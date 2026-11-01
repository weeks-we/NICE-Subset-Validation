"""New-run metadata with explicit, checksum-verified reuse of old raw inputs."""
import copy
from pathlib import Path

from nice_validation.core import config, read, write, require_commit, sha
from nice_validation.experiment_design import DESIGN


def locate_old_model(old, old_cfg, old_records, identifier):
    """Prefer the recorded path; recover moved storage only from matching locks."""
    recorded = old_cfg.get("model", {}).get("path")
    if recorded:
        path = Path(recorded)
        path = (path if path.is_absolute() else old / path).resolve()
        if path.exists():
            return path  # Existing but corrupt evidence must fail, not fall back.
    revision = old_records.get("model", {}).get("revision")
    locks = list((old / "models").glob("*/model_lock.json"))
    locks += [old / "models/model_lock.json"]
    if revision:
        cache = old / "cache/hub" / ("models--" + identifier.replace("/", "--"))
        locks.append(cache / "snapshots" / revision / "model_lock.json")
    matches = set()
    for lock in locks:
        if lock.is_file():
            metadata = read(lock)
            if metadata.get("id") == identifier and metadata.get("revision") == revision:
                matches.add(lock.parent.resolve())
    if len(matches) > 1:
        raise ValueError("Ambiguous reused model locations; supply the actual --model-path")
    if matches:
        return next(iter(matches))
    raise ValueError("Reused model directory not found in snapshot/models/cache; supply the actual --model-path")


def locate_old_raw(old, entry, kind):
    recorded = Path(entry["path"])
    path = (recorded if recorded.is_absolute() else old / recorded).resolve()
    if not path.exists():
        # A moved storage root can invalidate the original absolute prefix.
        path = (old / "data" / recorded.name).resolve()
        if not path.is_relative_to(old):
            raise ValueError("Relocated raw source escaped old storage: " + kind)
    if not path.is_file() or sha(path) != entry["sha256"]:
        raise ValueError("Old raw source checksum mismatch: " + kind)
    sidecar = path.with_suffix(path.suffix + ".source.json")
    if not sidecar.is_file():
        raise ValueError("Old raw source provenance sidecar missing")
    metadata = read(sidecar)
    if any(metadata.get(k) != entry.get(k) for k in ["id", "revision", "split", "limit", "sha256", "records"]):
        raise ValueError("Old raw source lock/sidecar mismatch: " + kind)
    return path


def configure(storage, gpus, destination, reuse_from=None, model_path=None, port=8765,
              base_config=None):
    from nice_validation.gpus import validate_pool
    from nice_validation.online import resolve_sources
    validate_pool(gpus)
    if len(gpus) < 2 or port < 1 or port + len(gpus) - 1 > 65535:
        raise ValueError("At least two authorized GPUs and a valid port range are required")
    storage, destination = Path(storage).resolve(), Path(destination).resolve()
    cfg = config(base_config or Path(__file__).resolve().parents[1] / "configs/main.yaml")
    if cfg["experiment"].get("design") != DESIGN:
        raise ValueError("The new server entry requires verified-selection-v1")
    old = Path(reuse_from).resolve() if reuse_from else None
    if old and (old == storage or old.is_relative_to(storage) or storage.is_relative_to(old)):
        raise ValueError("New storage and reused storage must be separate, non-nested directories")
    old_cfg, old_records, raw_entries = {}, {}, []
    if old:
        snapshot = old / "runs/main/config/snapshot.json"
        if snapshot.is_file():
            old_cfg = read(snapshot)["config"]
        source_file = old / "data/source_revisions.json"
        if not source_file.is_file():
            raise ValueError("--reuse-from must identify old storage with data/source_revisions.json")
        old_records = read(source_file)
        for name in ["source_lock.json", "transfer_sources.json"]:
            path = old / "data" / name
            if path.is_file():
                raw_entries.extend(read(path).values())
        if not model_path:
            model_path = locate_old_model(old, old_cfg, old_records, cfg["model"]["id"])

    def old_source(kind, identifier):
        keys = {"training": ["training"], "validation": ["verified", "validation"],
                "smith": ["smith", "validation"], "model": ["model"],
                "swebench": ["swebench"], "swesmith": ["swesmith"]}[kind]
        matches = [old_records[k] for k in keys if k in old_records and old_records[k].get("id") == identifier]
        if not matches or len({r["revision"] for r in matches}) != 1:
            raise ValueError(f"Missing/ambiguous pinned {kind} revision in old storage")
        return matches[0]
    model_path = Path(model_path).resolve() if model_path else storage / "models/base"
    if storage.is_relative_to(model_path) or model_path.is_relative_to(storage / "runs"):
        raise ValueError("Model path must not contain storage or reside under experiment outputs")
    if reuse_from and not model_path.is_dir():
        raise ValueError("Reused model directory not found; supply the actual --model-path")
    local_model = None
    if model_path.exists():
        lock = model_path / "model_lock.json"
        if not lock.is_file():
            raise ValueError("Existing model requires a verified model_lock.json; do not overwrite it")
        local_model = read(lock)
        if local_model.get("id") != cfg["model"]["id"] or not local_model.get("files"):
            raise ValueError("Existing model ID/files do not match this experiment")
        require_commit(local_model.get("revision"), "model.revision")
        architecture = local_model.get("architecture", {})
        max_positions = architecture.get("text_config", architecture).get("max_position_embeddings")
        if type(max_positions) is not int or max_positions < max(cfg["data"]["max_length"], cfg["agent"]["context_length"]):
            raise ValueError("Reused model's official context capacity is insufficient or unknown")
        for name, item in local_model["files"].items():
            file = (model_path / name).resolve()
            if not file.is_relative_to(model_path) or not file.is_file() or sha(file) != item["sha256"]:
                raise ValueError("Existing model file checksum mismatch")
    source_file = storage / "data/source_revisions.json"
    if source_file.exists():
        records = read(source_file)
    elif old:
        records = {}
        for kind, identifier in [("training", cfg["data"]["training_id"]),
                                 ("validation", cfg["data"]["validation_id"]),
                                 ("smith", cfg["data"]["smith_id"])]:
            entry = old_source(kind, identifier)
            records[kind] = {"id": identifier, "revision": entry["revision"]}
        for kind in ["model", "swebench", "swesmith"]:
            records[kind] = old_records[kind]
    else:
        records = resolve_sources(cfg)
        from huggingface_hub import HfApi
        records["smith"] = {"id": cfg["data"]["smith_id"],
                            "revision": HfApi().dataset_info(cfg["data"]["smith_id"]).sha}
    if local_model:
        if source_file.exists() and records["model"]["revision"] != local_model["revision"]:
            raise ValueError("Model differs from this run's frozen source revision")
        records["model"] = {"id": local_model["id"], "revision": local_model["revision"]}
    for kind in ["training", "validation", "smith", "model", "swebench", "swesmith"]:
        require_commit(records[kind]["revision"], kind)
        identifier = (cfg["data"][kind + "_id"] if kind in {"training", "validation", "smith"}
                      else cfg["model"]["id"] if kind == "model" else "SWE-bench/" + ("SWE-bench" if kind == "swebench" else "SWE-smith"))
        if records[kind]["id"] != identifier:
            raise ValueError("Pinned source repository mismatch: " + kind)
        if old:
            if old_source(kind, identifier)["revision"] != records[kind]["revision"]:
                raise ValueError("Reused source revision changed: " + kind)
    cfg["execution"] = "server"
    cfg["output"] = str(storage / "runs/main")
    cfg["model"].update(revision=records["model"]["revision"], path=str(model_path),
                         tokenizer_path=str(model_path), gpu_uuids=[gpus[0]],
                         adapter_path=str(storage / "models/initial_adapter"))
    slots = cfg["agent"].get("tasks_per_gpu", 1)
    cfg["resources"] = {"gpu_count": len(gpus), "gpu_uuids": list(gpus),
                         "inference_gpu_uuids": list(gpus) * slots, "layout": "verified-selection"}
    raw_reuse = {}
    for kind, split in [("training", "tool"), ("validation", "test"), ("smith", "train")]:
        cfg["data"][kind + "_revision"] = records[kind]["revision"]
        target = storage / "data" / (kind + ".jsonl")
        if old:
            matches = [r for r in raw_entries if r.get("id") == records[kind]["id"]
                       and r.get("revision") == records[kind]["revision"] and r.get("split") == split]
            if len(matches) > 1:
                raise ValueError("Ambiguous old raw source " + kind)
            if matches:
                previous = locate_old_raw(old, matches[0], kind)
                if matches[0].get("limit") is None:
                    target = previous
                    raw_reuse[kind] = {"action": "reuse", "path": str(previous),
                        "revision": records[kind]["revision"], "records": matches[0].get("records"),
                        "sha256": matches[0]["sha256"]}
                else:
                    raw_reuse[kind] = {"action": "download_full_pinned_split", "reason": "old source is prefix-limited",
                        "old_path": str(previous), "revision": records[kind]["revision"]}
        raw_reuse.setdefault(kind, {"action": "download_full_pinned_split", "revision": records[kind]["revision"]})
        cfg["data"][kind + "_path"] = str(target)
    cfg["data"]["source_lock"] = str(storage / "data/source_lock.json")
    for key, name in [("hub_cache", "hub"), ("datasets_cache", "datasets")]:
        cfg["data"][key] = str(Path(old_cfg.get("data", {}).get(key) or ((old or storage) / "cache" / name)).resolve())
    cfg["agent"].update(swebench_revision=records["swebench"]["revision"],
                         swesmith_revision=records["swesmith"]["revision"],
                         workers=len(gpus) * slots, service_url=f"http://127.0.0.1:{port + 1}",
                         service_endpoints=[f"http://127.0.0.1:{port + i}" for i in range(len(gpus))] * slots,
                         image_map=str(storage / "data/task_images.json"),
                         image_revision_lock=str(storage / "data/task_images.lock.json"),
                         smoke_task_path=str(storage / "data/development_task.json"))
    files = {destination / "server.local.yaml": cfg}
    services = []
    for index, gpu in enumerate(gpus):
        service = copy.deepcopy(cfg)
        service["model"]["gpu_uuids"] = [gpu]
        path = destination / f"service-{index}.local.yaml"
        files[path] = service
        services.append(str(path))
    for path, value in files.items():
        if path.exists() and config(path) != value:
            raise ValueError("Refusing to overwrite different configuration: " + str(path))
    write(source_file, records)
    for path, value in files.items():
        if not path.exists():
            write(path, value)
    write(storage / "data/reuse.json", {"reuse_from": str(old) if old else None,
          "model_path": str(model_path), "raw_sources": {k: cfg["data"][k + "_path"] for k in ["training", "validation", "smith"]},
          "model_revision": records["model"]["revision"], "raw_source_actions": raw_reuse,
          "policy": "reuse only frozen base/tokenizer/raw sources/cache; never reuse old gradients, task splits or trained adapters"})
    return {"server_config": str(destination / "server.local.yaml"), "service_configs": services}


def acquire(cfg):
    from nice_validation.acquire import hf_split
    entries = {}
    for kind, split in [("training", "tool"), ("validation", "test"), ("smith", "train")]:
        target = Path(cfg["data"][kind + "_path"])
        if target.exists():
            entry = read(target.with_suffix(target.suffix + ".source.json"))
            if any(entry.get(k) != v for k, v in {"id": cfg["data"][kind + "_id"],
                   "revision": cfg["data"][kind + "_revision"], "split": split, "limit": None}.items()) or entry["sha256"] != sha(target):
                raise ValueError("Raw source provenance/checksum mismatch: " + kind)
        else:
            entry = hf_split(cfg["data"][kind + "_id"], cfg["data"][kind + "_revision"],
                             split, target, "download", True, cfg["data"])
        entries[kind] = {**entry, "path": str(target)}
    lock = Path(cfg["data"]["source_lock"])
    if lock.exists() and read(lock) != entries:
        raise ValueError("Raw source lock changed")
    if not lock.exists():
        write(lock, entries)


def prepare_online(cfg, download_model=True):
    from nice_validation.model import server_only, fetch_model, initialize_adapter
    from nice_validation.preflight import preflight
    from nice_validation.environments import official_identity, prepare_environments
    from nice_validation.core import Run
    from nice_validation.data import prepare
    from nice_validation.online import development_task
    server_only(cfg)
    readiness = preflight(cfg)
    write(Path(cfg["output"]) / "preflight-bootstrap.json", readiness)
    if readiness["status"] != "PASS":
        raise ValueError("Bootstrap preflight failed; inspect saved report")
    official_identity(cfg)
    if download_model:
        fetch_model(cfg)  # Existing verified snapshots return without download/write.
    acquire(cfg)
    initialize_adapter(cfg)
    run = Run(cfg)
    prepare(run)
    development_task(run)
    prepare_environments(run)
    return run
