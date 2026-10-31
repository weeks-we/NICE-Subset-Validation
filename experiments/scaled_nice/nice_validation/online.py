"""Server bootstrap and online preparation, before freezing research run inputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import urllib.request

from nice_validation.core import (
    Run,
    config,
    read,
    require_commit,
    sha,
    write,
)
from nice_validation.model import server_only
from nice_validation.compatibility import SWEBENCH_REF


def resolve_sources(cfg: dict) -> dict:
    """Resolve mutable upstream heads once; subsequent downloads use full commits."""
    from huggingface_hub import HfApi

    api = HfApi()
    result = {}
    for kind in ["training", "validation", "model"]:
        identifier = (
            cfg["model"]["id"] if kind == "model" else cfg["data"][kind + "_id"]
        )
        info = (
            api.model_info(identifier)
            if kind == "model"
            else api.dataset_info(identifier)
        )
        require_commit(info.sha, kind + ".revision")
        result[kind] = {"id": identifier, "revision": info.sha}
    request = urllib.request.Request(
        f"https://api.github.com/repos/SWE-bench/SWE-bench/commits/{SWEBENCH_REF}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "nice-validation",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        commit = json.load(response)["sha"]
    require_commit(commit, "swebench.revision")
    result["swebench"] = {"id": "SWE-bench/SWE-bench", "revision": commit}

    def github(repository, ref="main"):
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repository}/commits/{ref}",
            headers={"User-Agent": "nice-validation"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            revision = json.load(response)["sha"]
        require_commit(revision, repository)
        return {"id": repository, "revision": revision}

    result["swesmith"] = github("SWE-bench/SWE-smith")
    if cfg.get("transfer", {}).get("enabled"):
        result["harbor"] = github("harbor-framework/harbor")
        result["live_harness"] = github("microsoft/SWE-bench-Live", "python-only")
        result["pro"] = github(cfg["transfer"]["suites"]["pro"]["id"])
        for suite in ["verified", "live"]:
            identifier = cfg["transfer"]["suites"][suite]["id"]
            info = api.dataset_info(identifier)
            require_commit(info.sha, suite)
            result[suite] = {"id": identifier, "revision": info.sha}
        url = f"https://raw.githubusercontent.com/harbor-framework/harbor/{result['harbor']['revision']}/registry.json"
        with urllib.request.urlopen(url, timeout=60) as response:
            registry = json.load(response)
        entry = next(
            item
            for item in registry
            if item["name"] == "terminal-bench" and str(item["version"]) == "2.0"
        )
        revisions = {t["git_commit_id"] for t in entry["tasks"]}
        urls = {t["git_url"] for t in entry["tasks"]}
        if len(revisions) != 1 or urls != {
            "https://github.com/laude-institute/terminal-bench-2.git"
        }:
            raise ValueError(
                "Terminal-Bench 2.0 registry has an unexpected mixed source layout"
            )
        revision = next(iter(revisions))
        require_commit(revision, "terminal-bench@2.0")
        result["terminal"] = {
            "id": "laude-institute/terminal-bench-2",
            "revision": revision,
            "version": "2.0",
            "registered_tasks": [
                {"name": t["name"], "path": t["path"]} for t in entry["tasks"]
            ],
        }
    return result


def configure_server(
    storage: Path,
    train_gpu: str,
    service_gpu: str,
    destination: Path,
    model_path: Path | None = None,
    port: int = 8765,
    service_gpus: list[str] | None = None,
    base_config: Path | None = None,
    model_id: str | None = None,
    model_loader: str | None = None,
) -> dict:
    """Generate local configs and a metadata lock; no weights or GPU imports."""
    from nice_validation.gpus import validate_pool

    inference_gpus = list(service_gpus) if service_gpus is not None else [service_gpu]
    validate_pool([train_gpu, *inference_gpus])
    if not inference_gpus or inference_gpus[0] != service_gpu:
        raise ValueError("Primary service GPU must be first in the inference pool")
    if not 1 <= port <= 65535 or port + len(inference_gpus) - 1 > 65535:
        raise ValueError("Invalid service port")
    storage, destination = storage.resolve(), destination.resolve()
    cfg = config(
        base_config or Path(__file__).resolve().parent.parent / "configs/main.yaml"
    )
    if model_id:
        cfg["model"]["id"] = model_id
    if model_loader:
        cfg["model"]["loader"] = model_loader
    if not cfg["model"].get("id"):
        raise ValueError(
            "Model is pending: explicitly select --model-id before online preparation"
        )
    phased = cfg.get("schedule") == "phased-independent"
    authorized_gpus = [train_gpu, *inference_gpus]
    if phased:
        if cfg.get("transfer", {}).get("enabled"):
            raise ValueError(
                "Independent scaled schedule is main-only; disable transfer"
            )
        inference_gpus = [*inference_gpus, train_gpu]
        if port + len(inference_gpus) - 1 > 65535:
            raise ValueError("Invalid phased service port range")
        if not cfg["model"].get("loader"):
            raise ValueError("Explicitly select --model-loader for the pending model")
    local_model = None
    if model_path and model_path.exists():
        lock = model_path / "model_lock.json"
        if not lock.is_file():
            raise ValueError(
                "Existing --model-path needs a verified model_lock.json; otherwise omit it and download a pinned snapshot"
            )
        local_model = read(lock)
        if local_model.get("id") != cfg["model"]["id"]:
            raise ValueError("Existing model lock belongs to another model")
        require_commit(local_model.get("revision"), "existing model revision")
        for name, entry in local_model.get("files", {}).items():
            file = (model_path / name).resolve()
            if (
                not file.is_relative_to(model_path.resolve())
                or not file.is_file()
                or sha(file) != entry["sha256"]
            ):
                raise ValueError("Existing model file differs from its verified lock")
        if not local_model.get("files"):
            raise ValueError("Existing model lock has no file evidence")
    chosen_model = (
        model_path.resolve()
        if model_path
        else storage / "models" / ("base" if phased else "Qwen3.8-27B")
    )
    if (storage / "runs/main").is_relative_to(chosen_model):
        raise ValueError("Model directory cannot contain experiment outputs")
    lock_path = storage / "data/source_revisions.json"
    records = read(lock_path) if lock_path.exists() else resolve_sources(cfg)
    if local_model:
        if (
            lock_path.exists()
            and records["model"]["revision"] != local_model["revision"]
        ):
            raise ValueError(
                "Existing model revision differs from frozen source configuration"
            )
        records["model"] = {
            "id": local_model["id"],
            "revision": local_model["revision"],
        }
    for kind in ["training", "validation", "model", "swebench", "swesmith"]:
        if kind not in records:
            raise ValueError(
                "Existing metadata lock is incomplete; choose a new storage root or supply the missing pinned source"
            )
        identifier = (
            cfg["model"]["id"]
            if kind == "model"
            else (
                {"swebench": "SWE-bench/SWE-bench", "swesmith": "SWE-bench/SWE-smith"}[
                    kind
                ]
                if kind in {"swebench", "swesmith"}
                else cfg["data"][kind + "_id"]
            )
        )
        if records[kind]["id"] != identifier:
            raise ValueError("Existing source lock identifies a different repository")
        require_commit(records[kind]["revision"], kind + ".revision")
    cfg["execution"] = "server"
    cfg["resources"] = {
        "gpu_count": len(authorized_gpus),
        "gpu_uuids": authorized_gpus,
        "layout": "phased-independent" if phased else "one-gradient-rest-inference",
        "inference_gpu_uuids": inference_gpus,
    }
    cfg["output"] = str(storage / "runs/main")
    cfg["model"].update(
        {
            "revision": records["model"]["revision"],
            "path": str(chosen_model),
            "loader": cfg["model"].get("loader") or "AutoModelForImageTextToText",
            "gpu_uuids": [train_gpu],
            "adapter_path": str(storage / "models/initial_adapter"),
        }
    )
    cfg["model"]["tokenizer_path"] = cfg["model"]["path"]
    cfg["data"].update(
        {
            "training_revision": records["training"]["revision"],
            "validation_revision": records["validation"]["revision"],
            **{
                key: str(storage / "data" / filename)
                for key, filename in [
                    ("training_path", "training.jsonl"),
                    ("validation_path", "validation.jsonl"),
                    ("source_lock", "source_lock.json"),
                ]
            },
            "hub_cache": str(storage / "cache/hub"),
            "datasets_cache": str(storage / "cache/datasets"),
        }
    )
    cfg["agent"].update(
        {
            "swebench_revision": records["swebench"]["revision"],
            "swesmith_revision": records["swesmith"]["revision"],
            "service_url": f"http://127.0.0.1:{port}",
            "workers": len(inference_gpus),
            "service_endpoints": [
                f"http://127.0.0.1:{port + i}" for i in range(len(inference_gpus))
            ],
            **{
                key: str(storage / "data" / filename)
                for key, filename in [
                    ("image_map", "task_images.json"),
                    ("image_revision_lock", "task_images.lock.json"),
                    ("smoke_task_path", "development_task.json"),
                ]
            },
        }
    )
    if cfg["transfer"]["enabled"]:
        for suite in ["verified", "pro", "live", "terminal"]:
            entry = records[suite]
            if entry["id"] != cfg["transfer"]["suites"][suite]["id"]:
                raise ValueError(
                    "Benchmark metadata lock identifies a different source"
                )
            require_commit(entry["revision"], suite)
            cfg["transfer"]["suites"][suite]["revision"] = entry["revision"]
        registered = records["terminal"].get("registered_tasks")
        if not registered:
            raise ValueError(
                "Terminal source lock lacks the 2.0 registry task list; use a new storage root"
            )
        cfg["transfer"]["suites"]["terminal"]["registered_tasks"] = registered
        cfg["transfer"].update(
            {
                "harbor_revision": records["harbor"]["revision"],
                "live_revision": records["live_harness"]["revision"],
                "live_python": str(storage / "envs/live/bin/python"),
                "checkouts": str(storage / "data/benchmark_repos"),
                "source_lock": str(storage / "data/transfer_sources.json"),
            }
        )
    paths = {destination / "server.local.yaml": cfg}
    service_files = []
    for i, gpu in enumerate(inference_gpus):
        service = copy.deepcopy(cfg)
        service["model"]["gpu_uuids"] = [gpu]
        path = destination / (
            "service.local.yaml" if i == 0 else f"service-{i + 1}.local.yaml"
        )
        paths[path] = service
        service_files.append(str(path))
    # JSON is valid YAML. Check every destination before writing either config.
    for path, value in paths.items():
        if path.exists() and config(path) != value:
            raise ValueError(
                f"Refusing to overwrite a different local configuration: {path}"
            )
    write(lock_path, records)
    for path, value in paths.items():
        if not path.exists():
            write(path, value)
    return {
        "server_config": str(destination / "server.local.yaml"),
        "service_config": str(destination / "service.local.yaml"),
        "service_configs": service_files,
        "sources": records,
    }


def development_task(run: Run) -> dict:
    """Choose by ID outside main validation; never inspect model outcomes."""
    path = Path(run.cfg["agent"]["smoke_task_path"])
    selected = {t["instance_id"] for t in read(run.root / "manifests/validation.json")}
    expected = read(run.root / "manifests/development_task.json")
    if path.exists():
        task = read(path)
        if task["instance_id"] in selected or task != expected:
            raise ValueError(
                "Existing development task overlaps validation or differs from frozen source"
            )
    else:
        task = expected
        write(path, task)
    with run.stage("prepare-online-inputs") as artifacts:
        record = run.root / "manifests/online_inputs.json"
        value = {
            "development_task_id": task["instance_id"],
            "development_task_sha256": sha(path),
            "validation_source_sha256": sha(Path(run.cfg["data"]["validation_path"])),
            "selection": "held-out development manifest; excluded from training and transfer; no model outcomes",
        }
        if artifacts is None:
            if read(record) != value:
                raise ValueError("Online preparation inputs changed")
        else:
            write(record, value)
            artifacts.append(record)
    return task


def prepare_online(
    cfg: dict,
    download_model: bool = False,
    mode: str = "download",
    recover_lock: bool = False,
) -> Run:
    """Acquire all external inputs before constructing the immutable Run."""
    server_only(cfg)
    from nice_validation.preflight import preflight
    from nice_validation.environments import official_identity, prepare_environments
    from nice_validation.data import fetch_data, prepare
    from nice_validation.model import fetch_model, initialize_adapter

    readiness = preflight(cfg)
    write(Path(cfg["output"]) / "preflight-bootstrap.json", readiness)
    if readiness["status"] != "PASS":
        raise ValueError(
            "Online preparation prerequisites failed; inspect preflight-bootstrap.json"
        )
    official_identity(
        cfg
    )  # Fail before downloading if the harness installation differs.
    plan = fetch_data(cfg, mode, None, False)
    write(Path(cfg["output"]) / "online-download-plan.json", plan)
    model_root = Path(cfg["model"]["path"])
    if download_model:
        print("Acquiring the explicitly requested pinned model snapshot", flush=True)
        fetch_model(cfg)
    lock = model_root / "model_lock.json"
    if not lock.is_file():
        raise ValueError(
            "Model not frozen: use --download-model, or verify existing files and freeze-model"
        )
    if any(read(lock).get(k) != cfg["model"][k] for k in ["id", "revision"]):
        raise ValueError("Model lock differs from online configuration")
    print("Acquiring complete pinned tool/test data splits", flush=True)
    fetch_data(cfg, mode, None, True)
    if cfg.get("transfer", {}).get("enabled"):
        from nice_validation.transfer_sources import acquire_transfer

        acquire_transfer(cfg, mode)
    initialize_adapter(cfg)
    # Creating this earlier would freeze missing source/adapter files into identity.
    run = Run(cfg)
    if recover_lock:
        from nice_validation.cli import recover_lock as recover

        recover(run)
    prepare(run)
    development_task(run)
    prepare_environments(run)
    return run


def run_online(cfg: dict, recover_lock: bool = False) -> None:
    """Continue through the existing smoke/training/evaluation pipeline."""
    server_only(cfg)
    if cfg.get("schedule") == "phased-independent":
        raise ValueError(
            "Use scripts/run_scaled_server.py to coordinate phased GPU ownership"
        )
    from nice_validation.preflight import preflight
    from nice_validation.smoke import smoke
    from nice_validation.cli import pipeline

    run = Run(cfg)
    if recover_lock:
        from nice_validation.cli import recover_lock as recover

        recover(run)
    if not run.valid("prepare-data") or not run.valid("prepare-environments"):
        raise ValueError("Complete prepare-online before run-online")
    development_task(run)
    readiness = preflight(cfg, prepared=True)
    write(run.root / "preflight-prepared.json", readiness)
    if readiness["status"] != "PASS":
        raise ValueError(
            "Prepared preflight failed; start the matching service and inspect the report"
        )
    smoke(run)
    pipeline(run)
