"""Prebuilt task-image catalog: no registry access in the compute process."""
from pathlib import Path
import re

from nice_validation.core import digest, read, sha


def frozen_tasks(root):
    root = Path(root)
    tasks = []
    for name in ("validation", "sft_validation", "benchmark", "development_task"):
        value = read(root / "manifests" / (name + ".json"))
        tasks.extend(value if isinstance(value, list) else [value])
    ids = [t["instance_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Frozen task manifests overlap")
    return tasks


def catalog_for(cfg, tasks, client, identity):
    deployment = cfg.get("deployment", {})
    path = Path(deployment["task_catalog"])
    if sha(path) != deployment["task_catalog_sha256"]:
        raise ValueError("Task catalog checksum mismatch")
    catalog = read(path)
    if catalog["official_harness"] != identity:
        raise ValueError("Task catalog harness mismatch")
    result, failures = {}, []
    for task in tasks:
        key = task["instance_id"]
        try:
            record = catalog["tasks"][key]
            if record["task_sha256"] != digest(task):
                raise ValueError("frozen task differs from catalog")
            entry = record["entry"]
            if entry["base_commit"] != task["base_commit"] or entry.get("public_environment_only") is not True:
                raise ValueError("task base/isolation mismatch")
            for role, field in (("agent", "image"), ("grader", "grader_image")):
                reference = entry[field]
                if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", reference):
                    raise ValueError("catalog requires immutable registry digests")
                image = client.images.get(reference)  # Deliberately never pull here.
                if image.id != record[role + "_image_id"]:
                    raise ValueError("local image ID differs from prepared image")
            alias = client.images.get(entry["official_image_name"])
            if alias.id != record["grader_image_id"]:
                raise ValueError("official grader alias missing/changed; run load phase")
            result[key] = entry
        except Exception as exc:
            failures.append({"task_id": key, "reason": str(exc)})
    if failures:
        raise ValueError("Prebuilt task resources incomplete (no downloads attempted): " + str(failures))
    return result


def local_inputs(cfg):
    """Fail with all missing paths before model initialization or acquisition."""
    missing = []
    model = Path(cfg["model"].get("path") or "/__missing_model__")
    if not (model / "model_lock.json").is_file():
        missing.append(str(model / "model_lock.json"))
    for kind in ("training", "validation", "smith"):
        path = Path(cfg["data"][kind + "_path"])
        for item in (path, path.with_suffix(path.suffix + ".source.json")):
            if not item.is_file():
                missing.append(str(item))
    if missing:
        raise ValueError("Offline resources missing; prepare before GPU job: " + str(missing))
