"""Configuration, content identities, atomic artifacts and exclusive stages."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def merge(base: dict, patch: dict) -> dict:
    result = dict(base)
    for key, val in patch.items():
        result[key] = (
            merge(result[key], val)
            if isinstance(val, dict) and isinstance(result.get(key), dict)
            else val
        )
    return result


def config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = yaml.safe_load(text)
    parent = value.pop("extends", None)
    value = merge(config(path.parent / parent), value) if parent else value
    if value["mode"] not in {"real", "mock"}:
        raise ValueError("mode must be real or mock")
    exp = value["experiment"]
    if (
        exp["candidates"],
        exp["validation"],
        exp["rollouts"],
        exp["subset_size"],
        exp["seeds"],
    ) != (300, 30, 4, 30, [11, 22, 33]):
        raise ValueError(
            "Fixed research design requires 300/30/4/30 and seeds [11,22,33]"
        )
    tr = value["training"]
    if exp["split_seed"] != 42 or exp["shuffle_seed"] != 2026:
        raise ValueError(
            "Fixed research design requires split_seed=42 and shuffle_seed=2026"
        )
    if (tr["epochs"], tr["effective_batch"], tr["weight_decay"]) != (1, 1, 0):
        raise ValueError("Research design requires one epoch, batch=1, weight_decay=0")
    if (
        value["score"]["adam_epsilon"] != tr["epsilon"]
        or value["score"]["max_grad_norm"] != tr["max_grad_norm"]
    ):
        raise ValueError("Scoring direction and SFT optimizer must match")
    workers = value["agent"]["workers"]
    if not isinstance(workers, int) or not 1 <= workers <= len(
        value["agent"]["service_endpoints"]
    ):
        raise ValueError(
            "Provide at least one explicit service endpoint per task worker"
        )
    if value["agent"]["infra_retries"] != 2:
        raise ValueError("Research protocol permits at most two infrastructure retries")
    transfer = value.get("transfer", {})
    if transfer.get("enabled") and (
        transfer["sample_size"] != 50
        or transfer["seed"] != 42
        or set(transfer["suites"]) != {"verified", "pro", "live", "terminal"}
    ):
        raise ValueError(
            "Transfer design requires four fixed suites, 50 tasks each, seed=42"
        )
    return value


def require_commit(value: str | None, field: str) -> None:
    if not value or not re.fullmatch(r"[a-f0-9]{40}", value):
        raise ValueError(f"{field}: freeze an actual 40-hex source commit first")


def source_identity() -> dict:
    root = Path(__file__).resolve().parent.parent
    paths = [
        *root.joinpath("nice_validation").rglob("*.py"),
        *root.joinpath("scripts").glob("*.py"),
        root / "pyproject.toml",
    ]
    return {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in sorted(paths)}


def external_identity(cfg: dict) -> dict:
    environment = environment_identity(cfg["mode"])
    if cfg["mode"] == "mock":
        return {"fixture_version": "synthetic-390-60-v1", "environment": environment}
    inputs = {"environment": environment}
    for key in ["training_path", "validation_path", "source_lock"]:
        path = Path(cfg["data"][key])
        inputs["data:" + key] = sha(path) if path.is_file() else None
    if cfg.get("transfer", {}).get("enabled"):
        path = Path(cfg["transfer"]["source_lock"])
        inputs["transfer_source_lock"] = sha(path) if path.is_file() else None
        if path.is_file():
            inputs["transfer_sources"] = {
                suite: (
                    sha(Path(entry["path"])) if Path(entry["path"]).is_file() else None
                )
                for suite, entry in read(path).items()
            }
        interpreter = cfg["transfer"].get("live_python")
        if interpreter and Path(interpreter).is_file():
            from nice_validation.environments import live_info

            inputs["live_environment"] = live_info(cfg)
    for key in ["path", "tokenizer_path", "adapter_path"]:
        directory = cfg["model"].get(key)
        if not directory:
            inputs["model:" + key] = None
            continue
        root = Path(directory)
        files = [
            p
            for p in root.rglob("*")
            if p.is_file()
            and ".cache" not in p.parts
            and (
                key == "adapter_path"
                or p.suffix in {".json", ".jinja", ".txt", ".model"}
            )
        ]
        inputs["model:" + key] = {
            str(p.relative_to(root)): sha(p) for p in sorted(files)
        }
    return inputs


def environment_identity(mode: str = "real") -> dict:
    """Fingerprint installed numerical/protocol code without importing GPU libraries."""
    packages = ["numpy", "PyYAML", "scipy", "matplotlib", "Pillow", "packaging"]
    if mode == "real":
        packages += [
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "tokenizers",
            "safetensors",
            "huggingface-hub",
            "datasets",
            "langchain-core",
            "docker",
            "swebench",
            "swesmith",
            "harbor",
        ]
    versions, vcs = {}, {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
            raw = importlib.metadata.distribution(package).read_text("direct_url.json")
            if raw:
                metadata = json.loads(raw)
                if metadata.get("vcs_info"):
                    vcs[package] = metadata["vcs_info"]
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": versions,
        "vcs": vcs,
    }


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class Run:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.event_lock = threading.Lock()
        self.root = Path(cfg["output"])
        self.external = external_identity(cfg)
        self.identity = digest(
            {
                "config": cfg,
                "source": source_identity(),
                "external_inputs": self.external,
            }
        )
        self.root.mkdir(parents=True, exist_ok=True)
        frozen = self.root / "config" / "snapshot.json"
        if frozen.exists():
            if read(frozen)["identity"] != self.identity:
                raise ValueError(
                    "Run configuration/source identity changed; use a new output directory"
                )
        else:
            write(
                frozen,
                {
                    "identity": self.identity,
                    "config": cfg,
                    "source": source_identity(),
                    "external_inputs": self.external,
                    "created_utc": utc(),
                    "timezone": cfg["timezone"],
                    "platform": platform.platform(),
                },
            )
        self.state_path = self.root / "run_state.json"

    def event(self, stage: str, event: str, **data: Any) -> None:
        with self.event_lock:
            append(
                self.root / "logs" / "events.jsonl",
                {
                    "utc": utc(),
                    "run_id": self.root.name,
                    "stage": stage,
                    "event": event,
                    "subset_id": None,
                    "task_id": None,
                    "attempt_id": None,
                    "mode": self.cfg["mode"],
                    **data,
                },
            )

    def state(self) -> dict:
        return read(self.state_path) if self.state_path.exists() else {}

    def valid(self, name: str) -> bool:
        item = self.state().get(name, {})
        return (
            item.get("status") == "completed"
            and bool(item.get("artifacts"))
            and all(
                (self.root / p).is_file() and sha(self.root / p) == h
                for p, h in item["artifacts"].items()
            )
        )

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[list[Path] | None]:
        lock = self.root / ".run.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(
                "Run is locked. Inspect owner and use resume --recover-lock after process exit"
            ) from None
        os.write(
            fd,
            json.dumps(
                {"pid": os.getpid(), "host": platform.node(), "utc": utc()}
            ).encode(),
        )
        os.close(fd)
        start = time.monotonic()
        artifacts: list[Path] = []
        try:
            state = self.state()
            if self.valid(name):
                yield None
                return
            if state.get(name, {}).get("immutable_artifact_failure"):
                raise ValueError(
                    f"{name}: corrupt completed evidence is preserved; use a new run"
                )
            if state.get(name, {}).get("status") == "completed":
                raise ValueError(
                    f"{name}: completed artifacts missing/corrupt; preserve evidence and use new run"
                )
            state[name] = {"status": "running", "started_utc": utc()}
            write(self.state_path, state)
            self.event(name, "started")
            yield artifacts
            if not artifacts or any(not p.is_file() for p in artifacts):
                raise ValueError(f"{name}: missing output artifacts")
            state = self.state()
            state[name] = {
                "status": "completed",
                "ended_utc": utc(),
                "artifacts": {str(p.relative_to(self.root)): sha(p) for p in artifacts},
            }
            write(self.state_path, state)
            self.event(name, "completed")
        except BaseException as exc:
            state = self.state()
            previous = state.get(name, {})
            state[name] = {
                **previous,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "utc": utc(),
            }
            if previous.get("artifacts"):
                state[name]["immutable_artifact_failure"] = True
            write(self.state_path, state)
            self.event(name, "failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            duration = time.monotonic() - start
            assigned = (
                self.cfg["model"]["gpu_uuids"]
                if self.cfg["mode"] == "real"
                and (name in {"score", "smoke"} or name.startswith("train:"))
                else []
            )
            memory = {
                "gpu_memory_allocated_peak": None,
                "gpu_memory_reserved_peak": None,
            }
            torch = sys.modules.get("torch")
            if assigned and torch is not None and torch.cuda.is_initialized():
                memory = {
                    "gpu_memory_allocated_peak": torch.cuda.max_memory_allocated(),
                    "gpu_memory_reserved_peak": torch.cuda.max_memory_reserved(),
                }
            self.event(
                name,
                "stage_cost",
                wall_seconds=duration,
                assigned_gpu_uuids=assigned,
                gpu_allocated_seconds=duration * len(assigned) if assigned else None,
                allocation_definition=(
                    "explicit device assignment for stage lifetime"
                    if assigned
                    else None
                ),
                **memory,
                prefill_tokens=None,
                decode_tokens=None,
                gradient_replay_tokens=None,
                padding_tokens=None,
            )
            lock.unlink(missing_ok=True)


def doctor(cfg: dict) -> dict:
    import importlib.metadata
    import shutil

    versions = {}
    for package in [
        "numpy",
        "torch",
        "transformers",
        "peft",
        "datasets",
        "langchain-core",
        "docker",
        "swebench",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "utc": utc(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "packages": versions,
        "gpu": None,
        "docker": None,
        "network": "not_probed",
        "model_exists": bool(
            cfg["model"]["path"] and Path(cfg["model"]["path"]).is_dir()
        ),
    }
    for key, cmd in [
        (
            "gpu",
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total",
                "--format=csv,noheader",
            ],
        ),
        ("docker", ["docker", "info", "--format", "{{.ServerVersion}}"]),
    ]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            result[key] = {
                "returncode": p.returncode,
                "stdout": p.stdout.strip(),
                "stderr": p.stderr.strip(),
            }
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            result[key] = {"unavailable": type(exc).__name__}
    try:
        import psutil

        result["ram_bytes"] = psutil.virtual_memory().total
    except ImportError:
        result["ram_bytes"] = None
    return result
