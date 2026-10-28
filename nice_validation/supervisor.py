"""Own the server run, service lifecycle and final evidence without manual terminals."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import urllib.request

from nice_validation.core import (
    config,
    environment_identity,
    read,
    source_identity,
    utc,
    write,
)
from nice_validation.model import server_only
from nice_validation.service import service_identity


def child_environment(gpu: str | None = None) -> dict:
    env = os.environ.copy()
    bypass = {"127.0.0.1", "localhost"}
    for key in ["NO_PROXY", "no_proxy"]:
        bypass.update(
            part.strip() for part in env.get(key, "").split(",") if part.strip()
        )
    env["NO_PROXY"] = env["no_proxy"] = ",".join(sorted(bypass))
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def health(url: str) -> dict | None:
    try:
        # The model service is loopback only; it must never traverse a proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url.rstrip("/") + "/health", timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def matches_service(value: dict, cfg: dict) -> bool:
    return (
        value.get("status") == "ready"
        and value.get("service_identity") == service_identity(cfg)
        and value.get("gpu_uuids") == cfg["model"]["gpu_uuids"]
    )


@contextlib.contextmanager
def controller_lock(root: Path, recover: bool):
    path = root / "controller.lock"
    if path.exists() and recover:
        import psutil

        owner = read(path)
        if owner["host"] != platform.node() or psutil.pid_exists(owner["pid"]):
            raise ValueError("Controller lock owner may be active; refusing recovery")
        path.unlink()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(
            "Controller already owns this storage root; recover only after exit"
        ) from None
    os.write(
        fd,
        json.dumps(
            {"pid": os.getpid(), "host": platform.node(), "utc": utc()}
        ).encode(),
    )
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def command(root: Path, name: str, argv: list[str], env: dict) -> None:
    folder = root / "controller" / name
    attempt = folder / f"attempt_{len(list(folder.glob('attempt_*'))):03d}"
    attempt.mkdir(parents=True)
    write(attempt / "intent.json", {"argv": argv, "utc": utc()})
    with (
        (attempt / "stdout.log").open("w", encoding="utf-8") as out,
        (attempt / "stderr.log").open("w", encoding="utf-8") as err,
    ):
        result = subprocess.run(
            argv,
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
        )
    write(attempt / "result.json", {"returncode": result.returncode, "utc": utc()})
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {attempt}")


def start_service(root: Path, cfg: dict, service_file: Path, port: int):
    url = f"http://127.0.0.1:{port}"
    existing = health(url)
    if existing:
        if not matches_service(existing, cfg):
            raise ValueError(
                "Existing service identity/GPU differs; use a free port or the matching config"
            )
        return None, None
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise ValueError(
                "Service port is occupied but not ready; refusing a duplicate service"
            )
    log = (root / f"controller/service-{port}.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nice_validation",
            "--config",
            str(service_file),
            "serve",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        env=child_environment(cfg["model"]["gpu_uuids"][0]),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    write(
        root / f"controller/service-{port}.json",
        {"pid": process.pid, "config": str(service_file), "utc": utc()},
    )
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Inference service exited; inspect controller/service-{port}.log"
                )
            value = health(url)
            if value:
                if not matches_service(value, cfg):
                    raise ValueError("Started service has an unexpected identity")
                return process, log
            time.sleep(0.5)
        raise RuntimeError("Inference service readiness timed out")
    except BaseException:
        stop_service(process, log)
        raise


def stop_service(process, log) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    if log is not None:
        log.close()


def stop_services(services: list) -> None:
    errors = []
    for process, log in reversed(services):
        try:
            stop_service(process, log)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("Service cleanup failed: " + "; ".join(errors))


def run_server(
    storage: Path,
    train_gpu: str,
    service_gpu: str,
    config_dir: Path,
    port: int = 8765,
    model_path: Path | None = None,
    recover: bool = False,
    service_gpus: list[str] | None = None,
) -> dict:
    if platform.system() != "Linux" or sys.version_info < (3, 12):
        raise ValueError("Full server execution requires Linux and Python 3.12+")
    from nice_validation.online import configure_server

    storage, config_dir = storage.resolve(), config_dir.resolve()
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "controller").mkdir(exist_ok=True)
    with controller_lock(storage, recover):
        recorded = storage / "controller/source.json"
        source = source_identity()
        if recorded.is_file() and read(recorded) != source:
            raise ValueError(
                "Controller source changed; preserve evidence and use a new storage root"
            )
        write(recorded, source)
        arguments = (storage, train_gpu, service_gpu, config_dir, model_path, port)
        paths = configure_server(
            *arguments,
            **({"service_gpus": service_gpus} if service_gpus is not None else {}),
        )
        server_file = Path(paths["server_config"])
        service_files = [
            Path(p) for p in paths.get("service_configs", [paths["service_config"]])
        ]
        cfg = config(server_file)
        server_only(cfg)
        env = child_environment(train_gpu)
        frozen = Path(cfg["output"]) / "config/snapshot.json"
        if frozen.is_file():
            previous = read(frozen)
            if previous["external_inputs"]["environment"] != environment_identity(
                "real"
            ):
                raise ValueError(
                    "Frozen run environment changed; do not reinstall into an existing run"
                )
        else:
            installed = storage / "controller/harness_environment.json"
            if not installed.is_file() or read(installed) != environment_identity(
                "real"
            ):
                command(
                    storage,
                    "install-harness",
                    [
                        sys.executable,
                        "scripts/install_harness.py",
                        "--config",
                        str(server_file),
                    ],
                    env,
                )
                write(installed, environment_identity("real"))
        executable = [
            sys.executable,
            "-m",
            "nice_validation",
            "--config",
            str(server_file),
        ]
        services = []
        try:
            command(storage, "preflight", executable + ["preflight"], env)
            options = ["prepare-online", "--download-model"] + (
                ["--recover-lock"] if recover else []
            )
            command(storage, "prepare", executable + options, env)
            for i, service_file in enumerate(service_files):
                services.append(
                    start_service(storage, config(service_file), service_file, port + i)
                )
            options = ["run-online"] + (["--recover-lock"] if recover else [])
            command(storage, "experiment", executable + options, env)
        except BaseException as exc:
            write(
                storage / "controller/completion.json",
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "utc": utc(),
                },
            )
            raise
        finally:
            try:
                stop_services(services)
            except BaseException as exc:
                write(
                    storage / "controller/completion.json",
                    {
                        "status": "failed",
                        "phase": "stop-services",
                        "error": f"{type(exc).__name__}: {exc}",
                        "utc": utc(),
                    },
                )
                raise
        # The owned service has now flushed its complete reservation cost.
        try:
            command(storage, "analyze", executable + ["analyze", "--refresh"], env)
            command(storage, "audit", executable + ["audit"], env)
            audit = read(Path(cfg["output"]) / "audit.json")
            if audit["status"] == "FAIL":
                raise ValueError("Final audit failed")
            bundle = (
                storage / "controller" / ("results-" + utc().replace(":", "-") + ".zip")
            )
            command(
                storage,
                "bundle",
                [
                    sys.executable,
                    "scripts/bundle_results.py",
                    "--run",
                    cfg["output"],
                    "--output",
                    str(bundle),
                ],
                env,
            )
        except BaseException as exc:
            write(
                storage / "controller/completion.json",
                {
                    "status": "failed",
                    "phase": "finalize",
                    "error": f"{type(exc).__name__}: {exc}",
                    "utc": utc(),
                },
            )
            raise
        result = {
            "status": "completed",
            "audit": audit["status"],
            "bundle": str(bundle),
            "utc": utc(),
            "run": cfg["output"],
            "service_reused": any(process is None for process, _ in services),
            "gpu_count": cfg.get("resources", {}).get("gpu_count", 2),
            "inference_services": len(services),
        }
        write(storage / "controller/completion.json", result)
        return result
