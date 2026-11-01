"""Loopback exact-token inference service. Start explicitly on the server."""

from __future__ import annotations

import json
import os
import time
import signal
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from nice_validation.core import (
    append,
    utc,
    digest,
    environment_identity,
    source_identity,
    sha,
)
from nice_validation.model import Runtime, server_only, tree_identity


class RuntimeManager:
    """Load adapters transactionally; failed switches leave a retryable empty state."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.runtime = None
        self.key = None

    def get(self, adapter: Path, checksum: str):
        key = (str(adapter), checksum)
        if self.key == key and self.runtime is not None:
            return self.runtime
        if self.runtime is not None and self.cfg.get("model", {}).get("reuse_base_for_inference"):
            try:
                self.runtime.switch_adapter(adapter)
                if self.runtime.adapter_identity != checksum:
                    raise ValueError("Switched adapter checksum mismatch")
            except BaseException:
                self.close()
                raise
            self.key = key
            return self.runtime
        if self.runtime is not None:
            previous, self.runtime, self.key = self.runtime, None, None
            previous.close()
        candidate = Runtime(self.cfg, adapter)
        self.runtime, self.key = candidate, key
        return candidate

    def close(self):
        if self.runtime is not None:
            previous, self.runtime, self.key = self.runtime, None, None
            previous.close()


def service_identity(cfg: dict) -> str:
    model = {k: v for k, v in cfg["model"].items() if k != "gpu_uuids"}
    lock = Path(cfg["model"]["path"] or ".") / "model_lock.json"
    return digest(
        {
            "model": model,
            "model_lock": sha(lock) if lock.is_file() else None,
            "agent": cfg["agent"],
            "output": str(Path(cfg["output"]).resolve()),
            "source": source_identity(),
            "environment": environment_identity("real"),
        }
    )


def matching_health(value, cfg: dict, expected_gpu: str | None = None) -> bool:
    if not isinstance(value, dict) or value.get("status") != "ready":
        return False
    gpus = value.get("gpu_uuids")
    if not isinstance(gpus, list) or len(gpus) != 1 or not isinstance(gpus[0], str):
        return False
    if expected_gpu is not None and gpus != [expected_gpu]:
        return False
    if expected_gpu is None and gpus == cfg["model"]["gpu_uuids"]:
        return False
    return value.get("service_identity") == service_identity(cfg)


def serve(cfg: dict, host: str, port: int) -> None:
    server_only(cfg)
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "The unauthenticated local inference service must bind to loopback"
        )
    manager = RuntimeManager(cfg)
    identity = service_identity(cfg)
    allowed_roots = [
        Path(cfg["model"]["adapter_path"]).resolve(),
        (Path(cfg["output"]) / "training").resolve(),
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            response = json.dumps(
                {
                    "status": "ready",
                    "model_loaded": manager.runtime is not None,
                    "adapter": manager.key[0] if manager.key else None,
                    "service_identity": identity,
                    "gpu_uuids": cfg["model"]["gpu_uuids"],
                    "pid": os.getpid(),
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            try:
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                adapter = Path(payload["adapter_path"]).resolve()
                if not any(
                    adapter == root or root in adapter.parents for root in allowed_roots
                ):
                    raise ValueError("adapter path outside frozen experiment roots")
                if tree_identity(adapter) != payload["adapter_sha256"]:
                    raise ValueError("adapter checksum mismatch")
                runtime = manager.get(adapter, payload["adapter_sha256"])
                result = runtime.generate(
                    payload["messages"],
                    payload["nice"],
                    payload["seed"],
                    payload["max_tokens"],
                )
                response = json.dumps(result, allow_nan=False).encode()
                self.send_response(200)
            except Exception as exc:
                response = json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"}
                ).encode()
                self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            try:
                self.wfile.write(response)
            except (BrokenPipeError, ConnectionResetError):
                append(
                    log,
                    {
                        "utc": utc(),
                        "stage": "inference-service",
                        "event": "client_disconnected",
                        "run_id": Path(cfg["output"]).name,
                        "mode": "real",
                    },
                )

    print(
        f"Inference service listening on http://{host}:{port}; loads local frozen adapters only",
        flush=True,
    )
    started = time.monotonic()
    log = Path(cfg["output"]) / "logs" / f"service-{os.getpid()}.jsonl"
    append(
        log,
        {
            "utc": utc(),
            "stage": "inference-service",
            "event": "started",
            "run_id": Path(cfg["output"]).name,
            "mode": "real",
            "assigned_gpu_uuids": cfg["model"]["gpu_uuids"],
            "host": host,
            "port": port,
        },
    )
    server = HTTPServer((host, port), Handler)
    previous_signal = None
    if threading.current_thread() is threading.main_thread():
        previous_signal = signal.getsignal(signal.SIGTERM)

        def stop_service(*args):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop_service)
    try:
        server.serve_forever()
    finally:
        manager.close()
        duration = time.monotonic() - started
        append(
            log,
            {
                "utc": utc(),
                "stage": "inference-service",
                "event": "stage_cost",
                "run_id": Path(cfg["output"]).name,
                "mode": "real",
                "wall_seconds": duration,
                "gpu_allocated_seconds": duration * len(cfg["model"]["gpu_uuids"]),
                "allocation_definition": "dedicated service GPU reservation for process lifetime",
                "assigned_gpu_uuids": cfg["model"]["gpu_uuids"],
            },
        )
        server.server_close()
        if previous_signal is not None:
            signal.signal(signal.SIGTERM, previous_signal)
