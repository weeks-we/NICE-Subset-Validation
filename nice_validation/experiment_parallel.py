"""Overlap four independent SFT workers with checkpoint validation on spare GPUs."""
import contextlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import uuid

from nice_validation.core import Run, read, read_jsonl, append, write, utc
from nice_validation.experiment import EndpointPool, evaluate
from nice_validation.experiment_training import validate_point
from nice_validation.parallel_training import validate_receipt


@contextlib.contextmanager
def worker_lease(run, gpu):
    import psutil
    folder = run.root / "training_leases"
    folder.mkdir(exist_ok=True)
    path = folder / (gpu + ".json")
    if path.exists():
        owner = read(path)
        if owner["host"] != platform.node() or psutil.pid_exists(owner["pid"]):
            raise ValueError("A training worker may still own this GPU; inspect its lease")
        path.rename(path.with_name(path.name + ".stale-" + uuid.uuid4().hex))
    owner = {"host": platform.node(), "pid": os.getpid(), "created": psutil.Process().create_time(),
             "run_identity": run.identity, "gpu_uuid": gpu, "utc": utc()}
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        import json
        json.dump(owner, file)
        file.flush()
        os.fsync(file.fileno())
    try:
        yield
    finally:
        if path.exists() and read(path) == owner:
            path.unlink()


def assert_no_workers(run):
    import psutil
    for path in (run.root / "training_leases").glob("GPU-*.json"):
        owner = read(path)
        if owner["host"] != platform.node() or psutil.pid_exists(owner["pid"]):
            raise ValueError("Training worker may still be active; preserve the lease and inspect processes")


def run_training_and_selection(run, activate_released=None):
    if run.cfg["mode"] != "real":
        raise ValueError("Hybrid execution is server-only; use the explicit mock pipeline locally")
    assert_no_workers(run)
    entries = read(run.root / "manifests/training_jobs.json")
    for job in entries:
        stage = "train:" + job["subset_id"]
        previous = run.state().get(stage, {})
        if previous.get("immutable_artifact_failure") or previous.get("status") == "completed" and not run.valid(stage):
            raise ValueError("Corrupt completed training evidence; do not launch a replacement worker")
    points = read(run.root / "manifests/checkpoints.json")
    gpus = run.cfg["resources"]["gpu_uuids"]
    width = min(len(entries), max(1, len(gpus) // 2))
    urls = list(dict.fromkeys(run.cfg["agent"]["service_endpoints"]))
    pool = EndpointPool(urls[width:], capacity=run.cfg["agent"]["workers"],
                        slots_per_endpoint=run.cfg["agent"].get("tasks_per_gpu", 1))
    abort = threading.Event()
    processes, completed = [], []
    guard = threading.Lock()
    repo = Path(__file__).resolve().parents[1]

    def watcher():
        try:
            pending = [j for j in entries if not run.valid("train:" + j["subset_id"])]
            with contextlib.ExitStack() as stack:
                for start in range(0, len(pending), width):
                    batch = []
                    for job, gpu in zip(pending[start:start + width], gpus[:width]):
                        if abort.is_set():
                            return
                        sid = job["subset_id"]
                        folder = run.root / "logs/parallel-training" / sid / uuid.uuid4().hex
                        folder.mkdir(parents=True)
                        stdout = stack.enter_context((folder / "stdout.log").open("w", encoding="utf-8"))
                        stderr = stack.enter_context((folder / "stderr.log").open("w", encoding="utf-8"))
                        with guard:
                            if abort.is_set():
                                return
                            child = subprocess.Popen([sys.executable, "scripts/train_branch_worker.py",
                                "--snapshot", str(run.root / "config/snapshot.json"), "--subset", sid,
                                "--gpu-uuid", gpu, "--log", str(folder / "events.jsonl"),
                                "--coordinator-pid", str(os.getpid())], cwd=repo,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                            processes.append(child)
                        batch.append((job, child))
                    while any(p.poll() is None for j, p in batch):
                        if abort.is_set():
                            return
                        if any(p.poll() not in {None, 0} for j, p in batch):
                            raise RuntimeError("A training worker failed; inspect logs/parallel-training")
                        time.sleep(.2)
                    for job, process in batch:
                        if process.returncode:
                            raise RuntimeError("Training worker failed: " + job["subset_id"])
                        receipt = read(run.root / "training" / job["subset_id"] / "worker_receipt.json")
                        validate_receipt(run, job["subset_id"], receipt)
                        completed.append((job, receipt))
            if activate_released:
                activate_released()  # Start inference services after every training worker is reaped.
            pool.add(urls[:width])
        except BaseException as exc:
            pool.error = exc

    monitor = threading.Thread(target=watcher, name="training-monitor")
    monitor.start()
    try:
        remaining = [p for p in points if not run.valid("selection:" + p["model_id"])]
        while remaining:
            if pool.error is not None:
                raise RuntimeError("Hybrid training failed") from pool.error
            ready = [p for p in remaining if (run.root / "training" / p["model_id"] / "checkpoint.json").is_file()]
            if not ready:
                if not monitor.is_alive():
                    raise ValueError("Training finished without every milestone")
                time.sleep(.2)
                continue
            for point in ready:
                validate_point(run, point)
                evaluate(run, "selection", point["model_id"], endpoint_pool=pool)
                remaining.remove(point)
        monitor.join()
        if pool.error is not None:
            raise RuntimeError("Hybrid training failed") from pool.error
        for job, receipt in completed:
            with run.stage("train:" + job["subset_id"]) as artifacts:
                if artifacts is None:
                    continue
                for event in read_jsonl(run.root / receipt["worker_log"]):
                    append(run.root / "logs/events.jsonl", event)
                artifacts.extend([run.root / "training" / job["subset_id"] / "worker_receipt.json",
                                  run.root / receipt["worker_log"], *[run.root / n for n in receipt["artifacts"]]])
    finally:
        abort.set()
        with guard:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        monitor.join(timeout=30)
        if monitor.is_alive():
            raise RuntimeError("Training monitor did not exit; inspect worker/service processes")
