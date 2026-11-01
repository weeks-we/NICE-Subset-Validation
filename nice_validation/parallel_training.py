"""Independent branch processes; one coordinator publishes immutable Run stages."""

from __future__ import annotations

import contextlib
import copy
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from nice_validation.core import Run, append, read, read_jsonl, sha, utc, write
from nice_validation.gpus import validate_pool
from nice_validation.training import train


class WorkerRun:
    """A bounded worker writes only its branch; the parent owns global state/lock."""

    def __init__(self, parent: Run, subset_id: str, gpu: str, log: Path):
        self.parent = parent
        self.cfg = copy.deepcopy(parent.cfg)
        self.cfg["model"]["gpu_uuids"] = [gpu]
        self.root, self.identity = parent.root, parent.identity
        self.subset_id, self.gpu, self.log = subset_id, gpu, log

    def valid(self, stage):
        return self.parent.valid(stage)

    def event(self, stage, event, **data):
        append(
            self.log,
            {
                "utc": utc(),
                "run_id": self.root.name,
                "stage": stage,
                "event": event,
                "mode": self.cfg["mode"],
                "worker_gpu_uuid": self.gpu,
                **data,
            },
        )

    @contextlib.contextmanager
    def stage(self, name):
        if name != "train:" + self.subset_id:
            raise ValueError("Worker attempted an unauthorized stage")
        artifacts = []
        started = time.monotonic()
        yield artifacts
        if not artifacts or any(not path.is_file() for path in artifacts):
            raise ValueError("Training worker did not publish complete evidence")
        elapsed = time.monotonic() - started
        self.event(
            name,
            "stage_cost",
            wall_seconds=elapsed,
            assigned_gpu_uuids=[self.gpu] if self.cfg["mode"] == "real" else [],
            gpu_allocated_seconds=elapsed if self.cfg["mode"] == "real" else None,
            allocation_definition="one independent training process on one authorized GPU",
        )
        receipt = {
            "run_identity": self.identity,
            "subset_id": self.subset_id,
            "gpu_uuid": self.gpu,
            "worker_log": str(self.log.relative_to(self.root)),
            "worker_log_sha256": sha(self.log),
            "artifacts": {
                str(path.relative_to(self.root)): sha(path) for path in artifacts
            },
        }
        write(self.root / "training" / self.subset_id / "worker_receipt.json", receipt)


def worker(snapshot: Path, subset_id: str, gpu: str, log: Path, coordinator_pid=None):
    cfg = read(snapshot)["config"]
    parent = Run(cfg)
    allowed = cfg.get("resources", {}).get("gpu_uuids", [])
    if gpu not in allowed:
        raise ValueError("Training worker GPU is outside the frozen authorized pool")
    validate_pool(allowed)
    view = WorkerRun(parent, subset_id, gpu, log)
    if cfg["experiment"].get("design") == "verified-selection-v1":
        from nice_validation.experiment_parallel import worker_lease
        view.cfg["_training_coordinator_pid"] = coordinator_pid
        with worker_lease(parent, gpu):
            train(view, subset_id)
    else:
        train(view, subset_id)


def train_parallel(run: Run):
    if not run.valid("make-subsets"):
        raise ValueError("Complete frozen subset selection before parallel training")
    gpus = run.cfg.get("resources", {}).get("gpu_uuids", [])
    if not gpus:
        raise ValueError("Parallel training requires an explicit authorized GPU pool")
    validate_pool(gpus)
    from nice_validation.batch_tuning import tune

    tune(run)
    subsets = read(run.root / "manifests/subsets.json")
    repo = Path(__file__).resolve().parents[1]
    for start in range(0, len(subsets), len(gpus)):
        batch = subsets[start : start + len(gpus)]
        pending = [row for row in batch if not run.valid("train:" + row["subset_id"])]
        if not pending:
            continue
        with run.stage(f"parallel-training:{start // len(gpus)}") as artifacts:
            if artifacts is not None:
                processes = []
                with contextlib.ExitStack() as stack:
                    try:
                        for row, gpu in zip(pending, gpus):
                            sid = row["subset_id"]
                            attempt = (
                                run.root
                                / "logs/parallel-training"
                                / sid
                                / uuid.uuid4().hex
                            )
                            attempt.mkdir(parents=True)
                            stdout = stack.enter_context(
                                (attempt / "stdout.log").open("w", encoding="utf-8")
                            )
                            stderr = stack.enter_context(
                                (attempt / "stderr.log").open("w", encoding="utf-8")
                            )
                            env = {
                                **os.environ,
                                "CUDA_VISIBLE_DEVICES": gpu,
                                "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                            }
                            argv = [
                                sys.executable,
                                "scripts/train_branch_worker.py",
                                "--snapshot",
                                str(run.root / "config/snapshot.json"),
                                "--subset",
                                sid,
                                "--gpu-uuid",
                                gpu,
                                "--log",
                                str(attempt / "events.jsonl"),
                            ]
                            processes.append(
                                (
                                    sid,
                                    subprocess.Popen(
                                        argv,
                                        cwd=repo,
                                        env=env,
                                        stdin=subprocess.DEVNULL,
                                        stdout=stdout,
                                        stderr=stderr,
                                    ),
                                )
                            )
                        for sid, process in processes:
                            if process.wait():
                                raise RuntimeError(
                                    f"Parallel training failed: {sid}; inspect logs/parallel-training"
                                )
                    finally:
                        for _, process in processes:
                            if process.poll() is None:
                                process.terminate()
                                try:
                                    process.wait(timeout=15)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait()
                for row in pending:
                    receipt_path = (
                        run.root / "training" / row["subset_id"] / "worker_receipt.json"
                    )
                    receipt = read(receipt_path)
                    validate_receipt(run, row["subset_id"], receipt)
                    artifacts.extend(
                        [
                            receipt_path,
                            run.root / receipt["worker_log"],
                            *[run.root / name for name in receipt["artifacts"]],
                        ]
                    )
        # Publish serially only after all workers in the batch have been reaped.
        for row in pending:
            sid = row["subset_id"]
            receipt_path = run.root / "training" / sid / "worker_receipt.json"
            receipt = read(receipt_path)
            validate_receipt(run, sid, receipt)
            with run.stage("train:" + sid) as artifacts:
                if artifacts is None:
                    continue
                for event in read_jsonl(run.root / receipt["worker_log"]):
                    append(run.root / "logs/events.jsonl", event)
                artifacts.extend(
                    [
                        receipt_path,
                        run.root / receipt["worker_log"],
                        *[run.root / name for name in receipt["artifacts"]],
                    ]
                )


def validate_receipt(run, subset_id, receipt):
    if (
        receipt.get("run_identity") != run.identity
        or receipt.get("subset_id") != subset_id
        or receipt.get("gpu_uuid") not in run.cfg["resources"]["gpu_uuids"]
    ):
        raise ValueError("Training worker identity differs from coordinator")
    for name, expected in {
        **receipt["artifacts"],
        receipt["worker_log"]: receipt["worker_log_sha256"],
    }.items():
        path = (run.root / name).resolve()
        if (
            not path.is_relative_to(run.root.resolve())
            or not path.is_file()
            or sha(path) != expected
        ):
            raise ValueError("Training worker evidence is missing or corrupt")
