"""Independent per-example LoRA gradients on a frozen, authorized GPU pool.

No DDP reduction and no optimizer: every feature retains its own sample/task ID.
Only the coordinator owns Run state; workers publish atomic, verified item caches.
"""

import contextlib
import copy
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

import numpy as np

from nice_validation.core import Run, append, digest, read, sha, utc, write
from nice_validation.gpus import validate_pool
from nice_validation.scoring import normalize
from nice_validation.score_features import (
    candidate_feature,
    validation_feature,
    fingerprint,
)


def inputs(run):
    if not run.valid("prepare-data") or not run.valid("rollout-nice"):
        raise ValueError(
            "Complete prepare-data and rollout-nice before parallel scoring"
        )
    candidates = read(run.root / "manifests/candidates.json")
    tasks = read(run.root / "manifests/validation.json")
    episodes = read(run.root / "rollouts/nice/results.json")
    exp = run.cfg["experiment"]
    if (
        len(candidates) < 2
        or not len(candidates)
        == len({r["data_id"] for r in candidates})
        == exp["candidates"]
    ):
        raise ValueError("Candidate IDs missing, duplicated or wrong count")
    if (
        not tasks
        or not len(tasks) == len({t["instance_id"] for t in tasks}) == exp["validation"]
    ):
        raise ValueError("Validation IDs missing, duplicated or wrong count")
    grouped = {t["instance_id"]: [] for t in tasks}
    for row in episodes:
        if (
            row["task_id"] not in grouped
            or type(row["resolved"]) is not bool
            or row["status"] == "INFRA_ERROR"
            or row.get("run_identity") != run.identity
        ):
            raise ValueError("NICE rollout identity/status mismatch")
        grouped[row["task_id"]].append(row)
    for records in grouped.values():
        if (
            len(records) != exp["rollouts"]
            or any(type(r.get("repeat")) is not int for r in records)
            or sorted(r["repeat"] for r in records) != list(range(exp["rollouts"]))
        ):
            raise ValueError("Missing or duplicated NICE rollout repeat")
        records.sort(key=lambda r: r["repeat"])
    return candidates, tasks, episodes, grouped


def split_ids(rows, identifier, weights, workers):
    shards, loads = [[] for _ in range(workers)], [0] * workers
    for row in sorted(rows, key=lambda r: (-weights[r[identifier]], r[identifier])):
        index = min(range(workers), key=lambda n: (loads[n], len(shards[n]), n))
        shards[index].append(row[identifier])
        loads[index] += weights[row[identifier]]
    return shards


def make_plan(run, candidates, tasks, episodes, grouped):
    gpus = validate_pool(run.cfg.get("resources", {}).get("gpu_uuids", []))
    key = digest(
        {
            "run": run.identity,
            "candidates": digest(candidates),
            "tasks": digest(tasks),
            "episodes": digest(episodes),
            "algorithm": "parallel-per-example-v1",
        }
    )
    c = split_ids(
        candidates,
        "data_id",
        {r["data_id"]: max(1, int(r["input_tokens"])) for r in candidates},
        len(gpus),
    )
    v = split_ids(
        tasks,
        "instance_id",
        {
            t["instance_id"]: max(
                1,
                sum(
                    int(r.get("input_tokens") or 0) + int(r.get("output_tokens") or 0)
                    for r in grouped[t["instance_id"]]
                    if r["resolved"]
                ),
            )
            for t in tasks
        },
        len(gpus),
    )
    reference_task = min(
        (
            t["instance_id"]
            for t in tasks
            if any(r["resolved"] for r in grouped[t["instance_id"]])
        ),
        key=lambda identifier: (
            sum(
                int(r.get("input_tokens") or 0) + int(r.get("output_tokens") or 0)
                for r in grouped[identifier]
                if r["resolved"]
            ),
            identifier,
        ),
        default=tasks[0]["instance_id"],
    )
    return {
        "run_identity": run.identity,
        "input_key": key,
        "mode": run.cfg["mode"],
        "reference_candidate": min(
            candidates, key=lambda r: (r["input_tokens"], r["data_id"])
        )["data_id"],
        "reference_task": reference_task,
        "shards": [
            {"index": i, "gpu_uuid": gpu, "candidate_ids": c[i], "validation_ids": v[i]}
            for i, gpu in enumerate(gpus)
        ],
    }


def item_path(run, kind, identifier):
    # Bounded names also work inside a nested Windows test checkout.
    return (
        run.root / "scores/shards/items" / kind / digest(identifier)[:24] / "cache.json"
    )


def save_arrays(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_item(run, plan, kind, identifier, fp):
    path = item_path(run, kind, identifier)
    if not path.exists():
        return None
    meta = read(path)
    if (
        meta.get("input_key") != plan["input_key"]
        or meta.get("kind") != kind
        or meta.get("id") != identifier
        or meta.get("fingerprint_sha256") != digest(fp)
    ):
        raise ValueError("Gradient cache identity/parameter order mismatch")
    data = (path.parent / meta["arrays_file"]).resolve()
    if (
        data.parent != path.parent.resolve()
        or not data.is_file()
        or sha(data) != meta["arrays_sha256"]
    ):
        raise ValueError(
            "Gradient cache checksum/path mismatch; preserve corrupt evidence"
        )
    names = {"raw", "adam"} if kind == "candidate" else {"validation"}
    with np.load(data, allow_pickle=False) as payload:
        if set(payload.files) != names:
            raise ValueError("Gradient cache array schema mismatch")
        arrays = {name: payload[name] for name in names}
    if any(
        a.shape != (run.cfg["score"]["projection_dim"],)
        or a.dtype != np.float64
        or not np.isfinite(a).all()
        for a in arrays.values()
    ):
        raise ValueError("Invalid projected gradient shape/dtype/value")
    return arrays, meta, [path, path.parent / meta["arrays_file"]]


def compute_item(run, runtime, plan, fp, kind, row, grouped):
    identifier = row["data_id"] if kind == "candidate" else row["instance_id"]
    cached = load_item(run, plan, kind, identifier, fp)
    if cached is not None:
        run.event("score", "gradient_cache_hit", kind=kind, item_id=identifier)
        return cached
    if kind == "candidate":
        arrays, extra = candidate_feature(run, runtime, row)
    else:
        arrays, extra = validation_feature(run, runtime, row, grouped[identifier])
    path = item_path(run, kind, identifier)
    data = path.parent / ("feature-" + uuid.uuid4().hex[:12] + ".npz")
    save_arrays(data, arrays)
    write(
        path,
        {
            "input_key": plan["input_key"],
            "kind": kind,
            "id": identifier,
            "fingerprint_sha256": digest(fp),
            "arrays_file": data.name,
            "arrays_sha256": sha(data),
            **extra,
        },
    )
    return load_item(run, plan, kind, identifier, fp)


@contextlib.contextmanager
def lease(run, index, token=None):
    import psutil

    path = run.root / "scores/shards/leases" / f"shard-{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        owner = read(path)
        if owner["host"] != platform.node() or psutil.pid_exists(owner["pid"]):
            raise ValueError(
                "Scoring shard owner may still be active; refuse concurrent workers"
            )
        path.rename(path.with_name(path.name + ".stale-" + uuid.uuid4().hex[:8]))
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    owner = {
        "pid": os.getpid(),
        "host": platform.node(),
        "run_identity": run.identity,
        "utc": utc(),
        "worker_token": token,
        "process_create_time": psutil.Process().create_time(),
    }
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(owner, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            if read(path) == owner:
                path.unlink()


class ShardRun:
    def __init__(self, parent, gpu, log, reference=False):
        self.root, self.identity = parent.root, parent.identity
        self.cfg = copy.deepcopy(parent.cfg)
        self.cfg["model"]["gpu_uuids"] = [gpu]
        self.gpu, self.log, self.reference = gpu, log, reference

    def event(self, stage, event, **data):
        append(
            self.log,
            {
                "utc": utc(),
                "run_id": self.root.name,
                "stage": "score-reference" if self.reference else stage,
                "event": event,
                "mode": self.cfg["mode"],
                "worker_gpu_uuid": self.gpu,
                **data,
            },
        )


def worker(snapshot, plan_path, index, gpu, attempt, coordinator_pid=None):
    import psutil

    parent = Run(read(snapshot)["config"])
    candidates, tasks, episodes, grouped = inputs(parent)
    plan = read(plan_path)
    if (
        type(index) is not int
        or not 0 <= index < len(plan["shards"])
        or plan != make_plan(parent, candidates, tasks, episodes, grouped)
        or plan["shards"][index]["gpu_uuid"] != gpu
    ):
        raise ValueError("Scoring worker plan/GPU/source mismatch")
    if not attempt.resolve().is_relative_to((parent.root / "scores/shards").resolve()):
        raise ValueError("Scoring worker output escapes shard directory")
    shard, coordinator = plan["shards"][index], coordinator_pid or os.getppid()
    attempt.mkdir(parents=True, exist_ok=True)
    view = ShardRun(parent, gpu, attempt / "events.jsonl")
    by_c = {r["data_id"]: r for r in candidates}
    by_t = {t["instance_id"]: t for t in tasks}
    with lease(parent, index, str(attempt.relative_to(parent.root))):
        runtime = None
        try:
            if parent.cfg["mode"] == "real":
                from nice_validation.model import Runtime

                runtime = Runtime(view.cfg)
            fp = fingerprint(runtime)
            ref_view = ShardRun(parent, gpu, view.log, reference=True)
            ref_c, _ = candidate_feature(
                ref_view, runtime, by_c[plan["reference_candidate"]]
            )
            ref_v, _ = validation_feature(
                ref_view,
                runtime,
                by_t[plan["reference_task"]],
                grouped[plan["reference_task"]],
            )
            reference = attempt / "reference.npz"
            save_arrays(reference, {**ref_c, **ref_v})
            ready = {
                "run_identity": parent.identity,
                "input_key": plan["input_key"],
                "shard": shard,
                "fingerprint": fp,
                "reference_file": str(reference.relative_to(parent.root)),
                "reference_sha256": sha(reference),
            }

            def alive():
                if (attempt.parent / "abort.json").exists():
                    raise RuntimeError("Scoring coordinator cancelled this attempt")
                if not psutil.pid_exists(coordinator):
                    raise RuntimeError(
                        "Scoring coordinator exited; worker stops before more gradients"
                    )

            for identifier in shard["validation_ids"]:
                alive()
                compute_item(
                    view, runtime, plan, fp, "validation", by_t[identifier], grouped
                )
            write(attempt / "validation_ready.json", ready)
            release = attempt.parent / "candidates.go.json"
            while not release.exists():
                alive()
                time.sleep(0.1)
            if read(release) != {
                "input_key": plan["input_key"],
                "run_identity": parent.identity,
            }:
                raise ValueError("Scoring validation barrier identity mismatch")
            for identifier in shard["candidate_ids"]:
                alive()
                compute_item(
                    view, runtime, plan, fp, "candidate", by_c[identifier], grouped
                )
            view.event("score", "shard_completed", shard_index=index)
            write(
                attempt / "receipt.json",
                {
                    **ready,
                    "worker_log": str(view.log.relative_to(parent.root)),
                    "worker_log_sha256": sha(view.log),
                },
            )
        finally:
            if runtime is not None:
                runtime.close()


def validate_ready(run, plan, shard, value):
    if (
        value.get("run_identity") != run.identity
        or value.get("input_key") != plan["input_key"]
        or value.get("shard") != shard
    ):
        raise ValueError("Scoring shard IDs/GPU/identity differ from frozen plan")
    fp = value["fingerprint"]
    if run.cfg["mode"] == "real":
        initial = read(Path(run.cfg["model"]["adapter_path"]) / "identity.json")
        if fp["adapter_identity"] != initial["adapter_sha256"]:
            raise ValueError("Scoring worker used another initial adapter")
    data = (run.root / value["reference_file"]).resolve()
    if (
        not data.is_relative_to(run.root.resolve())
        or not data.is_file()
        or sha(data) != value["reference_sha256"]
    ):
        raise ValueError("Scoring reference checksum/path mismatch")
    with np.load(data, allow_pickle=False) as payload:
        if set(payload.files) != {"raw", "adam", "validation"}:
            raise ValueError("Scoring reference schema mismatch")
        arrays = {n: payload[n] for n in payload.files}
    if any(
        a.shape != (run.cfg["score"]["projection_dim"],)
        or a.dtype != np.float64
        or not np.isfinite(a).all()
        for a in arrays.values()
    ):
        raise ValueError("Invalid scoring reference features")
    return fp, arrays


def check_references(run, plan, ready):
    if len(ready) != len(plan["shards"]):
        raise ValueError("Missing scoring GPU reference receipt")
    reference_fp, reference = validate_ready(run, plan, plan["shards"][0], ready[0])
    maximum = 0.0
    for shard, value in zip(plan["shards"], ready):
        fp, arrays = validate_ready(run, plan, shard, value)
        if fp != reference_fp:
            raise ValueError(
                "Scoring model identity/parameter order differs across GPUs"
            )
        for name, target in reference.items():
            for actual, expected in [
                (arrays[name], target),
                (normalize(arrays[name]), normalize(target)),
            ]:
                maximum = max(maximum, float(np.max(np.abs(actual - expected))))
                if not np.allclose(
                    actual,
                    expected,
                    atol=run.cfg["score"].get("parallel_check_atol", 1e-6),
                    rtol=run.cfg["score"].get("parallel_check_rtol", 1e-4),
                ):
                    raise ValueError("Scoring cross-GPU serial reference mismatch")
    return reference_fp, maximum


def import_events(run, attempts):
    events = run.root / "logs/events.jsonl"
    known = set()
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if value.get("scoring_worker_event_id"):
                known.add(value["scoring_worker_event_id"])
    for attempt in attempts:
        path = attempt / "events.jsonl"
        if not path.exists():
            continue
        for index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(keepends=True)
        ):
            if not line.endswith("\n"):
                run.event(
                    "score",
                    "incomplete_worker_log_tail",
                    path=str(path.relative_to(run.root)),
                )
                break
            key = digest([str(path.relative_to(run.root)), index])
            if key not in known:
                append(events, {**json.loads(line), "scoring_worker_event_id": key})
                known.add(key)


def monitor(processes, paths=None):
    while True:
        for item in processes:
            code = item["process"].poll()
            if code is not None:
                item.setdefault("ended", time.monotonic())
                if code:
                    raise RuntimeError(
                        f"NICE scoring worker {item['index']} failed; inspect {item['attempt']}"
                    )
                if paths is not None and not paths[item["index"]].is_file():
                    raise ValueError("Scoring worker exited without validation receipt")
        if paths is not None and all(p.is_file() for p in paths):
            return
        if paths is None and all(p["process"].poll() is not None for p in processes):
            return
        time.sleep(0.05)


def load_validation(run, plan, tasks, fp):
    features, cost = [], 0
    for task in tasks:
        cached = load_item(run, plan, "validation", task["instance_id"], fp)
        if cached is None:
            raise ValueError("Missing validation gradient item")
        arrays, meta, _ = cached
        features.append(arrays["validation"])
        cost += meta["validation_tokens"]
    matrix = np.stack(features)
    coverage = sum(
        float(np.linalg.norm(v)) > run.cfg["score"]["zero_epsilon"] for v in matrix
    )
    if not coverage:
        write(
            run.root / "scores/failure.json",
            {"reason": "ALL_VALIDATION_DIRECTIONS_ZERO", "tasks": len(tasks)},
        )
        raise ValueError(
            "All validation reward directions are zero; scoring/training blocked"
        )
    return matrix, cost, coverage


def publish(
    run, plan, candidates, tasks, episodes, fp, h, cost, coverage, maximum, out
):
    from scipy.stats import rankdata

    cache_identity = digest(
        {
            "run": run.identity,
            "candidates": digest(candidates),
            "rollouts": digest(episodes),
            "adapter": fp["adapter_identity"],
            "algorithm": "countsketch-v1",
        }
    )
    # Preserve zero tasks and the original task order/denominator. Precompute q
    # once instead of normalizing all validation vectors for every candidate.
    epsilon = run.cfg["score"]["zero_epsilon"]
    q = np.mean([normalize(v, epsilon) for v in h], axis=0)
    scores = []
    for row in candidates:
        cached = load_item(run, plan, "candidate", row["data_id"], fp)
        if cached is None:
            raise ValueError("Missing candidate gradient item")
        arrays, meta, artifacts = cached
        out.extend(artifacts)
        scores.append(
            {
                "data_id": row["data_id"],
                "source_id": row["source_id"],
                "repo": row["repo"],
                "fresh_adam_cosine": float(
                    np.sum(normalize(arrays["adam"], epsilon) * q)
                ),
                "raw_grad_cosine": float(np.sum(normalize(arrays["raw"], epsilon) * q)),
                "supervised_tokens": row["supervised_tokens"],
                "input_tokens": row["input_tokens"],
                "tool_tokens": row["tool_tokens"],
                "max_context": row["max_context"],
                "grad_norm": meta["diagnostic"]["grad_norm"],
                "cache_identity": cache_identity,
                "mode": run.cfg["mode"],
            }
        )
    for task in tasks:
        out.extend(load_item(run, plan, "validation", task["instance_id"], fp)[2])
    ordered = sorted(scores, key=lambda r: (r["fresh_adam_cosine"], r["data_id"]))
    ranks = rankdata([r["fresh_adam_cosine"] for r in ordered], method="average")
    for row, rank in zip(ordered, ranks):
        row["percentile"] = float((rank - 1) / (len(ordered) - 1))
    summary = {
        "run_identity": run.identity,
        "input_key": plan["input_key"],
        "mode": run.cfg["mode"],
        "gpu_uuids": [s["gpu_uuid"] for s in plan["shards"]],
        "candidate_counts": [len(s["candidate_ids"]) for s in plan["shards"]],
        "validation_counts": [len(s["validation_ids"]) for s in plan["shards"]],
        "candidate_count": len(candidates),
        "validation_count": len(tasks),
        "reference_checks_passed": True,
        "reference_max_abs_difference": maximum,
        "fingerprint_sha256": digest(fp),
        "per_example_batch": 1,
        "optimizer_updates": 0,
        "projection_algorithm": "countsketch-v1",
    }
    diagnostics = {
        "zero_gradient_tasks": len(tasks) - coverage,
        "reward_coverage": coverage / len(tasks),
        "warning": (
            "LOW_REWARD_COVERAGE"
            if coverage / len(tasks) < run.cfg["score"]["low_reward_coverage"]
            else None
        ),
        "cache_identity": cache_identity,
        "projection_algorithm": "countsketch-v1",
        "projection_stability": (
            "requires server smoke exact/projected comparison"
            if run.cfg["mode"] == "real"
            else "CPU regression tests"
        ),
        "validation_gradient_replay_tokens": cost,
        "parameter_order": [p["name"] for p in fp["parameters"]],
        "adapter_identity": fp["adapter_identity"],
        "parallel_scoring": summary,
    }
    for name, value in [
        ("candidate_scores", ordered),
        ("validation_features", h.tolist()),
        ("diagnostics", diagnostics),
        ("parallel_summary", summary),
    ]:
        path = run.root / "scores" / (name + ".json")
        write(path, value)
        out.append(path)


def score_parallel(run):
    candidates, tasks, episodes, grouped = inputs(run)
    plan = make_plan(run, candidates, tasks, episodes, grouped)
    with run.stage("score") as out:
        if out is None:
            return
        path = run.root / "scores/shards/plan.json"
        if path.exists() and read(path) != plan:
            raise ValueError("Frozen scoring shard plan differs; use a new run")
        if not path.exists():
            write(path, plan)
        out.append(path)
        folder = run.root / "scores/shards" / ("attempt-" + uuid.uuid4().hex[:12])
        folder.mkdir()
        repo = Path(__file__).resolve().parents[1]
        processes, attempts = [], []
        with contextlib.ExitStack() as handles:
            try:
                for shard in plan["shards"]:
                    index, gpu = shard["index"], shard["gpu_uuid"]
                    attempt = folder / f"worker-{index}"
                    attempt.mkdir()
                    attempts.append(attempt)
                    stdout, stderr = attempt / "stdout.log", attempt / "stderr.log"
                    a = handles.enter_context(stdout.open("w", encoding="utf-8"))
                    b = handles.enter_context(stderr.open("w", encoding="utf-8"))
                    started = time.monotonic()
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "scripts/score_shard_worker.py",
                            "--snapshot",
                            str(run.root / "config/snapshot.json"),
                            "--plan",
                            str(path),
                            "--shard-index",
                            str(index),
                            "--gpu-uuid",
                            gpu,
                            "--attempt",
                            str(attempt),
                            "--coordinator-pid",
                            str(os.getpid()),
                        ],
                        cwd=repo,
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": gpu,
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                        },
                        stdin=subprocess.DEVNULL,
                        stdout=a,
                        stderr=b,
                    )
                    processes.append(
                        {
                            "index": index,
                            "gpu": gpu,
                            "process": process,
                            "started": started,
                            "attempt": attempt,
                        }
                    )
                ready_paths = [a / "validation_ready.json" for a in attempts]
                monitor(processes, ready_paths)
                ready = [read(p) for p in ready_paths]
                fp, maximum = check_references(run, plan, ready)
                h, cost, coverage = load_validation(run, plan, tasks, fp)
                write(
                    folder / "candidates.go.json",
                    {"input_key": plan["input_key"], "run_identity": run.identity},
                )
                monitor(processes)
                for shard, attempt in zip(plan["shards"], attempts):
                    receipt_path = attempt / "receipt.json"
                    receipt = read(receipt_path)
                    validate_ready(run, plan, shard, receipt)
                    if receipt["fingerprint"] != fp:
                        raise ValueError(
                            "Completed shard parameter fingerprint mismatch"
                        )
                    log = attempt / "events.jsonl"
                    if (
                        receipt["worker_log"] != str(log.relative_to(run.root))
                        or sha(log) != receipt["worker_log_sha256"]
                    ):
                        raise ValueError("Scoring worker log checksum/path mismatch")
                    out.extend(
                        [
                            receipt_path,
                            attempt / "validation_ready.json",
                            attempt / "reference.npz",
                            log,
                            attempt / "stdout.log",
                            attempt / "stderr.log",
                        ]
                    )
                publish(
                    run,
                    plan,
                    candidates,
                    tasks,
                    episodes,
                    fp,
                    h,
                    cost,
                    coverage,
                    maximum,
                    out,
                )
                out.append(folder / "candidates.go.json")
            finally:
                if any(p["process"].poll() is None for p in processes):
                    write(
                        folder / "abort.json",
                        {"input_key": plan["input_key"], "run_identity": run.identity},
                    )
                    deadline = time.monotonic() + 2
                    while (
                        any(p["process"].poll() is None for p in processes)
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.05)
                for item in processes:
                    process = item["process"]
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    item.setdefault("ended", time.monotonic())
                    # A Windows venv launcher can have a different Python PID.
                    # Reap only the interpreter with our token and birth time.
                    owned_lease = (
                        run.root
                        / "scores/shards/leases"
                        / f"shard-{item['index']}.json"
                    )
                    if owned_lease.exists():
                        try:
                            owner = read(owned_lease)
                        except FileNotFoundError:
                            owner = {}
                        if (
                            owner.get("worker_token")
                            == str(item["attempt"].relative_to(run.root))
                            and owner.get("host") == platform.node()
                            and owner.get("run_identity") == run.identity
                        ):
                            import psutil

                            with contextlib.suppress(psutil.NoSuchProcess):
                                interpreter = psutil.Process(owner["pid"])
                                if (
                                    interpreter.create_time()
                                    == owner["process_create_time"]
                                ):
                                    try:
                                        interpreter.wait(timeout=15)
                                    except psutil.TimeoutExpired:
                                        interpreter.terminate()
                                        try:
                                            interpreter.wait(timeout=15)
                                        except psutil.TimeoutExpired:
                                            interpreter.kill()
                                            interpreter.wait(timeout=15)
                            with contextlib.suppress(FileNotFoundError):
                                owned_lease.rename(
                                    owned_lease.with_name(
                                        owned_lease.name
                                        + ".stopped-"
                                        + uuid.uuid4().hex[:8]
                                    )
                                )
                    duration = item["ended"] - item["started"]
                    real = run.cfg["mode"] == "real"
                    run.event(
                        "score",
                        "stage_cost",
                        wall_seconds=duration,
                        assigned_gpu_uuids=[item["gpu"]] if real else [],
                        gpu_allocated_seconds=duration if real else None,
                        allocation_definition="one independent NICE scoring worker process; includes model loading, reference checks and validation barrier wait",
                        shard_index=item["index"],
                        returncode=process.returncode,
                    )
                import_events(run, attempts)
