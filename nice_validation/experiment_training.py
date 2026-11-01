"""Four independent LoRA runs, twelve atomic milestone adapters, shared selection."""
import copy
import os
import time
import uuid
from pathlib import Path

from nice_validation.core import read, write, digest, sha
from nice_validation.experiment_design import jobs, points
from nice_validation.batching import batch_spec, shuffled_order, metric_members


def publish_directory(staging, target):
    # Windows readers/virus scanners can briefly hold a directory handle while
    # the coordinator polls for a new milestone. Retry the same atomic rename;
    # never fall back to copying partially published adapter files.
    for attempt in range(8):
        if target.exists():
            raise FileExistsError(target)
        try:
            staging.rename(target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.01 * 2**attempt, .25))


def ensure_jobs(run):
    if not run.valid("make-subsets"):
        raise ValueError("Freeze low/high subsets before training jobs")
    with run.stage("training-jobs") as artifacts:
        if artifacts is None:
            return
        entries = jobs(read(run.root / "manifests/subsets.json"), run.cfg)
        for name, value in [("training_jobs", entries),
                            ("checkpoints", [p for job in entries for p in points(job, run.cfg)])]:
            path = run.root / "manifests" / (name + ".json")
            write(path, value)
            artifacts.append(path)


def validate_point(run, point):
    from nice_validation.model import tree_identity
    folder = run.root / "training" / point["model_id"]
    record = read(folder / "checkpoint.json")
    if any(record.get(k) != v for k, v in point.items()) or record.get("run_identity") != run.identity:
        raise ValueError("Milestone identity differs from the frozen training plan")
    if tree_identity(folder / "adapter") != record["adapter_sha256"]:
        raise ValueError("Milestone adapter checksum mismatch")
    return record


def training_identity(run, job, rows, training_cfg, plan_sha):
    return digest({"run": run.identity, "job": job, "rows": rows,
                   "training": training_cfg, "batch_plan": plan_sha})


def publish_point(run, point, runtime=None, members=None, state_sha=None, training_id=None):
    from nice_validation.model import tree_identity
    target = run.root / "training" / point["model_id"]
    if target.exists():
        record = validate_point(run, point)
        if record.get("optimizer_checkpoint_sha256") != state_sha:
            raise ValueError("Milestone optimizer boundary changed")
        if record.get("training_identity") != training_id:
            raise ValueError("Milestone training/data identity changed")
        return
    staging = target.with_name(".milestone-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    if runtime is None:
        write(staging / "adapter/mock.json", {"synthetic": True, "point": point, "members": members})
    else:
        runtime.model.save_pretrained(staging / "adapter", safe_serialization=True)
    write(staging / "checkpoint.json", {**point, "run_identity": run.identity,
          "mode": run.cfg["mode"], "adapter_sha256": tree_identity(staging / "adapter"),
          "training_identity": training_id,
          "optimizer_checkpoint_sha256": state_sha})
    publish_directory(staging, target)


def train_job(run, job_id):
    from nice_validation.batch_tuning import training_config
    from nice_validation.training import sft_steps, batch_metric, load_checkpoint
    parent = getattr(run, "parent", run)
    if not all(parent.valid(stage) for stage in ["prepare-data", "make-subsets", "training-jobs"]):
        raise ValueError("Complete and preserve frozen data, subsets and training jobs before SFT")
    prior = parent.state().get("train:" + job_id, {})
    if (prior.get("status") == "completed" and not parent.valid("train:" + job_id)) or prior.get("immutable_artifact_failure"):
        raise ValueError("Corrupt completed training evidence must not be overwritten")
    entries = read(run.root / "manifests/training_jobs.json")
    matches = [j for j in entries if j["subset_id"] == job_id]
    if len(matches) != 1:
        raise ValueError("Unknown training job")
    job = matches[0]
    training_cfg, plan_sha = training_config(run)
    training_cfg["learning_rate"] = job["learning_rate"]
    milestones = points(job, run.cfg)
    folder = run.root / "training" / job_id
    rows = [r for r in read(run.root / "manifests/candidates.json") if r["data_id"] in job["members"]]
    if len(rows) != len(set(job["members"])) or {r["data_id"] for r in rows} != set(job["members"]):
        raise ValueError("Missing/duplicate training members")
    with run.stage("train:" + job_id) as artifacts:
        if artifacts is None:
            return
        started = time.monotonic()
        order = shuffled_order(rows)
        effective, _ = batch_spec(training_cfg)
        identity = training_identity(run, job, rows, training_cfg, plan_sha)
        runtime = None
        if run.cfg["mode"] == "mock":
            for index in range(len(rows) // effective):
                batch = [rows[i] for i in order[index * effective:(index + 1) * effective]]
                metric = {**batch_metric(batch, index + 1, training_cfg), "loss": None, "grad_norm": None,
                          "padding_tokens": 0, "learning_rate": job["learning_rate"], "mode": "mock"}
                write(folder / f"step_{index + 1:04d}.metrics.json", metric)
                for point in milestones:
                    if point["step"] == index + 1:
                        publish_point(run, point, members=[rows[i]["data_id"] for i in order[:point["exposures"]]], training_id=identity)
        else:
            import torch
            from nice_validation.model import Runtime
            runtime = Runtime(run.cfg)
            if runtime.initial_identity["adapter_sha256"] != read(run.root / "scores/diagnostics.json")["adapter_identity"]:
                raise ValueError("SFT and NICE must share the initial LoRA")

            def publish_from_state(point):
                path = folder / f"step_{point['step']:04d}.pt"
                if not path.with_suffix(".json").is_file():
                    return
                if (run.root / "training" / point["model_id"]).exists():
                    previous = validate_point(run, point)
                    if previous["optimizer_checkpoint_sha256"] != sha(path) or previous.get("training_identity") != identity:
                        raise ValueError("Published milestone differs from optimizer state")
                    return
                state = load_checkpoint(path, identity)
                saved = {n: p.detach().clone() for n, p in runtime.parameters}
                try:
                    with torch.no_grad():
                        for n, p in runtime.parameters:
                            p.copy_(state["adapter"][n].to(p.device))
                    publish_point(run, point, runtime=runtime, state_sha=sha(path), training_id=identity)
                finally:
                    with torch.no_grad():
                        for n, p in runtime.parameters:
                            p.copy_(saved[n])

            def on_step(metric):
                run.event("train:" + job_id, "optimizer_step", subset_id=job_id, **metric)
                for point in milestones:
                    if point["step"] == metric["step"]:
                        publish_from_state(point)

            def intent(metric):
                coordinator = run.cfg.get("_training_coordinator_pid")
                if coordinator:
                    import psutil
                    if not psutil.pid_exists(coordinator):
                        raise RuntimeError("Training coordinator exited; worker stops before next update")
                run.event("train:" + job_id, "training_forward_intent", subset_id=job_id, **metric)

            try:
                # Repair publication after interruption using trusted optimizer-boundary states.
                for point in milestones:
                    publish_from_state(point)
                sft_steps(runtime.model, runtime.parameters, rows, runtime.loss, training_cfg,
                          folder, identity, on_step, on_intent=intent)
                for point in milestones:
                    publish_from_state(point)
            finally:
                runtime.close()
        metrics = [read(p) for p in sorted(folder.glob("step_*.metrics.json"))]
        seen = [member for m in metrics for member in metric_members(m)]
        if len(metrics) != len(rows) // effective or len(seen) != len(set(seen)) or set(seen) != set(job["members"]):
            raise ValueError("Training exposure/update coverage mismatch")
        records = [validate_point(run, p) for p in milestones]
        report = {"subset_id": job_id, "group": job["group"], "learning_rate": job["learning_rate"],
                  "training_identity": identity,
                  "status": "completed", "mode": run.cfg["mode"], "members": job["members"],
                  "updates": len(metrics), "exposures": len(seen), "effective_batch": effective,
                  "micro_batch_size": training_cfg["micro_batch_size"], "batch_plan_sha256": plan_sha,
                  "checkpoint_models": [r["model_id"] for r in records],
                  "logical_input_tokens": sum(m["logical_input_tokens"] for m in metrics),
                  "supervised_tokens": sum(m["supervised_tokens"] for m in metrics),
                  "padding_tokens": sum(m["padding_tokens"] for m in metrics),
                  "wall_seconds": time.monotonic() - started}
        write(folder / "training.json", report)
        artifacts.extend([folder / "training.json", *sorted(folder.glob("step_*.metrics.json")),
                          *sorted(folder.glob("step_*.pt")), *sorted(folder.glob("step_*.json"))])
        for point in milestones:
            artifacts.extend(p for p in (run.root / "training" / point["model_id"]).rglob("*") if p.is_file())
