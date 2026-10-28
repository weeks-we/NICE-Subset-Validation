"""Held-out benchmark evaluation, paired by task and separate from NICE selection."""

import math
from pathlib import Path
import subprocess
import sys
import time

from nice_validation.core import Run, digest, read, sha, write
from nice_validation.harness import InfraError
from nice_validation.transfer_sources import SUITES, validate_native_task


def native_job(
    task: dict,
    folder: Path,
    config_file: Path,
    adapter: Path,
    patch: Path | None = None,
    expected_images: dict | None = None,
) -> dict:
    agent = {
        "import_path": "nice_validation.harbor_agent:NiceAgent",
        "skip_model_parsing": True,
        "override_timeout_sec": 1290,
        "kwargs": {
            "config_file": str(config_file),
            "adapter_path": str(adapter.resolve()),
            "capture_patch": task["benchmark"] == "pro",
            "expected_images": expected_images,
        },
    }
    if patch is not None:
        agent["import_path"] = "nice_validation.harbor_agent:NicePatchReplay"
        agent["kwargs"] = {
            "patch_path": str(patch.resolve()),
            "patch_sha256": sha(patch),
            "expected_images": expected_images,
        }
    return {
        "job_name": "regrade" if patch is not None else "generation",
        "jobs_dir": str(folder.resolve()),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "retry": {"max_retries": 0},
        "environment": {"type": "docker", "delete": True},
        "agents": [agent],
        "tasks": [{"path": task["task_path"]}],
    }


def one_native_result(job_dir: Path, task_id: str) -> tuple[dict, Path]:
    matches = []
    for path in job_dir.glob("*/result.json"):
        value = read(path)
        if value.get("task_name", "").split("/")[-1] == task_id:
            matches.append((value, path))
    if len(matches) != 1:
        raise InfraError("native_trial_missing_or_duplicated")
    return matches[0]


def reward(value: dict) -> float:
    if value.get("exception_info"):
        kind = value["exception_info"].get("exception_type", "unknown")
        if kind == "AgentTimeoutError":
            return 0.0
        raise InfraError("native_harness:" + kind)
    rewards = (value.get("verifier_result") or {}).get("rewards") or {}
    observed = rewards.get("reward")
    if (
        not isinstance(observed, (int, float))
        or not math.isfinite(observed)
        or not 0 <= observed <= 1
    ):
        raise InfraError("native_verifier_reward_missing_or_invalid")
    return float(observed)


def native_episode(run: Run, task: dict, model_id: str, folder: Path) -> dict:
    from nice_validation.model import server_only

    server_only(run.cfg)
    validate_native_task(task)
    image_lock = read(run.root / "transfer/native_images.json")
    proof = image_lock[task["benchmark"] + ":" + task["instance_id"]]
    start = time.monotonic()
    adapter = (
        Path(run.cfg["model"]["adapter_path"])
        if model_id == "initial"
        else run.root / "training" / model_id / "adapter"
    )
    runtime_config = folder / "runtime.local.yaml"
    write(runtime_config, run.cfg)
    jobs = folder / "native_jobs"

    def execute(spec, filename):
        destination = folder / filename
        write(destination, spec)
        with (
            (folder / (filename + ".stdout.log")).open("w", encoding="utf-8") as out,
            (folder / (filename + ".stderr.log")).open("w", encoding="utf-8") as err,
        ):
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "nice_validation.harbor_runner",
                        str(destination.resolve()),
                    ],
                    stdout=out,
                    stderr=err,
                    check=True,
                    timeout=run.cfg["agent"]["wall_seconds"]
                    + run.cfg["agent"]["evaluation_timeout"]
                    + 1800,
                )
            except subprocess.SubprocessError as exc:
                raise InfraError("native_harness_process") from exc

    generated = native_job(
        task, jobs, runtime_config.resolve(), adapter, expected_images=proof
    )
    generated["agents"][0]["override_timeout_sec"] = (
        run.cfg["agent"]["wall_seconds"] + 90
    )
    execute(generated, "generation-job.json")
    original, result_path = one_native_result(jobs / "generation", task["instance_id"])
    first_reward = reward(original)
    agent_dir = result_path.parent / "agent"
    episode = (
        read(agent_dir / "episode.json")
        if (agent_dir / "episode.json").is_file()
        else {}
    )
    raw_reward = first_reward
    if task["benchmark"] == "pro" and not original.get("exception_info"):
        patch = agent_dir / "model.patch"
        if not patch.is_file():
            raise InfraError("pro_patch_evidence_missing")
        execute(
            native_job(task, jobs, runtime_config.resolve(), adapter, patch, proof),
            "regrade-job.json",
        )
        regraded, _ = one_native_result(jobs / "regrade", task["instance_id"])
        raw_reward = reward(regraded)
    failed = episode.get("task_failure", False) or bool(original.get("exception_info"))
    effective = 0.0 if failed else raw_reward
    resolved = effective == 1.0
    return {
        "status": "RESOLVED" if resolved else "TASK_FAILED",
        "resolved": resolved,
        "native_reward": effective,
        "raw_verifier_reward": raw_reward,
        "in_sandbox_reward": first_reward,
        "stop_reason": episode.get("stop_reason", "NATIVE_AGENT_TIMEOUT"),
        "input_tokens": episode.get("input_tokens"),
        "output_tokens": episode.get("output_tokens"),
        "wall_seconds": time.monotonic() - start,
        "benchmark": task["benchmark"],
        "pristine_regrade": task["benchmark"] == "pro",
    }


def evaluate_transfer(run: Run, model_id: str) -> None:
    if not run.cfg.get("transfer", {}).get("enabled"):
        return
    if not run.valid("prepare-transfer"):
        raise ValueError("Freeze transfer task samples before evaluation")
    if model_id != "initial" and not run.valid("train:" + model_id):
        raise ValueError(
            "Finish the selected training branch before transfer evaluation"
        )
    if run.cfg["mode"] == "real":
        from nice_validation.smoke import require_real_smoke

        require_real_smoke(run)
        if not run.valid("prepare-environments"):
            raise ValueError("Complete main and transfer environment preparation")
        frozen = read(run.root / "manifests/validation_freeze.json")
        if (
            not run.valid("prepare-native-environments")
            or frozen["environment_lock_sha256"]
            != sha(Path(run.cfg["agent"]["image_revision_lock"]))
            or frozen["validation_sha256"]
            != sha(run.root / "manifests/validation.json")
        ):
            raise ValueError("Frozen main/transfer environment evidence changed")
    selections = read(run.root / "transfer/tasks.json")
    for suite in SUITES:
        stage = f"transfer:{suite}:{model_id}"
        with run.stage(stage) as artifacts:
            if artifacts is None:
                continue
            folder = run.root / "transfer/evaluation" / suite / model_id
            tasks = selections[suite]
            if run.cfg["mode"] == "mock":
                # Synthetic results contain no fabricated model traces or token costs.
                results = []
                for task in tasks:
                    resolved = (
                        int(digest([suite, model_id, task["instance_id"]])[:8], 16) % 3
                        == 0
                    )
                    results.append(
                        {
                            "task_id": task["instance_id"],
                            "model_id": model_id,
                            "benchmark": suite,
                            "resolved": resolved,
                            "status": "RESOLVED" if resolved else "TASK_FAILED",
                            "native_reward": float(resolved),
                            "mode": "mock",
                            "run_identity": run.identity,
                        }
                    )
                    run.event(stage, "episode_attempt", **results[-1])
            else:
                import copy
                from nice_validation.harness import _episode_job

                workers = run.cfg["agent"]["workers"]
                endpoints = run.cfg["agent"]["service_endpoints"]

                def worker_shard(worker_index):
                    scoped = copy.copy(run)
                    scoped.cfg = copy.deepcopy(run.cfg)
                    scoped.cfg["_evaluation_scope"] = stage
                    scoped.cfg["agent"]["service_url"] = endpoints[worker_index]
                    return [
                        _episode_job(scoped, stage, model_id, False, i, task, 0, folder)
                        for i, task in enumerate(tasks)
                        if i % workers == worker_index
                    ]

                if workers == 1:
                    results = worker_shard(0)
                else:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        results = [
                            row
                            for shard in pool.map(worker_shard, range(workers))
                            for row in shard
                        ]
                results.sort(key=lambda row: row["task_id"])
            destination = folder / "results.json"
            write(destination, results)
            artifacts.extend(p for p in sorted(folder.rglob("*")) if p.is_file())
            if len(results) != 50 or any(r["status"] == "INFRA_ERROR" for r in results):
                raise ValueError(
                    "Transfer evaluation incomplete; retain fixed denominator and repair infrastructure"
                )


def prepare_native_environments(run: Run) -> None:
    if not run.cfg.get("transfer", {}).get("enabled"):
        return
    from nice_validation.model import server_only

    server_only(run.cfg)
    from nice_validation.transfer_sources import transfer_tasks

    with run.stage("prepare-native-environments") as artifacts:
        if artifacts is None:
            return
        proofs = {}
        for task in transfer_tasks(run):
            if task["evaluation_backend"] != "harbor":
                continue
            validate_native_task(task)
            base_folder = (
                run.root
                / "transfer/preflight"
                / task["benchmark"]
                / task["instance_id"]
            )
            folder = (
                base_folder / f"attempt_{len(list(base_folder.glob('attempt_*'))):03d}"
            )
            cfg_file = folder / "runtime.local.yaml"
            write(cfg_file, run.cfg)
            job = native_job(
                task,
                folder / "jobs",
                cfg_file.resolve(),
                Path(run.cfg["model"]["adapter_path"]),
            )
            job["install_only"] = True
            job_file = folder / "job.json"
            write(job_file, job)
            with (
                (folder / "stdout.log").open("w", encoding="utf-8") as out,
                (folder / "stderr.log").open("w", encoding="utf-8") as err,
            ):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "nice_validation.harbor_runner",
                        str(job_file.resolve()),
                    ],
                    stdout=out,
                    stderr=err,
                    check=True,
                    timeout=3600,
                )
            value, result = one_native_result(
                folder / "jobs/generation", task["instance_id"]
            )
            if value.get("exception_info"):
                raise ValueError(
                    "Native environment preflight failed; preserve the selected task and repair environment"
                )
            evidence = result.parent / "agent/environment_images.json"
            proofs[task["benchmark"] + ":" + task["instance_id"]] = read(evidence)
        destination = run.root / "transfer/native_images.json"
        write(destination, proofs)
        artifacts.append(destination)
        artifacts.extend(
            p
            for p in sorted((run.root / "transfer/preflight").rglob("*"))
            if p.is_file()
        )
