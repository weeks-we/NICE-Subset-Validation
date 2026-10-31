"""Measure physical batches on every authorized GPU before any branch SFT."""

import copy
import math
import os
import random
from pathlib import Path
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from nice_validation.core import Run, read, write, sha
from nice_validation.batching import batch_spec, backward_batch


def choose_micro(reports, candidates):
    """Use a common feasible size; within 5% of fastest prefer lower memory."""
    feasible = {}
    for micro in candidates:
        rates = []
        for report in reports:
            matches = [t for t in report["trials"] if t["micro_batch_size"] == micro]
            if len(matches) != 1:
                raise ValueError("Missing or duplicated micro-batch trial")
            trial = matches[0]
            if trial["status"] != "PASS":
                if trial["status"] != "OOM":
                    raise ValueError("Unknown micro-batch trial status")
                break
            if trial.get("capacity_update_completed") is not True:
                raise ValueError("Missing micro-batch capacity evidence")
            rate = trial["supervised_tokens_per_second"]
            if (
                not isinstance(rate, (float, int))
                or not math.isfinite(rate)
                or rate <= 0
            ):
                raise ValueError("Invalid measured training throughput")
            rates.append(rate)
        if len(rates) == len(reports) and rates:
            feasible[micro] = sum(rates)
    if not feasible:
        raise ValueError("No micro-batch passed capacity and throughput on every GPU")
    fastest = max(feasible.values())
    return min(n for n, rate in feasible.items() if rate >= fastest * 0.95)


def validate_plan(run, plan):
    effective, micro = batch_spec(run.cfg["training"])
    if (
        plan.get("run_identity") != run.identity
        or plan.get("mode") != run.cfg["mode"]
        or plan.get("effective_batch") != effective
        or plan.get("gpu_uuids") != run.cfg.get("resources", {}).get("gpu_uuids", [])
        or plan.get("selected_micro_batch_size")
        not in run.cfg["training"].get("micro_batch_candidates", [1, 2, 4])
    ):
        raise ValueError("Training batch plan identity/configuration mismatch")
    selected = plan["selected_micro_batch_size"]
    if micro != "auto" or effective % selected:
        raise ValueError("Training batch plan does not match auto policy")
    if run.cfg["mode"] == "real":
        reports = plan.get("reports", [])
        if [r.get("gpu_uuid") for r in reports] != plan["gpu_uuids"] or any(
            r.get("run_identity") != run.identity for r in reports
        ):
            raise ValueError("Training batch plan is missing GPU evidence")
        if (
            choose_micro(reports, run.cfg["training"]["micro_batch_candidates"])
            != selected
        ):
            raise ValueError("Training batch plan selection differs from measurements")


def training_config(run):
    cfg = copy.deepcopy(run.cfg["training"])
    _, micro = batch_spec(cfg)
    if micro != "auto":
        return cfg, None
    if not run.valid("tune-training-batch"):
        raise ValueError("Complete tune-training-batch before branch training")
    path = run.root / "training_batch/plan.json"
    plan = read(path)
    validate_plan(run, plan)
    cfg["micro_batch_size"] = plan["selected_micro_batch_size"]
    return cfg, sha(path)


def tune(run):
    effective, micro = batch_spec(run.cfg["training"])
    if micro != "auto":
        return
    if not run.valid("make-subsets"):
        raise ValueError("Freeze subsets before tuning training throughput")
    if any(
        run.valid("train:" + s["subset_id"])
        for s in read(run.root / "manifests/subsets.json")
    ) and not run.valid("tune-training-batch"):
        raise ValueError("Cannot choose a new batch after branch training")
    from nice_validation.gpus import validate_pool

    gpus = validate_pool(run.cfg.get("resources", {}).get("gpu_uuids", []))
    with run.stage("tune-training-batch") as out:
        if out is None:
            validate_plan(run, read(run.root / "training_batch/plan.json"))
            return
        candidates = run.cfg["training"]["micro_batch_candidates"]
        reports = []
        if run.cfg["mode"] == "mock":
            # No fabricated speeds/capacity claims. Deterministic execution fixture.
            selected = min(candidates, key=lambda n: (abs(n - 2), n))
        else:
            repo = Path(__file__).resolve().parents[1]
            folder = run.root / "training_batch" / ("attempt-" + uuid.uuid4().hex)
            folder.mkdir(parents=True)

            def probe(item):
                index, gpu = item
                output = folder / f"gpu-{index}.json"
                stdout, stderr = (
                    folder / f"gpu-{index}.stdout.log",
                    folder / f"gpu-{index}.stderr.log",
                )
                with (
                    stdout.open("w", encoding="utf-8") as a,
                    stderr.open("w", encoding="utf-8") as b,
                ):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "scripts/training_batch_worker.py",
                            "--snapshot",
                            str(run.root / "config/snapshot.json"),
                            "--gpu-uuid",
                            gpu,
                            "--output",
                            str(output),
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
                        timeout=1800,
                    )
                if result.returncode:
                    raise RuntimeError(
                        f"Training batch probe failed on {gpu}; inspect {stderr}"
                    )
                report = read(output)
                if (
                    report.get("gpu_uuid") != gpu
                    or report.get("run_identity") != run.identity
                ):
                    raise ValueError("Batch probe returned another run/GPU identity")
                return report, [output, stdout, stderr]

            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                for report, artifacts in pool.map(probe, enumerate(gpus)):
                    reports.append(report)
                    out.extend(artifacts)
                    run.event(
                        "tune-training-batch",
                        "stage_cost",
                        wall_seconds=report["wall_seconds"],
                        assigned_gpu_uuids=[report["gpu_uuid"]],
                        gpu_allocated_seconds=report["wall_seconds"],
                        allocation_definition="disposable physical-batch profiling process; excluded from formal SFT exposures",
                    )
            selected = choose_micro(reports, candidates)
        plan = {
            "mode": run.cfg["mode"],
            "run_identity": run.identity,
            "gpu_uuids": gpus,
            "effective_batch": effective,
            "selected_micro_batch_size": selected,
            "gradient_accumulation_steps": effective // selected,
            "selection_rule": "smallest common size within 5% of fastest aggregate supervised-token throughput",
            "formal_capacity_evidence": run.cfg["mode"] == "real",
            "reports": reports,
        }
        validate_plan(run, plan)
        path = run.root / "training_batch/plan.json"
        write(path, plan)
        run.event(
            "tune-training-batch",
            "batch_plan_selected",
            effective_batch=effective,
            micro_batch_size=selected,
            gradient_accumulation_steps=effective // selected,
        )
        out.append(path)


def profile(snapshot, gpu, output):
    """Independent throwaway updates; never select on loss or validation answers."""
    import torch
    from nice_validation.model import Runtime

    parent = Run(read(snapshot)["config"])
    profile_started = time.monotonic()
    if gpu not in parent.cfg["resources"]["gpu_uuids"]:
        raise ValueError("Unauthorized profiling GPU")
    cfg = copy.deepcopy(parent.cfg)
    cfg["model"]["gpu_uuids"] = [gpu]
    members = {
        m for s in read(parent.root / "manifests/subsets.json") for m in s["members"]
    }
    rows = sorted(
        [
            r
            for r in read(parent.root / "manifests/candidates.json")
            if r["data_id"] in members
        ],
        key=lambda r: (r["input_tokens"], r["data_id"]),
    )
    effective, _ = batch_spec(cfg["training"])
    if not rows:
        raise ValueError("No frozen training rows to profile")
    # Same token-length quantiles, with repetition only inside disposable probes.
    probe_rows = [
        rows[round(i * (len(rows) - 1) / max(1, 2 * effective - 1))]
        for i in range(2 * effective)
    ]
    random.Random(2026).shuffle(probe_rows)
    runtime = Runtime(cfg)
    initial = {n: p.detach().cpu().clone() for n, p in runtime.parameters}
    trials = []
    try:
        for micro in cfg["training"]["micro_batch_candidates"]:
            training = {**cfg["training"], "micro_batch_size": micro}
            optimizer = None
            try:
                with torch.no_grad():
                    for n, p in runtime.parameters:
                        p.copy_(initial[n].to(p.device))
                torch.manual_seed(2026)
                runtime.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                optimizer = torch.optim.AdamW(
                    [p for _, p in runtime.parameters],
                    lr=training["learning_rate"],
                    betas=tuple(training["betas"]),
                    eps=training["epsilon"],
                    weight_decay=0,
                )

                def step(batch):
                    backward_batch(
                        runtime.model, runtime.parameters, batch, runtime.loss, training
                    )
                    optimizer.step()
                    if any(
                        not bool(torch.isfinite(p).all()) for _, p in runtime.parameters
                    ):
                        raise ValueError("Nonfinite adapter in disposable batch probe")

                # Worst combination of actual selected lengths, including Adam state.
                step([rows[-1]] * effective)
                step(probe_rows[:effective])  # untimed warmup
                torch.cuda.synchronize()
                started = time.perf_counter()
                for i in range(0, len(probe_rows), effective):
                    step(probe_rows[i : i + effective])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                if seconds <= 0:
                    raise ValueError(
                        "No positive profiling duration; cannot select a batch"
                    )
                tokens = sum(r["supervised_tokens"] for r in probe_rows)
                trials.append(
                    {
                        "micro_batch_size": micro,
                        "status": "PASS",
                        "seconds": seconds,
                        "supervised_tokens": tokens,
                        "supervised_tokens_per_second": tokens / seconds,
                        "capacity_input_tokens_per_row": rows[-1]["input_tokens"],
                        "capacity_update_completed": True,
                        "gpu_memory_allocated_peak": torch.cuda.max_memory_allocated(),
                        "gpu_memory_reserved_peak": torch.cuda.max_memory_reserved(),
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                trials.append(
                    {
                        "micro_batch_size": micro,
                        "status": "OOM",
                        "error": str(exc),
                        "supervised_tokens_per_second": None,
                    }
                )
            finally:
                del optimizer
                runtime.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
    finally:
        runtime.close()
    write(
        output,
        {
            "run_identity": parent.identity,
            "gpu_uuid": gpu,
            "trials": trials,
            "probe_data_ids": [r["data_id"] for r in probe_rows],
            "excluded_from_research_training": True,
            "wall_seconds": time.monotonic() - profile_started,
        },
    )
