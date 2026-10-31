"""Independent engineering smoke; never consumes the 510 main episodes."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from nice_validation.core import Run, digest, read, write
from nice_validation.design import counts
from nice_validation.scoring import aggregate_score, fresh_adam, sketch


def require_real_smoke(run: Run) -> None:
    if not run.valid("smoke"):
        raise ValueError("Complete independent real smoke before main episodes")
    result = read(run.root / "smoke/result.json")
    capacity = result.get("capacity_probe", {})
    if (
        result.get("mode") != "real"
        or result.get("target_model_tested") is not True
        or not capacity.get("update_completed")
        or capacity.get("input_tokens", 0) < run.cfg["data"]["max_length"]
    ):
        raise ValueError(
            "Real smoke must include an optimizer update at the full candidate context length"
        )


def smoke(run: Run, task_file: str | None = None) -> None:
    with run.stage("smoke") as out:
        if out is None:
            return
        start = time.monotonic()
        if run.cfg["mode"] == "mock":
            rng = np.random.default_rng(42)
            g, h = rng.normal(size=8192), rng.normal(size=(6, 8192))
            dim = run.cfg["score"]["projection_dim"]
            errors = []
            for seed in [42, 43, 44]:
                u = fresh_adam(g, 1e-8, 1)
                exact = aggregate_score(u, h)
                approx = aggregate_score(
                    sketch([u], dim, seed),
                    np.array([sketch([v], dim, seed) for v in h]),
                )
                errors.append(
                    {
                        "seed": seed,
                        "exact": exact,
                        "projected": approx,
                        "absolute_error": abs(exact - approx),
                    }
                )
            result = {
                "mode": "mock",
                "status": "synthetic_fixture",
                "target_model_tested": False,
                "projection_diagnostics": errors,
                "wall_seconds": time.monotonic() - start,
            }
        else:
            task_file = task_file or run.cfg["agent"].get("smoke_task_path")
            if not task_file:
                raise ValueError(
                    "Real smoke requires --task-file with one independent development task"
                )
            from nice_validation.data import tokenize
            from nice_validation.harness import real_episode
            from nice_validation.model import Runtime
            from nice_validation.training import sft_steps
            import torch

            task = read(Path(task_file))
            validation = (
                read(run.root / "manifests" / "validation.json")
                if (run.root / "manifests" / "validation.json").exists()
                else []
            )
            if task["instance_id"] in {t["instance_id"] for t in validation}:
                raise ValueError(
                    "Smoke task must be outside the frozen main validation set"
                )
            runtime = Runtime(run.cfg)
            row = {
                "data_id": "independent_smoke",
                **tokenize(
                    runtime.tokenizer,
                    [
                        {
                            "role": "user",
                            "content": "Write a Python function that returns 1.",
                        },
                        {"role": "assistant", "content": "def one():\n    return 1"},
                    ],
                    run.cfg["model"]["thinking"],
                    run.cfg["data"]["max_length"],
                ),
            }
            runtime.candidate_backward(row)
            original = [
                (
                    p.grad.detach().float().cpu().numpy().reshape(-1)
                    if p.grad is not None
                    else np.zeros(p.numel())
                )
                for _, p in runtime.parameters
            ]
            # Smoke only: concatenate one gradient as an exact reference if RAM permits.
            exact = np.concatenate(original).astype(np.float64)
            adam = fresh_adam(
                exact,
                run.cfg["score"]["adam_epsilon"],
                run.cfg["score"]["max_grad_norm"],
            )
            diagnostics = []
            reference_row = {
                "data_id": "independent_smoke_reference",
                **tokenize(
                    runtime.tokenizer,
                    [
                        {
                            "role": "user",
                            "content": "Write a Python function that returns 2.",
                        },
                        {"role": "assistant", "content": "def two():\n    return 2"},
                    ],
                    run.cfg["model"]["thinking"],
                    run.cfg["data"]["max_length"],
                ),
            }
            runtime.candidate_backward(reference_row)
            reference = np.concatenate(
                [
                    (
                        p.grad.detach().float().cpu().numpy().reshape(-1)
                        if p.grad is not None
                        else np.zeros(p.numel())
                    )
                    for _, p in runtime.parameters
                ]
            ).astype(np.float64)
            for seed in [42, 43, 44]:
                dim = run.cfg["score"]["projection_dim"]
                e = aggregate_score(adam, np.array([reference]))
                a = aggregate_score(
                    sketch([adam], dim, seed),
                    np.array([sketch([reference], dim, seed)]),
                )
                diagnostics.append(
                    {
                        "seed": seed,
                        "exact_cosine": e,
                        "projected_cosine": a,
                        "absolute_error": abs(e - a),
                    }
                )
            folder = run.root / "smoke"
            sft_steps(
                runtime.model,
                runtime.parameters,
                [row],
                runtime.loss,
                run.cfg["training"],
                folder / "training",
                digest({"run": run.identity, "purpose": "smoke"}),
                lambda m: run.event("smoke", "optimizer_step", **m),
            )
            # A short prompt does not exercise the memory footprint of a full
            # candidate. This independent synthetic capacity probe is excluded
            # from all research exposures and is never used for task selection.
            token_count = run.cfg["data"]["max_length"]
            body = runtime.tokenizer(
                "def memory_probe(x):\n    return x\n", add_special_tokens=False
            )["input_ids"]
            if not body:
                raise ValueError("Capacity probe tokenizer returned no text tokens")
            ids = (body * ((token_count + len(body) - 1) // len(body)))[:token_count]
            capacity_row = {
                "data_id": "synthetic-capacity-probe",
                "input_ids": ids,
                "labels": [-100] + ids[1:],
                "input_tokens": token_count,
                "supervised_tokens": token_count - 1,
            }
            torch.cuda.reset_peak_memory_stats()
            sft_steps(
                runtime.model,
                runtime.parameters,
                [capacity_row],
                runtime.loss,
                run.cfg["training"],
                folder / "capacity",
                digest(
                    {
                        "run": run.identity,
                        "purpose": "full-context-capacity",
                        "tokens": token_count,
                    }
                ),
                lambda m: run.event("smoke", "capacity_optimizer_step", **m),
            )
            capacity = {
                "input_tokens": token_count,
                "supervised_tokens": token_count - 1,
                "update_completed": True,
                "kind": "independent_synthetic_capacity_probe",
                "gpu_memory_allocated_peak": torch.cuda.max_memory_allocated(),
                "gpu_memory_reserved_peak": torch.cuda.max_memory_reserved(),
            }
            runtime.model.save_pretrained(folder / "adapter", safe_serialization=True)
            expected = {
                name: p.detach().cpu().clone() for name, p in runtime.parameters
            }
            runtime.close()
            reloaded = Runtime(run.cfg, folder / "adapter")
            reload_equal = all(
                torch.equal(expected[name], p.detach().cpu())
                for name, p in reloaded.parameters
            )
            reloaded.close()
            # Service uses the initial adapter, so this is outside SFT branches.
            episodes_folder = folder / "episodes"
            episode_folder = (
                episodes_folder
                / f"attempt_{len(list(episodes_folder.glob('attempt_*'))):03d}"
            )
            episode_folder.mkdir(parents=True, exist_ok=False)
            outcome = real_episode(run, task, "initial", True, 727, episode_folder)
            trace = read(episode_folder / "trace.json")
            if not trace:
                raise ValueError(
                    "Smoke produced no model calls; generation/replay alignment cannot be verified"
                )
            replay = Runtime(run.cfg)
            alignment = []
            for step in trace[:3]:
                replay.model.zero_grad(set_to_none=True)
                alignment.append(replay.replay_backward(step, 1.0))
            replay.close()
            result = {
                "mode": "real",
                "status": "completed",
                "target_model_tested": True,
                "adapter_reload_equal": reload_equal,
                "capacity_probe": capacity,
                "projection_diagnostics": diagnostics,
                "generation_replay_alignment": alignment,
                "episode": outcome,
                "wall_seconds": time.monotonic() - start,
                "gpu_memory_allocated_peak": torch.cuda.max_memory_allocated(),
                "gpu_memory_reserved_peak": torch.cuda.max_memory_reserved(),
            }
            if not reload_equal:
                raise ValueError("Smoke adapter save/reload mismatch")
        folder = run.root / "smoke"
        write(folder / "result.json", result)
        observed = result["wall_seconds"]
        design = counts(run.cfg)
        estimate = {
            "mode": run.cfg["mode"],
            "observed_smoke_wall_seconds": observed,
            "estimated_remaining_seconds": None,
            "assumptions": "One smoke is insufficient for a reliable episode-tail or training throughput estimate; update with measured main progress. No schedule promise.",
            "main_episode_budget": design["main_episodes"],
            "transfer_episode_budget": design["transfer_episodes"],
            "transfer_component_estimate_seconds": None,
            "episode_estimate_scope": "SWE-smith main tasks only; no extrapolation from a single smoke to Pro/Live/Verified/Terminal",
            "training_trajectory_exposures": design["training_updates"],
        }
        if result.get("episode"):
            seconds = result["episode"]["wall_seconds"]
            estimate.update(
                {
                    "episode_component_estimate_seconds": seconds
                    * design["main_episodes"],
                    "episode_component_range_seconds": [
                        seconds * design["main_episodes"] * 0.5,
                        seconds * design["main_episodes"] * 3,
                    ],
                    "assumptions": "Single episode extrapolation, 1 worker, 0.5–3x sensitivity; excludes gradient scoring/training/download; not a confidence interval.",
                }
            )
        write(run.root / "cost_estimate.json", estimate)
        out.extend([p for p in sorted(folder.rglob("*")) if p.is_file()])
        out.append(run.root / "cost_estimate.json")
