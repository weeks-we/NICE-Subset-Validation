"""Independent LoRA feature calculations shared by sharded scoring and probes."""

import time
import numpy as np

from nice_validation.core import digest, read
from nice_validation.scoring import sketch, fresh_adam, gradient_features


def fingerprint(runtime):
    if runtime is None:
        return {
            "adapter_identity": "mock",
            "model_identity": "mock",
            "parameters": [{"name": "synthetic", "shape": [128], "dtype": "float64"}],
        }
    return {
        "adapter_identity": runtime.adapter_identity,
        "model_identity": runtime.identity,
        "parameters": [
            {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
            for n, p in runtime.parameters
        ],
    }


def validation_feature(run, runtime, task, records):
    cfg, started = run.cfg, time.perf_counter()
    repeats = cfg["experiment"]["rollouts"]
    if len(records) != repeats or any(r["status"] == "INFRA_ERROR" for r in records):
        raise ValueError("Validation rollout incomplete")
    tokens, diagnostic = 0, {}
    if runtime is None:
        rng = np.random.default_rng(int(digest(task["instance_id"])[:8], 16))
        g = rng.normal(size=128) * sum(r["resolved"] for r in records) / repeats
        feature = sketch(
            [g], cfg["score"]["projection_dim"], cfg["score"]["projection_seed"]
        )
    else:
        runtime.model.zero_grad(set_to_none=True)
        for record in records:
            if record["resolved"]:
                path = (run.root / record["evidence_path"] / "trace.json").resolve()
                if not path.is_relative_to(run.root.resolve()):
                    raise ValueError("Replay evidence escapes this run")
                replay = read(path)
                if not replay:
                    raise ValueError("Successful NICE rollout has no replay trace")
                for step in replay:
                    alignment = runtime.replay_backward(step, scale=1 / repeats)
                    run.event(
                        "score",
                        "generation_replay_alignment",
                        task_id=task["instance_id"],
                        **alignment,
                    )
                    tokens += len(step["input_ids"]) + len(step["output_ids"])
        feature, diagnostic = gradient_features(runtime, cfg)
    run.event(
        "score",
        "validation_gradient",
        task_id=task["instance_id"],
        wall_seconds=time.perf_counter() - started,
        **diagnostic,
    )
    return {"validation": feature}, {
        "diagnostic": diagnostic,
        "validation_tokens": tokens,
    }


def candidate_feature(run, runtime, row):
    cfg, started = run.cfg, time.perf_counter()
    if runtime is not None:
        runtime.candidate_backward(row)
        raw, diagnostic = gradient_features(runtime, cfg)
        adam, _ = gradient_features(runtime, cfg, adam=True)
    else:
        rng = np.random.default_rng(int(row["data_id"][:8], 16))
        g = rng.normal(size=128)
        raw = sketch(
            [g], cfg["score"]["projection_dim"], cfg["score"]["projection_seed"]
        )
        adam = sketch(
            [
                fresh_adam(
                    g, cfg["score"]["adam_epsilon"], cfg["score"]["max_grad_norm"]
                )
            ],
            cfg["score"]["projection_dim"],
            cfg["score"]["projection_seed"],
        )
        diagnostic = {
            "grad_norm": float(np.sqrt(np.sum(g**2))),
            "layer_norms": {"synthetic": float(np.sqrt(np.sum(g**2)))},
        }
    run.event(
        "score",
        "candidate_gradient",
        data_id=row["data_id"],
        gradient_replay_tokens=row["input_tokens"],
        wall_seconds=time.perf_counter() - started,
        **diagnostic,
    )
    return {"raw": raw, "adam": adam}, {"diagnostic": diagnostic}
