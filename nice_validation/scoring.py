"""NICE-style directions, streaming CountSketch and fixed-denominator cosine."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

import numpy as np

from nice_validation.core import Run, digest, read, write


def normalize(vector: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    length = float(np.sqrt(np.sum(np.asarray(vector, dtype=np.float64) ** 2)))
    return vector / length if length > epsilon else np.zeros_like(vector)


def fresh_adam(gradient: np.ndarray, epsilon: float, clip: float) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=np.float64)
    length = float(np.sqrt(np.sum(gradient**2)))
    clipped = gradient * min(1.0, clip / (length + 1e-6))
    return clipped / (np.abs(clipped) + epsilon)


def aggregate_score(
    direction: np.ndarray, validation: np.ndarray, epsilon: float = 1e-12
) -> float:
    q = np.mean([normalize(row, epsilon) for row in validation], axis=0)
    # q must not be normalized again: zero tasks remain in the denominator.
    return float(np.sum(normalize(direction, epsilon) * q))


def explicit_score(
    direction: np.ndarray, validation: np.ndarray, epsilon: float = 1e-12
) -> float:
    return float(
        np.mean(
            [
                np.sum(normalize(direction, epsilon) * normalize(row, epsilon))
                for row in validation
            ]
        )
    )


def sketch(chunks: Iterable[np.ndarray], dim: int, seed: int) -> np.ndarray:
    """CountSketch-v1; coordinate hash, fixed global offsets, bounded block memory.

    Hash arithmetic wraps uint64. Every original coordinate maps to one bucket
    with a random sign. Chunk boundaries do not change the projection.
    """
    projected = np.zeros(dim, dtype=np.float64)
    offset = 0
    for chunk in chunks:
        flat = np.asarray(chunk, dtype=np.float64).reshape(-1)
        for start in range(0, len(flat), 262144):
            values = flat[start : start + 262144]
            idx = np.arange(
                offset + start, offset + start + len(values), dtype=np.uint64
            )
            with np.errstate(over="ignore"):
                bits = idx + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
                bits = (bits ^ (bits >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                bits = (bits ^ (bits >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                bits ^= bits >> np.uint64(31)
            buckets = (bits % np.uint64(dim)).astype(np.int64)
            signs = np.where((bits >> np.uint64(63)) == 0, 1.0, -1.0)
            projected += np.bincount(buckets, weights=values * signs, minlength=dim)
        offset += len(flat)
    if not np.all(np.isfinite(projected)):
        raise ValueError("Nonfinite projected gradients")
    return projected


def gradient_features(
    runtime, cfg: dict, adam: bool = False
) -> tuple[np.ndarray, dict]:
    norm_sq, layer_norms = 0.0, {}
    for name, param in runtime.parameters:
        norm = (
            float(param.grad.detach().float().norm()) if param.grad is not None else 0.0
        )
        layer_norms[name] = norm
        norm_sq += norm * norm
    norm = math.sqrt(norm_sq)
    if not math.isfinite(norm):
        raise ValueError("Nonfinite gradients")
    clip = min(1.0, cfg["score"]["max_grad_norm"] / (norm + 1e-6))

    def chunks():
        for _, parameter in runtime.parameters:
            values = (
                parameter.grad.detach().float().cpu().numpy().reshape(-1)
                if parameter.grad is not None
                else np.zeros(parameter.numel())
            )
            for start in range(0, len(values), 262144):
                block = values[start : start + 262144].astype(np.float64)
                if adam:
                    block *= clip
                    block = block / (np.abs(block) + cfg["score"]["adam_epsilon"])
                yield block

    return sketch(
        chunks(), cfg["score"]["projection_dim"], cfg["score"]["projection_seed"]
    ), {"grad_norm": norm, "layer_norms": layer_norms, "clipping_factor": clip}


def score(run: Run) -> None:
    if not run.valid("prepare-data") or not run.valid("rollout-nice"):
        raise ValueError("Complete prepare-data and rollout-nice first")
    with run.stage("score") as out:
        if out is None:
            return
        cfg = run.cfg
        candidates = read(run.root / "manifests" / "candidates.json")
        tasks = read(run.root / "manifests" / "validation.json")
        episodes = read(run.root / "rollouts" / "nice" / "results.json")
        runtime = None
        if cfg["mode"] == "real":
            from nice_validation.model import Runtime

            runtime = Runtime(cfg)
        h, coverage = [], 0
        validation_cost = 0
        for task in tasks:
            records = [r for r in episodes if r["task_id"] == task["instance_id"]]
            if len(records) != 4 or any(r["status"] == "INFRA_ERROR" for r in records):
                raise ValueError(
                    "Validation rollout incomplete; fixed four-episode reward estimate unavailable"
                )
            if cfg["mode"] == "mock":
                rng = np.random.default_rng(int(digest(task["instance_id"])[:8], 16))
                direction = (
                    rng.normal(size=128) * sum(r["resolved"] for r in records) / 4
                )
                h.append(
                    sketch(
                        [direction],
                        cfg["score"]["projection_dim"],
                        cfg["score"]["projection_seed"],
                    )
                )
            else:
                runtime.model.zero_grad(set_to_none=True)
                for record in records:
                    if record["resolved"]:
                        replay = read(run.root / record["evidence_path"] / "trace.json")
                        for step in replay:
                            alignment = runtime.replay_backward(step, scale=0.25)
                            run.event(
                                "score",
                                "generation_replay_alignment",
                                task_id=task["instance_id"],
                                **alignment,
                            )
                            validation_cost += len(step["input_ids"]) + len(
                                step["output_ids"]
                            )
                feature, diagnostic = gradient_features(runtime, cfg)
                run.event(
                    "score",
                    "validation_gradient",
                    task_id=task["instance_id"],
                    **diagnostic,
                )
                h.append(feature)
            coverage += int(
                float(np.sqrt(np.sum(h[-1] ** 2))) > cfg["score"]["zero_epsilon"]
            )
        h = np.asarray(h)
        if coverage == 0:
            write(
                run.root / "scores" / "failure.json",
                {"reason": "ALL_VALIDATION_DIRECTIONS_ZERO", "tasks": 30},
            )
            raise ValueError(
                "All validation reward directions are zero; scoring/training blocked"
            )
        cache_identity = digest(
            {
                "run": run.identity,
                "candidates": digest(candidates),
                "rollouts": digest(episodes),
                "adapter": runtime.adapter_identity if runtime else "mock",
                "algorithm": "countsketch-v1",
            }
        )
        scores = []
        for row in candidates:
            started = time.monotonic()
            if runtime:
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
                            g,
                            cfg["score"]["adam_epsilon"],
                            cfg["score"]["max_grad_norm"],
                        )
                    ],
                    cfg["score"]["projection_dim"],
                    cfg["score"]["projection_seed"],
                )
                diagnostic = {
                    "grad_norm": float(np.sqrt(np.sum(g**2))),
                    "layer_norms": {"synthetic": float(np.sqrt(np.sum(g**2)))},
                }
            scores.append(
                {
                    "data_id": row["data_id"],
                    "source_id": row["source_id"],
                    "repo": row["repo"],
                    "fresh_adam_cosine": aggregate_score(
                        adam, h, cfg["score"]["zero_epsilon"]
                    ),
                    "raw_grad_cosine": aggregate_score(
                        raw, h, cfg["score"]["zero_epsilon"]
                    ),
                    "supervised_tokens": row["supervised_tokens"],
                    "input_tokens": row["input_tokens"],
                    "tool_tokens": row["tool_tokens"],
                    "max_context": row["max_context"],
                    "grad_norm": diagnostic["grad_norm"],
                    "cache_identity": cache_identity,
                    "mode": cfg["mode"],
                }
            )
            run.event(
                "score",
                "candidate_gradient",
                data_id=row["data_id"],
                **diagnostic,
                gradient_replay_tokens=row["input_tokens"],
                wall_seconds=time.monotonic() - started,
            )
        ordered = sorted(scores, key=lambda x: (x["fresh_adam_cosine"], x["data_id"]))
        from scipy.stats import rankdata

        ranks = rankdata([r["fresh_adam_cosine"] for r in ordered], method="average")
        for item, rank in zip(ordered, ranks):
            item["percentile"] = float((rank - 1) / (len(scores) - 1))
        for filename, value in [
            ("candidate_scores", ordered),
            ("validation_features", h.tolist()),
            (
                "diagnostics",
                {
                    "zero_gradient_tasks": 30 - coverage,
                    "reward_coverage": coverage / 30,
                    "warning": (
                        "LOW_REWARD_COVERAGE"
                        if coverage / 30 < cfg["score"]["low_reward_coverage"]
                        else None
                    ),
                    "cache_identity": cache_identity,
                    "projection_algorithm": "countsketch-v1",
                    "projection_stability": (
                        "requires server smoke exact/projected comparison"
                        if runtime
                        else "CPU regression tests"
                    ),
                    "validation_gradient_replay_tokens": validation_cost,
                    "parameter_order": (
                        [n for n, _ in runtime.parameters] if runtime else ["synthetic"]
                    ),
                    "adapter_identity": runtime.adapter_identity if runtime else "mock",
                },
            ),
        ]:
            path = run.root / "scores" / f"{filename}.json"
            write(path, value)
            out.append(path)
        if runtime:
            runtime.close()
