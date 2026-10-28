"""Score strata and token-budget-conditional, uniform subset sampling."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from nice_validation.core import Run, read, write


def statistics(values: list[float]) -> dict:
    a = np.asarray(values)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "quantiles": np.quantile(a, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "histogram": {
            "counts": np.histogram(a, bins=10)[0].tolist(),
            "edges": np.histogram(a, bins=10)[1].tolist(),
        },
    }


def make_subsets(run: Run) -> None:
    if not run.valid("score"):
        raise ValueError("Complete score first")
    with run.stage("make-subsets") as out:
        if out is None:
            return
        cfg = run.cfg["experiment"]
        rows = sorted(
            read(run.root / "scores" / "candidate_scores.json"),
            key=lambda r: (r["fresh_adam_cosine"], r["data_id"]),
        )
        if len(rows) != 300 or len({r["data_id"] for r in rows}) != 300:
            raise ValueError("Expected 300 unique scored candidates")
        if rows[0]["fresh_adam_cosine"] == rows[-1]["fresh_adam_cosine"]:
            raise ValueError(
                "All scores tied: effective score stratification is unavailable"
            )
        rng = random.Random(2026)
        probes = [
            sum(r["supervised_tokens"] for r in rng.sample(rows, 30))
            for _ in range(cfg["budget_probes"])
        ]
        budget = float(np.median(probes))
        selected = []
        member_sets = []
        groups = {
            "low": rows[:100],
            "middle": rows[100:200],
            "high": rows[200:],
            "random": rows,
        }
        for group, source in groups.items():
            reachable = sorted(r["supervised_tokens"] for r in source)
            for seed in cfg["seeds"]:
                rng = random.Random(f"subset-v1:{group}:{seed}")
                for attempt in range(cfg["sample_attempts"]):
                    candidate = rng.sample(source, 30)
                    total = sum(r["supervised_tokens"] for r in candidate)
                    members = sorted(r["data_id"] for r in candidate)
                    if (
                        abs(total - budget) <= budget * cfg["budget_tolerance"]
                        and members not in member_sets
                    ):
                        break
                else:
                    path = run.root / "manifests" / "subset_failure.json"
                    write(
                        path,
                        {
                            "group": group,
                            "seed": seed,
                            "target": budget,
                            "reachable_min": sum(reachable[:30]),
                            "reachable_max": sum(reachable[-30:]),
                            "attempt_limit": cfg["sample_attempts"],
                        },
                    )
                    raise ValueError(
                        f"Token budget unreachable within attempt limit; see {path}"
                    )
                member_sets.append(members)
                selected.append(
                    {
                        "subset_id": f"{group}_{seed}",
                        "group": group,
                        "sampling_seed": seed,
                        "members": members,
                        "member_count": 30,
                        "supervised_tokens": total,
                        "budget_target": budget,
                        "sampling_attempts": attempt + 1,
                        "fresh_adam_cosine": statistics(
                            [r["fresh_adam_cosine"] for r in candidate]
                        ),
                        "raw_grad_cosine": statistics(
                            [r["raw_grad_cosine"] for r in candidate]
                        ),
                        "repos": dict(Counter(r["repo"] for r in candidate)),
                        "mode": run.cfg["mode"],
                    }
                )
        intersections = [
            [len(set(a) & set(b)) for b in member_sets] for a in member_sets
        ]
        jaccard = [
            [len(set(a) & set(b)) / len(set(a) | set(b)) for b in member_sets]
            for a in member_sets
        ]
        boundaries = [
            {
                "rank": i,
                "left_score": rows[i - 1]["fresh_adam_cosine"],
                "right_score": rows[i]["fresh_adam_cosine"],
                "gap": rows[i]["fresh_adam_cosine"] - rows[i - 1]["fresh_adam_cosine"],
                "tie_count": sum(
                    r["fresh_adam_cosine"] == rows[i]["fresh_adam_cosine"] for r in rows
                ),
            }
            for i in [100, 200]
        ]
        for name, value in [
            ("subsets", selected),
            ("strata", {g: [r["data_id"] for r in s] for g, s in groups.items()}),
            (
                "overlap",
                {
                    "subset_ids": [s["subset_id"] for s in selected],
                    "intersection": intersections,
                    "jaccard": jaccard,
                },
            ),
            (
                "budget",
                {
                    "target": budget,
                    "probe_seed": 2026,
                    "probe_count": len(probes),
                    "probe_totals": probes,
                    "tolerance": cfg["budget_tolerance"],
                    "boundaries": boundaries,
                },
            ),
        ]:
            path = run.root / "manifests" / f"{name}.json"
            write(path, value)
            out.append(path)
