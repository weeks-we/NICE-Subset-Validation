"""Counts derived from the independently frozen experiment configuration."""

import math


def counts(cfg: dict) -> dict:
    exp = cfg["experiment"]
    branches = 4 * len(exp["seeds"])
    nice = exp["validation"] * exp["rollouts"]
    evaluation = exp["validation"] * (branches + 1)
    transfer = (
        4 * cfg["transfer"]["sample_size"] * (branches + 1)
        if cfg.get("transfer", {}).get("enabled")
        else 0
    )
    return {
        "candidates": exp["candidates"],
        "validation": exp["validation"],
        "subset_size": exp["subset_size"],
        "branches": branches,
        "nice_episodes": nice,
        "evaluation_episodes": evaluation,
        "main_episodes": nice + evaluation,
        "transfer_episodes": transfer,
        "all_episodes": nice + evaluation + transfer,
        "training_updates": branches
        * math.ceil(exp["subset_size"] / cfg["training"]["effective_batch"]),
        "training_exposures": branches * exp["subset_size"],
        "selection_fraction": exp["subset_size"] / exp["candidates"],
    }


def strata_ranges(candidates: int) -> dict:
    first, second = candidates // 3, 2 * candidates // 3
    return {"low": (0, first), "middle": (first, second), "high": (second, candidates)}
