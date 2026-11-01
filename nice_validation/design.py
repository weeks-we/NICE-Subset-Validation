"""Counts derived from the independently frozen experiment configuration."""

import math


def sampling_groups(cfg: dict) -> list[str]:
    """Ordered SFT groups; omitted configuration preserves the legacy design."""
    return list(cfg["experiment"].get("groups", ["low", "middle", "high", "random"]))


def counts(cfg: dict) -> dict:
    if cfg["experiment"].get("design") == "verified-selection-v1":
        from nice_validation.experiment_design import counts as new_counts
        return new_counts(cfg)
    exp = cfg["experiment"]
    branches = len(sampling_groups(cfg)) * len(exp["seeds"])
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


def strata_ranges(candidates: int, scheme: str = "thirds") -> dict:
    if scheme == "halves":
        return {"low": (0, candidates // 2), "high": (candidates // 2, candidates)}
    if scheme != "thirds":
        raise ValueError("Unknown score stratum scheme")
    first, second = candidates // 3, 2 * candidates // 3
    return {"low": (0, first), "middle": (first, second), "high": (second, candidates)}
