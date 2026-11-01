"""Predeclared Verified NICE / SWE-smith selection / held-out Verified design."""
import math

DESIGN = "verified-selection-v1"


def validate(cfg):
    e, t, a = cfg["experiment"], cfg["training"], cfg["agent"]
    for name in ["candidates", "validation", "sft_validation", "benchmark", "rollouts", "subset_size"]:
        if type(e.get(name)) is not int or e[name] <= 0:
            raise ValueError(f"experiment.{name} must be a positive integer")
    if e["groups"] != ["low", "high"] or e["seeds"] != [11] or e.get("strata") != "halves":
        raise ValueError("This design requires low/high halves and one sampling seed=11")
    if e["rollouts"] != 8 or e["subset_size"] > e["candidates"] // 2:
        raise ValueError("Eight NICE rollouts and a feasible half-pool subset are required")
    if t.get("learning_rates") != [1e-6, 5e-6] or t.get("checkpoints") != [.25, .5, 1.0]:
        raise ValueError("Freeze the two learning rates and quarter/half/full checkpoints")
    batch = t["effective_batch"]
    if type(batch) is not int or batch <= 0 or e["subset_size"] % (4 * batch):
        raise ValueError("Quarter/half/full exposures must be exact optimizer boundaries")
    if a.get("history_observations") != 5:
        raise ValueError("Freeze the five-most-recent-tool-observations history policy")
    if type(a.get("tasks_per_gpu", 1)) is not int or not 1 <= a.get("tasks_per_gpu", 1) <= 2:
        raise ValueError("Use one or two concurrent task slots per GPU")
    if cfg["data"].get("validation_kind") != "swebench" or cfg["data"].get("validation_split") != "test":
        raise ValueError("NICE and final benchmark must use the Verified test source")
    if cfg.get("transfer", {}).get("enabled"):
        raise ValueError("The old four-suite transfer experiment must remain disabled")


def counts(cfg):
    e = cfg["experiment"]
    jobs = len(e["groups"]) * len(cfg["training"]["learning_rates"])
    points = jobs * len(cfg["training"]["checkpoints"])
    nice = e["validation"] * e["rollouts"]
    selection = e["sft_validation"] * (points + 1)
    benchmark = e["benchmark"] * (jobs + 1)
    retest = e["validation"] * (jobs + 1)
    return dict(candidates=e["candidates"], validation=e["validation"],
                sft_validation=e["sft_validation"], benchmark=e["benchmark"],
                subset_size=e["subset_size"], subset_groups=2, branches=jobs,
                checkpoints=points, selected_models=jobs, nice_episodes=nice,
                selection_episodes=selection, benchmark_episodes=benchmark, nice_retest_episodes=retest,
                evaluation_episodes=selection + benchmark + retest, main_episodes=nice + selection + benchmark + retest,
                transfer_episodes=0, all_episodes=nice + selection + benchmark + retest,
                training_updates=jobs * e["subset_size"] // cfg["training"]["effective_batch"],
                training_exposures=jobs * e["subset_size"],
                selection_fraction=e["subset_size"] / e["candidates"])


def jobs(subsets, cfg):
    result = []
    for subset in subsets:
        for index, lr in enumerate(cfg["training"]["learning_rates"]):
            result.append({**subset, "subset_id": f"{subset['group']}_lr{index + 1}",
                           "selection_subset_id": subset["subset_id"], "learning_rate": lr})
    return result


def points(job, cfg):
    size, batch = cfg["experiment"]["subset_size"], cfg["training"]["effective_batch"]
    return [{"model_id": job["subset_id"] + f"_p{int(f * 100):03d}",
             "job_id": job["subset_id"], "group": job["group"],
             "learning_rate": job["learning_rate"], "fraction": f,
             "exposures": int(size * f), "step": int(size * f) // batch}
            for f in cfg["training"]["checkpoints"]]
