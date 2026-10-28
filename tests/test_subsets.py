import pytest

from nice_validation.core import Run, config, read, write
from nice_validation.subsets import make_subsets


def scored_run(tmp_path, constant=False, budget_trap=False):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path)
    run = Run(cfg)
    rows = [
        {
            "data_id": f"{i:04d}",
            "fresh_adam_cosine": 0 if constant else i / 300,
            "raw_grad_cosine": i / 600,
            "supervised_tokens": 1 if budget_trap and i < 100 else 100,
            "repo": "r",
        }
        for i in range(300)
    ]
    with run.stage("score") as out:
        p = run.root / "scores/candidate_scores.json"
        write(p, rows)
        out.append(p)
    return run


def test_subsets_deterministic_budgeted_and_random_scores_present(tmp_path):
    a = scored_run(tmp_path / "a")
    b = scored_run(tmp_path / "b")
    make_subsets(a)
    make_subsets(b)
    sa = read(a.root / "manifests/subsets.json")
    sb = read(b.root / "manifests/subsets.json")
    assert sa == sb and len(sa) == 12
    assert all(
        len(set(s["members"])) == 30
        and abs(s["supervised_tokens"] - s["budget_target"]) <= 0.1 * s["budget_target"]
        for s in sa
    )
    assert len({tuple(s["members"]) for s in sa}) == 12
    assert sum(s["group"] == "random" and "raw_grad_cosine" in s for s in sa) == 3
    assert read(a.root / "manifests/overlap.json")["jaccard"][0][0] == 1


def test_all_ties_block_effective_stratification(tmp_path):
    run = scored_run(tmp_path, constant=True)
    with pytest.raises(ValueError, match="All scores tied"):
        make_subsets(run)


def test_unreachable_budget_never_silently_relaxes(tmp_path):
    run = scored_run(tmp_path, budget_trap=True)
    with pytest.raises(ValueError, match="Token budget unreachable"):
        make_subsets(run)
    failure = read(run.root / "manifests/subset_failure.json")
    assert failure["reachable_max"] < failure["target"] * 0.9
