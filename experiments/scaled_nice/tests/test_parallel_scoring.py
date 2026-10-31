"""Numerical equivalence, shard coverage, cache recovery and worker ownership."""

import copy
import platform
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from nice_validation.core import Run, read, read_jsonl, sha, write
from nice_validation.data import prepare
from nice_validation.harness import episodes
from nice_validation.scoring import score
from nice_validation.score_features import fingerprint
from nice_validation.parallel_scoring import (
    inputs,
    make_plan,
    split_ids,
    item_path,
    save_arrays,
    check_references,
    lease,
)
from test_scaled_experiment import small_scaled


def ready_run(path, parallel=True, invalid=None, zero=False):
    cfg = small_scaled(path)
    cfg["experiment"].update(candidates=30, validation=5, subset_size=3)
    cfg["score"]["parallel"] = parallel
    run = Run(cfg)
    prepare(run)
    if zero or invalid:
        records = [
            {
                "task_id": t["instance_id"],
                "repeat": r,
                "resolved": False,
                "status": "TASK_FAILED",
                "run_identity": run.identity,
            }
            for t in read(run.root / "manifests/validation.json")
            for r in range(4)
        ]
        if invalid == "duplicate":
            records[0]["repeat"] = 1
        elif invalid == "identity":
            records[0]["run_identity"] = "foreign"
        elif invalid == "infra":
            records[0]["status"] = "INFRA_ERROR"
        with run.stage("rollout-nice") as out:
            dest = run.root / "rollouts/nice/results.json"
            write(dest, records)
            out.append(dest)
    else:
        episodes(run, "initial", nice=True)
    return run


def test_weighted_shards_cover_every_id_once_even_with_empty_workers():
    rows = [{"id": str(i)} for i in range(5)]
    shards = split_ids(rows, "id", {str(i): i + 1 for i in range(5)}, 8)
    assert sorted(x for s in shards for x in s) == [str(i) for i in range(5)]
    assert len(shards) == 8 and sum(not s for s in shards) == 3
    assert shards == split_ids(
        list(reversed(rows)), "id", {str(i): i + 1 for i in range(5)}, 8
    )


def test_serial_and_four_process_scores_are_identical_and_completed_resume_skips(
    tmp_path,
):
    serial = ready_run(tmp_path / "serial", parallel=False)
    parallel = ready_run(tmp_path / "parallel")
    score(serial)
    score(parallel)
    left, right = read(serial.root / "scores/candidate_scores.json"), read(
        parallel.root / "scores/candidate_scores.json"
    )
    assert [{k: v for k, v in r.items() if k != "cache_identity"} for r in left] == [
        {k: v for k, v in r.items() if k != "cache_identity"} for r in right
    ]
    assert read(serial.root / "scores/validation_features.json") == read(
        parallel.root / "scores/validation_features.json"
    )
    summary = read(parallel.root / "scores/parallel_summary.json")
    assert (
        sum(summary["candidate_counts"]) == 30
        and sum(summary["validation_counts"]) == 5
    )
    assert (
        len(summary["gpu_uuids"]) == 4 and summary["reference_max_abs_difference"] == 0
    )
    assert summary["optimizer_updates"] == 0 and parallel.valid("score")
    state = parallel.state()
    with patch(
        "nice_validation.parallel_scoring.subprocess.Popen",
        side_effect=AssertionError("reran scoring"),
    ):
        score(parallel)
    assert parallel.state() == state


def unfinished(run):
    with patch(
        "nice_validation.parallel_scoring.publish",
        side_effect=RuntimeError("publication interrupted"),
    ):
        with pytest.raises(RuntimeError, match="publication interrupted"):
            score(run)
    assert run.state()["score"]["status"] == "failed"


def test_partial_score_resume_reuses_every_item_without_duplicate_gradients(tmp_path):
    run = ready_run(tmp_path)
    unfinished(run)
    files = list((run.root / "scores/shards/items").rglob("*.npz"))
    previous = {str(p): sha(p) for p in files}
    assert len(previous) == 35
    score(run)
    assert run.valid("score") and previous == {str(p): sha(p) for p in files}
    events = read_jsonl(run.root / "logs/events.jsonl")
    assert (
        sum(
            e["event"] == "candidate_gradient" and e["stage"] == "score" for e in events
        )
        == 30
    )
    assert (
        sum(
            e["event"] == "validation_gradient" and e["stage"] == "score"
            for e in events
        )
        == 5
    )
    assert sum(e["event"] == "gradient_cache_hit" for e in events) == 35
    imported = [
        e["scoring_worker_event_id"] for e in events if "scoring_worker_event_id" in e
    ]
    assert len(imported) == len(set(imported))


def test_corrupt_partial_item_blocks_recovery_and_is_preserved(tmp_path):
    run = ready_run(tmp_path)
    unfinished(run)
    row = read(run.root / "manifests/candidates.json")[0]
    path = item_path(run, "candidate", row["data_id"])
    meta = read(path)
    meta["arrays_sha256"] = "corrupt"
    write(path, meta)
    with pytest.raises(RuntimeError, match="scoring worker"):
        score(run)
    assert read(path)["arrays_sha256"] == "corrupt" and not run.valid("score")


def test_zero_reward_barrier_stops_before_candidate_pool(tmp_path):
    run = ready_run(tmp_path, zero=True)
    with pytest.raises(ValueError, match="All validation reward directions are zero"):
        score(run)
    assert read(run.root / "scores/failure.json")["tasks"] == 5
    assert not list((run.root / "scores/shards/items/candidate").rglob("cache.json"))
    assert not list((run.root / "scores/shards/leases").glob("shard-*.json"))
    assert not (run.root / ".run.lock").exists()


@pytest.mark.parametrize("invalid", ["duplicate", "identity", "infra"])
def test_invalid_rollouts_fail_before_launching_workers(tmp_path, invalid):
    run = ready_run(tmp_path, invalid=invalid)
    with patch(
        "nice_validation.parallel_scoring.subprocess.Popen",
        side_effect=AssertionError("launched on bad input"),
    ):
        with pytest.raises(ValueError):
            score(run)


@pytest.mark.parametrize("change", ["order", "values", "checksum"])
def test_cross_gpu_reference_detects_parameter_and_numerical_mismatches(
    tmp_path, change
):
    run = ready_run(tmp_path)
    c, t, e, grouped = inputs(run)
    plan = make_plan(run, c, t, e, grouped)
    ready = []
    for shard in plan["shards"]:
        path = run.root / "scores" / f"ref-{shard['index']}.npz"
        arrays = {n: np.linspace(0, 1, 128) for n in ["raw", "adam", "validation"]}
        save_arrays(path, arrays)
        ready.append(
            {
                "run_identity": run.identity,
                "input_key": plan["input_key"],
                "shard": shard,
                "fingerprint": fingerprint(None),
                "reference_file": str(path.relative_to(run.root)),
                "reference_sha256": sha(path),
            }
        )
    assert check_references(run, plan, ready)[1] == 0
    if change == "order":
        ready[1]["fingerprint"]["parameters"][0]["name"] = "other-order"
    elif change == "checksum":
        ready[1]["reference_sha256"] = "bad"
    else:
        path = run.root / ready[1]["reference_file"]
        save_arrays(path, {n: np.ones(128) for n in ["raw", "adam", "validation"]})
        ready[1]["reference_sha256"] = sha(path)
    with pytest.raises(ValueError):
        check_references(run, plan, ready)


def test_active_lease_is_never_stolen_and_dead_owner_is_archived(tmp_path):
    run = ready_run(tmp_path)
    with lease(run, 0):
        with pytest.raises(ValueError, match="still be active"):
            with lease(run, 0):
                pass
    path = run.root / "scores/shards/leases/shard-0.json"
    write(path, {"pid": -1, "host": platform.node()})
    with lease(run, 0):
        assert path.exists()
    assert not path.exists() and list(path.parent.glob("*.stale-*"))


def test_worker_failure_reaps_other_workers_without_publishing_score(
    tmp_path, monkeypatch
):
    run = ready_run(tmp_path)
    workers = []

    class Process:
        def __init__(self, argv, **kwargs):
            self.index = int(argv[argv.index("--shard-index") + 1])
            self.returncode = 1 if self.index == 0 else None
            assert (
                kwargs["env"]["CUDA_VISIBLE_DEVICES"]
                == run.cfg["resources"]["gpu_uuids"][self.index]
            )
            workers.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr("nice_validation.parallel_scoring.subprocess.Popen", Process)
    with pytest.raises(RuntimeError, match="scoring worker 0 failed"):
        score(run)
    assert len(workers) == 4 and all(p.returncode is not None for p in workers)
    assert not run.valid("score") and not (run.root / ".run.lock").exists()
