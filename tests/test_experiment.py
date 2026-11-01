"""New A design: source reuse, split isolation, checkpoint selection and delivery."""
import copy
from pathlib import Path
from unittest.mock import patch
import pytest

from nice_validation.core import Run, config, read, write, sha, read_jsonl
from nice_validation.experiment_design import counts, points
from nice_validation.experiment_data import prepare, mock_sources
from nice_validation.experiment import pipeline, evaluate, select_models, EndpointPool
from nice_validation.experiment_training import ensure_jobs, validate_point
from nice_validation.experiment_online import configure, acquire
from nice_validation.harness import episodes
from nice_validation.scoring import score
from nice_validation.subsets import make_subsets
from nice_validation.batch_tuning import tune


def small_cfg(path):
    cfg = config("configs/mock.yaml")
    cfg["output"] = str(path)
    cfg["experiment"].update(candidates=64, subset_size=16, validation=4, sft_validation=6, benchmark=7)
    cfg["training"].update(effective_batch=4, micro_batch_size=2)
    return cfg


def ready(path):
    run = Run(small_cfg(path))
    prepare(run)
    episodes(run, "initial", True)
    score(run)
    make_subsets(run)
    ensure_jobs(run)
    tune(run)
    return run


def test_default_design_and_all_counts():
    cfg = config("configs/main.yaml")
    c = counts(cfg)
    assert c["candidates"] == 2000 and c["subset_size"] == 512
    assert c["branches"] == c["selected_models"] == 4 and c["checkpoints"] == 12
    assert c["nice_episodes"] == 160 and c["selection_episodes"] == 1300
    assert c["benchmark_episodes"] == 500 and c["nice_retest_episodes"] == 100 and c["all_episodes"] == 2060
    assert c["training_updates"] == 256 and c["training_exposures"] == 2048
    assert cfg["training"]["effective_batch"] == 8
    assert cfg["training"]["learning_rates"] == [1e-6, 5e-6]
    assert cfg["agent"]["max_actions"] == 75
    assert (cfg["agent"]["max_new_tokens"], cfg["agent"]["total_new_tokens"], cfg["agent"]["context_length"]) == (32768, 262144, 131072)
    assert cfg["agent"]["wall_seconds"] == 3600 and cfg["agent"]["tool_seconds"] == 300
    assert not cfg["transfer"]["enabled"]
    assert not Path("experiments/scaled_nice").exists()
    assert not Path("docs/HANDOFF_SCALED.md").exists()


@pytest.mark.parametrize("tokens", [8192, 32767, 32768])
def test_full_context_smoke_requires_the_enlarged_output_span(tmp_path, tokens):
    from types import SimpleNamespace
    from nice_validation.smoke import require_real_smoke
    cfg = config("configs/main.yaml")
    write(tmp_path / "smoke/result.json", {
        "mode": "real", "target_model_tested": True,
        "capacity_probe": {"update_completed": True, "input_tokens": cfg["data"]["max_length"]},
        "replay_capacity_probe": {"backward_completed": True, "total_tokens": 131072, "output_tokens": tokens}})
    view = SimpleNamespace(root=tmp_path, cfg=cfg, valid=lambda stage: True)
    if tokens < 32768:
        with pytest.raises(ValueError, match="full agent-context"):
            require_real_smoke(view)
    else:
        require_real_smoke(view)


@pytest.mark.parametrize("section,key,value", [
    ("experiment", "candidates", 0), ("experiment", "sft_validation", True),
    ("experiment", "subset_size", 513), ("experiment", "rollouts", 4),
    ("experiment", "seeds", [11, 22]), ("experiment", "groups", ["low", "random"]),
    ("experiment", "strata", "thirds"), ("training", "learning_rates", [1e-5]),
    ("training", "checkpoints", [.5, 1]), ("training", "effective_batch", 3),
    ("agent", "history_observations", 0), ("agent", "tasks_per_gpu", True),
    ("data", "validation_kind", "swesmith"), ("transfer", "enabled", True),
])
def test_invalid_new_design_fails_early(tmp_path, section, key, value):
    cfg = config("configs/main.yaml")
    cfg[section][key] = value
    path = tmp_path / "bad.json"
    write(path, cfg)
    with pytest.raises(ValueError):
        config(path)


def test_task_splits_join_unused_trajectory_ids_and_eight_unique_seed_streams(tmp_path):
    run = Run(small_cfg(tmp_path))
    prepare(run)
    ids = read(run.root / "manifests/split_ids.json")
    all_ids = [i for group in ids.values() for i in group]
    assert len(all_ids) == len(set(all_ids))
    training, verified, smith = mock_sources(run.cfg)
    assert set(ids["sft_validation"]) <= {r["instance_id"] for r in training} & {r["instance_id"] for r in smith}
    assert not set(ids["candidates"]) & set(ids["sft_validation"])
    assert set(ids["nice"] + ids["benchmark"]) <= {r["instance_id"] for r in verified}
    episodes(run, "initial", True)
    results = read(run.root / "rollouts/nice/results.json")
    assert len(results) == 32 and len({r["seed"] for r in results}) == 32
    assert all(sorted(r["repeat"] for r in results if r["task_id"] == task) == list(range(8)) for task in ids["nice"])


@pytest.mark.parametrize("family", ["raw_only", "transitive"])
def test_holdout_join_preserves_one_sided_and_transitive_raw_families(tmp_path, monkeypatch, family):
    baseline = Run(small_cfg(tmp_path / "baseline"))
    prepare(baseline)
    split = read(baseline.root / "manifests/split_ids.json")
    candidate_id, unused_id = split["candidates"][0], split["sft_validation"][0]
    cfg = small_cfg(tmp_path / "joined")
    training, verified, smith = mock_sources(cfg)
    candidate = next(r for r in training if r["instance_id"] == candidate_id)
    unused = next(r for r in training if r["instance_id"] == unused_id)
    if family == "raw_only":
        unused["source_instance_id"] = candidate_id
    else:
        candidate["source_instance_id"] = "family__root"
        unused["source_instance_id"] = "family__child"
        smith.append({**smith[0], "instance_id": "family__child", "source_instance_id": "family__root",
                      "problem_statement": "A separate family metadata bridge."})
    monkeypatch.setattr("nice_validation.experiment_data.mock_sources", lambda c: copy.deepcopy((training, verified, smith)))
    run = Run(cfg)
    prepare(run)
    current = read(run.root / "manifests/split_ids.json")
    assert candidate_id in current["candidates"]
    assert unused_id not in current["sft_validation"] + current["development"]
    from nice_validation.core import digest
    saved = next(r for r in read(run.root / "manifests/candidates.json") if r["instance_id"] == candidate_id)
    assert saved["source_sha256"] == digest(candidate)
    assert "source_family_id" not in candidate


@pytest.mark.parametrize("problem", ["empty_training", "empty_verified", "empty_smith", "duplicate_verified", "missing_matching_ids"])
def test_missing_or_duplicate_raw_inputs_stop_before_model_contact(tmp_path, monkeypatch, problem):
    cfg = small_cfg(tmp_path)
    training, verified, smith = mock_sources(cfg)
    if problem == "empty_training": training = []
    elif problem == "empty_verified": verified = []
    elif problem == "empty_smith": smith = []
    elif problem == "duplicate_verified": verified.append(verified[0])
    else:
        smith = [{**r, "instance_id": "other-" + r["instance_id"]} for r in smith]
    monkeypatch.setattr("nice_validation.experiment_data.mock_sources", lambda c: (training, verified, smith))
    with pytest.raises(ValueError):
        prepare(Run(cfg))


def test_full_small_flow_preserves_matched_lr_members_and_resumes(tmp_path, monkeypatch):
    run = Run(small_cfg(tmp_path))
    pipeline(run)
    summary = read(run.root / "summary.json")
    assert summary["main_episode_count"] == 165
    assert summary["nice_retest"]["independent_test"] is False
    assert summary["nice_retest"]["used_for_checkpoint_selection"] is False
    assert len(summary["nice_retest"]["summary"]) == 5
    assert "nice_retest" in summary["logical_token_usage"]
    assert summary["successful_updates"] == 16 and summary["training_exposures"] == 64
    assert not summary["formal_evidence"]
    selected = read(run.root / "selection.json")["selected"]
    assert len(selected) == 4 and {(p["group"], p["learning_rate"]) for p in selected} == {
        (g, lr) for g in ["low", "high"] for lr in [1e-6, 5e-6]}
    entries = read(run.root / "manifests/training_jobs.json")
    for group in ["low", "high"]:
        pair = [j for j in entries if j["group"] == group]
        assert pair[0]["members"] == pair[1]["members"]
        for a, b in zip(points(pair[0], run.cfg), points(pair[1], run.cfg)):
            left = read(run.root / "training" / a["model_id"] / "adapter/mock.json")
            right = read(run.root / "training" / b["model_id"] / "adapter/mock.json")
            assert left["members"] == right["members"] and len(left["members"]) == a["exposures"]
    before = {str(p): sha(p) for p in (run.root / "scores/shards/items").rglob("*") if p.is_file()}
    events = read_jsonl(run.root / "logs/events.jsonl")
    pipeline(run)
    assert sum(r["event"] == "episode_attempt" for r in read_jsonl(run.root / "logs/events.jsonl")) == 165
    assert before == {str(p): sha(p) for p in (run.root / "scores/shards/items").rglob("*") if p.is_file()}
    not_selected = next(p for p in read(run.root / "manifests/checkpoints.json") if p["model_id"] not in {r["model_id"] for r in selected})
    with pytest.raises(ValueError, match="restricted"):
        evaluate(run, "benchmark", not_selected["model_id"])
    with pytest.raises(ValueError, match="restricted"):
        evaluate(run, "nice_retest", not_selected["model_id"])
    nice_ids = {t["instance_id"] for t in read(run.root / "manifests/validation.json")}
    for model in ["initial", *[p["model_id"] for p in selected]]:
        rows = read(run.root / "evaluation/nice_retest" / model / "results.json")
        assert len(rows) == 4 and {r["task_id"] for r in rows} == nice_ids
        assert all(r["model_id"] == model for r in rows)
    assert all(r["status"] == "PASS" for r in read(run.root / "figures/layout_audit.json"))
    import sys
    import zipfile
    from scripts.bundle_results import main as bundle
    target = tmp_path / "small-results.zip"
    monkeypatch.setattr(sys, "argv", ["bundle", "--run", str(run.root), "--output", str(target)])
    bundle()
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert "selection.json" in archive.namelist()
        assert "manifests/split_ids.json" in archive.namelist()
        assert "manifests/sft_validation.json" not in archive.namelist()
        assert "manifests/benchmark.json" not in archive.namelist()
        assert "tables/nice_retest_summary.csv" in archive.namelist()
        assert "tables/nice_retest_task_results.csv" in archive.namelist()
        assert "tables/nice_retest_paired_tasks.csv" in archive.namelist()
        assert "tables/nice_retest_high_low_comparison.csv" in archive.namelist()


@pytest.mark.parametrize("split", ["benchmark", "nice_retest"])
def test_benchmark_is_blocked_before_validation_selection(tmp_path, split):
    run = Run(small_cfg(tmp_path))
    prepare(run)
    with pytest.raises(ValueError, match="Freeze validation-only"):
        evaluate(run, split, "initial")


def test_changed_candidate_manifest_is_refused_before_training(tmp_path):
    from nice_validation.experiment_training import train_job
    run = ready(tmp_path)
    rows = read(run.root / "manifests/candidates.json")
    rows[0]["labels"][-1] += 1
    write(run.root / "manifests/candidates.json", rows)
    job = read(run.root / "manifests/training_jobs.json")[0]
    with pytest.raises(ValueError, match="frozen"):
        train_job(run, job["subset_id"])
    assert not (run.root / "training" / job["subset_id"]).exists()


def test_milestone_publication_retries_transient_locks_without_partial_copy(tmp_path, monkeypatch):
    from nice_validation.experiment_training import publish_point
    run = Run(small_cfg(tmp_path))
    point = {"model_id": "low_lr1_p025", "job_id": "low_lr1", "group": "low",
             "learning_rate": 1e-6, "fraction": .25, "exposures": 4, "step": 1}
    rename = Path.rename
    attempts = []
    def transient(source, target):
        if source.name.startswith(".milestone-"):
            attempts.append(True)
            if len(attempts) < 3:
                assert not target.exists()
                raise PermissionError("simulated temporary reader handle")
        return rename(source, target)
    monkeypatch.setattr(Path, "rename", transient)
    publish_point(run, point, members=["a", "b", "c", "d"])
    assert len(attempts) == 3
    assert validate_point(run, point)["model_id"] == point["model_id"]


def test_selection_is_within_each_group_and_lr_and_ties_prefer_earlier(tmp_path):
    run = ready(tmp_path)
    ids = [r["instance_id"] for r in read(run.root / "manifests/sft_validation.json")]
    entries = read(run.root / "manifests/training_jobs.json")
    for ji, job in enumerate(entries):
        with run.stage("train:" + job["subset_id"]) as out:
            p = run.root / "training" / job["subset_id"] / "unit.json"
            write(p, {"unit_test": True})
            out.append(p)
        for pi, point in enumerate(points(job, run.cfg)):
            correct = [2, 2, 2][pi] if ji == 0 else [1, 4, 3][pi]
            with run.stage("selection:" + point["model_id"]) as out:
                p = run.root / "evaluation/selection" / point["model_id"] / "results.json"
                write(p, [{"task_id": task, "model_id": point["model_id"], "run_identity": run.identity,
                           "status": "RESOLVED" if index < correct else "TASK_FAILED", "resolved": index < correct}
                          for index, task in enumerate(ids)])
                out.append(p)
    write(run.root / "evaluation/benchmark/fake/results.json", [{"would_change_choice": True}])
    select_models(run)
    chosen = read(run.root / "selection.json")
    assert [p["fraction"] for p in chosen["selected"]] == [.25, .5, .5, .5]
    assert chosen["benchmark_used_for_selection"] is False


def old_storage(root):
    old = root / "old"
    model = old / "models/frozen-base"
    model.mkdir(parents=True)
    cfg = config("configs/legacy.yaml")
    architecture = {"model_type": "qwen3_5", "text_config": {"max_position_embeddings": 262144}}
    write(model / "config.json", architecture)
    write(model / "model_lock.json", {"id": cfg["model"]["id"], "revision": "c"*40,
          "architecture": architecture, "files": {"config.json": {"sha256": sha(model / "config.json"), "size": (model / "config.json").stat().st_size}}})
    records = {"training": {"id": cfg["data"]["training_id"], "revision": "a"*40},
               "validation": {"id": "SWE-bench/SWE-smith", "revision": "b"*40},
               "verified": {"id": "SWE-bench/SWE-bench_Verified", "revision": "e"*40},
               "model": {"id": cfg["model"]["id"], "revision": "c"*40},
               "swesmith": {"id": "SWE-bench/SWE-smith", "revision": "d"*40},
               "swebench": {"id": "SWE-bench/SWE-bench", "revision": "f"*40}}
    write(old / "data/source_revisions.json", records)
    raw = {}
    for kind, filename, split in [("training", "training", "tool"), ("validation", "validation", "train"), ("verified", "verified", "test")]:
        target = old / "data" / (filename + ".jsonl")
        target.write_text('{"fixture":true}\n', encoding="utf-8")
        raw[kind] = {**records[kind], "path": str(target.resolve()), "sha256": sha(target),
                     "split": split, "limit": None, "records": 1, "files": []}
        write(target.with_suffix(".jsonl.source.json"), raw[kind])
    write(old / "data/source_lock.json", {k: raw[k] for k in ["training", "validation"]})
    write(old / "data/transfer_sources.json", {"verified": raw["verified"]})
    write(old / "runs/main/config/snapshot.json", {"config": {"model": {"path": str(model.resolve())}, "data": {}}})
    return old, model


def test_reuse_old_model_and_all_three_sources_without_network_or_copy(tmp_path):
    old, model = old_storage(tmp_path)
    before = {str(p): sha(p) for p in old.rglob("*") if p.is_file()}
    gpus = config("configs/mock.yaml")["resources"]["gpu_uuids"]
    with patch("nice_validation.online.resolve_sources", side_effect=AssertionError("network")):
        paths = configure(tmp_path / "new", gpus, tmp_path / "cfg", reuse_from=old)
        cfg = config(paths["server_config"])
        acquire(cfg)
    assert cfg["model"]["path"] == str(model.resolve())
    assert cfg["data"]["smith_revision"] == "b" * 40
    assert cfg["agent"]["swesmith_revision"] == "d" * 40
    assert cfg["data"]["validation_revision"] == "e" * 40
    assert cfg["agent"]["workers"] == 16 and len(paths["service_configs"]) == 8
    assert all(Path(cfg["data"][k + "_path"]).is_relative_to(old.resolve()) for k in ["training", "validation", "smith"])
    assert not (tmp_path / "new/models/base").exists()
    assert before == {str(p): sha(p) for p in old.rglob("*") if p.is_file()}


@pytest.mark.parametrize("broken", ["model", "raw", "missing_verified_pin", "unknown_context", "nested_storage"])
def test_reuse_refuses_corrupt_or_ambiguous_inputs(tmp_path, broken):
    old, model = old_storage(tmp_path)
    storage = tmp_path / "new"
    if broken == "model": (model / "config.json").write_text("changed")
    elif broken == "raw": (old / "data/training.jsonl").write_text("changed")
    elif broken == "missing_verified_pin":
        records = read(old / "data/source_revisions.json")
        records.pop("verified")
        write(old / "data/source_revisions.json", records)
    elif broken == "unknown_context":
        lock = read(model / "model_lock.json")
        lock["architecture"] = {}
        write(model / "model_lock.json", lock)
    else: storage = old / "nested"
    with pytest.raises(ValueError):
        configure(storage, config("configs/mock.yaml")["resources"]["gpu_uuids"], tmp_path / "cfg", reuse_from=old)
