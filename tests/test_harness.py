import pytest

from nice_validation.core import Run, config, read, read_jsonl, write
from nice_validation.harness import InfraError, _episode_job, episodes


def test_infrastructure_retry_uses_fresh_attempt_and_keeps_cost(monkeypatch, tmp_path):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path)
    cfg["mode"] = "real"
    run = Run(cfg)
    calls = []

    def episode(run, task, model_id, nice, seed, folder):
        calls.append(folder)
        if len(calls) < 3:
            raise InfraError("container_start:failed")
        write(folder / "trace.json", [])
        return {
            "resolved": True,
            "status": "RESOLVED",
            "wall_seconds": 0.2,
            "input_tokens": 3,
            "output_tokens": 2,
            "tool_calls": 1,
        }

    monkeypatch.setattr("nice_validation.harness.real_episode", episode)
    result = _episode_job(
        run,
        "test",
        "initial",
        False,
        0,
        {"instance_id": "task"},
        0,
        run.root / "evaluation/initial",
    )
    assert len(calls) == 3 and len(set(calls)) == 3
    assert result["status"] == "RESOLVED"
    events = read_jsonl(run.root / "logs/events.jsonl")
    assert len(events) == 3 and sum(e["status"] == "INFRA_ERROR" for e in events) == 2
    assert all((p / "result.json").exists() for p in calls)


def test_model_failure_is_not_selectively_retried(monkeypatch, tmp_path):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path)
    cfg["mode"] = "real"
    run = Run(cfg)
    calls = []

    def episode(*args):
        calls.append(1)
        return {
            "resolved": False,
            "status": "TASK_FAILED",
            "wall_seconds": 0.1,
            "input_tokens": 2,
            "output_tokens": 1,
            "tool_calls": 0,
        }

    monkeypatch.setattr("nice_validation.harness.real_episode", episode)
    _episode_job(
        run,
        "test",
        "initial",
        False,
        0,
        {"instance_id": "task"},
        0,
        run.root / "evaluation/initial",
    )
    assert len(calls) == 1


def test_retry_cap_persists_across_resume(monkeypatch, tmp_path):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path)
    cfg["mode"] = "real"
    run = Run(cfg)
    calls = []

    def episode(*args):
        calls.append(1)
        raise InfraError("daemon_down:same")

    monkeypatch.setattr("nice_validation.harness.real_episode", episode)
    for _ in range(2):
        result = _episode_job(
            run,
            "test",
            "initial",
            False,
            0,
            {"instance_id": "task"},
            0,
            run.root / "evaluation/initial",
        )
        assert result["status"] == "INFRA_ERROR"
    assert len(calls) == 3


def test_episode_cache_validates_evidence(monkeypatch, tmp_path):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path)
    run = Run(cfg)
    folder = run.root / "evaluation/initial"
    result = _episode_job(
        run, "test", "initial", False, 0, {"instance_id": "task"}, 0, folder
    )
    (run.root / result["evidence_path"] / "trace.json").write_text('["tampered"]')
    with pytest.raises(ValueError, match="corrupted"):
        _episode_job(
            run, "test", "initial", False, 0, {"instance_id": "task"}, 0, folder
        )


def test_parallel_worker_shards_preserve_all_task_ids(tmp_path):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path)
    cfg["agent"]["workers"] = 2
    cfg["agent"]["service_endpoints"] = [
        "http://127.0.0.1:8765",
        "http://127.0.0.1:8766",
    ]
    run = Run(cfg)
    with run.stage("prepare-data") as out:
        path = run.root / "manifests/validation.json"
        write(path, [{"instance_id": f"task-{i}"} for i in range(5)])
        out.append(path)
    episodes(run, "initial", nice=True)
    results = read(run.root / "rollouts/nice/results.json")
    assert (
        len(results) == 20 and len({(r["task_id"], r["repeat"]) for r in results}) == 20
    )
    assert run.valid("rollout-nice")
