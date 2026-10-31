import subprocess
import sys

import pytest

from nice_validation.analysis import analyze, audit
from nice_validation.core import Run, config, read, write
from nice_validation.scoring import score


def test_all_zero_reward_directions_block_scoring(tmp_path):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path)
    run = Run(cfg)
    tasks = [{"instance_id": f"task-{i}"} for i in range(30)]
    with run.stage("prepare-data") as out:
        for name, value in [("validation", tasks), ("candidates", [])]:
            path = run.root / "manifests" / f"{name}.json"
            write(path, value)
            out.append(path)
    with run.stage("rollout-nice") as out:
        path = run.root / "rollouts/nice/results.json"
        write(
            path,
            [
                {
                    "task_id": t["instance_id"],
                    "resolved": False,
                    "status": "TASK_FAILED",
                }
                for t in tasks
                for _ in range(4)
            ],
        )
        out.append(path)
    with pytest.raises(ValueError, match="All validation reward directions are zero"):
        score(run)
    assert (
        read(run.root / "scores/failure.json")["reason"]
        == "ALL_VALIDATION_DIRECTIONS_ZERO"
    )
    assert run.state()["score"]["status"] == "failed"


def test_formal_analysis_rejects_mock_artifacts(tmp_path):
    cfg = config("configs/development.yaml")
    cfg["mode"], cfg["output"] = "real", str(tmp_path)
    run = Run(cfg)
    write(run.root / "scores/candidate_scores.json", [{"mode": "mock"}])
    with pytest.raises(ValueError, match="cannot enter formal analysis"):
        analyze(run)


def test_incomplete_audit_is_machine_readable_not_division_by_zero(tmp_path):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path)
    report = audit(Run(cfg))
    assert report["status"] == "FAIL"
    assert any(
        c["check"] == "evaluation_denominator:initial" and c["status"] == "FAIL"
        for c in report["checks"]
    )


def test_import_and_local_fetch_model_do_not_load_weights(tmp_path):
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import nice_validation.model, sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr
    guard = subprocess.run(
        [
            sys.executable,
            "-m",
            "nice_validation",
            "--output",
            str(tmp_path),
            "fetch-model",
        ],
        capture_output=True,
        text=True,
    )
    assert guard.returncode != 0 and "execution: server" in guard.stderr
    assert not list(tmp_path.iterdir())


def test_raw_source_mutation_invalidates_cache(tmp_path):
    cfg = config("configs/development.yaml")
    cfg["mode"], cfg["output"] = "real", str(tmp_path / "run")
    data = tmp_path / "training.jsonl"
    data.write_text('{"id":1}\n')
    cfg["data"]["training_path"] = str(data)
    Run(cfg)
    data.write_text('{"id":2}\n')
    with pytest.raises(ValueError, match="identity changed"):
        Run(cfg)
