import json
import subprocess
import sys


def test_complete_mock_pipeline_and_resumable_cli(tmp_path):
    output = tmp_path / "mock"
    command = [
        sys.executable,
        "-m",
        "nice_validation",
        "--config",
        "configs/development.yaml",
        "--output",
        str(output),
    ]
    first = subprocess.run(
        command + ["run", "--profile", "mock"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["main_episode_count"] == 510
    assert summary["successful_updates"] == 360
    assert summary["transfer_episode_count"] == 2600
    transfer = json.loads((output / "transfer/summary.json").read_text())
    assert (
        len(transfer["rows"]) == 52 and transfer["planned_all_agent_episodes"] == 3110
    )
    assert all(
        row["denominator"] == 50
        and row["success_rate"] is None
        and row["delta"] is None
        for row in transfer["rows"]
    )
    assert summary["formal_evidence"] is False and summary["correlations"] is None
    assert all(
        s["success_rate"] is None and s["delta"] is None for s in summary["subsets"]
    )
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == "WARN"
    assert not [c for c in audit["checks"] if c["status"] == "FAIL"]
    assert len(list((output / "figures").glob("*.pdf"))) == 7
    scores = (output / "scores/candidate_scores.json").read_bytes()
    resumed = subprocess.run(
        command + ["resume"], capture_output=True, text=True, timeout=300
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (output / "scores/candidate_scores.json").read_bytes() == scores
    assert list((output / "history").iterdir())
