import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from nice_validation.core import Run, config, write
from nice_validation.harness import EDITOR_PROGRAM
from nice_validation.preflight import preflight
from nice_validation.service import RuntimeManager
from nice_validation.smoke import require_real_smoke
from nice_validation.cleanup import cleanup_containers


def test_service_adapter_switch_failure_can_recover(tmp_path):
    closed = []

    def factory(cfg, adapter):
        if adapter.name == "bad":
            raise ValueError("failed model load")

        class Model:
            def close(self):
                closed.append(str(adapter))

        return Model()

    manager = RuntimeManager({})
    with patch("nice_validation.service.Runtime", side_effect=factory):
        original = manager.get(tmp_path / "first", "a")
        assert manager.get(tmp_path / "first", "a") is original
        with pytest.raises(ValueError):
            manager.get(tmp_path / "bad", "b")
        assert manager.runtime is None and manager.key is None
        assert manager.get(tmp_path / "first", "a") is not original
        manager.close()
    assert len(closed) == 2


def test_service_reloads_changed_adapter_at_same_path(tmp_path):
    manager = RuntimeManager({})

    class Model:
        def __init__(self, *args):
            self.closed = False

        def close(self):
            self.closed = True

    with patch("nice_validation.service.Runtime", Model):
        first = manager.get(tmp_path, "old")
        second = manager.get(tmp_path, "new")
        assert first.closed and first is not second


def editor(tmp_path, **args):
    env = {
        **os.environ,
        "NICE_EDITOR_HISTORY": str(tmp_path / "history"),
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.run(
        [sys.executable, "-c", EDITOR_PROGRAM],
        input=json.dumps(args),
        capture_output=True,
        text=True,
        env=env,
        encoding="utf-8",
    )


def test_editor_mutation_undo_and_failed_edit_history(tmp_path):
    file = tmp_path / "source.py"
    assert (
        editor(
            tmp_path, command="create", path=str(file), file_text="original"
        ).returncode
        == 0
    )
    assert (
        editor(
            tmp_path,
            command="str_replace",
            path=str(file),
            old_str="original",
            new_str="changed",
        ).returncode
        == 0
    )
    assert (
        editor(
            tmp_path,
            command="str_replace",
            path=str(file),
            old_str="missing",
            new_str="bad",
        ).returncode
        != 0
    )
    assert editor(tmp_path, command="undo_edit", path=str(file)).returncode == 0
    assert file.read_text() == "original"
    assert editor(tmp_path, command="undo_edit", path=str(file)).returncode != 0


def test_editor_insert_into_file_without_last_newline(tmp_path):
    file = tmp_path / "source.py"
    file.write_bytes(b"first")
    assert (
        editor(
            tmp_path, command="insert", path=str(file), insert_line=1, new_str="second"
        ).returncode
        == 0
    )
    assert file.read_bytes() == b"first\nsecond\n"


def test_preflight_reports_all_missing_prerequisites_without_weights():
    cfg = config("configs/legacy.yaml")
    info = {"gpu": {"unavailable": "test"}, "docker": {"unavailable": "test"}}
    with patch("nice_validation.preflight.doctor", return_value=info):
        report = preflight(cfg)
    assert report["status"] == "FAIL"
    assert report["weights_loaded"] is report["weights_downloaded"] is False
    failed = {c["check"] for c in report["checks"] if c["status"] == "FAIL"}
    assert {
        "model.revision",
        "agent.swebench_revision",
        "authorized_gpu",
        "model.path",
    } <= failed


def test_short_real_smoke_cannot_authorize_main_rollouts(tmp_path):
    cfg = config("configs/development.yaml")
    cfg["mode"], cfg["output"] = "real", str(tmp_path)
    run = Run(cfg)
    with run.stage("smoke") as output:
        path = run.root / "smoke/result.json"
        write(
            path,
            {
                "mode": "real",
                "target_model_tested": True,
                "capacity_probe": {"input_tokens": 128, "update_completed": True},
            },
        )
        output.append(path)
    with pytest.raises(ValueError, match="full candidate context"):
        require_real_smoke(run)


def test_cleanup_only_removes_containers_with_the_exact_run_label(tmp_path):
    write(tmp_path / "config/snapshot.json", {"identity": "a" * 64})
    calls = []

    def docker(command, **kwargs):
        calls.append(command)
        value = (
            "c" * 12
            if command[1] == "ps"
            else json.dumps({"nice.run": "a" * 64}) if command[1] == "inspect" else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=value, stderr="")

    with patch("nice_validation.cleanup.subprocess.run", side_effect=docker):
        listed = cleanup_containers(tmp_path)
        assert listed["containers"][0]["removed"] is False
        assert not any(command[1] == "rm" for command in calls)
        cleanup_containers(tmp_path, execute=True)
    assert calls[-1] == ["docker", "rm", "-f", "c" * 12]


def test_cleanup_refuses_foreign_labels(tmp_path):
    write(tmp_path / "config/snapshot.json", {"identity": "a" * 64})

    def docker(command, **kwargs):
        assert command[1] != "rm"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "c" * 12 if command[1] == "ps" else json.dumps({"nice.run": "b" * 64})
            ),
            stderr="",
        )

    with patch("nice_validation.cleanup.subprocess.run", side_effect=docker):
        with pytest.raises(ValueError, match="label differs"):
            cleanup_containers(tmp_path, execute=True)
