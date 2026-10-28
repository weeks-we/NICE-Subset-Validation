import json
from pathlib import Path
import os
import runpy
import subprocess
import sys
import types

import pytest

from nice_validation.core import write
from nice_validation.environments import prepare_smith_image


def test_prepared_agent_image_has_no_hidden_parent_objects(tmp_path):
    # A separate synthetic Git repository; never writes the user's project .git.
    repo = tmp_path / "fixture-repo"
    repo.mkdir()
    assert repo.resolve().is_relative_to(Path.cwd().resolve())
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }

    def git(*args, input=None, check=True):
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            input=input,
            capture_output=True,
            text=True,
            env=env,
            check=check,
        )

    git("init")
    (repo / "public.py").write_text("public task code")
    (repo / "hidden_test.py").write_text("hidden test fixture")
    git("add", ".")
    parent = git(
        "commit-tree", git("write-tree").stdout.strip(), input="hidden fixture parent\n"
    ).stdout.strip()
    (repo / "hidden_test.py").unlink()
    git("add", "-u")
    tip = git(
        "commit-tree",
        git("write-tree").stdout.strip(),
        "-p",
        parent,
        input="public fixture tip\n",
    ).stdout.strip()
    git("update-ref", "refs/heads/fixture", tip)
    git("symbolic-ref", "HEAD", "refs/heads/fixture")
    git("remote", "add", "origin", "https://example.invalid/fixture.git")
    closed, creation = [], []

    class Container:
        def start(self):
            pass

        def exec_run(self, command, **kwargs):
            if command[:2] == ["git", "fetch"]:
                (repo / ".git/FETCH_HEAD").write_text(tip + "\n")
                return types.SimpleNamespace(exit_code=0, output=b"")
            if command[0] == "python":
                command = [sys.executable, *command[1:]]
            result = subprocess.run(command, cwd=repo, env=env, capture_output=True)
            return types.SimpleNamespace(
                exit_code=result.returncode, output=result.stdout + result.stderr
            )

        def commit(self):
            assert not (repo / "hidden_test.py").exists()
            assert git("cat-file", "-e", parent, check=False).returncode != 0
            assert git("rev-parse", "HEAD").stdout.strip() == tip
            return types.SimpleNamespace(id="sha256:" + "a" * 64)

        def remove(self, **kwargs):
            closed.append(kwargs)

    def create(*args, **kwargs):
        creation.append(kwargs)
        return Container()

    client = types.SimpleNamespace(containers=types.SimpleNamespace(create=create))
    image, commit = prepare_smith_image(
        client, {"instance_id": "fixture-task"}, types.SimpleNamespace(id="base-image")
    )
    assert commit == tip and image == "sha256:" + "a" * 64
    assert "environment" not in creation[0]  # Proxy credentials were exec-only.
    assert closed == [{"force": True}]


def test_failed_task_ref_fetch_does_not_publish_an_agent_image():
    closed = []

    class Container:
        def start(self):
            pass

        def exec_run(self, command, **kwargs):
            return types.SimpleNamespace(exit_code=1, output=b"fixture failure")

        def commit(self):
            raise AssertionError("Must not publish an incomplete checkout")

        def remove(self, **kwargs):
            closed.append(True)

    client = types.SimpleNamespace(
        containers=types.SimpleNamespace(create=lambda *a, **k: Container())
    )
    with pytest.raises(ValueError, match="preparation failed"):
        prepare_smith_image(
            client,
            {"instance_id": "fixture-task"},
            types.SimpleNamespace(id="base-image"),
        )
    assert closed == [True]


def test_grader_uses_the_frozen_sha_and_official_report_parser(tmp_path, monkeypatch):
    constants = types.ModuleType("swebench.harness.constants")
    constants.RUN_EVALUATION_LOG_DIR = tmp_path / "logs"
    constants.LOG_REPORT, constants.LOG_TEST_OUTPUT, constants.LOG_INSTANCE = (
        "report.json",
        "test_output.txt",
        "run.log",
    )
    constants.APPLY_PATCH_FAIL = "fixture-patch-apply-failed"
    build = types.ModuleType("swebench.harness.docker_build")
    build.close_logger = lambda logger: None
    utils = types.ModuleType("swesmith.harness.utils")
    commits = []

    def run_patch(task, run_id, root, timeout, **kwargs):
        commits.append(kwargs["commit"])
        folder = root / run_id / task["instance_id"]
        folder.mkdir(parents=True)
        (folder / "test_output.txt").write_text("synthetic official test output")
        return object(), False

    utils.run_patch_in_container = run_patch
    grading = types.ModuleType("swesmith.harness.grading")
    grading.get_eval_report = lambda pred, task, output, **kwargs: {"resolved": True}
    for module in [constants, build, utils, grading]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    dataset, predictions = tmp_path / "task.json", tmp_path / "predictions.jsonl"
    write(dataset, [{"instance_id": "fixture-task"}])
    predictions.write_text(
        json.dumps({"instance_id": "fixture-task", "model_patch": "fixture diff"})
        + "\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swesmith_grade.py",
            "--dataset_path",
            str(dataset),
            "--predictions_path",
            str(predictions),
            "--run_id",
            "fixture-run",
            "--commit",
            "b" * 40,
            "--timeout",
            "60",
        ],
    )
    runpy.run_path(
        str(Path("scripts/swesmith_grade.py").resolve()), run_name="__main__"
    )
    assert commits == ["b" * 40]
    assert (
        json.loads(
            (tmp_path / "logs/fixture-run/fixture-task/report.json").read_text()
        )["resolved"]
        is True
    )
