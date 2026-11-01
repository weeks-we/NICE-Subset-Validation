"""Actual Git commits, staged/binary/new files and pristine replay boundaries."""

import asyncio
import importlib
from pathlib import Path
import shlex
import subprocess
import sys
import types

import pytest

from nice_validation.core import config, read
from nice_validation.harness import DockerTask, InfraError, TaskFailure


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout


def repository(tmp_path):
    root = tmp_path / "repository with spaces"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "unit@example.invalid")
    git(root, "config", "user.name", "CPU unit test")
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "config", "core.hooksPath", str(root / ".git" / "disabled-hooks"))
    (root / "main.txt").write_text("original\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "starting task")
    return root, git(root, "rev-parse", "HEAD").strip()


def execute_patch(command, original_run, binary=False):
    value = None
    for part in command.split(" && "):
        value = original_run(
            shlex.split(part),
            capture_output=True,
            **({} if binary else {"text": True, "encoding": "utf-8"}),
        )
        if value.returncode:
            break
    return value


def native_module(monkeypatch):
    module = types.ModuleType("harbor.agents.base")
    module.BaseAgent = object
    monkeypatch.setitem(sys.modules, "harbor.agents.base", module)
    sys.modules.pop("nice_validation.harbor_agent", None)
    return importlib.import_module("nice_validation.harbor_agent")


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_main_patch_survives_agent_commit_and_replays_all_change_types(
    tmp_path, monkeypatch, newline
):
    root, baseline = repository(tmp_path)
    (root / "main.txt").write_bytes(b"committed fix" + newline)
    git(root, "add", ".")
    git(root, "commit", "-m", "agent committed fix")
    (root / "new.txt").write_text("new source\n")
    (root / "binary.bin").write_bytes(bytes(range(256)))
    git(root, "add", "new.txt")
    expected = {
        name: (root / name).read_bytes()
        for name in ["main.txt", "new.txt", "binary.bin"]
    }
    cfg = config("configs/development.yaml")
    cfg["agent"]["workdir"] = str(root)
    task = DockerTask(cfg, {}, tmp_path)
    task.container, task.checkout_commit = "no-real-container", baseline
    original_run = subprocess.run
    monkeypatch.setattr(
        "nice_validation.harness.subprocess.run",
        lambda argv, **kwargs: execute_patch(argv[-1], original_run, binary=True),
    )
    patch = task.patch()
    assert (
        "committed fix" in patch and "GIT binary patch" in patch and "new.txt" in patch
    )
    monkeypatch.setattr("nice_validation.harness.subprocess.run", original_run)
    patch_file = tmp_path / "model.patch"
    patch_file.write_bytes(patch.encode("utf-8"))
    git(root, "reset", "--hard", baseline)
    # Only explicitly named test files are removed; no recursive deletion.
    for name in ["new.txt", "binary.bin"]:
        (root / name).unlink(missing_ok=True)
    git(root, "apply", "--binary", str(patch_file))
    assert {name: (root / name).read_bytes() for name in expected} == expected


@pytest.mark.parametrize(
    "message,exception",
    [
        ("fatal: not a git repository", TaskFailure),
        ("Cannot connect to the Docker daemon", InfraError),
        ("Error response from daemon: No such container", InfraError),
    ],
)
def test_repository_damage_is_not_confused_with_daemon_failure(
    tmp_path, monkeypatch, message, exception
):
    task = DockerTask(config("configs/development.yaml"), {}, tmp_path)
    task.checkout_commit, task.container = "a" * 40, "unit"
    monkeypatch.setattr(
        "nice_validation.harness.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=1, stderr=message),
    )
    with pytest.raises(exception):
        task.patch()


def test_pro_patch_keeps_committed_changes_and_records_frozen_base(
    tmp_path, monkeypatch
):
    module = native_module(monkeypatch)
    root, baseline = repository(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "unit.txt").write_text("random fixture")
    original_run = subprocess.run

    class Environment:
        async def exec(self, command, **kwargs):
            return execute_patch(command, original_run)

    def generation(*args, **kwargs):
        (root / "main.txt").write_bytes(b"pro committed fix\r\n")
        (root / "binary.bin").write_bytes(bytes(range(256)))
        git(root, "add", ".")
        git(root, "commit", "-m", "agent commits")
        return {
            "usage": {"input_tokens": 4, "output_tokens": 2},
            "parse_error": None,
            "message": {"role": "assistant", "content": "done", "tool_calls": []},
        }

    # Native API spells the exit field return_code; Git subprocess uses returncode.
    class NativeEnvironment(Environment):
        async def exec(self, **kwargs):
            if "--output=" in kwargs["command"]:
                tokens = shlex.split(kwargs["command"])
                remote = next(
                    t.removeprefix("--output=")
                    for t in tokens
                    if t.startswith("--output=")
                )
                kwargs["command"] = kwargs["command"].replace(
                    remote,
                    shlex.quote(str(tmp_path / "remote.patch").replace("\\", "/")),
                )
            result = await super().exec(**kwargs)
            return types.SimpleNamespace(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )

        async def download_file(self, source_path, target_path):
            Path(target_path).write_bytes((tmp_path / "remote.patch").read_bytes())

    monkeypatch.setattr(module, "request_generation", generation)
    base = {"root": str(root), "commit": baseline}
    asyncio.run(
        module.service_loop(
            NativeEnvironment(),
            config("configs/legacy.yaml"),
            adapter,
            tmp_path / "logs",
            "public issue",
            types.SimpleNamespace(),
            capture_patch=True,
            patch_base=base,
        )
    )
    assert read(tmp_path / "logs/patch_base.json") == base
    assert "pro committed fix" in (tmp_path / "logs/model.patch").read_text()
    assert read(tmp_path / "logs/episode.json")["task_failure"] is False
    expected = {name: (root / name).read_bytes() for name in ["main.txt", "binary.bin"]}
    git(root, "reset", "--hard", baseline)
    git(root, "apply", "--binary", str(tmp_path / "logs/model.patch"))
    assert {name: (root / name).read_bytes() for name in expected} == expected


@pytest.mark.parametrize(
    "stdout,code",
    [("/app\n" + "a" * 40 + "\n M file\n", 0), ("/app\nnot-a-sha\n", 0), ("", 128)],
)
def test_pro_starting_repository_must_be_clean_and_identified(
    tmp_path, monkeypatch, stdout, code
):
    module = native_module(monkeypatch)

    class Environment:
        async def exec(self, **kwargs):
            return types.SimpleNamespace(stdout=stdout, return_code=code)

    with pytest.raises(InfraError):
        asyncio.run(module.pro_checkout(Environment()))


def test_pro_regrade_rejects_a_different_starting_commit(tmp_path, monkeypatch):
    module = native_module(monkeypatch)
    agent = object.__new__(module.NicePatchReplay)
    agent.expected_images = {"main": "sha256:unit"}
    agent.expected_checkout = {"root": "/app", "commit": "a" * 40}
    monkeypatch.setattr(module, "environment_images", lambda env: agent.expected_images)

    class Environment:
        async def exec(self, **kwargs):
            return types.SimpleNamespace(
                stdout="/app\n" + "b" * 40 + "\n", return_code=0
            )

    with pytest.raises(InfraError, match="starting_checkout_changed"):
        asyncio.run(agent.setup(Environment()))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "3"])
def test_invalid_download_limits_fail_before_any_network_or_import(limit, tmp_path):
    from nice_validation.acquire import hf_split

    with pytest.raises(ValueError, match="positive integer"):
        hf_split(
            "unit/source",
            "a" * 40,
            "tool",
            tmp_path / "rows.jsonl",
            "stream",
            True,
            {},
            limit,
        )


def test_invalid_download_mode_fails_before_network(tmp_path):
    from nice_validation.acquire import hf_split

    with pytest.raises(ValueError, match="mode"):
        hf_split(
            "unit/source",
            "a" * 40,
            "tool",
            tmp_path / "rows.jsonl",
            "invalid",
            True,
            {},
        )


@pytest.mark.parametrize("kind", ["timeout", "task_failed", "missing_base", "healthy"])
def test_pro_regrade_evidence_matches_the_attempt_actually_executed(
    tmp_path, monkeypatch, kind
):
    from nice_validation.core import Run, write
    from nice_validation.transfer import native_episode

    cfg = config("configs/development.yaml")
    cfg["output"], cfg["execution"] = str(tmp_path / "run"), "server"
    cfg["model"]["adapter_path"] = str(tmp_path / "adapter")
    run = Run(cfg)
    cfg["mode"] = "real"
    write(
        run.root / "transfer/native_images.json", {"pro:unit": {"main": "sha256:unit"}}
    )
    task = {"instance_id": "unit", "benchmark": "pro", "task_path": str(tmp_path)}
    result_path = tmp_path / "trial/result.json"
    agent = result_path.parent / "agent"
    write(
        agent / "episode.json",
        {"task_failure": kind == "task_failed", "stop_reason": "SUBMITTED"},
    )
    (agent / "model.patch").write_bytes(b"unit patch\r\n")
    base = {"root": "/app", "commit": "a" * 40}
    if kind != "missing_base":
        write(agent / "patch_base.json", base)
    calls = []
    monkeypatch.setattr(
        "nice_validation.transfer.validate_native_task", lambda task: None
    )
    monkeypatch.setattr(
        "nice_validation.transfer.subprocess.run",
        lambda argv, **kwargs: calls.append(argv)
        or types.SimpleNamespace(returncode=0),
    )
    value = {"verifier_result": {"rewards": {"reward": 1}}}
    if kind == "timeout":
        value["exception_info"] = {"exception_type": "AgentTimeoutError"}
    monkeypatch.setattr(
        "nice_validation.transfer.one_native_result", lambda *args: (value, result_path)
    )
    if kind == "missing_base":
        with pytest.raises(InfraError, match="starting_checkout_evidence_missing"):
            native_episode(run, task, "initial", tmp_path / "attempt")
        assert len(calls) == 1
        return
    result = native_episode(run, task, "initial", tmp_path / "attempt")
    assert result["pristine_regrade"] is (kind == "healthy")
    assert result["resolved"] is (kind == "healthy")
    assert len(calls) == (2 if kind == "healthy" else 1)
    if kind == "healthy":
        spec = read(tmp_path / "attempt/regrade-job.json")
        assert spec["agents"][0]["kwargs"]["expected_checkout"] == base
