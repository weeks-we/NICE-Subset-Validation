"""CPU-only online setup, source provenance and orchestration contract tests."""

import copy
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from nice_validation.core import config, read, sha, write
from nice_validation.online import configure_server
from nice_validation import supervisor as sup
from nice_validation.transfer_sources import checkout, acquire_transfer
from test_online import GPU_A, GPU_B, records


def test_existing_model_preserves_actual_revision_and_checks_content(tmp_path):
    model = tmp_path / "model"
    write(model / "config.json", {"fixture": True})
    write(
        model / "model_lock.json",
        {
            "id": "Qwen/Qwen3.8-27B",
            "revision": "9" * 40,
            "files": {"config.json": {"sha256": sha(model / "config.json")}},
        },
    )
    with patch(
        "nice_validation.online.resolve_sources", side_effect=lambda c: records()
    ):
        paths = configure_server(
            tmp_path / "store", GPU_A, GPU_B, tmp_path / "cfg", model
        )
    assert config(paths["server_config"])["model"]["revision"] == "9" * 40
    assert (
        read(tmp_path / "store/data/source_revisions.json")["model"]["revision"]
        == "9" * 40
    )
    (model / "config.json").write_text("tampered")
    with pytest.raises(ValueError, match="differs"):
        configure_server(tmp_path / "store", GPU_A, GPU_B, tmp_path / "cfg", model)


def test_unproven_model_and_output_parent_are_refused(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(ValueError, match="verified model_lock"):
        configure_server(tmp_path / "store", GPU_A, GPU_B, tmp_path / "cfg", model)
    with pytest.raises(ValueError, match="cannot contain"):
        configure_server(
            tmp_path / "future/store",
            GPU_A,
            GPU_B,
            tmp_path / "cfg",
            tmp_path / "future",
        )


@pytest.mark.parametrize("interrupt", [None, "init", "fetch"])
def test_real_git_first_checkout_recovery_and_dirty_rejection(
    tmp_path, monkeypatch, interrupt
):
    source, target = tmp_path / "upstream", tmp_path / "checkout"
    source.mkdir()
    original = subprocess.run

    def git(*args, **kwargs):
        return original(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
            **kwargs,
        )

    git("init")
    (source / "instruction.md").write_text("frozen task")
    git("add", "instruction.md")
    git(
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = git("rev-parse", "HEAD").stdout.strip()
    bundle = tmp_path / "fixture.bundle"
    git("bundle", "create", str(bundle), "HEAD")
    attempts = []

    def execute(argv, **kwargs):
        args = list(argv)
        if interrupt and not attempts and interrupt in args:
            attempts.append("interrupted")
            raise ConnectionError("fixture interrupted")
        if "fetch" in args:
            # Real Git bundle transport avoids Windows sandbox sh/pipe restrictions.
            args[args.index("origin")] = str(bundle)
            args[-1] = "HEAD"
        return original(args, **kwargs)

    monkeypatch.setattr("nice_validation.transfer_sources.subprocess.run", execute)
    if interrupt:
        with pytest.raises(ConnectionError):
            checkout("fixture/repo", revision, target)
    assert checkout("fixture/repo", revision, target) == target
    assert (target / "instruction.md").read_text() == "frozen task"
    assert checkout("fixture/repo", revision, target) == target
    (target / "instruction.md").write_text("dirty")
    with pytest.raises(ValueError, match="changed"):
        checkout("fixture/repo", revision, target)


def test_terminal_acquisition_uses_only_registered_tasks(tmp_path, monkeypatch):
    cfg = config("configs/main.yaml")
    cfg["execution"] = "server"
    cfg["transfer"].update(
        {
            "source_lock": str(tmp_path / "sources.json"),
            "checkouts": str(tmp_path / "repos"),
            "sample_size": 1,
        }
    )
    repo = tmp_path / "native"
    for name in ["registered", "extra"]:
        write(repo / "tasks" / name / "task.toml", {})
        (repo / "tasks" / name / "instruction.md").write_text(name)
    for spec in cfg["transfer"]["suites"].values():
        spec["revision"] = "a" * 40
    cfg["transfer"]["suites"]["terminal"]["registered_tasks"] = [
        {"name": "registered", "path": "tasks/registered"}
    ]
    monkeypatch.setattr("nice_validation.transfer_sources.checkout", lambda *a: repo)
    monkeypatch.setattr(
        "nice_validation.acquire.hf_split", lambda *a: {"fixture": True}
    )
    # Pro's task root is deliberately separate from Terminal's registry list.
    cfg["transfer"]["suites"]["pro"]["task_subdir"] = "tasks"
    result = acquire_transfer(cfg)
    assert result["terminal"]["records"] == 1
    assert [r["instance_id"] for r in read(Path(result["terminal"]["path"]))] == [
        "registered"
    ]


def test_child_environment_routes_gpu_and_keeps_loopback_off_proxy(monkeypatch):
    monkeypatch.setenv("no_proxy", "internal.example")
    env = sup.child_environment(GPU_A)
    assert env["CUDA_VISIBLE_DEVICES"] == GPU_A
    assert {"localhost", "127.0.0.1", "internal.example"} <= set(
        env["NO_PROXY"].split(",")
    )
    assert env["NO_PROXY"] == env["no_proxy"]


def test_service_reuse_requires_exact_model_source_and_gpu(tmp_path, monkeypatch):
    cfg = config("configs/main.yaml")
    cfg["model"]["gpu_uuids"] = [GPU_B]
    monkeypatch.setattr(sup, "service_identity", lambda c: c["model"]["revision"])
    cfg["model"]["revision"] = "a" * 40
    health = {"status": "ready", "service_identity": "a" * 40, "gpu_uuids": [GPU_B]}
    monkeypatch.setattr(sup, "health", lambda u: health)
    monkeypatch.setattr(
        sup.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    assert sup.start_service(tmp_path, cfg, tmp_path / "cfg", 8765) == (None, None)
    health["gpu_uuids"] = [GPU_A]
    with pytest.raises(ValueError, match="identity/GPU"):
        sup.start_service(tmp_path, cfg, tmp_path / "cfg", 8765)


@pytest.mark.parametrize("failure", [None, "preflight", "experiment", "audit"])
def test_server_orders_entire_flow_and_always_stops_owned_service(
    tmp_path, monkeypatch, failure
):
    store = tmp_path / "store"
    with patch(
        "nice_validation.online.resolve_sources", side_effect=lambda c: records()
    ):
        paths = configure_server(store, GPU_A, GPU_B, tmp_path / "configs")
    cfg = config(paths["server_config"])
    monkeypatch.setattr("nice_validation.online.configure_server", lambda *a: paths)
    monkeypatch.setattr(sup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sup.sys, "version_info", (3, 12))
    monkeypatch.setattr(sup, "server_only", lambda c: None)
    calls = []

    def command(root, name, argv, env):
        assert env["CUDA_VISIBLE_DEVICES"] == GPU_A
        calls.append(name)
        if name == failure:
            raise RuntimeError("injected failure")
        if name == "audit":
            write(Path(cfg["output"]) / "audit.json", {"status": "PASS"})

    monkeypatch.setattr(sup, "command", command)
    monkeypatch.setattr(
        sup,
        "start_service",
        lambda *a: (calls.append("start-service") or "owned", "log"),
    )
    monkeypatch.setattr(
        sup, "stop_service", lambda p, log: calls.append("stop-service")
    )
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            sup.run_server(store, GPU_A, GPU_B, tmp_path / "configs")
        assert read(store / "controller/completion.json")["status"] == "failed"
        assert ("stop-service" in calls) == (failure != "preflight")
        assert "bundle" not in calls
    else:
        result = sup.run_server(store, GPU_A, GPU_B, tmp_path / "configs")
        assert result["status"] == "completed" and result["service_reused"] is False
        assert calls == [
            "install-harness",
            "preflight",
            "prepare",
            "start-service",
            "experiment",
            "stop-service",
            "analyze",
            "audit",
            "bundle",
        ]
    assert not (store / "controller.lock").exists()


def test_health_identity_changes_with_model_output_and_role_is_independent(tmp_path):
    from nice_validation.service import service_identity

    cfg = config("configs/main.yaml")
    cfg["output"] = str(tmp_path)
    original = service_identity(cfg)
    changed = copy.deepcopy(cfg)
    changed["model"]["gpu_uuids"] = [GPU_A]
    assert service_identity(changed) == original
    changed["model"]["revision"] = "a" * 40
    assert service_identity(changed) != original
    changed = copy.deepcopy(cfg)
    changed["output"] = str(tmp_path / "other")
    assert service_identity(changed) != original
