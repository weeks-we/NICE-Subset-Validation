"""Failure injection for unpublished adapters, checkpoint chains and service health."""

import io
import json
from pathlib import Path
import sys
import types

import pytest

from nice_validation.core import config, read, write
from nice_validation.model import initialize_adapter, tree_identity
from nice_validation.service import matching_health, service_identity
from nice_validation import supervisor as sup
from test_online import GPU_A, GPU_B
from test_training import train_tiny


def initialization_fixture(tmp_path, monkeypatch):
    cfg = config("configs/legacy.yaml")
    cfg["execution"] = "server"
    cfg["model"].update(
        {
            "path": str(tmp_path / "base"),
            "revision": "c" * 40,
            "adapter_path": str(tmp_path / "initial"),
        }
    )
    write(tmp_path / "base/model_lock.json", {"synthetic": True})
    fake = types.ModuleType("peft")
    fake.LoraConfig = lambda **kwargs: kwargs
    state = {"fail": False, "saved": []}

    class Adapter:
        def save_pretrained(self, path, **kwargs):
            state["saved"].append(path)
            path.mkdir(parents=True)
            (path / "fixture.txt").write_text("synthetic adapter without real weights")
            if state["fail"]:
                raise OSError("injected interrupted save")

    fake.get_peft_model = lambda *a, **k: Adapter()
    monkeypatch.setitem(sys.modules, "peft", fake)
    model = types.SimpleNamespace(
        named_modules=lambda: [
            ("language_model." + name, None) for name in cfg["model"]["lora_targets"]
        ]
    )
    monkeypatch.setattr("nice_validation.model.load_base", lambda c: model)
    return cfg, state


def test_initial_adapter_is_published_only_after_complete_save_and_retries_safely(
    tmp_path, monkeypatch
):
    cfg, state = initialization_fixture(tmp_path, monkeypatch)
    state["fail"] = True
    with pytest.raises(OSError, match="interrupted"):
        initialize_adapter(cfg)
    target = Path(cfg["model"]["adapter_path"])
    assert not target.exists()
    abandoned = state["saved"][0]
    assert abandoned.is_dir() and not (abandoned / "identity.json").exists()
    state["fail"] = False
    identity = initialize_adapter(cfg)
    assert (target / "identity.json").is_file()
    assert tree_identity(target) == identity["adapter_sha256"]
    assert abandoned.is_dir()
    assert initialize_adapter(cfg) == identity
    assert len(state["saved"]) == 2


@pytest.mark.parametrize("change", ["model-lock", "targets", "incomplete"])
def test_initial_adapter_reuse_rejects_changed_inputs_and_preserves_incomplete_dirs(
    tmp_path, monkeypatch, change
):
    cfg, _ = initialization_fixture(tmp_path, monkeypatch)
    target = Path(cfg["model"]["adapter_path"])
    if change == "incomplete":
        target.mkdir()
        (target / "keep.txt").write_text("preserve old evidence")
    else:
        initialize_adapter(cfg)
        if change == "model-lock":
            write(tmp_path / "base/model_lock.json", {"synthetic": "changed"})
        else:
            cfg["model"]["lora_targets"] = ["q_proj"]
    with pytest.raises(ValueError, match="incomplete|initialization config mismatch"):
        initialize_adapter(cfg)
    assert target.is_dir()


@pytest.mark.parametrize("change", ["earlier-file", "step-number", "identity"])
def test_resume_rejects_corruption_anywhere_in_published_checkpoint_chain(
    tmp_path, change
):
    folder = tmp_path / "branch"
    train_tiny(folder, stop=3)
    earlier = folder / "step_0001.pt"
    if change == "earlier-file":
        earlier.write_bytes(b"corrupt earlier boundary, latest file intact")
    else:
        metadata = read(earlier.with_suffix(".json"))
        metadata["step" if change == "step-number" else "identity"] = (
            99 if change == "step-number" else "other-run"
        )
        write(earlier.with_suffix(".json"), metadata)
    with pytest.raises(ValueError, match="checkpoint chain"):
        train_tiny(folder)


@pytest.mark.parametrize("change", [None, "model", "output", "gpu", "malformed"])
def test_service_health_requires_exact_model_run_and_device(tmp_path, change):
    cfg = config("configs/legacy.yaml")
    cfg["output"] = str(tmp_path / "run")
    cfg["model"]["gpu_uuids"] = [GPU_A]
    health = {
        "status": "ready",
        "service_identity": service_identity(cfg),
        "gpu_uuids": [GPU_B],
    }
    if change == "model":
        cfg["model"]["revision"] = "e" * 40
    elif change == "output":
        cfg["output"] = str(tmp_path / "other")
    elif change == "gpu":
        health["gpu_uuids"] = [GPU_A]
    elif change == "malformed":
        health = ["ready"]
    assert matching_health(health, cfg, GPU_B) is (change is None)


def test_prepared_preflight_checks_service_identity_and_bypasses_proxy(
    tmp_path, monkeypatch
):
    from nice_validation.preflight import preflight

    cfg = config("configs/legacy.yaml")
    cfg["output"] = str(tmp_path / "run")
    cfg["model"]["gpu_uuids"] = [GPU_A]
    cfg["resources"] = {"gpu_count": 2, "gpu_uuids": [GPU_A, GPU_B]}
    monkeypatch.setattr(
        "nice_validation.preflight.doctor", lambda c: {"gpu": {}, "docker": {}}
    )
    value = {"status": "ready", "service_identity": "wrong-run", "gpu_uuids": [GPU_B]}
    proxies = []

    class Opener:
        def open(self, *args, **kwargs):
            return io.BytesIO(json.dumps(value).encode())

    def build(handler):
        proxies.append(handler.proxies)
        return Opener()

    monkeypatch.setattr("nice_validation.preflight.urllib.request.build_opener", build)

    def service_check():
        return next(
            row
            for row in preflight(cfg, prepared=True)["checks"]
            if row["check"].startswith("service:")
        )

    assert service_check()["status"] == "FAIL"
    value["service_identity"] = service_identity(cfg)
    assert service_check()["status"] == "PASS"
    assert proxies == [{}, {}]


def test_service_is_stopped_if_pid_evidence_cannot_be_published(tmp_path, monkeypatch):
    (tmp_path / "controller").mkdir()
    cfg = config("configs/legacy.yaml")
    cfg["model"]["gpu_uuids"] = [GPU_B]
    monkeypatch.setattr(sup, "health", lambda url: None)

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, value):
            pass

        def connect_ex(self, address):
            return 1

    monkeypatch.setattr(sup.socket, "socket", Probe)
    process = types.SimpleNamespace(pid=12345)
    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: process)

    def disk_full(*args):
        raise OSError("injected disk full during PID publication")

    monkeypatch.setattr(sup, "write", disk_full)
    stopped = []

    def stop(owned, log):
        stopped.append(owned)
        log.close()

    monkeypatch.setattr(sup, "stop_service", stop)
    with pytest.raises(OSError, match="disk full"):
        sup.start_service(tmp_path, cfg, tmp_path / "service.local.yaml", 8765)
    assert stopped == [process]


@pytest.mark.parametrize("resolved", [False, True])
def test_completed_episode_is_recovered_without_rerunning_after_final_publication_interruption(
    tmp_path, monkeypatch, resolved
):
    from nice_validation.core import Run, sha
    from nice_validation.harness import _episode_job

    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path / "run")
    cfg["mode"] = "real"
    run = Run(cfg)
    task = {"instance_id": "fixture-task"}
    folder = run.root / "evaluation/initial"
    final = folder / "fixture-task/rollout_0/result.json"
    calls = []

    def episode(run, task, model, nice, seed, attempt):
        calls.append(attempt)
        write(attempt / "trace.json", [{"synthetic": "completed interaction"}])
        (attempt / "patch.diff").write_text("synthetic patch")
        return {
            "status": "RESOLVED" if resolved else "TASK_FAILED",
            "resolved": resolved,
            "stop_reason": "SUBMITTED",
        }

    monkeypatch.setattr("nice_validation.harness.real_episode", episode)

    def interrupted(path, value):
        if path == final:
            raise OSError("interrupted at final result publication")
        write(path, value)

    monkeypatch.setattr("nice_validation.harness.write", interrupted)
    with pytest.raises(OSError, match="final result"):
        _episode_job(run, "test", "initial", False, 0, task, 0, folder)
    trace = calls[0] / "trace.json"
    before = sha(trace)
    monkeypatch.setattr("nice_validation.harness.write", write)
    recovered = _episode_job(run, "test", "initial", False, 0, task, 0, folder)
    assert recovered["resolved"] is resolved and recovered["attempt_id"] == 0
    assert len(calls) == 1 and sha(trace) == before
    assert final.is_file()
    assert len(list(final.parent.glob("attempt_*"))) == 1


@pytest.mark.parametrize(
    "status,resolved", [("RESOLVED", False), ("TASK_FAILED", True)]
)
def test_contradictory_result_labels_cannot_produce_a_formal_success_rate(
    status, resolved
):
    from nice_validation.analysis import task_rate

    result = task_rate(
        [{"task_id": "a", "status": status, "resolved": resolved}], ["a"]
    )
    assert result["status"] == "incomplete" and result["rate"] is None
    assert result["completed_tasks"] == 0


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "learning_rate", 0),
        ("training", "learning_rate", -1),
        ("training", "learning_rate", float("inf")),
        ("training", "epsilon", 0),
        ("training", "max_grad_norm", -1),
        ("agent", "max_actions", 0),
        ("agent", "context_length", -1),
        ("agent", "total_new_tokens", True),
        ("data", "max_length", 0),
        ("score", "projection_dim", 0),
        ("experiment", "sample_attempts", -1),
    ],
)
def test_invalid_numeric_budgets_and_zero_learning_rate_are_rejected_before_execution(
    tmp_path, section, key, value
):
    path = tmp_path / "invalid.yaml"
    import yaml

    path.write_text(
        yaml.safe_dump(
            {"extends": str(Path("configs/legacy.yaml").resolve()), section: {key: value}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive"):
        config(path)
