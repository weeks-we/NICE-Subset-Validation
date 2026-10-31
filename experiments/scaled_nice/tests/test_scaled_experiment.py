"""Independent scaled design, GPU phases and budget-exit grading contracts."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nice_validation.core import Run, config, read, write
from nice_validation.data import prepare
from nice_validation.design import counts, strata_ranges
from nice_validation.online import configure_server, resolve_sources
from nice_validation.parallel_training import train_parallel, validate_receipt
from nice_validation.scoring import score
from nice_validation.harness import episodes, real_episode
from nice_validation.subsets import make_subsets
from nice_validation import supervisor as sup
from test_online import GPU_A, GPU_B, records


GPU_C, GPU_D = GPU_A.replace("a", "c"), GPU_A.replace("a", "d")


def small_scaled(tmp_path):
    cfg = config("configs/scaled-mock.yaml")
    cfg["output"] = str(tmp_path)
    cfg["experiment"].update(candidates=90, validation=6, subset_size=9)
    return cfg


def test_scaled_default_budgets_and_smaller_model():
    cfg = config("configs/scaled.yaml")
    assert counts(cfg) == {
        "candidates": 2000,
        "validation": 100,
        "subset_size": 150,
        "branches": 12,
        "nice_episodes": 400,
        "evaluation_episodes": 1300,
        "main_episodes": 1700,
        "transfer_episodes": 0,
        "all_episodes": 1700,
        "training_updates": 456,
        "training_exposures": 1800,
        "selection_fraction": 0.075,
    }
    assert cfg["model"]["id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert cfg["model"]["loader"] == "AutoModelForCausalLM"
    assert cfg["agent"]["max_actions"] == 75
    assert not cfg["transfer"]["enabled"]
    assert strata_ranges(1000) == {
        "low": (0, 333),
        "middle": (333, 666),
        "high": (666, 1000),
    }
    assert strata_ranges(2000) == {
        "low": (0, 666),
        "middle": (666, 1333),
        "high": (1333, 2000),
    }


@pytest.mark.parametrize(
    "key,value",
    [("candidates", 0), ("validation", True), ("subset_size", 1000), ("rollouts", 0)],
)
def test_invalid_scaled_dimensions_fail_before_execution(tmp_path, key, value):
    cfg = config("configs/scaled.yaml")
    cfg["experiment"][key] = value
    path = tmp_path / "bad.json"
    write(path, cfg)
    with pytest.raises(ValueError):
        config(path)


def test_model_pending_blocks_all_source_resolution(tmp_path):
    pending = config("configs/scaled.yaml")
    pending["model"]["id"] = None
    base = tmp_path / "pending.json"
    write(base, pending)
    with patch(
        "nice_validation.online.resolve_sources",
        side_effect=AssertionError("network called"),
    ):
        with pytest.raises(ValueError, match="Model is pending"):
            configure_server(
                tmp_path / "storage",
                GPU_A,
                GPU_B,
                tmp_path / "cfg",
                base_config=base,
            )


def test_configure_four_shared_phase_gpus_and_selected_model(tmp_path):
    def sources(cfg):
        result = records()
        result["model"]["id"] = cfg["model"]["id"]
        assert cfg["transfer"]["enabled"] is False
        return result

    with patch("nice_validation.online.resolve_sources", side_effect=sources):
        paths = configure_server(
            tmp_path / "storage",
            GPU_A,
            GPU_B,
            tmp_path / "cfg",
            service_gpus=[GPU_B, GPU_C, GPU_D],
            base_config=Path("configs/scaled.yaml"),
            model_id="unit/small",
            model_loader="AutoModelForCausalLM",
        )
    cfg = config(paths["server_config"])
    assert cfg["model"]["id"] == "unit/small"
    assert cfg["model"]["loader"] == "AutoModelForCausalLM"
    assert cfg["resources"]["gpu_count"] == 4
    assert cfg["resources"]["gpu_uuids"] == [GPU_A, GPU_B, GPU_C, GPU_D]
    assert cfg["resources"]["inference_gpu_uuids"] == [GPU_B, GPU_C, GPU_D, GPU_A]
    assert cfg["agent"]["workers"] == len(paths["service_configs"]) == 4
    assert [
        config(path)["model"]["gpu_uuids"][0] for path in paths["service_configs"]
    ] == [GPU_B, GPU_C, GPU_D, GPU_A]
    assert not cfg["transfer"]["enabled"]


def test_scaled_source_resolution_never_contacts_external_benchmarks(monkeypatch):
    import huggingface_hub

    datasets, models, urls = [], [], []

    class API:
        def model_info(self, identifier):
            models.append(identifier)
            return SimpleNamespace(sha="c" * 40)

        def dataset_info(self, identifier):
            datasets.append(identifier)
            return SimpleNamespace(sha="a" * 40)

    def github(request, **kwargs):
        urls.append(request.full_url)
        return io.BytesIO(json.dumps({"sha": "d" * 40}).encode())

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr("urllib.request.urlopen", github)
    sources = resolve_sources(config("configs/scaled.yaml"))
    assert set(sources) == {"training", "validation", "model", "swebench", "swesmith"}
    assert datasets == ["SWE-bench/SWE-smith-trajectories", "SWE-bench/SWE-smith"]
    assert models == ["Qwen/Qwen2.5-Coder-7B-Instruct"]
    assert len(urls) == 2
    assert all(
        "/SWE-bench/SWE-bench/" in url or "/SWE-bench/SWE-smith/" in url for url in urls
    )


@pytest.mark.parametrize(
    "failure",
    [None, "score", "training-batch-probe", "parallel-training", "evaluate-low_11"],
)
def test_scaled_supervisor_releases_all_inference_before_training_and_cleans_up(
    tmp_path, monkeypatch, failure
):
    store = tmp_path / "store"
    with patch(
        "nice_validation.online.resolve_sources",
        side_effect=lambda cfg: {
            **records(),
            "model": {"id": cfg["model"]["id"], "revision": "c" * 40},
        },
    ):
        paths = configure_server(
            store,
            GPU_A,
            GPU_B,
            tmp_path / "cfg",
            service_gpus=[GPU_B, GPU_C, GPU_D],
            base_config=Path("configs/scaled.yaml"),
        )
    cfg = config(paths["server_config"])
    monkeypatch.setattr(
        "nice_validation.online.configure_server", lambda *a, **kw: paths
    )
    monkeypatch.setattr(sup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sup.sys, "version_info", (3, 12))
    monkeypatch.setattr(sup, "server_only", lambda c: None)
    active, calls = set(), []

    def start(root, service, filename, port):
        gpu = service["model"]["gpu_uuids"][0]
        assert gpu not in active
        active.add(gpu)
        calls.append("start")
        return gpu, None

    def stop(gpu, log):
        active.remove(gpu)
        calls.append("stop")

    def command(root, name, argv, env):
        calls.append(name)
        if name in {
            "score",
            "subsets",
            "training-batch-probe",
            "parallel-training",
            "analyze",
            "audit",
            "bundle",
        }:
            assert not active
        elif name in {
            "smoke",
            "nice-rollouts",
            "initial-evaluation",
            "preflight-prepared",
            "preflight-evaluation",
        } or name.startswith("evaluate-"):
            assert len(active) == 4
        if name == failure:
            raise RuntimeError("injected failure")
        if name == "subsets":
            write(
                Path(cfg["output"]) / "manifests/subsets.json",
                [
                    {"subset_id": f"{group}_{seed}"}
                    for group in ["low", "middle", "high", "random"]
                    for seed in [11, 22, 33]
                ],
            )
        if name == "audit":
            write(Path(cfg["output"]) / "audit.json", {"status": "PASS"})

    monkeypatch.setattr(sup, "start_service", start)
    monkeypatch.setattr(sup, "stop_service", stop)
    monkeypatch.setattr(sup, "command", command)
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            sup.run_server(store, GPU_A, GPU_B, tmp_path / "cfg")
        assert read(store / "controller/completion.json")["status"] == "failed"
        assert "bundle" not in calls
    else:
        result = sup.run_server(store, GPU_A, GPU_B, tmp_path / "cfg")
        assert result["status"] == "completed" and result["gpu_count"] == 4
        assert calls.count("start") == calls.count("stop") == 8
        assert calls.index("parallel-training") < calls.index("evaluate-low_11")
        assert not any("transfer" in name for name in calls)
    assert not active and not (store / "controller.lock").exists()


def test_scaled_prepare_rollout_score_parallel_training_resume(tmp_path):
    run = Run(small_scaled(tmp_path))
    prepare(run)
    assert len(read(run.root / "manifests/candidates.json")) == 90
    assert len(read(run.root / "manifests/validation.json")) == 6
    assert not (run.root / "transfer").exists()
    episodes(run, "initial", nice=True)
    assert len(read(run.root / "rollouts/nice/results.json")) == 24
    score(run)
    diagnostic = read(run.root / "scores/diagnostics.json")
    assert diagnostic["reward_coverage"] == (6 - diagnostic["zero_gradient_tasks"]) / 6
    make_subsets(run)
    train_parallel(run)
    for subset in read(run.root / "manifests/subsets.json"):
        sid = subset["subset_id"]
        assert run.valid("train:" + sid)
        report = read(run.root / "training" / sid / "training.json")
        assert report["updates"] == 3 and report["exposures"] == 9
        assert report["effective_batch"] == 4 and report["micro_batch_size"] == 2
        assert len(report["members"]) == 9
    original = read(run.state_path)
    with patch(
        "nice_validation.parallel_training.subprocess.Popen",
        side_effect=AssertionError("reran completed training"),
    ):
        train_parallel(run)
    assert read(run.state_path) == original
    receipt = read(run.root / "training/low_11/worker_receipt.json")
    receipt["gpu_uuid"] = GPU_A.replace("a", "f")
    with pytest.raises(ValueError, match="identity"):
        validate_receipt(run, "low_11", receipt)


@pytest.mark.parametrize(
    "policy,patch_text,expected",
    [("fail", "fix", False), ("grade_patch", "fix", True), ("grade_patch", "", False)],
)
def test_budget_exit_patch_grading_changes_only_explicit_policy(
    tmp_path, monkeypatch, policy, patch_text, expected
):
    import nice_validation.harness as module

    cfg = config("configs/development.yaml")
    cfg["agent"].update(max_actions=1, budget_exhaustion_policy=policy)
    cfg["model"]["adapter_path"] = str(tmp_path / "adapter")
    adapter = Path(cfg["model"]["adapter_path"])
    adapter.mkdir()
    monkeypatch.setattr(
        "nice_validation.model.tree_identity", lambda path: "adapter-id"
    )

    class Environment:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def tools(self):
            return {"bash": SimpleNamespace(invoke=lambda args: "ok")}

        def patch(self):
            return patch_text

    response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "true"}},
                }
            ],
        },
        "usage": {"input_tokens": 2, "output_tokens": 1},
        "input_ids": [1, 2],
    }
    monkeypatch.setattr(module, "DockerTask", Environment)
    monkeypatch.setattr(
        module, "request_generation", lambda *a, **kw: copy.deepcopy(response)
    )
    graded = []
    monkeypatch.setattr(module, "official_grade", lambda *a: graded.append(a) or True)
    run = SimpleNamespace(
        cfg=cfg, identity="unit", root=tmp_path, event=lambda *a, **kw: None
    )
    result = real_episode(
        run,
        {"instance_id": "unit", "problem_statement": "fix"},
        "initial",
        False,
        2026,
        tmp_path / "episode",
    )
    assert result["stop_reason"] == "ACTION_LIMIT"
    assert result["resolved"] is expected
    assert bool(graded) is expected
    assert result["automatic_submission"] is expected
    assert result["patch_sent_for_grading"] is expected
