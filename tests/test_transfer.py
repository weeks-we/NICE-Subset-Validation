import asyncio
import importlib
import io
import json
from pathlib import Path
import sys
import types

import pytest

from nice_validation.core import Run, append, config, read, sha, write
from nice_validation.data import FixtureTokenizer, fixture, prepare
from nice_validation.harness import InfraError, official_grade
from nice_validation.online import resolve_sources
from nice_validation.transfer import evaluate_transfer, native_job, reward
from nice_validation.transfer_analysis import analyze_transfer, transfer_summary
from nice_validation.transfer_sources import (
    SUITES,
    prepare_transfer,
    validate_native_task,
    task_files,
)


def run_at(path):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(path)
    return Run(cfg)


def test_four_samples_are_frozen_reproducible_and_outcome_independent(tmp_path):
    left, right = run_at(tmp_path / "a"), run_at(tmp_path / "b")
    prepare_transfer(left)
    prepare_transfer(right)
    tasks = read(left.root / "transfer/tasks.json")
    assert tasks == read(right.root / "transfer/tasks.json")
    assert set(tasks) == set(SUITES)
    assert all(
        len(rows) == len({t["instance_id"] for t in rows}) == 50
        for rows in tasks.values()
    )
    assert len({t["instance_id"] for rows in tasks.values() for t in rows}) == 200
    write(
        left.root / "transfer/evaluation/verified/initial/results.json",
        [{"resolved": True}],
    )
    prepare_transfer(left)
    assert read(left.root / "transfer/tasks.json") == tasks
    assert left.valid("prepare-transfer")


def test_swesmith_main_tasks_are_held_out_from_corresponding_training_source(
    tmp_path, monkeypatch
):
    cfg = config("configs/development.yaml")
    cfg["mode"], cfg["output"] = "real", str(tmp_path / "run")
    cfg["transfer"]["enabled"] = False
    cfg["data"].update(
        {
            "training_revision": "a" * 40,
            "validation_revision": "b" * 40,
            "training_path": str(tmp_path / "train.jsonl"),
            "validation_path": str(tmp_path / "tasks.jsonl"),
            "source_lock": str(tmp_path / "source.json"),
        }
    )
    trajectories, _ = fixture()
    tasks = [
        {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "problem_statement": row["messages"][0]["content"],
            "image_name": "synthetic-image",
            "FAIL_TO_PASS": ["tests/test_fixture.py::test_fix"],
            "PASS_TO_PASS": [],
        }
        for row in trajectories
    ]
    for path, rows in [
        (cfg["data"]["training_path"], trajectories),
        (cfg["data"]["validation_path"], tasks),
    ]:
        for row in rows:
            append(Path(path), row)
    write(
        Path(cfg["data"]["source_lock"]),
        {
            kind: {
                "revision": cfg["data"][kind + "_revision"],
                "sha256": sha(Path(cfg["data"][kind + "_path"])),
            }
            for kind in ["training", "validation"]
        },
    )
    monkeypatch.setattr(
        "nice_validation.model.load_tokenizer", lambda c: FixtureTokenizer()
    )
    run = Run(cfg)
    prepare(run)
    main = read(run.root / "manifests/validation.json")
    candidates = read(run.root / "manifests/candidates.json")
    dev = read(run.root / "manifests/development_task.json")
    assert len(main) == 30 and len(candidates) == 300
    assert all(
        task["evaluation_backend"] == "swesmith"
        and task["base_commit"] == task["instance_id"]
        for task in main
    )
    assert not {row["instance_id"] for row in candidates}.intersection(
        {t["instance_id"] for t in main} | {dev["instance_id"]}
    )
    assert (
        read(run.root / "manifests/filter_report.json")["filtered_reasons"][
            "cross_set_identity_or_family"
        ]
        == 31
    )


def test_transfer_fixed_denominator_and_mock_metrics_do_not_claim_formal_success(
    tmp_path,
):
    run = run_at(tmp_path)
    prepare_transfer(run)
    evaluate_transfer(run, "initial")
    result = transfer_summary(run)
    assert len(result["rows"]) == 4
    assert all(
        row["completed_tasks"] == 50 and row["success_rate"] is None
        for row in result["rows"]
    )
    path = run.root / "transfer/evaluation/live/initial/results.json"
    write(path, read(path)[:-1])
    incomplete = next(
        row for row in transfer_summary(run)["rows"] if row["benchmark"] == "live"
    )
    assert incomplete["denominator"] == 50 and incomplete["status"] == "incomplete"
    assert incomplete["synthetic_demo_rate"] is None


def test_formal_transfer_rejects_synthetic_rows(tmp_path):
    run = run_at(tmp_path)
    prepare_transfer(run)
    evaluate_transfer(run, "initial")
    run.cfg["mode"] = "real"
    with pytest.raises(ValueError, match="Mock transfer"):
        transfer_summary(run)


def test_cleaning_excludes_a_selected_external_task_from_training(
    tmp_path, monkeypatch
):
    run = run_at(tmp_path)
    prepare_transfer(run)
    heldout = read(run.root / "transfer/tasks.json")["verified"][0]
    trajectories, tasks = fixture()
    trajectories.append(
        {
            **trajectories[0],
            "instance_id": heldout["instance_id"],
            "traj_id": "external-leak-fixture",
        }
    )
    monkeypatch.setattr("nice_validation.data.fixture", lambda: (trajectories, tasks))
    prepare(run)
    candidates = read(run.root / "manifests/candidates.json")
    assert heldout["instance_id"] not in {row["instance_id"] for row in candidates}
    assert (
        read(run.root / "manifests/filter_report.json")["filtered_reasons"][
            "cross_set_identity_or_family"
        ]
        == 1
    )


def test_modified_transfer_source_invalidates_the_run_cache(tmp_path):
    cfg = config("configs/development.yaml")
    cfg["mode"], cfg["output"] = "real", str(tmp_path / "run")
    raw = tmp_path / "verified.jsonl"
    raw.write_text('{"instance_id":"one"}\n')
    lock = tmp_path / "transfer_sources.json"
    write(lock, {"verified": {"path": str(raw), "sha256": sha(raw)}})
    cfg["transfer"]["source_lock"] = str(lock)
    Run(cfg)
    raw.write_text('{"instance_id":"changed"}\n')
    with pytest.raises(ValueError, match="identity changed"):
        Run(cfg)


def test_native_preflight_builds_without_model_calls_and_freezes_image_proof(
    tmp_path, monkeypatch
):
    from nice_validation.transfer import prepare_native_environments

    task_dir = tmp_path / "native-task"
    write(task_dir / "fixture.json", {"synthetic": True})
    task = {
        "instance_id": "native-task",
        "task_path": str(task_dir),
        "task_files": task_files(task_dir),
        "benchmark": "terminal",
        "evaluation_backend": "harbor",
    }
    cfg = config("configs/development.yaml")
    cfg["execution"], cfg["mode"], cfg["output"] = (
        "server",
        "real",
        str(tmp_path / "run"),
    )
    cfg["model"]["adapter_path"] = str(tmp_path / "adapter")
    run = Run(cfg)
    monkeypatch.setattr(
        "nice_validation.transfer_sources.transfer_tasks", lambda r: [task]
    )
    calls = []

    def native_process(command, **kwargs):
        calls.append(command)
        job = read(Path(command[-1]))
        assert job["install_only"] is True and job["n_attempts"] == 1
        trial = Path(job["jobs_dir"]) / job["job_name"] / "native-task__fixture"
        write(
            trial / "result.json", {"task_name": "native-task", "exception_info": None}
        )
        write(trial / "agent/environment_images.json", {"main": "sha256:fixture"})
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("nice_validation.transfer.subprocess.run", native_process)
    prepare_native_environments(run)
    assert run.valid("prepare-native-environments")
    assert read(run.root / "transfer/native_images.json") == {
        "terminal:native-task": {"main": "sha256:fixture"}
    }
    prepare_native_environments(run)
    assert len(calls) == 1


def test_incremental_transfer_analysis_archives_and_refreshes_old_summary(
    tmp_path, monkeypatch
):
    run = run_at(tmp_path)
    prepare_transfer(run)
    write(
        run.root / "manifests/subsets.json",
        [
            {
                "subset_id": "low_11",
                "group": "low",
                "sampling_seed": 11,
                "fresh_adam_cosine": {"mean": -0.1},
            }
        ],
    )
    evaluate_transfer(run, "initial")
    monkeypatch.setattr(
        "nice_validation.transfer_analysis.plot_transfer", lambda *args: []
    )
    analyze_transfer(run)
    first = read(run.root / "transfer/summary.json")
    assert (
        next(
            r
            for r in first["rows"]
            if r["benchmark"] == "verified" and r["model_id"] == "low_11"
        )["status"]
        == "incomplete"
    )
    records = [
        {**r, "model_id": "low_11"}
        for r in read(run.root / "transfer/evaluation/verified/initial/results.json")
    ]
    write(run.root / "transfer/evaluation/verified/low_11/results.json", records)
    analyze_transfer(run)
    second = read(run.root / "transfer/summary.json")
    assert (
        next(
            r
            for r in second["rows"]
            if r["benchmark"] == "verified" and r["model_id"] == "low_11"
        )["status"]
        == "completed"
    )
    assert first["input_identity"] != second["input_identity"]
    assert list((run.root / "history").glob("transfer-*/previous_state.json"))


@pytest.mark.parametrize("value", [None, float("nan"), -1, 2, "1"])
def test_native_reward_requires_complete_bounded_numeric_evidence(value):
    with pytest.raises(InfraError):
        reward({"verifier_result": {"rewards": {"reward": value}}})


def test_native_partial_reward_is_retained_and_infra_is_not_a_task_failure():
    assert reward({"verifier_result": {"rewards": {"reward": 0.5}}}) == 0.5
    assert reward({"exception_info": {"exception_type": "AgentTimeoutError"}}) == 0
    with pytest.raises(InfraError, match="native_harness"):
        reward({"exception_info": {"exception_type": "SandboxBuildFailedError"}})


def test_native_job_has_single_attempt_and_separate_pristine_patch_replay(tmp_path):
    patch_path = tmp_path / "model.patch"
    patch_path.write_text("synthetic diff")
    task = {"benchmark": "pro", "task_path": str(tmp_path / "task")}
    proof = {"main": "sha256:fixture"}
    generation = native_job(
        task,
        tmp_path,
        tmp_path / "cfg.local.yaml",
        tmp_path / "adapter",
        expected_images=proof,
    )
    replay = native_job(
        task,
        tmp_path,
        tmp_path / "cfg.local.yaml",
        tmp_path / "adapter",
        patch_path,
        proof,
    )
    assert generation["n_attempts"] == 1 and generation["retry"]["max_retries"] == 0
    assert generation["agents"][0]["kwargs"]["capture_patch"] is True
    assert replay["agents"][0]["kwargs"]["patch_sha256"] == sha(patch_path)
    assert replay["agents"][0]["import_path"].endswith(":NicePatchReplay")
    assert "config_file" not in replay["agents"][0]["kwargs"]


def test_native_task_changes_are_rejected(tmp_path):
    (tmp_path / "instruction.md").write_text("public task")
    (tmp_path / "task.toml").write_text("version = '1.0'")
    task = {"task_path": str(tmp_path), "task_files": task_files(tmp_path)}
    validate_native_task(task)
    (tmp_path / "instruction.md").write_text("changed task")
    with pytest.raises(ValueError, match="content changed"):
        validate_native_task(task)


@pytest.mark.parametrize(
    "backend,module",
    [
        ("swesmith", "swesmith.harness.eval"),
        ("swebench", "swebench.harness.run_evaluation"),
        ("live", "swebench.harness.run_evaluation"),
    ],
)
def test_official_grader_dispatch_keeps_benchmarks_separate(
    tmp_path, monkeypatch, backend, module
):
    cfg = config("configs/main.yaml")
    cfg["agent"]["image_map"] = str(tmp_path / "images.json")
    cfg["transfer"]["live_python"] = "/isolated/live/python"
    write(
        Path(cfg["agent"]["image_map"]),
        {
            "task": {
                "image": "sha256:x",
                "official_image_name": "official/image:tag",
                "checkout_commit": "a" * 40,
            }
        },
    )
    task = {"instance_id": "task", "evaluation_backend": backend}
    commands = []

    def process(command, **kwargs):
        commands.append(command)
        if command[0] == "docker":
            return types.SimpleNamespace(stdout="sha256:one\n", returncode=0)
        run_id = command[command.index("--run_id") + 1]
        write(
            tmp_path / "grading/logs/run_evaluation" / run_id / "task/report.json",
            {"task": {"resolved": True}},
        )
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("nice_validation.environments.official_identity", lambda c: {})
    monkeypatch.setattr("nice_validation.harness.subprocess.run", process)
    assert official_grade(cfg, task, "synthetic patch", tmp_path / "grading") is True
    selected = next(command for command in commands if command[0] != "docker")
    if backend == "swesmith":
        assert selected[1].endswith("swesmith_grade.py")
        assert selected[selected.index("--commit") + 1] == "a" * 40
    else:
        assert selected[2] == module
    assert (selected[0] == "/isolated/live/python") == (backend == "live")
    if backend == "live":
        assert "--namespace" in selected
        assert selected[selected.index("--namespace") + 1] == "starryzhang"
    elif backend == "swebench":
        assert selected[selected.index("--namespace") + 1] == "swebench"


def harbor_module(monkeypatch):
    fake = types.ModuleType("harbor.agents.base")
    fake.BaseAgent = object
    monkeypatch.setitem(sys.modules, "harbor.agents.base", fake)
    sys.modules.pop("nice_validation.harbor_agent", None)
    return importlib.import_module("nice_validation.harbor_agent")


def test_native_agent_tool_round_trip_uses_local_adapter_and_public_instruction(
    tmp_path, monkeypatch
):
    module = harbor_module(monkeypatch)
    cfg = config("configs/main.yaml")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "fixture.txt").write_text("synthetic adapter")
    observed, commands = [], []
    calls = [
        {"id": "a", "function": {"name": "bash", "arguments": {"command": "pwd"}}},
        {"id": "b", "function": {"name": "submit", "arguments": {}}},
    ]

    def generation(url, payload, timeout, deadline):
        observed.append(payload)
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [calls[len(observed) - 1]],
            },
            "parse_error": None,
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

    class Environment:
        async def exec(self, **kwargs):
            commands.append(kwargs["command"])
            return types.SimpleNamespace(stdout="/app", stderr="", return_code=0)

    monkeypatch.setattr(module, "request_generation", generation)
    context = types.SimpleNamespace()
    asyncio.run(
        module.service_loop(
            Environment(),
            cfg,
            adapter,
            tmp_path / "logs",
            "PUBLIC INSTRUCTION",
            context,
        )
    )
    assert (
        len(commands) == 1
        and "pwd" in commands[0]
        and commands[0].startswith("timeout --signal=KILL")
        and len(observed) == 2
    )
    assert observed[0]["messages"][1]["content"] == "PUBLIC INSTRUCTION"
    assert observed[1]["messages"][-2]["role"] == "tool"
    assert observed[0]["adapter_path"] == str(adapter.resolve())
    assert (
        context.n_output_tokens == 4 and context.metadata["stop_reason"] == "SUBMITTED"
    )
    assert read(tmp_path / "logs/episode.json")["task_failure"] is False


def test_native_setup_rejects_a_changed_container_image(tmp_path, monkeypatch):
    module = harbor_module(monkeypatch)
    agent = object.__new__(module.NiceAgent)
    agent.expected_images = {"main": "sha256:frozen"}
    agent.logs_dir = tmp_path
    monkeypatch.setattr(
        module, "environment_images", lambda environment: {"main": "sha256:changed"}
    )
    with pytest.raises(InfraError, match="image_changed"):
        asyncio.run(agent.setup(object()))


def test_source_resolver_freezes_live_and_terminal_version(monkeypatch):
    hub = types.ModuleType("huggingface_hub")

    class API:
        def model_info(self, identifier):
            return types.SimpleNamespace(sha="a" * 40)

        dataset_info = model_info

    hub.HfApi = API
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    queried = []

    def remote(request, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        queried.append(url)
        value = (
            [
                {
                    "name": "terminal-bench",
                    "version": "2.0",
                    "tasks": [
                        {
                            "git_commit_id": "c" * 40,
                            "name": "registered-fixture",
                            "path": "tasks/registered-fixture",
                            "git_url": "https://github.com/laude-institute/terminal-bench-2.git",
                        }
                    ],
                }
            ]
            if url.endswith("registry.json")
            else {"sha": "b" * 40}
        )
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr("urllib.request.urlopen", remote)
    sources = resolve_sources(config("configs/main.yaml"))
    assert sources["terminal"]["revision"] == "c" * 40
    assert any("python-only" in url for url in queried)
    assert any("/" + "b" * 40 + "/registry.json" in url for url in queried)


def test_environment_replacement_cannot_move_a_training_task_into_validation(tmp_path):
    from nice_validation.environments import replace_validation

    run = run_at(tmp_path)
    run.cfg["mode"] = "real"
    folder = run.root / "manifests"
    old = {
        "instance_id": "failed",
        "repo": "org/repo",
        "problem_statement": "old issue",
    }
    trained = {
        "instance_id": "trained",
        "repo": "org/repo",
        "problem_statement": "training issue",
    }
    reserve = {
        "instance_id": "heldout",
        "repo": "org/repo",
        "problem_statement": "different heldout issue",
    }
    with run.stage("prepare-data") as artifacts:
        for name, value in [
            ("validation", [old]),
            ("validation_reserve", [trained, reserve]),
            ("checksums", {}),
            (
                "candidates",
                [
                    {
                        **trained,
                        "messages": [{"role": "user", "content": "training issue"}],
                    }
                ],
            ),
        ]:
            path = folder / (name + ".json")
            write(path, value)
            artifacts.append(path)
    write(
        folder / "environment_preflight.json",
        {"failures": [{"task_id": "failed", "reason": "fixture environment error"}]},
    )
    assert replace_validation(run, "failed")["replacement"]["instance_id"] == "heldout"
