"""Regression proofs for the final upstream compatibility and data-flow review."""

import copy
import io
import json
import sys
import types

import pytest

from nice_validation.core import Run, config, read
from nice_validation.compatibility import harness_api, SWEBENCH_REF
from nice_validation.environments import verified_image
from nice_validation.data import fixture, prepare
from nice_validation.transfer_sources import task_identities


def modules(monkeypatch, names):
    result = {}
    for name in names:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            key = ".".join(parts[:i])
            if key not in result:
                result[key] = types.ModuleType(key)
                result[key].__path__ = []
                monkeypatch.setitem(sys.modules, key, result[key])
                if i > 1:
                    setattr(result[".".join(parts[: i - 1])], parts[i - 1], result[key])
    return result


@pytest.mark.parametrize("broken", [None, "constant", "signature"])
def test_harness_api_checks_actual_required_names_and_signatures(monkeypatch, broken):
    names = [
        "swebench.harness.constants",
        "swebench.harness.docker_build",
        "swebench.harness.test_spec.test_spec",
        "swesmith.harness.utils",
        "swesmith.harness.grading",
    ]
    installed = modules(monkeypatch, names)
    constants = installed[names[0]]
    for name in [
        "DOCKER_PATCH",
        "DOCKER_USER",
        "DOCKER_WORKDIR",
        "KEY_INSTANCE_ID",
        "KEY_PREDICTION",
        "UTF8",
        "LOG_REPORT",
        "LOG_TEST_OUTPUT",
        "LOG_INSTANCE",
        "RUN_EVALUATION_LOG_DIR",
    ]:
        setattr(constants, name, name)
    installed[names[1]].close_logger = lambda logger: None
    installed[names[2]].make_test_spec = lambda task, namespace=None: None
    installed[names[3]].run_patch_in_container = (
        lambda task, patch=None, commit=None: None
    )
    installed[names[4]].get_eval_report = (
        lambda pred, task, output, f2p_only=False: None
    )
    if broken == "constant":
        del constants.DOCKER_PATCH
    elif broken == "signature":
        installed[names[3]].run_patch_in_container = lambda task, patch=None: None
    cfg = config("configs/legacy.yaml")
    cfg["transfer"]["enabled"] = False
    if broken:
        with pytest.raises(ValueError, match="mismatch"):
            harness_api(cfg)
    else:
        assert harness_api(cfg) == {"swebench_ref": "v4.1.0", "swesmith_api": "PASS"}


def test_verified_preparation_uses_the_official_remote_image_namespace(monkeypatch):
    installed = modules(monkeypatch, ["swebench.harness.test_spec.test_spec"])

    def make(task, namespace=None):
        assert namespace == "swebench"
        return types.SimpleNamespace(
            instance_image_key=namespace + "/sweb.eval.x86_64.org_1776_repo-1:latest"
        )

    installed["swebench.harness.test_spec.test_spec"].make_test_spec = make
    assert verified_image({"instance_id": "org__repo-1"}).startswith("swebench/")


def test_source_resolver_freezes_compatible_release_not_mutable_main(monkeypatch):
    from nice_validation.online import resolve_sources

    installed = modules(monkeypatch, ["huggingface_hub"])

    class API:
        def model_info(self, identifier):
            return types.SimpleNamespace(sha="a" * 40)

        dataset_info = model_info

    installed["huggingface_hub"].HfApi = API
    queried = []

    def remote(request, **kwargs):
        queried.append(request.full_url)
        return io.BytesIO(json.dumps({"sha": "b" * 40}).encode())

    monkeypatch.setattr("urllib.request.urlopen", remote)
    cfg = config("configs/legacy.yaml")
    cfg["transfer"]["enabled"] = False
    assert resolve_sources(cfg)["swebench"]["revision"] == "b" * 40
    assert queried[0].endswith("/commits/" + SWEBENCH_REF)


@pytest.mark.parametrize("source", ["external-original", "external-variant"])
def test_training_cannot_leak_an_external_task_through_one_sided_family_metadata(
    tmp_path, monkeypatch, source
):
    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path / "run")
    training, validation = fixture()
    # The held-out family can be declared on either side, not only on both.
    training[0]["source_instance_id"] = source
    external = {
        "instance_id": "external-original",
        "source_instance_id": "external-variant",
        "problem_statement": "An unrelated public external task",
    }
    monkeypatch.setattr("nice_validation.data.fixture", lambda: (training, validation))
    monkeypatch.setattr(
        "nice_validation.transfer_sources.transfer_tasks", lambda r: [external]
    )
    run = Run(cfg)
    prepare(run)
    assert (
        read(run.root / "manifests/filter_report.json")["filtered_reasons"][
            "cross_set_identity_or_family"
        ]
        == 1
    )
    assert training[0]["instance_id"] not in {
        r["instance_id"] for r in read(run.root / "manifests/candidates.json")
    }
    assert task_identities(external) == {"external-original", "external-variant"}


def test_main_multiround_loop_returns_each_tool_observation_before_next_generation(
    tmp_path, monkeypatch
):
    from nice_validation.harness import real_episode

    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path / "run")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "fixture.txt").write_text("synthetic adapter, no weights")
    cfg["model"]["adapter_path"] = str(adapter)
    run = Run(cfg)
    observations, actions = [], []
    calls = [
        [
            {
                "id": "view",
                "function": {"name": "bash", "arguments": {"command": "pwd"}},
            },
            {
                "id": "test",
                "function": {"name": "bash", "arguments": {"command": "run-tests"}},
            },
        ],
        [
            {
                "id": "fix",
                "function": {"name": "bash", "arguments": {"command": "apply-fix"}},
            }
        ],
        [{"id": "submit", "function": {"name": "submit", "arguments": {}}}],
    ]

    def generation(url, payload, *args, **kwargs):
        observations.append(copy.deepcopy(payload["messages"]))
        i = len(observations) - 1
        return {
            "message": {"role": "assistant", "content": "", "tool_calls": calls[i]},
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "input_ids": [1],
            "output_ids": [2],
            "parse_error": None,
        }

    def invoke(args):
        actions.append(args["command"])
        return {
            "pwd": "/testbed",
            "run-tests": "FAIL: assertion\n[exit_code=1]",
            "apply-fix": "file updated",
        }[args["command"]]

    class Environment:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def tools(self):
            return {"bash": types.SimpleNamespace(invoke=invoke)}

        def patch(self):
            return "synthetic patch"

    monkeypatch.setattr("nice_validation.harness.DockerTask", Environment)
    monkeypatch.setattr("nice_validation.harness.request_generation", generation)
    monkeypatch.setattr("nice_validation.harness.official_grade", lambda *a: True)
    folder = tmp_path / "episode"
    result = real_episode(
        run,
        {"instance_id": "fixture", "problem_statement": "PUBLIC ISSUE"},
        "initial",
        False,
        2026,
        folder,
    )
    assert actions == ["pwd", "run-tests", "apply-fix"]
    assert [m["tool_call_id"] for m in observations[1] if m["role"] == "tool"] == [
        "view",
        "test",
    ]
    assert "assertion" in observations[1][-1]["content"]
    assert observations[2][-1]["content"] == "file updated"
    assert result["stop_reason"] == "SUBMITTED" and result["tool_calls"] == 3
    assert len(read(folder / "trace.json")) == 3


@pytest.mark.parametrize(
    "episode",
    [
        None,
        {"task_failure": "false", "stop_reason": "SUBMITTED"},
        {"task_failure": False, "stop_reason": "SUBMITTED"},
    ],
)
def test_native_reward_cannot_be_accepted_without_agent_episode_evidence(
    tmp_path, monkeypatch, episode
):
    from nice_validation.transfer import native_episode
    from nice_validation.harness import InfraError
    from nice_validation.core import write

    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path / "run")
    cfg["execution"] = "server"
    cfg["model"]["adapter_path"] = str(tmp_path / "adapter")
    run = Run(cfg)
    cfg["mode"] = "real"
    task = {
        "instance_id": "fixture",
        "benchmark": "terminal",
        "task_path": str(tmp_path),
    }
    write(
        run.root / "transfer/native_images.json",
        {"terminal:fixture": {"main": "sha256:fixture"}},
    )
    result_path = tmp_path / "trial/result.json"
    if episode is not None:
        write(result_path.parent / "agent/episode.json", episode)
    monkeypatch.setattr(
        "nice_validation.transfer.validate_native_task", lambda task: None
    )
    monkeypatch.setattr(
        "nice_validation.transfer.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        "nice_validation.transfer.one_native_result",
        lambda *a: (
            {"task_name": "fixture", "verifier_result": {"rewards": {"reward": 1}}},
            result_path,
        ),
    )
    if episode is None or not isinstance(episode["task_failure"], bool):
        with pytest.raises(InfraError, match="episode_evidence"):
            native_episode(run, task, "initial", tmp_path / "attempt")
    else:
        assert (
            native_episode(run, task, "initial", tmp_path / "attempt")["resolved"]
            is True
        )
