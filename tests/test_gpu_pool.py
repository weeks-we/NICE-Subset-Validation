"""CPU proofs of configurable GPU selection, service lifecycle and task routing."""

import copy
import threading
from contextlib import contextmanager
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from nice_validation.core import Run, config, read, sha, write
from nice_validation.gpus import select_gpus
from nice_validation.online import configure_server
from nice_validation import supervisor as sup
from test_online import records


GPUS = [f"GPU-{i:08x}-aaaa-aaaa-aaaa-aaaaaaaaaaaa" for i in range(12)]


def test_preflight_checks_every_selected_gpu_before_preparation(monkeypatch):
    from nice_validation.preflight import preflight

    cfg = config("configs/legacy.yaml")
    cfg["model"]["gpu_uuids"] = GPUS[:1]
    cfg["resources"] = {"gpu_count": 8, "gpu_uuids": GPUS[:8]}
    observed = {
        "gpu": {"returncode": 0, "stdout": ",".join(GPUS[:4])},
        "docker": {"returncode": 0},
    }
    monkeypatch.setattr("nice_validation.preflight.doctor", lambda c: observed)
    assert (
        next(
            c for c in preflight(cfg)["checks"] if c["check"] == "authorized_gpu_pool"
        )["status"]
        == "FAIL"
    )
    observed["gpu"]["stdout"] = ",".join(GPUS[:8])
    assert (
        next(
            c for c in preflight(cfg)["checks"] if c["check"] == "authorized_gpu_pool"
        )["status"]
        == "PASS"
    )


@pytest.mark.parametrize("count", [2, 4, 8])
def test_requested_count_controls_all_service_configs_and_ports(tmp_path, count):
    with patch(
        "nice_validation.online.resolve_sources", side_effect=lambda c: records()
    ):
        paths = configure_server(
            tmp_path / "storage",
            GPUS[0],
            GPUS[1],
            tmp_path / "configs",
            service_gpus=GPUS[1:count],
        )
    cfg = config(paths["server_config"])
    assert cfg["resources"]["gpu_uuids"] == GPUS[:count]
    assert cfg["agent"]["workers"] == count - 1
    assert len(paths["service_configs"]) == count - 1
    for i, name in enumerate(paths["service_configs"]):
        service = config(name)
        assert service["model"]["gpu_uuids"] == [GPUS[i + 1]]
        assert service["agent"]["service_endpoints"] == [
            f"http://127.0.0.1:{8765 + j}" for j in range(count - 1)
        ]
    with pytest.raises(ValueError, match="overwrite"):
        configure_server(
            tmp_path / "storage",
            GPUS[0],
            GPUS[1],
            tmp_path / "configs",
            service_gpus=GPUS[1 : count + 1],
        )


def test_selection_defaults_to_eight_and_honors_visible_device_order(monkeypatch):
    result = subprocess.CompletedProcess(
        [], 0, "\n".join(f"{i}, {gpu}" for i, gpu in enumerate(GPUS))
    )
    monkeypatch.setattr("nice_validation.gpus.subprocess.run", lambda *a, **k: result)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert select_gpus() == GPUS[:8]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,3,2,1")
    assert select_gpus(4) == [GPUS[i] for i in [5, 3, 2, 1]]
    with pytest.raises(ValueError, match="only 4"):
        select_gpus()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(GPUS[3:7]))
    assert select_gpus(2) == GPUS[3:5]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(ValueError, match="CUDA_VISIBLE"):
        select_gpus(2)


@pytest.mark.parametrize(
    "pool,count",
    [(GPUS, 1), (GPUS[:4], 8), ([GPUS[0]] * 8, 8), (["GPU-bad", GPUS[1]], 2)],
)
def test_invalid_insufficient_or_duplicate_gpu_pool_is_refused(pool, count):
    with pytest.raises(ValueError):
        select_gpus(count, pool)


@pytest.mark.parametrize("failed_index", [None, 2])
def test_controller_owns_all_seven_services_and_cleans_partial_start(
    tmp_path, monkeypatch, failed_index
):
    with patch(
        "nice_validation.online.resolve_sources", side_effect=lambda c: records()
    ):
        paths = configure_server(
            tmp_path / "storage",
            GPUS[0],
            GPUS[1],
            tmp_path / "configs",
            service_gpus=GPUS[1:8],
        )
    cfg = config(paths["server_config"])
    monkeypatch.setattr(
        "nice_validation.online.configure_server", lambda *a, **k: paths
    )
    monkeypatch.setattr(sup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sup.sys, "version_info", (3, 12))
    starts, stops, commands = [], [], []

    def start(root, service, name, port):
        i = port - 8765
        assert service["model"]["gpu_uuids"] == [GPUS[i + 1]]
        if i == failed_index:
            raise RuntimeError("injected service startup failure")
        starts.append(port)
        return port, None

    def command(root, name, argv, env):
        assert env["CUDA_VISIBLE_DEVICES"] == GPUS[0]
        commands.append(name)
        if name == "audit":
            write(Path(cfg["output"]) / "audit.json", {"status": "PASS"})

    monkeypatch.setattr(sup, "start_service", start)
    monkeypatch.setattr(sup, "stop_service", lambda p, log: stops.append(p))
    monkeypatch.setattr(sup, "command", command)
    args = (tmp_path / "storage", GPUS[0], GPUS[1], tmp_path / "configs")
    if failed_index is not None:
        with pytest.raises(RuntimeError, match="injected"):
            sup.run_server(*args, service_gpus=GPUS[1:8])
        assert "experiment" not in commands
        assert (
            read(tmp_path / "storage/controller/completion.json")["status"] == "failed"
        )
    else:
        result = sup.run_server(*args, service_gpus=GPUS[1:8])
        assert result["inference_services"] == 7 and result["gpu_count"] == 8
    assert stops == list(reversed(starts))


def test_cleanup_attempts_every_service_even_after_one_stop_fails(monkeypatch):
    calls = []

    def stop(process, log):
        calls.append(process)
        if process == 2:
            raise OSError("cannot stop")

    monkeypatch.setattr(sup, "stop_service", stop)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        sup.stop_services([(1, None), (2, None), (3, None)])
    assert calls == [3, 2, 1]


def test_transfer_routes_each_frozen_task_once_with_real_parallel_workers(
    tmp_path, monkeypatch
):
    from nice_validation.transfer import evaluate_transfer
    from nice_validation.transfer_sources import prepare_transfer

    cfg = config("configs/development.yaml")
    cfg["output"] = str(tmp_path / "run")
    cfg["agent"]["workers"] = 3
    cfg["agent"]["service_endpoints"] = [
        f"http://127.0.0.1:{9000 + i}" for i in range(3)
    ]
    run = Run(cfg)
    prepare_transfer(run)
    selections = read(run.root / "transfer/tasks.json")
    run.cfg["mode"] = "real"
    monkeypatch.setattr(run, "valid", lambda name: True)
    image_lock = tmp_path / "images.json"
    write(image_lock, {})
    run.cfg["agent"]["image_revision_lock"] = str(image_lock)
    write(run.root / "manifests/validation.json", [])
    write(
        run.root / "manifests/validation_freeze.json",
        {
            "environment_lock_sha256": sha(image_lock),
            "validation_sha256": sha(run.root / "manifests/validation.json"),
        },
    )
    monkeypatch.setattr("nice_validation.smoke.require_real_smoke", lambda r: None)

    @contextmanager
    def stage(name):
        yield []

    monkeypatch.setattr(run, "stage", stage)
    calls, lock, barrier = [], threading.Lock(), threading.Barrier(3)

    def job(scoped, stage, model, nice, i, task, repeat, folder):
        if i < 3:
            barrier.wait(timeout=10)
        endpoint = scoped.cfg["agent"]["service_url"]
        assert endpoint == cfg["agent"]["service_endpoints"][i % 3]
        assert scoped.cfg["_evaluation_scope"] == stage
        with lock:
            calls.append((task["benchmark"], task["instance_id"], i, endpoint))
        return {
            "task_id": task["instance_id"],
            "status": "TASK_FAILED",
            "resolved": False,
        }

    monkeypatch.setattr("nice_validation.harness._episode_job", job)
    original_agent = copy.deepcopy(run.cfg["agent"])
    evaluate_transfer(run, "initial")
    assert len(calls) == len(set((suite, task) for suite, task, _, _ in calls)) == 200
    assert run.cfg["agent"] == original_agent
    for suite, selected in selections.items():
        results = read(run.root / f"transfer/evaluation/{suite}/initial/results.json")
        assert [r["task_id"] for r in results] == sorted(
            t["instance_id"] for t in selected
        )
