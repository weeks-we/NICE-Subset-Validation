"""CPU process doubles verify GPU roles, publication and failure cleanup."""
import copy
from types import SimpleNamespace
from pathlib import Path
import pytest

from nice_validation.core import read, write
from nice_validation.experiment_parallel import run_training_and_selection, worker_lease, assert_no_workers
from nice_validation.parallel_training import WorkerRun
from nice_validation.experiment_training import train_job
from test_experiment import ready, small_cfg


@pytest.mark.parametrize("fail", [False, True])
def test_hybrid_assigns_four_training_gpus_activates_released_cards_and_reaps(tmp_path, monkeypatch, fail):
    import nice_validation.experiment_parallel as module
    run = ready(tmp_path)
    run.cfg = copy.deepcopy(run.cfg)
    run.cfg["mode"] = "real"  # Only the coordinator is under test; workers explicitly use synthetic data.
    launched, evaluated, activated = [], [], []

    class Process:
        def __init__(self, argv, **kwargs):
            self.returncode = None
            sid = argv[argv.index("--subset") + 1]
            gpu = argv[argv.index("--gpu-uuid") + 1]
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == gpu
            self.sid, self.gpu = sid, gpu
            launched.append(self)
            if fail and len(launched) == 2:
                self.returncode = 1
                return
            log = Path(argv[argv.index("--log") + 1])
            view = WorkerRun(run, sid, gpu, log)
            view.cfg["mode"] = "mock"
            train_job(view, sid)
            self.returncode = 0

        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode
        def terminate(self): self.returncode = -15
        def kill(self): self.returncode = -9

    def evaluate(parent, split, model_id, endpoint_pool):
        assert split == "selection"
        assert endpoint_pool.capacity == 16
        evaluated.append(model_id)

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module, "evaluate", evaluate)
    if fail:
        with pytest.raises(RuntimeError, match="failed"):
            run_training_and_selection(run, lambda: activated.append(True))
        assert not activated
    else:
        run_training_and_selection(run, lambda: activated.append(True))
        assert activated == [True]
        assert len(launched) == 4 and {p.gpu for p in launched} == set(run.cfg["resources"]["gpu_uuids"][:4])
        assert len(evaluated) == len(set(evaluated)) == 12
        assert all(run.valid("train:" + p.sid) for p in launched)
    assert all(p.poll() is not None for p in launched)


def test_active_training_lease_cannot_be_stolen(tmp_path):
    run = ready(tmp_path)
    gpu = run.cfg["resources"]["gpu_uuids"][0]
    with worker_lease(run, gpu):
        with pytest.raises(ValueError, match="still"):
            assert_no_workers(run)
        with pytest.raises(ValueError, match="still"):
            with worker_lease(run, gpu):
                pass
    assert_no_workers(run)


def test_server_owns_all_service_transitions_and_delivery(tmp_path, monkeypatch):
    import nice_validation.experiment_server as server
    import nice_validation.experiment_parallel as parallel
    from nice_validation.core import Run
    from nice_validation.data import prepare
    from nice_validation.smoke import smoke
    from nice_validation.harness import episodes
    from nice_validation.scoring import score
    from nice_validation.subsets import make_subsets
    from nice_validation.batch_tuning import tune
    from nice_validation.training import train
    from nice_validation.experiment import evaluate

    storage = tmp_path / "storage"
    cfg = small_cfg(storage / "runs/main")
    gpus = cfg["resources"]["gpu_uuids"]
    filename = tmp_path / "cfg/server.json"
    write(filename, cfg)
    files = []
    for index, gpu in enumerate(gpus):
        service = copy.deepcopy(cfg)
        service["model"]["gpu_uuids"] = [gpu]
        file = tmp_path / "cfg" / f"service-{index}.json"
        write(file, service)
        files.append(str(file))
    monkeypatch.setattr(server.platform, "system", lambda: "Linux")
    monkeypatch.setattr(server.sys, "version_info", (3, 12))
    monkeypatch.setattr(server, "configure", lambda *a, **k: {"server_config": str(filename), "service_configs": files})
    active, starts, stopped = set(), [], []

    def start(root, service, file, port):
        gpu = service["model"]["gpu_uuids"][0]
        assert gpu not in active
        active.add(gpu)
        starts.append(gpu)
        return gpu, None

    def stop(handles):
        for gpu, log in handles:
            active.remove(gpu)
            stopped.append(gpu)

    def command(root, name, argv, env):
        run = Run(cfg)
        if name == "prepare": prepare(run)
        elif name == "smoke":
            assert len(active) == 8
            smoke(run)
        elif name == "nice-rollouts":
            assert len(active) == 8
            episodes(run, "initial", True)
        elif name == "parallel-score":
            assert not active
            score(run)
        elif name == "subsets": make_subsets(run)
        elif name == "training-batch-probe":
            assert not active
            tune(run)
        elif name == "bundle":
            assert not active
            Path(argv[argv.index("--output") + 1]).write_bytes(b"unit-bundle")

    def hybrid(run, activate_released):
        assert active == set(gpus[4:])
        for job in read(run.root / "manifests/training_jobs.json"):
            train(run, job["subset_id"])
        activate_released()
        assert active == set(gpus)
        for point in read(run.root / "manifests/checkpoints.json"):
            evaluate(run, "selection", point["model_id"])

    monkeypatch.setattr(server, "start_service", start)
    monkeypatch.setattr(server, "stop_services", stop)
    monkeypatch.setattr(server, "command", command)
    monkeypatch.setattr(parallel, "run_training_and_selection", hybrid)
    result = server.run_server(storage, gpus, tmp_path / "cfg")
    assert result["status"] == "completed" and len(result["selected_models"]) == 4
    assert not active and len(starts) == len(stopped) == 16
    assert read(storage / "controller/completion.json")["status"] == "completed"
