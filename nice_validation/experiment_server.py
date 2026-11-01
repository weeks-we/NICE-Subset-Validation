"""Own all server phases, old-input reuse, hybrid SFT/eval and final delivery."""
from pathlib import Path
import platform
import sys
import threading

from nice_validation.core import Run, config, read, write, source_identity, environment_identity, utc
from nice_validation.supervisor import (controller_lock, command, start_service, stop_services,
                                        child_environment)
from nice_validation.experiment_online import configure


def run_server(storage, gpus, destination, model_path=None, reuse_from=None,
               port=8765, recover=False, base_config=None):
    if platform.system() != "Linux" or sys.version_info < (3, 12):
        raise ValueError("Full execution requires the authorized Linux server and Python 3.12+")
    storage = Path(storage).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "controller").mkdir(exist_ok=True)
    services, service_guard = {}, threading.Lock()
    with controller_lock(storage, recover):
        source_path = storage / "controller/source.json"
        if source_path.is_file() and read(source_path) != source_identity():
            raise ValueError("Source changed; preserve old evidence and use new storage")
        write(source_path, source_identity())
        try:
            paths = configure(storage, gpus, destination, reuse_from, model_path, port, base_config)
            cfg = config(paths["server_config"])
            env = child_environment(gpus[0])
            executable = [sys.executable, "-m", "nice_validation", "--config", paths["server_config"]]
            frozen = storage / "runs/main/config/snapshot.json"
            if frozen.exists():
                if read(frozen)["external_inputs"]["environment"] != environment_identity("real"):
                    raise ValueError("Frozen environment changed; use a new environment/storage pair")
            else:
                command(storage, "install-harness", [sys.executable, "scripts/install_harness.py",
                         "--config", paths["server_config"]], env)
            command(storage, "preflight", executable + ["preflight"], env)
            if recover and frozen.is_file():
                from nice_validation.cli import recover_lock
                recover_lock(Run(cfg))
            command(storage, "prepare", executable + ["prepare-online", "--download-model"], env)

            def start(indices):
                with service_guard:
                    for index in indices:
                        if index in services:
                            continue
                        file = Path(paths["service_configs"][index])
                        handle = start_service(storage, config(file), file, port + index)
                        if handle[0] is None:
                            raise ValueError("Inference services must be owned by this controller")
                        services[index] = handle

            def stop_all():
                with service_guard:
                    stop_services(list(services.values()))
                    services.clear()

            start(range(len(gpus)))
            command(storage, "preflight-prepared", executable + ["preflight", "--prepared"], env)
            command(storage, "smoke", executable + ["smoke"], env)
            command(storage, "nice-rollouts", executable + ["rollout-nice"], env)
            run = Run(cfg)
            from nice_validation.experiment import evaluate, select_models
            evaluate(run, "selection", "initial")
            stop_all()
            command(storage, "parallel-score", executable + ["score"], env)
            command(storage, "subsets", executable + ["make-subsets"], env)
            from nice_validation.experiment_training import ensure_jobs
            ensure_jobs(run)
            command(storage, "training-batch-probe", executable + ["tune-training-batch"], env)
            width = min(4, max(1, len(gpus) // 2))
            start(range(width, len(gpus)))
            from nice_validation.experiment_parallel import run_training_and_selection
            run_training_and_selection(run, activate_released=lambda: start(range(width)))
            select_models(run)
            start(range(len(gpus)))
            for model_id in ["initial", *[p["model_id"] for p in read(run.root / "selection.json")["selected"]]]:
                evaluate(run, "benchmark", model_id)
                evaluate(run, "nice_retest", model_id)
            stop_all()
            from nice_validation.experiment_report import report
            report(run)
            audit = read(run.root / "audit.json")
            if audit["status"] == "FAIL":
                raise ValueError("Final audit failed")
            bundle = storage / "controller" / ("results-" + utc().replace(":", "-") + ".zip")
            command(storage, "bundle", [sys.executable, "scripts/bundle_results.py", "--run", str(run.root),
                                        "--output", str(bundle)], env)
            result = {"status": "completed", "utc": utc(), "run": str(run.root),
                      "bundle": str(bundle), "gpu_count": len(gpus), "audit": audit["status"],
                      "selected_models": [p["model_id"] for p in read(run.root / "selection.json")["selected"]]}
            write(storage / "controller/completion.json", result)
            return result
        except BaseException as exc:
            write(storage / "controller/completion.json", {"status": "failed", "utc": utc(),
                  "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            with service_guard:
                stop_services(list(services.values()))
