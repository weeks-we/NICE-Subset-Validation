"""One server entry: install dependencies, prepare inputs, own service, run and bundle."""

import argparse
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.gpus import add_gpu_arguments, gpu_arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", required=True, type=Path)
    add_gpu_arguments(parser)
    parser.add_argument("--config-dir", default="configs", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-loader")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--recover-lock", action="store_true")
    parser.add_argument(
        "--execute-server",
        action="store_true",
        help="Explicitly authorize server downloads, model/GPU execution and local dependency installation",
    )
    args = parser.parse_args()
    if args.num_gpus < 2:
        parser.error(
            "--num-gpus must be at least 2: one training and one inference GPU"
        )
    if not args.execute_server:
        phased = args.base_config and args.base_config.name == "scaled.yaml"
        print(
            f"Plan: {2 if args.train_gpu or args.service_gpu else args.num_gpus} GPUs "
            + (
                "(phased: all independent training / all inference); "
                if phased
                else "(one training, remaining inference); "
            )
            + "install dependencies -> freeze source/config -> install harnesses -> online prepare -> own services -> smoke and full experiment -> stop owned services -> analyze/audit/bundle. Add --execute-server on the authorized Linux server."
        )
        return
    if platform.system() != "Linux" or sys.version_info < (3, 12):
        raise SystemExit(
            "Full execution requires the authorized Linux server with Python 3.12+"
        )
    gpus = gpu_arguments(args)
    if not 1 <= args.port <= 65535 or args.port + len(gpus) - 1 > 65535:
        raise SystemExit("Invalid service port")
    repo = Path(__file__).resolve().parents[1]
    args.config_dir = (repo / args.config_dir).resolve()
    # Never reinstall before resuming a frozen environment.
    if not (args.storage_root / "runs/main/config/snapshot.json").exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(repo) + "[server,dev]"],
            cwd=repo,
            check=True,
        )
    sys.path.insert(0, str(repo))
    from nice_validation.supervisor import run_server

    print(
        run_server(
            args.storage_root,
            gpus[0],
            gpus[1],
            args.config_dir,
            args.port,
            args.model_path,
            args.recover_lock,
            service_gpus=gpus[1:],
            base_config=args.base_config,
            model_id=args.model_id,
            model_loader=args.model_loader,
        )
    )


if __name__ == "__main__":
    main()
