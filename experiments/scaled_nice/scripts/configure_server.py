"""Generate pinned server/service configs online; never downloads weights."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.online import configure_server
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
    args = parser.parse_args()
    gpus = gpu_arguments(args)
    result = configure_server(
        args.storage_root,
        gpus[0],
        gpus[1],
        args.config_dir,
        args.model_path,
        args.port,
        service_gpus=gpus[1:],
        base_config=args.base_config,
        model_id=args.model_id,
        model_loader=args.model_loader,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    commit = result["sources"]["swebench"]["revision"]
    print(
        f"Install the pinned harness before preparing: python -m pip install 'git+https://github.com/SWE-bench/SWE-bench.git@{commit}'"
    )


if __name__ == "__main__":
    main()
