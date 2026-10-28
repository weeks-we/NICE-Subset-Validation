"""Select an explicit, bounded GPU pool before importing model dependencies."""

import os
import re
import subprocess


def validate_pool(uuids: list[str]) -> list[str]:
    if (
        len(uuids) < 2
        or len(set(uuids)) != len(uuids)
        or any(
            not re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value
            )
            for value in uuids
        )
    ):
        raise ValueError("Supply at least two different authorized full GPU UUIDs")
    return uuids


def select_gpus(count: int = 8, allowed: list[str] | None = None) -> list[str]:
    if count < 2:
        raise ValueError(
            "At least 2 GPUs are required: one gradient/training and one inference GPU"
        )
    if allowed is not None:
        pool = validate_pool(allowed)
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        devices = [
            tuple(part.strip() for part in line.split(","))
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        by_index = dict(devices)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            pool = [uuid for _, uuid in devices]
        else:
            pool = []
            for token in visible.split(","):
                token = token.strip()
                if token in by_index:
                    pool.append(by_index[token])
                elif token in by_index.values():
                    pool.append(token)
                else:
                    raise ValueError(
                        "CUDA_VISIBLE_DEVICES must contain valid physical indices or full GPU UUIDs"
                    )
        validate_pool(pool)
    if len(pool) < count:
        raise ValueError(
            f"Requested {count} GPUs but only {len(pool)} are available in the selected pool; set --num-gpus explicitly"
        )
    return pool[:count]


def add_gpu_arguments(parser) -> None:
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Total GPUs; default 8, minimum 2 (one training + remaining inference)",
    )
    parser.add_argument(
        "--gpu-uuids",
        nargs="+",
        help="Authorized pool in order; otherwise discover visible devices on the authorized server",
    )
    parser.add_argument(
        "--train-gpu", help="Legacy two-GPU mode; use together with --service-gpu"
    )
    parser.add_argument(
        "--service-gpu", help="Legacy two-GPU mode; use together with --train-gpu"
    )


def gpu_arguments(args) -> list[str]:
    if args.train_gpu or args.service_gpu:
        if (
            not args.train_gpu
            or not args.service_gpu
            or args.gpu_uuids
            or args.num_gpus not in {2, 8}
        ):
            raise ValueError(
                "Legacy two-GPU flags need both UUIDs and cannot be combined with a GPU pool/count other than 2"
            )
        return validate_pool([args.train_gpu, args.service_gpu])
    return select_gpus(args.num_gpus, args.gpu_uuids)
