"""Internal NICE worker: independent per-example gradients, never Run state."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.parallel_scoring import worker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--coordinator-pid", required=True, type=int)
    args = parser.parse_args()
    worker(
        args.snapshot,
        args.plan,
        args.shard_index,
        args.gpu_uuid,
        args.attempt,
        args.coordinator_pid,
    )
