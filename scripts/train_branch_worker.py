"""Internal worker: the coordinator holds the run lock and owns stage publication."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.parallel_training import worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--coordinator-pid", type=int)
    args = parser.parse_args()
    worker(args.snapshot, args.subset, args.gpu_uuid, args.log, args.coordinator_pid)
