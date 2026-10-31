"""Internal disposable single-GPU throughput/capacity probe."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.batch_tuning import profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile(args.snapshot, args.gpu_uuid, args.output)
