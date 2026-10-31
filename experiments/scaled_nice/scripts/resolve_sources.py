"""Explicit metadata lookup only; never downloads model weights."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nice_validation.core import config, write
from nice_validation.online import resolve_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit(
            "Source revisions already exist; preserve them or choose a new output file"
        )
    records = resolve_sources(config(args.config))
    write(Path(args.output), records)
    print(__import__("json").dumps(records, indent=2))


if __name__ == "__main__":
    main()
