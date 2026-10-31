"""Record exact installed distributions and source checkout without machine paths."""

import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.core import source_identity, utc, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    }
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    write(
        Path(args.output),
        {
            "utc": utc(),
            "python": sys.version,
            "packages": dict(sorted(packages.items())),
            "source": source_identity(),
            "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
