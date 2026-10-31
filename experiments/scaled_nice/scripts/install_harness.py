"""Explicitly install the official benchmark at the generated frozen commit."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.core import config, require_commit
from nice_validation.environments import official_identity
from nice_validation.model import server_only
from nice_validation.compatibility import harness_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    server_only(cfg)
    commit = cfg["agent"]["swebench_revision"]
    require_commit(commit, "agent.swebench_revision")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/SWE-bench/SWE-bench.git@" + commit,
        ],
        check=True,
    )
    commit = cfg["agent"]["swesmith_revision"]
    require_commit(commit, "agent.swesmith_revision")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "swesmith[validate] @ git+https://github.com/SWE-bench/SWE-smith.git@"
            + commit,
        ],
        check=True,
    )
    if cfg.get("transfer", {}).get("enabled"):
        import venv

        harbor = cfg["transfer"]["harbor_revision"]
        require_commit(harbor, "transfer.harbor_revision")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/harbor-framework/harbor.git@" + harbor,
            ],
            check=True,
        )
        live = cfg["transfer"]["live_revision"]
        require_commit(live, "transfer.live_revision")
        interpreter = Path(cfg["transfer"]["live_python"])
        if not interpreter.is_file():
            venv.EnvBuilder(with_pip=True).create(interpreter.parent.parent)
        subprocess.run(
            [
                str(interpreter),
                "-m",
                "pip",
                "install",
                "git+https://github.com/microsoft/SWE-bench-Live.git@" + live,
            ],
            check=True,
        )
        subprocess.run([str(interpreter), "-m", "pip", "check"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print(official_identity(cfg))
    print(harness_api(cfg))


if __name__ == "__main__":
    main()
