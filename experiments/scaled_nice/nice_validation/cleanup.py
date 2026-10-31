"""Recover only containers carrying the selected frozen run's exact label."""

from pathlib import Path
import json
import os
import platform
import re
import subprocess

from nice_validation.core import read, utc, write


def cleanup_containers(root: Path, execute: bool = False) -> dict:
    frozen = read(root / "config/snapshot.json")
    identity = frozen["identity"]
    if not re.fullmatch(r"[a-f0-9]{64}", identity):
        raise ValueError("Invalid frozen run identity")
    lock = root / ".run.lock"
    if execute and lock.is_file():
        import psutil

        owner = read(lock)
        if owner["host"] != platform.node() or psutil.pid_exists(owner["pid"]):
            raise ValueError(
                "Run owner may still be active; refusing container cleanup"
            )
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=nice.run=" + identity],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    containers = []
    for container in result.stdout.split():
        if not re.fullmatch(r"[a-f0-9]{12,64}", container):
            raise ValueError("Docker returned an invalid container ID")
        inspected = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", container],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        labels = json.loads(inspected.stdout)
        if labels.get("nice.run") != identity:
            raise ValueError("Container label differs from the selected run")
        if execute:
            subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
        containers.append(
            {"container_id": container, "labels": labels, "removed": execute}
        )
    value = {
        "utc": utc(),
        "run_identity": identity,
        "execute": execute,
        "containers": containers,
    }
    write(root / "logs" / f"container-cleanup-{os.getpid()}.json", value)
    return value
