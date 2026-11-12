"""Read-only reuse inventory and container-side network gate for reevaluation.

This module deliberately does not change the legacy DockerTask or run training.
"""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from nice_validation.core import read, sha


def independent_paths(old: Path, output: Path):
    old, output = old.resolve(), output.resolve()
    if old == output or old.is_relative_to(output) or output.is_relative_to(old):
        raise ValueError("Old run and new deployment output must be separate, non-nested directories")


def network_environment(endpoint: str) -> dict:
    url = urlsplit(endpoint)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError("Use an http(s) endpoint without embedded credentials")
    if url.query or url.fragment or url.path not in {"", "/"}:
        raise ValueError("Anthropic base URL must be the service root, without /v1 or query")
    host = url.hostname
    if host.lower() in {"localhost", "localhost.localdomain"}:
        raise ValueError("Container localhost is not the host inference server")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None  # DNS is resolved INSIDE the actual task container below.
    if address and (address.is_loopback or address.is_unspecified or address.is_multicast):
        raise ValueError("Use a container-reachable unicast service address")
    no_proxy = ",".join(dict.fromkeys([host, "localhost", "127.0.0.1", "::1"]))
    # Clear both cases; a corporate HTTP proxy must not intercept local model calls.
    env = {name: "" for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")}
    env.update(NO_PROXY=no_proxy, no_proxy=no_proxy,
               ANTHROPIC_BASE_URL=endpoint.rstrip("/"),
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
               DISABLE_AUTOUPDATER="1")
    return env


def inspect_image(reference: str, execute=None) -> dict:
    execute = execute or subprocess.run
    result = execute(["docker", "image", "inspect", reference], check=True,
                     capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)[0]


def inventory(old: Path, inspect=None) -> dict:
    """Only local inspect: no pull, fetch, image build, GPU use, or old writes."""
    old = Path(old)
    snapshot = read(old / "config/snapshot.json")
    cfg = snapshot["config"]
    selected = read(old / "selection.json")["selected"]
    expected = {"low_lr1_p100", "low_lr2_p025", "high_lr1_p050", "high_lr2_p100"}
    if {x["model_id"] for x in selected} != expected:
        raise ValueError("Original selected models differ; do not silently substitute checkpoints")
    models = {"initial": cfg["model"]["adapter_path"]}
    models.update({name: str(old / "training" / name / "adapter") for name in sorted(expected)})
    issues, adapters = [], {}
    for name, directory in models.items():
        folder = Path(directory)
        files = [folder / "adapter_config.json", folder / "adapter_model.safetensors"]
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            issues.append({"model": name, "missing": missing})
        else:
            adapters[name] = {"path": str(folder.resolve()), "files": {p.name: sha(p) for p in files}}
    base = Path(cfg["model"]["path"])
    base_lock = base / "model_lock.json"
    if not base_lock.is_file():
        issues.append({"base": str(base), "reason": "model_lock.json missing"})
    else:
        lock = read(base_lock)
        if lock.get("revision") != cfg["model"]["revision"] or lock.get("id") != cfg["model"]["id"]:
            issues.append({"base": str(base), "reason": "model identity differs"})
        for name, meta in lock.get("files", {}).items():
            path = (base / name).resolve()
            if not path.is_relative_to(base.resolve()) or not path.is_file() or sha(path) != meta["sha256"]:
                issues.append({"base_file": name, "reason": "missing or checksum mismatch"})
    image_cache, scopes, ids = {}, {}, set()
    inspector = inspect or inspect_image
    for scope, filename, expected_count, image_scope in [
        ("benchmark", "benchmark", 100, "benchmark"),
        ("nice_retest", "validation", 20, "nice"),
    ]:
        task_path = old / "manifests" / (filename + ".json")
        tasks = read(task_path)
        if len(tasks) != expected_count:
            raise ValueError("Frozen task count differs: " + scope)
        mapping_path = old / "environments" / image_scope / "task_images.json"
        mapping = read(mapping_path)
        lock_path = mapping_path.with_name("task_images.lock.json")
        if read(lock_path)["image_map_sha256"] != sha(mapping_path):
            raise ValueError("Original task image map checksum differs")
        scopes[scope] = {"manifest": str(task_path.resolve()), "sha256": sha(task_path), "tasks": []}
        for task in tasks:
            key = task["instance_id"]
            if key in ids:
                raise ValueError("Task manifests overlap")
            ids.add(key)
            entry = mapping.get(key)
            if not entry or entry.get("public_environment_only") is not True or entry["base_commit"] != task["base_commit"]:
                raise ValueError("Task image mapping differs: " + key)
            record = {"task_id": key, "entry": entry}
            for role, reference in [("agent", entry["image"]),
                                    ("grader", entry.get("grader_image", entry["image"])),
                                    ("official_alias", entry["official_image_name"])]:
                if reference not in image_cache:
                    try:
                        image_cache[reference] = inspector(reference)["Id"]
                    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
                        image_cache[reference] = None
                        issues.append({"image": reference, "reason": str(exc)})
                record[role + "_image_id"] = image_cache[reference]
            if record["grader_image_id"] and record["official_alias_image_id"] and record["grader_image_id"] != record["official_alias_image_id"]:
                issues.append({"task_id": key, "reason": "official grader alias content mismatch"})
            scopes[scope]["tasks"].append(record)
    return {"status": "BLOCKED" if issues else "RESOURCES_PRESENT", "old_run": str(old.resolve()),
            "old_identity": snapshot["identity"], "snapshot_sha256": sha(old / "config/snapshot.json"),
            "base_path": str(base), "base_revision": cfg["model"]["revision"],
            "adapters": adapters, "scopes": scopes, "issues": issues,
            "formal_episodes": 600, "scope": "Local resource inventory, not GPU/CLI/grade acceptance"}


PROBE = r'''
import ipaddress,json,socket,urllib.request,sys
from urllib.parse import urlsplit
url=sys.argv[1];host=urlsplit(url).hostname
addresses=sorted({r[4][0] for r in socket.getaddrinfo(host,None)})
if not addresses or any(not ipaddress.ip_address(a).is_private or ipaddress.ip_address(a).is_loopback or ipaddress.ip_address(a).is_unspecified for a in addresses):
    raise RuntimeError('Endpoint must resolve to reachable private service addresses: '+str(addresses))
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(url.rstrip('/')+'/health',timeout=20) as r:
    if r.status!=200:raise RuntimeError('Unhealthy inference endpoint')
print(json.dumps({'status':'PASS','addresses':addresses,'proxy_bypassed':True,'scope':'DNS/TCP/HTTP health only; not model or tool acceptance'}))
'''


def probe_network(image: str, network: str, endpoint: str, execute=None):
    if not network or network in {"none", "host", "bridge"}:
        raise ValueError("Supply an existing dedicated Docker network, not host/none/default bridge")
    env = network_environment(endpoint)
    command = ["docker", "run", "--rm", "--pull=never", "--network", network,
               "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=128"]
    for key, value in env.items():
        command += ["--env", key + "=" + value]
    command += ["--entrypoint", "python", image, "-c", PROBE, endpoint]
    execute = execute or subprocess.run
    result = execute(command, capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])
