"""Read-only server readiness checks. Never loads or downloads model weights."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path
import tomllib
import urllib.error
import urllib.request

from packaging.requirements import Requirement

from nice_validation.core import doctor, read, require_commit, sha, utc


def preflight(cfg: dict, prepared: bool = False, network: bool = False) -> dict:
    checks = []

    def add(name, ok, detail):
        checks.append(
            {"check": name, "status": "PASS" if ok else "FAIL", "detail": detail}
        )

    add(
        "server_execution",
        cfg["execution"] == "server" and cfg["mode"] == "real",
        "requires mode=real, execution=server",
    )
    package_file = Path(__file__).resolve().parent.parent / "pyproject.toml"
    project = tomllib.loads(package_file.read_text(encoding="utf-8"))["project"]
    for declaration in (
        project["dependencies"] + project["optional-dependencies"]["server"]
    ):
        req = Requirement(declaration)
        try:
            version = importlib.metadata.version(req.name)
            add(
                "dependency:" + req.name,
                version in req.specifier,
                {"required": declaration, "installed": version},
            )
        except importlib.metadata.PackageNotFoundError:
            add(
                "dependency:" + req.name,
                False,
                {"required": declaration, "installed": None},
            )
    for section, key in [
        ("model", "revision"),
        ("data", "training_revision"),
        ("data", "validation_revision"),
        ("agent", "swebench_revision"),
        ("agent", "swesmith_revision"),
    ]:
        try:
            require_commit(cfg[section][key], section + "." + key)
            add(section + "." + key, True, cfg[section][key])
        except ValueError as exc:
            add(section + "." + key, False, str(exc))
    for section, key in [
        ("model", "path"),
        ("model", "adapter_path"),
        ("model", "loader"),
        ("agent", "image_map"),
        ("agent", "image_revision_lock"),
        ("agent", "smoke_task_path"),
    ]:
        value = cfg[section].get(key)
        add(section + "." + key, bool(value) and "REPLACE_" not in str(value), value)
    info = doctor(cfg)
    try:
        from nice_validation.compatibility import harness_api

        add("official_harness_api", True, harness_api(cfg))
    except (ImportError, AttributeError, ValueError, RuntimeError, TypeError) as exc:
        add("official_harness_api", False, f"{type(exc).__name__}: {exc}")
    if cfg.get("transfer", {}).get("enabled"):
        for field in ["harbor_revision", "live_revision"]:
            try:
                require_commit(cfg["transfer"][field], "transfer." + field)
                add("transfer." + field, True, cfg["transfer"][field])
            except ValueError as exc:
                add("transfer." + field, False, str(exc))
        for suite, spec in cfg["transfer"]["suites"].items():
            try:
                require_commit(spec["revision"], suite + ".revision")
                add("transfer.revision:" + suite, True, spec["revision"])
            except ValueError as exc:
                add("transfer.revision:" + suite, False, str(exc))
        add(
            "live_harness_python",
            bool(cfg["transfer"].get("live_python"))
            and Path(cfg["transfer"]["live_python"]).is_file(),
            cfg["transfer"].get("live_python"),
        )
    for key in ["gpu", "docker"]:
        add(key, info[key].get("returncode") == 0, info[key])
    commands = [("git_cli", ["git", "--version"])]
    if cfg.get("transfer", {}).get("enabled"):
        commands.append(("docker_compose", ["docker", "compose", "version", "--short"]))
    for name, argv in commands:
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
            add(
                name,
                result.returncode == 0,
                {"stdout": result.stdout, "stderr": result.stderr},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            add(name, False, f"{type(exc).__name__}: {exc}")
    uuids = cfg["model"]["gpu_uuids"]
    gpu_info = info["gpu"].get("stdout", "")
    add(
        "authorized_gpu",
        len(uuids) == 1 and uuids[0] in gpu_info,
        {"requested": uuids, "observed": gpu_info},
    )
    resources = cfg.get("resources", {})
    pool = resources.get("gpu_uuids", uuids)
    add(
        "authorized_gpu_pool",
        len(pool) == resources.get("gpu_count", len(pool))
        and len(pool) == len(set(pool))
        and bool(pool)
        and all(gpu in gpu_info for gpu in pool),
        {"requested": pool, "observed": gpu_info},
    )
    if cfg["model"]["path"]:
        base = Path(cfg["model"]["path"]).resolve()
        output = Path(cfg["output"]).resolve()
        add(
            "output_outside_base_model",
            base != output and base not in output.parents,
            str(output),
        )
    if prepared:
        try:
            from nice_validation.compatibility import model_loader_api

            add(
                "model_loader_available",
                True,
                model_loader_api(cfg, Path(cfg["model"]["path"] or ".")),
            )
        except (ImportError, RuntimeError, ValueError, OSError, TypeError) as exc:
            add("model_loader_available", False, f"{type(exc).__name__}: {exc}")
        paths = [
            ("model_lock", Path(cfg["model"]["path"] or ".") / "model_lock.json"),
            (
                "initial_adapter",
                Path(cfg["model"]["adapter_path"] or ".") / "identity.json",
            ),
        ]
        paths += [
            (key, Path(cfg["data"][key]))
            for key in ["training_path", "validation_path", "source_lock"]
        ]
        paths += [
            (key, Path(cfg["agent"].get(key) or "."))
            for key in ["image_map", "image_revision_lock", "smoke_task_path"]
        ]
        if cfg.get("transfer", {}).get("enabled"):
            paths += [
                ("transfer_source_lock", Path(cfg["transfer"]["source_lock"])),
                ("transfer_tasks", Path(cfg["output"]) / "transfer/tasks.json"),
                ("native_images", Path(cfg["output"]) / "transfer/native_images.json"),
            ]
        for key, path in paths:
            add("artifact:" + key, path.is_file(), str(path))
        lock_path = Path(cfg["data"]["source_lock"])
        if lock_path.is_file():
            lock = read(lock_path)
            for kind in ["training", "validation"]:
                path = Path(cfg["data"][kind + "_path"])
                entry = lock.get(kind, {})
                add(
                    "source_identity:" + kind,
                    path.is_file()
                    and entry.get("revision") == cfg["data"][kind + "_revision"]
                    and entry.get("sha256") == sha(path),
                    str(path),
                )
        from nice_validation.service import matching_health

        for index, endpoint in enumerate(
            cfg["agent"]["service_endpoints"][: cfg["agent"]["workers"]]
        ):
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(
                    endpoint.rstrip("/") + "/health", timeout=5
                ) as response:
                    health = json.load(response)
                selected = cfg.get("resources", {}).get("gpu_uuids", [])
                expected_gpu = (
                    selected[index + 1] if index + 1 < len(selected) else None
                )
                add(
                    "service:" + endpoint,
                    matching_health(health, cfg, expected_gpu),
                    health,
                )
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                add("service:" + endpoint, False, type(exc).__name__)
    if network:
        for name, url in [
            ("huggingface", "https://huggingface.co/api/models/" + cfg["model"]["id"]),
            ("github", "https://api.github.com/repos/SWE-bench/SWE-bench"),
            ("registry", "https://registry-1.docker.io/v2/"),
        ]:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    add(
                        "network:" + name,
                        response.status == 200,
                        {"status_code": response.status},
                    )
            except urllib.error.HTTPError as exc:
                add(
                    "network:" + name,
                    name == "registry" and exc.code == 401,
                    {
                        "status_code": exc.code,
                        "note": "registry reachability only; image download not verified",
                    },
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                add("network:" + name, False, type(exc).__name__)
    return {
        "utc": utc(),
        "status": "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS",
        "phase": "prepared" if prepared else "bootstrap",
        "checks": checks,
        "system": info,
        "weights_loaded": False,
        "weights_downloaded": False,
        "scope": "Prerequisites only. Real smoke including full-context update is still required.",
    }
