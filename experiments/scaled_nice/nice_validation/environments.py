"""Acquire and freeze official task images before any model outcomes are seen."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import json
import subprocess
import os
import re

from nice_validation.core import Run, digest, read, require_commit, sha, write


def distribution_identity(name: str, expected: str, namespace: str) -> dict:
    require_commit(expected, name + ".revision")
    dist = importlib.metadata.distribution(name)
    direct = dist.read_text("direct_url.json")
    metadata = json.loads(direct) if direct else {}
    if metadata.get("vcs_info", {}).get("commit_id") != expected:
        raise ValueError(
            f"Install official {name} at the frozen commit; distribution direct_url.json must match"
        )
    files = {
        str(f): sha(Path(dist.locate_file(f)))
        for f in dist.files or []
        if str(f).endswith(".py")
        and str(f).replace("\\", "/").startswith(namespace + "/")
    }
    return {
        "version": dist.version,
        "commit": expected,
        "source_sha256": digest(files),
        "direct_url": metadata,
    }


def live_info(cfg: dict, task_file: Path | None = None) -> dict:
    command = [
        cfg["transfer"]["live_python"],
        str(Path(__file__).resolve().parent.parent / "scripts/live_info.py"),
    ]
    if task_file:
        command += [
            "--task",
            str(task_file),
            "--namespace",
            cfg["transfer"]["suites"]["live"]["namespace"],
        ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=60
    )
    value = json.loads(result.stdout.strip().splitlines()[-1])
    if (
        value["source"].get("vcs_info", {}).get("commit_id")
        != cfg["transfer"]["live_revision"]
    ):
        raise ValueError("Live isolated harness differs from frozen commit")
    return value


def official_identity(cfg: dict) -> dict:
    result = distribution_identity(
        "swebench", cfg["agent"]["swebench_revision"], "swebench"
    )
    if cfg["data"].get("validation_kind") == "swesmith":
        result["swesmith"] = distribution_identity(
            "swesmith", cfg["agent"]["swesmith_revision"], "swesmith"
        )
    if cfg.get("transfer", {}).get("enabled"):
        result["harbor"] = distribution_identity(
            "harbor", cfg["transfer"]["harbor_revision"], "harbor"
        )
        result["live"] = live_info(cfg)
    return result


def prepare_smith_image(client, task: dict, base_image) -> tuple[str, str]:
    """Fetch during trusted preparation, then keep only public checkout history."""
    container = client.containers.create(
        base_image.id,
        entrypoint="/bin/bash",
        command=["-lc", "sleep infinity"],
        detach=True,
    )
    try:
        container.start()
        proxies = {
            key: os.environ[key]
            for key in [
                "HTTPS_PROXY",
                "HTTP_PROXY",
                "NO_PROXY",
                "https_proxy",
                "http_proxy",
                "no_proxy",
            ]
            if key in os.environ
        }

        def execute(command, environment=None):
            result = container.exec_run(
                command, workdir="/testbed", environment=environment
            )
            if result.exit_code:
                raise ValueError(
                    "SWE-smith task preparation failed; inspect task refs and preparation-container network"
                )
            return result.output.decode("utf-8").strip()

        execute(["git", "fetch", "origin", task["instance_id"]], proxies)
        execute(["git", "checkout", "--detach", "FETCH_HEAD"])
        commit = execute(["git", "rev-parse", "HEAD"])
        require_commit(commit, "SWE-smith checkout SHA")
        if execute(["git", "status", "--porcelain", "--untracked-files=all"]):
            raise ValueError("Prepared SWE-smith checkout is not clean")
        for ref in execute(["git", "for-each-ref", "--format=%(refname)"]).splitlines():
            execute(["git", "update-ref", "-d", ref])
        execute(["git", "remote", "remove", "origin"])
        execute(
            [
                "python",
                "-c",
                "from pathlib import Path; Path('.git/shallow').write_text("
                + repr(commit + "\n")
                + ")",
            ]
        )
        for command in [
            ["git", "reflog", "expire", "--expire=now", "--all"],
            ["git", "repack", "-a", "-d"],
            ["git", "prune", "--expire=now"],
        ]:
            execute(command)
        if execute(["git", "rev-list", "--all", "--count"]) != "1":
            raise ValueError("SWE-smith hidden history isolation failed")
        image = container.commit()
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image.id):
            raise ValueError("Prepared image has no immutable content identity")
        return image.id, commit
    finally:
        container.remove(force=True)


def verified_image(task: dict) -> str:
    from swebench.harness.test_spec.test_spec import make_test_spec

    spec = make_test_spec(task, namespace="swebench")
    return getattr(spec, "image", None) or getattr(spec, "instance_image_key", None)


def prepare_environments(run: Run) -> None:
    """Preflight only. Failed images stop before rollout; no outcome-based replacement."""
    if run.cfg["mode"] != "real" or run.cfg["execution"] != "server":
        raise ValueError("Task image preparation is an explicit server operation")
    if not run.valid("prepare-data"):
        raise ValueError("Freeze prepare-data before environment preparation")
    if run.cfg.get("transfer", {}).get("enabled"):
        from nice_validation.transfer import prepare_native_environments

        prepare_native_environments(run)
    with run.stage("prepare-environments") as out:
        if out is None:
            return
        identity = official_identity(run.cfg)
        import docker

        client = docker.from_env()
        mapping, failures = {}, []
        tasks = read(run.root / "manifests/validation.json")
        if run.cfg.get("transfer", {}).get("enabled"):
            from nice_validation.transfer_sources import transfer_tasks

            if not run.valid("prepare-transfer"):
                raise ValueError("Freeze transfer tasks before environment preparation")
            tasks += [
                t for t in transfer_tasks(run) if t["evaluation_backend"] != "harbor"
            ]
        if run.cfg["agent"].get("smoke_task_path"):
            development = read(Path(run.cfg["agent"]["smoke_task_path"]))
            if development["instance_id"] in {t["instance_id"] for t in tasks}:
                raise ValueError(
                    "Development smoke task must be outside the main validation set"
                )
            tasks.append(development)
        try:
            for task in tasks:
                try:
                    backend = task.get(
                        "evaluation_backend",
                        run.cfg["data"].get("validation_kind", "swesmith"),
                    )
                    if backend == "swesmith":
                        image_name = task["image_name"]
                    elif backend == "live":
                        source = (
                            run.root
                            / "preflight/live-source"
                            / (task["instance_id"] + ".json")
                        )
                        write(source, task)
                        image_name = live_info(run.cfg, source)["image"]
                    else:
                        image_name = verified_image(task)
                    if not image_name:
                        raise ValueError(
                            "Installed official API exposes no task image name"
                        )
                    try:
                        image = client.images.get(image_name)
                    except docker.errors.ImageNotFound:
                        image = client.images.pull(image_name)
                    digests = image.attrs.get("RepoDigests", [])
                    image_id = digests[0] if digests else image.id
                    grader_image = image_id
                    checkout_sha = task["base_commit"]
                    if backend == "swesmith":
                        image_id, checkout_sha = prepare_smith_image(
                            client, task, image
                        )
                    # Only use public clean base images. Grader scripts/patches are
                    # never copied to these containers. Grading creates its own environment.
                    mapping[task["instance_id"]] = {
                        "image": image_id,
                        "base_commit": task["base_commit"],
                        "public_environment_only": True,
                        "official_image_name": image_name,
                        "backend": backend,
                        "checkout_commit": checkout_sha,
                        "grader_image": grader_image,
                        "shell_setup": "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then source /opt/miniconda3/etc/profile.d/conda.sh; conda activate testbed; fi",
                    }
                except (docker.errors.DockerException, ValueError, KeyError) as exc:
                    failures.append(
                        {
                            "task_id": task["instance_id"],
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
        finally:
            client.close()
        path = Path(run.cfg["agent"]["image_map"])
        write(path, mapping)
        write(
            Path(run.cfg["agent"]["image_revision_lock"]),
            {"image_map_sha256": sha(path), "official_harness": identity},
        )
        from nice_validation.harness import DockerTask, InfraError, TaskFailure

        for task in tasks:
            if task["instance_id"] not in mapping:
                continue
            try:
                with DockerTask(
                    {**run.cfg, "_run_identity": run.identity},
                    task,
                    run.root / "preflight" / task["instance_id"],
                ) as environment:
                    output = environment.command("python --version", 60)
                    if "[exit_code=" in output:
                        raise InfraError("task_python_unavailable:" + output)
            except (InfraError, TaskFailure) as exc:
                failures.append({"task_id": task["instance_id"], "reason": str(exc)})
        report = {
            "task_count": run.cfg["experiment"]["validation"],
            "images_ready": len(mapping),
            "failures": failures,
            "official_harness": identity,
            "replacement_policy": "No replacements applied. Repair environment and rerun; model outcomes not consulted.",
        }
        destination = run.root / "manifests/environment_preflight.json"
        write(destination, report)
        out.append(destination)
        if failures:
            raise ValueError(
                "Environment preflight incomplete; inspect manifests/environment_preflight.json"
            )
        frozen = run.root / "manifests/validation_freeze.json"
        write(
            frozen,
            {
                "status": "final_frozen",
                "validation_sha256": sha(run.root / "manifests/validation.json"),
                "environment_lock_sha256": sha(
                    Path(run.cfg["agent"]["image_revision_lock"])
                ),
                "selection_has_not_seen_model_outcomes": True,
            },
        )
        out.append(frozen)


def replace_validation(run: Run, task_id: str) -> dict:
    """Deterministic same-repository reserve replacement, before main model contact."""
    from nice_validation.core import utc

    if run.cfg["mode"] != "real":
        raise ValueError(
            "Validation replacement only applies to real environment preflight"
        )
    if any(
        name.startswith("rollout-nice") or name.startswith("evaluate:")
        for name in run.state()
    ):
        raise ValueError(
            "Main model contact has started; the validation denominator is now immutable"
        )
    if (run.root / "manifests/validation_freeze.json").exists():
        raise ValueError(
            "Final validation set is frozen; repair infrastructure without replacing tasks"
        )
    report = read(run.root / "manifests/environment_preflight.json")
    errors = [x for x in report["failures"] if x["task_id"] == task_id]
    if not errors:
        raise ValueError(
            "Only a documented environment preflight failure permits replacement"
        )
    with run.stage("replace-validation:" + task_id) as out:
        if out is None:
            return read(run.root / "manifests" / f"replacement_{task_id}.json")
        validation_path = run.root / "manifests/validation.json"
        reserve_path = run.root / "manifests/validation_reserve.json"
        validation, reserve = read(validation_path), read(reserve_path)
        index = next(
            (i for i, t in enumerate(validation) if t["instance_id"] == task_id), None
        )
        if index is None:
            raise ValueError("Task is not in current validation selection")
        selected_repo = validation[index]["repo"]
        development_path = run.cfg["agent"].get("smoke_task_path")
        development_id = (
            read(Path(development_path))["instance_id"]
            if development_path and Path(development_path).is_file()
            else None
        )
        from nice_validation.data import normalized, near_prompt

        candidates_path = run.root / "manifests/candidates.json"
        candidates = read(candidates_path) if candidates_path.is_file() else []
        trained_ids = {row["instance_id"] for row in candidates}
        trained_prompts = [
            normalized(
                "\n".join(
                    m.get("content") or ""
                    for m in row.get("messages", [])
                    if m["role"] == "user"
                )
            )
            for row in candidates
        ]

        def eligible(task):
            prompt = normalized(task.get("problem_statement", ""))
            return (
                task["repo"] == selected_repo
                and task["instance_id"] != development_id
                and task["instance_id"] not in trained_ids
                and not any(
                    prompt == prior
                    or near_prompt(
                        prompt, prior, run.cfg["data"]["duplicate_similarity"]
                    )
                    for prior in trained_prompts
                    if prior
                )
            )

        reserve_index = next(
            (i for i, t in enumerate(reserve) if eligible(t)),
            None,
        )
        if reserve_index is None:
            raise ValueError(
                "No same-repository reserve remains; cannot silently alter repository quotas"
            )
        old = validation[index]
        replacement = reserve.pop(reserve_index)
        record = {
            "removed": old,
            "replacement": replacement,
            "reason": errors,
            "preflight": report,
            "utc": utc(),
            "rule": "first eligible same-repo reserve excluding training and development tasks and near prompts; no main model outcomes observed",
        }
        validation[index] = replacement
        write(validation_path, validation)
        write(reserve_path, reserve)
        checksums_path = run.root / "manifests/checksums.json"
        checksums = read(checksums_path)
        checksums["validation.json"], checksums["validation_reserve.json"] = sha(
            validation_path
        ), sha(reserve_path)
        write(checksums_path, checksums)
        state = run.state()
        for artifact in [validation_path, reserve_path, checksums_path]:
            state["prepare-data"]["artifacts"][str(artifact.relative_to(run.root))] = (
                sha(artifact)
            )
        write(run.state_path, state)
        p = run.root / "manifests" / f"replacement_{task_id}.json"
        write(p, record)
        out.append(p)
        return record
