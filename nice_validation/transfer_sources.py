"""Pinned external benchmark sources. No model execution or outcome-based sampling."""

from pathlib import Path
import random
import subprocess

from nice_validation.core import (
    Run,
    digest,
    read,
    read_jsonl,
    require_commit,
    sha,
    write,
)


SUITES = ("verified", "pro", "live", "terminal")


def checkout(repository: str, revision: str, target: Path) -> Path:
    require_commit(revision, repository)
    url = "https://github.com/" + repository + ".git"
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target.with_name(target.name + ".checkout.json")
    identity = {"repository": repository, "revision": revision}
    if marker.exists() and any(read(marker).get(k) != v for k, v in identity.items()):
        raise ValueError("Interrupted checkout belongs to a different source")
    if not target.exists():
        target.mkdir()
        write(marker, {**identity, "status": "initializing"})
        subprocess.run(
            ["git", "-C", str(target), "init"], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(target), "remote", "add", "origin", url], check=True
        )
        write(marker, {**identity, "status": "fetching"})
    elif not (target / ".git").is_dir():
        if (
            marker.exists()
            and read(marker).get("status") == "initializing"
            and not any(target.iterdir())
        ):
            subprocess.run(
                ["git", "-C", str(target), "init"], check=True, capture_output=True
            )
        else:
            raise ValueError(
                "Existing checkout directory is not a repository; refusing overwrite"
            )
    if marker.exists() and read(marker).get("status") == "initializing":
        remotes = subprocess.run(
            ["git", "-C", str(target), "remote"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if "origin" not in remotes:
            subprocess.run(
                ["git", "-C", str(target), "remote", "add", "origin", url], check=True
            )
        write(marker, {**identity, "status": "fetching"})
    origin = subprocess.run(
        ["git", "-C", str(target), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if origin != url:
        raise ValueError("Benchmark checkout remote differs from frozen source")
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if head.returncode == 0 and (
        not marker.exists() or read(marker).get("status") == "completed"
    ):
        dirty = subprocess.run(
            ["git", "-C", str(target), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if head.stdout.strip() != revision or dirty:
            raise ValueError(
                "Benchmark checkout changed; preserve it and use a new source directory"
            )
        return target
    subprocess.run(
        ["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", revision], check=True
    )
    dirty = subprocess.run(
        ["git", "-C", str(target), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if dirty:
        raise ValueError("Pinned checkout is not clean after acquisition")
    write(marker, {**identity, "status": "completed"})
    return target


def task_files(path: Path) -> dict:
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            if not item.resolve().is_relative_to(path.resolve()):
                raise ValueError("Task file escapes the frozen benchmark directory")
            files[str(item.relative_to(path)).replace("\\", "/")] = sha(item)
    return files


def acquire_transfer(cfg: dict, mode: str = "download") -> dict:
    from nice_validation.acquire import hf_split
    from nice_validation.model import server_only

    server_only(cfg)
    transfer = cfg["transfer"]
    path = Path(transfer["source_lock"])
    sources = {}
    for suite in SUITES:
        spec = transfer["suites"][suite]
        require_commit(spec["revision"], suite + ".revision")
        if spec["backend"] != "harbor":
            target = path.parent / (suite + ".jsonl")
            entry = hf_split(
                spec["id"],
                spec["revision"],
                spec["split"],
                target,
                mode,
                True,
                cfg["data"],
            )
            sources[suite] = {"backend": spec["backend"], **entry}
        else:
            repo = checkout(
                spec["id"],
                spec["revision"],
                Path(transfer["checkouts"]) / f"{suite}-{spec['revision']}",
            )
            task_root = (repo / spec["task_subdir"]).resolve()
            rows = []
            if suite == "terminal":
                registered = spec.get("registered_tasks")
                if not registered or len({item["name"] for item in registered}) != len(
                    registered
                ):
                    raise ValueError("Terminal 2.0 needs its frozen registry task list")
                paths = [(repo / item["path"]).resolve() for item in registered]
                if any(not item.is_relative_to(repo.resolve()) for item in paths):
                    raise ValueError("Registered task path escapes checkout")
            else:
                paths = sorted(task_root.iterdir())
            for task in paths:
                if task.is_dir() and (task / "task.toml").is_file():
                    rows.append(
                        {
                            "instance_id": task.name,
                            "repo": suite,
                            "problem_statement": (task / "instruction.md").read_text(
                                encoding="utf-8"
                            ),
                            "task_path": str(task),
                            "task_files": task_files(task),
                        }
                    )
                elif suite == "terminal":
                    raise ValueError(
                        "Registered Terminal task is missing its native task.toml"
                    )
            if len(rows) < transfer["sample_size"]:
                raise ValueError(f"Not enough native {suite} tasks in frozen checkout")
            target = path.parent / (suite + ".json")
            if target.exists() and read(target) != rows:
                raise ValueError("Native benchmark source changed")
            write(target, rows)
            sources[suite] = {
                "backend": "harbor",
                "id": spec["id"],
                "revision": spec["revision"],
                "version": spec["version"],
                "path": str(target),
                "sha256": sha(target),
                "records": len(rows),
            }
    if path.exists() and read(path) != sources:
        raise ValueError(
            "Transfer source identity changed; use a new source/run directory"
        )
    write(path, sources)
    return sources


def source_rows(cfg: dict) -> dict:
    lock = read(Path(cfg["transfer"]["source_lock"]))
    result = {}
    for suite in SUITES:
        entry = lock[suite]
        expected = cfg["transfer"]["suites"][suite]
        path = Path(entry["path"])
        if (
            entry["id"] != expected["id"]
            or entry["revision"] != expected["revision"]
            or sha(path) != entry["sha256"]
        ):
            raise ValueError(
                "Transfer source differs from frozen configuration/checksum"
            )
        result[suite] = read(path) if path.suffix == ".json" else read_jsonl(path)
    return result


def prepare_transfer(run: Run) -> None:
    if not run.cfg.get("transfer", {}).get("enabled"):
        return
    with run.stage("prepare-transfer") as artifacts:
        if artifacts is None:
            return
        if run.cfg["mode"] == "mock":
            sources = {
                suite: [
                    {
                        "instance_id": f"{suite}__fixture-{i:03d}",
                        "repo": suite,
                        "problem_statement": f"Synthetic independent {suite} transfer task {i}",
                    }
                    for i in range(80)
                ]
                for suite in SUITES
            }
        else:
            sources = source_rows(run.cfg)
        tasks, report, seen_ids, seen_prompts = {}, {}, set(), set()
        for suite in SUITES:
            rows = sorted(sources[suite], key=lambda r: r["instance_id"])
            if len(rows) != len({r["instance_id"] for r in rows}):
                raise ValueError("Duplicate task IDs in benchmark source")
            # Cross-suite duplicates are excluded deterministically before drawing,
            # never because of environment success or model scores.
            eligible = [
                r
                for r in rows
                if r["instance_id"] not in seen_ids
                and " ".join(r["problem_statement"].casefold().split())
                not in seen_prompts
            ]
            if len(eligible) < run.cfg["transfer"]["sample_size"]:
                raise ValueError(f"Insufficient distinct {suite} transfer tasks")
            rng = random.Random(f"transfer-v1:{run.cfg['transfer']['seed']}:{suite}")
            chosen = rng.sample(eligible, run.cfg["transfer"]["sample_size"])
            chosen = [
                {
                    **row,
                    "evaluation_backend": run.cfg["transfer"]["suites"][suite][
                        "backend"
                    ],
                    "benchmark": suite,
                }
                for row in chosen
            ]
            tasks[suite] = chosen
            seen_ids.update(r["instance_id"] for r in chosen)
            seen_prompts.update(
                " ".join(r["problem_statement"].casefold().split()) for r in chosen
            )
            report[suite] = {
                "population": len(rows),
                "eligible": len(eligible),
                "selected": len(chosen),
                "seed": run.cfg["transfer"]["seed"],
                "selection_sha256": digest(chosen),
            }
        for name, value in [("tasks", tasks), ("selection", report)]:
            destination = run.root / "transfer" / f"{name}.json"
            write(destination, value)
            artifacts.append(destination)


def transfer_tasks(run: Run) -> list[dict]:
    path = run.root / "transfer/tasks.json"
    return (
        [task for rows in read(path).values() for task in rows]
        if path.is_file()
        else []
    )


def validate_native_task(task: dict) -> None:
    if task_files(Path(task["task_path"])) != task["task_files"]:
        raise ValueError("Native task environment/verifier/instruction content changed")
