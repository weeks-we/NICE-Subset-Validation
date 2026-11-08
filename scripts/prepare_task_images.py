"""Separate network-enabled preparation/load phases; never runs model experiments."""
import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.core import config, digest, read, write
from nice_validation.deployment import frozen_tasks
from nice_validation.environments import official_identity, prepare_smith_image, verified_image


def publish(client, image, repository):
    image.tag(repository, "prepared")
    for event in client.images.push(repository, tag="prepared", stream=True, decode=True):
        if event.get("error"):
            raise RuntimeError(event["error"])
    # Pull back the published manifest to obtain its registry digest and verify content.
    published = client.images.pull(repository, tag="prepared")
    if published.id != image.id:
        raise ValueError("Published image content differs")
    matches = [d for d in published.attrs.get("RepoDigests", []) if d.startswith(repository + "@sha256:")]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely identify published digest")
    return matches[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["prepare", "load", "check"])
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True, type=Path, help="Read-only frozen run with manifests")
    p.add_argument("--catalog", required=True, type=Path, help="Separate deployment output, never old run artifacts")
    p.add_argument("--registry-prefix", help="Internal registry/project, without scheme")
    p.add_argument("--execute-server", action="store_true")
    a = p.parse_args()
    if not a.execute_server:
        p.error("Explicit --execute-server required; Docker preparation must run on server/build host")
    if a.catalog.resolve().is_relative_to(a.run.resolve()):
        p.error("Catalog must be outside preserved run")
    cfg, tasks = config(a.config), frozen_tasks(a.run)
    if cfg.get("transfer", {}).get("enabled"):
        p.error("This deployment supports the current Verified/Smith experiment only")
    identity = official_identity(cfg)
    import docker
    client = docker.from_env()
    try:
        if a.phase == "prepare":
            prefix = (a.registry_prefix or "").rstrip("/")
            if not re.fullmatch(r"[a-z0-9][a-z0-9.\-:/]*", prefix) or "://" in prefix:
                p.error("Supply a valid internal --registry-prefix")
            catalog = read(a.catalog) if a.catalog.exists() else {"official_harness": identity, "tasks": {}}
            if catalog["official_harness"] != identity:
                raise ValueError("Existing catalog harness differs")
            for task in tasks:
                key = task["instance_id"]
                previous = catalog["tasks"].get(key)
                if previous:
                    # Verify completed preparations; never rebuild them just to continue.
                    if previous["task_sha256"] != digest(task):
                        raise ValueError("Existing catalog task differs: " + key)
                    for role, field in (("agent", "image"), ("grader", "grader_image")):
                        image = client.images.pull(previous["entry"][field])
                        if image.id != previous[role + "_image_id"]:
                            raise ValueError("Previously published content changed")
                    continue
                backend = task.get("evaluation_backend", "swesmith" if "image_name" in task else "swebench")
                if backend not in {"swesmith", "swebench"}:
                    raise ValueError("Unsupported backend: " + backend)
                name = task["image_name"] if backend == "swesmith" else verified_image(task)
                try:
                    base = client.images.get(name)
                except docker.errors.ImageNotFound:
                    base = client.images.pull(name)
                checkout, agent_image = task["base_commit"], base
                if backend == "swesmith":
                    image_id, checkout = prepare_smith_image(client, task, base)
                    agent_image = client.images.get(image_id)
                slug = digest({"task": task, "harness": identity})[:32]
                grader_ref = publish(client, base, prefix + "/grader-" + slug)
                agent_ref = grader_ref if agent_image.id == base.id else publish(client, agent_image, prefix + "/agent-" + slug)
                entry = {"image": agent_ref, "grader_image": grader_ref, "base_commit": task["base_commit"],
                         "checkout_commit": checkout, "backend": backend, "public_environment_only": True,
                         "official_image_name": name,
                         "shell_setup": "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then source /opt/miniconda3/etc/profile.d/conda.sh; conda activate testbed; fi"}
                catalog["tasks"][key] = {"task_sha256": digest(task), "entry": entry,
                                         "agent_image_id": agent_image.id, "grader_image_id": base.id}
                write(a.catalog, catalog)  # Incremental receipt for interrupted preparation.
        else:
            catalog = read(a.catalog)
            if catalog["official_harness"] != identity:
                raise ValueError("Catalog harness differs")
            if a.phase == "load":
                for task in tasks:
                    record = catalog["tasks"][task["instance_id"]]
                    if record["task_sha256"] != digest(task):
                        raise ValueError("Frozen task mismatch")
                    for role, field in (("agent", "image"), ("grader", "grader_image")):
                        image = client.images.pull(record["entry"][field])
                        if image.id != record[role + "_image_id"]:
                            raise ValueError("Loaded image ID mismatch")
                    client.images.get(record["entry"]["grader_image"]).tag(record["entry"]["official_image_name"])
            from nice_validation.core import sha
            from nice_validation.deployment import catalog_for
            cfg["deployment"] = {"task_catalog": str(a.catalog), "task_catalog_sha256": sha(a.catalog)}
            catalog_for(cfg, tasks, client, identity)
        print(f"{a.phase}: {len(tasks)} frozen tasks; catalog={a.catalog}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
