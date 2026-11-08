import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from nice_validation.core import digest, sha, write
from nice_validation.deployment import catalog_for, frozen_tasks, local_inputs


def setup_catalog(tmp_path):
    task = {"instance_id": "repo-1", "base_commit": "a" * 40}
    ref = "internal.example/tasks/a@sha256:" + "b" * 64
    entry = {"image": ref, "grader_image": ref, "base_commit": task["base_commit"],
             "public_environment_only": True, "official_image_name": "official:a"}
    record = {"task_sha256": digest(task), "entry": entry,
              "agent_image_id": "sha256:content", "grader_image_id": "sha256:content"}
    path = tmp_path / "catalog.json"
    value = {"official_harness": {"commit": "c" * 40}, "tasks": {"repo-1": record}}
    write(path, value)
    cfg = {"deployment": {"task_catalog": str(path), "task_catalog_sha256": sha(path)}}
    calls = []

    def get(name):
        calls.append(name)
        return SimpleNamespace(id="sha256:content")

    # No pull/create API provided: checks cannot accidentally acquire resources.
    client = SimpleNamespace(images=SimpleNamespace(get=get))
    return cfg, task, value, client, calls


def test_prebuilt_accepts_exact_task_harness_and_images_without_download(tmp_path):
    cfg, task, value, client, calls = setup_catalog(tmp_path)
    assert catalog_for(cfg, [task], client, value["official_harness"])["repo-1"]["public_environment_only"]
    assert len(calls) == 3


@pytest.mark.parametrize("failure", ["checksum", "task", "harness", "missing", "image_id", "mutable", "isolation"])
def test_prebuilt_rejects_wrong_or_missing_resources(tmp_path, failure):
    cfg, task, value, client, calls = setup_catalog(tmp_path)
    identity = copy.deepcopy(value["official_harness"])
    if failure == "checksum":
        cfg["deployment"]["task_catalog_sha256"] = "0" * 64
    elif failure == "task":
        task["base_commit"] = "d" * 40
    elif failure == "harness":
        identity["commit"] = "e" * 40
    elif failure == "missing":
        def missing(name):
            raise RuntimeError("image unavailable: " + name)
        client.images.get = missing
    else:
        record = value["tasks"]["repo-1"]
        if failure == "image_id":
            record["agent_image_id"] = "wrong"
        elif failure == "mutable":
            record["entry"]["image"] = "internal/task:latest"
        else:
            record["entry"]["public_environment_only"] = False
        path = Path(cfg["deployment"]["task_catalog"])
        write(path, value)
        cfg["deployment"]["task_catalog_sha256"] = sha(path)
    with pytest.raises(ValueError):
        catalog_for(cfg, [task], client, identity)


def test_frozen_tasks_reads_all_scopes_and_rejects_overlap(tmp_path):
    folder = tmp_path / "manifests"
    for name in ("validation", "sft_validation", "benchmark", "development_task"):
        task = {"instance_id": name}
        write(folder / (name + ".json"), task if name == "development_task" else [task])
    assert len(frozen_tasks(tmp_path)) == 4
    write(folder / "benchmark.json", [{"instance_id": "validation"}])
    with pytest.raises(ValueError, match="overlap"):
        frozen_tasks(tmp_path)


def test_offline_missing_inputs_are_reported_together(tmp_path):
    cfg = {"model": {"path": str(tmp_path / "base")},
           "data": {k + "_path": str(tmp_path / (k + ".jsonl")) for k in ("training", "validation", "smith")}}
    with pytest.raises(ValueError) as exc:
        local_inputs(cfg)
    for name in ("model_lock.json", "training.jsonl", "validation.jsonl", "smith.jsonl"):
        assert name in str(exc.value)


def test_publish_uses_registry_digest_and_checks_pushed_content():
    from scripts.prepare_task_images import publish
    image = SimpleNamespace(id="sha256:content", tag=lambda *args: None)
    published = SimpleNamespace(id=image.id, attrs={"RepoDigests": ["internal/a@sha256:" + "a" * 64]})
    api = SimpleNamespace(push=lambda *args, **kw: iter([{"status": "done"}]),
                          pull=lambda *args, **kw: published)
    client = SimpleNamespace(images=api)
    assert publish(client, image, "internal/a") == published.attrs["RepoDigests"][0]
    published.id = "sha256:other"
    with pytest.raises(ValueError, match="content differs"):
        publish(client, image, "internal/a")


def test_publish_does_not_accept_registry_push_error():
    from scripts.prepare_task_images import publish
    image = SimpleNamespace(id="sha256:content", tag=lambda *args: None)
    api = SimpleNamespace(push=lambda *args, **kw: iter([{"error": "unauthorized"}]))
    with pytest.raises(RuntimeError, match="unauthorized"):
        publish(SimpleNamespace(images=api), image, "internal/a")
