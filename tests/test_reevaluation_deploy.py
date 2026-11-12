import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nice_validation.core import sha, write
from nice_validation.reevaluation_deploy import independent_paths, inventory, network_environment, probe_network


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://127.0.0.1:8000", "http://0.0.0.0:8000", "https://user:secret@service", "http://service/v1", "http://service?token=x", "file:///tmp/model"])
def test_invalid_network_endpoint_rejected(url):
    with pytest.raises(ValueError):
        network_environment(url)


def test_local_service_bypasses_corporate_proxy():
    env = network_environment("http://inference.company.internal:8000")
    assert env["HTTP_PROXY"] == env["http_proxy"] == env["ALL_PROXY"] == ""
    assert "inference.company.internal" in env["NO_PROXY"].split(",")
    assert env["no_proxy"] == env["NO_PROXY"]


@pytest.mark.parametrize("name", ["host", "none", "bridge", ""])
def test_probe_refuses_unrestricted_or_disconnected_defaults(name):
    with pytest.raises(ValueError):
        probe_network("sha256:existing", name, "http://10.1.1.2:8000")


def test_probe_runs_inside_container_without_pulling_or_mounting_old_data():
    calls = []
    def execute(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=json.dumps({"status": "PASS"}))
    assert probe_network("sha256:existing", "nice-private", "http://10.1.1.2:8000", execute)["status"] == "PASS"
    cmd = calls[0]
    assert "--pull=never" in cmd and "nice-private" in cmd
    assert "--privileged" not in cmd and "-v" not in cmd
    assert cmd[cmd.index("--entrypoint") + 1] == "python"


def test_old_run_cannot_be_overwritten(tmp_path):
    for output in [tmp_path, tmp_path / "child"]:
        with pytest.raises(ValueError):
            independent_paths(tmp_path, output)


def fixture_run(tmp_path):
    old = tmp_path / "old"
    base = tmp_path / "base"
    write(base / "model_lock.json", {"id": "Qwen", "revision": "a" * 40, "files": {"config.json": {"sha256": "bad"}}})
    initial = tmp_path / "initial"
    selected = ["low_lr1_p100", "low_lr2_p025", "high_lr1_p050", "high_lr2_p100"]
    write(old / "config/snapshot.json", {"identity": "original", "config": {"model": {
        "path": str(base), "id": "Qwen", "revision": "a" * 40, "adapter_path": str(initial)}}})
    write(old / "selection.json", {"selected": [{"model_id": x} for x in selected]})
    for folder in [initial, *[old / "training" / name / "adapter" for name in selected]]:
        write(folder / "adapter_config.json", {"r": 16})
        (folder / "adapter_model.safetensors").write_bytes(b"fixture only")
    for filename, image_scope, count in [("benchmark", "benchmark", 100), ("validation", "nice", 20)]:
        tasks = [{"instance_id": filename + str(i), "base_commit": "b" * 40} for i in range(count)]
        write(old / "manifests" / (filename + ".json"), tasks)
        mapping = {t["instance_id"]: {"image": "sha256:agent", "grader_image": "sha256:grader",
                   "official_image_name": "official:tag", "base_commit": t["base_commit"],
                   "public_environment_only": True} for t in tasks}
        path = old / "environments" / image_scope / "task_images.json"
        write(path, mapping)
        write(path.with_name("task_images.lock.json"), {"image_map_sha256": sha(path)})
    return old, base


def test_inventory_reads_exact_120_tasks_and_preserves_old_evidence(tmp_path):
    old, base = fixture_run(tmp_path)
    write(base / "config.json", {})
    write(base / "model_lock.json", {"id": "Qwen", "revision": "a" * 40,
                                     "files": {"config.json": {"sha256": sha(base / "config.json")}}})
    before = {str(p): sha(p) for p in old.rglob('*') if p.is_file()}
    calls = []
    def inspect(ref):
        calls.append(ref)
        return {"Id": "sha256:agent" if ref == "sha256:agent" else "sha256:grader"}
    result = inventory(old, inspect)
    assert result["status"] == "RESOURCES_PRESENT" and result["formal_episodes"] == 600
    assert len(result["scopes"]["benchmark"]["tasks"]) == 100
    assert len(result["scopes"]["nice_retest"]["tasks"]) == 20
    assert len(calls) == 3  # Shared layers/aliases inspected once, not downloaded.
    assert before == {str(p): sha(p) for p in old.rglob('*') if p.is_file()}


def test_inventory_reports_bad_base_and_missing_images_without_repair(tmp_path):
    old, base = fixture_run(tmp_path)
    def missing(ref):
        raise ValueError("not on this daemon")
    result = inventory(old, missing)
    assert result["status"] == "BLOCKED"
    assert any("base_file" in x for x in result["issues"])
    assert any("image" in x for x in result["issues"])


def test_inventory_rejects_modified_frozen_image_map(tmp_path):
    old, base = fixture_run(tmp_path)
    write(old / "environments/benchmark/task_images.json", {})
    with pytest.raises(ValueError, match="checksum"):
        inventory(old, lambda ref: {"Id": "fixture"})
