"""Old storage relocation and complete-source reuse without touching old evidence."""
from pathlib import Path
from unittest.mock import patch

import pytest

from nice_validation.core import config, read, sha, write
from nice_validation.experiment_online import configure, acquire
from test_experiment import old_storage


def pool():
    return config("configs/mock.yaml")["resources"]["gpu_uuids"]


def test_relocated_model_is_found_under_explicit_old_storage(tmp_path):
    old, model = old_storage(tmp_path)
    snapshot = old / "runs/main/config/snapshot.json"
    value = read(snapshot)
    value["config"]["model"]["path"] = str(tmp_path / "unavailable-old-mount/models/frozen-base")
    write(snapshot, value)
    before = {str(p): sha(p) for p in old.rglob("*") if p.is_file()}
    paths = configure(tmp_path / "new", pool(), tmp_path / "cfg", reuse_from=old)
    assert config(paths["server_config"])["model"]["path"] == str(model.resolve())
    assert before == {str(p): sha(p) for p in old.rglob("*") if p.is_file()}


def test_relative_and_relocated_raw_paths_resolve_within_old_storage(tmp_path):
    old, model = old_storage(tmp_path)
    for name in ["source_lock.json", "transfer_sources.json"]:
        path = old / "data" / name
        entries = read(path)
        for kind, row in entries.items():
            row["path"] = ("data/" + Path(row["path"]).name if kind == "training"
                           else str(tmp_path / "unavailable-old-mount/data" / Path(row["path"]).name))
        write(path, entries)
    before = {str(p): sha(p) for p in old.rglob("*") if p.is_file()}
    paths = configure(tmp_path / "new", pool(), tmp_path / "cfg", reuse_from=old)
    cfg = config(paths["server_config"])
    acquire(cfg)
    assert all(Path(cfg["data"][k + "_path"]).is_relative_to(old) for k in ["training", "validation", "smith"])
    assert before == {str(p): sha(p) for p in old.rglob("*") if p.is_file()}


def test_prefix_only_source_downloads_full_same_revision_into_new_storage(tmp_path):
    old, model = old_storage(tmp_path)
    source = old / "data/transfer_sources.json"
    entry = read(source)
    entry["verified"]["limit"] = 1
    write(source, entry)
    sidecar = old / "data/verified.jsonl.source.json"
    write(sidecar, {**read(sidecar), "limit": 1})
    before = {str(p): sha(p) for p in old.rglob("*") if p.is_file()}
    storage = tmp_path / "new"
    paths = configure(storage, pool(), tmp_path / "cfg", reuse_from=old)
    cfg = config(paths["server_config"])
    assert cfg["data"]["validation_path"] == str(storage / "data/validation.jsonl")
    calls = []
    def fetch(identifier, revision, split, target, mode, execute, caches):
        calls.append((identifier, revision, split))
        assert target == storage / "data/validation.jsonl" and mode == "download" and execute
        target.write_text('{"fixture":true}\n', encoding="utf-8")
        result = {"id": identifier, "revision": revision, "split": split, "limit": None,
                  "records": 1, "sha256": sha(target), "path": str(target), "files": []}
        write(target.with_suffix(".jsonl.source.json"), result)
        return result
    with patch("nice_validation.acquire.hf_split", fetch):
        acquire(cfg)
    assert calls == [("SWE-bench/SWE-bench_Verified", "e" * 40, "test")]
    assert before == {str(p): sha(p) for p in old.rglob("*") if p.is_file()}


def test_ambiguous_relocated_models_require_explicit_path(tmp_path):
    import shutil
    old, model = old_storage(tmp_path)
    snapshot = old / "runs/main/config/snapshot.json"
    write(snapshot, {"config": {"model": {"path": str(tmp_path / "missing")}}})
    shutil.copytree(model, old / "models/other-base")
    with pytest.raises(ValueError, match="Ambiguous.*model"):
        configure(tmp_path / "new", pool(), tmp_path / "cfg", reuse_from=old)
    paths = configure(tmp_path / "new", pool(), tmp_path / "cfg", reuse_from=old, model_path=model)
    assert config(paths["server_config"])["model"]["path"] == str(model.resolve())


@pytest.mark.parametrize("problem", ["valid", "missing", "corrupt", "wrong_id", "prefix", "missing_revision"])
def test_prepared_preflight_checks_the_smith_task_source(tmp_path, monkeypatch, problem):
    import types
    import urllib.error
    import nice_validation.preflight as module
    old, model = old_storage(tmp_path)
    paths = configure(tmp_path / "new", pool(), tmp_path / "cfg", reuse_from=old)
    cfg = config(paths["server_config"])
    acquire(cfg)
    task = Path(cfg["data"]["smith_path"])
    if problem == "missing":
        task.unlink()
    elif problem == "corrupt":
        task.write_text("changed", encoding="utf-8")
    elif problem in {"wrong_id", "prefix"}:
        path = Path(cfg["data"]["source_lock"])
        lock = read(path)
        lock["smith"]["id" if problem == "wrong_id" else "limit"] = "other/repo" if problem == "wrong_id" else 1
        write(path, lock)
    elif problem == "missing_revision":
        cfg["data"]["smith_revision"] = None
    monkeypatch.setattr(module, "doctor", lambda c: {"gpu": {"returncode": 0, "stdout": " ".join(pool())}, "docker": {"returncode": 0}})
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr("nice_validation.compatibility.harness_api", lambda c: {})
    monkeypatch.setattr("nice_validation.compatibility.model_loader_api", lambda *a: {})
    class Offline:
        def open(self, *a, **k):
            raise urllib.error.URLError("unit test: no services")
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *a: Offline())
    checks = {r["check"]: r["status"] for r in module.preflight(cfg, prepared=True)["checks"]}
    assert checks["data.smith_revision"] == ("FAIL" if problem == "missing_revision" else "PASS")
    assert checks["artifact:smith_path"] == ("FAIL" if problem == "missing" else "PASS")
    assert checks["source_identity:smith"] == ("PASS" if problem == "valid" else "FAIL")
