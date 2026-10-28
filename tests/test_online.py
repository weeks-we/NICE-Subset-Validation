import io
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

from nice_validation.core import Run, config, read, sha, write
from nice_validation.data import FixtureTokenizer, fetch_data, fixture, resumable_rows
from nice_validation.model import fetch_model
from nice_validation.online import (
    configure_server,
    development_task,
    prepare_online,
    resolve_sources,
    run_online,
)


GPU_A = (
    "GPU-" + "a" * 8 + "-" + "a" * 4 + "-" + "a" * 4 + "-" + "a" * 4 + "-" + "a" * 12
)
GPU_B = GPU_A.replace("a", "b")


def records():
    cfg = config("configs/main.yaml")
    return {
        "training": {"id": cfg["data"]["training_id"], "revision": "a" * 40},
        "validation": {"id": cfg["data"]["validation_id"], "revision": "b" * 40},
        "model": {"id": cfg["model"]["id"], "revision": "c" * 40},
        "swebench": {"id": "SWE-bench/SWE-bench", "revision": "d" * 40},
        "swesmith": {"id": "SWE-bench/SWE-smith", "revision": "d" * 40},
        "harbor": {"id": "harbor-framework/harbor", "revision": "d" * 40},
        "live_harness": {"id": "microsoft/SWE-bench-Live", "revision": "d" * 40},
        "verified": {
            "id": cfg["transfer"]["suites"]["verified"]["id"],
            "revision": "e" * 40,
        },
        "pro": {"id": cfg["transfer"]["suites"]["pro"]["id"], "revision": "f" * 40},
        "live": {"id": cfg["transfer"]["suites"]["live"]["id"], "revision": "a" * 40},
        "terminal": {
            "id": cfg["transfer"]["suites"]["terminal"]["id"],
            "revision": "b" * 40,
            "registered_tasks": [{"name": "fixture", "path": "tasks/fixture"}],
        },
    }


def configured(tmp_path):
    with patch("nice_validation.online.resolve_sources", return_value=records()):
        paths = configure_server(
            tmp_path / "storage", GPU_A, GPU_B, tmp_path / "configs"
        )
    cfg = config(paths["server_config"])
    # These acquisition tests isolate main sources; external suites have their
    # own integration tests and must never perform real network calls here.
    cfg["transfer"]["enabled"] = False
    return cfg, paths


def fake_downloads(monkeypatch, training=None, validation=None):
    if training is None:
        training, validation = fixture()
    downloads, loads = [], []
    hub = types.ModuleType("huggingface_hub")

    class API:
        def list_repo_tree(self, repo, **kwargs):
            prefix = (
                "tool"
                if repo.endswith("trajectories")
                else "train" if repo.endswith("SWE-smith") else "test"
            )
            # Deliberately mixed: only the exact requested split is acquired.
            return [
                types.SimpleNamespace(path=f"data/{prefix}-00001.parquet", size=3),
                types.SimpleNamespace(path="data/xml-00000.parquet", size=5),
                types.SimpleNamespace(path=f"data/{prefix}-00000.parquet", size=2),
            ]

    hub.HfApi = API

    def download(repo, filename, **kwargs):
        downloads.append((repo, filename, kwargs))
        return str(Path("fake") / filename)

    hub.hf_hub_download = download
    datasets = types.ModuleType("datasets")

    def load(name, **kwargs):
        loads.append(kwargs)
        return iter(training if kwargs["split"] == "tool" else validation)

    datasets.load_dataset = load
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    return downloads, loads, hub


def test_resolver_pins_all_four_sources_without_weights(monkeypatch):
    _, _, hub = fake_downloads(monkeypatch)

    class API:
        def model_info(self, identifier):
            return types.SimpleNamespace(sha="c" * 40)

        def dataset_info(self, identifier):
            return types.SimpleNamespace(
                sha=("a" if identifier.endswith("trajectories") else "b") * 40
            )

    hub.HfApi = API
    with patch(
        "urllib.request.urlopen",
        side_effect=lambda *a, **k: io.BytesIO(json.dumps({"sha": "d" * 40}).encode()),
    ):
        cfg = config("configs/main.yaml")
        cfg["transfer"]["enabled"] = False
        assert resolve_sources(cfg) == {
            k: records()[k]
            for k in ["training", "validation", "model", "swebench", "swesmith"]
        }


def test_generated_configs_reuse_pins_and_preserve_existing_files(tmp_path):
    cfg, paths = configured(tmp_path)
    assert cfg["execution"] == "server" and cfg["data"]["training_revision"] == "a" * 40
    assert config(paths["service_config"])["model"]["gpu_uuids"] == [GPU_B]
    for value in [
        cfg["output"],
        cfg["data"]["training_path"],
        cfg["data"]["hub_cache"],
        cfg["model"]["path"],
    ]:
        assert Path(value).is_absolute()
    before = Path(paths["server_config"]).read_bytes()
    with patch(
        "nice_validation.online.resolve_sources",
        side_effect=AssertionError("must reuse lock"),
    ):
        configure_server(tmp_path / "storage", GPU_A, GPU_B, tmp_path / "configs")
        with pytest.raises(ValueError, match="overwrite"):
            configure_server(
                tmp_path / "storage", GPU_A, GPU_B, tmp_path / "configs", port=9000
            )
    assert Path(paths["server_config"]).read_bytes() == before


def test_configuration_requires_separate_authorized_gpus(tmp_path):
    with pytest.raises(ValueError, match="different authorized"):
        configure_server(tmp_path, GPU_A, GPU_A, tmp_path)


def test_source_download_is_split_scoped_checksummed_and_idempotent(
    tmp_path, monkeypatch
):
    cfg, _ = configured(tmp_path)
    downloads, loads, _ = fake_downloads(monkeypatch)
    plan = fetch_data(cfg, "download", None, False)
    assert not downloads and plan["training"]["download_bytes"] == 5
    result = fetch_data(cfg, "download", None, True)
    assert len(downloads) == 4 and len(loads) == 2
    assert all("xml" not in f for _, f, _ in downloads)
    assert all(p["cache_dir"] == cfg["data"]["hub_cache"] for _, _, p in downloads)
    assert result["training"]["sha256"] == sha(Path(cfg["data"]["training_path"]))
    assert read(Path(cfg["data"]["source_lock"])) == result
    assert fetch_data(cfg, "download", None, True) == result
    assert len(downloads) == 4  # no duplicate downloads or conversion
    Path(cfg["data"]["training_path"]).write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        fetch_data(cfg, "download", None, True)


def test_source_interruption_resumes_without_duplicate_rows(tmp_path, monkeypatch):
    cfg, _ = configured(tmp_path)
    rows, validation = fixture()
    _, _, _ = fake_downloads(monkeypatch, rows, validation)
    original = sys.modules["datasets"].load_dataset

    def broken(*args, **kwargs):
        yield rows[0]
        raise ConnectionError("simulated network interruption")

    sys.modules["datasets"].load_dataset = broken
    with pytest.raises(ConnectionError):
        fetch_data(cfg, "stream", None, True)
    partial = Path(cfg["data"]["training_path"] + ".partial")
    with partial.open("ab") as handle:
        handle.write(b'{"torn":')
    sys.modules["datasets"].load_dataset = original
    result = fetch_data(cfg, "stream", None, True)
    assert result["training"]["records"] == len(rows)
    actual = Path(cfg["data"]["training_path"]).read_text().splitlines()
    assert len(actual) == len(rows) and json.loads(actual[0]) == rows[0]
    assert list(partial.parent.glob("*.interrupted-*"))


def test_partial_complete_last_line_gets_terminator(tmp_path):
    path = tmp_path / "rows.partial"
    path.write_bytes(b'{"id": 1}')
    assert resumable_rows(path) == 1
    assert path.read_bytes().endswith(b"\n")
    path.write_bytes(b"{bad}\n")
    with pytest.raises(ValueError, match="Corrupt complete"):
        resumable_rows(path)


def test_source_recovers_if_interrupted_between_provenance_and_publish(
    tmp_path, monkeypatch
):
    cfg, _ = configured(tmp_path)
    downloads, _, _ = fake_downloads(monkeypatch)
    replace = Path.replace

    def interrupted(path, target):
        if str(path).endswith("training.jsonl.partial"):
            raise OSError("simulated interruption before source publication")
        return replace(path, target)

    with patch.object(Path, "replace", interrupted):
        with pytest.raises(OSError, match="publication"):
            fetch_data(cfg, "download", None, True)
    target = Path(cfg["data"]["training_path"])
    assert not target.exists()
    assert target.with_suffix(".jsonl.source.json").is_file()
    result = fetch_data(cfg, "download", None, True)
    assert target.is_file() and result["training"]["sha256"] == sha(target)
    assert all("xml" not in f for _, f, _ in downloads)


def test_model_download_resumes_after_config_arrives(tmp_path, monkeypatch):
    cfg, _ = configured(tmp_path)
    _, _, hub = fake_downloads(monkeypatch)
    attempts = []

    def snapshot(identifier, **kwargs):
        attempts.append(kwargs)
        write(Path(kwargs["local_dir"]) / "config.json", {"synthetic_fixture": True})
        if len(attempts) == 1:
            raise ConnectionError("interrupted after metadata")
        (Path(kwargs["local_dir"]) / "fixture.txt").write_text("no model weights")

    hub.snapshot_download = snapshot
    with pytest.raises(ConnectionError):
        fetch_model(cfg)
    assert not (Path(cfg["model"]["path"]) / "model_lock.json").exists()
    fetch_model(cfg)
    assert len(attempts) == 2
    assert read(Path(cfg["model"]["path"]) / "model_lock.json")["revision"] == "c" * 40
    fetch_model(cfg)
    assert len(attempts) == 2


def test_online_prepare_to_processing_and_existing_pipeline(tmp_path, monkeypatch):
    cfg, _ = configured(tmp_path)
    training, validation = fixture()
    validation += [
        {
            **validation[i % 60],
            "instance_id": f"valid__task-{i:04d}",
            "problem_statement": f"Unrelated synthetic validation issue {i}",
        }
        for i in range(60, 500)
    ]
    downloads, _, hub = fake_downloads(monkeypatch, training, validation)

    def snapshot(identifier, **kwargs):
        write(Path(kwargs["local_dir"]) / "config.json", {"fixture": True})

    hub.snapshot_download = snapshot

    def adapter(c):
        # Source data must exist before the run identity is frozen.
        assert Path(c["data"]["source_lock"]).is_file()
        write(Path(c["model"]["adapter_path"]) / "identity.json", {"fixture": True})

    def environments(run):
        assert run.valid("prepare-data")
        assert Path(cfg["agent"]["smoke_task_path"]).exists()
        with run.stage("prepare-environments") as artifacts:
            if artifacts is not None:
                p = run.root / "manifests/environment-fixture.json"
                write(p, {"mode": "synthetic"})
                artifacts.append(p)

    monkeypatch.setattr(
        "nice_validation.preflight.preflight", lambda *a, **k: {"status": "PASS"}
    )
    monkeypatch.setattr("nice_validation.environments.official_identity", lambda c: {})
    monkeypatch.setattr("nice_validation.model.initialize_adapter", adapter)
    monkeypatch.setattr(
        "nice_validation.model.load_tokenizer", lambda c: FixtureTokenizer()
    )
    monkeypatch.setattr(
        "nice_validation.environments.prepare_environments", environments
    )
    run = prepare_online(cfg, download_model=True)
    assert len(read(run.root / "manifests/candidates.json")) == 300
    assert len(read(run.root / "manifests/validation.json")) == 30
    assert run.identity == Run(cfg).identity
    source_hash = sha(Path(cfg["data"]["training_path"]))
    assert prepare_online(cfg, download_model=True).identity == run.identity
    assert sha(Path(cfg["data"]["training_path"])) == source_hash
    assert len(downloads) == 4
    called = []
    monkeypatch.setattr("nice_validation.smoke.smoke", lambda r: called.append("smoke"))
    monkeypatch.setattr(
        "nice_validation.cli.pipeline", lambda r: called.append("pipeline")
    )
    run_online(cfg)
    assert called == ["smoke", "pipeline"]
    called.clear()
    monkeypatch.setattr(
        "nice_validation.preflight.preflight", lambda *a, **k: {"status": "FAIL"}
    )
    with pytest.raises(ValueError, match="Prepared preflight failed"):
        run_online(cfg)
    assert not called
    task = read(Path(cfg["agent"]["smoke_task_path"]))
    task["problem_statement"] = "tampered"
    write(Path(cfg["agent"]["smoke_task_path"]), task)
    with pytest.raises(ValueError, match="differs from frozen"):
        development_task(run)


@pytest.mark.parametrize("function", [prepare_online, run_online])
def test_online_commands_refuse_local_execution(function):
    with pytest.raises(ValueError, match="execution: server"):
        function(config("configs/main.yaml"))


def test_replacement_keeps_development_task_outside_main_validation(tmp_path):
    from nice_validation.environments import replace_validation

    cfg, _ = configured(tmp_path)
    run = Run(cfg)
    manifests = run.root / "manifests"
    old = {"instance_id": "old", "repo": "org/repo"}
    dev = {"instance_id": "dev", "repo": "org/repo"}
    other = {"instance_id": "reserve", "repo": "org/repo"}
    write(Path(cfg["agent"]["smoke_task_path"]), dev)
    with run.stage("prepare-data") as artifacts:
        for name, value in [
            ("validation", [old]),
            ("validation_reserve", [dev, other]),
            ("checksums", {}),
        ]:
            path = manifests / (name + ".json")
            write(path, value)
            artifacts.append(path)
    write(
        manifests / "environment_preflight.json",
        {"failures": [{"task_id": "old", "reason": "fixture image failure"}]},
    )
    result = replace_validation(run, "old")
    assert result["replacement"]["instance_id"] == "reserve"
    assert read(manifests / "validation_reserve.json") == [dev]
    assert run.valid("prepare-data")
