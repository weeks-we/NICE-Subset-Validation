"""Use real Arrow, datasets and HF metadata classes with transport isolated."""

from datetime import datetime
import json
from pathlib import Path
import uuid

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("datasets")
hub = pytest.importorskip("huggingface_hub")
from huggingface_hub.hf_api import RepoFile

from nice_validation.acquire import hf_split
from nice_validation.core import read_jsonl, sha
from conftest import TEST_RUNS


def test_actual_parquet_loader_preserves_fields_and_recovers_a_torn_export(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    source = tmp_path / "tool.parquet"
    rows = [
        {
            "instance_id": "owner__repo.task-" + str(i),
            "traj_id": "trajectory-" + str(i),
            "messages": json.dumps(
                [
                    {"role": "user", "content": "fix"},
                    {"role": "assistant", "content": "done"},
                ]
            ),
            "resolved": True,
            "created_at": datetime(2025, 1, 2, 3, 4, i),
            "optional": None,
        }
        for i in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    metadata = RepoFile(
        path="data/tool-00000-of-00001.parquet",
        size=source.stat().st_size,
        oid="b" * 40,
    )

    class API:
        def list_repo_tree(self, *args, **kwargs):
            assert kwargs["revision"] == "a" * 40 and kwargs["recursive"] is True
            return [metadata]

    monkeypatch.setattr(hub, "HfApi", API)
    monkeypatch.setattr(hub, "hf_hub_download", lambda *args, **kwargs: str(source))
    # datasets includes the cache's absolute path in a lock filename. A short,
    # unique workspace path also supports Windows without long-path support.
    caches = {
        "datasets_cache": str((TEST_RUNS / ("p" + uuid.uuid4().hex[:6])).resolve())
    }
    first = tmp_path / "export.jsonl"
    result = hf_split(
        "offline-fixture/unit", "a" * 40, "tool", first, "download", True, caches
    )
    values = read_jsonl(first)
    assert result["records"] == 3 and result["sha256"] == sha(first)
    assert values[0]["created_at"] == "2025-01-02T03:04:00"
    assert values[0]["optional"] is None and values[0]["resolved"] is True
    assert json.loads(values[0]["messages"])[1]["content"] == "done"

    resumed = tmp_path / "resumed.jsonl"
    partial = resumed.with_suffix(".jsonl.partial")
    partial.write_bytes(first.read_bytes().splitlines(keepends=True)[0] + b'{"torn":')
    from nice_validation.core import write

    plan = hf_split(
        "offline-fixture/unit", "a" * 40, "tool", resumed, "download", False, caches
    )
    write(partial.with_suffix(".partial.source.json"), plan)
    second = hf_split(
        "offline-fixture/unit", "a" * 40, "tool", resumed, "download", True, caches
    )
    assert second["records"] == 3 and first.read_bytes() == resumed.read_bytes()
    assert list(tmp_path.glob("*.interrupted-*"))
