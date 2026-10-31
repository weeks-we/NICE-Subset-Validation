"""Fixed-revision, split-scoped JSONL acquisition shared by all HF sources."""

from pathlib import Path

from nice_validation.core import append, read, require_commit, sha, write


def hf_split(
    repo: str,
    revision: str,
    split: str,
    target: Path,
    mode: str,
    execute: bool,
    caches: dict,
    limit: int | None = None,
) -> dict:
    if mode not in {"stream", "download"}:
        raise ValueError("Acquisition mode must be stream or download")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Acquisition limit must be a positive integer or None")
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset
    from nice_validation.data import resumable_rows

    require_commit(revision, repo + ".revision")
    files = HfApi().list_repo_tree(
        repo, repo_type="dataset", revision=revision, recursive=True
    )
    prefix = f"data/{split}-"
    shards = sorted(
        [
            f
            for f in files
            if hasattr(f, "size")
            and f.path.startswith(prefix)
            and f.path.endswith(".parquet")
        ],
        key=lambda f: f.path,
    )
    if not shards:
        raise ValueError(f"No {prefix} parquet shards at frozen revision {revision}")
    entry = {
        "id": repo,
        "revision": revision,
        "split": split,
        "files": [{"path": f.path, "bytes": f.size} for f in shards],
        "download_bytes": sum(f.size for f in shards),
        "limit": limit,
        "selection": (
            "complete source"
            if limit is None
            else "bounded prefix; deterministic within fetched prefix"
        ),
    }
    if not execute:
        return entry
    sidecar = target.with_suffix(target.suffix + ".source.json")
    if target.exists():
        if not sidecar.exists():
            raise ValueError(f"Existing source has no provenance sidecar: {target}")
        previous = read(sidecar)
        if any(
            previous.get(k) != entry.get(k)
            for k in ["id", "revision", "split", "limit", "files"]
        ):
            raise ValueError(f"Existing source identity differs: {target}")
        if previous["sha256"] != sha(target):
            raise ValueError(f"Existing source checksum differs: {target}")
        return previous
    urls = [f"hf://datasets/{repo}@{revision}/{f.path}" for f in shards]
    if mode == "download":
        urls = [
            hf_hub_download(
                repo,
                f.path,
                repo_type="dataset",
                revision=revision,
                cache_dir=caches.get("hub_cache"),
            )
            for f in shards
        ]
    dataset = load_dataset(
        "parquet",
        data_files={split: urls},
        split=split,
        streaming=mode == "stream",
        cache_dir=caches.get("datasets_cache"),
    )
    partial = target.with_suffix(target.suffix + ".partial")
    partial_lock = partial.with_suffix(partial.suffix + ".source.json")
    if partial.exists():
        if not partial_lock.exists() or read(partial_lock) != entry:
            raise ValueError(f"Partial source identity differs: {partial}")
        completed = resumable_rows(partial)
    else:
        write(partial_lock, entry)
        completed = 0
    count = 0
    saved = partial.open(encoding="utf-8") if completed else None
    try:
        import json
        from nice_validation.core import digest

        for row in dataset:
            # Arrow timestamp columns occur in Live metadata; preserve their
            # values as ISO-8601 strings in the canonical JSONL export.
            from datetime import date, datetime

            def encode(value):
                if isinstance(value, (date, datetime)):
                    return value.isoformat()
                raise TypeError(f"Unsupported source value: {type(value).__name__}")

            row = json.loads(json.dumps(row, default=encode))
            count += 1
            if count <= completed:
                if digest(json.loads(saved.readline())) != digest(row):
                    raise ValueError(
                        "Resumable prefix differs from the frozen remote source"
                    )
            else:
                append(partial, row)
            if limit and count >= limit:
                break
    finally:
        if saved:
            saved.close()
    if count < completed:
        raise ValueError(
            "Frozen source has fewer rows than the resumable partial artifact"
        )
    if count == 0:
        raise ValueError(f"Frozen source split is empty: {repo}/{split}")
    entry.update({"records": count, "sha256": sha(partial), "path": str(target)})
    write(sidecar, entry)
    partial.replace(target)
    partial_lock.unlink(missing_ok=True)
    return entry
