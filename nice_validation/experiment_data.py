"""Three task splits frozen before rollout, with task/family/prompt isolation."""
from collections import Counter
from pathlib import Path

from nice_validation.core import read, read_jsonl, write, sha, digest, require_commit
from nice_validation.data import (fixture, FixtureTokenizer, proportional, repo_of,
                                  normalized, near_prompt, tokenize, main_task_records)
from nice_validation.protocol import validate_messages, TOOLS
from nice_validation.schemas import training_record
from nice_validation.transfer_sources import task_identities


def issue(row):
    return normalized(row["problem_statement"])


def source_families(*collections):
    """Known family closure uses both raw trajectories and corresponding tasks."""
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        trail = []
        while parents[value] != value:
            trail.append(value)
            value = parents[value]
        for node in trail:
            parents[node] = value
        return value
    for rows in collections:
        for row in rows:
            identifier = row.get("instance_id")
            if not isinstance(identifier, str) or not identifier:
                continue
            find(identifier)
            for key in ["source_instance_id", "source_family_id"]:
                source = row.get(key)
                if isinstance(source, str) and source:
                    left, right = find(identifier), find(source)
                    parents[max(left, right)] = min(left, right)
    return {identifier: find(identifier) for identifier in list(parents)}


def overlaps(row, held, threshold, prompt=None):
    ids = task_identities(row)
    prompt = prompt if prompt is not None else issue(row)
    return any(ids.intersection(task_identities(other)) or prompt == issue(other)
               or near_prompt(prompt, issue(other), threshold) for other in held)


def mock_sources(cfg):
    e = cfg["experiment"]
    training, verified = fixture(e["candidates"] + e["sft_validation"] + 80,
                                 e["validation"] + e["benchmark"] + 30)
    smith = [{"instance_id": row["instance_id"], "repo": row["repo"],
              "problem_statement": row["messages"][0]["content"],
              "base_commit": "fixture", "image_name": "synthetic",
              "evaluation_backend": "swesmith"} for row in training]
    verified = [{**row, "evaluation_backend": "swebench"} for row in verified]
    return training, verified, smith


def raw_sources(cfg):
    if cfg["mode"] == "mock":
        return (*mock_sources(cfg), FixtureTokenizer(), {"mode": "mock"})
    from nice_validation.model import load_tokenizer
    lock = read(Path(cfg["data"]["source_lock"]))
    rows = []
    for kind in ["training", "validation", "smith"]:
        path = Path(cfg["data"][kind + "_path"])
        require_commit(cfg["data"][kind + "_revision"], kind)
        entry = lock[kind]
        if entry["revision"] != cfg["data"][kind + "_revision"] or entry["sha256"] != sha(path):
            raise ValueError(f"Frozen {kind} source checksum/revision mismatch")
        rows.append(read_jsonl(path))
    rows[1], _ = main_task_records(rows[1], "swebench")
    rows[2], _ = main_task_records(rows[2], "swesmith")
    rows[1] = [{**r, "evaluation_backend": "swebench"} for r in rows[1]]
    rows[2] = [{**r, "evaluation_backend": "swesmith"} for r in rows[2]]
    return (*rows, load_tokenizer(cfg), lock)


def prepare(run):
    with run.stage("prepare-data") as artifacts:
        if artifacts is None:
            return
        cfg, e, folder = run.cfg, run.cfg["experiment"], run.root / "manifests"
        training, verified, smith, tokenizer, lock = raw_sources(cfg)
        for name, rows in [("Verified", verified), ("SWE-smith tasks", smith)]:
            if len({r["instance_id"] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate task IDs in {name}")
        families = source_families(training, verified, smith)
        verified = [{**r, "repo": repo_of(r), "source_family_id": families[r["instance_id"]]} for r in verified]
        smith = [{**r, "source_family_id": families[r["instance_id"]]} for r in smith]
        threshold = cfg["data"]["duplicate_similarity"]
        nice, remainder = proportional(verified, e["validation"], e["split_seed"])
        remainder = [r for r in remainder if not overlaps(r, nice, threshold)]
        benchmark, _ = proportional(remainder, e["benchmark"], e["split_seed"] + 1)
        # Freeze the two Verified subsets before processing any training record.
        held = nice + benchmark
        smith = [{**r, "repo": repo_of(r), "base_commit": r.get("base_commit") or r["instance_id"]}
                 for r in smith if not overlaps(r, held, threshold)]
        smith_by_id = {r["instance_id"]: r for r in smith}
        filtered, reasons, seen, mapping = [], Counter(), set(), []
        for row in sorted(training, key=lambda r: (str(r.get("instance_id", "")), str(r.get("traj_id", "")))):
            try:
                training_record(row)
                messages = validate_messages(row["messages"])
                text = normalized("\n".join(m.get("content") or "" for m in messages if m["role"] == "user"))
                if row["instance_id"] not in smith_by_id:
                    raise ValueError("missing_corresponding_independent_smith_task")
                if overlaps({**row, "source_family_id": families[row["instance_id"]]}, held, threshold, text):
                    raise ValueError("verified_identity_family_or_prompt_overlap")
                if row["instance_id"] in seen:
                    raise ValueError("duplicate_instance")
                tokens = tokenize(tokenizer, messages, cfg["model"]["thinking"], cfg["data"]["max_length"])
                item = {"data_id": digest({"source_id": row["instance_id"], "traj_id": row["traj_id"], "messages": messages}),
                        "instance_id": row["instance_id"], "source_id": row["instance_id"],
                        "traj_id": row["traj_id"], "repo": repo_of(row), "messages": messages,
                        **tokens, "source_sha256": digest(row), "source_model": row.get("model"),
                        "source_instance_id": row.get("source_instance_id"),
                        "source_family_id": families[row["instance_id"]],
                        "source_metadata": {k: v for k, v in row.items() if k not in {"messages", "patch"}},
                        "mode": cfg["mode"], "raw_messages": row["messages"],
                        "resolved_evidence": "source_resolved_true_not_reverified"}
                filtered.append(item)
                mapping.append({k: item[k] for k in ["data_id", "instance_id", "traj_id", "source_sha256"]})
                seen.add(row["instance_id"])
            except (ValueError, KeyError, TypeError) as exc:
                reasons[str(exc)] += 1
        pool, unused = proportional(filtered, e["candidates"], e["split_seed"])
        pool_tasks = [smith_by_id[r["instance_id"]] for r in pool]
        # Candidates, not just selected low/high members, are excluded from selection validation.
        unused = [r for r in unused if not overlaps(smith_by_id[r["instance_id"]], pool_tasks, threshold)]
        validation_rows, unused = proportional(unused, e["sft_validation"], e["split_seed"] + 2)
        sft = [smith_by_id[r["instance_id"]] for r in validation_rows]
        devs = [smith_by_id[r["instance_id"]] for r in unused
                if not overlaps(smith_by_id[r["instance_id"]], pool_tasks + sft, threshold)]
        if not devs:
            raise ValueError("No independent development task remains")
        development = sorted(devs, key=lambda r: r["instance_id"])[-1]
        # Validate prompt leakage using actual training messages as well as task metadata.
        for row in pool:
            prompt = normalized("\n".join(m.get("content") or "" for m in row["messages"] if m["role"] == "user"))
            if overlaps(row, held + sft + [development], threshold, prompt):
                raise ValueError("Candidate prompt/family overlaps a held-out task; do not silently redraw")
        for name, rows in [("NICE", nice), ("benchmark", benchmark), ("SFT validation", sft)]:
            if not rows:
                raise ValueError(f"Empty {name} split")
        values = {"source_lock": lock, "candidates": pool, "validation": nice,
                  "sft_validation": sft, "benchmark": benchmark, "validation_reserve": [],
                  "development_task": development, "mapping": mapping,
                  "protocol": {"tools": TOOLS, "version": cfg["agent"]["protocol"]},
                  "split_ids": {"candidates": [r["instance_id"] for r in pool],
                                "nice": [r["instance_id"] for r in nice],
                                "sft_validation": [r["instance_id"] for r in sft],
                                "benchmark": [r["instance_id"] for r in benchmark],
                                "development": [development["instance_id"]]},
                  "split_family_ids": {"candidates": [r["source_family_id"] for r in pool],
                                       "nice": [r["source_family_id"] for r in nice],
                                       "sft_validation": [r["source_family_id"] for r in sft],
                                       "benchmark": [r["source_family_id"] for r in benchmark],
                                       "development": [development["source_family_id"]]},
                  "filter_report": {"raw_read": len(training), "eligible_unique_tasks": len(filtered),
                                    "filtered_reasons": dict(reasons), "sft_validation_source": "unused trajectory IDs joined to SWE-smith tasks",
                                    "scope": "IDs, transitive explicit source families joined from raw trajectories and tasks, exact/near prompts; no pretraining contamination guarantee"}}
        for name, value in values.items():
            path = folder / (name + ".json")
            write(path, value)
            artifacts.append(path)
        write(folder / "checksums.json", {p.name: sha(p) for p in artifacts})
        artifacts.append(folder / "checksums.json")
