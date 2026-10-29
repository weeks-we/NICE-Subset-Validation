"""Deterministic source preparation, leakage checks and assistant-only masks."""

from __future__ import annotations

import json
import random
import re
import uuid
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from nice_validation.core import (
    Run,
    digest,
    read,
    read_jsonl,
    require_commit,
    sha,
    write,
)
from nice_validation.protocol import TOOLS, validate_messages
from nice_validation.schemas import record_identity, task_record, training_record


def proportional(
    rows: list[dict], size: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Largest remainder allocation; stable repository/id ordering before sampling."""
    if size > len(rows):
        raise ValueError(f"Not enough distinct records: {len(rows)} < {size}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["repo"]].append(row)
    exact = {repo: size * len(items) / len(rows) for repo, items in groups.items()}
    quotas = {repo: int(x) for repo, x in exact.items()}
    for repo in sorted(groups, key=lambda r: (-(exact[r] - quotas[r]), r))[
        : size - sum(quotas.values())
    ]:
        quotas[repo] += 1
    selected, reserve = [], []
    for repo in sorted(groups):
        items = sorted(groups[repo], key=lambda r: r.get("data_id", r["instance_id"]))
        random.Random(f"{seed}:{repo}").shuffle(items)
        selected.extend(items[: quotas[repo]])
        reserve.extend(items[quotas[repo] :])
    return selected, reserve


def repo_of(row: dict) -> str:
    if row.get("repo"):
        return row["repo"]
    owner, rest = row["instance_id"].split("__", 1)
    repository = re.sub(r"-\d+.*$", "", rest.split(".", 1)[0])
    return owner + "/" + repository


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def near_prompt(left: str, right: str, threshold: float) -> bool:
    if min(len(left), len(right)) <= 100:
        return False
    # These are upper bounds on SequenceMatcher.ratio, so they do not change
    # the filtering definition and avoid expensive comparisons of unlike sizes.
    if 2 * min(len(left), len(right)) / (len(left) + len(right)) < threshold:
        return False
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    return matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold


def main_task_records(rows: list[dict], backend: str) -> tuple[list[dict], dict]:
    """Remove absent issue statements before drawing the fixed held-out sample."""
    eligible, excluded = [], []
    for row in rows:
        record_identity(row)
        problem = row.get("problem_statement")
        if problem is None or (isinstance(problem, str) and not problem.strip()):
            excluded.append(row["instance_id"])
            continue
        eligible.append(task_record(row, backend))
    return eligible, {
        "raw_read": len(rows),
        "eligible_with_statement": len(eligible),
        "excluded_missing_problem_statement": len(excluded),
        "excluded_task_ids": excluded,
        "selection": "source-schema filtering before sampling; independent of model outcomes",
    }


def render(
    tokenizer, messages: list[dict], thinking: bool = False, generation: bool = False
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=generation,
        enable_thinking=thinking,
    )


def tokenize(tokenizer, messages: list[dict], thinking: bool, max_length: int) -> dict:
    """Mark assistant rendered deltas. Verify every prefix against the final template.

    This version supervises assistant body/tool syntax and end markers, excluding
    assistant headers. Templates that rewrite earlier turns fail closed.
    """
    full = render(tokenizer, messages, thinking)
    encoded = tokenizer(
        full, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )
    ids, offsets = list(encoded["input_ids"]), encoded["offset_mapping"]
    if len(ids) > max_length:
        raise ValueError("context_length_exceeded")
    intervals = []
    tool_intervals = []
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            prefix = render(tokenizer, messages[:i], thinking, True)
            end = render(tokenizer, messages[: i + 1], thinking)
            # Some official Qwen templates insert this empty non-thinking scaffold
            # only in a generation prompt. It is a header, never a target span.
            scaffold = "<think>\n\n</think>\n\n"
            if (
                not thinking
                and not end.startswith(prefix)
                and prefix.endswith(scaffold)
            ):
                prefix = prefix[: -len(scaffold)]
            if not end.startswith(prefix) or not full.startswith(end):
                raise ValueError(
                    "template_not_prefix_stable: implement a reviewed model-specific mask"
                )
            intervals.append((len(prefix), len(end)))
        elif message["role"] == "tool" and (
            i == 0 or messages[i - 1]["role"] != "tool"
        ):
            # Qwen wraps consecutive observations in one user message. Only the
            # complete group boundary is prefix-stable in the full rendering.
            group_end = i + 1
            while group_end < len(messages) and messages[group_end]["role"] == "tool":
                group_end += 1
            before = render(tokenizer, messages[:i], thinking)
            end = render(tokenizer, messages[:group_end], thinking)
            if not full.startswith(before) or not full.startswith(end):
                raise ValueError("template_not_prefix_stable")
            tool_intervals.append((len(before), len(end)))
    mask = [
        i > 0 and any(a <= start < stop <= b for a, b in intervals)
        for i, (start, stop) in enumerate(offsets)
    ]
    if not any(mask):
        raise ValueError("no_supervised_tokens")
    labels = [tok if use else -100 for tok, use in zip(ids, mask)]
    return {
        "input_ids": ids,
        "labels": labels,
        "supervised_tokens": sum(mask),
        "input_tokens": len(ids),
        "tool_tokens": sum(
            any(a <= x < y <= b for a, b in tool_intervals) for x, y in offsets
        ),
        "max_context": len(ids),
        "estimated": False,
    }


class FixtureTokenizer:
    """Character tokenizer for explicit mock fixtures only."""

    is_fast = True

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        text = "".join(
            f"<{m['role']}>"
            + (m.get("content") or "")
            + (
                json.dumps(m["tool_calls"], sort_keys=True)
                if m.get("tool_calls")
                else ""
            )
            + "</turn>"
            for m in messages
        )
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) % 127 + 1 for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def fixture() -> tuple[list[dict], list[dict]]:
    candidates, validation = [], []
    for i in range(390):
        repo = ["alpha/core", "beta/lib", "gamma/utils"][i % 3]
        messages = [
            {"role": "user", "content": f"Synthetic training task number {i}."},
            {"role": "assistant", "content": "Fix " + "x" * (20 + i % 35)},
        ]
        candidates.append(
            {
                "instance_id": f"train__task-{i:04d}",
                "traj_id": f"trajectory-{i:04d}",
                "resolved": True,
                "repo": repo,
                "messages": messages,
                "model": "synthetic_fixture",
            }
        )
    for i in range(60):
        validation.append(
            {
                "instance_id": f"valid__task-{i:04d}",
                "repo": ["alpha/core", "beta/lib", "gamma/utils"][i % 3],
                "problem_statement": f"Independent synthetic validation problem {i}.",
                "base_commit": "fixture",
            }
        )
    return candidates, validation


def prepare(run: Run) -> None:
    from nice_validation.transfer_sources import (
        prepare_transfer,
        transfer_tasks,
        task_identities,
    )

    prepare_transfer(run)
    with run.stage("prepare-data") as out:
        if out is None:
            return
        cfg, root = run.cfg, run.root / "manifests"
        if cfg["mode"] == "mock":
            training, validation = fixture()
            tokenizer = FixtureTokenizer()
            lock = {
                "mode": "mock",
                "status": "synthetic_fixture",
                "training_revision": None,
                "validation_revision": None,
            }
        else:
            from nice_validation.model import load_tokenizer

            for key in ["training_revision", "validation_revision"]:
                require_commit(cfg["data"][key], key)
            tokenizer = load_tokenizer(cfg)
            training = read_jsonl(Path(cfg["data"]["training_path"]))
            validation = read_jsonl(Path(cfg["data"]["validation_path"]))
            lock = read(Path(cfg["data"]["source_lock"]))
            for key in ["training", "validation"]:
                if lock[key]["revision"] != cfg["data"][key + "_revision"]:
                    raise ValueError("Source revision does not match config")
                if lock[key]["sha256"] != sha(Path(cfg["data"][key + "_path"])):
                    raise ValueError("Source checksum mismatch")
            if len(validation) < 31:
                raise ValueError(
                    "Main task source must contain at least 31 distinct tasks"
                )
            validation, task_report = main_task_records(
                validation, cfg["data"].get("validation_kind", "swesmith")
            )
            task_report_path = root / "main_task_filter_report.json"
            write(task_report_path, task_report)
            out.append(task_report_path)
        if len({r["instance_id"] for r in validation}) != len(validation):
            raise ValueError("Duplicate validation task ids")
        validation = [
            {
                **row,
                "repo": repo_of(row),
                "base_commit": row.get("base_commit") or row["instance_id"],
                "evaluation_backend": cfg["data"].get("validation_kind", "swesmith"),
            }
            for row in validation
        ]
        if len(validation) < 31:
            raise ValueError(
                "Not enough main tasks after excluding transfer overlap; need 30 validation plus one development task"
            )
        external = transfer_tasks(run)
        external_ids = {
            identity for row in external for identity in task_identities(row)
        }
        external_prompts = [normalized(row["problem_statement"]) for row in external]
        validation = [
            row
            for row in validation
            if not task_identities(row).intersection(external_ids)
            and not any(
                normalized(row["problem_statement"]) == p
                or near_prompt(
                    normalized(row["problem_statement"]),
                    p,
                    cfg["data"]["duplicate_similarity"],
                )
                for p in external_prompts
            )
        ]
        if len(validation) < 31:
            raise ValueError(
                "Not enough main tasks after excluding transfer overlap; need 30 validation plus one development task"
            )
        selected, reserve = proportional(validation, 30, 42)
        development = max(reserve, key=lambda row: row["instance_id"])
        reserve = [
            row for row in reserve if row["instance_id"] != development["instance_id"]
        ]
        heldout = selected + [development] + external
        validation_ids = {
            identity for row in heldout for identity in task_identities(row)
        }
        problems = [normalized(row["problem_statement"]) for row in heldout]
        filtered, counts, mapping = [], Counter(), []
        tools_seen = Counter()
        for row in sorted(
            training,
            key=lambda r: (str(r.get("instance_id", "")), str(r.get("traj_id", ""))),
        ):
            try:
                training_record(row)
                messages = validate_messages(row["messages"])
                for msg in messages:
                    for call in msg.get("tool_calls", []):
                        tools_seen[call["function"]["name"]] += 1
                if task_identities(row).intersection(validation_ids):
                    raise ValueError("cross_set_identity_or_family")
                prompt = normalized(
                    "\n".join(
                        m.get("content") or "" for m in messages if m["role"] == "user"
                    )
                )
                if any(
                    prompt == problem
                    or near_prompt(prompt, problem, cfg["data"]["duplicate_similarity"])
                    for problem in problems
                ):
                    raise ValueError("cross_set_exact_or_near_prompt")
                tokens = tokenize(
                    tokenizer,
                    messages,
                    cfg["model"]["thinking"],
                    cfg["data"]["max_length"],
                )
                data_id = digest(
                    {
                        "source_id": row["instance_id"],
                        "traj_id": row["traj_id"],
                        "messages": messages,
                    }
                )
                item = {
                    "data_id": data_id,
                    "instance_id": row["instance_id"],
                    "source_id": row["instance_id"],
                    "traj_id": row["traj_id"],
                    "repo": repo_of(row),
                    "messages": messages,
                    **tokens,
                    "source_sha256": digest(row),
                    "resolved_evidence": "source_resolved_true_not_reverified",
                    "source_model": row.get("model"),
                    "source_metadata": {
                        k: v for k, v in row.items() if k not in {"messages", "patch"}
                    },
                    "mode": cfg["mode"],
                    "raw_messages": row["messages"],
                    "conversion": "text-blocks-concatenated-v1; object-arguments; original-tool-semantics; missing-return-ID-inferred-only-for-one-outstanding-call",
                }
                filtered.append(item)
                mapping.append(
                    {
                        "data_id": data_id,
                        "source_sha256": digest(row),
                        "instance_id": row["instance_id"],
                        "traj_id": row["traj_id"],
                    }
                )
            except (ValueError, KeyError, TypeError) as exc:
                counts[str(exc)] += 1
        unique = {}
        for row in filtered:
            if row["instance_id"] in unique:
                counts["duplicate_instance"] += 1
            else:
                unique[row["instance_id"]] = row
        report = {
            "raw_read": len(training),
            "eligible_trajectories": len(filtered),
            "unique_tasks": len(unique),
            "filtered_reasons": dict(counts),
            "tools_seen": dict(tools_seen),
            "leakage_scope": "held-out SWE-smith validation/development plus frozen transfer tasks: IDs, explicit source families and near prompts; no pretraining contamination guarantee",
            "unknown_family_metadata": sum(
                not row.get("source_instance_id") for row in training
            ),
        }
        write(root / "filter_report.json", report)
        if len(unique) < 300:
            raise ValueError(
                f"Only {len(unique)} eligible unique tasks; need 300. See {root / 'filter_report.json'}"
            )
        pool, _ = proportional(list(unique.values()), 300, 42)
        for name, value in [
            ("source_lock", lock),
            ("candidates", pool),
            ("validation", selected),
            ("validation_reserve", reserve),
            ("development_task", development),
            ("mapping", mapping),
            ("protocol", {"tools": TOOLS, "version": cfg["agent"]["protocol"]}),
        ]:
            path = root / f"{name}.json"
            write(path, value)
            out.append(path)
        out.append(root / "filter_report.json")
        write(root / "checksums.json", {p.name: sha(p) for p in out})
        out.append(root / "checksums.json")


def resumable_rows(path: Path) -> int:
    """Count complete JSONL rows, retaining any torn final write before recovery."""
    count = 0
    with path.open("rb+") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                return count
            try:
                json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                if line.endswith(b"\n"):
                    raise ValueError(
                        "Corrupt complete row in resumable source"
                    ) from exc
                evidence = path.with_name(
                    path.name + ".interrupted-" + uuid.uuid4().hex
                )
                evidence.write_bytes(line)
                handle.seek(start)
                handle.truncate()
                return count
            count += 1
            if not line.endswith(b"\n"):
                handle.write(b"\n")


def fetch_data(cfg: dict, mode: str, limit: int | None, execute: bool) -> dict:
    """Acquire training tool trajectories and the configured main task split."""
    from nice_validation.acquire import hf_split

    info = {}
    for kind, split in [
        ("training", "tool"),
        ("validation", cfg["data"].get("validation_split", "train")),
    ]:
        info[kind] = hf_split(
            cfg["data"][kind + "_id"],
            cfg["data"][kind + "_revision"],
            split,
            Path(cfg["data"][kind + "_path"]),
            mode,
            execute,
            cfg["data"],
            limit if kind == "training" else None,
        )
    if execute:
        write(Path(cfg["data"]["source_lock"]), info)
    return info
