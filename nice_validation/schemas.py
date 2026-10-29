"""Validate source records before sampling or invoking an official harness."""

import json

from nice_validation.core import require_commit


def required_text(row: dict, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing_or_empty_{field}")
    return value


def record_identity(row: dict) -> None:
    if not isinstance(row, dict):
        raise ValueError("source_record_must_be_an_object")
    value = required_text(row, "instance_id")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("instance_id_must_be_a_single_path_component")


def task_record(row: dict, backend: str) -> dict:
    """Keep all native fields; normalize only JSON-encoded test-name lists."""
    record_identity(row)
    required_text(row, "repo")
    required_text(row, "problem_statement")
    result = dict(row)
    if backend == "harbor":
        required_text(row, "task_path")
        if not isinstance(row.get("task_files"), dict) or not row["task_files"]:
            raise ValueError("missing_or_empty_task_files")
        return result
    if backend == "swesmith":
        required_text(row, "image_name")
    elif backend in {"swebench", "live"}:
        require_commit(row.get("base_commit"), "task.base_commit")
        if backend == "swebench":
            required_text(row, "version")
        else:
            # Live's Python-only harness uses per-task commands/parsers, not
            # SWE-bench's repository/version environment lookup table.
            required_text(row, "log_parser")
            commands = row.get("test_cmds")
            if (
                not isinstance(commands, list)
                or not commands
                or any(
                    not isinstance(command, str) or not command.strip()
                    for command in commands
                )
            ):
                raise ValueError("invalid_test_cmds")
    else:
        raise ValueError(f"unknown_task_backend:{backend}")
    for field in ["FAIL_TO_PASS", "PASS_TO_PASS"]:
        value = row.get(field)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_{field}") from exc
        if (
            not isinstance(value, list)
            or any(not isinstance(v, str) or not v.strip() for v in value)
            or (field == "FAIL_TO_PASS" and not value)
        ):
            raise ValueError(f"invalid_{field}")
        result[field] = value
    return result


def training_record(row: dict) -> None:
    record_identity(row)
    required_text(row, "traj_id")
    if row.get("resolved") is not True:
        raise ValueError("unresolved")
    if "messages" not in row:
        raise ValueError("missing_messages")
