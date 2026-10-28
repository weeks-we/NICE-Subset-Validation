"""Auditable SWE-agent shell/editor semantics. Unknown tools are rejected."""

from __future__ import annotations

import json
import copy
import re
from typing import Any


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the task repository.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "View or edit a UTF-8 file in the task repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "view",
                            "create",
                            "str_replace",
                            "insert",
                            "undo_edit",
                        ],
                    },
                    "path": {"type": "string"},
                    "file_text": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "insert_line": {"type": "integer"},
                    "view_range": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["command", "path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Finish and submit the current repository diff.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def text_content(content: Any) -> str | None:
    """Concatenate text blocks exactly as the official template does.

    Raw records are retained by the preparation manifest. Non-text blocks fail
    rather than being silently dropped from the model context.
    """
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("nontext_content")
    parts = []
    for block in content:
        if (
            not isinstance(block, dict)
            or block.get("type") != "text"
            or not isinstance(block.get("text"), str)
        ):
            raise ValueError("unsupported_content_block")
        parts.append(block["text"])
    return "".join(parts)


def _native_function(body: str) -> dict:
    """Parse the Qwen function/parameter grammar without evaluating payloads."""
    match = re.fullmatch(r"\s*<function=([\w-]+)>\s*\n?(.*?)</function>\s*", body, re.S)
    if not match:
        raise ValueError("malformed_function_block")
    name, payload = match.groups()
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
    if name not in schemas:
        raise ValueError(f"unsupported_tool:{name}")
    parameters, cursor = {}, 0
    while cursor < len(payload):
        if not payload[cursor:].strip():
            break
        header = re.match(r"\s*<parameter=([\w-]+)>", payload[cursor:])
        if not header:
            raise ValueError("malformed_parameter_block")
        key = header[1]
        start = cursor + header.end()
        end = payload.find("</parameter>", start)
        if end < 0 or key in parameters:
            raise ValueError("unclosed_or_duplicate_parameter")
        value = payload[start:end]
        # Remove only the grammar's one wrapping line break; payload whitespace
        # and multiline file edits must survive unchanged.
        for line_break in ["\r\n", "\n"]:
            if value.startswith(line_break):
                value = value[len(line_break) :]
                break
        for line_break in ["\r\n", "\n"]:
            if value.endswith(line_break):
                value = value[: -len(line_break)]
                break
        schema = schemas[name].get("properties", {}).get(key)
        if schema is None:
            raise ValueError(f"unknown_tool_parameter:{key}")
        parameters[key] = value if schema["type"] == "string" else json.loads(value)
        cursor = end + len("</parameter>")
    return {"name": name, "arguments": parameters}


def parse_completion(text: str, call_prefix: str, thinking: bool = False) -> dict:
    """Return canonical mapping-valued calls; preserve invalid raw output upstream."""
    content = text
    reasoning = None
    if "</think>" in content and (thinking or content.lstrip().startswith("<think>")):
        reasoning, content = content.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
    matches = list(re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.S))
    calls, error = [], None
    try:
        if content.count("<tool_call>") != len(matches) or content.count(
            "</tool_call>"
        ) != len(matches):
            raise ValueError("unclosed_tool_call")
        for index, match in enumerate(matches):
            body = match[1]
            function = (
                json.loads(body)
                if body.lstrip().startswith("{")
                else _native_function(body)
            )
            if not isinstance(function, dict):
                raise ValueError("malformed_tool_call")
            name, values = arguments(function)
            calls.append(
                {
                    "id": f"{call_prefix}_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": values},
                }
            )
    except (ValueError, KeyError, TypeError) as exc:
        error = f"INVALID_TOOL_CALL:{exc}"
        calls = []
    message = {
        "role": "assistant",
        "content": re.sub(
            r"<tool_call>.*?</tool_call>", "", content, flags=re.S
        ).strip(),
        "tool_calls": calls,
    }
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"message": message, "parse_error": error}


def arguments(call: dict) -> tuple[str, dict]:
    if not isinstance(call, dict):
        raise ValueError("tool call must be an object")
    function = call.get("function", call)
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError("tool function must have a string name")
    name = function["name"]
    args = function.get("arguments", {})
    args = json.loads(args) if isinstance(args, str) else args
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be an object")
    if name not in {t["function"]["name"] for t in TOOLS}:
        raise ValueError(f"unsupported_tool:{name}")
    if name == "bash" and (
        set(args) != {"command"} or not isinstance(args["command"], str)
    ):
        raise ValueError(
            "bash expects exactly one string command; renaming tools is not a conversion"
        )
    if name == "submit" and args:
        raise ValueError("submit expects no arguments")
    if name == "str_replace_editor":
        if args.get("command") not in {
            "view",
            "create",
            "str_replace",
            "insert",
            "undo_edit",
        } or not isinstance(args.get("path"), str):
            raise ValueError("unsupported_editor_operation")
        for field in {
            "create": ["file_text"],
            "str_replace": ["old_str"],
            "insert": ["new_str", "insert_line"],
        }.get(args["command"], []):
            if field not in args:
                raise ValueError(f"editor missing {field}")
        allowed = {
            "command",
            "path",
            "file_text",
            "old_str",
            "new_str",
            "insert_line",
            "view_range",
        }
        if set(args) - allowed:
            raise ValueError("unknown_editor_arguments")
        for field in ["command", "path", "file_text", "old_str", "new_str"]:
            if field in args and not isinstance(args[field], str):
                raise ValueError(f"editor {field} must be a string")
        if "insert_line" in args and type(args["insert_line"]) is not int:
            raise ValueError("editor insert_line must be an integer")
        if "view_range" in args and (
            not isinstance(args["view_range"], list)
            or len(args["view_range"]) != 2
            or any(type(v) is not int for v in args["view_range"])
        ):
            raise ValueError("editor view_range must contain two integers")
    return name, args


def validate_messages(messages: Any) -> list[dict]:
    if isinstance(messages, str):
        messages = json.loads(messages)
    messages = copy.deepcopy(messages)
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("malformed_messages")
    pending: set[str] = set()
    seen: set[str] = set()
    assistant = 0
    last_assistant = None
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("malformed_message")
        if msg.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unknown_role")
        if "content" in msg:
            msg["content"] = text_content(msg["content"])
        if "reasoning_content" in msg:
            msg["reasoning_content"] = text_content(msg["reasoning_content"])
        if msg["role"] == "assistant":
            assistant += 1
            last_assistant = msg
            if pending:
                raise ValueError("unanswered_tool_calls")
            if msg.get("tool_calls") is None:
                msg.pop("tool_calls", None)
            if not isinstance(msg.get("tool_calls", []), list):
                raise ValueError("tool_calls must be a list")
            for call in msg.get("tool_calls", []):
                arguments(call)
                if isinstance(call["function"].get("arguments"), str):
                    call["function"]["arguments"] = json.loads(
                        call["function"]["arguments"]
                    )
                cid = call.get("id")
                if not isinstance(cid, str) or not cid or cid in seen:
                    raise ValueError("duplicate_or_missing_call_id")
                pending.add(cid)
                seen.add(cid)
        elif msg["role"] == "tool":
            cid = msg.get("tool_call_id")
            if not cid and len(pending) == 1:
                # Older SWE-agent exports omit tool_call_id. A sole outstanding
                # call provides a unique association; raw messages remain saved.
                cid = next(iter(pending))
                msg["tool_call_id"] = cid
            if not isinstance(cid, str) or cid not in pending:
                raise ValueError("orphan_tool_response")
            pending.remove(cid)
        elif pending:
            raise ValueError("interrupted_tool_calls")
    terminal_calls = last_assistant.get("tool_calls", []) if last_assistant else []
    terminal_submit = (
        len(terminal_calls) == 1 and terminal_calls[0]["function"]["name"] == "submit"
    )
    allowed_pending = {terminal_calls[0]["id"]} if terminal_submit else set()
    if (
        (pending and pending != allowed_pending)
        or not assistant
        or (messages[-1]["role"] != "assistant" and not terminal_submit)
    ):
        raise ValueError("incomplete_trajectory")
    return messages
