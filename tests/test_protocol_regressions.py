"""Regression coverage for the seven readiness-audit findings."""

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import types
import urllib.error
from unittest.mock import patch

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from nice_validation.core import Run, config
from nice_validation.data import FixtureTokenizer, render, tokenize
from nice_validation.harness import (
    DockerTask,
    InfraError,
    TaskFailure,
    request_generation,
)
from nice_validation.protocol import parse_completion, validate_messages


NATIVE = "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>"


def test_text_blocks_are_lossless_and_raw_messages_are_not_mutated():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": " first\n"},
                {"type": "text", "text": " second  "},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    ]
    raw = copy.deepcopy(messages)
    normalized = validate_messages(json.dumps(messages))
    assert normalized[0]["content"] == " first\n second  "
    assert normalized[1]["content"] == "answer"
    assert messages == raw


@pytest.mark.parametrize(
    "content",
    [[{"type": "image", "url": "x"}], [{"text": "missing type"}], ["not a block"]],
)
def test_nontext_blocks_fail_without_being_discarded(content):
    with pytest.raises(ValueError, match="unsupported_content_block"):
        validate_messages(
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "done"},
            ]
        )


@pytest.mark.parametrize(
    "text",
    [NATIVE, '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'],
)
def test_native_and_json_calls_have_mapping_arguments(text):
    result = parse_completion(text, "call")
    assert result["parse_error"] is None
    assert result["message"]["tool_calls"] == [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "bash", "arguments": {"command": "pwd"}},
        }
    ]


def test_multiline_edit_parameters_keep_payload_whitespace_and_types():
    old, new = "  old\n\tline\n", "  new\n\tline\n"
    text = f"<tool_call>\n<function=str_replace_editor>\n<parameter=command>\nstr_replace\n</parameter>\n<parameter=path>\n/testbed/a.py\n</parameter>\n<parameter=old_str>\n{old}\n</parameter>\n<parameter=new_str>\n{new}\n</parameter>\n</function>\n</tool_call>"
    result = parse_completion(text, "edit")
    assert result["parse_error"] is None
    args = result["message"]["tool_calls"][0]["function"]["arguments"]
    assert args["old_str"] == old and args["new_str"] == new
    text = "<tool_call><function=str_replace_editor><parameter=command>view</parameter><parameter=path>/testbed/a.py</parameter><parameter=view_range>\n[2, -1]\n</parameter></function></tool_call>"
    args = parse_completion(text, "view")["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert args["view_range"] == [2, -1]


def test_multiple_calls_and_reasoning_survive_parsing():
    result = parse_completion(
        "reason\n</think>\n"
        + NATIVE
        + "\n<tool_call><function=submit>\n</function></tool_call>",
        "m",
        thinking=True,
    )
    assert result["parse_error"] is None
    assert result["message"]["reasoning_content"] == "reason"
    assert [c["function"]["name"] for c in result["message"]["tool_calls"]] == [
        "bash",
        "submit",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call><function=bash>",
        "<tool_call><function=unknown></function></tool_call>",
        "<tool_call><function=bash><parameter=command>a</parameter><parameter=command>b</parameter></function></tool_call>",
        "<tool_call><function=bash><parameter=typo>a</parameter></function></tool_call>",
        "<tool_call><function=bash></function></tool_call>",
    ],
)
def test_malformed_calls_are_reported_and_none_are_executed(text):
    result = parse_completion(text, "bad")
    assert result["parse_error"].startswith("INVALID_TOOL_CALL")
    assert not result["message"]["tool_calls"]


# An independent renderer with grouped observations exercises the boundary
# invariant without copying the implementation under test.
class GroupedTokenizer(FixtureTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        parts = []
        for index, message in enumerate(messages):
            if message["role"] == "tool":
                if not index or messages[index - 1]["role"] != "tool":
                    parts.append("<user>")
                parts.append("<observation>" + message["content"] + "</observation>")
                if index + 1 == len(messages) or messages[index + 1]["role"] != "tool":
                    parts.append("</turn>")
            else:
                parts.append(
                    f"<{message['role']}>" + (message.get("content") or "") + "</turn>"
                )
        return "".join(parts) + ("<assistant>" if add_generation_prompt else "")


def conversation():
    first = parse_completion(NATIVE + "\n" + NATIVE, "c")["message"]
    return [
        {"role": "user", "content": "PUBLIC"},
        first,
        {"role": "tool", "tool_call_id": "c_0", "content": "SECRET_ONE"},
        {"role": "tool", "tool_call_id": "c_1", "content": "SECRET_TWO"},
        {"role": "assistant", "content": "ANSWER"},
    ]


def test_parallel_observations_are_not_supervised():
    messages = validate_messages(conversation())
    tokenizer = GroupedTokenizer()
    value = tokenize(tokenizer, messages, False, 32768)
    text = render(tokenizer, messages)
    targets = "".join(
        text[i] for i, label in enumerate(value["labels"]) if label != -100
    )
    assert "SECRET_ONE" not in targets and "SECRET_TWO" not in targets
    assert "PUBLIC" not in targets and "ANSWER" in targets
    assert value["tool_tokens"] > len("SECRET_ONESECRET_TWO")


def test_official_cached_template_roundtrip_and_parallel_mask():
    path = Path(
        os.environ.get(
            "NICE_CHAT_TEMPLATE",
            "runs/engineering-audit-20261026/official_chat_template.jinja",
        )
    )
    if not path.is_file():
        pytest.skip(
            "Set NICE_CHAT_TEMPLATE to a locally obtained official chat_template.jinja"
        )
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)

    def fail(message):
        raise ValueError(message)

    env.globals["raise_exception"] = fail
    template = env.from_string(path.read_text(encoding="utf-8"))

    class OfficialTokenizer(FixtureTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return template.render(messages=messages, **kwargs)

    tokenizer = OfficialTokenizer()
    messages = validate_messages(conversation())
    value = tokenize(tokenizer, messages, False, 32768)
    text = render(tokenizer, messages)
    targets = "".join(
        text[i] for i, label in enumerate(value["labels"]) if label != -100
    )
    assert "<function=bash>" in targets and "ANSWER" in targets
    assert "SECRET_ONE" not in targets and "SECRET_TWO" not in targets
    # Also exercise the exact history handed back by Runtime.generate.
    next_prompt = render(tokenizer, messages[:4], False, True)
    assert "<parameter=command>" in next_prompt


def test_editor_wrapper_only_serializes_declared_parameters(tmp_path):
    module = types.ModuleType("langchain_core.tools")

    class Capture:
        @staticmethod
        def from_function(fn, **kwargs):
            return fn

    module.StructuredTool = Capture
    with patch.dict(
        sys.modules,
        {
            "langchain_core": types.ModuleType("langchain_core"),
            "langchain_core.tools": module,
        },
    ):
        task = DockerTask(config("configs/development.yaml"), {}, tmp_path)
        sent = []
        task.command = lambda *args, **kwargs: sent.append(json.loads(args[2])) or "ok"
        assert (
            task.tools()["str_replace_editor"](
                command="insert",
                path="/testbed/a.py",
                new_str="  hello\n",
                insert_line=0,
            )
            == "ok"
        )
    assert sent == [
        {
            "command": "insert",
            "path": "/testbed/a.py",
            "new_str": "  hello\n",
            "insert_line": 0,
        }
    ]


def test_episode_timeout_is_task_failure_and_connection_error_is_infra():
    clock = iter([10.0, 12.0])
    with (
        patch(
            "nice_validation.harness.time.monotonic", side_effect=lambda: next(clock)
        ),
        patch("urllib.request.urlopen", side_effect=TimeoutError()),
    ):
        with pytest.raises(TaskFailure, match="WALL_LIMIT"):
            request_generation("http://127.0.0.1:8765", {}, 1, deadline=11.0)
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(InfraError):
            request_generation(
                "http://127.0.0.1:8765", {}, 1, deadline=time.monotonic() + 100
            )


def test_changed_library_version_rejects_run_cache(tmp_path):
    cfg = config("configs/development.yaml")
    cfg["mode"], cfg["output"] = "real", str(tmp_path)
    Run(cfg)
    original = importlib.metadata.version

    def changed(package):
        return "999.0.0" if package == "torch" else original(package)

    with patch("importlib.metadata.version", side_effect=changed):
        with pytest.raises(ValueError, match="identity changed"):
            Run(cfg)
