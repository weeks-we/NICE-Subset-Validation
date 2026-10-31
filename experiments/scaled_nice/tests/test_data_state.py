from pathlib import Path

import pytest

from nice_validation.core import Run, config, write
from nice_validation.data import FixtureTokenizer, proportional, repo_of, tokenize
from nice_validation.protocol import arguments, validate_messages


def cfg(tmp_path):
    c = config("configs/development.yaml")
    c["output"] = str(tmp_path / "run")
    return c


def test_config_rejects_research_drift(tmp_path):
    p = tmp_path / "invalid.yaml"
    p.write_text(
        f"extends: {Path('configs/main.yaml').resolve().as_posix()}\nexperiment:\n  candidates: 2000\n"
    )
    with pytest.raises(ValueError, match="Fixed research"):
        config(p)


def test_assistant_mask_excludes_user_tool_and_template_headers():
    messages = [
        {"role": "user", "content": "PUBLIC_USER"},
        {"role": "assistant", "content": "ACTION"},
        {"role": "tool", "content": "TOOL_SECRET"},
        {"role": "assistant", "content": "FINAL"},
    ]
    tokenizer = FixtureTokenizer()
    out = tokenize(tokenizer, messages, False, 10000)
    text = tokenizer.apply_chat_template(messages)
    indices = [i for i, l in enumerate(out["labels"]) if l != -100]
    supervised = "".join(text[i] for i in indices)
    assert "ACTION" in supervised and "FINAL" in supervised
    assert "PUBLIC_USER" not in supervised and "TOOL_SECRET" not in supervised
    assert "<assistant>" not in supervised
    assert out["supervised_tokens"] == len(indices)
    with pytest.raises(ValueError, match="context_length"):
        tokenize(tokenizer, messages, False, 5)


def test_tool_pairing_unknown_tools_and_incomplete_records():
    call = {"id": "a", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}
    messages = [
        {"role": "user", "content": "fix"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "a", "content": "/testbed"},
        {"role": "assistant", "content": "done"},
    ]
    normalized = validate_messages(messages)
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert isinstance(messages[1]["tool_calls"][0]["function"]["arguments"], str)
    with pytest.raises(ValueError, match="unanswered|incomplete"):
        validate_messages(messages[:2])
    with pytest.raises(ValueError, match="unsupported_tool"):
        arguments({"function": {"name": "execute_python", "arguments": {}}})
    with pytest.raises(ValueError, match="orphan"):
        validate_messages(
            [
                messages[0],
                {"role": "tool", "tool_call_id": "b", "content": "x"},
                messages[-1],
            ]
        )


def test_proportional_allocation_stable_and_no_duplicate_sampling():
    rows = [{"instance_id": str(i), "repo": "a" if i < 7 else "b"} for i in range(10)]
    chosen, reserve = proportional(rows, 4, 42)
    assert sum(x["repo"] == "a" for x in chosen) == 3
    assert len(chosen) + len(reserve) == 10
    assert proportional(rows[::-1], 4, 42) == (chosen, reserve)
    assert len({r["instance_id"] for r in chosen}) == 4
    with pytest.raises(ValueError, match="Not enough"):
        proportional(rows, 11, 42)


def test_repository_names_with_hyphens_and_swesmith_suffix():
    assert (
        repo_of({"instance_id": "scikit-learn__scikit-learn-123"})
        == "scikit-learn/scikit-learn"
    )
    assert (
        repo_of({"instance_id": "django-money__django-money.835c1ab8.func_x"})
        == "django-money/django-money"
    )


def test_terminal_submit_does_not_require_nonexistent_tool_observation():
    messages = [
        {"role": "user", "content": "fix"},
        {
            "role": "assistant",
            "content": "done",
            "tool_calls": [
                {"id": "submit-1", "function": {"name": "submit", "arguments": "{}"}}
            ],
        },
    ]
    assert (
        validate_messages(messages)[-1]["tool_calls"][0]["function"]["name"] == "submit"
    )
    messages.append(
        {"role": "tool", "tool_call_id": "submit-1", "content": "submitted"}
    )
    assert len(validate_messages(messages)) == 3


def test_missing_tool_return_id_inferred_only_if_unambiguous():
    call = {"id": "a", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}
    messages = [
        {"role": "user", "content": "fix"},
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "content": "/testbed"},
        {"role": "assistant", "content": "done"},
    ]
    assert validate_messages(messages)[2]["tool_call_id"] == "a"
    assert "tool_call_id" not in messages[2]


def test_known_empty_non_thinking_scaffold_is_header_only():
    class QwenLike(FixtureTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            out = super().apply_chat_template(messages, **kwargs)
            return (
                out + "<think>\n\n</think>\n\n"
                if kwargs.get("add_generation_prompt")
                else out
            )

    result = tokenize(
        QwenLike(),
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "ANSWER"}],
        False,
        999,
    )
    assert result["supervised_tokens"] == len("ANSWER</turn>")


def test_stage_lock_completion_and_corruption(tmp_path):
    run = Run(cfg(tmp_path))
    with run.stage("one") as out:
        p = run.root / "artifact.json"
        write(p, {"a": 1})
        out.append(p)
        with pytest.raises(RuntimeError, match="locked"):
            with run.stage("two"):
                pass
    assert run.valid("one")
    with run.stage("one") as out:
        assert out is None
    write(p, {"a": 2})
    assert not run.valid("one")
    with pytest.raises(ValueError, match="corrupt"):
        with run.stage("one"):
            pass


def test_configuration_identity_change_rejects_cache(tmp_path):
    c = cfg(tmp_path)
    Run(c)
    c["model"]["thinking"] = True
    with pytest.raises(ValueError, match="identity changed"):
        Run(c)


def test_failure_retains_events_and_failure_state(tmp_path):
    run = Run(cfg(tmp_path))
    with pytest.raises(ValueError, match="intentional"):
        with run.stage("bad"):
            run.event("bad", "cost", input_tokens=17)
            raise ValueError("intentional")
    assert run.state()["bad"]["status"] == "failed"
    assert '"input_tokens": 17' in (run.root / "logs/events.jsonl").read_text()
    assert not (run.root / ".run.lock").exists()
