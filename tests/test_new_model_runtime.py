"""Real random-weight Transformers/PEFT checks for new memory/performance paths."""
import copy
from pathlib import Path
import torch
import pytest
from test_official_model_interface import runtime, row
from nice_validation.model import Runtime, token_ce_sum, tree_identity
from nice_validation.service import RuntimeManager


@pytest.mark.parametrize("chunk_size", [1, 7, 512])
def test_chunked_ce_preserves_all_tokens_masks_loss_and_gradients(chunk_size):
    from nice_validation.model import chunked_ce_sum
    torch.manual_seed(42)
    logits = torch.randn(1027, 17, requires_grad=True)
    labels = torch.randint(17, (1027,))
    labels[2:8] = -100
    expected = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
    expected.backward()
    gradient = logits.grad.detach().clone()
    logits.grad = None
    actual = chunked_ce_sum(logits, labels, chunk_size)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=2e-6)
    torch.testing.assert_close(logits.grad, gradient, atol=1e-7, rtol=1e-6)


def test_selected_logits_preserve_loss_and_lora_gradients(runtime):
    actual, cfg = runtime
    rows = [row(), {**row(), "input_ids": [5, 6, 7, 8], "labels": [-100, -100, 7, 8]}]
    actual.model.eval()
    actual.model.zero_grad(set_to_none=True)
    full = actual.loss(rows)
    full.backward()
    gradients = {n: p.grad.detach().clone() for n, p in actual.parameters}
    actual.model.zero_grad(set_to_none=True)
    cfg["model"]["selective_logits"] = True
    selected = actual.loss(rows)
    selected.backward()
    torch.testing.assert_close(full, selected, atol=1e-6, rtol=1e-6)
    for n, p in actual.parameters:
        torch.testing.assert_close(p.grad, gradients[n], atol=2e-6, rtol=2e-4)


def test_nice_checkpointed_mode_is_zero_dropout_and_preserves_replay(runtime):
    actual, cfg = runtime
    cfg["experiment"]["design"] = "verified-selection-v1"
    before = {n: p.detach().clone() for n, p in actual.model.named_parameters()}
    actual.model.eval()
    with torch.no_grad():
        nll = float(token_ce_sum(actual.model, [5, 6, 7], [8, 9]))
    step = {"model_identity": actual.identity, "input_ids": [5, 6, 7], "output_ids": [8, 9],
            "sampling": {"temperature": 1, "top_p": 1, "top_k": 0}, "logprob_sum": -nll}
    result = actual.replay_backward(step, .125)
    assert result["absolute_logprob_difference"] <= result["tolerance"]
    assert actual.model.training
    assert all(torch.equal(before[n], p) for n, p in actual.model.named_parameters())
    assert all(p.grad is None for n, p in actual.model.named_parameters() if not p.requires_grad)


def test_history_keeps_latest_five_observations_and_full_evidence_unmodified(runtime):
    actual, cfg = runtime
    cfg["agent"]["history_observations"] = 5
    actual.tokenizer.chat_template = "{{ messages[0]['content'] }}"
    messages = [{"role": "user", "content": "5"}] + [
        {"role": "tool", "content": str(i), "tool_call_id": "call_" + str(i)} for i in range(10)]
    original = copy.deepcopy(messages)
    trace = actual.generate(messages, True, 2026, 2)
    assert messages == original
    assert all("omitted" in m["content"] for m in trace["messages"][1:6])
    assert trace["messages"][6:] == messages[6:]
    assert [m.get("tool_call_id") for m in trace["messages"]] == [m.get("tool_call_id") for m in messages]


def test_greedy_evaluation_does_not_repeat_a_teacher_forcing_forward(runtime, monkeypatch):
    actual, cfg = runtime
    actual.tokenizer.chat_template = "{{ messages[0]['content'] }}"
    monkeypatch.setattr("nice_validation.model.token_ce_sum", lambda *a, **k: pytest.fail("Greedy evaluation must not repeat teacher-forcing inference"))
    trace = actual.generate([{"role": "user", "content": "5"}], False, 2026, 2)
    assert trace["logprob_sum"] is None
    assert trace["usage"]["logprob_replay_tokens"] == 0
    assert trace["usage"]["output_tokens"] == len(trace["output_ids"])


def test_service_swaps_only_lora_and_matches_fresh_load(runtime, tmp_path, monkeypatch):
    actual, cfg = runtime
    cfg["model"]["reuse_base_for_inference"] = True
    initial = Path(cfg["model"]["adapter_path"])
    changed = tmp_path / "changed"
    with torch.no_grad():
        for n, p in actual.parameters:
            p.add_(.01)
    actual.model.save_pretrained(changed, safe_serialization=True)
    created = []
    monkeypatch.setattr("nice_validation.service.Runtime", lambda *a: created.append(True) or actual)
    manager = RuntimeManager(cfg)
    manager.get(initial, tree_identity(initial))
    base = actual.model.get_base_model()
    switched = manager.get(changed, tree_identity(changed))
    assert switched is actual and actual.model.get_base_model() is base and len(created) == 1
    reference = Runtime(cfg, changed)
    torch.testing.assert_close(switched.loss(row()), reference.loss(row()), atol=1e-6, rtol=1e-6)
    reference.close()
    manager.get(initial, tree_identity(initial))
    assert actual.model.get_base_model() is base and len(created) == 1
    reference = Runtime(cfg)
    torch.testing.assert_close(actual.loss(row()), reference.loss(row()), atol=1e-6, rtol=1e-6)
    reference.close()
