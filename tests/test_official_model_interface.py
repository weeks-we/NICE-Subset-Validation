"""Optional CPU integration with real Transformers/PEFT classes and random weights.

Run in an isolated server-dependency environment. No network or target weights.
"""

import copy

import pytest

pytest.importorskip("transformers", minversion="5.8")
pytest.importorskip("peft", minversion="0.18")

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen3_5Config,
)

from nice_validation.compatibility import model_loader_api
from nice_validation.core import config
from nice_validation.model import (
    Runtime,
    freeze_model,
    initialize_adapter,
    token_ce_sum,
)
from nice_validation.training import sft_steps


def tiny_config():
    return Qwen3_5Config(
        text_config={
            "vocab_size": 256,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "max_position_embeddings": 128,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 1, 0],
            },
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "num_position_embeddings": 16,
        },
        image_token_id=251,
        video_token_id=252,
        vision_start_token_id=253,
        vision_end_token_id=254,
    )


@pytest.fixture(params=["qwen35", "qwen2"])
def runtime(tmp_path, monkeypatch, request):
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    root = tmp_path / "tiny-base"
    causal = request.param == "qwen2"
    architecture = (
        Qwen2Config(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        if causal
        else tiny_config()
    )
    loader = AutoModelForCausalLM if causal else AutoModelForImageTextToText
    model = loader.from_config(architecture, attn_implementation="sdpa")
    model.save_pretrained(root, safe_serialization=True)
    del model
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({str(i): i for i in range(256)}, unk_token="0")
        ),
        unk_token="0",
        pad_token="0",
        bos_token="1",
        eos_token="2",
    )
    tokenizer.save_pretrained(root)
    cfg = config("configs/legacy.yaml")
    cfg["execution"] = "server"
    cfg["model"].update(
        {
            "id": "local-random-tiny-qwen-interface-test",
            "revision": "c" * 40,
            "path": str(root),
            "tokenizer_path": str(root),
            "adapter_path": str(tmp_path / "initial"),
            "loader": loader.__name__,
            "dtype": "float32",
            "device": "cpu",
        }
    )
    cfg["output"] = str(tmp_path / "run")
    freeze_model(cfg)
    monkeypatch.setattr("nice_validation.model.bind_devices", lambda cfg: None)
    initialize_adapter(cfg)
    return Runtime(cfg), cfg


def row():
    return {
        "data_id": "cpu-interface",
        "input_ids": [5, 6, 7, 8, 9, 10],
        "labels": [-100, -100, -100, 8, 9, 10],
        "input_tokens": 6,
        "supervised_tokens": 3,
    }


def test_official_qwen_loader_lora_gradient_update_and_exact_reload(runtime, tmp_path):
    actual, cfg = runtime
    api = model_loader_api(cfg, cfg["model"]["path"])
    assert (
        api["model_class"]
        == (
            "Qwen2ForCausalLM"
            if cfg["model"]["loader"] == "AutoModelForCausalLM"
            else "Qwen3_5ForConditionalGeneration"
        )
        and not api["weights_loaded"]
    )
    assert all(
        "lora_" in name and "visual" not in name for name, _ in actual.parameters
    )
    scoring_before = {
        name: p.detach().clone() for name, p in actual.model.named_parameters()
    }
    actual.candidate_backward(row())
    assert all(
        torch.equal(scoring_before[name], p)
        for name, p in actual.model.named_parameters()
    )
    assert all(p.grad is None for p in actual.model.parameters() if not p.requires_grad)
    name, parameter = max(
        actual.parameters, key=lambda pair: float(pair[1].grad.abs().max())
    )
    index = int(parameter.grad.abs().argmax())
    derivative = float(parameter.grad.flatten()[index])
    original = float(parameter.data.flatten()[index])
    delta = 1e-3
    with torch.no_grad():
        parameter.data.flatten()[index] = original + delta
        plus = float(actual.loss(row()))
        parameter.data.flatten()[index] = original - delta
        minus = float(actual.loss(row()))
        parameter.data.flatten()[index] = original
    assert derivative == pytest.approx(
        (plus - minus) / (2 * delta), rel=0.03, abs=0.003
    )
    before = {name: p.detach().clone() for name, p in actual.model.named_parameters()}
    frozen = {
        name for name, p in actual.model.named_parameters() if not p.requires_grad
    }
    metrics = []
    sft_steps(
        actual.model,
        actual.parameters,
        [row()],
        actual.loss,
        cfg["training"],
        tmp_path / "sft",
        actual.identity,
        metrics.append,
    )
    assert len(metrics) == 1
    assert all(
        torch.equal(before[n], p)
        for n, p in actual.model.named_parameters()
        if n in frozen
    )
    assert any(not torch.equal(before[n], p) for n, p in actual.parameters)
    actual.model.save_pretrained(tmp_path / "trained", safe_serialization=True)
    restored = Runtime(cfg, tmp_path / "trained")
    for (_, left), (_, right) in zip(actual.parameters, restored.parameters):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_official_replay_does_not_update_and_rejects_mismatched_sampling(runtime):
    actual, cfg = runtime
    actual.model.zero_grad(set_to_none=True)
    with torch.no_grad():
        nll = float(token_ce_sum(actual.model, [5, 6, 7], [8, 9, 10]))
    before = {name: p.detach().clone() for name, p in actual.parameters}
    step = {
        "model_identity": actual.identity,
        "input_ids": [5, 6, 7],
        "output_ids": [8, 9, 10],
        "sampling": {"temperature": 1, "top_p": 1, "top_k": 0},
        "logprob_sum": -nll,
    }
    result = actual.replay_backward(step, 0.25)
    assert result["absolute_logprob_difference"] == 0
    assert all(torch.equal(before[name], p) for name, p in actual.parameters)
    assert all(p.grad is None for p in actual.model.parameters() if not p.requires_grad)
    for field in ["input_ids", "output_ids"]:
        bad = copy.deepcopy(step)
        bad[field] = []
        with pytest.raises(ValueError, match="nonempty"):
            actual.replay_backward(bad, 0.25)
    bad = copy.deepcopy(step)
    bad["sampling"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="untruncated"):
        actual.replay_backward(bad, 0.25)


def test_official_generation_and_nice_backward_use_only_lora_and_keep_zero_pad_id(
    runtime,
):
    actual, cfg = runtime
    actual.tokenizer.chat_template = "{{ messages[0]['content'] }}"
    before = {name: p.detach().clone() for name, p in actual.model.named_parameters()}
    actual.model.zero_grad(set_to_none=True)
    trace = actual.generate(
        [{"role": "user", "content": "5"}], nice=True, seed=2026, max_tokens=2
    )
    assert 0 < len(trace["output_ids"]) <= 2
    assert trace["sampling"]["effective_generation_config"]["pad_token_id"] == 0
    result = actual.replay_backward(trace, 0.25)
    assert result["absolute_logprob_difference"] == 0
    assert any(
        p.grad is not None and bool(p.grad.abs().any()) for _, p in actual.parameters
    )
    assert all(
        "lora_" in name
        for name, p in actual.model.named_parameters()
        if p.requires_grad
    )
    assert all(p.grad is None for p in actual.model.parameters() if not p.requires_grad)
    assert all(
        torch.equal(before[name], p) for name, p in actual.model.named_parameters()
    )


def test_official_loader_rejects_wrong_class_and_absent_config(runtime, tmp_path):
    actual, cfg = runtime
    wrong = copy.deepcopy(cfg)
    wrong["model"]["loader"] = (
        "AutoModelForImageTextToText"
        if cfg["model"]["loader"] == "AutoModelForCausalLM"
        else "AutoModelForCausalLM"
    )
    with pytest.raises(ValueError, match="support|architecture"):
        model_loader_api(wrong, cfg["model"]["path"])
    with pytest.raises((ValueError, OSError)):
        model_loader_api(cfg, tmp_path / "absent")


def test_available_context_clamps_output_and_preserves_nice_replay(runtime):
    actual, cfg = runtime
    actual.tokenizer.chat_template = "{{ messages[0]['content'] }}"
    cfg["agent"].update(context_length=4, context_budget_policy="available")
    trace = actual.generate(
        [{"role": "user", "content": "5"}], nice=True, seed=2026, max_tokens=8
    )
    assert trace["generation_budget"] == {
        "requested_max_tokens": 8,
        "effective_max_tokens": 3,
        "available_context_tokens": 3,
        "context_clamped": True,
        "reached_effective_cap": len(trace["output_ids"]) == 3,
    }
    assert trace["sampling"]["effective_generation_config"]["max_new_tokens"] == 3
    assert len(trace["input_ids"]) + len(trace["output_ids"]) <= 4
    assert actual.replay_backward(trace, .25)["absolute_logprob_difference"] == 0
    cfg["agent"]["context_length"] = 1
    with pytest.raises(ValueError, match="CONTEXT_LIMIT"):
        actual.generate([{"role": "user", "content": "5"}], True, 2026, 8)
    cfg["agent"].update(context_length=4, context_budget_policy="strict")
    with pytest.raises(ValueError, match="CONTEXT_LIMIT"):
        actual.generate([{"role": "user", "content": "5"}], True, 2026, 8)


@pytest.mark.parametrize("last_token,reason", [(2, "eos"), (10, "length")])
def test_generation_logs_eos_separately_from_reaching_cap(runtime, monkeypatch, last_token, reason):
    from types import SimpleNamespace

    actual, cfg = runtime
    actual.tokenizer.chat_template = "{{ messages[0]['content'] }}"
    monkeypatch.setattr(
        actual.model, "generate",
        lambda **kwargs: SimpleNamespace(sequences=torch.tensor([[5, 8, last_token]])),
    )
    trace = actual.generate([{"role": "user", "content": "5"}], True, 2026, 2)
    assert trace["finish_reason"] == reason
    assert trace["generation_budget"]["reached_effective_cap"]
    assert not trace["generation_budget"]["context_clamped"]


def test_true_batch_padding_preserves_equal_trajectory_loss_and_gradients(runtime):
    actual, cfg = runtime
    short = {
        "data_id": "short",
        "input_ids": [5, 6, 7, 8],
        "labels": [-100, -100, -100, 8],
        "input_tokens": 4,
        "supervised_tokens": 1,
    }
    rows = [row(), short]
    actual.model.eval()
    actual.model.zero_grad(set_to_none=True)
    expected = sum(actual.loss(r) for r in rows) / 2
    expected.backward()
    reference = {
        n: None if p.grad is None else p.grad.detach().clone()
        for n, p in actual.parameters
    }
    actual.model.zero_grad(set_to_none=True)
    physical_shapes = []
    handle = actual.model.register_forward_pre_hook(
        lambda module, args, kwargs: physical_shapes.append(
            tuple(kwargs["input_ids"].shape)
        ),
        with_kwargs=True,
    )
    try:
        batched = actual.loss(rows)
        batched.backward()
    finally:
        handle.remove()
    assert physical_shapes == [(2, 6)]
    torch.testing.assert_close(
        batched.detach(), expected.detach(), atol=1e-6, rtol=1e-6
    )
    for n, p in actual.parameters:
        if reference[n] is not None:
            torch.testing.assert_close(p.grad, reference[n], atol=2e-6, rtol=2e-4)
    with pytest.raises(ValueError, match="supervised"):
        actual.loss([{**short, "labels": [-100] * 4}])


def test_parallel_feature_math_uses_only_lora_and_never_steps_optimizer(
    runtime, tmp_path, monkeypatch
):
    import numpy as np
    from types import SimpleNamespace
    from nice_validation.core import write
    from nice_validation.scoring import gradient_features
    from nice_validation.score_features import candidate_feature, validation_feature

    actual, cfg = runtime
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["rollouts"] = 4
    view = SimpleNamespace(cfg=cfg, root=tmp_path, event=lambda *a, **k: None)
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("NICE updated weights")
    )
    before = {n: p.detach().clone() for n, p in actual.model.named_parameters()}
    arrays, meta = candidate_feature(view, actual, row())
    actual.candidate_backward(row())
    for name, adam in [("raw", False), ("adam", True)]:
        expected, _ = gradient_features(actual, cfg, adam=adam)
        np.testing.assert_array_equal(arrays[name], expected)
    with torch.no_grad():
        nll = float(token_ce_sum(actual.model, [5, 6, 7], [8, 9]))
    step = {
        "model_identity": actual.identity,
        "input_ids": [5, 6, 7],
        "output_ids": [8, 9],
        "sampling": {"temperature": 1, "top_p": 1, "top_k": 0},
        "logprob_sum": -nll,
    }
    records = []
    for index in range(4):
        write(tmp_path / f"trace-{index}/trace.json", [step])
        records.append(
            {
                "resolved": index % 2 == 0,
                "status": "RESOLVED" if index % 2 == 0 else "TASK_FAILED",
                "evidence_path": f"trace-{index}",
            }
        )
    result, diagnostic = validation_feature(
        view, actual, {"instance_id": "unit"}, records
    )
    actual.model.zero_grad(set_to_none=True)
    actual.replay_backward(step, scale=0.5)
    expected, _ = gradient_features(actual, cfg)
    np.testing.assert_array_equal(result["validation"], expected)
    assert diagnostic["validation_tokens"] == 10
    assert all(torch.equal(before[n], p) for n, p in actual.model.named_parameters())
    assert all(p.grad is None for p in actual.model.parameters() if not p.requires_grad)
