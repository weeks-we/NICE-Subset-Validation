import copy

import numpy as np
import pytest
import torch

from nice_validation.analysis import ranking, task_rate
from nice_validation.model import token_ce_sum
from nice_validation.scoring import aggregate_score, explicit_score, fresh_adam, sketch


def test_adam_direction_matches_actual_first_optimizer_step():
    parameter = torch.nn.Parameter(
        torch.tensor([2.0, -1.0, 0.4, 0.0], dtype=torch.float64)
    )
    original = parameter.detach().clone()
    gradient = torch.tensor([1.0, -4.0, 1e-9, 0.0], dtype=torch.float64)
    parameter.grad = gradient.clone()
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, eps=1e-8, weight_decay=0)
    torch.nn.utils.clip_grad_norm_([parameter], 1)
    optimizer.step()
    observed = ((original - parameter) / 1e-3).detach().numpy()
    np.testing.assert_allclose(
        fresh_adam(gradient.numpy(), 1e-8, 1), observed, atol=1e-12
    )


def test_taskwise_cosine_aggregation_and_fixed_zero_denominator():
    direction = np.array([1.0, 0.0])
    h = np.array([[1.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
    assert aggregate_score(direction, h) == pytest.approx(1 / 3)
    assert aggregate_score(direction, h) == explicit_score(direction, h)
    assert aggregate_score(np.zeros(2), h) == 0
    assert aggregate_score(direction, h) != pytest.approx(1 / np.sqrt(101))


def test_countsketch_is_chunk_invariant_and_repeatable():
    rng = np.random.default_rng(41)
    a = rng.normal(size=999)
    np.testing.assert_allclose(
        sketch([a], 128, 42), sketch([a[:33], a[33:91], a[91:]], 128, 42), atol=1e-14
    )
    assert not np.array_equal(sketch([a], 128, 42), sketch([a], 128, 43))


def test_projection_accuracy_diagnostic_on_seeded_large_dimension():
    rng = np.random.default_rng(5)
    x = rng.normal(size=10000)
    y = 0.5 * x + rng.normal(size=10000)
    exact = aggregate_score(x, np.array([y]))
    approx = aggregate_score(sketch([x], 8192, 42), np.array([sketch([y], 8192, 42)]))
    assert abs(exact - approx) < 0.04


def test_adam_must_precede_nonlinear_projection():
    g = np.array([1.0, 10.0, -2.0, 4.0])
    correct = sketch([fresh_adam(g, 1e-8, 1)], 2, 42)
    wrong = fresh_adam(sketch([g], 2, 42), 1e-8, 1)
    assert not np.allclose(correct, wrong)


class SmallLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 5)
        self.output = torch.nn.Linear(5, 17)

    def forward(self, input_ids, attention_mask=None):
        class Output:
            pass

        out = Output()
        out.logits = self.output(self.embedding(input_ids).cumsum(1))
        return out


def test_replay_supervises_generated_tokens_only_and_sums_length():
    torch.manual_seed(4)
    model = SmallLM()
    prompt, output = [1, 2, 3, 4], [5, 6]
    nll = token_ce_sum(model, prompt, output)
    logits = model(torch.tensor([prompt + output])).logits
    explicit = -torch.log_softmax(logits.float(), -1)[0, [3, 4], [5, 6]].sum()
    assert float(nll) == pytest.approx(float(explicit))
    assert float(nll) != pytest.approx(float(explicit) / len(output))


def test_weighted_segments_equal_mean_gradient_and_order_independent():
    torch.manual_seed(11)
    model = SmallLM()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    turns = [([1, 2], [3]), ([1, 2, 3, 4], [5, 6, 7])]
    total_tokens = sum(len(out) for _, out in turns)

    def gradient(order):
        model.zero_grad(set_to_none=True)
        for i in order:
            prompt, out = turns[i]
            (token_ce_sum(model, prompt, out) / total_tokens).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    g1, g2 = gradient([0, 1]), gradient([1, 0])
    torch.testing.assert_close(g1, g2)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert optimizer_before == optimizer.state_dict()


def test_ties_and_missing_evaluation_never_create_false_rate():
    assert task_rate([], []) == {
        "status": "incomplete",
        "resolved_count": 0,
        "completed_tasks": 0,
        "denominator": 0,
        "rate": None,
    }
    assert ranking([1, 1, 1], [1, 2, 3])["spearman"] is None
    rank = ranking([1, 1, 3], [2, 2, 5])
    assert rank["concordance"] == 1 and rank["tied_pairs"] == 1
    records = [{"task_id": "a", "resolved": True, "status": "RESOLVED"}]
    assert task_rate(records, ["a", "b"])["rate"] is None
    assert task_rate(records * 2, ["a", "b"])["rate"] is None
    assert task_rate(records, ["a"])["rate"] == 1
