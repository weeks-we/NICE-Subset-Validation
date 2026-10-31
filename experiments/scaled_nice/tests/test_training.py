import torch
import pytest

from nice_validation.training import load_checkpoint, sft_steps


def train_tiny(folder, stop=None):
    torch.manual_seed(27)
    model = torch.nn.Linear(2, 1, bias=False)
    parameters = list(model.named_parameters())
    rows = [
        {
            "data_id": str(i),
            "supervised_tokens": i + 1,
            "input_tokens": i + 3,
            "x": [i / 10, 1.0],
            "target": i * 0.2,
        }
        for i in range(5)
    ]

    def loss(row):
        output = model(torch.tensor(row["x"]))
        return ((output - row["target"]) ** 2).mean()

    cfg = {
        "learning_rate": 1e-3,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "max_grad_norm": 1,
    }
    report = sft_steps(
        model,
        parameters,
        rows,
        loss,
        cfg,
        folder,
        "frozen-test",
        lambda m: None,
        stop_after=stop,
    )
    return model.weight.detach().clone(), report


def test_resume_matches_uninterrupted_and_independent_branches(tmp_path):
    full, _ = train_tiny(tmp_path / "full")
    _, paused = train_tiny(tmp_path / "resume", stop=2)
    assert paused["status"] == "paused_at_step_boundary"
    resumed, report = train_tiny(tmp_path / "resume")
    torch.testing.assert_close(full, resumed, rtol=0, atol=0)
    independent, _ = train_tiny(tmp_path / "independent")
    torch.testing.assert_close(full, independent, rtol=0, atol=0)
    assert report["updates"] == 5
    state = load_checkpoint(tmp_path / "resume/step_0005.pt", "frozen-test")
    assert len(set(state["order"])) == 5


def test_checkpoint_metric_recovered_after_publication_interruption(tmp_path):
    expected, _ = train_tiny(tmp_path / "full")
    train_tiny(tmp_path / "resume", stop=2)
    (tmp_path / "resume/step_0002.metrics.json").unlink()
    # An unpublished .pt must never hide the last valid optimizer boundary.
    (tmp_path / "resume/step_0003.pt").write_bytes(b"interrupted")
    actual, _ = train_tiny(tmp_path / "resume")
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert (tmp_path / "resume/step_0002.metrics.json").is_file()
    assert list((tmp_path / "resume/interrupted").iterdir())


def test_corrupted_checkpoint_is_rejected(tmp_path):
    train_tiny(tmp_path, stop=2)
    (tmp_path / "step_0002.pt").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        train_tiny(tmp_path)
