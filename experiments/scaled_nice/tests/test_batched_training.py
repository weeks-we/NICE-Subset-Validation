"""True batch forwards, exact tails, accumulation equivalence and recovery."""

import copy
from pathlib import Path

import pytest
import torch

from nice_validation.batching import batch_spec, metric_members
from nice_validation.batch_tuning import choose_micro, tune, training_config
from nice_validation.core import Run, config, read, write
from nice_validation.training import sft_steps, load_checkpoint


def tiny(folder, micro=2, effective=4, stop=None, fail=None):
    torch.manual_seed(27)
    model = torch.nn.Linear(2, 1, bias=False).double()
    rows = [
        {
            "data_id": str(i),
            "input_tokens": i + 3,
            "supervised_tokens": i + 1,
            "x": [i / 10, 1.0],
            "target": i * 0.2,
        }
        for i in range(10)
    ]
    forwards = []

    def loss(value):
        batch = value if isinstance(value, list) else [value]
        forwards.append(len(batch))
        if fail == len(forwards):
            raise RuntimeError("interrupted mid accumulation")
        x = torch.tensor([r["x"] for r in batch], dtype=torch.float64)
        targets = torch.tensor([[r["target"]] for r in batch], dtype=torch.float64)
        return ((model(x) - targets) ** 2).mean()

    cfg = {
        "learning_rate": 1e-3,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "max_grad_norm": 1,
        "effective_batch": effective,
        "micro_batch_size": micro,
    }
    result = sft_steps(
        model,
        list(model.named_parameters()),
        rows,
        loss,
        cfg,
        folder,
        "batched-test",
        lambda m: None,
        stop_after=stop,
    )
    return model.weight.detach().clone(), result, forwards


def test_true_microbatch_tail_and_all_samples_once(tmp_path):
    weight, report, forwards = tiny(tmp_path, micro=2)
    assert forwards == [
        2,
        2,
        2,
        2,
        2,
    ]  # physical model forwards, not sequential batch=1
    assert report == {"status": "completed", "updates": 3, "exposures": 10}
    metrics = [read(p) for p in sorted(tmp_path.glob("step_*.metrics.json"))]
    assert [m["batch_size"] for m in metrics] == [4, 4, 2]
    members = [n for m in metrics for n in metric_members(m)]
    assert len(members) == len(set(members)) == 10
    saved = load_checkpoint(tmp_path / "step_0003.pt", "batched-test")
    assert saved["samples_seen"] == 10


@pytest.mark.parametrize("micro", [1, 2, 4])
def test_accumulation_matches_true_batch_and_resume(tmp_path, micro):
    reference, _, _ = tiny(tmp_path / "reference", micro=4)
    actual, _, _ = tiny(tmp_path / "full", micro=micro)
    torch.testing.assert_close(reference, actual, rtol=1e-12, atol=1e-12)
    tiny(tmp_path / "resume", micro=micro, stop=1)
    resumed, _, _ = tiny(tmp_path / "resume", micro=micro)
    torch.testing.assert_close(actual, resumed, rtol=0, atol=0)


def test_interrupted_partial_accumulation_is_not_published_or_replayed(tmp_path):
    reference, _, _ = tiny(tmp_path / "full")
    with pytest.raises(RuntimeError, match="mid accumulation"):
        tiny(tmp_path / "resume", fail=3)
    assert len(list((tmp_path / "resume").glob("step_*.pt"))) == 1
    actual, _, _ = tiny(tmp_path / "resume")
    torch.testing.assert_close(reference, actual, rtol=0, atol=0)


@pytest.mark.parametrize("changes", [{"effective": 2}, {"micro": 1}])
def test_resume_rejects_changed_batch_contract(tmp_path, changes):
    tiny(tmp_path, stop=1)
    with pytest.raises(ValueError, match="batch contract"):
        tiny(tmp_path, **changes)


@pytest.mark.parametrize(
    "cfg",
    [
        {"effective_batch": True},
        {"effective_batch": 0},
        {"effective_batch": 4, "micro_batch_size": 3},
        {
            "effective_batch": 4,
            "micro_batch_size": "auto",
            "micro_batch_candidates": [1, 3],
        },
        {
            "effective_batch": 4,
            "micro_batch_size": "auto",
            "micro_batch_candidates": [2, 2],
        },
    ],
)
def test_invalid_batch_config_fails_early(cfg):
    with pytest.raises(ValueError):
        batch_spec(cfg)


def measurements(a=10, b=20, c=None):
    return {
        "trials": [
            {
                "micro_batch_size": n,
                "status": "PASS" if rate else "OOM",
                "capacity_update_completed": bool(rate),
                "supervised_tokens_per_second": rate,
            }
            for n, rate in [(1, a), (2, b), (4, c)]
        ]
    }


def test_tuner_requires_all_gpus_and_prefers_small_near_ties():
    assert choose_micro([measurements(), measurements(c=25)], [1, 2, 4]) == 2
    assert choose_micro([measurements(b=None), measurements()], [1, 2, 4]) == 1
    assert choose_micro([measurements(a=19.2, b=20)], [1, 2, 4]) == 1
    assert choose_micro([measurements(c=30)], [1, 2, 4]) == 4
    with pytest.raises(ValueError, match="No micro-batch"):
        choose_micro([measurements(a=None, b=None)], [1, 2, 4])
    bad = measurements()
    bad["trials"][0]["capacity_update_completed"] = False
    with pytest.raises(ValueError, match="capacity evidence"):
        choose_micro([bad], [1, 2, 4])


def test_missing_plan_never_silently_uses_batch_one(tmp_path):
    cfg = config("configs/scaled-mock.yaml")
    cfg["output"] = str(tmp_path)
    with pytest.raises(ValueError, match="tune-training-batch"):
        training_config(Run(cfg))


@pytest.mark.parametrize("fatal", [False, True])
def test_physical_probe_only_falls_back_on_oom_and_closes_runtime(
    tmp_path, monkeypatch, fatal
):
    from nice_validation.batch_tuning import profile

    cfg = config("configs/scaled-mock.yaml")
    cfg["output"] = str(tmp_path)
    run = Run(cfg)
    rows = [
        {"data_id": str(i), "input_tokens": i + 3, "supervised_tokens": i + 1}
        for i in range(10)
    ]
    write(run.root / "manifests/candidates.json", rows)
    write(
        run.root / "manifests/subsets.json", [{"members": [r["data_id"] for r in rows]}]
    )
    seen, closed = [], []

    class TinyRuntime:
        def __init__(self, cfg):
            self.model = torch.nn.Linear(1, 1).double()
            self.parameters = list(self.model.named_parameters())

        def loss(self, value):
            batch = value if isinstance(value, list) else [value]
            seen.append(len(batch))
            if len(batch) == 4:
                if fatal:
                    raise ValueError("invalid model interface")
                raise torch.cuda.OutOfMemoryError("simulated capacity")
            return (
                self.model(torch.ones(len(batch), 1, dtype=torch.float64))
                .square()
                .mean()
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr("nice_validation.model.Runtime", TinyRuntime)
    for name in ["empty_cache", "synchronize", "reset_peak_memory_stats"]:
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ["max_memory_allocated", "max_memory_reserved"]:
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    output = tmp_path / "profile.json"
    if fatal:
        with pytest.raises(ValueError, match="model interface"):
            profile(
                run.root / "config/snapshot.json",
                cfg["resources"]["gpu_uuids"][0],
                output,
            )
        assert not output.exists()
    else:
        profile(
            run.root / "config/snapshot.json", cfg["resources"]["gpu_uuids"][0], output
        )
        report = read(output)
        assert [t["status"] for t in report["trials"]] == ["PASS", "PASS", "OOM"]
        assert report["excluded_from_research_training"] is True
        assert set(seen) == {1, 2, 4}
    assert closed == [True]


def test_real_tuner_routes_each_gpu_freezes_plan_and_reuses_it(tmp_path, monkeypatch):
    from types import SimpleNamespace

    cfg = config("configs/scaled-mock.yaml")
    cfg["mode"], cfg["output"] = "real", str(tmp_path)
    run = Run(cfg)
    path = run.root / "manifests/subsets.json"
    with run.stage("make-subsets") as out:
        write(path, [{"subset_id": "low_11", "members": []}])
        out.append(path)
    seen = []

    def probe(argv, **kwargs):
        gpu = argv[argv.index("--gpu-uuid") + 1]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == gpu
        seen.append(gpu)
        write(
            Path(argv[argv.index("--output") + 1]),
            {
                **measurements(),
                "gpu_uuid": gpu,
                "run_identity": run.identity,
                "wall_seconds": 0.5,
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("nice_validation.batch_tuning.subprocess.run", probe)
    tune(run)
    assert set(seen) == set(cfg["resources"]["gpu_uuids"]) and len(seen) == 4
    selected, checksum = training_config(run)
    assert selected["micro_batch_size"] == 2 and selected["effective_batch"] == 4
    tune(run)
    assert len(seen) == 4
    write(run.root / "training_batch/plan.json", {"corrupt": True})
    with pytest.raises(ValueError, match="tune-training-batch"):
        training_config(run)
