"""Independent one-pass SFT, checkpoint publication and deterministic recovery."""

from __future__ import annotations

import os
import random
import time
import uuid
from pathlib import Path

import numpy as np

from nice_validation.core import Run, digest, read, sha, write


def save_checkpoint(path: Path, state: dict, identity: str) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        write(
            path.with_suffix(".json"),
            {"identity": identity, "sha256": sha(path), "step": state["step"]},
        )
    finally:
        temp.unlink(missing_ok=True)


def load_checkpoint(path: Path, identity: str) -> dict:
    import torch

    metadata = read(path.with_suffix(".json"))
    if metadata["identity"] != identity or metadata["sha256"] != sha(path):
        raise ValueError("Checkpoint identity/checksum mismatch")
    # Checkpoints are trusted local artifacts only. Never load third-party pickle files.
    return torch.load(path, map_location="cpu", weights_only=False)


def sft_steps(
    model,
    parameters,
    rows: list[dict],
    loss_function,
    cfg: dict,
    folder: Path,
    identity: str,
    on_step,
    stop_after: int | None = None,
    on_intent=None,
) -> dict:
    """Publish adapter/optimizer/RNG/order at each complete optimizer boundary."""
    import torch

    order = sorted(range(len(rows)), key=lambda i: rows[i]["data_id"])
    random.Random(2026).shuffle(order)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    optimizer = torch.optim.AdamW(
        [p for _, p in parameters],
        lr=cfg["learning_rate"],
        betas=tuple(cfg["betas"]),
        eps=cfg["epsilon"],
        weight_decay=0,
    )
    start = 0
    uses_cuda = any(p.is_cuda for _, p in parameters)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(
        p for p in folder.glob("step_*.pt") if p.with_suffix(".json").is_file()
    )
    for orphan in sorted(
        p for p in folder.glob("step_*.pt") if not p.with_suffix(".json").is_file()
    ):
        archive = folder / "interrupted" / (orphan.name + "." + uuid.uuid4().hex)
        archive.parent.mkdir(exist_ok=True)
        orphan.rename(archive)
    if [p.name for p in checkpoints] != [
        f"step_{i:04d}.pt" for i in range(1, len(checkpoints) + 1)
    ]:
        raise ValueError(
            "Published checkpoints are not contiguous; preserve evidence and investigate"
        )
    if checkpoints:
        saved = load_checkpoint(checkpoints[-1], identity)
        if saved["order"] != order:
            raise ValueError("Checkpoint data order mismatch")
        for name, param in parameters:
            param.data.copy_(saved["adapter"][name].to(param.device))
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if uses_cuda and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start = saved["step"]
        metric_path = folder / f"step_{start:04d}.metrics.json"
        if not metric_path.exists():
            write(
                metric_path,
                {**saved["metric"], "checkpoint_sha256": sha(checkpoints[-1])},
            )
    metrics = []
    for index in range(start, len(rows)):
        row = rows[order[index]]
        start_time = time.monotonic()
        intent = {
            "step": index + 1,
            "data_id": row["data_id"],
            "supervised_tokens": row["supervised_tokens"],
            "logical_input_tokens": row["input_tokens"],
            "identity": identity,
        }
        write(
            folder / "intents" / f"step_{index + 1:04d}_{uuid.uuid4().hex}.json", intent
        )
        if on_intent:
            on_intent(intent)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        loss = loss_function(row)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss; branch stopped")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for _, p in parameters], cfg["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for _, p in parameters):
            raise ValueError(
                "Nonfinite adapter after update; branch stopped before checkpoint publication"
            )
        metric = {
            "step": index + 1,
            "data_id": row["data_id"],
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "learning_rate": cfg["learning_rate"],
            "supervised_tokens": row["supervised_tokens"],
            "logical_input_tokens": row["input_tokens"],
            "padding_tokens": 0,
            "packed_tokens": 0,
            "wall_seconds": time.monotonic() - start_time,
        }
        state = {
            "step": index + 1,
            "adapter": {n: p.detach().cpu().clone() for n, p in parameters},
            "optimizer": optimizer.state_dict(),
            "order": order,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if uses_cuda else None,
            "metric": metric,
        }
        path = folder / f"step_{index + 1:04d}.pt"
        save_checkpoint(path, state, identity)
        metric["checkpoint_sha256"] = sha(path)
        write(folder / f"step_{index + 1:04d}.metrics.json", metric)
        on_step(metric)
        metrics.append(metric)
        if stop_after and index + 1 >= stop_after:
            return {"status": "paused_at_step_boundary", "updates": index + 1}
    return {"status": "completed", "updates": len(rows)}


def train(run: Run, subset_id: str) -> None:
    if not run.valid("make-subsets"):
        raise ValueError("Complete make-subsets first")
    subsets = read(run.root / "manifests" / "subsets.json")
    matches = [s for s in subsets if s["subset_id"] == subset_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown subset: {subset_id}")
    subset = matches[0]
    with run.stage("train:" + subset_id) as out:
        if out is None:
            return
        folder = run.root / "training" / subset_id
        rows = [
            r
            for r in read(run.root / "manifests" / "candidates.json")
            if r["data_id"] in subset["members"]
        ]
        start = time.monotonic()
        if run.cfg["mode"] == "mock":
            # Explicit synthetic lifecycle; not a claim of model optimization.
            initial = "mock-common-adapter-v1"
            for i, row in enumerate(sorted(rows, key=lambda r: r["data_id"])):
                metric = {
                    "step": i + 1,
                    "data_id": row["data_id"],
                    "loss": None,
                    "grad_norm": None,
                    "learning_rate": run.cfg["training"]["learning_rate"],
                    "supervised_tokens": row["supervised_tokens"],
                    "logical_input_tokens": row["input_tokens"],
                    "padding_tokens": 0,
                    "mode": "mock",
                }
                write(folder / f"step_{i + 1:04d}.metrics.json", metric)
            write(
                folder / "adapter" / "mock.json",
                {"synthetic": True, "members": subset["members"]},
            )
            adapter_identity = digest(subset["members"])
        else:
            from nice_validation.model import Runtime, tree_identity

            runtime = Runtime(run.cfg)
            initial = runtime.initial_identity["adapter_sha256"]
            score_initial = read(run.root / "scores" / "diagnostics.json")[
                "adapter_identity"
            ]
            if initial != score_initial:
                raise ValueError("SFT and scoring must share the same initial adapter")
            identity = digest(
                {
                    "run": run.identity,
                    "subset": subset,
                    "initial": initial,
                    "rows": rows,
                }
            )
            sft_steps(
                runtime.model,
                runtime.parameters,
                rows,
                runtime.loss,
                run.cfg["training"],
                folder,
                identity,
                lambda m: run.event(
                    "train:" + subset_id, "optimizer_step", subset_id=subset_id, **m
                ),
                on_intent=lambda m: run.event(
                    "train:" + subset_id,
                    "training_forward_intent",
                    subset_id=subset_id,
                    **m,
                ),
            )
            runtime.model.save_pretrained(folder / "adapter", safe_serialization=True)
            adapter_identity = tree_identity(folder / "adapter")
            runtime.close()
        metrics = [read(p) for p in sorted(folder.glob("step_*.metrics.json"))]
        if len(metrics) != 30 or len({r["data_id"] for r in metrics}) != 30:
            raise ValueError("One-pass exposure/update audit failed")
        report = {
            "subset_id": subset_id,
            "status": "completed",
            "mode": run.cfg["mode"],
            "initial_adapter_sha256": initial,
            "final_adapter_sha256": adapter_identity,
            "updates": 30,
            "exposures": 30,
            "members": subset["members"],
            "supervised_tokens": sum(m["supervised_tokens"] for m in metrics),
            "wall_seconds": time.monotonic() - start,
            "gpu_hours": (
                (time.monotonic() - start) * len(run.cfg["model"]["gpu_uuids"]) / 3600
                if run.cfg["mode"] == "real"
                else None
            ),
            "logical_input_tokens": sum(m["logical_input_tokens"] for m in metrics),
        }
        write(folder / "training.json", report)
        out.extend(
            [
                folder / "training.json",
                *sorted(folder.glob("step_*.metrics.json")),
                *sorted((folder / "adapter").glob("*")),
            ]
        )
