"""Independent one-pass SFT, checkpoint publication and deterministic recovery."""

from __future__ import annotations

import os
import random
import time
import uuid
from pathlib import Path

import numpy as np

from nice_validation.core import Run, digest, read, sha, write
from nice_validation.batching import (
    batch_spec,
    backward_batch,
    shuffled_order,
    steps_per_epoch,
    metric_members,
)


def batch_metric(rows, step, cfg):
    effective, micro = batch_spec(cfg)
    return {
        "step": step,
        "data_id": rows[0]["data_id"],
        "data_ids": [r["data_id"] for r in rows],
        "batch_size": len(rows),
        "effective_batch": effective,
        "micro_batch_size": micro,
        "micro_batches": (len(rows) + micro - 1) // micro,
        "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
        "logical_input_tokens": sum(r["input_tokens"] for r in rows),
    }


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

    effective, micro = batch_spec(cfg)
    if micro == "auto":
        raise ValueError("Resolve the measured micro-batch plan before SFT")
    contract = {
        "effective_batch": effective,
        "micro_batch_size": micro,
        "objective": "mean-of-trajectory-token-means-v1",
    }
    order = shuffled_order(rows)
    total_steps = steps_per_epoch(len(rows), cfg)
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
        for index, checkpoint in enumerate(checkpoints, 1):
            metadata = read(checkpoint.with_suffix(".json"))
            if (
                metadata.get("identity") != identity
                or metadata.get("step") != index
                or metadata.get("sha256") != sha(checkpoint)
            ):
                raise ValueError(
                    "Published checkpoint chain identity/step/checksum mismatch"
                )
        saved = load_checkpoint(checkpoints[-1], identity)
        if saved["step"] != len(checkpoints) or not 0 < saved["step"] <= total_steps:
            raise ValueError(
                "Checkpoint optimizer boundary differs from the published step chain"
            )
        if saved["order"] != order:
            raise ValueError("Checkpoint data order mismatch")
        if saved.get("batch_contract") != contract or saved.get("samples_seen") != min(
            saved["step"] * effective, len(rows)
        ):
            raise ValueError("Checkpoint batch contract/sample cursor mismatch")
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
    for index in range(start, total_steps):
        batch = [rows[i] for i in order[index * effective : (index + 1) * effective]]
        start_time = time.monotonic()
        intent = {
            **batch_metric(batch, index + 1, cfg),
            "identity": identity,
        }
        write(
            folder / "intents" / f"step_{index + 1:04d}_{uuid.uuid4().hex}.json", intent
        )
        if on_intent:
            on_intent(intent)
        optimizer.zero_grad(set_to_none=True)
        loss, grad_norm, padding = backward_batch(
            model, parameters, batch, loss_function, cfg
        )
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for _, p in parameters):
            raise ValueError(
                "Nonfinite adapter after update; branch stopped before checkpoint publication"
            )
        metric = {
            **batch_metric(batch, index + 1, cfg),
            "loss": loss,
            "grad_norm": float(grad_norm),
            "learning_rate": cfg["learning_rate"],
            "padding_tokens": padding,
            "packed_tokens": 0,
            "wall_seconds": time.monotonic() - start_time,
        }
        state = {
            "step": index + 1,
            "samples_seen": min((index + 1) * effective, len(rows)),
            "batch_contract": contract,
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
    return {"status": "completed", "updates": total_steps, "exposures": len(rows)}


def train(run: Run, subset_id: str) -> None:
    if run.cfg["experiment"].get("design") == "verified-selection-v1":
        from nice_validation.experiment_training import train_job
        return train_job(run, subset_id)
    if not run.valid("make-subsets"):
        raise ValueError("Complete make-subsets first")
    subsets = read(run.root / "manifests" / "subsets.json")
    matches = [s for s in subsets if s["subset_id"] == subset_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown subset: {subset_id}")
    subset = matches[0]
    from nice_validation.batch_tuning import training_config

    training_cfg, plan_sha = training_config(run)
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
        if len(rows) != len(set(subset["members"])) or {
            r["data_id"] for r in rows
        } != set(subset["members"]):
            raise ValueError("Training subset members are missing or duplicated")
        if run.cfg["mode"] == "mock":
            # Explicit synthetic lifecycle; not a claim of model optimization.
            initial = "mock-common-adapter-v1"
            order = shuffled_order(rows)
            effective, _ = batch_spec(training_cfg)
            for i in range(steps_per_epoch(len(rows), training_cfg)):
                batch = [rows[j] for j in order[i * effective : (i + 1) * effective]]
                metric = {
                    **batch_metric(batch, i + 1, training_cfg),
                    "loss": None,
                    "grad_norm": None,
                    "learning_rate": run.cfg["training"]["learning_rate"],
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
                    "training": training_cfg,
                    "batch_plan_sha256": plan_sha,
                }
            )
            sft_steps(
                runtime.model,
                runtime.parameters,
                rows,
                runtime.loss,
                training_cfg,
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
        subset_size = run.cfg["experiment"]["subset_size"]
        seen = [member for metric in metrics for member in metric_members(metric)]
        updates = steps_per_epoch(subset_size, training_cfg)
        if (
            len(metrics) != updates
            or not len(seen) == len(set(seen)) == subset_size
            or set(seen) != set(subset["members"])
        ):
            raise ValueError("One-pass exposure/update audit failed")
        report = {
            "subset_id": subset_id,
            "status": "completed",
            "mode": run.cfg["mode"],
            "initial_adapter_sha256": initial,
            "final_adapter_sha256": adapter_identity,
            "updates": updates,
            "exposures": subset_size,
            "effective_batch": training_cfg.get("effective_batch", 1),
            "micro_batch_size": training_cfg.get("micro_batch_size", 1),
            "gradient_accumulation_steps": training_cfg.get("effective_batch", 1)
            // training_cfg.get("micro_batch_size", 1),
            "loss_reduction": "mean-of-trajectory-token-means-v1",
            "batch_plan_sha256": plan_sha,
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
