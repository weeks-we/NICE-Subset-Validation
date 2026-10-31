"""A fixed effective batch with explicit physical micro-batches and tail handling."""

import math


def batch_spec(cfg):
    effective = cfg.get("effective_batch", 1)
    micro = cfg.get("micro_batch_size", 1)
    if type(effective) is not int or effective < 1:
        raise ValueError("effective_batch must be a positive integer")
    if micro == "auto":
        candidates = cfg.get("micro_batch_candidates", [1, 2, 4])
        if (
            not candidates
            or len(set(candidates)) != len(candidates)
            or any(type(n) is not int or n < 1 or effective % n for n in candidates)
        ):
            raise ValueError(
                "Auto micro-batch candidates must be distinct divisors of effective_batch"
            )
    elif type(micro) is not int or micro < 1 or effective % micro:
        raise ValueError("micro_batch_size must divide effective_batch")
    return effective, micro


def steps_per_epoch(samples, cfg):
    return math.ceil(samples / batch_spec(cfg)[0])


def shuffled_order(rows):
    import random

    if not rows or len({r["data_id"] for r in rows}) != len(rows):
        raise ValueError("Training rows must be nonempty with unique data IDs")
    order = sorted(range(len(rows)), key=lambda i: rows[i]["data_id"])
    random.Random(2026).shuffle(order)
    return order


def metric_members(metric):
    return metric.get("data_ids", [metric["data_id"]] if "data_id" in metric else [])


def backward_batch(model, parameters, rows, loss_function, cfg):
    """Equal trajectory weights and one global clip; caller owns optimizer.step."""
    import torch

    effective, micro = batch_spec(cfg)
    if micro == "auto":
        raise ValueError("Resolve the measured micro-batch plan before training")
    if not rows or len(rows) > effective:
        raise ValueError("Invalid optimizer batch size")
    model.train()
    model.zero_grad(set_to_none=True)
    loss_value, padding = 0.0, 0
    for start in range(0, len(rows), micro):
        part = rows[start : start + micro]
        # Single-row callbacks remain compatible with the original scoring API.
        loss = loss_function(part[0] if len(part) == 1 else part)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite training loss; branch stopped")
        weight = len(part) / len(rows)
        (loss * weight).backward()
        loss_value += float(loss.detach()) * weight
        padding += max(r["input_tokens"] for r in part) * len(part) - sum(
            r["input_tokens"] for r in part
        )
    norm = torch.nn.utils.clip_grad_norm_(
        [p for _, p in parameters], cfg["max_grad_norm"], error_if_nonfinite=True
    )
    return loss_value, float(norm), padding
