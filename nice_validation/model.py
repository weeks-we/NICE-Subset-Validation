"""Local-only model loading, shared LoRA initialization and exact-token replay."""

from __future__ import annotations

import importlib.metadata
import os
import random
import re
import subprocess
import uuid
from pathlib import Path

import numpy as np

from nice_validation.core import digest, read, require_commit, sha, utc, write
from nice_validation.data import render


def tree_identity(path: Path) -> str:
    return digest(
        {
            str(p.relative_to(path)): sha(p)
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.name != "identity.json"
        }
    )


def server_only(cfg: dict) -> None:
    if cfg["execution"] != "server" or cfg["mode"] != "real":
        raise ValueError(
            "Real weights/training require an explicit execution: server configuration"
        )


def bind_devices(cfg: dict) -> None:
    server_only(cfg)
    uuids = cfg["model"]["gpu_uuids"]
    if len(uuids) != 1 or any(
        not re.fullmatch(r"GPU-[a-fA-F0-9-]{36}", u) for u in uuids
    ):
        raise ValueError(
            "This model replica requires one full GPU UUID; deploy independent replicas for task shards"
        )
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    if not any(
        line.split(",")[0].strip() == uuids[0] for line in p.stdout.splitlines()
    ):
        raise ValueError("Requested GPU UUID is not present")
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing and existing != uuids[0]:
        raise ValueError("CUDA_VISIBLE_DEVICES conflicts with the authorized full UUID")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuids[0]
    write(
        Path(cfg["output"]) / "logs" / "resources" / f"gpu-{os.getpid()}.json",
        {"utc": utc(), "authorized_uuids": uuids, "nvidia_smi_identity": p.stdout},
    )


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    path = cfg["model"]["tokenizer_path"] or cfg["model"]["path"]
    if not path or not Path(path).is_dir():
        raise ValueError(
            "Set a local tokenizer_path; no implicit model/tokenizer download is permitted"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError(
            "Assistant mask requires a fast tokenizer with offset mappings"
        )
    return tokenizer


def freeze_model(cfg: dict) -> dict:
    server_only(cfg)
    require_commit(cfg["model"]["revision"], "model.revision")
    if not cfg["model"]["path"]:
        raise ValueError("Set model.path to an inspected local snapshot")
    root = Path(cfg["model"]["path"])
    if not (root / "config.json").is_file():
        raise ValueError("Local model config.json missing")
    files = {
        str(p.relative_to(root)): {"sha256": sha(p), "size": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.name != "model_lock.json"
        and ".cache" not in p.parts
        and ".git" not in p.parts
    }
    result = {
        "id": cfg["model"]["id"],
        "revision": cfg["model"]["revision"],
        "files": files,
        "architecture": read(root / "config.json"),
        "declared_origin": "operator verified local snapshot or explicit fetch-model",
    }
    write(root / "model_lock.json", result)
    return result


def fetch_model(cfg: dict) -> str:
    server_only(cfg)
    require_commit(cfg["model"]["revision"], "model.revision")
    target = cfg["model"]["path"]
    if not target:
        raise ValueError("Set model.path")
    marker = Path(target).with_name(Path(target).name + ".download.json")
    request = {"id": cfg["model"]["id"], "revision": cfg["model"]["revision"]}
    if marker.exists() and any(read(marker).get(k) != v for k, v in request.items()):
        raise ValueError("Interrupted model download belongs to a different source")
    if (Path(target) / "model_lock.json").is_file():
        lock = read(Path(target) / "model_lock.json")
        if any(lock.get(k) != v for k, v in request.items()):
            raise ValueError("Existing model lock differs from requested source")
        for name, metadata in lock["files"].items():
            path = Path(target) / name
            if not path.is_file() or sha(path) != metadata["sha256"]:
                raise ValueError(f"Frozen model file changed: {name}")
        return str(target)
    if (Path(target) / "config.json").exists() and not marker.exists():
        return "Existing local model found; inspect it and run freeze-model. No download performed."
    from huggingface_hub import hf_hub_download, snapshot_download
    from nice_validation.compatibility import model_loader_api

    write(marker, {**request, "status": "in_progress"})
    hf_hub_download(
        cfg["model"]["id"],
        "config.json",
        revision=cfg["model"]["revision"],
        local_dir=target,
    )
    api = model_loader_api(cfg, Path(target))
    write(marker, {**request, "status": "in_progress", "model_api": api})
    snapshot_download(
        cfg["model"]["id"], revision=cfg["model"]["revision"], local_dir=target
    )
    freeze_model(cfg)
    write(marker, {**request, "status": "completed", "model_api": api})
    return str(target)


def load_base(cfg: dict):
    bind_devices(cfg)
    import torch
    import transformers

    root = Path(cfg["model"]["path"])
    require_commit(cfg["model"]["revision"], "model.revision")
    lock = read(root / "model_lock.json")
    if lock["id"] != cfg["model"]["id"] or lock["revision"] != cfg["model"]["revision"]:
        raise ValueError("Local model source lock differs from frozen model identity")
    for name, metadata in lock["files"].items():
        p = root / name
        if not p.is_file() or sha(p) != metadata["sha256"]:
            raise ValueError(f"Model source file changed: {name}")
    loader = cfg["model"]["loader"]
    if (
        not loader
        or not loader.startswith("AutoModelFor")
        or not hasattr(transformers, loader)
    ):
        raise ValueError(
            "Inspect architecture and set an installed official AutoModelFor* loader explicitly"
        )
    from nice_validation.compatibility import model_loader_api

    model_loader_api(cfg, root)
    model = getattr(transformers, loader).from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attention"],
    )
    model.to(cfg["model"]["device"])
    model.config.use_cache = False
    if cfg["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    return model


def initialize_adapter(cfg: dict) -> dict:
    server_only(cfg)
    import torch
    from peft import LoraConfig, get_peft_model

    if not cfg["model"]["adapter_path"]:
        raise ValueError("Set model.adapter_path for the common initial LoRA")
    path = Path(cfg["model"]["adapter_path"])
    if path.exists():
        if not (path / "identity.json").is_file():
            raise ValueError(
                "Initial adapter directory is incomplete; preserve it and choose a new adapter directory"
            )
        identity = read(path / "identity.json")
        if identity["adapter_sha256"] != tree_identity(path):
            raise ValueError("Initial adapter checksum mismatch")
        if (
            identity["rank"] != cfg["model"]["rank"]
            or identity["alpha"] != cfg["model"]["alpha"]
            or identity["model_revision"] != cfg["model"]["revision"]
            or identity["model_lock_sha256"]
            != sha(Path(cfg["model"]["path"]) / "model_lock.json")
            or {name.rsplit(".", 1)[-1] for name in identity["targets"]}
            != set(cfg["model"]["lora_targets"])
        ):
            raise ValueError("Initial adapter initialization config mismatch")
        return identity
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    model = load_base(cfg)
    targets = [
        name
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in cfg["model"]["lora_targets"]
        and not any(
            part in name.casefold() for part in ["vision", "visual", "image", "audio"]
        )
    ]
    if not targets or set(n.rsplit(".", 1)[-1] for n in targets) != set(
        cfg["model"]["lora_targets"]
    ):
        raise ValueError(
            "LoRA target modules missing; inspect actual named_modules before changing configuration"
        )
    adapter = get_peft_model(
        model,
        LoraConfig(
            r=cfg["model"]["rank"],
            lora_alpha=cfg["model"]["alpha"],
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    # Publish only a complete adapter. Interrupted staging directories remain as evidence.
    staging = path.with_name(path.name + ".initializing-" + uuid.uuid4().hex)
    adapter.save_pretrained(staging, safe_serialization=True)
    identity = {
        "adapter_sha256": tree_identity(staging),
        "model_revision": cfg["model"]["revision"],
        "model_lock_sha256": sha(Path(cfg["model"]["path"]) / "model_lock.json"),
        "targets": targets,
        "rank": cfg["model"]["rank"],
        "alpha": cfg["model"]["alpha"],
        "seed": 2026,
    }
    write(staging / "identity.json", identity)
    staging.rename(path)
    del adapter, model
    torch.cuda.empty_cache()
    return identity


def token_ce_sum(model, input_ids, target_ids):
    """Sum model-generated token CE; observations appear only as conditions."""
    import torch

    if not target_ids or not input_ids:
        raise ValueError("Replay requires nonempty exact prompt and output IDs")
    device = next(model.parameters()).device
    joined = torch.tensor([input_ids + target_ids], device=device)
    logits = model(input_ids=joined, attention_mask=torch.ones_like(joined)).logits
    start = len(input_ids) - 1
    return torch.nn.functional.cross_entropy(
        logits[0, start : start + len(target_ids)].float(),
        torch.tensor(target_ids, device=device),
        reduction="sum",
    )


class Runtime:
    def __init__(self, cfg: dict, adapter: Path | None = None):
        server_only(cfg)
        from peft import PeftModel

        self.cfg = cfg
        if not cfg["model"]["adapter_path"]:
            raise ValueError("Set model.adapter_path for the common initial LoRA")
        initial = Path(cfg["model"]["adapter_path"])
        self.initial_identity = read(initial / "identity.json")
        if (
            self.initial_identity["rank"] != cfg["model"]["rank"]
            or self.initial_identity["alpha"] != cfg["model"]["alpha"]
            or {name.rsplit(".", 1)[-1] for name in self.initial_identity["targets"]}
            != set(cfg["model"]["lora_targets"])
        ):
            raise ValueError("Initial adapter modules/rank/alpha differ from config")
        if self.initial_identity["adapter_sha256"] != tree_identity(initial):
            raise ValueError("Common initial LoRA adapter has changed")
        if self.initial_identity["model_revision"] != cfg["model"]["revision"]:
            raise ValueError("Initial adapter belongs to a different base revision")
        if self.initial_identity["model_lock_sha256"] != sha(
            Path(cfg["model"]["path"]) / "model_lock.json"
        ):
            raise ValueError("Initial adapter model source lock mismatch")
        path = adapter or initial
        self.adapter_identity = tree_identity(path)
        self.tokenizer = load_tokenizer(cfg)
        self.model = PeftModel.from_pretrained(
            load_base(cfg), path, is_trainable=True, local_files_only=True
        )
        self.model.eval()
        self.parameters = sorted(
            (name, p) for name, p in self.model.named_parameters() if p.requires_grad
        )
        if not self.parameters or any(
            "lora_" not in name for name, _ in self.parameters
        ):
            raise ValueError("Only LoRA parameters may be trainable")
        self.identity = digest(
            {
                "model_lock": sha(Path(cfg["model"]["path"]) / "model_lock.json"),
                "adapter": self.adapter_identity,
                "dtype": cfg["model"]["dtype"],
                "tokenizer": {
                    str(
                        p.relative_to(
                            Path(cfg["model"]["tokenizer_path"] or cfg["model"]["path"])
                        )
                    ): sha(p)
                    for p in sorted(
                        Path(
                            cfg["model"]["tokenizer_path"] or cfg["model"]["path"]
                        ).glob("*")
                    )
                    if p.is_file() and ("token" in p.name or "template" in p.name)
                },
                "thinking": cfg["model"]["thinking"],
                "mask": cfg["score"]["mask_version"],
                "versions": {
                    k: importlib.metadata.version(k)
                    for k in ["torch", "transformers", "peft"]
                },
            }
        )

    def loss(self, row: dict):
        import torch

        device = next(self.model.parameters()).device
        ids = torch.tensor([row["input_ids"]], device=device)
        labels = torch.tensor([row["labels"]], device=device)
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )

    def close(self) -> None:
        import gc
        import torch

        self.parameters = []
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def candidate_backward(self, row: dict) -> None:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        loss = self.loss(row)
        if not loss.isfinite():
            raise ValueError("Nonfinite candidate loss")
        loss.backward()

    def replay_backward(self, step: dict, scale: float) -> dict:
        if step["model_identity"] != self.identity:
            raise ValueError("Generation and gradient model identities differ")
        if (
            step["sampling"]["temperature"] != 1
            or step["sampling"]["top_p"] != 1
            or step["sampling"]["top_k"] != 0
        ):
            raise ValueError(
                "NICE replay requires the untruncated on-policy distribution"
            )
        self.model.eval()
        nll = token_ce_sum(self.model, step["input_ids"], step["output_ids"])
        difference = abs(float(nll.detach()) + step["logprob_sum"])
        tolerance = self.cfg["score"].get("logprob_atol_per_token", 0.005) * len(
            step["output_ids"]
        )
        if difference > tolerance:
            raise ValueError(
                f"Generation/replay logprob mismatch: {difference} > {tolerance}"
            )
        (nll * scale).backward()
        return {
            "absolute_logprob_difference": difference,
            "tolerance": tolerance,
            "output_tokens": len(step["output_ids"]),
        }

    def generate(
        self, messages: list[dict], nice: bool, seed: int, max_tokens: int
    ) -> dict:
        import torch
        from transformers import GenerationConfig
        from nice_validation.protocol import TOOLS

        torch.manual_seed(seed)
        self.model.eval()
        text = render(self.tokenizer, messages, self.cfg["model"]["thinking"], True)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) + max_tokens > self.cfg["agent"]["context_length"]:
            raise ValueError("CONTEXT_LIMIT")
        generation = GenerationConfig(
            do_sample=nice,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=max_tokens,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        tensor = torch.tensor([ids], device=next(self.model.parameters()).device)
        with torch.no_grad():
            result = self.model.generate(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                generation_config=generation,
            )
        output_ids = result.sequences[0, len(ids) :].tolist()
        raw_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        from nice_validation.protocol import parse_completion

        parsed = parse_completion(
            content, f"call_{seed}", self.cfg["model"]["thinking"]
        )
        with torch.no_grad():
            nll = token_ce_sum(self.model, ids, output_ids)
        return {
            **parsed,
            "input_ids": ids,
            "output_ids": output_ids,
            "raw_output": raw_text,
            "messages": messages,
            "tools": TOOLS,
            "rendered_prompt_sha256": digest(text),
            "model_identity": self.identity,
            "adapter_identity": self.adapter_identity,
            "logprob_sum": -float(nll),
            "sampling": {
                "temperature": 1 if nice else 0,
                "top_p": 1,
                "top_k": 0,
                "grammar_constraints": False,
                "effective_generation_config": generation.to_dict(),
            },
            "usage": {
                "input_tokens": len(ids),
                "output_tokens": len(output_ids),
                "thinking_tokens": None,
                "cached_input_tokens": None,
                "logprob_replay_tokens": len(ids) + len(output_ids),
                "estimated": False,
            },
        }
