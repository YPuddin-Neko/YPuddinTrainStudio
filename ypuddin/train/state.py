"""RNG capture and full-state checkpoints (exact resume at optimizer-step boundaries)."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def capture_rng(
    generators: dict[str, torch.Generator] | None = None, *, device: torch.device | None = None
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
        if device is not None and device.type == "cuda":
            state["cuda_device"] = device.index if device.index is not None else torch.cuda.current_device()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if generators:
        state["generators"] = {k: g.get_state() for k, g in generators.items()}
    return state


def restore_rng(
    state: dict[str, Any],
    generators: dict[str, torch.Generator] | None = None,
    *,
    device: torch.device | None = None,
) -> None:
    cuda = state.get("cuda") if torch.cuda.is_available() else None
    source = state.get("cuda_device")
    target = device.index if device is not None and device.type == "cuda" else None
    if cuda is not None:
        target = torch.cuda.current_device() if target is None else target
        visible = torch.cuda.device_count()
        if len(cuda) != visible and source is None:
            # Legacy states hold one RNG per GPU visible to the saving worker and no device
            # index. The supervisor can recover the actual device from durable job progress.
            legacy = os.environ.get("YPUDDIN_LEGACY_CUDA_RNG_INDEX")
            source = int(legacy) if legacy is not None and legacy.isdecimal() else None
        if (len(cuda) != visible and source is None) or (
            source is not None and (type(source) is not int or not 0 <= source < len(cuda))
        ):
            raise ValueError("无法确定旧训练状态使用的显卡，请从原任务恢复训练，以保留正确的随机状态。")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if cuda is not None:
        if len(cuda) == visible:
            torch.cuda.set_rng_state_all(cuda)
        if source is not None and (len(cuda) != visible or source != target):
            torch.cuda.set_rng_state(cuda[source], device=target)
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if generators and "generators" in state:
        for k, g in generators.items():
            if k in state["generators"]:
                g.set_state(state["generators"][k])


@dataclass
class Progress:
    step: int = 0  # optimizer steps completed
    epoch: int = 0
    batch_in_epoch: int = 0  # micro-batches consumed in the current epoch
    samples_seen: int = 0
    nan_skips: int = 0
    total_steps: int = 0
    steps_per_epoch: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_checkpoint(
    path: str | Path,
    *,
    adapter_tensors: dict[str, torch.Tensor],
    adapter_metadata: dict[str, str],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler_state: dict[str, Any],
    progress: Progress,
    rng: dict[str, Any],
    training_tensors: dict[str, torch.Tensor] | None = None,
    ema_tensors: dict[str, torch.Tensor] | None = None,
    config_hash: str = "",
    dataset_fingerprint: str = "",
    model_identity: str = "",
    training_kind: str = "adapter",
) -> Path:
    """Write everything needed for an exact resume into ``path`` (atomic via temp dir + rename)."""
    final = Path(path)
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in adapter_tensors.items()},
        str(tmp / ("model.safetensors" if training_kind == "full-model" else "adapter.safetensors")),
        metadata=adapter_metadata,
    )
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save(scheduler.state_dict() if scheduler is not None else {}, tmp / "scheduler.pt")
    torch.save(rng, tmp / "rng.pt")
    if training_tensors is not None and training_kind != "full-model":
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in training_tensors.items()},
            str(tmp / "training.safetensors"),
        )
    if ema_tensors:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in ema_tensors.items()}, str(tmp / "ema.safetensors")
        )
    meta = {
        "progress": progress.to_dict(),
        "sampler": sampler_state,
        "config_hash": config_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "model_identity": model_identity,
        "format": 3 if training_kind == "full-model" else 2 if training_tensors is not None else 1,
        "training_kind": training_kind,
    }
    (tmp / "state.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    meta = json.loads((p / "state.json").read_text(encoding="utf-8"))
    version = meta.get("format", 1)
    if version not in (1, 2, 3):
        raise ValueError(f"unsupported checkpoint format {version}")
    out: dict[str, Any] = {
        "adapter": load_file(str(p / ("model.safetensors" if version == 3 else "adapter.safetensors"))),
        "training_kind": meta.get("training_kind", "adapter"),
        "optimizer": torch.load(p / "optimizer.pt", map_location="cpu", weights_only=False),
        "scheduler": torch.load(p / "scheduler.pt", map_location="cpu", weights_only=False),
        "rng": torch.load(p / "rng.pt", map_location="cpu", weights_only=False),
        "progress": Progress(**meta["progress"]),
        "sampler": meta["sampler"],
        "config_hash": meta.get("config_hash", ""),
        "dataset_fingerprint": meta.get("dataset_fingerprint", ""),
        "model_identity": meta.get("model_identity", ""),
        "format": version,
    }
    if version == 3:
        out["training"] = out["adapter"]
    elif version == 2:
        out["training"] = load_file(str(p / "training.safetensors"))
    if (p / "ema.safetensors").exists():
        out["ema"] = load_file(str(p / "ema.safetensors"))
    return out
