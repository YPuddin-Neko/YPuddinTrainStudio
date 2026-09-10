"""RNG capture and full-state checkpoints (exact resume at optimizer-step boundaries)."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def capture_rng(generators: dict[str, torch.Generator] | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if generators:
        state["generators"] = {k: g.get_state() for k, g in generators.items()}
    return state


def restore_rng(state: dict[str, Any], generators: dict[str, torch.Generator] | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
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
    ema_tensors: dict[str, torch.Tensor] | None = None,
    config_hash: str = "",
    dataset_fingerprint: str = "",
) -> Path:
    """Write everything needed for an exact resume into ``path`` (atomic via temp dir + rename)."""
    final = Path(path)
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in adapter_tensors.items()},
        str(tmp / "adapter.safetensors"),
        metadata=adapter_metadata,
    )
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save(scheduler.state_dict() if scheduler is not None else {}, tmp / "scheduler.pt")
    torch.save(rng, tmp / "rng.pt")
    if ema_tensors:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in ema_tensors.items()}, str(tmp / "ema.safetensors")
        )
    meta = {
        "progress": progress.to_dict(),
        "sampler": sampler_state,
        "config_hash": config_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "format": 1,
    }
    (tmp / "state.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    meta = json.loads((p / "state.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "adapter": load_file(str(p / "adapter.safetensors")),
        "optimizer": torch.load(p / "optimizer.pt", map_location="cpu", weights_only=False),
        "scheduler": torch.load(p / "scheduler.pt", map_location="cpu", weights_only=False),
        "rng": torch.load(p / "rng.pt", map_location="cpu", weights_only=False),
        "progress": Progress(**meta["progress"]),
        "sampler": meta["sampler"],
        "config_hash": meta.get("config_hash", ""),
        "dataset_fingerprint": meta.get("dataset_fingerprint", ""),
    }
    if (p / "ema.safetensors").exists():
        out["ema"] = load_file(str(p / "ema.safetensors"))
    return out
