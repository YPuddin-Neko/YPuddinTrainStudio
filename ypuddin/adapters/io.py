"""Save / load adapter weights as safetensors with kohya-compatible keys and rich metadata."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

import ypuddin

from .base import AdapterModule, refuse_tucker
from .full import Full
from .loha import LoHa
from .lokr import LoKr
from .lora import LoRA

SAVE_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Suffixes that identify each algorithm in a kohya/LyCORIS-style file.
_DETECT = (
    ("lokr", ("lokr_w1", "lokr_w1_a")),
    ("loha", ("hada_w1_a",)),
    ("lora", ("lora_down.weight", "lora_A.weight")),
    ("full", ("diff",)),
)
ALGO_CLASSES: dict[str, type[AdapterModule]] = {"lora": LoRA, "lokr": LoKr, "loha": LoHa, "full": Full}


def detect_algo(suffixes: set[str]) -> str | None:
    for algo, keys in _DETECT:
        if any(k in suffixes for k in keys):
            return algo
    return None


def group_by_module(tensors: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """``lora_unet_blocks_0_q.lora_down.weight`` -> ``{"lora_unet_blocks_0_q": {"lora_down.weight": t}}``."""
    out: dict[str, dict[str, Tensor]] = {}
    for key, t in tensors.items():
        module, _, suffix = key.partition(".")
        out.setdefault(module, {})[suffix] = t
    return out


def sha256_of_tensors(tensors: dict[str, Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors):
        h.update(k.encode())
        h.update(tensors[k].detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def build_metadata(
    *,
    targets: dict[str, Any],
    adapter_cfg: dict[str, Any],
    family: str,
    architecture: str,
    title: str,
    resolution: str | None = None,
    extra: dict[str, str] | None = None,
    config_hash: str | None = None,
    dataset_fingerprint: str | None = None,
    steps: int | None = None,
    epoch: int | None = None,
) -> dict[str, str]:
    """kohya ``ss_*`` + ModelSpec ``modelspec.*`` + our ``ypuddin.*`` keys (all values are strings)."""
    adapter_cfg = {key: value for key, value in adapter_cfg.items() if key != "resume_weights"}
    algo = adapter_cfg.get("algo")

    def exported(rank: Any, alpha: Any) -> tuple[Any, Any]:
        # OrthoLoRA and T-LoRA are exported as plain LoRA; the kohya keys describe that file.
        if algo == "tlora" and adapter_cfg.get("tlora_ortho") and isinstance(rank, int):
            return 2 * rank, 2 * rank  # trained and frozen terms together, with the scale folded in
        if algo == "ortho" and isinstance(rank, int):
            return rank, rank
        return rank, alpha

    rank, alpha = exported(adapter_cfg.get("rank"), adapter_cfg.get("alpha"))
    args = ("algo", "factor", "decompose_both", "rs_lora", "dora", "preset", "init")
    if algo == "tlora":
        args += ("tlora_min_rank", "tlora_power", "tlora_ortho")
    if adapter_cfg.get("dora"):
        args += ("dora_axis",)
    network_args = {k: adapter_cfg.get(k) for k in args}
    # kohya's conv_dim / conv_alpha: the rank and alpha of kernels larger than 1×1 (Full layers have none).
    if any(
        layer.get("algo") != "full" and any(size != 1 for size in layer.get("kernel") or ())
        for layer in targets.values()
        if isinstance(layer, dict)
    ):
        conv_rank, conv_alpha = adapter_cfg.get("conv_rank"), adapter_cfg.get("conv_alpha")
        network_args["conv_dim"], network_args["conv_alpha"] = exported(
            adapter_cfg.get("rank") if conv_rank is None else conv_rank,
            adapter_cfg.get("alpha") if conv_alpha is None else conv_alpha,
        )
    meta: dict[str, str] = {
        "ss_network_module": "ypuddin.adapters",
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
        "ss_network_args": json.dumps(network_args, ensure_ascii=False),
        "ss_base_model_version": family,
        "ss_training_finished_at": str(time.time()),
        "modelspec.sai_model_spec": "1.0.1",
        "modelspec.architecture": architecture,
        "modelspec.implementation": "ypuddin",
        "modelspec.title": title,
        "modelspec.date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ypuddin.version": ypuddin.__version__,
        "ypuddin.format": "1",
        "ypuddin.family": family,
        "ypuddin.targets": json.dumps(targets, ensure_ascii=False),
        "ypuddin.adapter": json.dumps(adapter_cfg, ensure_ascii=False),
    }
    if resolution:
        meta["modelspec.resolution"] = resolution
        meta["ss_resolution"] = resolution
    if steps is not None:
        meta["ss_steps"] = str(steps)
    if epoch is not None:
        meta["ss_epoch"] = str(epoch)
    if config_hash:
        meta["ypuddin.config_hash"] = config_hash
    if dataset_fingerprint:
        meta["ypuddin.dataset_fingerprint"] = dataset_fingerprint
    if extra:
        meta.update({k: str(v) for k, v in extra.items()})
    return meta


def save_adapter_file(
    path: str | Path, tensors: dict[str, Tensor], metadata: dict[str, str], *, dtype: str = "bf16"
) -> Path:
    """Atomically write tensors (cast to ``dtype``, ``alpha``/``dora_scale`` kept fp32) with metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Tensor] = {}
    target = SAVE_DTYPES[dtype]
    for k, t in tensors.items():
        t = t.detach().cpu().contiguous()
        if k.endswith(".alpha") or k.endswith(".dora_scale"):
            out[k] = t.to(torch.float32)
        else:
            out[k] = t.to(target)
    meta = dict(metadata)
    meta["modelspec.hash_sha256"] = "0x" + sha256_of_tensors(out)
    tmp = p.with_suffix(p.suffix + ".tmp")
    save_file(out, str(tmp), metadata=meta)
    tmp.replace(p)
    return p


def load_adapter_file(path: str | Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    from safetensors import safe_open

    p = Path(path)
    tensors = load_file(str(p))
    with safe_open(str(p), framework="pt") as f:
        metadata = dict(f.metadata() or {})
    return tensors, metadata


def parse_targets_metadata(metadata: dict[str, str]) -> dict[str, Any]:
    raw = metadata.get("ypuddin.targets")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def modules_from_tensors(
    tensors: dict[str, Tensor],
    metadata: dict[str, str] | None = None,
    *,
    prefix: str = "lora_unet",
    dtype: torch.dtype = torch.float32,
    kernels: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, tuple[AdapterModule, Tensor | None]]:
    """Rebuild standalone adapter modules keyed by canonical-ish name (underscored kohya name).

    ``kernels`` names each target layer's kernel (``()`` for a linear layer) where the caller knows the model.
    """
    targets_meta = parse_targets_metadata(metadata or {})
    by_underscored = {k.replace(".", "_"): v for k, v in targets_meta.items()}
    out: dict[str, tuple[AdapterModule, Tensor | None]] = {}
    for module_key, sub in group_by_module(tensors).items():
        if not module_key.startswith(prefix + "_"):
            continue
        algo = detect_algo(set(sub))
        if algo is None:
            continue
        dora = sub.pop("dora_scale", None)
        refuse_tucker(sub)
        if "lora_A.weight" in sub:  # PEFT-style naming
            sub["lora_down.weight"] = sub.pop("lora_A.weight")
            sub["lora_up.weight"] = sub.pop("lora_B.weight")
        meta = targets_meta.get(module_key) or by_underscored.get(module_key[len(prefix) + 1 :])
        extra = {"kernel": kernels[module_key]} if kernels and module_key in kernels else {}
        out[module_key] = (ALGO_CLASSES[algo].from_tensors(sub, meta, dtype=dtype, **extra), dora)
    return out
