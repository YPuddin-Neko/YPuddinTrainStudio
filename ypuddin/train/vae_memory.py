"""Image VAE memory per GPU while caching latents, encoding online and decoding previews.

Workspace coefficients are the peak bytes allocated per image pixel above the VAE weights and the
FP32 pixel batch, measured with real encodes and decodes on an RTX 5070 Ti (PyTorch 2.11, CUDA 12.8).
Reserved-memory floors cover allocator peaks in those probes, including multi-image FP32 encoding.
Fused SDPA keeps the mid-block attention linear in its token count; PyTorch's math SDPA materializes
FP32 attention scores.
``EncodeMonitor`` logs what an encode phase actually ran and the memory it took.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ypuddin.config import TrainConfig
from ypuddin.models.vae_tiling import CACHE_TILE_PIXELS, CACHE_TILE_THRESHOLD, VAE_TILE_PIXELS

log = logging.getLogger(__name__)

_DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
# (encode, decode) bytes per pixel: the Qwen-Image VAE keeps 96 channels at full resolution in its
# compute precision; the SDXL and FLUX.2 AutoencoderKL keep 128 channels and always run in FP32.
_PIXEL_WORKSPACE = {("qwen", 2): (966, 1345), ("qwen", 4): (1536, 2306), ("kl", 4): (2560, 3842)}
# Allocator fragmentation around those peaks.
_FRAGMENTATION = 1.1
# Math SDPA holds FP32 scores, their softmax and a mask: 9 bytes per score on PyTorch 2.11 CUDA. The
# composite safe softmax of earlier releases adds another FP32 copy, which HIP builds may still use.
_SCORE_BYTES = 13
_QUERY_CHUNK = 2048
_MARGIN_MB = 512
# Reserved workspace above the weights and input, from the same 5070 Ti probes.
# A cap confines isolated small/mid-size pool peaks to their measured envelope;
# larger calls retain the convolution/attention estimate when that is higher.
# Key: (VAE kind, element bytes, decode, attention); value: (bytes/pixel, cap MiB/image).
_RESERVED_WORKSPACE = {
    ("qwen", 4, False, "fused"): (2016, None),
    ("qwen", 4, False, "math"): (3200, 1800),
    ("qwen", 2, True, "fused"): (1824, None),
    ("qwen", 2, True, "chunked"): (2080, None),
    ("qwen", 4, True, "fused"): (3328, None),
    ("kl", 4, False, "math"): (4480, 2520),
    ("kl", 4, True, "fused"): (7680, 1920),
}
# SDXL FP32 encode batches 2/3 reserved up to 1.45 times their live workspace.
_KL_BATCH_RESERVED_BYTES_PER_PIXEL = 3840  # 1.5 * 2560 allocated bytes/pixel
_PROBE_TIMEOUT = 20.0
_PROBE_CACHE: dict[tuple[str, ...], tuple[float, str]] = {}
_PROBE_LOCK = threading.Lock()
_PROBE = r"""
import json, sys, warnings
import torch
backend = "unknown"
if torch.cuda.is_available() and torch.version.hip:
    with torch.no_grad(), torch.cuda.device(sys.argv[1]):
        shape, dtype = (1, 1, 8, int(sys.argv[3])), getattr(torch, sys.argv[2])
        q, k, v = [torch.empty(shape, device=sys.argv[1], dtype=dtype) for _ in range(3)]
        cuda = torch.backends.cuda
        try:
            params = cuda.SDPAParams(q, k, v, None, 0.0, False, False)
        except TypeError:
            params = cuda.SDPAParams(q, k, v, None, 0.0, False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fused = (cuda.flash_sdp_enabled() and cuda.can_use_flash_attention(params, debug=False)) or (
                cuda.mem_efficient_sdp_enabled() and cuda.can_use_efficient_attention(params, debug=False)
            )
        backend = "fused" if fused else "math"
print(json.dumps({"backend": backend}))
"""


def attention_backend_probe(device: str | torch.device, dtype: torch.dtype, channels: int, profile: str) -> str:
    """Ask a short-lived process whether HIP's native SDPA fuses the VAE's single-head attention.

    The planner itself never creates a GPU context. The query allocates three tiny tensors and runs no
    attention. Answers last for the service's lifetime; an unanswered query is retried after a minute.
    """
    if not getattr(torch.version, "hip", None):
        # The query would run this same PyTorch build, which has no HIP runtime to ask.
        return "unknown"
    key = (
        sys.executable, str(torch.__version__), str(getattr(torch.version, "hip", None)),
        str(device), _DTYPES[dtype], str(channels), profile,
        *(os.environ.get(name, "") for name in (
            "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
        )),
    )
    with _PROBE_LOCK:
        cached = _PROBE_CACHE.get(key)
        if cached and (cached[1] != "unknown" or time.monotonic() - cached[0] < 60):
            return cached[1]
        try:
            result = subprocess.run(
                [sys.executable, "-c", _PROBE, str(device), _DTYPES[dtype], str(channels)],
                capture_output=True, text=True, timeout=_PROBE_TIMEOUT, check=False, stdin=subprocess.DEVNULL,
            )
            backend = (
                json.loads(result.stdout.strip().splitlines()[-1])["backend"] if result.returncode == 0 else "unknown"
            )
            if backend not in {"fused", "math"}:
                backend = "unknown"
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
            backend = "unknown"
        _PROBE_CACHE[key] = (time.monotonic(), backend)
        return backend


@lru_cache(maxsize=1)
def _qwen_vae_parameter_count() -> int:
    from ypuddin.models.anima.vendor.qwen_image_vae_2d import AutoencoderKLQwenImage2D

    # The image loader discards temporal kernels; checkpoint file size counts them.
    with torch.device("meta"):
        vae = AutoencoderKLQwenImage2D()
    return sum(t.numel() for t in (*vae.parameters(), *vae.buffers()))


@lru_cache(maxsize=16)
def _autoencoder_kl_geometry(family: str, config_json: str) -> tuple[int, int]:
    """FP32 storage bytes and mid-block attention width of a Diffusers AutoencoderKL, built on meta."""
    from diffusers import AutoencoderKL, AutoencoderKLFlux2

    model_type = AutoencoderKLFlux2 if family == "flux2" else AutoencoderKL
    with torch.device("meta"):
        vae = model_type.from_config(json.loads(config_json))
    tensors = {id(t): t for t in (*vae.parameters(), *vae.buffers())}
    storage = sum(t.numel() * (4 if t.is_floating_point() else t.element_size()) for t in tensors.values())
    return storage, int(vae.config.block_out_channels[-1])


def _autoencoder_kl_config(cfg: TrainConfig) -> dict[str, Any]:
    """The configured VAE's geometry; an unreadable draft path uses the family's standard VAE."""
    from ypuddin.models.sdxl.loading import ASSETS

    try:
        if not cfg.model.dit_path and not cfg.model.vae_path:
            raise ValueError("no model path")
        if cfg.model.family == "sdxl":
            from ypuddin.models.sdxl.loading import component_config, component_path

            return component_config(component_path(cfg.model.dit_path or ".", "vae", cfg.model.vae_path), "vae")
        from ypuddin.models.flux2.loading import component, read_json

        path = component(cfg.model.dit_path or ".", "vae", cfg.model.vae_path)
        config_path = (path if path.is_dir() else path.parent) / "config.json"
        return read_json(config_path) if config_path.is_file() else {}
    except (OSError, ValueError, KeyError, TypeError):
        if cfg.model.family == "sdxl":
            return json.loads((Path(ASSETS) / "vae" / "config.json").read_text(encoding="utf-8"))
        return {}


@dataclass(frozen=True)
class VaeMemory:
    """One family's image VAE as it runs on the planned device."""

    kind: str  # "qwen": Anima / Krea 2; "kl": SDXL / FLUX.2
    dtype: torch.dtype
    weights_mb: float
    attention_channels: int
    # "fused", "math", or "unknown" (estimated as math)
    attention_backend: str
    query_chunking: bool = False
    tiling: bool = False
    cache_tiling: bool = False
    # Whether VAE attention chunking applies to this runtime at all (Qwen VAE on HIP).
    chunking_available: bool = False

    @property
    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()


def vae_memory(
    cfg: TrainConfig, dtype: torch.dtype, *, device: str | torch.device | None, profile: str
) -> VaeMemory | None:
    """The VAE a run loads, or None for a family without an image VAE on the device."""
    device_type = torch.device(device).type if device is not None else None
    hip = device_type == "cuda" and (profile == "linux-dtk" or bool(getattr(torch.version, "hip", None)))
    family = cfg.model.family
    if family in {"anima", "krea2"}:
        vae_dtype = torch.float32 if cfg.memory.no_half_vae or device_type == "cpu" else dtype
        kind, channels = "qwen", 384
        weights_mb = _qwen_vae_parameter_count() * torch.empty((), dtype=vae_dtype).element_size() / 2**20
    elif family in {"sdxl", "flux2"}:
        vae_dtype, kind = torch.float32, "kl"
        storage, channels = _autoencoder_kl_geometry(
            family, json.dumps(_autoencoder_kl_config(cfg), sort_keys=True, default=str)
        )
        weights_mb = storage / 2**20
    else:
        return None
    if device_type == "mps":
        # MPS materializes these single-head scores.
        backend = "math"
    elif not hip:
        # CUDA's memory-efficient kernel covers FP32 and half precision at these widths; CPU uses its own.
        backend = "fused"
    elif profile == "linux-dtk" and cfg.loop.deterministic:
        # Reproducible DTK training turns fused SDPA off for the whole process.
        backend = "math"
    else:
        backend = attention_backend_probe(device, vae_dtype, channels, profile)
    qwen = kind == "qwen"
    return VaeMemory(
        kind=kind,
        dtype=vae_dtype,
        weights_mb=weights_mb,
        attention_channels=channels,
        attention_backend=backend,
        query_chunking=qwen and hip and cfg.memory.vae_attention_chunking,
        tiling=qwen and cfg.memory.vae_tiling,
        cache_tiling=qwen and cfg.memory.cache_encode_tiled,
        chunking_available=qwen and hip,
    )


def _reserved_workspace_floor(vae: VaeMemory, decode: bool, pixels: int, batch: int) -> float:
    """An allocator floor shared by cache, online, preview and setting alternatives."""
    backend = "fused" if vae.attention_backend == "fused" else "chunked" if vae.query_chunking else "math"
    coefficient, cap_mb = _RESERVED_WORKSPACE.get((vae.kind, vae.element_size, decode, backend), (0, None))
    if vae.kind == "kl" and not decode and backend == "fused" and batch > 1:
        coefficient = _KL_BATCH_RESERVED_BYTES_PER_PIXEL
    reserved = pixels * coefficient
    if cap_mb is not None:
        reserved = min(reserved, cap_mb * 2**20)
    # Every caller already includes this margin once in the enclosing phase.
    return max(0.0, batch * reserved - _MARGIN_MB * 2**20)


def _invocation(vae: VaeMemory, decode: bool, width: int, height: int, batch: int, tile: int | None) -> dict:
    """Input residency plus the largest convolution, attention or allocator workspace."""
    element = vae.element_size
    if tile:
        w, h, b = min(width, tile), min(height, tile), 1
    else:
        w, h, b = width, height, batch
    conv = b * w * h * _PIXEL_WORKSPACE[(vae.kind, max(2, element))][decode] * _FRAGMENTATION
    attention = 0.0
    tokens = (w // 8) * (h // 8)
    if vae.attention_backend != "fused":
        queries = min(tokens, _QUERY_CHUNK) if vae.query_chunking else tokens
        attention = b * (queries * tokens * _SCORE_BYTES + tokens * vae.attention_channels * (10 * element + 16))
    # Encoding keeps the FP32 batch and, while tiles run, its full cast copy on the device.
    resident = 0 if decode else batch * width * height * 3 * (4 + (element if tile and vae.kind == "qwen" else 0))
    reserved = _reserved_workspace_floor(vae, decode, w * h, b)
    return {
        "width": width,
        "height": height,
        "batch_size": batch,
        "tile_pixels": tile,
        "attention_tokens": tokens,
        "workspace_mb_estimate": round((resident + max(conv, attention, reserved)) / 2**20, 1),
        "attention_workspace_mb_estimate": round(attention / 2**20, 1),
    }


def _largest(vae: VaeMemory, decode: bool, shapes: dict[tuple[int, int], int], tile_for) -> dict | None:
    calls = [
        _invocation(vae, decode, width, height, batch, tile_for(vae, width, height))
        for (width, height), batch in sorted(shapes.items())
    ]
    return max(calls, key=lambda call: call["workspace_mb_estimate"], default=None)


def _cache_tile(vae: VaeMemory, width: int, height: int) -> int | None:
    if vae.cache_tiling and width * height > CACHE_TILE_THRESHOLD:
        return CACHE_TILE_PIXELS
    return VAE_TILE_PIXELS if vae.tiling else None


def _vae_tile(vae: VaeMemory, width: int, height: int) -> int | None:
    return VAE_TILE_PIXELS if vae.tiling else None


def _phase(vae: VaeMemory, decode: bool, shapes: dict[tuple[int, int], int], tile_for, alternatives) -> dict | None:
    """The largest call with the workspace each applicable setting would leave."""
    largest = _largest(vae, decode, shapes, tile_for)
    if largest is None:
        return None
    reductions = {}
    for path, changed in alternatives:
        call = _largest(changed, decode, shapes, tile_for)
        if call is not None and call["workspace_mb_estimate"] < largest["workspace_mb_estimate"]:
            reductions[path] = call["workspace_mb_estimate"]
    return {
        **largest,
        "dtype": _DTYPES[vae.dtype],
        "attention_backend": vae.attention_backend,
        "query_chunking": vae.query_chunking,
        "weights_mb": round(vae.weights_mb, 1),
        "workspace_reductions_mb": reductions,
    }


def _alternatives(vae: VaeMemory, *, caching: bool, shapes: dict[tuple[int, int], int]):
    """Settings that change this VAE's workspace here, each with the VAE as it would then run."""
    if vae.kind != "qwen":
        return []
    options = []
    if vae.chunking_available and vae.attention_backend != "fused" and not vae.query_chunking:
        options.append(("memory.vae_attention_chunking", replace(vae, query_chunking=True)))
    if not vae.tiling:
        options.append(("memory.vae_tiling", replace(vae, tiling=True)))
    if caching and not vae.cache_tiling and any(w * h > CACHE_TILE_THRESHOLD for w, h in shapes):
        options.append(("memory.cache_encode_tiled", replace(vae, cache_tiling=True)))
    return options


def latent_cache_phase(vae: VaeMemory, shapes: dict[tuple[int, int], int]) -> dict | None:
    """Caching runs before the backbone, text encoder or training state reach the device: only the VAE."""
    phase = _phase(vae, False, shapes, _cache_tile, _alternatives(vae, caching=True, shapes=shapes))
    if phase is not None:
        phase["peak_mb_estimate"] = round(vae.weights_mb + phase["workspace_mb_estimate"] + _MARGIN_MB)
    return phase


def online_encoding_phase(vae: VaeMemory, shapes: dict[tuple[int, int], int]) -> dict | None:
    """Encoding each batch beside the training state; the caller adds that residency."""
    return _phase(vae, False, shapes, _vae_tile, _alternatives(vae, caching=False, shapes=shapes))


def preview_decoding_phase(vae: VaeMemory, shapes: dict[tuple[int, int], int]) -> dict | None:
    """Decoding each preview image beside the training state; the caller adds that residency."""
    return _phase(vae, True, shapes, _vae_tile, _alternatives(vae, caching=False, shapes=shapes))


def preview_shapes(cfg: TrainConfig, align: int) -> dict[tuple[int, int], int]:
    """Preview sizes as the trainer aligns them, one image per decode."""
    sampling = cfg.sampling
    if sampling is None or not sampling.enabled:
        return {}
    sizes = [(prompt.width or sampling.width, prompt.height or sampling.height) for prompt in sampling.prompts]
    if sampling.prompts_file:
        # A prompt file can set its own sizes; the defaults stand in for them here.
        sizes.append((sampling.width, sampling.height))
    return {
        (width // align * align, height // align * align): 1
        for width, height in sizes
        if width >= align and height >= align
    }


# --------------------------------------------------------------------------- runtime log
_PRECISIONS = {torch.float32: "FP32", torch.bfloat16: "BF16", torch.float16: "FP16"}


def _precision(latent: Any) -> str:
    """The VAE's precision: its loaded weights' dtype, else the dtype it loads and encodes in."""
    vae = getattr(latent, "vae", None)
    dtype = None
    if isinstance(vae, torch.nn.Module):
        dtype = next((p.dtype for p in vae.parameters() if p.is_floating_point()), None)
    dtype = dtype or getattr(latent, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        return "unknown"
    return _PRECISIONS.get(dtype, str(dtype).removeprefix("torch."))


def _tiling(latent: Any, *, caching: bool) -> str:
    if getattr(latent, "vae_tiling", False):
        return "on"
    if caching and getattr(latent, "cache_encode_tiled", False):
        return "large-images"
    return "off"


class EncodeMonitor:
    """VAE encode settings and debug records of the memory each phase took.

    Peak allocated and peak reserved are this process's PyTorch counters. "Device in use" is the whole
    GPU as the driver reports it after each call, other programs included. The three are never mixed.
    Caching runs alone on the device, so its peaks start with the phase; an online encode reports the
    run's peaks so far, which the training step shares.
    """

    def __init__(
        self, latent: Any, device: torch.device | str, *, encode_batch: int, training_batch: int, caching: bool
    ) -> None:
        self.latent, self.device = latent, torch.device(device)
        self.encode_batch, self.training_batch, self.caching = encode_batch, training_batch, caching
        self.calls = self.images = self.largest = 0
        self.device_used = self.device_total = 0
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        # Cold allocators start with zero counters; a cache hit need not initialize CUDA.
        if self.cuda and torch.cuda.is_initialized():
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if caching:
                torch.cuda.reset_peak_memory_stats(self.device)

    def wrap(self, encode: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        def counted(pixels: torch.Tensor) -> torch.Tensor:
            if not self.calls:
                log.info(
                    "%s: batch %d, precision %s, tiling %s",
                    "VAE cache encode" if self.caching else "online VAE encode",
                    self.encode_batch,
                    _precision(self.latent),
                    _tiling(self.latent, caching=self.caching),
                )
            latents = encode(pixels)
            self.calls += 1
            self.images += len(pixels)
            self.largest = max(self.largest, len(pixels))
            if self.cuda:
                if self.device.index is None:
                    self.device = torch.device("cuda", torch.cuda.current_device())
                free, total = torch.cuda.mem_get_info(self.device)
                self.device_used, self.device_total = max(self.device_used, total - free), total
            return latents

        return counted

    def finish(self) -> None:
        if not self.calls:
            return
        log.debug(
            "%s: images %d, VAE calls %d, at most %d per call",
            "VAE encode finished" if self.caching else "online VAE encode, first training batch",
            self.images,
            self.calls,
            self.largest,
        )
        if not self.cuda:
            return
        log.debug(
            "VAE encode memory on %s: peak allocated %.2f GiB, peak reserved %.2f GiB (this process, %s); "
            "device in use up to %.2f of %.2f GiB (all processes)",
            self.device,
            torch.cuda.max_memory_allocated(self.device) / 2**30,
            torch.cuda.max_memory_reserved(self.device) / 2**30,
            "this phase" if self.caching else "run so far",
            self.device_used / 2**30,
            self.device_total / 2**30,
        )
