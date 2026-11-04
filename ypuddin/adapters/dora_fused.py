"""Resolve and freeze the standard DoRA execution backend before training."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

KERNEL_REVISION = "standard-linear-fp32-r1"
_VERIFIED: set[tuple[str, str, str]] = set()


def _layers(adapters):
    from .components import ComponentAdapterSet

    components = adapters.components.values() if isinstance(adapters, ComponentAdapterSet) else (adapters,)
    for component in components:
        for name, layer in getattr(component, "layers", {}).items():
            if layer.dora is not None and layer.dora.compute_mode == "standard_fused":
                yield component.export_key(name), layer


def _verify_cuda(device: torch.device, axis: str) -> None:
    key = (str(device), axis, KERNEL_REVISION)
    if key in _VERIFIED:
        return
    from ._dora_fused_kernels import standard_dora
    from .dora import decompose

    eps = torch.finfo(torch.float32).eps
    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        shape = (1, 4) if axis == "input" else (3, 1)
        for case in ("ordinary", "epsilon", "zero"):
            merged = torch.arange(1, 13, device=device, dtype=torch.float32).reshape(3, 4) / 19
            if case != "ordinary":
                merged.zero_()
                if case == "epsilon":
                    if axis == "input":
                        merged[0, :] = eps
                    else:
                        merged[:, 0] = eps
            base = (merged * .25).requires_grad_()
            delta = (merged * .75).requires_grad_()
            magnitude = torch.ones(shape, device=device, requires_grad=True)
            leaves = (base, delta, magnitude)
            expected = decompose(base + delta, magnitude)
            reference = torch.autograd.grad(expected.sum(), leaves)
            actual = standard_dora(*leaves, axis=axis)
            gradients = torch.autograd.grad(actual.sum(), leaves)
            for value, target in zip((actual, *gradients), (expected, *reference), strict=True):
                torch.testing.assert_close(value, target, rtol=1e-4, atol=1e-6)
                if not torch.isfinite(value).all():
                    raise RuntimeError("DoRA 融合内核预检查产生了非有限值")
    torch.cuda.synchronize(device)
    _VERIFIED.add(key)


def prepare(adapters, device: torch.device, *, resume: str | Path | None = None) -> dict:
    from .dora_contract import STATE_KEY, compute_contract, validate_resume_contract

    layers = list(_layers(adapters))
    if not layers:
        return {"triton_layers": 0, "torch_layers": 0, "fallback_reasons": []}
    device = torch.device(device)
    saved = None
    if resume:
        metadata = json.loads((Path(resume).expanduser() / "state.json").read_text(encoding="utf-8"))
        saved = metadata.get("progress", {}).get("extra", {}).get(STATE_KEY)
        if (not isinstance(saved, dict) or saved.get("mode") != "standard_fused"
                or saved.get("version") != 2 or saved.get("kernel_revision") != KERNEL_REVISION):
            raise ValueError("恢复点缺少匹配的 DoRA 融合后端记录，不能继续训练")
        validate_resume_contract(compute_contract(adapters), saved, allow_unresolved_fused=True)
    reports, reasons = [], set()
    for key, layer in layers:
        shape = tuple(layer.adapter.weight_shape)
        reason = None
        if device.type != "cuda" or not torch.version.cuda or torch.version.hip:
            reason = "当前设备使用 PyTorch 标准计算"
        elif len(shape) != 2 or not all(1 <= value <= 4096 for value in shape):
            reason = "当前权重形状使用 PyTorch 标准计算"
        elif importlib.util.find_spec("triton") is None:
            reason = "未安装 Triton，使用 PyTorch 标准计算"
        backend = "torch" if reason else "triton"
        if saved is not None:
            frozen = saved["layers"][key][2]
            if not isinstance(frozen, dict) or frozen.get("backend") not in {"torch", "triton"}:
                raise ValueError("恢复点中的 DoRA 后端记录无效")
            backend = frozen["backend"]
            if backend == "triton" and reason:
                raise ValueError(f"原训练使用 DoRA 融合内核，当前无法恢复：{reason}")
            if backend == "torch":
                reason = "沿用恢复点的 PyTorch 标准计算"
        runtime = {"backend": backend, "device_type": device.type, "torch": str(torch.__version__)}
        if backend == "triton":
            import triton

            _verify_cuda(device, layer.dora.axis)
            runtime["triton"] = triton.__version__
            runtime["cuda"] = torch.version.cuda
            runtime["capability"] = list(torch.cuda.get_device_capability(device))
        layer.dora._fused_runtime = runtime
        reports.append(backend)
        if reason:
            reasons.add(reason)
    if saved is not None:
        validate_resume_contract(compute_contract(adapters), saved)
    return {"triton_layers": reports.count("triton"), "torch_layers": reports.count("torch"),
            "fallback_reasons": sorted(reasons)}


def rescale_pair(dora, base, delta):
    runtime = getattr(dora, "_fused_runtime", None)
    if not runtime:
        raise RuntimeError("DoRA 融合后端尚未在训练准备阶段确定")
    base, delta, magnitude = base.float(), delta.float(), dora.dora_scale.float()
    if runtime["backend"] == "torch":
        from .dora import decompose

        return decompose(base + delta, magnitude)
    from ._dora_fused_kernels import standard_dora

    return standard_dora(base, delta, magnitude, axis=dora.axis)
