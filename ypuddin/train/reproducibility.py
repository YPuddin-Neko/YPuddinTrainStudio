"""Configure deterministic kernels before model loading or GPU computation."""

from __future__ import annotations

import os

import torch

from ypuddin.runtime_profiles import current_profile

_sdp_defaults: dict[str, bool] | None = None


def _configure_dtk_sdpa(enabled: bool) -> None:
    global _sdp_defaults
    cuda = torch.backends.cuda
    if enabled:
        if _sdp_defaults is None:
            _sdp_defaults = {
                name: getattr(cuda, f"{name}_sdp_enabled")()
                for name in ("flash", "mem_efficient", "math", "cudnn")
                if hasattr(cuda, f"{name}_sdp_enabled")
            }
        # Opt-in candidate for DTK reproducibility: disable fused native SDPA,
        # including direct F.scaled_dot_product_attention calls. Hardware traces
        # have not isolated SDPA as the source of forward differences. This does
        # not replace explicitly selected third-party backends.
        for name in _sdp_defaults:
            getattr(cuda, f"enable_{name}_sdp")(name == "math")
    elif _sdp_defaults is not None:
        for name, value in _sdp_defaults.items():
            getattr(cuda, f"enable_{name}_sdp")(value)
        _sdp_defaults = None


def configure_reproducibility(enabled: bool, device: torch.device) -> None:
    if enabled and device.type == "cuda" and not torch.version.hip:
        # cuBLAS reads this on first use. Preserve an explicitly supplied setting;
        # PyTorch will reject an incompatible one instead of silently ignoring it.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(enabled, warn_only=False)
    torch.backends.cudnn.deterministic = enabled
    if enabled:
        torch.backends.cudnn.benchmark = False
    _configure_dtk_sdpa(enabled and device.type == "cuda" and current_profile() == "linux-dtk")


def validate_resume_reproducibility(enabled: bool, saved: bool | None) -> None:
    if saved is None:
        if enabled:
            raise ValueError(
                "此训练状态未记录确定性计算设置，旧版本默认关闭。"
                "请关闭“可复现训练”（loop.deterministic=false）后恢复；"
                "若要启用，请从已有权重新建训练。"
            )
    elif saved != enabled:
        setting = "开启" if saved else "关闭"
        raise ValueError(
            f"恢复训练需要保持原来的计算方式，请{setting}“可复现训练”（loop.deterministic）后重试。"
        )
