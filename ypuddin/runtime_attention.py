"""Bounded attention checks and narrowly identified runtime dependency diagnostics."""

from __future__ import annotations

import time
from typing import Any

from ypuddin.runtime_profiles import current_profile


class AttentionEnvironmentError(RuntimeError):
    """A diagnosed runtime dependency problem; the original exception remains its cause."""

    code = "hip_sdpa_flash_library_missing"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(
            "当前 DTK / HIP 版 PyTorch 的 SDPA 计算缺少 FlashAttention 动态库。"
            "请在“设置 → 运行环境 → FlashAttention 2”安装与当前 DTK、PyTorch 匹配的厂商包，"
            "完成后重启服务并重新检测。原始原因：" + detail
        )


def attention_dependency_error(
    exc: BaseException, *, hip_runtime: str | None
) -> AttentionEnvironmentError | None:
    if (
        hip_runtime
        and isinstance(exc, RuntimeError)
        and "no matching libraries found for flash_attn_2_cuda" in str(exc).casefold()
    ):
        return AttentionEnvironmentError(str(exc))
    return None


def probe_sdpa(torch_module=None) -> dict[str, Any]:
    """Check small forward/backward SDPA paths in the existing isolated probe process.

    No kernel backend is forced: CUDA/HIP checks available half precisions; MPS
    checks FP32, matching the current training path. This does not identify or
    certify a Flash kernel, every model shape or every card. Inputs use under
    1 MiB per case; no model is loaded. Explicit CPU profiles never allocate GPU
    tensors, and the Apple profile never probes a CUDA device.
    """
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    hip = getattr(torch.version, "hip", None)
    result = {
        "status": "not_tested",
        "reason": None,
        "error": None,
        "detail": None,
        "checked_at": time.time(),
        "device": None,
        "device_name": None,
        "torch": str(torch.__version__),
        "hip_runtime": hip,
        "checks": [],
    }
    profile = current_profile()
    if profile.endswith("-cpu"):
        result["reason"] = "cpu_profile"
        return result
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    if profile == "macos-mps" or (
        profile == "legacy" and not torch.cuda.is_available() and mps_available
    ):
        if not mps_available:
            result["reason"] = "mps_unavailable"
            return result
        result.update(device="mps", device_name="Apple MPS")
        dtypes = [("fp32", torch.float32)]
        synchronize = torch.mps.synchronize
    else:
        if not torch.cuda.is_available():
            result["reason"] = "cuda_or_hip_unavailable"
            return result
        result.update(device=f"cuda:{torch.cuda.current_device()}", device_name=torch.cuda.get_device_name())
        dtypes = [("fp16", torch.float16)]
        if torch.cuda.is_bf16_supported():
            dtypes.append(("bf16", torch.bfloat16))
        synchronize = torch.cuda.synchronize
    for dtype_name, dtype in dtypes:
        for shape in ((1, 2, 32, 64), (1, 2, 256, 128)):
            check = {"dtype": dtype_name, "shape": list(shape), "passed": False, "error": None}
            q = k = v = y = None
            try:
                with torch.enable_grad():
                    q, k, v = [
                        torch.randn(*shape, device=result["device"], dtype=dtype, requires_grad=True)
                        for _ in range(3)
                    ]
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                    y.float().square().mean().backward()
                    synchronize()
                    if not bool(torch.isfinite(y).all()):
                        raise RuntimeError("SDPA output is not finite")
                    if any(t.grad is None or not bool(torch.isfinite(t.grad).all()) for t in (q, k, v)):
                        raise RuntimeError("SDPA gradient is missing or not finite")
                check["passed"] = True
            except Exception as exc:
                detail = str(exc)[-1500:]
                diagnosed = attention_dependency_error(
                    exc, hip_runtime=hip if result["device"].startswith("cuda:") else None
                )
                check["error"] = str(diagnosed) if diagnosed else detail
                result.update(
                    status="failed",
                    reason=diagnosed.code if diagnosed else "sdpa_probe_failed",
                    error=check["error"],
                    detail=detail,
                )
                result["checks"].append(check)
                # Continuing after a failed GPU kernel/OOM can obscure its cause.
                return result
            finally:
                del q, k, v, y
            result["checks"].append(check)
    result["status"] = "passed"
    return result
