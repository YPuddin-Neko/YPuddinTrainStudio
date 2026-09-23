"""Pinned mtlattn wheels and supported deployment requirements.

PyPI metadata declares only ``torch``. The publisher documents that these
extension wheels were built against Torch 2.13; package metadata alone therefore
cannot establish compatibility. Keep optional installation from changing Torch.
"""

from packaging.version import InvalidVersion, Version

from ypuddin.runtime_profiles import current_profile

VERSION = "0.4.1"
DOCS_URL = "https://github.com/lastowl/mtlattn"
# PyPI release metadata inspected 2026-09-23. Only launcher-supported Python ABIs.
WHEELS = {
    "mtlattn-0.4.1-cp311-cp311-macosx_13_0_universal2.whl": (
        434749,
        "61d84ad35d371f4ec5de99ef806a2bf82032b350cfbbbd65b7819177fcd9e9a8",
    ),
    "mtlattn-0.4.1-cp312-cp312-macosx_13_0_universal2.whl": (
        436477,
        "5f082b8d255bc2fdbf6b1fa2d201e2288d54b9719ce44ed7ab5ae501842a9135",
    ),
}

MESSAGES = {
    "metal_requires_apple_silicon": "Metal FlashAttention 需要 Apple Silicon macOS。",
    "metal_requires_mps_profile": "Metal FlashAttention 需要 Apple MPS 环境。",
    "metal_requires_macos_15": "本版 Metal FlashAttention 需要 macOS 15 或更新版本。",
    "metal_requires_python_311_312": "本版 Metal FlashAttention 需要 Python 3.11 或 3.12。",
    "metal_requires_torch_2_13": "mtlattn 0.4.1 的预编译包需要 PyTorch 2.13.x；请先创建并切换至兼容的 Apple MPS 环境。",
    "metal_requires_mps": "当前 PyTorch 无法使用 Apple MPS；Metal FlashAttention 检测或安装已停止。",
}


def incompatibility(runtime: dict, profile: str) -> str | None:
    if (
        runtime.get("platform") != "Darwin"
        or str(runtime.get("machine", "")).lower() not in {"arm64", "aarch64"}
        or runtime.get("cuda_runtime")
        or runtime.get("hip_runtime")
    ):
        return "metal_requires_apple_silicon"
    if profile not in {"legacy", "macos-mps"}:
        return "metal_requires_mps_profile"
    try:
        if Version(runtime.get("macos_version") or "0") < Version("15"):
            return "metal_requires_macos_15"
        if Version(runtime.get("python") or "0").release[:2] not in {(3, 11), (3, 12)}:
            return "metal_requires_python_311_312"
        if Version(runtime.get("torch") or "0").release[:2] != (2, 13):
            return "metal_requires_torch_2_13"
    except InvalidVersion:
        return "metal_requires_torch_2_13"
    if not runtime.get("mps_available"):
        return "metal_requires_mps"
    return None


def validate_release_file(filename: str, sha256: str, size: int | None = None) -> None:
    expected = WHEELS.get(filename)
    if expected is None:
        raise ValueError("Metal FlashAttention only accepts the reviewed mtlattn 0.4.1 macOS wheels")
    if sha256.lower() != expected[1] or (size is not None and size != expected[0]):
        raise ValueError("Metal FlashAttention wheel differs from the reviewed PyPI SHA256 or size")


def probe_metal_attention(torch_module=None) -> dict:
    """Bounded actual Metal backward probe; importing a package is insufficient."""
    result = {"importable": False, "kernel_tested": False, "error": None}
    if current_profile().endswith("-cpu"):
        result["error"] = MESSAGES["metal_requires_mps_profile"]
        return result
    try:
        from ypuddin.models.metal_attention import (
            metal_flash_eligible,
            metal_flash_sdpa,
            require_metal_flash,
        )

        require_metal_flash("mps")  # Validates runtime before importing the native extension.
        result["importable"] = True
        if torch_module is None:
            import torch as torch_module
        torch = torch_module
        for shape in ((1, 2, 32, 64), (1, 2, 64, 128)):
            with torch.enable_grad():
                q, k, v = [
                    torch.randn(*shape, device="mps", dtype=torch.float32, requires_grad=True)
                    for _ in range(3)
                ]
                if not metal_flash_eligible(q, k, v):
                    raise RuntimeError("Metal FlashAttention probe inputs are not eligible for its kernel")
                y = metal_flash_sdpa(q, k, v)
                pending, seen = [y.grad_fn], set()
                flash_backward = False
                while pending:
                    node = pending.pop()
                    if node is None or node in seen:
                        continue
                    seen.add(node)
                    flash_backward |= type(node).__name__ == "_VarlenAttnFnBackward"
                    pending.extend(child for child, _ in node.next_functions)
                if not flash_backward:
                    raise RuntimeError("Metal FlashAttention probe did not use the native varlen backward")
                y.square().mean().backward()
                torch.mps.synchronize()
                if not bool(torch.isfinite(y).all()) or any(
                    tensor.grad is None or not bool(torch.isfinite(tensor.grad).all())
                    for tensor in (q, k, v)
                ):
                    raise RuntimeError("Metal FlashAttention output or Q/K/V gradient is not finite")
        result["kernel_tested"] = True
    except Exception as exc:
        result["error"] = str(exc)[-1500:]
    return result
