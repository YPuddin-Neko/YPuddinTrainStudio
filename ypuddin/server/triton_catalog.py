"""Which Triton fits this runtime: the optional Triton row of the environment page.

Triton compiles GPU kernels for ``torch.compile``, some optional xFormers kernels and the
FLOP counting of Triton kernels; training runs without it.

* Linux CUDA: PyTorch's wheels pin their Triton release (``triton==X`` in the wheel
  metadata), so it is installed with PyTorch and not managed here.
* Windows CUDA: the separate ``triton-windows`` build. Each PyTorch minor version is only
  guaranteed to work with one Triton minor version (maintainer table:
  https://github.com/triton-lang/triton-windows/tree/readme#3-pytorch, checked 2026-10-04).
  Each pin is the newest ``triton-windows`` build of that Triton minor on PyPI on that date;
  the wheels bundle their CUDA toolchain and C compiler, so nothing is compiled on install.
* DTK: the vendor's Triton, installed with the vendor PyTorch (``…torch271`` for 2.7.1).
* Apple GPUs and CPUs: no Triton.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

PACKAGES = ("triton", "triton-windows")
DOCS_URL = "https://github.com/triton-lang/triton"
WINDOWS_DOCS_URL = "https://github.com/triton-lang/triton-windows"
# PyTorch minor version -> triton-windows build of the matching Triton minor version.
WINDOWS_BUILDS = {
    "2.4": "3.1.0.post17",
    "2.5": "3.1.0.post17",
    "2.6": "3.2.0.post21",
    "2.7": "3.3.1.post21",
    "2.8": "3.4.0.post21",
    "2.9": "3.5.1.post24",
    "2.10": "3.6.0.post26",
    "2.11": "3.6.0.post26",
    "2.12": "3.7.1.post27",
    "2.13": "3.7.1.post27",
    "2.14": "3.8.0.post29",
}


def _minor(version: str) -> str | None:
    try:
        release = Version(str(version)).release
    except InvalidVersion:
        return None
    return f"{release[0]}.{release[1]}" if len(release) >= 2 else None


def _capabilities(runtime: dict[str, Any]) -> list[tuple[int, ...]]:
    found = [
        tuple(gpu["compute_capability"][:2])
        for gpu in runtime.get("gpus") or []
        if gpu.get("cuda_available") and gpu.get("compute_capability")
    ]
    if not found and runtime.get("gpu_capability"):
        found = [tuple(runtime["gpu_capability"][:2])]
    return found


def _gpu_reason(triton: str, runtime: dict[str, Any]) -> str | None:
    """Supported GPUs per the maintainer's table: Volta/Turing only up to Triton 3.2, RTX 50 only
    from Triton 3.3 with PyTorch 2.7 and CUDA 12.8, nothing older than Volta."""
    release = Version(triton).release[:2]
    try:
        cuda = Version(str(runtime.get("cuda_runtime") or "0")).release[:2]
        torch = Version(str(runtime.get("torch") or "0")).release[:2]
    except InvalidVersion:
        cuda = torch = (0, 0)
    for capability in _capabilities(runtime):
        if capability < (7, 0):
            return "triton_unsupported_gpu"
        if release >= (3, 3) and capability < (8, 0):
            return "triton_requires_ampere"
        if capability >= (10, 0) and (release < (3, 3) or torch < (2, 7) or cuda < (12, 8)):
            return "triton_requires_cuda128"
    return None


def windows_build(runtime: dict[str, Any]) -> tuple[str | None, str | None]:
    """The triton-windows version for this PyTorch and GPU, or why there is none."""
    if runtime.get("platform") != "Windows" or runtime.get("hip_runtime") or not runtime.get("cuda_runtime"):
        return None, "triton_windows_cuda_only"
    if not runtime.get("cuda_available"):
        return None, "requires_cuda"
    pin = WINDOWS_BUILDS.get(_minor(str(runtime.get("torch") or "")) or "")
    if pin is None:
        return None, "triton_no_build"
    reason = _gpu_reason(pin, runtime)
    return (None, reason) if reason else (pin, None)


def windows_mismatch(version: str, runtime: dict[str, Any]) -> str | None:
    """Why a triton-windows version does not fit this PyTorch: another Triton minor than the pin."""
    pin, reason = windows_build(runtime)
    if reason:
        return reason
    return None if _minor(version) == _minor(pin) else "triton_version_mismatch"


def torch_requirement() -> Requirement | None:
    """The Triton requirement in the installed PyTorch's metadata for this platform."""
    try:
        requires = importlib.metadata.requires("torch") or []
    except importlib.metadata.PackageNotFoundError:
        return None
    for raw in requires:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) == "triton" and (
            requirement.marker is None or requirement.marker.evaluate()
        ):
            return requirement
    return None


def _requirement_text(requirement: Requirement | None) -> str | None:
    if requirement is None or not str(requirement.specifier):
        return None
    pins = [spec.version for spec in requirement.specifier if spec.operator in ("==", "===")]
    return pins[0] if len(requirement.specifier) == 1 and pins else str(requirement.specifier)


def _vendor_label(torch: str) -> str | None:
    """``torch271`` for the vendor PyTorch 2.7.1: the label the vendor's Triton build carries."""
    try:
        return "torch" + Version(torch).base_version.replace(".", "")
    except InvalidVersion:
        return None


def vendor_version(torch: str) -> str | None:
    """The reviewed vendor Triton of this DTK PyTorch, from the reviewed download sets."""
    from . import dtk_catalog

    try:
        base = Version(torch).base_version
    except InvalidVersion:
        return None
    return next(
        (version for package, version, _ in dtk_catalog._guidance_wheels(base) if package == "triton"), None
    )


def _platform(runtime: dict[str, Any], profile: str) -> str:
    if profile.endswith("-cpu") or runtime.get("platform") == "Darwin" or profile == "macos-mps":
        return "none"
    if profile == "linux-dtk" or runtime.get("hip_runtime"):
        return "dtk"
    if not runtime.get("cuda_runtime"):
        return "none"
    return "windows" if runtime.get("platform") == "Windows" else "linux"


def package_state(name: str, runtime: dict[str, Any], versions: dict[str, str], profile: str) -> tuple[bool, str]:
    """(whether this page installs it here, reason) for one Triton row."""
    where = _platform(runtime, profile)
    installed = versions.get(name)
    if name == "triton-windows":
        if where != "windows":
            return False, "triton_windows_cuda_only"
        pin, reason = windows_build(runtime)
        if reason:
            return False, reason
        if installed and _minor(installed) != _minor(pin):
            return True, "triton_version_mismatch"
        return True, "supported"
    if where == "none":
        return False, "triton_unavailable"
    if where == "windows":
        return False, "triton_windows_build"
    if not installed:
        return False, "triton_missing"
    if where == "dtk":
        label = _vendor_label(str(runtime.get("torch") or ""))
        vendor = "das" in installed and (label is None or installed.endswith(label))
        return False, "vendor_build" if vendor else "triton_version_mismatch"
    requirement = torch_requirement()
    if requirement is not None and not requirement.specifier.contains(installed, prereleases=True):
        return False, "triton_version_mismatch"
    return False, "bundled_with_torch"


def matching_version(runtime: dict[str, Any], profile: str) -> tuple[str, dict[str, Any]] | None:
    """The Triton version this runtime pairs with, shown beside the installed one."""
    where = _platform(runtime, profile)
    if where == "windows":
        return "triton-windows", {"version": windows_build(runtime)[0], "source": "pinned", "error": None}
    if where == "linux":
        return "triton", {"version": _requirement_text(torch_requirement()), "source": "pinned", "error": None}
    if where == "dtk":
        return "triton", {"version": vendor_version(str(runtime.get("torch") or "")), "source": "dtk", "error": None}
    return None


def message(reason: str, runtime: dict[str, Any], version: str | None = None) -> str:
    torch = str(runtime.get("torch") or "未知版本")
    pin = WINDOWS_BUILDS.get(_minor(torch) or "")
    messages = {
        "triton_windows_cuda_only": "triton-windows 只用于 Windows 的 NVIDIA CUDA 环境。",
        "requires_cuda": "Triton 需要能使用 NVIDIA CUDA 的 PyTorch。",
        "triton_no_build": f"PyTorch {torch} 没有已核对的配套 triton-windows 版本。",
        "triton_unsupported_gpu": "这块显卡的架构早于 Volta，Triton 不支持。",
        "triton_requires_ampere": f"与 PyTorch {torch} 配套的 Triton {_minor(pin or '') or ''} 需要 RTX 30 系（Ampere）或更新的显卡。",
        "triton_requires_cuda128": "RTX 50 系显卡上的 Triton 需要 PyTorch 2.7 及以上、CUDA 12.8 及以上的构建。",
        "triton_version_mismatch": f"triton-windows {version} 与 PyTorch {torch} 不配套，需要 {_minor(pin or '') or ''}.x 系列（{pin}）。",
    }
    return messages.get(reason, reason)
