"""Dependencies for the CUDA/HIP model-block compilation path."""

from __future__ import annotations

import importlib
import importlib.util


class CompileEnvironmentError(ValueError):
    """Compilation cannot start in the current Python environment."""


def triton_installed() -> bool:
    """Windows and DTK distributions also expose the ``triton`` import package."""
    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, ValueError):
        return False


def validate_compile_environment(enabled: bool, device_type: str | None) -> None:
    # CPU/MPS run eagerly; an offline plan has no execution environment to check.
    if not enabled or device_type != "cuda":
        return
    if not triton_installed():
        raise CompileEnvironmentError(
            "当前环境未安装 Triton，无法编译模型。"
            "请安装与当前 PyTorch 配套的 Triton，或关闭“编译模型”。"
        )
    try:
        importlib.import_module("triton")
    except Exception as error:  # A broken binary package can also fail during module initialization.
        raise CompileEnvironmentError(
            "当前环境无法加载 Triton，无法编译模型。"
            "请修复 Triton 环境，或关闭“编译模型”。原因：" + str(error)
        ) from error
