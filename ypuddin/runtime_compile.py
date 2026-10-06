"""Dependencies for the CUDA/HIP model-block compilation path."""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _probe_triton_import(origin: str, modified: int, interval: int) -> str | None:
    """Import binaries in a short-lived process so environment updates can replace them."""
    script = (
        "import json\n"
        "try:\n import triton\n error = None\n"
        "except Exception as exc:\n error = type(exc).__name__ + ': ' + str(exc)\n"
        "print('YPUDDIN_TRITON=' + json.dumps(error))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=20,
        )
    except subprocess.TimeoutExpired:
        return "Triton 加载检查超时。"
    except OSError:
        return "无法启动 Triton 加载检查。"
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("YPUDDIN_TRITON=") and result.returncode == 0:
            try:
                value = json.loads(line[len("YPUDDIN_TRITON="):])
            except ValueError:
                break
            if value is None or isinstance(value, str):
                return value
    return f"Triton 加载检查异常退出（退出码 {result.returncode}）。"


def _isolated_triton_error() -> str | None:
    spec = importlib.util.find_spec("triton")
    origin = str(spec.origin or "") if spec else ""
    try:
        modified = Path(origin).stat().st_mtime_ns
    except OSError:
        modified = 0
    # Reinstallation normally changes the package file; expire unchanged paths as well.
    return _probe_triton_import(origin, modified, int(time.monotonic() // 10))


def _load_error(reason: str) -> CompileEnvironmentError:
    detail = " ".join(reason.split())[:180]
    return CompileEnvironmentError(
        "当前环境无法加载 Triton，无法编译模型。"
        "请在“运行环境”检查 PyTorch 与 Triton 的配套版本，或关闭“编译模型”。原因：" + detail
    )


class CompileEnvironmentError(ValueError):
    """Compilation cannot start in the current Python environment."""


def triton_installed() -> bool:
    """Windows and DTK distributions also expose the ``triton`` import package."""
    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, ValueError):
        return False


def validate_compile_environment(enabled: bool, device_type: str | None, *, isolated: bool = False) -> None:
    # CPU/MPS run eagerly; an offline plan has no execution environment to check.
    if not enabled or device_type != "cuda":
        return
    if not triton_installed():
        raise CompileEnvironmentError(
            "当前环境未安装 Triton，无法编译模型。"
            "请在“运行环境”检查 PyTorch 与 Triton 的配套版本，或关闭“编译模型”。"
        )
    if isolated:
        if reason := _isolated_triton_error():
            raise _load_error(reason)
        return
    try:
        importlib.import_module("triton")
    except Exception as error:  # A broken binary package can also fail during module initialization.
        raise _load_error(str(error)) from error
