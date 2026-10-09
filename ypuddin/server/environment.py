"""Managed optional dependencies, with a reviewed wheel plan and a protected Torch runtime.

No shell, source builds, arbitrary package names, or implicit framework upgrades are exposed.
The installer runs outside the server process; optional CUDA libraries are probed there too so
Windows does not retain their DLLs in the service. Application restart releases maintenance.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Literal
from urllib.request import url2pathname

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ypuddin.package_sources import pypi_sources, run_sources, torch_sources
from ypuddin.runtime_profiles import current_profile, profile_root

from . import dtk_catalog, metal_attention_catalog, triton_catalog, windows_attention_catalog
from .db import Database, new_id, now
from .download_sources import probe_options
from .errors import ApiError

CATALOG = {
    "torch": ("torch", None, "https://pytorch.org/get-started/locally/"),
    "xformers": (
        "xformers.ops",
        "xformers",
        "https://github.com/facebookresearch/xformers#installing-xformers",
    ),
    "flash-attn": (
        "flash_attn",
        "flash_attn",
        "https://github.com/Dao-AILab/flash-attention#installation-and-features",
    ),
    "sageattention": ("sageattention", "sage", "https://github.com/thu-ml/SageAttention#installation"),
    "mtlattn": ("mtlattn", "metal_flash", metal_attention_catalog.DOCS_URL),
    "nvidia-ml-py": ("pynvml", None, "https://pypi.org/project/nvidia-ml-py/"),
    "tensorboard": ("tensorboard", None, "https://www.tensorflow.org/tensorboard/get_started"),
    "schedulefree": ("schedulefree", None, "https://github.com/facebookresearch/schedule_free"),
    # The 8-bit optimizers (AdamW 8-bit, Lion 8-bit).
    "bitsandbytes": ("bitsandbytes", None, "https://github.com/bitsandbytes-foundation/bitsandbytes"),
    # Automatic tagging and masks. Both builds provide the same module; one is installed at a time.
    "onnxruntime": ("onnxruntime", None, "https://onnxruntime.ai/docs/install/"),
    "onnxruntime-gpu": ("onnxruntime", None, "https://onnxruntime.ai/docs/install/#python-installs"),
    # Kernel compiler for torch.compile and optional extension kernels: installed with PyTorch on Linux
    # and DTK, the matching triton-windows build on Windows (triton_catalog).
    "triton": ("triton", None, triton_catalog.DOCS_URL),
    "triton-windows": ("triton", None, triton_catalog.WINDOWS_DOCS_URL),
}
ONNX_RUNTIMES = ("onnxruntime", "onnxruntime-gpu")
ATTENTION = ("auto", "sdpa", "xformers", "flash_attn", "sage", "metal_flash")
MUTATING = ("installing", "verifying")
BUSY = ("planning", *MUTATING)
MAX_WHEEL_BYTES = 2 * 1024**3
# Finished installs stay in the task center this long; failures of this run stay until dismissed.
TASK_KEEP_SECONDS = 600
PACKAGE_LABELS = {
    "xformers": "xFormers", "flash-attn": "FlashAttention 2", "sageattention": "SageAttention",
    "mtlattn": "Metal FlashAttention", "onnxruntime": "ONNX Runtime", "onnxruntime-gpu": "ONNX Runtime GPU",
    "triton-windows": "Triton",
}
_TASK_FIELDS = {"id", "package", "action", "status", "phase", "downloaded_bytes", "total_bytes", "restart_required",
                "dismissed_at", "created_at", "updated_at", "error"}


# Why a reviewed prebuilt wheel does not fit this machine, worded as in the wheel lists on the environment page.
_WHEEL_REASONS = {
    "dtk_profile_required": "请使用独立的 Linux DTK 启动入口",
    "requires_linux_x86_64": "需要 Linux x86_64 环境",
    "hip_runtime_unavailable": "当前 HIP 运行时无法访问显卡",
    "dtk_version_mismatch": "与当前 DTK 版本不匹配",
    "integrity_verification_pending": "尚未完成官方包完整性核验",
    "requires_windows_cuda": "需要 NVIDIA CUDA 环境",
    "requires_windows_x86_64": "需要 Windows 或 Linux x86_64 系统",
    "platform_mismatch": "与当前系统平台不匹配",
    "cuda_runtime_unavailable": "当前 PyTorch 无法使用 CUDA",
    "runtime_version_unrecognized": "无法识别当前运行时版本",
    "python_abi_mismatch": "与当前 Python 版本不匹配",
    "torch_version_mismatch": "与当前 PyTorch 版本不匹配",
    "cuda_version_mismatch": "CUDA 构建版本不匹配",
}


def wheel_reason(reason: str) -> str:
    if reason.startswith("requires_package:"):
        return f"需要先安装 {reason.removeprefix('requires_package:')}。"
    if reason.startswith("requires_torch_runtime_api"):
        return f"当前 PyTorch 缺少此包需要的功能，要求 {reason.removeprefix('requires_torch_runtime_api')}。"
    return _WHEEL_REASONS.get(reason, reason) + "。"


def task_error(message: str | None, fallback: str) -> str:
    """A failure reason for the task center: the recorded one when it is written in Chinese,
    otherwise a pointer to the page that shows the installer's own output."""
    return message if message and re.search(r"[\u4e00-\u9fff]", message) else fallback


class EnvironmentError(ApiError):
    def __init__(self, status: int, message: str):
        super().__init__(message, status=status, code="environment.request")


class EnvironmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package: Literal[
        "xformers",
        "flash-attn",
        "sageattention",
        "mtlattn",
        "nvidia-ml-py",
        "tensorboard",
        "schedulefree",
        "bitsandbytes",
        "onnxruntime",
        "onnxruntime-gpu",
        "triton-windows",
    ]
    action: Literal["install", "repair", "uninstall"] = "install"
    version: str | None = None
    wheel_id: str | None = None
    vendor_wheel_id: str | None = None

    @model_validator(mode="after")
    def one_wheel_source(self):
        if self.wheel_id and self.vendor_wheel_id:
            raise ValueError("上传的 wheel 和预编译 wheel 只能选择一个。")
        if self.action == "uninstall" and (self.wheel_id or self.vendor_wheel_id):
            raise ValueError("卸载时不能指定 wheel。")
        if self.package == "mtlattn" and self.action != "uninstall":
            if self.version not in (None, metal_attention_catalog.VERSION):
                raise ValueError("Metal FlashAttention 目前只支持 mtlattn 0.4.1。")
            self.version = metal_attention_catalog.VERSION
        return self

    @field_validator("version")
    @classmethod
    def version_only(cls, value):
        if value is None or not value.strip():
            return None
        try:
            return str(Version(value.strip()))
        except InvalidVersion as exc:
            raise ValueError("版本号需要是确切的包版本，例如 0.0.32.post2。") from exc


class EnvironmentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attention_default: Literal["auto", "sdpa", "xformers", "flash_attn", "sage", "metal_flash"] = "auto"


class EnvironmentRuntime(BaseModel):
    environment_profile: str = "legacy"
    python: str
    python_executable: str
    platform: str
    machine: str
    macos_version: str | None = None
    distribution: str | None = None
    distribution_id: str | None = None
    distribution_version: str | None = None
    kernel_release: str | None = None
    glibc_version: str | None = None
    dtk_root: str | None = None
    installed_dtk: str | None = None
    driver_version: str | None = None
    torch: str
    cuda_runtime: str | None
    hip_runtime: str | None = None
    compute_backend: Literal["cuda", "hip", "mps", "cpu"] = "cpu"
    cuda_available: bool
    mps_available: bool
    gpu_capability: list[int] | None
    cxx11_abi: bool | None = None
    gpus: list[dict[str, Any]]
    virtual_environment: bool
    cuda_device_count: int = 0
    distributed_available: bool = False
    nccl_available: bool = False
    gloo_available: bool = False
    multi_gpu_backend: Literal["nccl", "gloo"] | None = None
    multi_gpu_probe_required: bool = False
    multi_gpu_training: bool = False
    training_device_policy: Literal["single_device", "exclusive_devices"] = "single_device"
    cuda_applicable: bool = True
    nccl_applicable: bool = True
    distributed_purpose: str = "multi_process_communication"


class EnvironmentPackage(BaseModel):
    name: str
    version: str | None
    backend: str | None
    docs_url: str
    supported: bool
    reason: str
    importable: bool
    kernel_tested: bool
    error: str | None
    available: bool
    wheel_required: bool


class SdpaCheck(BaseModel):
    dtype: str
    shape: list[int]
    passed: bool
    error: str | None = None


class SdpaProbe(BaseModel):
    status: Literal["passed", "failed", "not_tested"]
    reason: str | None = None
    error: str | None = None
    detail: str | None = None
    checked_at: float | None = None
    device: str | None = None
    device_name: str | None = None
    torch: str | None = None
    hip_runtime: str | None = None
    checks: list[SdpaCheck] = Field(default_factory=list)


class EnvironmentSnapshot(BaseModel):
    runtime: EnvironmentRuntime
    packages: list[EnvironmentPackage]
    attention_default: str
    restart_required: bool
    maintenance: bool
    running_jobs: bool
    probe_deferred: bool
    sdpa: SdpaProbe | None = None
    # When the packages were last probed; None until the first probe of this service run.
    probed_at: float | None = None


class EnvironmentLatestPackage(BaseModel):
    # The newest version this runtime can install from the package's online source.
    version: str | None
    source: Literal["community", "pypi", "pytorch", "dtk", "pinned"]
    # The package index that answered, for sources tried in order.
    index: str | None = None
    error: str | None = None


class EnvironmentLatest(BaseModel):
    checked_at: float
    packages: dict[str, EnvironmentLatestPackage]


class EnvironmentWheel(BaseModel):
    wheel_id: str
    package: str
    version: str
    filename: str
    size: int
    sha256: str
    vendor_wheel_id: str | None = None


class EnvironmentOperation(BaseModel):
    environment_profile: str = "legacy"
    python_executable: str | None = None
    id: str
    package: str
    action: str
    status: Literal["planning", "ready", "installing", "verifying", "completed", "failed", "cancelled"]
    created_at: float
    updated_at: float
    plan: list[dict[str, Any]] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    error: str | None = None
    restart_required: bool = False
    dismissed_at: float | None = None
    vendor_wheel_id: str | None = None
    phase: str = "plan"
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    bytes_per_second: float | None = None
    eta_seconds: float | None = None


def environment_attention_default(context) -> str:
    value = context.db.get_kv("environment.settings", {}).get("attention_default", "auto")
    if value == "metal_flash":
        # This is only the default for new configurations. Keep the user's saved
        # preference and existing job/model settings intact when changing runtimes.
        profile = current_profile()
        if profile.endswith("-cpu") or metal_attention_catalog.incompatibility(runtime_info(), profile):
            return "auto"
        try:
            if importlib.metadata.version("mtlattn") != metal_attention_catalog.VERSION:
                return "auto"
        except importlib.metadata.PackageNotFoundError:
            return "auto"
    return value if value in ATTENTION else "auto"


def maintenance_reason(db: Database) -> str | None:
    """Why the queue may not start work now, worded for a waiting job; None when it may."""
    maintenance = db.get_kv("environment.maintenance", {})
    if maintenance.get("blocked"):
        if maintenance.get("restart_required"):
            return "环境已更新，重启服务后开始"
        if maintenance.get("restarting"):
            return "服务正在重启"
        if maintenance.get("torch_operation"):
            return "等待 PyTorch 环境准备完成"
        return "等待环境检测完成" if maintenance.get("probing") else "等待环境操作完成"
    if db.get_kv("regularization.reservation", {}).get("id"):
        return "等待正则图生成完成"
    return None


def maintenance_blocked(db: Database) -> bool:
    return maintenance_reason(db) is not None


def installed_versions() -> dict[str, str]:
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            versions.setdefault(canonicalize_name(name), distribution.version)
    return versions


def protected(name: str) -> bool:
    name = canonicalize_name(name)
    return (
        name in {"torch", "torchvision", "torchaudio", "triton", "triton-windows", "numpy"}
        or name.startswith(("nvidia-", "cuda-", "pytorch-", "dtk-", "dcu-", "hygon-", "hip-", "rocm-"))
        and name != "nvidia-ml-py"
    )


def gpu_checked(name: str) -> bool:
    """Packages that count as available only once they have run on the GPU: attention kernels and
    the 8-bit optimizers. An import alone does not show that their native code suits the device."""
    return bool(CATALOG[name][1]) or name in ("bitsandbytes", *triton_catalog.PACKAGES)


def environment_identity(versions: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()


def runtime_info() -> dict[str, Any]:
    import torch

    from .hardware import gpu_info

    profile = current_profile()
    accelerators_allowed = not profile.endswith("-cpu")
    gpus = gpu_info(include_unavailable=True)
    cuda = accelerators_allowed and profile != "macos-mps" and any(gpu.get("cuda_available") for gpu in gpus)
    hip = getattr(torch.version, "hip", None)
    mps = bool(
        accelerators_allowed
        and profile in ("legacy", "macos-mps")
        and any(gpu.get("kind") == "mps" for gpu in gpus)
    )
    distributed = getattr(torch, "distributed", None)
    distributed_available = bool(distributed and distributed.is_available())
    nccl_available = bool(distributed_available and distributed.is_nccl_available())
    gloo_available = bool(
        distributed_available and getattr(distributed, "is_gloo_available", lambda: False)()
    )
    device_count = sum(bool(gpu.get("cuda_available")) for gpu in gpus) if cuda else 0
    backend = None
    if not profile.endswith("-cpu") and cuda and device_count >= 2:
        if platform.system() == "Linux" and nccl_available:
            backend = "nccl"
        elif platform.system() == "Windows" and not hip and gloo_available:
            backend = "gloo"
    multi_gpu = backend is not None
    return {
        "environment_profile": profile,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.system(),
        "machine": platform.machine(),
        "macos_version": platform.mac_ver()[0] or None if platform.system() == "Darwin" else None,
        **dtk_catalog.system_info(),
        "driver_version": dtk_catalog.driver_version() if hip else None,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": hip,
        "compute_backend": "hip" if cuda and hip else "cuda" if cuda else "mps" if mps else "cpu",
        "cuda_available": cuda,
        "cuda_device_count": device_count,
        "distributed_available": distributed_available,
        "nccl_available": nccl_available,
        "gloo_available": gloo_available,
        "multi_gpu_backend": backend,
        "multi_gpu_probe_required": backend == "gloo",
        "cuda_applicable": platform.system() != "Darwin" and not hip,
        "nccl_applicable": platform.system() == "Linux" and cuda,
        "multi_gpu_training": multi_gpu,
        "training_device_policy": "exclusive_devices" if multi_gpu else "single_device",
        "mps_available": mps,
        "gpu_capability": next(
            (gpu.get("compute_capability") for gpu in gpus if gpu.get("cuda_available")), None
        ),
        "cxx11_abi": bool(torch.compiled_with_cxx11_abi())
        if hasattr(torch, "compiled_with_cxx11_abi")
        else None,
        "gpus": gpus,
        "virtual_environment": sys.prefix != sys.base_prefix,
    }


# Fixed program, never assembled from request text. Kernel probes use a separate process and
# tiny tensors; imports alone do not establish that a wheel works with the current GPU.
PROBE = r"""
import importlib, json, re, torch
from ypuddin.runtime_profiles import current_profile
names = {"xformers": "xformers.ops", "flash-attn": "flash_attn", "sageattention": "sageattention", "nvidia-ml-py": "pynvml", "tensorboard": "tensorboard", "schedulefree": "schedulefree", "bitsandbytes": "bitsandbytes", "onnxruntime": "onnxruntime", "onnxruntime-gpu": "onnxruntime"}
gpu = ("xformers", "flash-attn", "sageattention", "bitsandbytes")
out = {}
accelerators_allowed = not current_profile().endswith("-cpu")
from ypuddin.runtime_attention import probe_sdpa
try:
    out["sdpa"] = probe_sdpa(torch)
except Exception as exc:
    out["sdpa"] = {"status": "not_tested", "reason": "probe_failed", "error": str(exc)[-1500:]}
for name, module in names.items():
    if name in gpu and (not accelerators_allowed or current_profile() == "macos-mps"):
        out[name] = {"importable": False, "kernel_tested": False, "error": None}
        continue
    imported = False
    try:
        m = importlib.import_module(module)
        imported = True
        tested = False
        if name in gpu and accelerators_allowed:
            if not torch.cuda.is_available():
                try:
                    torch.cuda.init()
                    cause = ""
                except Exception as cuda_exc:
                    cause = "\n" + str(cuda_exc)[-500:]
                check = "未能运行 8-bit 优化器检测。" if name == "bitsandbytes" else "未能运行注意力内核检测。"
                raise RuntimeError("检测进程中 PyTorch 无法使用 CUDA，" + check + cause)
            if name == "bitsandbytes":
                # One AdamW 8-bit step; bitsandbytes keeps 8-bit state from 4096 values up.
                p = torch.nn.Parameter(torch.randn(64, 128, device="cuda"))
                p.grad = torch.randn_like(p)
                before = p.detach().clone()
                optimizer = importlib.import_module("bitsandbytes.optim").AdamW8bit([p], lr=1e-2)
                optimizer.step()
                torch.cuda.synchronize()
                if optimizer.state[p]["state1"].dtype != torch.uint8:
                    raise RuntimeError("优化器没有保持 8-bit 状态。")
                if not bool(torch.isfinite(p).all()) or torch.equal(p.detach(), before):
                    raise RuntimeError("8-bit 优化器的更新步骤没有改变权重。")
                tested = True
                out[name] = {"importable": True, "kernel_tested": tested, "error": None}
                continue
            q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float16, requires_grad=name != "sageattention")
            if name == "xformers": y = m.memory_efficient_attention(q, q, q)
            elif name == "flash-attn": y = m.flash_attn_func(q, q, q)
            else:
                with torch.no_grad(): y = m.sageattn(q.transpose(1,2), q.transpose(1,2), q.transpose(1,2), tensor_layout="HND", is_causal=False)
            if name != "sageattention": y.float().sum().backward()
            torch.cuda.synchronize()
            if not bool(torch.isfinite(y).all()): raise RuntimeError("注意力输出含有 NaN 或 Inf。")
            if name != "sageattention" and (q.grad is None or not bool(torch.isfinite(q.grad).all())):
                raise RuntimeError("注意力梯度缺失，或含有 NaN 或 Inf。")
            tested = True
        out[name] = {"importable": True, "kernel_tested": tested, "error": None}
    except Exception as exc:
        detail = str(exc)
        kernel_unavailable = name == "xformers" and imported and (
            re.search(r"no operator found for\s+[`']?memory_efficient_attention", detail, re.IGNORECASE)
            or "no kernel image is available for execution on the device" in detail.lower()
            or re.search(r"(?:device(?: with)?|compute) capability[^\n]*too new", detail, re.IGNORECASE)
        )
        error = detail[-1500:]
        if kernel_unavailable:
            capability = torch.cuda.get_device_capability()
            hip = getattr(torch.version, 'hip', None)
            device_label = 'HIP ' + str(getattr(torch.cuda.get_device_properties(0), 'gcnArchName', capability)) if hip else f'SM{capability[0]}{capability[1]}'
            runtime_label = 'DTK / HIP' if hip else 'CUDA'
            error = (
                f"xFormers 已安装并可导入，但当前 wheel 没有可用于此 GPU（{device_label}）的注意力计算内核。"
                f"可尝试匹配当前 Python、PyTorch、{runtime_label} 且支持此 GPU 的 FlashAttention 2 wheel，安装后重新检测。"
            )
            try:
                q_sdpa = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16, requires_grad=True)
                y_sdpa = torch.nn.functional.scaled_dot_product_attention(q_sdpa, q_sdpa, q_sdpa)
                y_sdpa.float().sum().backward()
                torch.cuda.synchronize()
                error += "PyTorch SDPA 正反向检测通过，可继续使用 SDPA。"
            except Exception as sdpa_exc:
                error += "PyTorch SDPA 检测也未通过，请检查运行环境。\nSDPA: " + str(sdpa_exc)[-500:]
            error += "\n原始错误：" + detail[-1500:]
        out[name] = {"importable": imported, "kernel_tested": False, "error": error}
        if kernel_unavailable:
            out[name]["kernel_unavailable"] = True
import importlib.metadata as metadata
for name in ("triton", "triton-windows"):
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        continue
    imported = False
    try:
        importlib.import_module("triton")
        imported = True
        tested = False
        if accelerators_allowed and current_profile() != "macos-mps" and torch.cuda.is_available():
            importlib.import_module("ypuddin.server.triton_check").run()
            tested = True
        out[name] = {"importable": True, "kernel_tested": tested, "error": None}
    except Exception as exc:
        out[name] = {"importable": imported, "kernel_tested": False, "error": str(exc)[-1500:]}
from ypuddin.server.metal_attention_catalog import probe_metal_attention
out["mtlattn"] = probe_metal_attention(torch)
print("YPUDDIN_ENV=" + json.dumps(out))
"""


def probe_packages() -> dict[str, Any]:
    try:
        # Some runtime imports (including ORT 1.30) write session files into cwd.
        # Probe in disposable storage, never the user's source/data directory.
        with tempfile.TemporaryDirectory(prefix="ypuddin-env-probe-") as work:
            proc = subprocess.run(
                [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=90, cwd=work
            )
        result = next(
            (
                line[len("YPUDDIN_ENV=") :]
                for line in reversed(proc.stdout.splitlines())
                if line.startswith("YPUDDIN_ENV=")
            ),
            None,
        )
        if result is None:
            raise RuntimeError(
                (proc.stderr or proc.stdout)[-1500:] or f"检测进程异常退出，退出码 {proc.returncode}。"
            )
        return json.loads(result)
    except Exception as exc:
        return {
            name: {"importable": False, "kernel_tested": False, "error": str(exc)}
            for name in CATALOG
            if name != "torch"
        } | {"sdpa": {"status": "not_tested", "reason": "probe_failed", "error": str(exc)}}


class Installer:
    """Fixed argument-list runner; separate bundled pip can target a venv without pip installed."""

    def __init__(self, root: Path, *, context=None):
        self.root = root
        self.context = context

    @property
    def cache_dir(self) -> Path:
        return self.context.package_cache_dir(current_profile()) if self.context else self.root / "cache"

    def command(self, log, cancel) -> list[str]:
        if importlib.util.find_spec("pip"):
            return [
                sys.executable,
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "--cache-dir",
                str(self.cache_dir),
            ]
        tool = self.root / "installer"
        python = tool / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists():
            log("正在准备独立的 pip 工具，训练环境不受影响。")
            self.run([sys.executable, "-m", "venv", str(tool)], log, cancel, timeout=120)
        return [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "--python",
            sys.executable,
            "--cache-dir",
            str(self.cache_dir),
        ]

    def run(self, args, log, cancel, *, timeout=1800):
        from .network import ProxyPolicy

        policy = ProxyPolicy.from_context(self.context) if self.context else ProxyPolicy()
        env = policy.subprocess_env()
        for key in tuple(env):
            if key.startswith("PIP_"):
                env.pop(key)
        env["PIP_CONFIG_FILE"] = os.devnull
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        # The output is read as UTF-8; on Windows a child would otherwise write the system code page.
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        log(policy.redact("$ " + subprocess.list2cmdline(args)))
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        lines = []

        def read():
            assert proc.stdout is not None
            for line in proc.stdout:
                line = policy.redact(line.rstrip("\r\n"))
                if line:
                    lines.append(line)
                    if len(lines) > 30:
                        lines.pop(0)
                    log(line)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        end = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if cancel.wait(0.1):
                    raise InterruptedError("已在修改依赖前取消。")
                if time.monotonic() > end:
                    raise TimeoutError(f"安装程序超过 {timeout} 秒仍未完成。")
            reader.join(timeout=5)
            if proc.returncode:
                raise RuntimeError(
                    f"安装程序异常退出，退出码 {proc.returncode}：\n" + "\n".join(lines[-12:])
                )
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            reader.join(timeout=5)
            if proc.stdout:
                proc.stdout.close()


class EnvironmentManager:
    def __init__(
        self,
        context,
        *,
        installer=None,
        versions=installed_versions,
        runtime=runtime_info,
        probe=probe_packages,
    ):
        self.context = context
        self.profile = current_profile()
        self.root = profile_root(context.data_root, self.profile)
        self.root.mkdir(parents=True, exist_ok=True)
        self.installer = installer or Installer(self.root, context=context)
        self.versions, self.runtime, self.probe = versions, runtime, probe
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="environment")
        self.lock = threading.RLock()
        self._cancel: dict[str, threading.Event] = {}
        self._probe_cache: dict[str, Any] | None = None
        self._probe_time = 0.0
        self._probed_at: float | None = None
        self._probe_versions: dict[str, str] | None = None
        self._latest: tuple[float, Any, dict] | None = None
        self._lora_latest: tuple[float, Any, float, dict] | None = None
        self._latest_lock = threading.Lock()
        self._lora_latest_lock = threading.Lock()
        self._windows_catalog = windows_attention_catalog.Catalog()
        self._closed = False
        # What the task center shows of the operations this service changes; it reads them every
        # second, so from memory rather than from the database.
        self._task_ops: dict[str, dict[str, Any]] = {}
        self._task_lock = threading.Lock()
        self._started_at = time.time()
        self.context.db.set_kv("environment.maintenance", {"blocked": False})
        if (tasks := getattr(context, "background_tasks", None)) is not None:
            tasks.add_source(self.background_tasks)
        # A fresh service has released imported DLLs. An interrupted mutation must still be
        # visible as a failed operation and can be repaired through the same workflow.
        for op in self.list():
            if op.status in BUSY:
                self._update(
                    op.id,
                    status="failed",
                    error="训练器在这项操作进行时停止了。请检查运行环境，并修复受影响的扩展。",
                    restart_required=False,
                )

    def list(self) -> list[EnvironmentOperation]:
        rows = self.context.db.fetchall("SELECT value FROM kv WHERE key LIKE 'environment.operation.%'")
        return sorted(
            (
                EnvironmentOperation.model_validate(value)
                for row in rows
                if (value := json.loads(row["value"])).get("environment_profile", "legacy") == self.profile
            ),
            key=lambda op: op.created_at,
            reverse=True,
        )

    def _remember(self, op: EnvironmentOperation) -> None:
        with self._task_lock:
            self._task_ops[op.id] = op.model_dump(include=_TASK_FIELDS)

    def background_tasks(self) -> list[dict[str, Any]]:
        """Extension installs for the task center, while they run and for a while after they end.
        A plan waiting for review is not running work."""
        cutoff = time.time() - TASK_KEEP_SECONDS

        def listed(op: dict[str, Any]) -> bool:
            if op["status"] in BUSY:
                return True
            failed_now = op["status"] == "failed" and op["updated_at"] >= self._started_at
            return not op["dismissed_at"] and (failed_now or op["updated_at"] >= cutoff)

        with self._task_lock:
            for key in [key for key, op in self._task_ops.items() if not listed(op)]:
                del self._task_ops[key]
            ops = sorted(self._task_ops.values(), key=lambda op: op["created_at"], reverse=True)
        tasks = []
        for op in ops:
            running = op["status"] in BUSY
            if op["status"] == "ready":
                continue
            downloading = op["status"] == "planning" and op["phase"] == "download"
            sized = downloading and bool(op["total_bytes"])
            detail = {
                "planning": "下载适配包" if downloading else "检查兼容性",
                "installing": {"uninstall": "正在卸载", "repair": "正在重装"}.get(op["action"], "正在安装"),
                "verifying": "验证环境",
                "completed": "已完成，重启训练器后生效" if op["restart_required"] else "已完成",
                "failed": "失败",
                "cancelled": "已取消",
            }[op["status"]]
            tasks.append({
                "id": f"environment-{op['id']}", "kind": "environment",
                "subject": PACKAGE_LABELS.get(op["package"], op["package"]),
                "state": "running" if running else op["status"], "detail": detail,
                "done": op["downloaded_bytes"] if sized else None, "total": op["total_bytes"] if sized else None,
                "unit": "bytes" if sized else None,
                "link": "/settings/environment", "cancellable": False, "started_at": op["created_at"],
                "finished_at": None if running else op["updated_at"],
                "error": task_error(op["error"], "安装未完成，原因见运行环境页的记录。") if op["status"] == "failed" else None,
            })
        return tasks

    def get(self, id_: str) -> EnvironmentOperation:
        item = self.context.db.get_kv("environment.operation." + id_)
        if not item or item.get("environment_profile", "legacy") != self.profile:
            raise EnvironmentError(404, "找不到这项环境操作。")
        return EnvironmentOperation.model_validate(item)

    def dismiss(self, id_: str) -> EnvironmentOperation:
        with self.lock:
            if self.get(id_).status not in ("completed", "failed", "cancelled"):
                raise EnvironmentError(409, "只能移除已结束的操作记录。")
            return self._update(id_, dismissed_at=now())

    def _update(self, id_, **fields):
        with self.lock:
            op = self.get(id_).model_dump()
            op.update(fields, updated_at=now())
            self.context.db.set_kv("environment.operation." + id_, op)
            result = EnvironmentOperation.model_validate(op)
            self._remember(result)
        return result

    def _log(self, id_, line):
        with self.lock:
            op = self.get(id_)
            self._update(id_, logs=[*op.logs, line[-4000:]][-300:])

    def _running(self):
        if self.context.db.get_kv("regularization.reservation", {}).get("id"):
            return True
        jobs = bool(
            self.context.db.fetchone(
                "SELECT id FROM jobs WHERE status IN ('running','pausing','cancelling') LIMIT 1"
            )
        ) or any(proc.poll() is None for proc in list(self.context.supervisor._procs.values()))
        if jobs:
            return True
        # Tagging owns a spawned ONNX process rather than a jobs-table worker. Its
        # queued row and maintenance admission share db.lock with pipeline.start.
        if self.context.db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dataset_pipeline_operations'"
        ):
            return bool(
                self.context.db.fetchone(
                    "SELECT id FROM dataset_pipeline_operations WHERE action='tag' AND job_id IS NULL "
                    "AND status IN ('queued','running','cancelling') LIMIT 1"
                )
            )
        return False

    def _idle(self):
        maintenance = self.context.db.get_kv("environment.maintenance", {})
        if maintenance.get("torch_operation") or maintenance.get("restarting"):
            raise EnvironmentError(409, "训练器正在切换或准备运行环境，请稍后再试。")
        # A model-test worker keeping its base model holds the installed libraries open.
        self.context.supervisor.release_models()
        if self._running():
            raise EnvironmentError(
                409,
                "训练、缓存、AI 正则图或数据处理任务正在运行。请先停止，等进程退出后再修改依赖。",
            )
        if self._closed:
            raise EnvironmentError(503, "训练器正在关闭，暂时无法修改运行环境。")

    def status(self, refresh=False):
        with self.lock:
            runtime = self.runtime()
            versions = self.versions()
            if self._probe_versions is not None and versions != self._probe_versions:
                self._probe_cache = None
                self._probed_at = None
            running = self._running()
            # Avoid taking VRAM for probes while a real job owns the GPU.
            if refresh and not running and not any(op.status in MUTATING for op in self.list()):
                with self.context.db.lock:
                    # Share the supervisor's launch boundary: the CUDA probe cannot race
                    # with a queued worker acquiring the same accelerator.
                    running = self._running()
                    previous = self.context.db.get_kv("environment.maintenance", {})
                    if not running:
                        self.context.db.set_kv(
                            "environment.maintenance", {**previous, "blocked": True, "probing": True}
                        )
                if not running:
                    try:
                        self._probe_cache = self.probe()
                        self._probe_time = time.monotonic()
                        self._probed_at = time.time()
                        self._probe_versions = dict(versions)
                    finally:
                        self.context.db.set_kv("environment.maintenance", previous)
            probes = self._probe_cache or {}
            packages = []
            for name, (_, backend, docs) in CATALOG.items():
                reason = "supported"
                triton_installable = False
                if name == "torch":
                    reason = "protected_runtime"
                elif name in triton_catalog.PACKAGES:
                    triton_installable, reason = triton_catalog.package_state(name, runtime, versions, self.profile)
                elif name == "mtlattn":
                    reason = metal_attention_catalog.incompatibility(runtime, self.profile) or "supported"
                elif self.profile == "linux-dtk" and name in ("sageattention", "nvidia-ml-py"):
                    reason = "requires_nvidia"
                elif gpu_checked(name) and not runtime["cuda_available"]:
                    reason = "requires_cuda"
                elif (
                    backend in ("flash_attn", "sage")
                    and not runtime.get("hip_runtime")
                    and runtime.get("gpu_capability")
                    and runtime["gpu_capability"][0] < 8
                ):
                    reason = "requires_ampere"
                elif name == "nvidia-ml-py" and runtime["platform"] == "Darwin":
                    reason = "requires_nvidia"
                elif name == "onnxruntime-gpu" and not runtime["cuda_available"]:
                    # The GPU build targets NVIDIA CUDA; DTK and Apple chips run the CPU build.
                    reason = "requires_cuda"
                probe = probes.get(name, {})
                # A Triton row can be shown, and checked, where this page does not install it.
                supported = triton_installable if name in triton_catalog.PACKAGES else reason == "supported"
                packages.append(
                    {
                        "name": name,
                        "version": versions.get(name),
                        "backend": backend,
                        "docs_url": docs,
                        "supported": supported,
                        "reason": reason,
                        "importable": probe.get("importable", name == "torch"),
                        "kernel_tested": probe.get("kernel_tested", False),
                        "error": probe.get("error") if name in versions else None,
                        "available": (supported or reason in ("bundled_with_torch", "vendor_build"))
                        and reason != "triton_version_mismatch"
                        and name in versions
                        and (name != "mtlattn" or versions[name] == metal_attention_catalog.VERSION)
                        and bool(probe.get("importable"))
                        and (not gpu_checked(name) or bool(probe.get("kernel_tested"))),
                        "wheel_required": bool(
                            (
                                backend in ("flash_attn", "sage")
                                and runtime["platform"] in ("Windows", "Linux")
                            )
                            # PyPI builds carry no kernels for Hygon GPUs.
                            or (
                                self.profile == "linux-dtk"
                                and name in ("flash-attn", "xformers", "bitsandbytes")
                            )
                        ),
                    }
                )
            maintenance = self.context.db.get_kv("environment.maintenance", {})
            return {
                "runtime": runtime,
                "packages": packages,
                "attention_default": environment_attention_default(self.context),
                "restart_required": maintenance.get("restart_required", False),
                "maintenance": maintenance_blocked(self.context.db),
                "running_jobs": running,
                "probe_deferred": running,
                "sdpa": probes.get("sdpa"),
                "probed_at": self._probed_at,
            }

    def save_settings(self, settings: EnvironmentSettings):
        if settings.attention_default not in ("auto", "sdpa"):
            status = self.status(refresh=True)
            if not any(
                p["backend"] == settings.attention_default and p["available"] for p in status["packages"]
            ):
                raise EnvironmentError(
                    422,
                    "所选注意力后端需要先在本机设备上通过正反向内核检测，才能设为默认。",
                )
        self.context.db.set_kv("environment.settings", settings.model_dump())
        return settings

    def validate_wheel(self, path: Path, *, package=None):
        try:
            name, version, _, tags = parse_wheel_filename(path.name)
        except Exception as exc:
            raise ValueError("wheel 文件名无效。") from exc
        name = canonicalize_name(name)
        if name not in CATALOG or name == "torch" or package and name != package:
            raise ValueError("wheel 必须是所选的可选扩展包。")
        if not tags.intersection(set(sys_tags())):
            raise ValueError("wheel 的 Python ABI 或平台与训练器所在的机器不符。")
        with zipfile.ZipFile(path) as archive:
            native_files = [
                i.filename
                for i in archive.infolist()
                if i.filename.endswith((".so", ".pyd", ".dll", ".dylib"))
            ]
            entries = [i for i in archive.infolist() if i.filename.endswith(".dist-info/METADATA")]
            if len(entries) != 1 or entries[0].file_size > 1024**2:
                raise ValueError("wheel 中必须有且只有一个有效的包元数据（METADATA）。")
            metadata = BytesParser().parsebytes(archive.read(entries[0]))
            build_info = None
            if name == "xformers" and "xformers/cpp_lib.json" in archive.namelist():
                entry = archive.getinfo("xformers/cpp_lib.json")
                if entry.file_size > 1024**2:
                    raise ValueError("xFormers 构建信息无效。")
                build_info = json.loads(archive.read(entry)).get("version", {})
            pure_sage = (
                name == "sageattention"
                and all(tag.platform == "any" for tag in tags)
                and not any(i.filename.endswith((".so", ".pyd", ".dll")) for i in archive.infolist())
            )
        if (
            canonicalize_name(metadata.get("Name", "")) != name
            or Version(metadata.get("Version", "0")) != version
        ):
            raise ValueError("wheel 文件名与包元数据不一致。")
        if metadata.get("Requires-Python") and not SpecifierSet(metadata["Requires-Python"]).contains(
            platform.python_version(), prereleases=True
        ):
            raise ValueError("wheel 要求的 Python 版本（Requires-Python）与训练器所在的机器不符。")
        versions = self.versions()
        torch_constraint = False
        torch_requirement = False
        for raw in metadata.get_all("Requires-Dist", []):
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate():
                continue
            dep = canonicalize_name(req.name)
            if protected(dep):
                if req.url:
                    raise ValueError(f"wheel 要求直接替换受保护的运行时组件 {dep}。")
                if dep not in versions or versions[dep] not in req.specifier:
                    raise ValueError(
                        f"wheel 需要 {req}，当前受保护的运行时组件为 {versions.get(dep, '未安装')}。"
                    )
                if dep == "torch" and any(
                    s.operator in ("==", "===") and "*" not in s.version for s in req.specifier
                ):
                    torch_constraint = True
                if dep == "torch":
                    torch_requirement = True
        runtime = self.runtime()
        vendor = None
        if name == "mtlattn":
            reason = metal_attention_catalog.incompatibility(runtime, self.profile)
            if reason:
                raise ValueError(metal_attention_catalog.MESSAGES[reason])
            if str(version) != metal_attention_catalog.VERSION or not torch_requirement:
                raise ValueError("Metal FlashAttention 需要经过审核的 mtlattn 0.4.1 Torch 扩展。")
            metal_attention_catalog.validate_release_file(path.name, self._hash(path), path.stat().st_size)
        elif self.profile == "linux-dtk" and CATALOG[name][1]:
            vendor = dtk_catalog.wheel_for_file(path)
            reason = dtk_catalog.incompatibility(vendor, runtime, versions, self.profile)
            if reason:
                raise ValueError("DTK 厂商 wheel 不适用于当前环境：" + wheel_reason(reason))
            if path.stat().st_size != vendor.size_bytes or self._hash(path) != vendor.sha256:
                raise ValueError("DTK 厂商 wheel 的 SHA256 或大小与官方审核记录不符。")
            if not vendor.binary and native_files:
                raise ValueError("这个审核为纯 Python 的厂商 wheel 意外包含本地二进制代码。")
        elif name == "triton-windows":
            reason = triton_catalog.windows_mismatch(str(version), runtime)
            if reason:
                raise ValueError(triton_catalog.message(reason, runtime, str(version)))
        elif CATALOG[name][1]:
            match_torch = re.search(r"torch(\d+\.\d+(?:\.\d+)?)", path.name)
            match_cuda = re.search(r"cu(\d{2,3})", path.name)
            if (
                match_torch
                and not str(runtime["torch"]).startswith(match_torch[1] + ".")
                and Version(str(runtime["torch"])).base_version != match_torch[1]
            ):
                raise ValueError("wheel 的 PyTorch 构建标记与当前 PyTorch 版本不符。")
            if match_cuda and match_cuda[1] != str(runtime.get("cuda_runtime") or "").replace(".", ""):
                raise ValueError("wheel 的 CUDA 构建标记与 PyTorch 的 CUDA 运行时不符。")
            abi = re.search(r"cxx11abi(true|false)", path.name, re.IGNORECASE)
            if (
                abi
                and runtime.get("cxx11_abi") is not None
                and (abi[1].lower() == "true") != runtime["cxx11_abi"]
            ):
                raise ValueError("wheel 的 C++ ABI 与当前 PyTorch 构建不符。")
            build_verified = False
            if build_info:
                cuda = str(runtime.get("cuda_runtime") or "").split(".")
                expected_cuda = int(cuda[0]) * 100 + int(cuda[1]) if len(cuda) == 2 else None
                if build_info.get("cuda") != expected_cuda:
                    raise ValueError("xFormers 编译时的 CUDA 版本与 PyTorch 的 CUDA 运行时不符。")
                # New xFormers wheels use the stable Torch ABI. Respect their declared
                # Torch range instead of rejecting a newer compatible runtime merely
                # because the wheel was compiled against an older version.
                build_verified = (
                    torch_requirement
                    or Version(str(build_info.get("torch", "0"))).base_version
                    == Version(str(runtime["torch"])).base_version
                )
            if (
                not pure_sage
                and not build_verified
                and not (match_cuda and (torch_constraint or match_torch))
            ):
                raise ValueError(
                    "无法确认这个加速扩展 wheel 与 Torch/CUDA 的兼容性。请使用文件名带 torch 和 cu 构建标记的 wheel，或带有 xFormers 编译信息的 wheel。"
                )
        return {
            "package": name,
            "version": str(version),
            "filename": path.name,
            "size": path.stat().st_size,
            "sha256": self._hash(path),
            "vendor_wheel_id": vendor.id if vendor else None,
        }

    @staticmethod
    def _validate_installed_requirements(path: Path, versions: dict[str, str]):
        with zipfile.ZipFile(path) as archive:
            filename = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(filename))
        for raw in metadata.get_all("Requires-Dist", []):
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate():
                continue
            actual = versions.get(canonicalize_name(req.name))
            if req.url or actual is None or actual not in req.specifier:
                raise ValueError(
                    f"厂商 wheel 需要 {req}，请先在这个 DTK 环境中准备兼容的依赖。"
                )

    def vendor_wheels(self):
        return dtk_catalog.catalog(self.runtime(), self.versions(), self.profile)

    def latest_versions(self, refresh=False):
        """The newest version of each optional package this runtime can install online, read from the
        sources an installation uses: the configured pip source for bitsandbytes and ONNX Runtime, the
        PyTorch source for xFormers, the community builds for FlashAttention 2 and the vendor list on
        DTK. Kept an hour."""
        requested_at = time.monotonic()
        with self._latest_lock:
            return self._latest_versions(refresh, requested_at)

    def _latest_versions(self, refresh, requested_at):
        from . import package_releases
        from .network import ProxyPolicy

        policy = ProxyPolicy.from_context(self.context)
        runtime, versions = self.runtime(), self.versions()
        downloads = self.context.settings().get("downloads", {})
        fallback = downloads.get("fallback", True)
        options = {"opener_factory": policy.opener, "cache_key": policy}
        hip = self.profile == "linux-dtk" or bool(runtime.get("hip_runtime"))
        # GPU telemetry changes constantly; only compatibility inputs identify an online lookup.
        identity = {name: runtime.get(name) for name in (
            "python", "torch", "platform", "machine", "cuda_runtime", "hip_runtime",
            "cuda_available", "gpu_capability", "installed_dtk",
        )}
        identity["gpus"] = [
            {name: gpu.get(name) for name in ("cuda_available", "compute_capability")}
            for gpu in runtime.get("gpus") or []
        ]
        matching = triton_catalog.matching_version(runtime, self.profile)
        identity["triton"] = matching
        if hip:
            dependencies = {"torch"} | {
                Requirement(value).name for wheel in dtk_catalog.WHEELS for value in wheel.requires_packages
            }
            identity["dtk_build"] = dtk_catalog.installed_build(runtime, versions)
            identity["packages"] = {name: versions.get(name) for name in sorted(dependencies)}
        sources = {"pypi": downloads.get("pypi", "auto"), "pytorch": downloads.get("pytorch", "auto"),
                   "fallback": fallback}
        cache_key = (policy, self.profile, json.dumps(identity, sort_keys=True), json.dumps(sources, sort_keys=True))
        if (
            self._latest
            and self._latest[1] == cache_key
            and time.monotonic() - self._latest[0] < 3600
            and (not refresh or self._latest[0] >= requested_at)
        ):
            return self._latest[2]

        pypi = [("index-url", url) for url in pypi_sources(downloads.get("pypi", "auto"), fallback, **options)]

        def newest(values, source, error=None):
            parsed = []
            for value in values:
                try:
                    parsed.append((Version(value), value))
                except InvalidVersion:
                    pass
            return {"version": max(parsed)[1] if parsed else None, "source": source, "error": error}

        def online(name, kind, sources, *, torch=True):
            try:
                release = package_releases.newest_release(
                    name,
                    torch=str(runtime.get("torch", "")) if torch else None,
                    python=str(runtime.get("python", "")),
                    opener=policy.opener(),
                    sources=sources,
                )
                return {"version": release.version, "source": kind, "index": release.index, "error": None}
            except Exception as exc:  # noqa: BLE001 - an unreachable index only leaves the version unknown
                return {"version": None, "source": kind, "error": policy.redact(exc)[-500:]}

        packages = {}
        cuda = not hip and runtime.get("platform") in ("Windows", "Linux") and bool(runtime.get("cuda_runtime"))
        if hip:
            wheels = dtk_catalog.catalog(runtime, versions, self.profile).wheels
            for name in ("xformers", "flash-attn", "bitsandbytes"):
                packages[name] = newest(
                    [w.version for w in wheels if w.package == name and w.compatible], "dtk"
                )
        elif cuda:
            catalog = self.windows_wheels(
                refresh=refresh, requested_at=requested_at, runtime=runtime, policy=policy,
            )
            packages["flash-attn"] = newest(
                [w.version for w in catalog.wheels if w.compatible], "community", catalog.error
            )
            # xFormers installs from the PyTorch source for this CUDA runtime, bitsandbytes from pip's.
            torch_index = torch_sources(
                "cu" + str(runtime["cuda_runtime"]).replace(".", ""), downloads.get("pytorch", "auto"), fallback,
                **options,
            )
            packages["xformers"] = online("xformers", "pytorch", torch_index)
            packages["bitsandbytes"] = online("bitsandbytes", "pypi", pypi)
        # Tagging and masks: the GPU build on NVIDIA; the CPU build everywhere, DTK and Apple chips included.
        for name in ("onnxruntime", *(("onnxruntime-gpu",) if cuda else ())):
            packages[name] = online(name, "pypi", pypi, torch=False)
        packages["mtlattn"] = {"version": metal_attention_catalog.VERSION, "source": "pinned", "error": None}
        if matching:
            packages[matching[0]] = matching[1]
        result = {"checked_at": time.time(), "packages": packages}
        self._latest = (time.monotonic(), cache_key, result)
        return result

    def lora_environment(self, refresh=False):
        """The LyCORIS release the built-in adapters match and upstream's; upstream is kept an hour."""
        requested_at = time.monotonic()
        with self._lora_latest_lock:
            return self._lora_environment(refresh, requested_at)

    def _lora_environment(self, refresh, requested_at):
        from .lora_environment import local_lycoris, upstream_lycoris
        from .network import ProxyPolicy

        local = local_lycoris()
        policy = ProxyPolicy.from_context(self.context)
        key = policy
        cached = self._lora_latest
        if (cached and cached[1] == key and time.monotonic() - cached[0] < 3600
                and (not refresh or cached[0] >= requested_at)):
            return {"checked_at": cached[2], "local": local, "upstream": cached[3]}
        try:
            upstream = upstream_lycoris(policy.opener())
        except Exception as exc:  # noqa: BLE001 - an unreachable upstream only leaves the versions unknown
            upstream = {"error": policy.redact(exc)[-500:]}
        checked = time.time()
        self._lora_latest = (time.monotonic(), key, checked, upstream)
        return {"checked_at": checked, "local": local, "upstream": upstream}

    def windows_wheels(self, refresh=False, *, requested_at=None, runtime=None, policy=None):
        from .network import ProxyPolicy

        return self._windows_catalog.snapshot(
            self.runtime() if runtime is None else runtime, self.profile,
            proxy=ProxyPolicy.from_context(self.context) if policy is None else policy, refresh=refresh,
            requested_at=requested_at,
        )

    def _wheel_provider(self, id_):
        if id_.startswith("mjun0812-"):
            return windows_attention_catalog, self._windows_catalog.find_wheel(id_)
        return dtk_catalog, dtk_catalog.find_wheel(id_)

    @staticmethod
    def _hash(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def register_wheel(self, path: Path):
        result = self.validate_wheel(path)
        wheel_id = new_id("wheel")
        result.update(wheel_id=wheel_id, path=str(path))
        self.context.db.set_kv("environment.wheel." + wheel_id, result)
        return {k: v for k, v in result.items() if k != "path"}

    def start(self, request: EnvironmentRequest):
        # Waits outside the locks for a released model-test worker to exit.
        self.context.supervisor.release_models(wait=10)
        with self.lock, self.context.db.lock:
            self._idle()
            if any(op.status in BUSY for op in self.list()):
                raise EnvironmentError(409, "另一项环境操作正在进行，请等它结束。")
            runtime = self.runtime()
            if request.package == "mtlattn" and request.action != "uninstall":
                reason = metal_attention_catalog.incompatibility(runtime, self.profile)
                if reason:
                    raise EnvironmentError(422, metal_attention_catalog.MESSAGES[reason])
            if request.package == "triton-windows" and request.action != "uninstall":
                pin, reason = triton_catalog.windows_build(runtime)
                installed = self.versions().get("triton-windows")
                if not reason and request.action == "repair" and installed:
                    reason = triton_catalog.windows_mismatch(installed, runtime)
                if reason:
                    raise EnvironmentError(422, triton_catalog.message(reason, runtime, installed))
                if request.action == "install" and not request.wheel_id:
                    if request.version not in (None, pin):
                        raise EnvironmentError(422, f"只安装与当前 PyTorch 配套的 triton-windows {pin}。")
                    request.version = pin
            if request.vendor_wheel_id:
                try:
                    provider, vendor = self._wheel_provider(request.vendor_wheel_id)
                    reason = provider.incompatibility(vendor, runtime, self.versions(), self.profile)
                    if reason:
                        raise ValueError("所选预编译 wheel 不可用：" + wheel_reason(reason))
                    if (
                        vendor.package != request.package
                        or request.version
                        and vendor.version != request.version
                    ):
                        raise ValueError("预编译 wheel 与所选的包或版本不符。")
                except ValueError as exc:
                    raise EnvironmentError(422, str(exc)) from exc
            if (
                request.action != "uninstall"
                and self.profile == "linux-dtk"
                and (
                    request.package in ("nvidia-ml-py", "sageattention")
                    or (gpu_checked(request.package) and not (request.vendor_wheel_id or request.wheel_id))
                )
            ):
                raise EnvironmentError(
                    422,
                    "DTK 环境需要匹配的 SourceFind 厂商 wheel 或经过校验的离线 wheel，不能安装 NVIDIA CUDA 包。",
                )
            if (
                request.action != "uninstall"
                and gpu_checked(request.package)
                and request.package != "mtlattn"
                and not runtime["cuda_available"]
            ):
                raise EnvironmentError(
                    422,
                    "8-bit 优化器需要能使用 CUDA 的 PyTorch。"
                    if request.package == "bitsandbytes"
                    else "这个注意力扩展需要能使用 NVIDIA CUDA 的 PyTorch。",
                )
            if (
                request.action != "uninstall"
                and request.package in ("flash-attn", "sageattention")
                and runtime["platform"] == "Windows"
                and not (request.wheel_id or request.vendor_wheel_id)
            ):
                raise EnvironmentError(
                    422,
                    "在 Windows 上安装需要上传兼容的预编译 wheel，不支持从源码编译。",
                )
            installed = self.versions().get(request.package)
            if request.action in ("repair", "uninstall") and not installed:
                raise EnvironmentError(422, "这个包尚未安装。")
            if request.action == "repair" and request.version and request.version != installed:
                raise EnvironmentError(
                    422, "修复会重装当前版本；要更换版本请选择安装。"
                )
            op = EnvironmentOperation(
                environment_profile=self.profile,
                python_executable=sys.executable,
                id=new_id("env"),
                package=request.package,
                action=request.action,
                status="planning",
                created_at=now(),
                updated_at=now(),
                vendor_wheel_id=request.vendor_wheel_id,
            )
            self.context.db.set_kv("environment.operation." + op.id, op.model_dump())
            self._remember(op)
            self._cancel[op.id] = threading.Event()
            self.pool.submit(self._plan, op.id, request)
            return op

    def _plan(self, id_, request):
        def log(line):
            self._log(id_, line)

        cancel = self._cancel[id_]
        try:
            versions = self.versions()
            self.context.db.set_kv("environment.identity." + id_, environment_identity(versions))
            if request.action == "uninstall":
                if cancel.is_set():
                    raise InterruptedError()
                self._update(
                    id_,
                    status="ready",
                    plan=[
                        {"name": request.package, "from_version": versions[request.package], "version": None}
                    ],
                )
                return
            if request.package in ONNX_RUNTIMES:
                other = next(name for name in ONNX_RUNTIMES if name != request.package)
                if other in versions:
                    # Both distributions write the same onnxruntime package; removing one breaks the other.
                    raise ValueError(f"已安装 {other}，请先卸载它再安装 {request.package}。")
            work = self.root / id_
            work.mkdir(exist_ok=True)
            constraints = work / "protected.txt"
            # Optional installs can add dependencies, but only the explicitly selected
            # package may change version. Keep the running service's other packages fixed.
            constraints.write_text(
                "".join(f"{k}==={v}\n" for k, v in versions.items() if k != request.package), encoding="utf-8"
            )
            version = versions.get(request.package) if request.action == "repair" else request.version
            target = request.package + (f"=={version}" if version else "")
            local_wheel = bool(request.wheel_id or request.vendor_wheel_id)
            vendor = None
            provider = None
            if request.vendor_wheel_id:
                provider, vendor = self._wheel_provider(request.vendor_wheel_id)
                reason = provider.incompatibility(vendor, self.runtime(), versions, self.profile)
                if reason:
                    raise ValueError("所选预编译 wheel 已不再兼容：" + wheel_reason(reason))
                self._update(id_, phase="download", total_bytes=vendor.size_bytes)
                log("下载经过审核的预编译 wheel：" + vendor.filename)

                def progress(downloaded, total, speed, eta):
                    self._update(
                        id_,
                        downloaded_bytes=downloaded,
                        total_bytes=total,
                        bytes_per_second=speed,
                        eta_seconds=eta,
                    )

                from .network import ProxyPolicy

                path = provider.download(
                    vendor,
                    self.installer.cache_dir / "reviewed-wheels" / vendor.sha256
                    if request.package == "flash-attn" and self.profile == "windows-cuda"
                    else work / "vendor",
                    cancel,
                    progress,
                    proxy=ProxyPolicy.from_context(self.context),
                )
                self._update(id_, phase="validate", bytes_per_second=None, eta_seconds=None)
                current = self.validate_wheel(path, package=request.package)
                if version and current["version"] != version:
                    raise ValueError("预编译 wheel 与要修复的版本不符。")
                target = str(path)
            if request.wheel_id:
                wheel = self.context.db.get_kv("environment.wheel." + request.wheel_id)
                if not wheel:
                    raise ValueError("找不到上传的 wheel，请重新上传。")
                path = Path(wheel["path"])
                current = self.validate_wheel(path, package=request.package)
                if current["sha256"] != wheel["sha256"] or version and current["version"] != version:
                    raise ValueError("上传的 wheel 已变化，或与所选版本不符。")
                if current.get("vendor_wheel_id"):
                    provider, vendor = self._wheel_provider(current["vendor_wheel_id"])
                target = str(path)
            cmd = self.installer.command(log, cancel)
            self._update(id_, phase="plan", bytes_per_second=None, eta_seconds=None)
            report = work / "report.json"
            args = [
                *cmd,
                "install",
                "--dry-run",
                "--report",
                str(report),
                "--only-binary=:all:",
                "--no-input",
                "--constraint",
                str(constraints),
            ]
            # The official CUDA-specific xFormers wheel index binds its compiled kernels to
            # the actual Torch runtime, not the driver's advertised maximum CUDA version.
            sources = self.context.settings().get("downloads", {})
            options = probe_options(self.context)
            indexes = pypi_sources(sources.get("pypi", "auto"), sources.get("fallback", True), **options)
            plan_sources = [("index-url", url) for url in indexes]
            if self.profile == "linux-dtk" and local_wheel:
                # Native/declared dependencies must exist. Only the reviewed implicit
                # Python dependencies are resolved separately below, never vendor kernels.
                self._validate_installed_requirements(Path(target), versions)
                args += ["--no-index", "--no-deps"]
                plan_sources = [("local", "local wheel")]
            elif request.package == "xformers" and not local_wheel:
                cuda = self.runtime().get("cuda_runtime")
                plan_sources = torch_sources(
                    "cu" + str(cuda).replace(".", ""),
                    sources.get("pytorch", "auto"),
                    sources.get("fallback", True),
                    **options,
                )
            if request.action == "repair":
                # Only the requested package is force-reinstalled. Dependencies are checked
                # separately against the installed environment and never reinstalled en masse.
                args += ["--force-reinstall", "--no-deps"]
            run_sources(
                self.installer,
                [
                    (
                        url,
                        [
                            *args,
                            *(
                                []
                                if kind == "local"
                                else ["--no-index", "--no-deps", "--find-links", url]
                                if kind == "find-links"
                                else ["--index-url", url]
                            ),
                            target,
                        ],
                    )
                    for kind, url in plan_sources
                ],
                log,
                cancel,
            )
            rows = json.loads(report.read_text(encoding="utf-8")).get("install", [])
            supplemental_names = set()
            supplemental_requirements = []
            if vendor and provider is dtk_catalog:
                for value in dtk_catalog.python_runtime_requirements(vendor):
                    requirement = Requirement(value)
                    actual = versions.get(canonicalize_name(requirement.name))
                    if actual is None or actual not in requirement.specifier:
                        supplemental_requirements.append(value)
            if supplemental_requirements:
                log("解析厂商 wheel 需要的 Python 依赖：" + ", ".join(supplemental_requirements))
                supplemental_report = work / "python-runtime-report.json"
                run_sources(
                    self.installer,
                    [
                        (
                            url,
                            [
                                *cmd,
                                "install",
                                "--dry-run",
                                "--report",
                                str(supplemental_report),
                                "--only-binary=:all:",
                                "--no-input",
                                "--constraint",
                                str(constraints),
                                "--index-url",
                                url,
                                *supplemental_requirements,
                            ],
                        )
                        for url in indexes
                    ],
                    log,
                    cancel,
                )
                supplemental_rows = json.loads(supplemental_report.read_text(encoding="utf-8")).get(
                    "install", []
                )
                supplemental_names = {canonicalize_name(row["metadata"]["name"]) for row in supplemental_rows}
                if any(protected(name) or name in CATALOG for name in supplemental_names):
                    raise ValueError(
                        "厂商 wheel 的 Python 依赖不能安装或替换本地二进制包和受管理的扩展。"
                    )
                rows.extend(supplemental_rows)
            planned_versions = {
                **versions,
                **{canonicalize_name(row["metadata"]["name"]): row["metadata"]["version"] for row in rows},
            }
            for value in supplemental_requirements:
                requirement = Requirement(value)
                actual = planned_versions.get(canonicalize_name(requirement.name))
                if actual is None or actual not in requirement.specifier:
                    raise ValueError("安装计划缺少厂商 wheel 需要的 Python 依赖：" + value)
            plan = []
            for row in rows:
                meta = row["metadata"]
                name, selected = canonicalize_name(meta["name"]), meta["version"]
                if protected(name) and not (name == request.package == "triton-windows"):
                    raise ValueError(
                        f"安装计划会改动受保护的运行时组件 {name}（{versions.get(name, '未安装')} → {selected}）。请选择兼容的扩展版本或 wheel。"
                    )
                info = row["download_info"]
                url = info["url"]
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme == "file":
                    if (
                        parsed.netloc
                        or not local_wheel
                        or Path(url2pathname(parsed.path)).resolve() != Path(target).resolve()
                    ):
                        raise ValueError("安装计划中出现了意外的本地依赖。")
                elif parsed.scheme != "https" or parsed.hostname not in (
                    "files.pythonhosted.org",
                    "download.pytorch.org",
                    "download-r2.pytorch.org",
                    "mirrors.ustc.edu.cn",
                    "pypi.tuna.tsinghua.edu.cn",
                    "mirrors.aliyun.com",
                    "mirror.sjtu.edu.cn",
                ):
                    raise ValueError("安装计划使用了不受支持的 wheel 来源。")
                filename = urllib.parse.unquote(parsed.path).rsplit("/", 1)[-1]
                if not filename.endswith(".whl"):
                    raise ValueError("不支持从源码构建，需要兼容的预编译 wheel。")
                if name in supplemental_names:
                    _, _, _, tags = parse_wheel_filename(filename)
                    if any(tag.abi != "none" or tag.platform != "any" for tag in tags):
                        raise ValueError("厂商 wheel 的补充依赖必须是纯 Python wheel。")
                if name != request.package and name in versions and selected != versions[name]:
                    raise ValueError(
                        f"安装计划会改动已有依赖 {name}；只有所选的包可以更换版本。"
                    )
                sha = info.get("archive_info", {}).get("hashes", {}).get("sha256")
                if not sha or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
                    raise ValueError("安装计划缺少 wheel 的 SHA256 校验值。")
                if name == "mtlattn":
                    if selected != metal_attention_catalog.VERSION:
                        raise ValueError("Metal FlashAttention 只支持 mtlattn 0.4.1。")
                    metal_attention_catalog.validate_release_file(filename, sha)
                # Same-version repair must still satisfy existing dependency requirements.
                if request.action == "repair" or name in supplemental_names:
                    for raw in meta.get("requires_dist", []):
                        req = Requirement(raw)
                        if req.marker and not req.marker.evaluate():
                            continue
                        if (
                            canonicalize_name(req.name) not in planned_versions
                            or planned_versions[canonicalize_name(req.name)] not in req.specifier
                        ):
                            raise ValueError(f"安装计划不满足依赖要求 {req}。")
                plan.append(
                    {
                        "name": name,
                        "from_version": versions.get(name),
                        "version": selected,
                        "url": url,
                        "sha256": sha,
                        "filename": filename,
                        **(
                            {
                                "source_url": vendor.url,
                                "provider": "sourcefind-dtk"
                                if provider is dtk_catalog
                                else windows_attention_catalog.PROVIDER,
                            }
                            if request.vendor_wheel_id and name == request.package
                            else {}
                        ),
                    }
                )
            if not plan:
                log("所选版本已经安装；如需重装请使用修复。")
                self._update(id_, status="completed")
                return
            if cancel.is_set():
                raise InterruptedError()
            self._update(id_, status="ready", plan=plan)
            log(
                "安装计划已就绪。现有的 Torch/CUDA/NumPy 运行时不会改动；请逐项检查包变更后再应用。"
            )
        except InterruptedError:
            self._update(id_, status="cancelled", bytes_per_second=None, eta_seconds=None)
        except Exception as exc:
            self._update(id_, status="failed", error=str(exc), bytes_per_second=None, eta_seconds=None)
            log(str(exc))

    def apply(self, id_):
        self.context.supervisor.release_models(wait=10)
        with self.lock, self.context.db.lock:
            self._idle()
            op = self.get(id_)
            if op.status != "ready":
                raise EnvironmentError(409, "只能应用已就绪的安装计划。")
            if op.python_executable and op.python_executable != sys.executable:
                raise EnvironmentError(409, "这个安装计划属于另一个 Python 环境，请重新生成计划。")
            if any(other.status in BUSY for other in self.list()):
                raise EnvironmentError(409, "另一项环境操作正在进行，请等它结束。")
            if environment_identity(self.versions()) != self.context.db.get_kv("environment.identity." + id_):
                raise EnvironmentError(
                    409, "生成安装计划后运行环境发生了变化，请重新生成计划。"
                )
            previous = self.context.db.get_kv("environment.maintenance", {})
            self.context.db.set_kv("environment.prior_maintenance." + id_, previous)
            self.context.db.set_kv(
                "environment.maintenance",
                {
                    "blocked": True,
                    "operation_id": id_,
                    "restart_required": previous.get("restart_required", False),
                },
            )
            self._update(id_, status="installing")
            self._update(id_, phase="install", bytes_per_second=None, eta_seconds=None)
            self.pool.submit(self._apply, id_)
            return self.get(id_)

    def _apply(self, id_):
        def log(line):
            self._log(id_, line)

        op = self.get(id_)
        before = self.versions()
        mutation_started = False
        # Mutation is not cancellable: interrupting pip can leave a partially uninstalled
        # extension. Downloads and resolution can be cancelled before Apply instead.
        no_cancel = threading.Event()
        try:
            cmd = self.installer.command(log, no_cancel)
            if op.action == "uninstall":
                mutation_started = True
                self.installer.run([*cmd, "uninstall", "--yes", op.package], log, no_cancel)
            else:
                work = self.root / id_ / "wheels"
                work.mkdir(parents=True, exist_ok=True)
                requirements = self.root / id_ / "resolved.txt"
                requirements.write_text(
                    "".join(f"{p['url']} --hash=sha256:{p['sha256']}\n" for p in op.plan), encoding="utf-8"
                )
                self.installer.run(
                    [
                        *cmd,
                        "download",
                        *(["--no-index"] if self.profile == "linux-dtk" else []),
                        "--no-deps",
                        "--only-binary=:all:",
                        "--require-hashes",
                        "--dest",
                        str(work),
                        "--requirement",
                        str(requirements),
                    ],
                    log,
                    no_cancel,
                )
                wheels = []
                for item in op.plan:
                    wheel = work / item["filename"]
                    if not wheel.is_file() or self._hash(wheel) != item["sha256"]:
                        raise ValueError(
                            f"下载的 wheel 校验值与安装计划不符：{item['name']}"
                        )
                    if item["name"] == op.package and CATALOG[op.package][1]:
                        self.validate_wheel(wheel, package=op.package)
                    wheels.append(str(wheel))
                mutation_started = True
                self.installer.run(
                    [*cmd, "install", "--no-deps", "--no-index", "--force-reinstall", *wheels], log, no_cancel
                )
            self._update(id_, status="verifying", phase="verify")
            after = self.versions()
            if {k: v for k, v in before.items() if protected(k) and k != op.package} != {
                k: v for k, v in after.items() if protected(k) and k != op.package
            }:
                raise RuntimeError(
                    "受保护的运行时组件意外发生了变化。请先恢复原来的运行环境再训练。"
                )
            if op.action == "uninstall":
                if op.package in after:
                    raise RuntimeError("卸载后这个包仍然存在。")
                if CATALOG[op.package][1] == self.context.db.get_kv("environment.settings", {}).get(
                    "attention_default", "auto"
                ):
                    self.context.db.set_kv("environment.settings", {"attention_default": "auto"})
                    log(
                        "已卸载当前默认的注意力后端，新配置改用 PyTorch SDPA（自动）。"
                    )
            else:
                for item in op.plan:
                    if after.get(item["name"]) != item["version"]:
                        raise RuntimeError(f"安装后的版本与安装计划不符：{item['name']}")
                probes = self.probe()
                self._probe_cache, self._probe_time = probes, time.monotonic()
                self._probed_at = time.time()
                self._probe_versions = dict(self.versions())
                probe = probes.get(op.package, {})
                if not probe.get("importable") or gpu_checked(op.package) and not probe.get("kernel_tested"):
                    if probe.get("kernel_unavailable"):
                        raise RuntimeError(probe["error"])
                    raise RuntimeError(
                        "包已安装，但运行检测未通过："
                        + str(probe.get("error") or "所选设备没有可用的内核。")
                    )
            if op.package in ONNX_RUNTIMES:
                # Only the tagging and head-mask child processes import ONNX Runtime; each job
                # starts a fresh one, so the running service and its queue need no restart.
                self._update(id_, status="completed", restart_required=False)
                log("依赖已更新，下次打标或自动遮罩时生效。")
            else:
                self._update(id_, status="completed", restart_required=True)
                log("依赖已更新。重启 Studio 后生效并恢复任务队列。")
        except Exception as exc:
            self._update(id_, status="failed", error=str(exc), restart_required=mutation_started)
            log(str(exc))
        finally:
            self._probe_time = 0
            previous = self.context.db.get_kv("environment.prior_maintenance." + id_, {})
            restart = (mutation_started and op.package not in ONNX_RUNTIMES) or previous.get(
                "restart_required", False
            )
            self.context.db.set_kv(
                "environment.maintenance",
                {"blocked": restart, "operation_id": id_, "restart_required": restart},
            )

    def cancel(self, id_):
        with self.lock:
            op = self.get(id_)
            if op.status in MUTATING:
                raise EnvironmentError(
                    409,
                    "正在应用依赖变更，中途停止不安全。请等待验证完成，然后重启训练器。",
                )
            if op.status == "planning":
                self._cancel.setdefault(id_, threading.Event()).set()
            elif op.status == "ready":
                self._update(id_, status="cancelled")
            return self.get(id_)

    def close(self):
        self._closed = True
        for op in self.list():
            if op.status == "planning":
                self._cancel.setdefault(op.id, threading.Event()).set()
        self.pool.shutdown(wait=True, cancel_futures=False)
