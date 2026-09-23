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

from . import dtk_catalog, metal_attention_catalog, windows_attention_catalog
from .db import Database, new_id, now
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
}
ATTENTION = ("auto", "sdpa", "xformers", "flash_attn", "sage", "metal_flash")
MUTATING = ("installing", "verifying")
BUSY = ("planning", *MUTATING)
MAX_WHEEL_BYTES = 2 * 1024**3


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
    ]
    action: Literal["install", "repair", "uninstall"] = "install"
    version: str | None = None
    wheel_id: str | None = None
    vendor_wheel_id: str | None = None

    @model_validator(mode="after")
    def one_wheel_source(self):
        if self.wheel_id and self.vendor_wheel_id:
            raise ValueError("Choose either an uploaded wheel or a vendor wheel")
        if self.action == "uninstall" and (self.wheel_id or self.vendor_wheel_id):
            raise ValueError("Uninstall does not accept a wheel source")
        if self.package == "mtlattn" and self.action != "uninstall":
            if self.version not in (None, metal_attention_catalog.VERSION):
                raise ValueError("Metal FlashAttention currently supports only mtlattn 0.4.1")
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
            raise ValueError("version must be an exact package version, for example 0.0.32.post2") from exc


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


def maintenance_blocked(db: Database) -> bool:
    return bool(
        db.get_kv("environment.maintenance", {}).get("blocked")
        or db.get_kv("regularization.reservation", {}).get("id")
    )


def installed_versions() -> dict[str, str]:
    return {
        canonicalize_name(d.metadata["Name"]): d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }


def protected(name: str) -> bool:
    name = canonicalize_name(name)
    return (
        name in {"torch", "torchvision", "torchaudio", "triton", "triton-windows", "numpy"}
        or name.startswith(("nvidia-", "cuda-", "pytorch-", "dtk-", "dcu-", "hygon-", "hip-", "rocm-"))
        and name != "nvidia-ml-py"
    )


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
names = {"xformers": "xformers.ops", "flash-attn": "flash_attn", "sageattention": "sageattention", "nvidia-ml-py": "pynvml", "tensorboard": "tensorboard", "schedulefree": "schedulefree"}
out = {}
accelerators_allowed = not current_profile().endswith("-cpu")
from ypuddin.runtime_attention import probe_sdpa
try:
    out["sdpa"] = probe_sdpa(torch)
except Exception as exc:
    out["sdpa"] = {"status": "not_tested", "reason": "probe_failed", "error": str(exc)[-1500:]}
for name, module in names.items():
    if name in ("xformers", "flash-attn", "sageattention") and (not accelerators_allowed or current_profile() == "macos-mps"):
        out[name] = {"importable": False, "kernel_tested": False, "error": None}
        continue
    imported = False
    try:
        m = importlib.import_module(module)
        imported = True
        tested = False
        if name in ("xformers", "flash-attn", "sageattention") and accelerators_allowed and torch.cuda.is_available():
            q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float16, requires_grad=name != "sageattention")
            if name == "xformers": y = m.memory_efficient_attention(q, q, q)
            elif name == "flash-attn": y = m.flash_attn_func(q, q, q)
            else:
                with torch.no_grad(): y = m.sageattn(q.transpose(1,2), q.transpose(1,2), q.transpose(1,2), tensor_layout="HND", is_causal=False)
            if name != "sageattention": y.float().sum().backward()
            torch.cuda.synchronize()
            if not bool(torch.isfinite(y).all()): raise RuntimeError("Attention output is not finite")
            if name != "sageattention" and (q.grad is None or not bool(torch.isfinite(q.grad).all())):
                raise RuntimeError("Attention gradient is missing or not finite")
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
                "需要兼容的 attention 内核。"
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
                (proc.stderr or proc.stdout)[-1500:] or f"probe exited with code {proc.returncode}"
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
            log("Preparing an isolated pip helper; the training environment is unchanged.")
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
                    raise InterruptedError("Operation cancelled before dependency mutation")
                if time.monotonic() > end:
                    raise TimeoutError(f"Installer exceeded {timeout} seconds")
            reader.join(timeout=5)
            if proc.returncode:
                raise RuntimeError(
                    f"Installer exited with code {proc.returncode}:\n" + "\n".join(lines[-12:])
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
        self._windows_catalog = windows_attention_catalog.Catalog()
        self._closed = False
        self.context.db.set_kv("environment.maintenance", {"blocked": False})
        # A fresh service has released imported DLLs. An interrupted mutation must still be
        # visible as a failed operation and can be repaired through the same workflow.
        for op in self.list():
            if op.status in BUSY:
                self._update(
                    op.id,
                    status="failed",
                    error="Service stopped during this operation; inspect the environment and repair the affected package.",
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

    def get(self, id_: str) -> EnvironmentOperation:
        item = self.context.db.get_kv("environment.operation." + id_)
        if not item or item.get("environment_profile", "legacy") != self.profile:
            raise EnvironmentError(404, "Environment operation not found")
        return EnvironmentOperation.model_validate(item)

    def dismiss(self, id_: str) -> EnvironmentOperation:
        with self.lock:
            if self.get(id_).status not in ("completed", "failed", "cancelled"):
                raise EnvironmentError(409, "Only a finished operation notification can be dismissed")
            return self._update(id_, dismissed_at=now())

    def _update(self, id_, **fields):
        with self.lock:
            op = self.get(id_).model_dump()
            op.update(fields, updated_at=now())
            self.context.db.set_kv("environment.operation." + id_, op)
        return EnvironmentOperation.model_validate(op)

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
            raise EnvironmentError(409, "The service is switching or preparing its runtime")
        if self._running():
            raise EnvironmentError(
                409,
                "A training, cache, AI regularization or data worker is running. Stop it and wait for the process to exit before modifying dependencies.",
            )
        if self._closed:
            raise EnvironmentError(503, "Environment manager is stopping")

    def status(self, refresh=False):
        with self.lock:
            runtime = self.runtime()
            versions = self.versions()
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
                    finally:
                        self.context.db.set_kv("environment.maintenance", previous)
            probes = self._probe_cache or {}
            packages = []
            for name, (_, backend, docs) in CATALOG.items():
                reason = "supported"
                if name == "torch":
                    reason = "protected_runtime"
                elif name == "mtlattn":
                    reason = metal_attention_catalog.incompatibility(runtime, self.profile) or "supported"
                elif self.profile == "linux-dtk" and name in ("sageattention", "nvidia-ml-py"):
                    reason = "requires_nvidia"
                elif backend and not runtime["cuda_available"]:
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
                probe = probes.get(name, {})
                supported = reason == "supported"
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
                        "available": supported
                        and name in versions
                        and (name != "mtlattn" or versions[name] == metal_attention_catalog.VERSION)
                        and bool(probe.get("importable"))
                        and (not backend or bool(probe.get("kernel_tested"))),
                        "wheel_required": bool(
                            (
                                backend in ("flash_attn", "sage")
                                and runtime["platform"] in ("Windows", "Linux")
                            )
                            or (self.profile == "linux-dtk" and name in ("flash-attn", "xformers"))
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
            }

    def save_settings(self, settings: EnvironmentSettings):
        if settings.attention_default not in ("auto", "sdpa"):
            status = self.status(refresh=True)
            if not any(
                p["backend"] == settings.attention_default and p["available"] for p in status["packages"]
            ):
                raise EnvironmentError(
                    422,
                    "The selected attention backend must pass its device's forward/backward kernel probe before becoming the default.",
                )
        self.context.db.set_kv("environment.settings", settings.model_dump())
        return settings

    def validate_wheel(self, path: Path, *, package=None):
        try:
            name, version, _, tags = parse_wheel_filename(path.name)
        except Exception as exc:
            raise ValueError("Invalid wheel filename") from exc
        name = canonicalize_name(name)
        if name not in CATALOG or name == "torch" or package and name != package:
            raise ValueError("Wheel must contain the selected managed optional package")
        if not tags.intersection(set(sys_tags())):
            raise ValueError("Wheel Python ABI or platform does not match this server")
        with zipfile.ZipFile(path) as archive:
            native_files = [
                i.filename
                for i in archive.infolist()
                if i.filename.endswith((".so", ".pyd", ".dll", ".dylib"))
            ]
            entries = [i for i in archive.infolist() if i.filename.endswith(".dist-info/METADATA")]
            if len(entries) != 1 or entries[0].file_size > 1024**2:
                raise ValueError("Wheel must have exactly one valid package METADATA")
            metadata = BytesParser().parsebytes(archive.read(entries[0]))
            build_info = None
            if name == "xformers" and "xformers/cpp_lib.json" in archive.namelist():
                entry = archive.getinfo("xformers/cpp_lib.json")
                if entry.file_size > 1024**2:
                    raise ValueError("Invalid xFormers build metadata")
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
            raise ValueError("Wheel filename and package metadata disagree")
        if metadata.get("Requires-Python") and not SpecifierSet(metadata["Requires-Python"]).contains(
            platform.python_version(), prereleases=True
        ):
            raise ValueError("Wheel Requires-Python does not match this server")
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
                    raise ValueError(f"Wheel requests a direct replacement of protected runtime {dep}")
                if dep not in versions or versions[dep] not in req.specifier:
                    raise ValueError(
                        f"Wheel requires {req}; current protected runtime has {versions.get(dep, 'not installed')}"
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
                raise ValueError("Metal FlashAttention requires the reviewed mtlattn 0.4.1 Torch extension")
            metal_attention_catalog.validate_release_file(path.name, self._hash(path), path.stat().st_size)
        elif self.profile == "linux-dtk" and CATALOG[name][1]:
            vendor = dtk_catalog.wheel_for_file(path)
            reason = dtk_catalog.incompatibility(vendor, runtime, versions, self.profile)
            if reason:
                raise ValueError("DTK vendor wheel does not match this environment: " + reason)
            if path.stat().st_size != vendor.size_bytes or self._hash(path) != vendor.sha256:
                raise ValueError("DTK vendor wheel differs from the reviewed official SHA256 or size")
            if not vendor.binary and native_files:
                raise ValueError("The reviewed Python-only vendor wheel unexpectedly contains native code")
        elif CATALOG[name][1]:
            match_torch = re.search(r"torch(\d+\.\d+(?:\.\d+)?)", path.name)
            match_cuda = re.search(r"cu(\d{2,3})", path.name)
            if (
                match_torch
                and not str(runtime["torch"]).startswith(match_torch[1] + ".")
                and Version(str(runtime["torch"])).base_version != match_torch[1]
            ):
                raise ValueError("Wheel PyTorch build tag does not match the current PyTorch version")
            if match_cuda and match_cuda[1] != str(runtime.get("cuda_runtime") or "").replace(".", ""):
                raise ValueError("Wheel CUDA build tag does not match the PyTorch CUDA runtime")
            abi = re.search(r"cxx11abi(true|false)", path.name, re.IGNORECASE)
            if (
                abi
                and runtime.get("cxx11_abi") is not None
                and (abi[1].lower() == "true") != runtime["cxx11_abi"]
            ):
                raise ValueError("Wheel C++ ABI does not match the current PyTorch build")
            build_verified = False
            if build_info:
                cuda = str(runtime.get("cuda_runtime") or "").split(".")
                expected_cuda = int(cuda[0]) * 100 + int(cuda[1]) if len(cuda) == 2 else None
                if build_info.get("cuda") != expected_cuda:
                    raise ValueError("xFormers compiled CUDA version does not match the PyTorch CUDA runtime")
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
                    "Cannot verify this accelerator wheel's Torch/CUDA compatibility; use explicit torch+cu build tags or xFormers compiled build metadata"
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
                    f"Vendor wheel requires {req}; prepare the compatible dependency in this DTK environment first"
                )

    def vendor_wheels(self):
        return dtk_catalog.catalog(self.runtime(), self.versions(), self.profile)

    def windows_wheels(self, refresh=False):
        from .network import ProxyPolicy

        return self._windows_catalog.snapshot(
            self.runtime(), self.profile, proxy=ProxyPolicy.from_context(self.context), refresh=refresh
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
        with self.lock, self.context.db.lock:
            self._idle()
            if any(op.status in BUSY for op in self.list()):
                raise EnvironmentError(409, "Another environment operation is in progress")
            runtime = self.runtime()
            if request.package == "mtlattn" and request.action != "uninstall":
                reason = metal_attention_catalog.incompatibility(runtime, self.profile)
                if reason:
                    raise EnvironmentError(422, metal_attention_catalog.MESSAGES[reason])
            if request.vendor_wheel_id:
                try:
                    provider, vendor = self._wheel_provider(request.vendor_wheel_id)
                    reason = provider.incompatibility(vendor, runtime, self.versions(), self.profile)
                    if reason:
                        raise ValueError("Selected prebuilt wheel is unavailable: " + reason)
                    if (
                        vendor.package != request.package
                        or request.version
                        and vendor.version != request.version
                    ):
                        raise ValueError("Vendor wheel does not match the selected package or version")
                except ValueError as exc:
                    raise EnvironmentError(422, str(exc)) from exc
            if (
                request.action != "uninstall"
                and self.profile == "linux-dtk"
                and (
                    request.package in ("nvidia-ml-py", "sageattention")
                    or (CATALOG[request.package][1] and not (request.vendor_wheel_id or request.wheel_id))
                )
            ):
                raise EnvironmentError(
                    422,
                    "DTK requires a matching SourceFind vendor wheel or verified offline wheel; NVIDIA CUDA packages cannot be installed",
                )
            if (
                request.action != "uninstall"
                and CATALOG[request.package][1]
                and request.package != "mtlattn"
                and not runtime["cuda_available"]
            ):
                raise EnvironmentError(
                    422, "This attention extension requires a working NVIDIA CUDA PyTorch runtime"
                )
            if (
                request.action != "uninstall"
                and request.package in ("flash-attn", "sageattention")
                and runtime["platform"] == "Windows"
                and not (request.wheel_id or request.vendor_wheel_id)
            ):
                raise EnvironmentError(
                    422,
                    "Windows installation requires an uploaded compatible prebuilt wheel. Source compilation is disabled.",
                )
            installed = self.versions().get(request.package)
            if request.action in ("repair", "uninstall") and not installed:
                raise EnvironmentError(422, "Package is not installed")
            if request.action == "repair" and request.version and request.version != installed:
                raise EnvironmentError(
                    422, "Repair reinstalls the current version; choose Install to change version"
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
                    raise ValueError("Selected prebuilt wheel is no longer compatible: " + reason)
                self._update(id_, phase="download", total_bytes=vendor.size_bytes)
                log("Download reviewed prebuilt wheel: " + vendor.filename)

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
                    raise ValueError("Vendor wheel does not match the requested repair version")
                target = str(path)
            if request.wheel_id:
                wheel = self.context.db.get_kv("environment.wheel." + request.wheel_id)
                if not wheel:
                    raise ValueError("Uploaded wheel not found")
                path = Path(wheel["path"])
                current = self.validate_wheel(path, package=request.package)
                if current["sha256"] != wheel["sha256"] or version and current["version"] != version:
                    raise ValueError("Uploaded wheel changed or does not match the selected version")
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
            indexes = pypi_sources(sources.get("pypi", "ustc"), sources.get("fallback", True))
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
                    sources.get("pytorch", "mirror"),
                    sources.get("fallback", True),
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
                log("Resolve vendor Python runtime dependencies: " + ", ".join(supplemental_requirements))
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
                        "Vendor Python dependencies cannot install or replace native/managed packages"
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
                    raise ValueError("Vendor Python runtime dependency is missing from the plan: " + value)
            plan = []
            for row in rows:
                meta = row["metadata"]
                name, selected = canonicalize_name(meta["name"]), meta["version"]
                if protected(name):
                    raise ValueError(
                        f"Plan would modify protected runtime {name} ({versions.get(name, 'absent')} -> {selected}). Choose a compatible extension version or wheel."
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
                        raise ValueError("Unexpected local dependency in installation plan")
                elif parsed.scheme != "https" or parsed.hostname not in (
                    "files.pythonhosted.org",
                    "download.pytorch.org",
                    "download-r2.pytorch.org",
                    "mirrors.ustc.edu.cn",
                    "pypi.tuna.tsinghua.edu.cn",
                    "mirrors.aliyun.com",
                    "mirror.sjtu.edu.cn",
                ):
                    raise ValueError("Installation plan uses an unsupported wheel source")
                filename = urllib.parse.unquote(parsed.path).rsplit("/", 1)[-1]
                if not filename.endswith(".whl"):
                    raise ValueError("Source builds are disabled; a compatible prebuilt wheel is required")
                if name in supplemental_names:
                    _, _, _, tags = parse_wheel_filename(filename)
                    if any(tag.abi != "none" or tag.platform != "any" for tag in tags):
                        raise ValueError("Vendor supplemental dependencies must be Python-only wheels")
                if name != request.package and name in versions and selected != versions[name]:
                    raise ValueError(
                        f"Plan would change existing dependency {name}; only the selected package may change version"
                    )
                sha = info.get("archive_info", {}).get("hashes", {}).get("sha256")
                if not sha or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
                    raise ValueError("Wheel plan is missing its SHA256 digest")
                if name == "mtlattn":
                    if selected != metal_attention_catalog.VERSION:
                        raise ValueError("Metal FlashAttention only accepts mtlattn 0.4.1")
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
                            raise ValueError(f"Installation plan does not satisfy {req}")
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
                log("Requested version is already installed; use Repair to reinstall it.")
                self._update(id_, status="completed")
                return
            if cancel.is_set():
                raise InterruptedError()
            self._update(id_, status="ready", plan=plan)
            log(
                "Plan ready. Existing Torch/CUDA/NumPy runtime is protected. Review every package change before applying."
            )
        except InterruptedError:
            self._update(id_, status="cancelled", bytes_per_second=None, eta_seconds=None)
        except Exception as exc:
            self._update(id_, status="failed", error=str(exc), bytes_per_second=None, eta_seconds=None)
            log(str(exc))

    def apply(self, id_):
        with self.lock, self.context.db.lock:
            self._idle()
            op = self.get(id_)
            if op.status != "ready":
                raise EnvironmentError(409, "Only a reviewed, ready plan can be applied")
            if op.python_executable and op.python_executable != sys.executable:
                raise EnvironmentError(409, "This plan belongs to another interpreter; create a new plan")
            if any(other.status in BUSY for other in self.list()):
                raise EnvironmentError(409, "Another environment operation is in progress")
            if environment_identity(self.versions()) != self.context.db.get_kv("environment.identity." + id_):
                raise EnvironmentError(
                    409, "Environment changed since this plan was created; create a new plan"
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
                            f"Downloaded wheel digest does not match the reviewed plan: {item['name']}"
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
            if {k: v for k, v in before.items() if protected(k)} != {
                k: v for k, v in after.items() if protected(k)
            }:
                raise RuntimeError(
                    "Protected runtime changed unexpectedly. Restore the original environment before training."
                )
            if op.action == "uninstall":
                if op.package in after:
                    raise RuntimeError("Package remains installed after uninstall")
                if CATALOG[op.package][1] == self.context.db.get_kv("environment.settings", {}).get(
                    "attention_default", "auto"
                ):
                    self.context.db.set_kv("environment.settings", {"attention_default": "auto"})
                    log(
                        "Removed the selected default backend; new configurations now use PyTorch SDPA (auto)."
                    )
            else:
                for item in op.plan:
                    if after.get(item["name"]) != item["version"]:
                        raise RuntimeError(f"Installed version does not match reviewed plan: {item['name']}")
                probes = self.probe()
                self._probe_cache, self._probe_time = probes, time.monotonic()
                probe = probes.get(op.package, {})
                if not probe.get("importable") or CATALOG[op.package][1] and not probe.get("kernel_tested"):
                    if probe.get("kernel_unavailable"):
                        raise RuntimeError(probe["error"])
                    raise RuntimeError(
                        "Package was installed but its runtime probe failed: "
                        + str(probe.get("error") or "Selected device kernel unavailable")
                    )
            self._update(id_, status="completed", restart_required=True)
            log("依赖已更新。重启 Studio 后生效并恢复任务队列。")
        except Exception as exc:
            self._update(id_, status="failed", error=str(exc), restart_required=mutation_started)
            log(str(exc))
        finally:
            self._probe_time = 0
            previous = self.context.db.get_kv("environment.prior_maintenance." + id_, {})
            restart = mutation_started or previous.get("restart_required", False)
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
                    "Applying dependency changes cannot be interrupted safely. Wait for verification, then restart Studio.",
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
