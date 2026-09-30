"""Reviewed SourceFind DTK attention wheels, not a PyPI/CUDA fallback.

The public directory was inspected on 2026-09-14 via
GET /api-static/file/ListFile?CategoryID=4&Path=/xformers (and /flash_attn).
Only immutable filenames whose METADATA and contents were inspected are listed.
Catalog compatibility permits preparation; only a real device probe establishes availability.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

from ypuddin import dtk_builds

SOURCE = "https://download.sourcefind.cn:65024"
SOURCE_PAGE = SOURCE + "/4/main/"


class DtkWheel(BaseModel):
    id: str
    package: str
    version: str
    filename: str
    url: str
    source_url: str = SOURCE_PAGE
    size_bytes: int
    sha256: str
    dtk: str
    torch: str
    declared_torch: str | None = None
    python_tag: str
    platform_tag: str
    binary: bool
    validation: str = "kernel_probe_required"
    requires_packages: list[str] = Field(default_factory=list)
    compatible: bool = False
    reason: str | None = None


class DtkCatalog(BaseModel):
    source_url: str = SOURCE_PAGE
    runtime: dict[str, str | None]
    wheels: list[DtkWheel]
    reason: str | None = None
    guidance: DtkGuidance


class DtkRuntimePackage(BaseModel):
    package: str
    version: str
    url: str


class DtkRuntimeRecommendation(BaseModel):
    dtk: str
    toolkit_url: str
    toolkit_checksum_url: str
    python_tag: str
    minimum_driver: str
    status: str = "candidate_requires_validation"
    reason: str = "matched_vendor_metadata_requires_driver_and_training_validation"
    wheels: list[DtkRuntimePackage]


class DtkGuidance(BaseModel):
    toolkit_source_url: str = SOURCE + "/1/main"
    driver_source_url: str = SOURCE + "/6/main"
    compatibility_source_url: str = SOURCE + "/file/1/" + urllib.parse.quote("DTK与驱动版本配套关系表.md")
    driver_version: str | None = None
    driver_verification: str = "manual_confirmation_required"
    current_stack_reason: str = "framework_combination_requires_validation"
    recommendation: DtkRuntimeRecommendation | None = None


def _url(path: str) -> str:
    return SOURCE + urllib.parse.quote(path, safe="/")


def system_info() -> dict[str, str | None]:
    """Read OS/toolkit identity without inspecting unrelated user data or changing the host."""
    release = {}
    if platform.system() == "Linux":
        try:
            release = platform.freedesktop_os_release()
        except OSError:
            pass
    root = Path(os.environ.get("DTK_ROOT", "/opt/dtk"))
    resolved = str(root.resolve()) if root.is_dir() else None
    toolkit = re.search(r"dtk[-_](\d{2}\.\d{2}(?:\.\d+)?)$", resolved or "", re.IGNORECASE)
    libc_name, libc_version = platform.libc_ver() if platform.system() == "Linux" else ("", "")
    return {
        "distribution": release.get("PRETTY_NAME"),
        "distribution_id": release.get("ID"),
        "distribution_version": release.get("VERSION_ID"),
        "kernel_release": platform.release(),
        "glibc_version": libc_version if libc_name == "glibc" else None,
        "dtk_root": resolved,
        "installed_dtk": toolkit[1] if toolkit else None,
    }


def driver_version() -> str | None:
    """Vendor rocm-smi emits text even with --json on some DTK releases."""
    if platform.system() != "Linux":
        return None
    executable = Path(os.environ.get("DTK_ROOT", "/opt/dtk")) / "bin" / "rocm-smi"
    if not executable.is_file():
        return None
    try:
        result = subprocess.run(
            [str(executable), "--showdriverversion"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        if result.returncode == 0:
            found = re.search(r"^\s*Driver Version:\s*([A-Za-z0-9_.+-]+)\s*$", result.stdout, re.MULTILINE)
            return found[1] if found else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _guidance_wheels(torch_version: str) -> tuple[tuple[str, str, str], ...]:
    # Manual download links from SourceFind's DAS1.8 inventory (2026-09-30).
    # They do not enter the checksum-pinned automatic installer catalog.
    if torch_version == "2.5.1":
        return tuple(
            (package, version, f"/file/4/{directory}/DAS1.8/{filename}")
            for package, directory, version, filename in (
                (
                    "torch", "pytorch", "2.5.1+das.opt1.dtk2604",
                    "torch-2.5.1+das.opt1.dtk2604-cp311-cp311-manylinux_2_28_x86_64.whl",
                ),
                (
                    "torchvision", "vision", "0.20.1+das.opt1.dtk2604.torch251",
                    "torchvision-0.20.1+das.opt1.dtk2604.torch251-cp311-cp311-manylinux_2_28_x86_64.whl",
                ),
                (
                    "triton", "triton", "3.1.0+das.opt1.dtk2604.torch251",
                    "triton-3.1.0+das.opt1.dtk2604.torch251-cp311-cp311-manylinux_2_28_x86_64.whl",
                ),
                (
                    "flash-attn", "flash_attn", "2.8.3+das.opt1.dtk2604.torch251",
                    "flash_attn-2.8.3+das.opt1.dtk2604.torch251-cp311-cp311-manylinux_2_28_x86_64.whl",
                ),
            )
        )
    if torch_version == "2.7.1":
        return (
            *(
                (package, version, path)
                for package, version, path, _, _ in dtk_builds.RUNTIME_SETS[("26.04", "cp311")]
            ),
            (
                "flash-attn",
                "2.8.3+das.opt1.dtk2604.torch271",
                "/file/4/flash_attn/DAS1.8/flash_attn-2.8.3+das.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl",
            ),
        )
    return ()


def guidance(runtime: dict, versions: dict[str, str] | None = None) -> DtkGuidance:
    result = DtkGuidance(driver_version=runtime.get("driver_version"))
    installed_torch = (versions or {}).get("torch") or runtime.get("torch")
    torch_version = None
    try:
        torch_version = Version(str(installed_torch)) if installed_torch else None
        if torch_version is not None and torch_version < Version("2.5"):
            result.current_stack_reason = "torch24_transformers5_diffusers040_conflict"
    except InvalidVersion:
        pass
    if (
        runtime.get("platform") == "Linux"
        and str(runtime.get("machine", "")).lower() in ("x86_64", "amd64")
        and runtime.get("distribution_id") == "ubuntu"
        and runtime.get("distribution_version") == "22.04"
        and re.match(r"^3\.11(?:\.|$)", str(runtime.get("python", "")))
    ):
        toolkit_version = runtime.get("installed_dtk")
        build_version = dtk_build_version({**runtime, "torch": installed_torch})
        if (toolkit_version and toolkit_version != "26.04") or (
            build_version and build_version != "26.04"
        ):
            if result.current_stack_reason != "torch24_transformers5_diffusers040_conflict":
                result.current_stack_reason = "no_matching_dtk_build"
            return result
        # Keep the current Torch line. The bootstrap default applies only when
        # no Torch is installed, never as an implicit upgrade recommendation.
        selected_torch = torch_version.base_version if torch_version is not None else "2.7.1"
        if installed_torch and (
            torch_version is None
            or torch_version.public != torch_version.base_version
            or torch_version.local not in (None, "das.opt1.dtk2604")
            or not (build_version or toolkit_version) == "26.04"
        ):
            result.current_stack_reason = "no_matching_torch_build"
            return result
        bundles = _guidance_wheels(selected_torch)
        if not bundles:
            if result.current_stack_reason != "torch24_transformers5_diffusers040_conflict":
                result.current_stack_reason = "no_matching_torch_build"
            return result
        toolkit = _url("/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz")
        result.recommendation = DtkRuntimeRecommendation(
            dtk="26.04",
            toolkit_url=toolkit,
            toolkit_checksum_url=toolkit + ".md5",
            python_tag="cp311",
            minimum_driver="6.3.30-V1.4.1a",
            reason="matches_installed_torch" if installed_torch else "new_environment",
            wheels=[
                DtkRuntimePackage(package=package, version=version, url=_url(path))
                for package, version, path in bundles
            ],
        )
    return result


WHEELS = (
    DtkWheel(
        id="sourcefind-flash-attn-2.6.1-dtk25041-cp311",
        package="flash-attn",
        version="2.6.1+das.opt1.dtk25041",
        filename="flash_attn-2.6.1+das.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl",
        url=_url(
            "/file/4/flash_attn/DAS1.6/flash_attn-2.6.1+das.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl"
        ),
        size_bytes=389084275,
        sha256="5ce0a673a64fc7fead3289a6268932200f09d4d1e0903a09231bf765f649e7f9",
        dtk="25.04.1",
        torch="==2.4.1",
        python_tag="cp311",
        platform_tag="manylinux_2_28_x86_64",
        binary=True,
        requires_packages=["einops", "triton==3.0.0+das.opt1.dtk25041"],
    ),
    DtkWheel(
        id="sourcefind-xformers-0.0.33-python",
        package="xformers",
        version="0.0.33+das.opt1.dtk2604.torch251",
        filename="xformers-0.0.33+das.opt1.dtk2604.torch251-py3-none-any.whl",
        url=_url("/file/4/xformers/DAS1.8/xformers-0.0.33+das.opt1.dtk2604.torch251-py3-none-any.whl"),
        size_bytes=251694,
        sha256="bbcf795c71ff248e261a56ba43a9a9dd8b14f944872c9d9ee8efd260b1bdb915",
        # These are the vendor's directory/build labels, not a native ABI dependency:
        # this wheel contains Python only; METADATA declares torch>=2.1.0.
        dtk="26.04 (vendor label; Python-only)",
        # The vendor METADATA is too broad: the real Torch 2.4.1 probe failed
        # while importing torch.distributed._symmetric_memory on 2026-09-14.
        torch=">=2.5",
        declared_torch=">=2.1.0",
        python_tag="py3",
        platform_tag="any",
        binary=False,
        requires_packages=["numpy", "flash-attn>=2.6.1"],
    ),
    DtkWheel(
        id="sourcefind-flash-attn-2.8.3-dtk2604-torch271-cp311",
        package="flash-attn",
        version="2.8.3+das.opt1.dtk2604.torch271",
        filename="flash_attn-2.8.3+das.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl",
        url=_url(
            "/file/4/flash_attn/DAS1.8/flash_attn-2.8.3+das.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl"
        ),
        size_bytes=658880343,
        sha256="d2cdd700de8622b2473bbac57328ca6682eb4025c274a6b7f7f50c687a3c554b",
        dtk="26.04",
        torch="==2.7.1",
        python_tag="cp311",
        platform_tag="manylinux_2_28_x86_64",
        binary=True,
        # The reviewed wheel's public __version__ remains 2.6.1. It exposes
        # flash_attn_func, but not Diffusers' wrapped forward/backward interfaces.
        # Post-install probing establishes only the public FA kernel capability;
        # model-specific integration and full training still require verification.
        requires_packages=["einops", "triton==3.1.0+das.opt1.dtk2604.torch271"],
    ),
)


def dtk_build_version(runtime: dict) -> str | None:
    # Vendor Torch records the DTK build; do not infer it from generic HIP numbering.
    match = re.search(r"dtk(\d{2})(\d{2})(\d*)", str(runtime.get("torch", "")))
    if match:
        return f"{match[1]}.{match[2]}" + (f".{match[3]}" if match[3] else "")
    root = Path(os.environ.get("DTK_ROOT", "/opt/dtk"))
    if root.is_dir():
        match = re.search(r"dtk[-_](\d{2}\.\d{2}(?:\.\d+)?)$", str(root.resolve()), re.IGNORECASE)
        return match[1] if match else None
    return None


def incompatibility(wheel: DtkWheel, runtime: dict, versions: dict[str, str], profile: str) -> str | None:
    from packaging.requirements import Requirement

    if profile != "linux-dtk":
        return "dtk_profile_required"
    if runtime.get("platform") != "Linux" or str(runtime.get("machine", "")).lower() not in (
        "x86_64",
        "amd64",
    ):
        return "requires_linux_x86_64"
    if not runtime.get("hip_runtime") or not runtime.get("cuda_available"):
        return "hip_runtime_unavailable"
    try:
        if Version(str(runtime.get("torch", "0"))) not in SpecifierSet(wheel.torch):
            return (
                "requires_torch_runtime_api" + wheel.torch
                if wheel.declared_torch
                else "torch_version_mismatch"
            )
    except InvalidVersion:
        return "torch_version_mismatch"
    if wheel.binary and (
        dtk_build_version(runtime) != wheel.dtk
        or runtime.get("installed_dtk")
        and runtime["installed_dtk"] != wheel.dtk
    ):
        return "dtk_version_mismatch"
    py = re.match(r"(\d+)\.(\d+)", str(runtime.get("python", "")))
    if not py or (wheel.python_tag != "py3" and wheel.python_tag != "cp" + py[1] + py[2]):
        return "python_abi_mismatch"
    for value in wheel.requires_packages:
        req = Requirement(value)
        actual = versions.get(req.name)
        if not actual or actual not in req.specifier:
            return "requires_package:" + value
    if not re.fullmatch(r"[a-f0-9]{64}", wheel.sha256):
        return "integrity_verification_pending"
    return None


def catalog(runtime: dict, versions: dict[str, str], profile: str) -> DtkCatalog:
    wheels = []
    for wheel in WHEELS:
        reason = incompatibility(wheel, runtime, versions, profile)
        wheels.append(wheel.model_copy(update={"compatible": reason is None, "reason": reason}))
    return DtkCatalog(
        runtime={
            "environment_profile": profile,
            "torch": str(runtime.get("torch", "")),
            "python": str(runtime.get("python", "")),
            "dtk": dtk_build_version(runtime),
            "machine": str(runtime.get("machine", "")),
            **{
                name: runtime.get(name)
                for name in (
                    "distribution",
                    "distribution_id",
                    "distribution_version",
                    "kernel_release",
                    "glibc_version",
                    "dtk_root",
                    "installed_dtk",
                    "hip_runtime",
                )
            },
        },
        guidance=guidance(runtime, versions),
        wheels=wheels,
        reason=None
        if any(w.compatible for w in wheels)
        else "no_matching_vendor_build"
        if profile == "linux-dtk"
        else "dtk_profile_required",
    )


def find_wheel(id_: str) -> DtkWheel:
    for wheel in WHEELS:
        if wheel.id == id_:
            return wheel
    raise ValueError("Unknown DTK vendor wheel")


def python_runtime_requirements(wheel: DtkWheel) -> tuple[str, ...]:
    # Both reviewed Flash builds import flash_attn_triton_interface from __init__,
    # which unconditionally imports modules using pytest. Their METADATA omits it.
    # The reviewed Python xFormers adapter imports the installed vendor Flash too.
    return ("pytest",) if wheel.package in ("flash-attn", "xformers") else ()


def wheel_for_file(path: Path) -> DtkWheel:
    for wheel in WHEELS:
        if wheel.filename == path.name:
            return wheel
    raise ValueError("Use a reviewed SourceFind vendor wheel matching this DTK environment")


def permitted_url(url: str) -> bool:
    value = urllib.parse.urlsplit(url)
    return (
        value.scheme == "https"
        and value.hostname == "download.sourcefind.cn"
        and value.port == 65024
        and not value.username
        and not value.password
        and value.path.startswith("/file/4/")
        and not value.query
        and not value.fragment
    )


class _VendorRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not permitted_url(newurl):
            raise ValueError("SourceFind download redirected outside its approved origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(
    wheel: DtkWheel,
    destination: Path,
    cancel: threading.Event,
    progress: Callable[[int, int, float | None, float | None], None],
    *,
    opener=None,
    proxy=None,
) -> Path:
    """Stream a pinned vendor wheel with bounded reads and measured progress; never execute it."""
    if cancel.is_set():
        raise InterruptedError()
    if not permitted_url(wheel.url) or not re.fullmatch(r"[a-f0-9]{64}", wheel.sha256):
        raise ValueError("Vendor source or pinned SHA256 is not verified")
    if Path(wheel.filename).name != wheel.filename or not 0 < wheel.size_bytes <= 2 * 1024**3:
        raise ValueError("Invalid vendor wheel filename or size")
    parse_wheel_filename(wheel.filename)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / wheel.filename
    if target.is_file() and target.stat().st_size == wheel.size_bytes and file_hash(target) == wheel.sha256:
        progress(wheel.size_bytes, wheel.size_bytes, None, 0)
        return target
    temporary = target.with_suffix(".whl.partial")
    downloaded = 0
    previous_at, previous_bytes, speed = time.monotonic(), 0, None
    progress(0, wheel.size_bytes, None, None)
    from .network import ProxyPolicy

    policy = proxy or ProxyPolicy()
    try:
        open_url = opener or policy.opener(_VendorRedirect()).open
        request = urllib.request.Request(wheel.url, headers={"User-Agent": "YPuddin-DTK/1"})
        with open_url(request, timeout=10) as response, temporary.open("wb") as stream:
            length = response.headers.get("Content-Length")
            if length and int(length) != wheel.size_bytes:
                raise ValueError("Vendor wheel size differs from the reviewed catalog")
            if not permitted_url(response.geturl()):
                raise ValueError("Unapproved vendor download response")
            digest = hashlib.sha256()
            while True:
                if cancel.is_set():
                    raise InterruptedError()
                chunk = response.read(1024**2)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > wheel.size_bytes:
                    raise ValueError("Vendor wheel exceeds its reviewed size")
                stream.write(chunk)
                digest.update(chunk)
                moment = time.monotonic()
                if moment - previous_at >= 0.25 or downloaded == wheel.size_bytes:
                    interval = moment - previous_at
                    measured = (downloaded - previous_bytes) / interval if interval > 0 else None
                    if measured is not None:
                        speed = measured if speed is None else 0.35 * measured + 0.65 * speed
                    progress(
                        downloaded,
                        wheel.size_bytes,
                        speed,
                        (wheel.size_bytes - downloaded) / speed if speed and speed > 0 else None,
                    )
                    previous_at, previous_bytes = moment, downloaded
            if downloaded != wheel.size_bytes or digest.hexdigest() != wheel.sha256:
                raise ValueError("Vendor wheel SHA256 or size differs from the reviewed catalog")
        if cancel.is_set():
            raise InterruptedError()
        temporary.replace(target)
        progress(downloaded, wheel.size_bytes, None, 0)
        return target
    except Exception as exc:
        message = policy.redact(exc)
        if message != str(exc):
            raise ValueError(message) from None
        raise
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()
