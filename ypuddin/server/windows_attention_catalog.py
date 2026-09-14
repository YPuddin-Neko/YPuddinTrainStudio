"""Bounded Windows FA2 discovery from a reviewed community publisher's fixed release.

The bundled asset metadata was checked against the maintainer's GitHub release on
2026-09-15. A filename match is an installation candidate, not a GPU qualification.
Only the Windows cp312/cu128/Torch2.11 build has prior project hardware acceptance.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel

from .network import ProxyPolicy
from .windows_attention_assets import ASSETS

REPOSITORY = "mjun0812/flash-attention-prebuild-wheels"
RELEASE = "v0.9.6"
SOURCE_URL = f"https://github.com/{REPOSITORY}/releases/tag/{RELEASE}"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{RELEASE}"
PREFIX = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE}/"
PROVIDER = "mjun0812-community-windows"
MAX_METADATA_BYTES = 2 * 1024**2
NAME = re.compile(r"flash_attn-(\d+\.\d+\.\d+)\+cu(\d{3})torch(\d+\.\d+)-(cp3\d+)-\4-win_amd64\.whl")


class WindowsAttentionWheel(BaseModel):
    id: str
    package: str = "flash-attn"
    version: str
    filename: str
    url: str
    source_url: str = SOURCE_URL
    provider: str = PROVIDER
    size_bytes: int
    sha256: str
    torch: str
    cuda: str
    python_tag: str
    platform_tag: str = "win_amd64"
    validation: str = "kernel_probe_required"
    compatible: bool = False
    reason: str | None = None


class WindowsAttentionCatalog(BaseModel):
    source_url: str = SOURCE_URL
    release: str = RELEASE
    provider: str = PROVIDER
    origin: Literal["live", "cached", "bundled"]
    checked_at: float | None = None
    error: str | None = None
    reason: str | None = None
    runtime: dict[str, str | None]
    wheels: list[WindowsAttentionWheel]


def parse_assets(document: dict) -> tuple[WindowsAttentionWheel, ...]:
    if document.get("tag_name") != RELEASE or document.get("draft") or document.get("prerelease"):
        raise ValueError("Release identity does not match the reviewed publisher release")
    wheels = []
    seen = set()
    for asset in document.get("assets", []):
        filename = asset.get("name", "")
        match = NAME.fullmatch(filename)
        if not match:
            continue  # FA3, Linux and source archives are not FA2 Windows candidates.
        digest = asset.get("digest", "")
        size = asset.get("size")
        url = asset.get("browser_download_url", "")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest)
            or not isinstance(size, int)
            or not 0 < size <= 2 * 1024**3
            or url != PREFIX + urllib.parse.quote(filename, safe="")
            or filename in seen
        ):
            raise ValueError("Release contains duplicate or unverified Windows wheel metadata")
        seen.add(filename)
        base, cuda, torch, python = match.groups()
        wheels.append(
            WindowsAttentionWheel(
                id="mjun0812-" + RELEASE + "-" + filename,
                version=f"{base}+cu{cuda}torch{torch}",
                filename=filename,
                url=url,
                size_bytes=size,
                sha256=digest[7:],
                torch=torch,
                cuda=cuda[:-1] + "." + cuda[-1],
                python_tag=python,
            )
        )
    if not wheels:
        raise ValueError("The reviewed release contains no verified Windows FA2 wheels")
    return tuple(wheels)


BUNDLED = parse_assets(ASSETS)


def incompatibility(wheel, runtime, versions=None, profile=None):
    if profile == "linux-dtk" or runtime.get("hip_runtime"):
        return "requires_windows_cuda"
    if runtime.get("platform") != "Windows" or str(runtime.get("machine", "")).lower() not in (
        "amd64",
        "x86_64",
    ):
        return "requires_windows_x86_64"
    if not runtime.get("cuda_available"):
        return "cuda_runtime_unavailable"
    try:
        python = Version(str(runtime.get("python", "")))
        torch = Version(str(runtime.get("torch", "")))
    except InvalidVersion:
        return "runtime_version_unrecognized"
    if python.is_prerelease or "cp" + "".join(map(str, python.release[:2])) != wheel.python_tag:
        return "python_abi_mismatch"
    if torch.is_prerelease or tuple(torch.release[:2]) != Version(wheel.torch).release:
        return "torch_version_mismatch"
    if str(runtime.get("cuda_runtime", "")) != wheel.cuda or torch.local not in (
        None,
        "cu" + wheel.cuda.replace(".", ""),
    ):
        return "cuda_version_mismatch"
    return None


class _ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # GitHub metadata needs no redirect. Do not follow changed repositories or login pages.
        raise ValueError("Release metadata redirected outside its fixed API endpoint")


class Catalog:
    def __init__(self):
        self._lock = threading.Lock()
        self._wheels = BUNDLED
        self._checked_at = None
        self._attempted_at = 0.0
        self._error = None
        self._policy = None

    def snapshot(self, runtime, profile, *, proxy=None, refresh=False):
        policy = proxy or ProxyPolicy()
        supported = (
            runtime.get("platform") == "Windows" and not runtime.get("hip_runtime") and profile != "linux-dtk"
        )
        with self._lock:
            if supported and (
                refresh or policy != self._policy or time.monotonic() - self._attempted_at > 300
            ):
                self._attempted_at, self._policy = time.monotonic(), policy
                try:
                    request = urllib.request.Request(
                        API_URL,
                        headers={
                            "Accept": "application/vnd.github+json",
                            "User-Agent": "YPuddin-Windows-FA2/1",
                        },
                    )
                    with policy.opener(_ReleaseRedirect()).open(request, timeout=8) as response:
                        if response.geturl() != API_URL:
                            raise ValueError("Unexpected release metadata response origin")
                        raw = response.read(MAX_METADATA_BYTES + 1)
                    if len(raw) > MAX_METADATA_BYTES:
                        raise ValueError("Release metadata exceeds the size limit")
                    self._wheels = parse_assets(json.loads(raw))
                    self._checked_at, self._error = time.time(), None
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    self._error = (
                        "无法更新社区版本目录，已使用已保存的版本信息。请检查网络或全局代理设置。 / Cannot refresh community releases; using saved metadata. Check network/proxy settings. "
                        + policy.redact(exc)
                    )
            wheels = [
                w.model_copy(
                    update={
                        "reason": (reason := incompatibility(w, runtime, profile=profile)),
                        "compatible": reason is None,
                    }
                )
                for w in self._wheels
            ]
            return WindowsAttentionCatalog(
                origin="bundled" if self._checked_at is None else "cached" if self._error else "live",
                checked_at=self._checked_at,
                error=self._error,
                reason=None if any(w.compatible for w in wheels) else "no_matching_build",
                runtime={
                    key: str(runtime.get(key) or "")
                    for key in ("python", "torch", "cuda_runtime", "platform", "machine")
                },
                wheels=wheels,
            )

    def find_wheel(self, id_):
        with self._lock:
            for wheel in self._wheels:
                if wheel.id == id_:
                    return wheel
        raise ValueError("Unknown reviewed Windows community wheel; refresh the build list")


def permitted_download(url: str) -> bool:
    value = urllib.parse.urlsplit(url)
    if (
        value.scheme != "https"
        or value.port not in (None, 443)
        or value.username
        or value.password
        or value.fragment
    ):
        return False
    if value.hostname == "github.com":
        return (
            url.startswith(PREFIX)
            and not value.query
            and bool(NAME.fullmatch(urllib.parse.unquote(value.path.rsplit("/", 1)[-1])))
        )
    # GitHub release assets redirect to this signed CDN; no arbitrary GitHubusercontent subdomains.
    return value.hostname == "release-assets.githubusercontent.com" and value.path.startswith(
        "/github-production-release-asset/"
    )


class _AssetRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not permitted_download(newurl):
            raise ValueError("Wheel download redirected outside the approved GitHub release hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(wheel, destination, cancel, progress, *, proxy=None, opener=None):
    """Download to staging only; hash/size must match before existing wheel/plan validation."""
    policy = proxy or ProxyPolicy()
    if cancel.is_set():
        raise InterruptedError()
    if (
        wheel.url != PREFIX + urllib.parse.quote(wheel.filename, safe="")
        or not NAME.fullmatch(wheel.filename)
        or not re.fullmatch(r"[a-f0-9]{64}", wheel.sha256)
    ):
        raise ValueError("Unverified Windows community wheel source")
    if not 0 < wheel.size_bytes <= 2 * 1024**3:
        raise ValueError("Invalid reviewed wheel size")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / wheel.filename
    temporary = target.with_suffix(".whl.partial")
    total, previous_bytes, previous_at, speed = 0, 0, time.monotonic(), None
    progress(0, wheel.size_bytes, None, None)
    try:
        request = urllib.request.Request(wheel.url, headers={"User-Agent": "YPuddin-Windows-FA2/1"})
        open_url = opener or policy.opener(_AssetRedirect()).open
        with open_url(request, timeout=10) as response, temporary.open("wb") as stream:
            if not permitted_download(response.geturl()):
                raise ValueError("Unapproved wheel download response origin")
            length = response.headers.get("Content-Length")
            if length and int(length) != wheel.size_bytes:
                raise ValueError("Wheel size differs from the reviewed release")
            digest = hashlib.sha256()
            while True:
                if cancel.is_set():
                    raise InterruptedError()
                chunk = response.read(1024**2)
                if not chunk:
                    break
                total += len(chunk)
                if total > wheel.size_bytes:
                    raise ValueError("Wheel exceeds its reviewed size")
                stream.write(chunk)
                digest.update(chunk)
                now = time.monotonic()
                if now - previous_at >= 0.25 or total == wheel.size_bytes:
                    measured = (total - previous_bytes) / (now - previous_at) if now > previous_at else None
                    if measured is not None:
                        speed = measured if speed is None else 0.35 * measured + 0.65 * speed
                    progress(
                        total, wheel.size_bytes, speed, (wheel.size_bytes - total) / speed if speed else None
                    )
                    previous_at, previous_bytes = now, total
            if total != wheel.size_bytes or digest.hexdigest() != wheel.sha256:
                raise ValueError("Wheel SHA256 or size differs from the reviewed release")
        if cancel.is_set():
            raise InterruptedError()
        temporary.replace(target)
        progress(total, wheel.size_bytes, None, 0)
        return target
    except InterruptedError:
        raise
    except (urllib.error.URLError, OSError) as exc:
        raise ValueError(
            "社区 wheel 下载失败；请检查全局代理，或从发布页手动下载后上传。 / Community wheel download failed; check proxy settings or download and upload manually. "
            + policy.redact(exc)
        ) from None
    finally:
        temporary.unlink(missing_ok=True)
