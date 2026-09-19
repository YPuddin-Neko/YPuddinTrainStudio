"""Bounded Windows FA2 discovery from a reviewed community publisher's releases.

The bundled asset metadata was checked against the maintainer's GitHub release on
2026-09-15. A filename match is an installation candidate, not a GPU qualification.
Only the Windows cp312/cu128/Torch2.11 build has prior project hardware acceptance.
"""

from __future__ import annotations

import hashlib
import http.client
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
BUNDLED_RELEASE = "v0.9.6"
SOURCE_URL = f"https://github.com/{REPOSITORY}/releases"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases"
DOWNLOAD_PREFIX = f"https://github.com/{REPOSITORY}/releases/download/"
PROVIDER = "mjun0812-community-windows"
MAX_METADATA_BYTES = 2 * 1024**2
RELEASES_PER_PAGE = 10
MAX_RELEASE_PAGES = 5
TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
NAME = re.compile(r"flash_attn-(2\.\d+\.\d+)\+cu(\d{3})torch(\d+\.\d+)-(cp3\d+)-\4-win_amd64\.whl")


def _download_prefix(release: str) -> str:
    if not TAG.fullmatch(release):
        raise ValueError("Invalid community release tag")
    return DOWNLOAD_PREFIX + urllib.parse.quote(release, safe="") + "/"


def _release_url(release: str) -> str:
    return SOURCE_URL + "/tag/" + urllib.parse.quote(release, safe="")


class WindowsAttentionWheel(BaseModel):
    id: str
    package: str = "flash-attn"
    version: str
    filename: str
    url: str
    source_url: str = _release_url(BUNDLED_RELEASE)
    release: str = BUNDLED_RELEASE
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
    release: str = BUNDLED_RELEASE
    release_count: int = 1
    limited: bool = False
    unverified_assets: int = 0
    provider: str = PROVIDER
    origin: Literal["live", "cached", "bundled"]
    checked_at: float | None = None
    error: str | None = None
    reason: str | None = None
    runtime: dict[str, str | None]
    wheels: list[WindowsAttentionWheel]


def parse_assets(document: dict, *, allow_empty=False) -> tuple[WindowsAttentionWheel, ...]:
    release = document.get("tag_name")
    if (
        not isinstance(release, str)
        or not TAG.fullmatch(release)
        or document.get("draft")
        or document.get("prerelease")
    ):
        raise ValueError("Release identity does not match the reviewed publisher release")
    prefix = _download_prefix(release)
    wheels = []
    seen = set()
    assets = document.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("Invalid release asset list")
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("name", ""), str):
            raise ValueError("Invalid release asset entry")
        filename = asset.get("name", "")
        match = NAME.fullmatch(filename)
        if not match:
            continue  # FA3, Linux and source archives are not FA2 Windows candidates.
        digest = asset.get("digest", "")
        size = asset.get("size")
        url = asset.get("browser_download_url", "")
        if (
            not isinstance(size, int)
            or not 0 < size <= 2 * 1024**3
            or url != prefix + urllib.parse.quote(filename, safe="")
            or filename in seen
        ):
            raise ValueError("Release contains duplicate or unverified Windows wheel metadata")
        seen.add(filename)
        # Older releases can lack GitHub's SHA256 metadata. Never turn them into
        # download candidates, but do not hide verified builds in other releases.
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
            continue
        base, cuda, torch, python = match.groups()
        wheels.append(
            WindowsAttentionWheel(
                id="mjun0812-" + release + "-" + filename,
                version=f"{base}+cu{cuda}torch{torch}",
                filename=filename,
                url=url,
                source_url=_release_url(release),
                release=release,
                size_bytes=size,
                sha256=digest[7:],
                torch=torch,
                cuda=cuda[:-1] + "." + cuda[-1],
                python_tag=python,
            )
        )
    if not wheels and not allow_empty:
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
        raise ValueError("Release metadata redirected outside its approved API endpoint")


def discover(policy):
    wheels, releases, seen = [], [], set()
    unverified_assets, limited = 0, False
    for page in range(1, MAX_RELEASE_PAGES + 1):
        url = f"{API_URL}?per_page={RELEASES_PER_PAGE}&page={page}"
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "YPuddin-Windows-FA2/1"},
        )
        with policy.opener(_ReleaseRedirect()).open(request, timeout=8) as response:
            if response.geturl() != url:
                raise ValueError("Unexpected release metadata response origin")
            raw = response.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("Release metadata exceeds the size limit")
        documents = json.loads(raw)
        if not isinstance(documents, list) or len(documents) > RELEASES_PER_PAGE:
            raise ValueError("Invalid community release listing")
        for document in documents:
            if not isinstance(document, dict):
                raise ValueError("Invalid community release entry")
            if document.get("draft") or document.get("prerelease"):
                continue
            candidates = parse_assets(document, allow_empty=True)
            releases.append(document["tag_name"])
            unverified_assets += sum(
                bool(NAME.fullmatch(asset.get("name", "")))
                and not re.fullmatch(r"sha256:[a-f0-9]{64}", str(asset.get("digest", "")))
                for asset in document.get("assets", [])
            )
            for wheel in candidates:
                identity = (wheel.filename, wheel.sha256)
                if identity not in seen:
                    seen.add(identity)
                    wheels.append(wheel)
        if len(documents) < RELEASES_PER_PAGE:
            break
        limited = page == MAX_RELEASE_PAGES
    if not wheels:
        raise ValueError("No verified Windows FA2 wheels were found in the queried releases")
    wheels.sort(
        key=lambda w: (Version(w.version.split("+")[0]), Version(w.torch), Version(w.cuda)), reverse=True
    )
    return tuple(wheels), releases, limited, unverified_assets


class Catalog:
    def __init__(self):
        self._lock = threading.Lock()
        self._wheels = BUNDLED
        self._checked_at = None
        self._attempted_at = 0.0
        self._error = None
        self._policy = None
        self._releases = [BUNDLED_RELEASE]
        self._limited = False
        self._unverified_assets = 0

    def snapshot(self, runtime, profile, *, proxy=None, refresh=False):
        policy = proxy or ProxyPolicy()
        supported = (
            runtime.get("platform") == "Windows" and not runtime.get("hip_runtime") and profile != "linux-dtk"
        )
        with self._lock:
            updated = False
            if supported and (
                refresh or policy != self._policy or time.monotonic() - self._attempted_at > 300
            ):
                self._attempted_at, self._policy = time.monotonic(), policy
                try:
                    self._wheels, self._releases, self._limited, self._unverified_assets = discover(policy)
                    self._checked_at, self._error = time.time(), None
                    updated = True
                except (OSError, http.client.HTTPException, ValueError, TypeError, KeyError) as exc:
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
                release=self._releases[0],
                release_count=len(self._releases),
                limited=self._limited,
                unverified_assets=self._unverified_assets,
                origin="bundled" if self._checked_at is None else "live" if updated else "cached",
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
        if not url.startswith(DOWNLOAD_PREFIX) or value.query:
            return False
        parts = url[len(DOWNLOAD_PREFIX) :].split("/")
        if len(parts) != 2:
            return False
        release, filename = map(urllib.parse.unquote, parts)
        return bool(TAG.fullmatch(release) and NAME.fullmatch(filename)) and url == (
            _download_prefix(release) + urllib.parse.quote(filename, safe="")
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
        wheel.url != _download_prefix(wheel.release) + urllib.parse.quote(wheel.filename, safe="")
        or not NAME.fullmatch(wheel.filename)
        or not re.fullmatch(r"[a-f0-9]{64}", wheel.sha256)
    ):
        raise ValueError("Unverified Windows community wheel source")
    if not 0 < wheel.size_bytes <= 2 * 1024**3:
        raise ValueError("Invalid reviewed wheel size")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / wheel.filename
    # Reuse only bytes verified against the current release metadata.
    if target.is_file() and target.stat().st_size == wheel.size_bytes:
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                if cancel.is_set():
                    raise InterruptedError()
                digest.update(chunk)
        if digest.hexdigest() == wheel.sha256:
            progress(wheel.size_bytes, wheel.size_bytes, None, 0)
            return target
    temporary = target.with_suffix(".whl.partial")
    total, previous_bytes, previous_at, speed = 0, 0, time.monotonic(), None
    progress(0, wheel.size_bytes, None, None)
    try:
        request = urllib.request.Request(wheel.url, headers={"User-Agent": "YPuddin-Windows-FA2/1"})
        open_url = opener or policy.opener(_AssetRedirect()).open
        with open_url(request, timeout=30) as response, temporary.open("wb") as stream:
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
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise ValueError(
            "社区 wheel 下载失败；请检查全局代理，或从发布页手动下载后上传。 / Community wheel download failed; check proxy settings or download and upload manually. "
            + policy.redact(exc)
        ) from None
    finally:
        temporary.unlink(missing_ok=True)
