"""The newest PyPI release of an optional package that the running PyTorch and Python can install."""

from __future__ import annotations

import json
import urllib.request

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

import ypuddin

PYPI = "https://pypi.org/pypi"
MAX_BYTES = 16 * 1024 * 1024
CHECKED = 8  # Newest releases looked at before giving up.


def _read(opener, url: str) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": f"YPuddinTrainStudio/{ypuddin.__version__}", "Accept": "application/json"}
    )
    with opener.open(request, timeout=15) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("The package index answer is too large")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("The package index answer is not an object")
    return data


def _installable(filename: str, supported: set) -> bool:
    try:
        return bool(parse_wheel_filename(filename)[3] & supported)
    except InvalidWheelFilename:
        return False


def _accepts(info: dict, torch: Version, python: Version) -> bool:
    try:
        if info.get("requires_python") and python not in SpecifierSet(info["requires_python"]):
            return False
    except InvalidSpecifier:
        return False
    for text in info.get("requires_dist") or []:
        try:
            requirement = Requirement(text)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) != "torch":
            continue
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        if not requirement.specifier.contains(torch, prereleases=True):
            return False
    return True


def newest_release(name: str, *, torch: str, python: str, opener) -> str | None:
    """The newest final release with a wheel for this interpreter whose requirements accept the
    installed PyTorch (without its local +cu tag) and Python; None when the newest few do not."""
    data = _read(opener, f"{PYPI}/{name}/json")
    supported = set(sys_tags())
    torch_version = Version(Version(torch).base_version)
    python_version = Version(python)
    candidates = []
    for text, files in (data.get("releases") or {}).items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease or not isinstance(files, list):
            continue
        if any(
            isinstance(item, dict)
            and item.get("packagetype") == "bdist_wheel"
            and not item.get("yanked")
            and _installable(str(item.get("filename", "")), supported)
            for item in files
        ):
            candidates.append((version, text))
    latest = data.get("info") or {}
    for _, text in sorted(candidates, reverse=True)[:CHECKED]:
        info = (
            latest
            if latest.get("version") == text
            else _read(opener, f"{PYPI}/{name}/{text}/json").get("info") or {}
        )
        if _accepts(info, torch_version, python_version):
            return text
    return None
