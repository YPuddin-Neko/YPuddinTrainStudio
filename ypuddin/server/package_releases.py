"""The newest release of an optional package that the running Python (and PyTorch) can install,
read from the same package sources the installer uses.

Sources are PEP 503 / PEP 691 simple indexes (``index-url``) or flat file listings
(``find-links``). They are tried in the installer's order; a source that cannot be reached, or that
has no compatible build, hands over to the next one, as it does during an installation.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.parser import HeaderParser
from html.parser import HTMLParser

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

import ypuddin

PYPI = "https://pypi.org/pypi"
MAX_BYTES = 16 * 1024 * 1024
CHECKED = 8  # Newest releases looked at before giving up.
SIMPLE_JSON = "application/vnd.pypi.simple.v1+json"


@dataclass(frozen=True)
class IndexFile:
    filename: str
    url: str
    version: Version
    tags: frozenset
    requires_python: str | None
    yanked: bool
    metadata: bool  # the source serves this wheel's METADATA at ``url + ".metadata"``


@dataclass(frozen=True)
class Release:
    version: str | None  # None: the sources answered, but none has a compatible build
    index: str  # the source that decided the answer


def _get(opener, url: str, accept: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url, headers={"User-Agent": f"YPuddinTrainStudio/{ypuddin.__version__}", "Accept": accept}
    )
    with opener.open(request, timeout=15) as response:
        raw = response.read(MAX_BYTES + 1)
        headers = getattr(response, "headers", None)
        kind = (headers.get("Content-Type", "") if headers is not None else "") or ""
    if len(raw) > MAX_BYTES:
        raise ValueError("The package index answer is too large")
    return raw, kind.split(";")[0].strip().lower()


def _read(opener, url: str) -> dict:
    raw, _ = _get(opener, url, "application/json")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("The package index answer is not an object")
    return data


class _Links(HTMLParser):
    """Anchors of a PEP 503 page or a flat listing, with their data-* attributes."""

    def __init__(self):
        super().__init__()
        self.links: list[tuple[dict[str, str | None], str]] = []
        self._current: dict[str, str | None] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._current, self._text = dict(attrs), []

    def handle_data(self, data):
        if self._current is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._current is not None:
            self.links.append((self._current, "".join(self._text).strip()))
            self._current = None


def _file(name: str, filename: str, url: str, requires_python, yanked, metadata) -> IndexFile | None:
    if not filename.endswith(".whl"):
        return None
    try:
        project, version, _, tags = parse_wheel_filename(filename)
    except (InvalidWheelFilename, InvalidVersion):
        return None
    if project != canonicalize_name(name):
        return None
    return IndexFile(
        filename=filename,
        url=url.split("#", 1)[0],
        version=version,
        tags=frozenset(tags),
        requires_python=requires_python or None,
        yanked=bool(yanked),
        metadata=bool(metadata) and metadata != "false",
    )


def index_files(name: str, kind: str, source: str, opener) -> list[IndexFile]:
    """Every wheel of ``name`` that one source lists."""
    page = source if kind == "find-links" else f"{source.rstrip('/')}/{canonicalize_name(name)}/"
    raw, content_type = _get(opener, page, f"{SIMPLE_JSON}, text/html;q=0.1")
    files: list[IndexFile | None] = []
    if content_type == SIMPLE_JSON or (not content_type and raw.lstrip()[:1] == b"{"):
        data = json.loads(raw)
        for item in data.get("files") or [] if isinstance(data, dict) else []:
            if not isinstance(item, dict):
                continue
            metadata = item.get("core-metadata", item.get("data-dist-info-metadata"))
            files.append(
                _file(
                    name,
                    str(item.get("filename", "")),
                    urllib.parse.urljoin(page, str(item.get("url", ""))),
                    item.get("requires-python"),
                    # PEP 691: true, or the reason as a string, marks a yanked file.
                    item.get("yanked") is True or isinstance(item.get("yanked"), str),
                    metadata,
                )
            )
    else:
        parser = _Links()
        parser.feed(raw.decode("utf-8", "replace"))
        for attrs, label in parser.links:
            href = attrs.get("href") or ""
            filename = label or urllib.parse.unquote(href.split("#", 1)[0].rsplit("/", 1)[-1])
            metadata = attrs.get("data-core-metadata", attrs.get("data-dist-info-metadata"))
            files.append(
                _file(
                    name,
                    filename,
                    urllib.parse.urljoin(page, href),
                    attrs.get("data-requires-python"),
                    "data-yanked" in attrs,
                    metadata if metadata is not None else None,
                )
            )
    return [item for item in files if item is not None]


def _python_ok(specifier: str | None, python: Version) -> bool:
    try:
        return not specifier or python in SpecifierSet(specifier)
    except InvalidSpecifier:
        return False


def _torch_ok(requirements: list[str], torch: Version) -> bool:
    for text in requirements:
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


def _requirements(name: str, file: IndexFile, opener) -> list[str]:
    """Requires-Dist of one wheel: from the source's metadata file when it serves one, else PyPI."""
    if file.metadata:
        raw, _ = _get(opener, file.url + ".metadata", "*/*")
        return HeaderParser().parsestr(raw.decode("utf-8", "replace")).get_all("Requires-Dist") or []
    info = _read(opener, f"{PYPI}/{name}/{file.version}/json").get("info") or {}
    return list(info.get("requires_dist") or [])


def _newest(
    name: str, files: list[IndexFile], *, torch: Version | None, python: Version, opener
) -> str | None:
    supported = set(sys_tags())
    usable: dict[Version, list[IndexFile]] = {}
    for file in files:
        if file.yanked or file.version.is_prerelease or file.version.is_devrelease:
            continue
        if not file.tags & supported or not _python_ok(file.requires_python, python):
            continue
        usable.setdefault(file.version, []).append(file)
    unverified = False
    for version in sorted(usable, reverse=True)[:CHECKED]:
        if torch is None:
            return str(version)
        candidates = sorted(usable[version], key=lambda item: not item.metadata)
        try:
            requirements = _requirements(name, candidates[0], opener)
        except Exception:  # noqa: BLE001 - a missing metadata file only leaves this build unchecked
            unverified = True
            continue
        if _torch_ok(requirements, torch):
            return str(version)
    if unverified:
        raise RuntimeError(f"cannot read the requirements of {name} from this source")
    return None


def newest_release(
    name: str,
    *,
    python: str,
    opener,
    sources: list[tuple[str, str]],
    torch: str | None = None,
) -> Release:
    """The newest final release with a wheel for this interpreter, from the first source in order
    that has one. With ``torch``, the wheel's requirements must also accept that PyTorch (without its
    local +cu tag). Raises when no source could be read."""
    torch_version = Version(Version(torch).base_version) if torch else None
    python_version = Version(python)
    answered: str | None = None
    failure: Exception | None = None
    for kind, source in sources:
        try:
            files = index_files(name, kind, source, opener)
            version = _newest(name, files, torch=torch_version, python=python_version, opener=opener)
        except Exception as exc:  # noqa: BLE001 - the next source may still answer
            failure = exc
            continue
        if version:
            return Release(version, source)
        answered = answered or source
    if answered:
        return Release(None, answered)
    raise failure or RuntimeError("no package source is configured")
