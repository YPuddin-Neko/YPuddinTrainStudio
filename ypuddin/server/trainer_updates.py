"""Read-only checks against the trainer's official source history."""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import ypuddin

from .network import ProxyPolicy

REPOSITORY = "YPuddin-Neko/YPuddinTrainStudio"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
API_URL = f"https://api.github.com/repos/{REPOSITORY}"
BRANCH = "main"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
MAX_RESPONSE = 2 * 1024 * 1024
CACHE_SECONDS = 60.0
SHA = re.compile(r"[0-9a-fA-F]{40}\Z")


class TrainerCurrentVersion(BaseModel):
    version: str
    commit: str | None = None
    branch: str | None = None
    source: Literal["git", "package", "unknown"] = "unknown"
    dirty: bool | None = None


class TrainerCommit(BaseModel):
    commit: str
    subject: str
    body: str
    author: str
    date: str
    url: str


class TrainerLatestVersion(TrainerCommit):
    download_url: str
    branch: str = BRANCH


class TrainerUpdateStatus(BaseModel):
    state: Literal["unchecked", "current", "available", "ahead", "diverged", "unknown", "error"] = "unchecked"
    checked_at: float | None = None
    error: str | None = None
    error_code: Literal["network", "rate_limit", "invalid_response", "unavailable"] | None = None
    current: TrainerCurrentVersion
    latest: TrainerLatestVersion | None = None
    commits: list[TrainerCommit] = Field(default_factory=list)
    history_kind: Literal["updates", "recent"] = "recent"
    total_commits: int | None = None
    has_more: bool = False
    compare_url: str | None = None
    repository_url: str = REPOSITORY_URL


def _sha(value: object) -> str | None:
    return value.lower() if isinstance(value, str) and SHA.fullmatch(value) else None


def _git(root: Path, *args: str) -> str | None:
    # Do not inherit a caller's GIT_DIR/GIT_WORK_TREE or execute a configured fsmonitor.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(
                ["git", "--no-pager", "-c", "core.fsmonitor=false", "-C", str(root), *args],
                stdout=output, stderr=subprocess.DEVNULL, timeout=2, env=env, check=False,
            )
            if result.returncode:
                return None
            output.seek(0)
            raw = output.read(16_385)
        if len(raw) > 16_384:
            return None
        return raw.decode("utf-8", errors="replace").strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def local_version(root: Path = SOURCE_ROOT) -> TrainerCurrentVersion:
    root = root.resolve()
    version = ypuddin.__version__
    top = _git(root, "rev-parse", "--show-toplevel")
    if top and Path(top).resolve() == root:
        commit = _sha(_git(root, "rev-parse", "HEAD"))
        if commit:
            branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
            changes = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
            return TrainerCurrentVersion(
                version=version, commit=commit, branch=None if branch == "HEAD" else branch,
                source="git", dirty=None if changes is None else bool(changes),
            )
    manifest = root / "SOURCE_MANIFEST.json"
    packaged = manifest.is_file()
    try:
        with manifest.open("rb") as stream:
            raw = stream.read(8 * 1024 * 1024 + 1)
        document = json.loads(raw) if len(raw) <= 8 * 1024 * 1024 else None
        build = document.get("build") if isinstance(document, dict) else None
        if isinstance(build, dict) and (commit := _sha(build.get("commit"))):
            branch = build.get("branch")
            return TrainerCurrentVersion(
                version=version,
                commit=commit, branch=branch if isinstance(branch, str) and len(branch) < 256 else None,
                source="package", dirty=build.get("dirty") if isinstance(build.get("dirty"), bool) else None,
            )
    except (OSError, ValueError, RecursionError):
        pass
    try:
        with (root / "ypuddin" / "_archive_revision.txt").open("rb") as stream:
            revision = stream.read(128).decode("ascii").strip()
        if commit := _sha(revision):
            return TrainerCurrentVersion(version=version, commit=commit, source="package")
    except (OSError, UnicodeError):
        pass
    return TrainerCurrentVersion(version=version, source="package" if packaged else "unknown")


class _CheckError(Exception):
    def __init__(self, code: str, status: int | None = None):
        self.code = code
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _CheckError("invalid_response")


def _read(policy: ProxyPolicy, path: str) -> object:
    request = urllib.request.Request(
        API_URL + path,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "YPuddin-Train-Studio",
                 "X-GitHub-Api-Version": "2022-11-28"},
    )
    try:
        with policy.opener(_NoRedirect()).open(request, timeout=5) as response:
            if response.geturl() != request.full_url:
                raise _CheckError("invalid_response")
            raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise _CheckError("invalid_response")
        return json.loads(raw)
    except urllib.error.HTTPError as exc:
        headers = exc.headers or {}
        limited = exc.code == 429 or (
            exc.code == 403 and (headers.get("X-RateLimit-Remaining") == "0" or headers.get("Retry-After"))
        )
        raise _CheckError("rate_limit" if limited else "unavailable", exc.code) from None
    except (OSError, urllib.error.URLError, http.client.HTTPException):
        raise _CheckError("network") from None
    except (ValueError, RecursionError):
        raise _CheckError("invalid_response") from None


def _text(value: object, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise _CheckError("invalid_response")
    return value


def _commit(document: object) -> TrainerCommit:
    if not isinstance(document, dict) or not (commit := _sha(document.get("sha"))):
        raise _CheckError("invalid_response")
    data = document.get("commit")
    if not isinstance(data, dict) or not isinstance(author := data.get("author"), dict):
        raise _CheckError("invalid_response")
    message = _text(data.get("message"), 32_768)
    subject, _, body = message.partition("\n")
    return TrainerCommit(
        commit=commit, subject=subject, body=body.strip(), author=_text(author.get("name"), 512),
        date=_text(author.get("date"), 128), url=f"{REPOSITORY_URL}/commit/{commit}",
    )


def _recent(policy: ProxyPolicy, target: str) -> tuple[list[TrainerCommit], bool]:
    document = _read(policy, f"/commits?sha={target}&per_page=21")
    if not isinstance(document, list) or not 1 <= len(document) <= 21:
        raise _CheckError("invalid_response")
    commits = [_commit(item) for item in document]
    if commits[0].commit != target or len({item.commit for item in commits}) != len(commits):
        raise _CheckError("invalid_response")
    return commits[:20], len(commits) > 20


def _comparison(policy: ProxyPolicy, base: str, latest: TrainerCommit) -> tuple[str, list[TrainerCommit], int]:
    target = latest.commit
    comparison = f"/compare/{base}...{target}"
    # GitHub includes file patches only on page 1; classification needs no diff.
    first = _read(policy, comparison + "?per_page=1&page=2")
    if not isinstance(first, dict):
        raise _CheckError("invalid_response")
    remote_state = first.get("status")
    if not isinstance(remote_state, str):
        raise _CheckError("invalid_response")
    state = {"identical": "current", "ahead": "available", "behind": "ahead", "diverged": "diverged"}.get(remote_state)
    total = first.get("total_commits")
    if state is None or type(total) is not int or not 0 <= total <= 10_000_000:
        raise _CheckError("invalid_response")
    if state != "available":
        return state, [], total
    if total == 0:
        raise _CheckError("invalid_response")
    if total == 1:
        return state, [latest], total
    last_page = (total + 19) // 20

    def page(number: int) -> list[TrainerCommit]:
        document = _read(policy, comparison + f"?per_page=20&page={number}")
        if not isinstance(document, dict) or document.get("status") != "ahead" or document.get("total_commits") != total:
            raise _CheckError("invalid_response")
        items = document.get("commits")
        expected = min(20, total - (number - 1) * 20)
        if not isinstance(items, list) or len(items) != expected:
            raise _CheckError("invalid_response")
        return [_commit(item) for item in items]

    commits = page(last_page)
    if len(commits) < 20 and last_page > 1:
        commits = page(last_page - 1) + commits
    commits = list(reversed(commits[-20:]))
    if commits[0].commit != target or len({item.commit for item in commits}) != len(commits):
        raise _CheckError("invalid_response")
    return state, commits, total


class TrainerUpdates:
    def __init__(self, context, root: Path = SOURCE_ROOT):
        self.context = context
        self.root = root.resolve()
        self.current = local_version(root)
        self._lock = threading.Lock()
        self._checking = threading.Lock()
        self._cache: dict[ProxyPolicy, tuple[float, TrainerUpdateStatus]] = {}

    def status(self) -> TrainerUpdateStatus:
        policy = ProxyPolicy.from_context(self.context)
        with self._lock:
            entry = self._cache.get(policy)
            return (entry[1] if entry else TrainerUpdateStatus(current=self.current)).model_copy(deep=True)

    def check(self) -> TrainerUpdateStatus:
        policy = ProxyPolicy.from_context(self.context)
        with self._checking:
            with self._lock:
                previous = self._cache.get(policy)
                if previous and time.monotonic() - previous[0] < CACHE_SECONDS:
                    return previous[1].model_copy(deep=True)
            try:
                latest_document = _read(policy, f"/commits?sha={BRANCH}&per_page=1")
                if not isinstance(latest_document, list) or len(latest_document) != 1:
                    raise _CheckError("invalid_response")
                latest_commit = _commit(latest_document[0])
                target = latest_commit.commit
                result = TrainerUpdateStatus(
                    state="unknown", current=self.current,
                    latest=TrainerLatestVersion(
                        **latest_commit.model_dump(), download_url=f"{REPOSITORY_URL}/archive/{target}.zip",
                    ),
                )
                if self.current.commit:
                    result.compare_url = f"{REPOSITORY_URL}/compare/{self.current.commit}...{target}"
                    if self.current.commit == target:
                        result.state = "current"
                    else:
                        try:
                            state, commits, total = _comparison(policy, self.current.commit, latest_commit)
                            result.state = state
                            if state == "available":
                                result.commits = commits
                                result.history_kind = "updates"
                                result.total_commits = total
                                result.has_more = total > 20
                        except _CheckError as exc:
                            if exc.status != 404:
                                raise
                if result.history_kind == "recent":
                    result.commits, result.has_more = _recent(policy, target)
            except _CheckError as exc:
                result = previous[1].model_copy(deep=True) if previous else TrainerUpdateStatus(current=self.current)
                result.state = "error"
                result.error_code = exc.code
                result.error = {
                    "network": "Could not connect to GitHub.",
                    "rate_limit": "GitHub request limit reached. Try again later.",
                    "invalid_response": "GitHub returned an invalid response.",
                    "unavailable": "GitHub update information is unavailable.",
                }[exc.code]
            result.checked_at = time.time()
            with self._lock:
                if policy not in self._cache and len(self._cache) >= 4:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[policy] = (time.monotonic(), result)
            return result.model_copy(deep=True)
