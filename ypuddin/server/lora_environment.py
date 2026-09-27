"""LyCORIS as the service has it and as upstream publishes it, for the runtime settings page.

YPuddin implements its adapters itself; LyCORIS is the reference their formats follow, so the page
shows which release and commit the service has next to the newest ones.
"""

from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import BaseModel

from .package_releases import PYPI, _read

PACKAGE = "lycoris-lora"
REPOSITORY = "https://api.github.com/repos/KohakuBlueleaf/LyCORIS"


class LoraLocal(BaseModel):
    version: str | None = None
    # Short commit of the installed source, when the install records one.
    commit: str | None = None


class LoraUpstream(BaseModel):
    version: str | None = None  # newest release on PyPI
    commit: str | None = None  # the commit that release was tagged on
    head: str | None = None  # newest commit on the default branch
    head_date: str | None = None
    # Commit of the installed release, looked up when the install does not record it.
    local_commit: str | None = None
    error: str | None = None


class LoraEnvironment(BaseModel):
    checked_at: float
    local: LoraLocal
    upstream: LoraUpstream


def local_lycoris() -> dict:
    try:
        distribution = metadata.distribution(PACKAGE)
    except metadata.PackageNotFoundError:
        return {"version": None, "commit": None}
    try:
        origin = json.loads(distribution.read_text("direct_url.json") or "null") or {}
    except (json.JSONDecodeError, OSError):
        origin = {}
    commit = (origin.get("vcs_info") or {}).get("commit_id")
    url = str(origin.get("url") or "")
    if not commit and "dir_info" in origin and url.startswith("file://"):
        # Installed from a local checkout: that checkout knows its commit.
        commit = _checkout_commit(Path(unquote(urlparse(url).path)))
    return {"version": distribution.version, "commit": commit[:7] if commit else None}


def _checkout_commit(path: Path) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = done.stdout.strip()
    return value if done.returncode == 0 and len(value) >= 7 else None


def upstream_lycoris(opener, local: dict) -> dict:
    """Newest release, its commit and the default branch head; raises when upstream cannot be read."""

    def commit(ref: str) -> tuple[str, str]:
        data = _read(opener, f"{REPOSITORY}/commits/{ref}")
        when = ((data.get("commit") or {}).get("committer") or {}).get("date") or ""
        return str(data["sha"])[:7], str(when)[:10]

    version = str(_read(opener, f"{PYPI}/{PACKAGE}/json")["info"]["version"])
    released, _ = commit(f"v{version}")
    head, head_date = commit("HEAD")
    local_commit = None
    if local.get("version") and not local.get("commit"):
        local_commit = released if local["version"] == version else commit(f"v{local['version']}")[0]
    return {
        "version": version,
        "commit": released,
        "head": head,
        "head_date": head_date,
        "local_commit": local_commit,
        "error": None,
    }
