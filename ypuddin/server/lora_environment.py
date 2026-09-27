"""The LyCORIS release the built-in adapters match, next to what upstream publishes.

YPuddin implements its adapters itself and does not need the LyCORIS package. The files it saves
follow LyCORIS's layouts; the page names the release they were checked against beside the newest
release and the default branch, so a newer upstream shows up there.
"""

from __future__ import annotations

from pydantic import BaseModel

from .package_releases import PYPI, _read

PACKAGE = "lycoris-lora"
REPOSITORY = "https://api.github.com/repos/KohakuBlueleaf/LyCORIS"
# Every algorithm and option the trainer saves loads in this release with the same weights.
COMPATIBLE = {"version": "4.0.0", "commit": "03270a3"}


class LoraLocal(BaseModel):
    # The service's own adapters; version and commit name the LyCORIS release their files match.
    builtin: bool = True
    version: str | None = None
    commit: str | None = None


class LoraUpstream(BaseModel):
    version: str | None = None  # newest release on PyPI
    commit: str | None = None  # the commit that release was tagged on
    head: str | None = None  # newest commit on the default branch
    head_date: str | None = None
    error: str | None = None


class LoraEnvironment(BaseModel):
    checked_at: float
    local: LoraLocal
    upstream: LoraUpstream


def local_lycoris() -> dict:
    return {"builtin": True, **COMPATIBLE}


def upstream_lycoris(opener) -> dict:
    """Newest release, its commit and the default branch head; raises when upstream cannot be read."""

    def commit(ref: str) -> tuple[str, str]:
        data = _read(opener, f"{REPOSITORY}/commits/{ref}")
        when = ((data.get("commit") or {}).get("committer") or {}).get("date") or ""
        return str(data["sha"])[:7], str(when)[:10]

    version = str(_read(opener, f"{PYPI}/{PACKAGE}/json")["info"]["version"])
    released, _ = commit(f"v{version}")
    head, head_date = commit("HEAD")
    return {"version": version, "commit": released, "head": head, "head_date": head_date, "error": None}
