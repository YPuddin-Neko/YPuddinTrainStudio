"""Process-owned deployment profile paths; legacy environments remain in place."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROFILES = {
    "legacy",
    "windows-cuda",
    "linux-cuda",
    "linux-dtk",
    "macos-mps",
    "windows-cpu",
    "linux-cpu",
    "macos-cpu",
}

# A profile directory sits directly under environment/, beside the entries a legacy
# install already keeps there. A name in both places would make one shadow the other.
_LEGACY_ENTRIES = {"runtimes", "service", "cache", "installer"}
assert not PROFILES & _LEGACY_ENTRIES, "profile name collides with a legacy environment entry"


def current_profile() -> str:
    prefix = Path(sys.prefix)
    inferred = None
    if prefix.name == "venv" and prefix.parent.parent.name == "environment":
        inferred = prefix.parent.name
    elif prefix.parent.name == "runtimes" and prefix.parent.parent.parent.name == "environment":
        inferred = prefix.parent.parent.name
    profile = os.environ.get("YPUDDIN_ENV_PROFILE") or inferred or "legacy"
    if profile not in PROFILES:
        raise ValueError(f"Unsupported deployment profile: {profile}")
    if inferred and inferred != profile:
        raise ValueError(f"Interpreter belongs to {inferred}, not {profile}")
    return profile


def profile_root(data_root: Path, profile: str | None = None) -> Path:
    profile = profile or current_profile()
    if profile not in PROFILES:
        raise ValueError("Invalid environment profile")
    root = Path(data_root) / "environment"
    return root if profile == "legacy" else root / profile


def selected_key(profile: str | None = None) -> str:
    profile = profile or current_profile()
    return "torch.selected" if profile == "legacy" else f"torch.selected.{profile}"
