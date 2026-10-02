"""Inspect official commit history in a separate, metadata-only Git repository."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .network import ProxyPolicy


class GitCheckUnavailable(Exception):
    pass


def _run(folder: Path, policy: ProxyPolicy, *args: str, timeout: int = 15) -> tuple[int, str]:
    env = policy.subprocess_env({key: value for key, value in os.environ.items() if not key.startswith("GIT_")})
    env.pop("SSH_ASKPASS", None)
    env.pop("SSH_ASKPASS_REQUIRE", None)
    # Public source checks must not run credential helpers, hooks, URL rewrites or prompts.
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0",
               GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1")
    command = ["git", "--no-pager", "-c", "core.hooksPath=" + os.devnull,
               "-c", "credential.helper=", "-c", "core.askPass=",
               "-c", "http.lowSpeedLimit=1", "-c", "http.lowSpeedTime=15", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
               "-c", "http.followRedirects=false", "-c", "maintenance.auto=false", "-c", "gc.auto=0",
               "--git-dir=" + str(folder), "-C", str(folder), *args]
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.DEVNULL,
                                    env=env, check=False, timeout=timeout)
            output.seek(0)
            raw = output.read(65_537)
        if len(raw) > 65_536:
            raise GitCheckUnavailable()
        return result.returncode, raw.decode("utf-8", errors="replace").strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitCheckUnavailable() from exc


def inspect_history(folder: Path, policy: ProxyPolicy, repository: str, branch: str,
                    current: str | None) -> tuple[str, list[str], int | None]:
    """Return classification and the latest commit, without touching the user's checkout."""
    try:
        if folder.is_symlink():
            raise GitCheckUnavailable()
        folder = folder.absolute()
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "HEAD").is_file():
            code, _ = _run(folder, policy, "init", "--bare", "--template=")
            if code:
                raise GitCheckUnavailable()
        code, bare = _run(folder, policy, "rev-parse", "--is-bare-repository")
        if code or bare != "true":
            raise GitCheckUnavailable()
        # tree:0 obtains complete ancestry, but no file contents or working tree.
        code, _ = _run(folder, policy, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules",
                       "--filter=tree:0", repository + ".git", f"+refs/heads/{branch}:refs/heads/checked", timeout=45)
        if code:
            raise GitCheckUnavailable()
        code, output = _run(folder, policy, "log", "-1", "--format=%H%x00%s%x00%b%x00%an%x00%aI", "refs/heads/checked")
        fields = output.split("\x00")
        if code or len(fields) != 5:
            raise GitCheckUnavailable()
        target = fields[0]
        if current == target:
            return "current", fields, 0
        if not current:
            return "unknown", fields, None
        code, _ = _run(folder, policy, "cat-file", "-e", current + "^{commit}")
        if code:
            # A local/ahead commit may not be part of main. Fetch only this known SHA.
            code, _ = _run(folder, policy, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules",
                           "--filter=tree:0", repository + ".git", current, timeout=30)
            if code:
                return "unknown", fields, None
        code, counts = _run(folder, policy, "rev-list", "--left-right", "--count", current + "..." + target)
        if code:
            raise GitCheckUnavailable()
        values = counts.split()
        if len(values) != 2 or not all(value.isdecimal() for value in values):
            raise GitCheckUnavailable()
        ahead, behind = map(int, values)
        state = "diverged" if ahead and behind else "ahead" if ahead else "available" if behind else "current"
        return state, fields, behind
    except OSError as exc:
        raise GitCheckUnavailable() from exc
