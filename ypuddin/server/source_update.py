"""Stage and apply a pinned official source update without touching runtime data."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .network import ProxyPolicy

REPOSITORY = "YPuddin-Neko/YPuddinTrainStudio"
ORIGIN = f"https://github.com/{REPOSITORY}.git"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 1024 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_FILES = 20_000
_ROOT_FILES = {
    ".gitattributes", ".gitignore", "CHANGELOG.md", "LICENSE", "README.md", "pyproject.toml",
    "studio-cpu.bat", "studio-cpu.sh", "studio-linux-cuda.sh", "studio-linux-dtk.sh",
    "studio-macos.command", "studio-windows-cuda.bat",
}
_SOURCE_DIRS = {".github", "docs", "frontend", "scripts", "ypuddin"}
_PRIVATE_PARTS = {".git", "node_modules", "__pycache__", "venv", ".venv", "studio_data", ".ssh", ".cache"}


class SourceUpdateError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str):
    raise SourceUpdateError(code, message)


def _emit(callback, phase: str, message: str, **values) -> None:
    if callback:
        callback({"phase": phase, "message": message, **values})


def _sha(value: str | None) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        _fail("unknown_revision", "The installed source revision could not be verified.")
    return value.lower()


def _relative(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        _fail("unsafe_path", "The update contains an unsafe file path.")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        _fail("unsafe_path", "The update contains an unsafe file path.")
    for part in path.parts:
        stem = part.split(".")[0].upper()
        if ":" in part or part.endswith((".", " ")) or stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            _fail("unsafe_path", "The update contains a file path unsupported on this platform.")
    return path.as_posix()


def _source_name(name: str) -> bool:
    path = PurePosixPath(_relative(name))
    if set(path.parts) & _PRIVATE_PARTS or path.parts[:2] == ("frontend", "dist"):
        return False
    if path.name in {".env", ".npmrc", ".pypirc", "secrets.json", "credentials.json"}:
        return False
    if path.name.startswith(".env.") and path.name not in {".env.example", ".env.sample"}:
        return False
    return name in _ROOT_FILES or (len(path.parts) > 1 and path.parts[0] in _SOURCE_DIRS)


def _path(root: Path, name: str) -> Path:
    name = _relative(name)
    if root.is_symlink():
        _fail("unsafe_path", "A source or update directory is a symbolic link.")
    path = root
    parts = PurePosixPath(name).parts
    for index, part in enumerate(parts):
        path = path / part
        if path.is_symlink():
            _fail("unsafe_path", "A source or update path is a symbolic link.")
        if index < len(parts) - 1 and path.exists() and not path.is_dir():
            _fail("path_conflict", f"A source directory conflicts with an existing file: {name}")
    if path.exists() and not path.is_file():
        _fail("path_conflict", f"A source file conflicts with a directory: {name}")
    return path


def _digest(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        _fail("file_too_large", "An update file exceeds the size limit.")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _hashes(root: Path, names) -> dict[str, str | None]:
    result = {}
    for name in sorted(names):
        path = _path(root, name)
        result[name] = _digest(path) if path.is_file() else None
    return result


def _json(path: Path) -> dict:
    try:
        if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError
        data = json.loads(path.read_bytes())
        if not isinstance(data, dict):
            raise ValueError
        return data
    except (OSError, ValueError, RecursionError):
        _fail("invalid_plan", "The update record could not be read.")


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink() or path.is_symlink():
        _fail("unsafe_path", "An update record is a symbolic link.")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _git(root: Path, *args: str, policy: ProxyPolicy | None = None, timeout: float = 20) -> str:
    env = (policy or ProxyPolicy()).subprocess_env()
    env = {key: value for key, value in env.items() if not key.startswith("GIT_")}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(
                ["git", "--no-pager", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}",
                 "-C", str(root), *args], stdout=output, stderr=subprocess.DEVNULL,
                env=env, timeout=timeout, check=False,
            )
            output.seek(0)
            raw = output.read(4 * 1024 * 1024 + 1)
        if result.returncode or len(raw) > 4 * 1024 * 1024:
            _fail("git_failed", "Git could not complete the source update operation.")
        return raw.decode("utf-8", errors="surrogateescape").strip()
    except (OSError, subprocess.TimeoutExpired):
        _fail("git_failed", "Git could not complete the source update operation.")


def _git_clean(root: Path, commit: str) -> None:
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        _fail("wrong_repository", "The source directory is not the Git repository root.")
    if _git(root, "rev-parse", "HEAD") != commit:
        _fail("source_changed", "The installed source revision has changed. Check for updates again.")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        _fail("local_changes", "The source directory contains local changes. Save them before updating.")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _fail("download_failed", "The source archive redirected to an unexpected address.")


def _download_archive(commit: str, destination: Path, policy: ProxyPolicy, callback) -> None:
    url = f"https://codeload.github.com/{REPOSITORY}/zip/{commit}"
    request = urllib.request.Request(url, headers={"User-Agent": "YPuddin-Train-Studio"})
    started = time.monotonic()
    received = 0
    _emit(callback, "downloading", "Downloading source archive.", archive=destination.name, completed=0)
    try:
        with policy.opener(_NoRedirect()).open(request, timeout=30) as response, destination.open("xb") as stream:
            if response.geturl() != url:
                _fail("download_failed", "The source archive address could not be verified.")
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > MAX_ARCHIVE_BYTES or time.monotonic() - started > 600:
                    _fail("download_limit", "The source archive exceeds the download limit.")
                stream.write(block)
                _emit(callback, "downloading", "Downloading source archive.", archive=destination.name, completed=received)
        _emit(callback, "downloaded", "Source archive downloaded.", archive=destination.name, completed=received)
    except SourceUpdateError:
        raise
    except (OSError, urllib.error.URLError, http.client.HTTPException):
        _fail("download_failed", "The official source archive could not be downloaded.")


def _extract(archive: Path, destination: Path, commit: str) -> dict[str, str]:
    prefix = f"YPuddinTrainStudio-{commit}/"
    files = {}
    seen = set()
    total = 0
    try:
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            if len(entries) > MAX_FILES:
                _fail("archive_limit", "The source archive contains too many entries.")
            for entry in entries:
                if not entry.filename.startswith(prefix):
                    _fail("archive_identity", "The source archive does not match the requested revision.")
                name = entry.filename[len(prefix):].rstrip("/")
                if not name:
                    continue
                _relative(name)
                mode = entry.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1:
                    _fail("unsafe_archive", "The source archive contains an unsupported entry.")
                key = unicodedata.normalize("NFC", name).casefold()
                if key in seen:
                    _fail("unsafe_archive", "The source archive contains duplicate file paths.")
                seen.add(key)
                if entry.is_dir():
                    continue
                if not _source_name(name):
                    _fail("unsupported_source", f"The archive contains a file outside the source directories: {name}")
                total += entry.file_size
                if entry.file_size > MAX_FILE_BYTES or total > MAX_EXPANDED_BYTES:
                    _fail("archive_limit", "The expanded source archive exceeds the size limit.")
                path = _path(destination, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as incoming, path.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                path.chmod(0o755 if mode & 0o111 else 0o644)
                files[name] = _digest(path)
    except SourceUpdateError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError):
        _fail("invalid_archive", "The source archive could not be verified.")
    required = {"pyproject.toml", "ypuddin/__init__.py", "ypuddin/cli.py", "scripts/bootstrap.py", "frontend/package.json"}
    if not required <= files.keys():
        _fail("invalid_archive", "The source archive is missing required trainer files.")
    marker = destination / "ypuddin/_archive_revision.txt"
    if marker.exists() and marker.read_bytes().strip() not in (commit.encode("ascii"), b"$Format:%H$"):
        _fail("archive_identity", "The source archive revision marker does not match.")
    return files


def _frontend(root: Path, *, required: bool) -> dict[str, str]:
    manifest = _path(root, "frontend/dist/.source-manifest.json")
    dist = root / "frontend/dist"
    if not manifest.exists():
        if required or dist.exists() and any(dist.iterdir()):
            _fail("frontend_unverified", "The frontend build is missing its source fingerprint.")
        return {}
    document = _json(manifest)
    if document.get("version") != 1 or not isinstance(document.get("inputs"), dict) or not isinstance(document.get("outputs"), dict):
        _fail("frontend_unverified", "The frontend build fingerprint is invalid.")
    if "index.html" not in document["outputs"]:
        _fail("frontend_unverified", "The frontend build is incomplete.")
    for directory, hashes in ((root / "frontend", document["inputs"]), (dist, document["outputs"])):
        for name, digest in hashes.items():
            path = _path(directory, name)
            if not isinstance(digest, str) or not path.is_file() or _digest(path) != digest:
                _fail("frontend_unverified", "The frontend build does not match its source fingerprint.")
    return {**{f"frontend/dist/{name}": digest for name, digest in document["outputs"].items()},
            "frontend/dist/.source-manifest.json": _digest(manifest)}


def _plan(path: Path) -> tuple[Path, dict]:
    path = Path(path).absolute()
    if path.name != "plan.json" or path.is_symlink() or path.parent.is_symlink():
        _fail("invalid_plan", "The update plan path is invalid.")
    plan = _json(path)
    if plan.get("format") != 1 or plan.get("source_kind") not in ("git", "package"):
        _fail("invalid_plan", "The update plan format is invalid.")
    _sha(plan.get("current_commit"))
    _sha(plan.get("target_commit"))
    if not isinstance(plan.get("root"), str) or Path(plan["root"]).resolve() != Path(plan["root"]):
        _fail("invalid_plan", "The update source path is invalid.")
    for field in ("before", "source_before", "source_after", "frontend_before", "after", "frontend_after"):
        values = plan.get(field, {})
        if not isinstance(values, dict) or len(values) > MAX_FILES:
            _fail("invalid_plan", "The update file inventory is invalid.")
        for name, digest in values.items():
            if not (_source_name(name) or name.startswith("frontend/dist/") or name == "SOURCE_MANIFEST.json"):
                _fail("invalid_plan", "The update plan includes a file outside the source directories.")
            if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                _fail("invalid_plan", "The update file inventory contains an invalid checksum.")
    return path, plan


def prepare_update(root, work_dir, current_commit, target_commit, source_kind, policy, progress_callback=None) -> dict:
    root = Path(root).resolve()
    work = Path(work_dir).absolute()
    current_commit, target_commit = _sha(current_commit), _sha(target_commit)
    if source_kind not in ("git", "package") or current_commit == target_commit:
        _fail("invalid_update", "The requested source update is not applicable.")
    if work.is_symlink() or work == root or work in root.parents:
        _fail("unsafe_path", "The update staging directory is invalid.")
    work = work.resolve()
    if root in work.parents and work.relative_to(root).parts[0] in _SOURCE_DIRS:
        _fail("unsafe_path", "The update staging directory overlaps the source files.")
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        _fail("staging_in_use", "The update staging directory is already in use.")
    _emit(progress_callback, "checking", "Checking installed source files.")
    if source_kind == "git":
        _git_clean(root, current_commit)
    elif (root / ".git").exists():
        _fail("wrong_repository", "A Git checkout cannot be updated as a source archive.")
    baseline, staged = work / "baseline", work / "staged"
    baseline.mkdir()
    staged.mkdir()
    _download_archive(current_commit, work / "baseline.zip", policy, progress_callback)
    _emit(progress_callback, "extracting", "Extracting source archive.", archive="baseline.zip")
    before_source = _extract(work / "baseline.zip", baseline, current_commit)
    _download_archive(target_commit, work / "target.zip", policy, progress_callback)
    _emit(progress_callback, "extracting", "Extracting source archive.", archive="target.zip")
    after_source = _extract(work / "target.zip", staged, target_commit)
    _emit(progress_callback, "verifying", "Verifying source files.")
    before = _hashes(root, before_source.keys() | after_source.keys())
    for name, digest in before_source.items():
        if source_kind == "package" and before[name] != digest:
            marker = name == "ypuddin/_archive_revision.txt" and before[name] is not None
            if marker and _path(root, name).read_bytes().strip() in (current_commit.encode("ascii"), b"$Format:%H$"):
                continue
            _fail("local_changes", f"A source file differs from the installed revision: {name}")
    for name in after_source.keys() - before_source.keys():
        if before[name] is not None:
            _fail("local_changes", f"A new source file would overwrite an existing file: {name}")
    old_frontend = _frontend(root, required=False)
    before.update(old_frontend)
    if source_kind == "package":
        before.update(_hashes(root, ["SOURCE_MANIFEST.json"]))
    else:
        _emit(progress_callback, "fetching", "Fetching the selected source revision.")
        _git(root, "fetch", "--no-tags", "--no-recurse-submodules", ORIGIN, target_commit, policy=policy, timeout=180)
        if _git(root, "rev-parse", "FETCH_HEAD") != target_commit:
            _fail("wrong_revision", "Git fetched a different source revision.")
        _git(root, "merge-base", "--is-ancestor", current_commit, target_commit)
    plan = {
        "format": 1, "root": str(root), "source_kind": source_kind,
        "current_commit": current_commit, "target_commit": target_commit,
        "plan_path": str(work / "plan.json"), "staged_root": str(staged), "phase": "prepared",
        "before": before, "source_before": before_source, "source_after": after_source,
        "frontend_before": old_frontend, "created_at": time.time(),
    }
    _write_json(work / "plan.json", plan)
    _emit(progress_callback, "prepared", "Source files are ready for the frontend build.")
    return plan


def seal_update(plan_path, progress_callback=None) -> dict:
    path, plan = _plan(plan_path)
    if plan["phase"] != "prepared":
        _fail("invalid_phase", "The update is not waiting for a frontend build.")
    root, staged = Path(plan["root"]), path.parent / "staged"
    if _hashes(staged, plan["source_after"]) != plan["source_after"]:
        _fail("staging_changed", "Source files changed while preparing the update.")
    frontend = _frontend(staged, required=True)
    after = {**plan["source_after"], **frontend}
    for name in frontend.keys() - plan["frontend_before"].keys():
        if _path(root, name).exists():
            _fail("local_changes", f"A frontend file would overwrite an existing file: {name}")
        plan["before"][name] = None
    if plan["source_kind"] == "package":
        version_text = (staged / "ypuddin/__init__.py").read_text(encoding="utf-8")
        version = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', version_text)
        manifest = {
            "build": {"version": version.group(1) if version else "unknown", "commit": plan["target_commit"],
                      "branch": "main", "dirty": False, "built_at": datetime.now(timezone.utc).isoformat()},
            "files": {name: {"sha256": digest, "bytes": _path(staged, name).stat().st_size} for name, digest in sorted(after.items())},
        }
        _write_json(staged / "SOURCE_MANIFEST.json", manifest)
        after["SOURCE_MANIFEST.json"] = _digest(staged / "SOURCE_MANIFEST.json")
    plan.update(after=after, frontend_after=frontend, phase="ready")
    _write_json(path, plan)
    _emit(progress_callback, "ready", "The source update and frontend build are ready.")
    return plan


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ypuddin-update-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, stream, 1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        shutil.copymode(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _git_frontend_fingerprint(path: Path, plan: dict) -> None:
    # Git may check out text with CRLF; the staged build used archive line endings.
    root, staged = Path(plan["root"]), path.parent / "staged"
    name = "frontend/dist/.source-manifest.json"
    manifest_path = _path(staged, name)
    manifest = _json(manifest_path)
    changed = False
    for relative in manifest["inputs"]:
        source = _path(staged / "frontend", relative)
        installed = _path(root / "frontend", relative)
        if not installed.is_file() or source.read_bytes().replace(b"\r\n", b"\n") != installed.read_bytes().replace(b"\r\n", b"\n"):
            _fail("checkout_changed", "Git changed a frontend build input beyond line endings.")
        digest = _digest(installed)
        changed = changed or manifest["inputs"][relative] != digest
        manifest["inputs"][relative] = digest
    if not changed:
        return
    _write_json(manifest_path, manifest)
    digest = _digest(manifest_path)
    plan["after"][name] = plan["frontend_after"][name] = digest
    _write_json(path, plan)


def _require_space(root: Path, work: Path, plan: dict, changed: set[str]) -> None:
    backup_bytes = 0
    growth_bytes = 0
    temporary_bytes = 0
    staged = work / "staged"
    for name in changed:
        old = _path(root, name)
        old_size = old.stat().st_size if old.exists() else 0
        new = _path(staged, name)
        new_size = new.stat().st_size if name in plan["after"] else 0
        if new_size and plan["source_kind"] == "git" and name in plan["source_after"]:
            # A Git checkout can expand LF to CRLF on Windows.
            with new.open("rb") as stream:
                new_size += sum(block.count(b"\n") for block in iter(lambda: stream.read(1024 * 1024), b""))
        backup_bytes += old_size
        growth_bytes += max(0, new_size - old_size)
        temporary_bytes = max(temporary_bytes, old_size, new_size)
    live_bytes = growth_bytes + temporary_bytes
    try:
        if root.stat().st_dev == work.stat().st_dev:
            enough = shutil.disk_usage(root).free >= backup_bytes + live_bytes
        else:
            enough = shutil.disk_usage(work).free >= backup_bytes and shutil.disk_usage(root).free >= live_bytes
    except OSError:
        _fail("disk_check_failed", "Free disk space could not be checked before applying the update.")
    if not enough:
        _fail("insufficient_space", "Not enough free disk space for the source update and its backup.")


def _clear_bytecode(root: Path, changed) -> None:
    for name in changed:
        source = PurePosixPath(name)
        if source.suffix != ".py":
            continue
        # Equal-sized edits within one timestamp tick can otherwise reuse old code.
        cache = _path(root, str(source.parent / "__pycache__" / ".cache-check")).parent
        if not cache.exists():
            continue
        for entry in cache.iterdir():
            if entry.name.startswith(source.stem + ".") and entry.name.endswith(".pyc"):
                _path(root, entry.relative_to(root).as_posix()).unlink()


def apply_update(plan_path, python=None, progress_callback=None) -> None:
    path, plan = _plan(plan_path)
    if plan["phase"] != "ready":
        _fail("invalid_phase", "The update is not ready to apply.")
    root, staged, backup = Path(plan["root"]), path.parent / "staged", path.parent / "backup"
    if _hashes(root, plan["before"]) != plan["before"]:
        _fail("source_changed", "Installed files changed after the update was prepared.")
    if _hashes(staged, plan["after"]) != plan["after"]:
        _fail("staging_changed", "Prepared update files changed before applying.")
    if plan["source_kind"] == "git":
        _git_clean(root, plan["current_commit"])
    changed = {name for name in plan["before"].keys() | plan["after"].keys()
               if plan["before"].get(name) != plan["after"].get(name)}
    _require_space(root, path.parent, plan, changed)
    backup.mkdir(exist_ok=False)
    for name in sorted(changed):
        if plan["before"].get(name) is not None:
            _copy(_path(root, name), _path(backup, name))
    plan.update(phase="applying", changed=sorted(changed))
    _write_json(path, plan)
    try:
        _emit(progress_callback, "applying", "Applying the prepared source update.")
        if plan["source_kind"] == "git":
            _git(root, "merge", "--ff-only", "--no-edit", plan["target_commit"], timeout=60)
            _git_frontend_fingerprint(path, plan)
        for name in sorted(changed):
            if plan["source_kind"] == "git" and not name.startswith("frontend/dist/"):
                continue
            target = _path(root, name)
            if name in plan["after"]:
                _copy(_path(staged, name), target)
            else:
                target.unlink(missing_ok=True)
        _clear_bytecode(root, changed)
        plan["phase"] = "applied"
        _write_json(path, plan)
        _emit(progress_callback, "applied", "The prepared source update has been applied.")
    except BaseException as error:
        try:
            rollback_update(path, python=python, progress_callback=progress_callback)
        except BaseException as rollback_error:
            raise SourceUpdateError("rollback_failed", "The update failed and could not be restored automatically. The backup has been kept.") from rollback_error
        raise SourceUpdateError("apply_failed", "The update could not be applied. The previous source files were restored.") from error


def rollback_update(plan_path, python=None, progress_callback=None) -> None:
    path, plan = _plan(plan_path)
    if plan["phase"] == "rolled_back":
        return
    if plan["phase"] not in ("applying", "applied"):
        _fail("invalid_phase", "This update has not changed the installed source files.")
    root, backup = Path(plan["root"]), path.parent / "backup"
    changed = plan["changed"]
    expected_backups = {name: plan["before"][name] for name in changed if plan["before"].get(name) is not None}
    if _hashes(backup, expected_backups) != expected_backups:
        _fail("backup_changed", "The source update backup could not be verified.")
    for name in changed:
        if plan["source_kind"] == "git" and not name.startswith("frontend/dist/"):
            continue
        live = _path(root, name)
        digest = _digest(live) if live.is_file() else None
        if digest not in (plan["before"].get(name), plan["after"].get(name)):
            _fail("rollback_conflict", f"A file changed after the update; its backup was preserved: {name}")
    _emit(progress_callback, "rolling_back", "Restoring the previous source files.")
    if plan["source_kind"] == "git":
        head = _git(root, "rev-parse", "HEAD")
        if head == plan["target_commit"]:
            _git_clean(root, head)
            _git(root, "read-tree", "-u", "-m", plan["target_commit"], plan["current_commit"])
            _git(root, "update-ref", "-m", "Restore previous trainer source", "HEAD", plan["current_commit"], plan["target_commit"])
        elif head != plan["current_commit"]:
            _fail("rollback_conflict", "The Git revision changed after the update; the backup was preserved.")
    for name in sorted(changed):
        if plan["source_kind"] == "git" and not name.startswith("frontend/dist/"):
            continue
        target = _path(root, name)
        if plan["before"].get(name) is None:
            target.unlink(missing_ok=True)
        else:
            _copy(_path(backup, name), target)
    _clear_bytecode(root, changed)
    plan["phase"] = "rolled_back"
    _write_json(path, plan)
    _emit(progress_callback, "rolled_back", "The previous source files were restored.")
