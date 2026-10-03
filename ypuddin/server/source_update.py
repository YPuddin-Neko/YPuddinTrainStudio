"""Stage and apply a pinned official source update without touching runtime data.

Applying replaces files in a fixed order with the version records last, so an installed tree
reports the new version only once every file is new. A plan records the SHA-256 of every file
before and after the update; an apply that stopped part-way is finished from the staged files
or undone from the backup. The module uses only the standard library, so the launcher can run
the copy saved with an update after the installed files changed.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network import ProxyPolicy

REPOSITORY = "YPuddin-Neko/YPuddinTrainStudio"
ORIGIN = f"https://github.com/{REPOSITORY}.git"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 1024 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_FILES = 20_000
GIT_MERGE_TIMEOUT = 600
_ROOT_FILES = {
    ".gitattributes", ".gitignore", "CHANGELOG.md", "LICENSE", "README.md", "pyproject.toml",
    "studio-cpu.bat", "studio-cpu.sh", "studio-linux-cuda.sh", "studio-linux-dtk.sh",
    "studio-macos.command", "studio-windows-cuda.bat",
}
_SOURCE_DIRS = {".github", "docs", "frontend", "scripts", "ypuddin"}
_PRIVATE_PARTS = {".git", "node_modules", "__pycache__", "venv", ".venv", "studio_data", ".ssh", ".cache"}
# Version records: written last when applying and restored first when rolling back.
_VERSION_FILES = ("ypuddin/_archive_revision.txt", "frontend/dist/.source-manifest.json", "SOURCE_MANIFEST.json")
# Fetching into the user's checkout never prompts, runs credential helpers or follows URL
# rewrites to another transport; local commands keep the user's Git settings.
_NETWORK_OPTIONS = (
    "-c", "credential.helper=", "-c", "core.askPass=", "-c", "protocol.allow=never",
    "-c", "protocol.https.allow=always", "-c", "http.lowSpeedLimit=1", "-c", "http.lowSpeedTime=30",
)
_GIT_STEPS = {"fetch": "拉取新版本", "merge": "切换到新版本", "read-tree": "恢复原版文件", "update-ref": "恢复原版提交"}


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


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        _fail("cancelled", "更新已取消。")


def _sha(value: str | None) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        _fail("unknown_revision", "无法确认已安装源码的版本。")
    return value.lower()


def _relative(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        _fail("unsafe_path", "更新包含不安全的文件路径。")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        _fail("unsafe_path", "更新包含不安全的文件路径。")
    for part in path.parts:
        stem = part.split(".")[0].upper()
        if ":" in part or part.endswith((".", " ")) or stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            _fail("unsafe_path", "更新包含当前系统不支持的文件路径。")
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
        _fail("unsafe_path", "源码或更新目录是符号链接。")
    path = root
    parts = PurePosixPath(name).parts
    for index, part in enumerate(parts):
        path = path / part
        if path.is_symlink():
            _fail("unsafe_path", "源码或更新路径是符号链接。")
        if index < len(parts) - 1 and path.exists() and not path.is_dir():
            _fail("path_conflict", f"更新需要的目录与已有文件同名：{name}")
    if path.exists() and not path.is_file():
        _fail("path_conflict", f"更新中的文件与已有目录同名：{name}")
    return path


def _digest(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        _fail("file_too_large", "更新中的文件超过大小限制。")
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


def _live(root: Path, name: str) -> str | None:
    path = _path(root, name)
    return _digest(path) if path.is_file() else None


def _json(path: Path) -> dict:
    try:
        if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError
        data = json.loads(path.read_bytes())
        if not isinstance(data, dict):
            raise ValueError
        return data
    except (OSError, ValueError, RecursionError):
        _fail("invalid_plan", "无法读取更新记录。")


def _replace(source: Path, target: Path) -> None:
    # A virus scanner or an open reader on Windows can hold the target for a moment.
    for attempt in range(30):
        try:
            source.replace(target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 29:
                raise
            time.sleep(0.1)


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink() or path.is_symlink():
        _fail("unsafe_path", "更新记录是符号链接。")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(temporary, path)


def _kill_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            process.kill()
        except OSError:
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _git_env(policy: ProxyPolicy | None, network: bool) -> dict[str, str]:
    env = policy.subprocess_env() if policy is not None else dict(os.environ)
    env = {key: value for key, value in env.items() if not key.startswith("GIT_")}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    if network:
        for key in ("SSH_ASKPASS", "SSH_ASKPASS_REQUIRE"):
            env.pop(key, None)
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_SSH_COMMAND="ssh -o BatchMode=yes",
                   GCM_INTERACTIVE="never")
    return env


def _git_failure(args: tuple[str, ...], errors: bytes, policy: ProxyPolicy | None) -> str:
    step = _GIT_STEPS.get(args[0], "完成源码更新操作")
    lines = [line.strip() for line in errors.decode("utf-8", errors="replace").splitlines() if line.strip()]
    detail = "\n".join(lines[-6:])[-1500:]
    if detail and policy is not None:
        detail = policy.redact(detail)
    return f"Git 未能{step}。" + (f"\n{detail}" if detail else "")


def _git(root: Path, *args: str, policy: ProxyPolicy | None = None, timeout: float = 20, cancel=None,
         network: bool = False, started=None, strip: bool = True) -> str:
    """Run Git in ``root``. Its whole process tree stops on timeout or cancellation; ``started``
    receives the process id."""
    command = ["git", "--no-pager", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}",
               *(_NETWORK_OPTIONS if network else ()), "-C", str(root), *args]
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    try:
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                                       env=_git_env(policy, network), **group)
            if started is not None:
                started(process.pid)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    if (cancel is not None and cancel.is_set()) or time.monotonic() > deadline:
                        _kill_tree(process)
                        _check(cancel)
                        _fail("git_failed", f"Git 在 {int(timeout)} 秒内没有完成：git {args[0]}。")
            output.seek(0)
            raw = output.read(4 * 1024 * 1024 + 1)
            errors.seek(0)
            detail = errors.read(64 * 1024)
        if process.returncode or len(raw) > 4 * 1024 * 1024:
            _fail("git_failed", _git_failure(args, detail, policy))
        text = raw.decode("utf-8", errors="surrogateescape")
        return text.strip() if strip else text
    except OSError as error:
        _fail("git_failed", f"无法运行 Git：{error}")


def _git_clean(root: Path, commit: str) -> None:
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        _fail("wrong_repository", "源码目录不是 Git 仓库的根目录。")
    if _git(root, "rev-parse", "HEAD") != commit:
        _fail("source_changed", "已安装的源码版本已变化，请重新检查更新。")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        _fail("local_changes", "源码目录有本地改动，请先保存或撤销后再更新。")


def _git_changes(root: Path) -> set[str]:
    """Changed and untracked files, as repository-relative names."""
    entries = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", strip=False).split("\0")
    names, index = set(), 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        names.add(entry[3:])
        if entry[0] in "RC":  # the original name of a rename follows
            names.add(entries[index])
            index += 1
    return names


def _running(pid: int, started_at: float) -> bool:
    """Whether a Git process an update started is still running."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            return process.status() != psutil.STATUS_ZOMBIE and process.create_time() >= started_at - 5
        except psutil.Error:
            return False
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_index_lock(root: Path, plan: dict, *, required: bool = True, wait: float = 15) -> None:
    """Wait for Git before restoring files or changing the index.

    index.lock has no owner field. Its timestamp cannot distinguish a stopped update's lock
    from one created later by another Git process, so recovery never removes it.
    """
    lock = root / ".git" / "index.lock"
    record = plan.get("git_command")
    started_at = record.get("started_at") if isinstance(record, dict) else None
    pid = record.get("pid") if isinstance(record, dict) else None
    deadline = time.monotonic() + wait
    while True:
        try:
            lock.stat()
        except FileNotFoundError:
            return
        running = isinstance(started_at, (int, float)) and isinstance(pid, int) and _running(pid, started_at)
        if not running and not required:
            return
        if time.monotonic() > deadline:
            if running:
                _fail("git_busy", "上次更新启动的 Git 仍在运行，请稍后重新启动训练器。")
            _fail("git_locked", "Git 索引锁仍存在（.git/index.lock），无法确认归属，已保留该文件。请确认没有 Git 操作运行后处理遗留锁，再重新启动训练器。")
        time.sleep(0.2)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _fail("download_failed", "源码包下载地址发生了意外跳转。")


def _close_on_cancel(response, cancel, done: threading.Event) -> None:
    while not done.wait(0.2):
        if cancel.is_set():
            try:
                response.fp.raw._sock.shutdown(2)  # wakes a read that waits for the network
            except (AttributeError, OSError):
                pass
            return


def _download_archive(commit: str, destination: Path, policy: ProxyPolicy, callback, cancel=None) -> None:
    url = f"https://codeload.github.com/{REPOSITORY}/zip/{commit}"
    request = urllib.request.Request(url, headers={"User-Agent": "YPuddin-Train-Studio"})
    started = time.monotonic()
    received = 0
    _emit(callback, "downloading", "正在下载源码包", archive=destination.name, completed=0)
    done = threading.Event()
    try:
        with policy.opener(_NoRedirect()).open(request, timeout=15) as response, destination.open("xb") as stream:
            if response.geturl() != url:
                _fail("download_failed", "无法确认源码包的下载地址。")
            if cancel is not None:
                threading.Thread(target=_close_on_cancel, args=(response, cancel, done), daemon=True).start()
            while block := response.read(1024 * 1024):
                _check(cancel)
                received += len(block)
                if received > MAX_ARCHIVE_BYTES or time.monotonic() - started > 600:
                    _fail("download_limit", "源码包超过下载限制。")
                stream.write(block)
                _emit(callback, "downloading", "正在下载源码包", archive=destination.name, completed=received)
            _check(cancel)
        _emit(callback, "downloaded", "源码包下载完成", archive=destination.name, completed=received)
    except SourceUpdateError:
        raise
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
        _check(cancel)
        reason = policy.redact(str(getattr(error, "reason", error)))[-300:]
        _fail("download_failed", "无法下载官方源码包" + (f"：{reason}" if reason else "。"))
    finally:
        done.set()


def _extract(archive: Path, destination: Path, commit: str, cancel=None) -> dict[str, str]:
    prefix = f"YPuddinTrainStudio-{commit}/"
    files = {}
    seen = set()
    total = 0
    try:
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            if len(entries) > MAX_FILES:
                _fail("archive_limit", "源码包中的文件过多。")
            for entry in entries:
                _check(cancel)
                if not entry.filename.startswith(prefix):
                    _fail("archive_identity", "源码包与要求的版本不符。")
                name = entry.filename[len(prefix):].rstrip("/")
                if not name:
                    continue
                _relative(name)
                mode = entry.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1:
                    _fail("unsafe_archive", "源码包含不支持的条目。")
                key = unicodedata.normalize("NFC", name).casefold()
                if key in seen:
                    _fail("unsafe_archive", "源码包含重复的文件路径。")
                seen.add(key)
                if entry.is_dir():
                    continue
                if not _source_name(name):
                    _fail("unsupported_source", f"源码包含源码目录以外的文件：{name}")
                total += entry.file_size
                if entry.file_size > MAX_FILE_BYTES or total > MAX_EXPANDED_BYTES:
                    _fail("archive_limit", "解压后的源码超过大小限制。")
                path = _path(destination, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as incoming, path.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                path.chmod(0o755 if mode & 0o111 else 0o644)
                files[name] = _digest(path)
    except SourceUpdateError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError):
        _fail("invalid_archive", "无法校验源码包。")
    required = {"pyproject.toml", "ypuddin/__init__.py", "ypuddin/cli.py", "scripts/bootstrap.py", "frontend/package.json"}
    if not required <= files.keys():
        _fail("invalid_archive", "源码包缺少训练器的必要文件。")
    marker = destination / "ypuddin/_archive_revision.txt"
    if marker.exists() and marker.read_bytes().strip() not in (commit.encode("ascii"), b"$Format:%H$"):
        _fail("archive_identity", "源码包的版本标记与要求的版本不符。")
    return files


def _frontend(root: Path, *, required: bool) -> dict[str, str]:
    manifest = _path(root, "frontend/dist/.source-manifest.json")
    dist = root / "frontend/dist"
    if not manifest.exists():
        if required or dist.exists() and any(dist.iterdir()):
            _fail("frontend_unverified", "前端构建缺少源码指纹。")
        return {}
    document = _json(manifest)
    if document.get("version") != 1 or not isinstance(document.get("inputs"), dict) or not isinstance(document.get("outputs"), dict):
        _fail("frontend_unverified", "前端构建指纹无效。")
    if "index.html" not in document["outputs"]:
        _fail("frontend_unverified", "前端构建不完整。")
    for directory, hashes in ((root / "frontend", document["inputs"]), (dist, document["outputs"])):
        for name, digest in hashes.items():
            path = _path(directory, name)
            if not isinstance(digest, str) or not path.is_file() or _digest(path) != digest:
                _fail("frontend_unverified", "前端构建与源码指纹不符。")
    return {**{f"frontend/dist/{name}": digest for name, digest in document["outputs"].items()},
            "frontend/dist/.source-manifest.json": _digest(manifest)}


def _plan(path: Path) -> tuple[Path, dict]:
    path = Path(path).absolute()
    if path.name != "plan.json" or path.is_symlink() or path.parent.is_symlink():
        _fail("invalid_plan", "更新计划的路径无效。")
    plan = _json(path)
    if plan.get("format") != 1 or plan.get("source_kind") not in ("git", "package"):
        _fail("invalid_plan", "更新计划的格式无效。")
    _sha(plan.get("current_commit"))
    _sha(plan.get("target_commit"))
    if not isinstance(plan.get("root"), str) or Path(plan["root"]).resolve() != Path(plan["root"]):
        _fail("invalid_plan", "更新的源码路径无效。")
    for field in ("before", "source_before", "source_after", "frontend_before", "after", "frontend_after"):
        values = plan.get(field, {})
        if not isinstance(values, dict) or len(values) > MAX_FILES:
            _fail("invalid_plan", "更新的文件清单无效。")
        for name, digest in values.items():
            if not (_source_name(name) or name.startswith("frontend/dist/") or name == "SOURCE_MANIFEST.json"):
                _fail("invalid_plan", "更新计划包含源码目录以外的文件。")
            if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                _fail("invalid_plan", "更新的文件清单含有无效的校验值。")
    return path, plan


def prepare_update(root, work_dir, current_commit, target_commit, source_kind, policy, progress_callback=None,
                   cancel=None) -> dict:
    root = Path(root).resolve()
    work = Path(work_dir).absolute()
    current_commit, target_commit = _sha(current_commit), _sha(target_commit)
    if source_kind not in ("git", "package") or current_commit == target_commit:
        _fail("invalid_update", "无法应用此源码更新。")
    if work.is_symlink() or work == root or work in root.parents:
        _fail("unsafe_path", "更新暂存目录无效。")
    work = work.resolve()
    if root in work.parents and work.relative_to(root).parts[0] in _SOURCE_DIRS:
        _fail("unsafe_path", "更新暂存目录与源码文件重叠。")
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        _fail("staging_in_use", "更新暂存目录已被占用。")
    _emit(progress_callback, "checking", "正在检查本地源码")
    if source_kind == "git":
        _git_clean(root, current_commit)
    elif (root / ".git").exists():
        _fail("wrong_repository", "Git 源码目录不能按源码包方式更新。")
    baseline, staged = work / "baseline", work / "staged"
    baseline.mkdir()
    staged.mkdir()
    _download_archive(current_commit, work / "baseline.zip", policy, progress_callback, cancel=cancel)
    _emit(progress_callback, "extracting", "正在解压源码包", archive="baseline.zip")
    before_source = _extract(work / "baseline.zip", baseline, current_commit, cancel)
    _download_archive(target_commit, work / "target.zip", policy, progress_callback, cancel=cancel)
    _emit(progress_callback, "extracting", "正在解压源码包", archive="target.zip")
    after_source = _extract(work / "target.zip", staged, target_commit, cancel)
    _emit(progress_callback, "verifying", "正在校验源码文件")
    before = _hashes(root, before_source.keys() | after_source.keys())
    for name, digest in before_source.items():
        if source_kind == "package" and before[name] != digest:
            marker = name == "ypuddin/_archive_revision.txt" and before[name] is not None
            if marker and _path(root, name).read_bytes().strip() in (current_commit.encode("ascii"), b"$Format:%H$"):
                continue
            _fail("local_changes", f"源码文件与已安装的版本不同：{name}")
    for name in after_source.keys() - before_source.keys():
        if before[name] is not None:
            _fail("local_changes", f"新版本的文件会覆盖已有文件：{name}")
    old_frontend = _frontend(root, required=False)
    before.update(old_frontend)
    if source_kind == "package":
        before.update(_hashes(root, ["SOURCE_MANIFEST.json"]))
    else:
        _emit(progress_callback, "fetching", "正在同步 Git 版本记录")
        _git(root, "fetch", "--no-tags", "--no-recurse-submodules", ORIGIN, target_commit, policy=policy,
             timeout=180, cancel=cancel, network=True)
        if _git(root, "rev-parse", "FETCH_HEAD") != target_commit:
            _fail("wrong_revision", "Git 拉取到的版本与要求的版本不同。")
        try:
            _git(root, "merge-base", "--is-ancestor", current_commit, target_commit)
        except SourceUpdateError:
            _fail("not_fast_forward", "新版本不是当前版本的后续提交，无法直接更新。")
    _check(cancel)
    plan = {
        "format": 1, "root": str(root), "source_kind": source_kind,
        "current_commit": current_commit, "target_commit": target_commit,
        "plan_path": str(work / "plan.json"), "staged_root": str(staged), "phase": "prepared",
        "before": before, "source_before": before_source, "source_after": after_source,
        "frontend_before": old_frontend, "created_at": time.time(),
    }
    _write_json(work / "plan.json", plan)
    _emit(progress_callback, "prepared", "源码准备完成")
    return plan


def seal_update(plan_path, progress_callback=None) -> dict:
    path, plan = _plan(plan_path)
    if plan["phase"] != "prepared":
        _fail("invalid_phase", "此更新不在等待前端构建的阶段。")
    root, staged = Path(plan["root"]), path.parent / "staged"
    if _hashes(staged, plan["source_after"]) != plan["source_after"]:
        _fail("staging_changed", "准备更新期间，暂存的源码文件发生了变化。")
    frontend = _frontend(staged, required=True)
    after = {**plan["source_after"], **frontend}
    for name in frontend.keys() - plan["frontend_before"].keys():
        if _path(root, name).exists():
            _fail("local_changes", f"新的前端文件会覆盖已有文件：{name}")
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
    _emit(progress_callback, "ready", "源码与前端构建校验完成")
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
        _replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _remove(root: Path, name: str) -> None:
    """Delete a file the update added, and the folders that only it created."""
    target = _path(root, name)
    target.unlink(missing_ok=True)
    folder = target.parent
    while folder != root and root in folder.parents:
        try:
            folder.rmdir()
        except OSError:
            break
        folder = folder.parent


def _discard_partial_copies(root: Path, names) -> None:
    """Delete the temporary copies a process stopped in the middle of ``_copy`` left beside installed files."""
    for folder in {_path(root, name).parent for name in names}:
        try:
            leftovers = [entry for entry in folder.glob(".ypuddin-update-*") if entry.is_file() and not entry.is_symlink()]
        except OSError:
            continue
        for entry in leftovers:
            entry.unlink(missing_ok=True)


def _apply_order(names) -> list[str]:
    """Version records last, so the tree reports the new version only once every file is new."""
    return sorted(names, key=lambda name: (name in _VERSION_FILES, _VERSION_FILES.index(name) if name in _VERSION_FILES else 0, name))


def _restore_order(names) -> list[str]:
    """Version records first, so a partly restored tree never reports the new version."""
    return sorted(names, key=lambda name: (name not in _VERSION_FILES, -_VERSION_FILES.index(name) if name in _VERSION_FILES else 0, name))


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
            _fail("checkout_changed", "Git 检出的前端构建输入除换行符外还有其他差异。")
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
        _fail("disk_check_failed", "安装更新前无法检查可用磁盘空间。")
    if not enough:
        _fail("insufficient_space", "磁盘空间不足，无法安装源码更新并保留备份。")


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


def _merge(path: Path, plan: dict) -> None:
    """Fast-forward the checkout, recording the Git process so a later recovery can tell its index lock."""
    plan["git_command"] = {"started_at": time.time(), "pid": None}
    _write_json(path, plan)

    def started(pid: int) -> None:
        plan["git_command"]["pid"] = pid
        _write_json(path, plan)

    _git(Path(plan["root"]), "merge", "--ff-only", "--no-edit", plan["target_commit"], timeout=GIT_MERGE_TIMEOUT,
         started=started)
    plan.pop("git_command")
    _write_json(path, plan)


def _install(path: Path, plan: dict, progress_callback, *, resume: bool) -> None:
    root, staged = Path(plan["root"]), path.parent / "staged"
    changed = plan["changed"]
    if plan["source_kind"] == "git":
        if _git(root, "rev-parse", "HEAD") != plan["target_commit"]:
            _merge(path, plan)
        _git_frontend_fingerprint(path, plan)
    for name in _apply_order(changed):
        if plan["source_kind"] == "git" and not name.startswith("frontend/dist/"):
            continue
        if name not in plan["after"]:
            _remove(root, name)
        elif not resume or _live(root, name) != plan["after"][name]:
            _copy(_path(staged, name), _path(root, name))
    _clear_bytecode(root, changed)
    plan["phase"] = "applied"
    _write_json(path, plan)
    _emit(progress_callback, "applied", "源码替换完成")


def _reason(error: BaseException) -> str:
    return error.message if isinstance(error, SourceUpdateError) else str(error) or type(error).__name__


def apply_update(plan_path, python=None, progress_callback=None) -> None:
    path, plan = _plan(plan_path)
    if plan["phase"] != "ready":
        _fail("invalid_phase", "此更新尚未准备好安装。")
    root, staged, backup = Path(plan["root"]), path.parent / "staged", path.parent / "backup"
    if _hashes(root, plan["before"]) != plan["before"]:
        _fail("source_changed", "准备更新后，已安装的文件发生了变化。")
    if _hashes(staged, plan["after"]) != plan["after"]:
        _fail("staging_changed", "准备好的更新文件在安装前发生了变化。")
    if plan["source_kind"] == "git":
        _git_clean(root, plan["current_commit"])
    changed = {name for name in plan["before"].keys() | plan["after"].keys()
               if plan["before"].get(name) != plan["after"].get(name)}
    _require_space(root, path.parent, plan, changed)
    if backup.is_dir() and not backup.is_symlink():
        shutil.rmtree(backup)  # left by an apply that stopped before changing installed files
    backup.mkdir(exist_ok=False)
    for name in sorted(changed):
        if plan["before"].get(name) is not None:
            _copy(_path(root, name), _path(backup, name))
    plan.update(phase="applying", changed=sorted(changed))
    _write_json(path, plan)
    try:
        _emit(progress_callback, "applying", "正在替换源码")
        _install(path, plan, progress_callback, resume=False)
    except BaseException as error:
        try:
            rollback_update(path, python=python, progress_callback=progress_callback)
        except BaseException as rollback_error:
            raise SourceUpdateError(
                "rollback_failed",
                f"更新未能安装：{_reason(error)}\n也未能自动恢复原版源码：{_reason(rollback_error)}\n备份保留在 {backup}。",
            ) from rollback_error
        raise SourceUpdateError("apply_failed", f"更新未能安装，已恢复原版源码：{_reason(error)}") from error


def _restore_git_files(path: Path, plan: dict) -> None:
    """Put back the files a stopped or failed fast-forward wrote while HEAD stayed at the old commit."""
    root, backup = Path(plan["root"]), path.parent / "backup"
    changes = _git_changes(root)
    unknown = sorted(changes - set(plan["changed"]))
    if unknown:
        _fail("rollback_failed", f"Git 源码中有更新以外的改动，未能恢复原版源码：{unknown[0]}。备份已保留。")
    for name in _restore_order(changes):
        if plan["before"].get(name) is None:
            _remove(root, name)
        else:
            _copy(_path(backup, name), _path(root, name))
    if _git_changes(root):
        _fail("rollback_failed", f"未能恢复原版 Git 源码，工作区仍有改动。备份保留在 {backup}。")


def _git_restore(path: Path, plan: dict) -> None:
    root = Path(plan["root"])
    head = _git(root, "rev-parse", "HEAD")
    # Restoring files needs no index lock; moving HEAD back does.
    _wait_for_index_lock(root, plan, required=head == plan["target_commit"])
    if head == plan["target_commit"]:
        if _git_changes(root):
            _fail("rollback_conflict", "更新后 Git 源码有新的改动，已保留备份。")
        _git(root, "read-tree", "-u", "-m", plan["target_commit"], plan["current_commit"])
        _git(root, "update-ref", "-m", "Restore previous trainer source", "HEAD", plan["current_commit"], plan["target_commit"])
    elif head != plan["current_commit"]:
        _fail("rollback_conflict", "更新后 Git 版本发生了变化，已保留备份。")
    _restore_git_files(path, plan)


def rollback_update(plan_path, python=None, progress_callback=None) -> None:
    path, plan = _plan(plan_path)
    if plan["phase"] == "rolled_back":
        return
    if plan["phase"] not in ("applying", "applied", "rolling_back"):
        _fail("invalid_phase", "此更新没有改动已安装的源码。")
    root, backup = Path(plan["root"]), path.parent / "backup"
    changed = plan["changed"]
    expected_backups = {name: plan["before"][name] for name in changed if plan["before"].get(name) is not None}
    if _hashes(backup, expected_backups) != expected_backups:
        _fail("backup_changed", "无法校验源码更新的备份。")
    for name in changed:
        if plan["source_kind"] == "git" and not name.startswith("frontend/dist/"):
            continue
        if _live(root, name) not in (plan["before"].get(name), plan["after"].get(name)):
            _fail("rollback_conflict", f"更新后有文件被改动，已保留备份：{name}")
    plan["phase"] = "rolling_back"
    _write_json(path, plan)
    _emit(progress_callback, "rolling_back", "正在恢复原版源码")
    if plan["source_kind"] == "git":
        _git_restore(path, plan)
    for name in _restore_order(changed):
        if plan["source_kind"] == "git" and not name.startswith("frontend/dist/"):
            continue
        if plan["before"].get(name) is None:
            _remove(root, name)
        elif _live(root, name) != plan["before"][name]:
            _copy(_path(backup, name), _path(root, name))
    _clear_bytecode(root, changed)
    plan["phase"] = "rolled_back"
    plan.pop("git_command", None)
    _write_json(path, plan)
    _emit(progress_callback, "rolled_back", "已恢复原版源码")


def _resume(path: Path, plan: dict, progress_callback) -> None:
    root, staged = Path(plan["root"]), path.parent / "staged"
    if _hashes(staged, plan["after"]) != plan["after"]:
        _fail("staging_changed", "准备好的更新文件已缺失或损坏。")
    for name in plan["changed"]:
        if plan["source_kind"] == "git" and not name.startswith("frontend/dist/"):
            continue
        if _live(root, name) not in (plan["before"].get(name), plan["after"].get(name)):
            _fail("rollback_conflict", f"更新中断后有文件被改动：{name}")
    if plan["source_kind"] == "git":
        head = _git(root, "rev-parse", "HEAD")
        if head == plan["current_commit"]:
            _wait_for_index_lock(root, plan)
            _restore_git_files(path, plan)
        elif head != plan["target_commit"] or _git_changes(root):
            _fail("source_changed", "更新中断后 Git 源码发生了变化。")
    _install(path, plan, progress_callback, resume=True)


def recover_update(plan_path, python=None, progress_callback=None) -> str:
    """Finish or undo an update that stopped while replacing the installed files; returns the phase.

    The update is finished when every staged file still matches its recorded checksum and
    nothing else changed the installed files; otherwise the previous files are restored.
    """
    path, plan = _plan(plan_path)
    if plan["phase"] in ("applying", "rolling_back"):
        _discard_partial_copies(Path(plan["root"]), plan["changed"])
    if plan["phase"] == "rolling_back":
        rollback_update(path, python=python, progress_callback=progress_callback)
        return "rolled_back"
    if plan["phase"] != "applying":
        return plan["phase"]
    _emit(progress_callback, "resuming", "正在完成上次中断的源码替换")
    try:
        _resume(path, plan, progress_callback)
        return "applied"
    except SourceUpdateError as error:
        _emit(progress_callback, "resume_failed", f"无法完成上次的源码替换：{error.message}")
    rollback_update(path, python=python, progress_callback=progress_callback)
    return "rolled_back"
