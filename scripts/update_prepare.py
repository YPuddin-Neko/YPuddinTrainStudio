"""Prepare an in-place source update without replacing the selected compute runtime.

The launcher copies this helper before replacing source files and runs its commands in
separate processes: ``build`` and ``deps`` prepare the frontend and the Python packages,
``apply`` replaces the source files and ``recover`` finishes or undoes an apply that
stopped part-way. Python package installation is not transactional.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"legacy", "windows-cuda", "linux-cuda", "linux-dtk", "macos-mps", "windows-cpu", "linux-cpu", "macos-cpu"}
_HELPER_FILES = (
    "scripts/update_prepare.py",
    "scripts/bootstrap.py",
    "ypuddin/__init__.py",
    "ypuddin/dtk_builds.py",
    "ypuddin/package_sources.py",
    "ypuddin/server/source_update.py",
)
LOCK_WAIT = 1800


class PreparationError(RuntimeError):
    pass


def snapshot_helpers(work_dir: str | Path) -> Path:
    """Freeze the current preparation code before the live checkout changes."""
    work = Path(work_dir).absolute()
    work.mkdir(parents=True, exist_ok=True)
    destination = work / "helper"
    if destination.exists():
        if not all((destination / name).is_file() for name in _HELPER_FILES):
            raise PreparationError("更新准备脚本不完整，请重新开始更新。")
        return destination / "scripts" / "update_prepare.py"
    with tempfile.TemporaryDirectory(prefix=".helper-", dir=work) as temporary:
        bundle = Path(temporary) / "helper"
        for name in _HELPER_FILES:
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE_ROOT / name, target)
        bundle.rename(destination)
    return destination / "scripts" / "update_prepare.py"


def _bootstrap(root: str | Path, settings: dict[str, Any]) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_ypuddin_update_bootstrap", Path(__file__).with_name("bootstrap.py")
    )
    if spec is None or spec.loader is None:
        raise PreparationError("无法加载更新准备脚本。")
    boot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boot)
    boot.ROOT = Path(root).resolve()
    boot.FRONTEND = boot.ROOT / "frontend"
    downloads = settings.get("downloads")
    boot.DOWNLOAD_SETTINGS = downloads if isinstance(downloads, dict) else None
    paths = settings.get("paths", {})
    boot.PACKAGE_CACHE_ROOT = paths.get("cache_dir") if isinstance(paths, dict) else None
    return boot


def prepare_frontend(staged_root: str | Path, settings: dict[str, Any], work_dir: str | Path) -> None:
    """Validate a complete prebuild or build inside the staged source tree."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    boot = _bootstrap(staged_root, settings)
    boot.FRONTEND_PURPOSE = "update"
    if not (boot.FRONTEND / "package.json").is_file():
        raise PreparationError("更新源码缺少前端 package.json。")
    boot.build_frontend()
    if boot.frontend_stale():
        raise PreparationError("更新后的前端文件校验失败。")


def _project_environment(boot: ModuleType) -> dict[str, Any]:
    return boot.venv_json(
        """
import json, platform, sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from packaging.specifiers import SpecifierSet
from pathlib import Path
project = tomllib.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['project']
required = project.get('requires-python', '')
print(json.dumps({
    'prefix': sys.prefix,
    'python': platform.python_version(),
    'requires_python': required,
    'python_ok': platform.python_version() in SpecifierSet(required),
    'extras': list(project.get('optional-dependencies', {})),
}))
""",
        str(boot.ROOT / "pyproject.toml"),
    )


def _install_options(boot: ModuleType, project: dict, runtime: dict) -> tuple[str, str]:
    """Extras and PyTorch tag for the existing environment; sets ``boot.PROFILE``.

    The environment keeps the profile its install record names, then the one its location
    names; only an environment with neither is classified by its PyTorch build. The launcher
    refuses an environment recorded under another profile.
    """
    try:
        saved = json.loads(boot.MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    if runtime.get("hip"):
        guessed, tag = "linux-dtk", "dtk"
    elif runtime.get("cuda"):
        guessed = "windows-cuda" if os.name == "nt" else "linux-cuda"
        tag = "cu" + runtime["cuda"].replace(".", "")
    else:
        guessed = (
            "macos-mps"
            if platform.system() == "Darwin"
            else ("windows-cpu" if os.name == "nt" else "linux-cpu")
        )
        tag = "cpu"
    venv = boot.VENV
    location = venv.parent.name if venv.name == "venv" and venv.parent.parent.name == "environment" else None
    recorded = saved.get("profile")
    boot.PROFILE = recorded if recorded in PROFILES else location if location in PROFILES else guessed
    declared = set(project["extras"])
    extras = [name for name in boot.EXTRAS_BASE.split(",") if name in declared]
    previous = saved.get("extras", "")
    if isinstance(previous, str):
        extras += [name for name in previous.split(",") if name in declared and name not in extras]
    hip = bool(runtime.get("hip")) and boot.PROFILE in ("linux-dtk", "legacy")
    cuda = bool(runtime.get("cuda")) and not boot.PROFILE.endswith("-cpu")
    for name, enabled in (("dtk", hip), ("nvidia", cuda)):
        if enabled and name in declared and name not in extras:
            extras.append(name)
    return ",".join(extras), tag


def _assert_native_unchanged(boot: ModuleType, protected: dict[str, str]) -> dict[str, str]:
    versions = boot.installed_versions()
    changed = [name for name, version in protected.items() if versions.get(name) != version]
    if changed:
        raise PreparationError("受保护的计算依赖版本发生变化：" + ", ".join(changed) + "。更新已停止。")
    return versions


def _dependency_installer(boot: ModuleType, python: Path) -> list[str]:
    # Shared DTK packages belong to the vendor interpreter; pip leaves them in place.
    shared = boot.PROFILE == "linux-dtk" and boot.recorded_vendor_stack()
    uv = None if shared else boot.uv_path()
    try:
        config = (boot.VENV / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError:
        config = ""
    uv_created = any(line.partition("=")[0].strip() == "uv" for line in config.splitlines())
    if uv_created and uv:
        return [uv, "pip", "install", "--python", str(python)]

    probe = subprocess.run(
        [str(python), "-m", "pip", "--version"],
        cwd=boot.ROOT, env=boot._env(), capture_output=True, text=True, timeout=30,
    )
    if probe.returncode:
        if uv:
            boot.log("当前环境缺少可用的 pip，使用 uv 安装更新依赖")
            return [uv, "pip", "install", "--python", str(python)]
        boot.log("当前环境缺少可用的 pip，尝试离线修复")
        repaired = subprocess.run(
            [str(python), "-m", "ensurepip", "--upgrade"],
            cwd=boot.ROOT, env=boot._env(), timeout=120,
        )
        probe = subprocess.run(
            [str(python), "-m", "pip", "--version"],
            cwd=boot.ROOT, env=boot._env(), capture_output=True, text=True, timeout=30,
        )
        if repaired.returncode or probe.returncode:
            reason = (probe.stderr or probe.stdout).strip()
            raise PreparationError(
                "pip 离线修复失败，请检查上方 ensurepip 日志。"
                "安装 uv（https://docs.astral.sh/uv/getting-started/installation/）"
                "或修复当前 Python 的 pip 后重试。" + (f"\n{reason}" if reason else "")
            )
    return [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]


def install_dependencies(
    root: str | Path,
    python: str | Path,
    settings: dict[str, Any],
    work_dir: str | Path,
) -> None:
    """Reconcile the target requirements in the existing interpreter after worker exit."""
    work = Path(work_dir).absolute()
    work.mkdir(parents=True, exist_ok=True)
    boot = _bootstrap(root, settings)
    # Resolving a venv's Python symlink would select its base interpreter instead.
    selected_python = Path(python).expanduser().absolute()
    if not selected_python.is_file():
        raise PreparationError("当前 Python 环境不存在，无法安装更新依赖。")
    boot.venv_python = lambda: selected_python
    project = _project_environment(boot)
    if not project["python_ok"]:
        raise PreparationError(
            f"此更新要求 Python {project['requires_python']}，当前为 {project['python']}。"
        )
    boot.VENV = Path(project["prefix"])
    boot.MARKER = boot.VENV / ".ypuddin-install.json"
    versions = boot.installed_versions()
    missing = [name for name in ("torch", "torchvision") if name not in versions]
    if missing:
        raise PreparationError("当前环境缺少 " + " / ".join(missing) + "，请先修复运行环境。")
    runtime = boot.torch_runtime()
    extras, tag = _install_options(boot, project, runtime)
    recorded = boot.PROFILE
    if runtime.get("hip") and recorded == "legacy":
        # A legacy environment with vendor HIP PyTorch keeps its identity, but its
        # dependencies follow the DTK rules.
        boot.PROFILE = "linux-dtk"
    if boot.PROFILE == "linux-dtk":
        boot.validate_dtk_runtime(runtime)
    protected = boot.protected_versions(versions)
    compatibility = boot.dtk_compatibility_constraints(versions)
    # Keep the original constraints for both the update and an old-source repair attempt.
    baseline = work / "native-stack.json"
    if baseline.exists():
        original = json.loads(baseline.read_text(encoding="utf-8"))
        if not isinstance(original, dict) or original.get("python") != str(selected_python):
            raise PreparationError("更新记录与当前 Python 环境不匹配。")
        protected = original["versions"]
        _assert_native_unchanged(boot, protected)
    else:
        baseline.write_text(
            json.dumps({"python": str(selected_python), "versions": protected}, indent=2),
            encoding="utf-8",
        )
    constraints = work / "native-stack.txt"
    constraints.write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(protected.items()))
        + "".join(value + "\n" for value in compatibility),
        encoding="utf-8",
    )
    downloads = boot.DOWNLOAD_SETTINGS or {}
    sources = boot.pypi_sources(downloads.get("pypi", "auto"), downloads.get("fallback", True))
    boot.log("检查更新依赖，保留当前 PyTorch 计算环境")
    installer = _dependency_installer(boot, selected_python)
    requirement = str(boot.ROOT) + (f"[{extras}]" if extras else "")
    for index, source in enumerate(sources):
        command = [
            *installer,
            "--constraint",
            str(constraints),
            "--index-url",
            source,
            "-e",
            requirement,
        ]
        result = subprocess.run(command, cwd=boot.ROOT, env=boot._env())
        after = _assert_native_unchanged(boot, protected)
        if result.returncode == 0:
            break
        if index + 1 < len(sources):
            boot.log("依赖安装失败，换下一个下载源重试")
    else:
        raise PreparationError("更新依赖安装失败，请检查上方安装日志。")
    issues = boot.dependency_issues(extras) + boot.dtk_compatibility_issues(after, compatibility)
    if issues:
        raise PreparationError("更新后的依赖不完整：" + "; ".join(issues[:12]))
    if boot.PROFILE == "linux-dtk":
        boot.validate_dtk_numpy_bridge(after)
    if "models" in extras.split(","):
        report = boot.model_runtime()
        if not report.get("ok"):
            raise PreparationError("更新后模型依赖加载失败：" + str(report.get("error", "unknown error")))
    if not boot.editable_install_ready():
        raise PreparationError("更新后的训练器安装信息校验失败。")
    boot.cleanup_build_metadata()
    boot.PROFILE = recorded
    # Use the target bootstrap's bytes for the next normal startup's signature.
    bootstrap_file = boot.ROOT / "scripts" / "bootstrap.py"
    if bootstrap_file.is_file():
        boot.__file__ = str(bootstrap_file)
    boot.MARKER.write_text(
        json.dumps(
            {
                "signature": boot.install_signature(tag, extras),
                "profile": boot.PROFILE,
                "arch": boot.host_arch(),
                "torch": tag,
                "torch_version": after["torch"],
                "extras": extras,
                "time": boot.time.time(),
            }
        ),
        encoding="utf-8",
    )
    boot.log("更新依赖检查完成")


def _source_update() -> ModuleType:
    """The update code saved with this helper, unaffected by the files it replaces."""
    spec = importlib.util.spec_from_file_location(
        "_ypuddin_update_source", SOURCE_ROOT / "ypuddin" / "server" / "source_update.py"
    )
    if spec is None or spec.loader is None:
        raise PreparationError("无法加载更新安装脚本。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _try_lock(stream) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def lock_update(work_dir: str | Path, *, wait: bool):
    """Hold the update folder for one apply or recovery; the system releases it when the process ends.

    Returns the open lock file, or None when another process holds it and ``wait`` is false.
    """
    stream = (Path(work_dir) / "apply.lock").open("a+b")
    deadline = time.monotonic() + LOCK_WAIT
    announced = False
    while not _try_lock(stream):
        if not wait or time.monotonic() > deadline:
            stream.close()
            return None
        if not announced:
            print("[studio] 等待上次的源码替换结束…", flush=True)
            announced = True
        time.sleep(0.5)
    return stream


def _print_progress(event: dict) -> None:
    print(f"[studio] {event['message']}", flush=True)


def apply_source(work_dir: str | Path) -> int:
    """Replace the installed files. 0: applied; 2: not applied, previous files restored; 1: failed."""
    work = Path(work_dir).absolute()
    lock = lock_update(work, wait=False)
    if lock is None:
        print("[studio] 另一个进程正在安装此更新。", flush=True)
        return 1
    with lock:
        source = _source_update()
        try:
            source.apply_update(work / "plan.json", progress_callback=_print_progress)
        except source.SourceUpdateError as error:
            print(f"[studio] {error.message}", flush=True)
            return 2 if error.code == "apply_failed" else 1
    return 0


def recover_source(work_dir: str | Path) -> int:
    """Finish or undo an apply that stopped part-way. 0: the installed files are consistent."""
    work = Path(work_dir).absolute()
    lock = lock_update(work, wait=True)
    if lock is None:
        print("[studio] 上次的源码替换仍未结束。", flush=True)
        return 1
    with lock:
        source = _source_update()
        try:
            phase = source.recover_update(work / "plan.json", progress_callback=_print_progress)
        except source.SourceUpdateError as error:
            print(f"[studio] 无法完成或撤销上次中断的更新：{error.message}", flush=True)
            return 1
    print(f"[studio] 更新记录状态：{phase}", flush=True)
    return 0


def _stop_with_parent(pid: int) -> None:
    """End this helper and the installers it started once the process waiting for it is gone."""

    def watch() -> None:
        if os.name == "nt":
            import ctypes

            kernel = ctypes.windll.kernel32
            handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if handle:
                kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
                kernel.CloseHandle(handle)
            subprocess.run(["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            os._exit(1)
        while os.getppid() == pid:
            time.sleep(0.5)
        if os.getpgrp() == os.getpid():  # the helper leads its own group: npm and pip go too
            os.killpg(os.getpgrp(), signal.SIGKILL)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "deps", "apply", "recover"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args(argv)
    if args.action == "apply":
        return apply_source(args.work_dir)
    if args.action == "recover":
        return recover_source(args.work_dir)
    if args.root is None:
        parser.error("--root is required")
    if args.parent_pid:
        _stop_with_parent(args.parent_pid)
    try:
        settings = json.loads(args.settings_file.read_text(encoding="utf-8")) if args.settings_file else {}
        if not isinstance(settings, dict):
            raise PreparationError("更新设置格式错误。")
        if args.action == "build":
            prepare_frontend(args.root, settings, args.work_dir)
        else:
            install_dependencies(args.root, args.python, settings, args.work_dir)
    except (PreparationError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[studio] {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
