"""Prepare an in-place source update without replacing the selected compute runtime.

The launcher copies this helper before replacing source files and runs its ``build``
and ``deps`` commands in separate processes. Source rollback belongs to the launcher;
Python package installation is not transactional.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1]
_HELPER_FILES = (
    "scripts/update_prepare.py",
    "scripts/bootstrap.py",
    "ypuddin/__init__.py",
    "ypuddin/dtk_builds.py",
    "ypuddin/package_sources.py",
)


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
    try:
        saved = json.loads(boot.MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    if runtime.get("hip"):
        boot.PROFILE = "linux-dtk"
        tag = "dtk"
    elif runtime.get("cuda"):
        boot.PROFILE = "windows-cuda" if os.name == "nt" else "linux-cuda"
        tag = "cu" + runtime["cuda"].replace(".", "")
    else:
        boot.PROFILE = (
            "macos-mps"
            if platform.system() == "Darwin"
            else ("windows-cpu" if os.name == "nt" else "linux-cpu")
        )
        tag = "cpu"
    # Legacy environments keep their cache/marker identity; HIP constraints still apply.
    if saved.get("profile") == "legacy" and not runtime.get("hip"):
        boot.PROFILE = "legacy"
    declared = set(project["extras"])
    extras = [name for name in boot.EXTRAS_BASE.split(",") if name in declared]
    previous = saved.get("extras", "")
    if isinstance(previous, str):
        extras += [name for name in previous.split(",") if name in declared and name not in extras]
    for name, enabled in (("dtk", bool(runtime.get("hip"))), ("nvidia", bool(runtime.get("cuda")))):
        if enabled and name in declared and name not in extras:
            extras.append(name)
    return ",".join(extras), tag


def _assert_native_unchanged(boot: ModuleType, protected: dict[str, str]) -> dict[str, str]:
    versions = boot.installed_versions()
    changed = [name for name, version in protected.items() if versions.get(name) != version]
    if changed:
        raise PreparationError("受保护的计算依赖版本发生变化：" + ", ".join(changed) + "。更新已停止。")
    return versions


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
    requirement = str(boot.ROOT) + (f"[{extras}]" if extras else "")
    for index, source in enumerate(sources):
        command = [
            str(selected_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "deps"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
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
