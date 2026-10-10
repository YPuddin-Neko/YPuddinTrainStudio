"""Inspect a candidate interpreter in a CPU-only child without changing its packages."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .issues import TtsIssue

_MARKER = "TTS_ENVIRONMENT "
_ENGINES = frozenset({"voxcpm1.5", "gpt-sovits-v5"})
_BASE_IMPORTS = {
    "voxcpm1.5": ("torch", "numpy", "torchaudio", "torchcodec", "soundfile", "safetensors", "transformers"),
    "gpt-sovits-v5": ("torch", "numpy", "torchaudio", "soundfile", "yaml", "transformers", "peft",
                      "pytorch_lightning", "tensorboard", "librosa", "resampy"),
}
_SOURCE_IMPORTS = {
    "voxcpm1.5": ("voxcpm", "voxcpm.model.voxcpm", "voxcpm.training"),
    "gpt-sovits-v5": ("module.models", "module.models_v5", "module.data_utils", "AR.data.data_module",
                      "AR.models.t2s_lightning_module", "process_ckpt", "tools.my_utils", "BigVGAN.bigvgan",
                      "GPT_SoVITS.f5_tts.model", "text", "utils"),
}

# This script deliberately has no imports from the service or third-party packaging tools.
_CHILD = r'''
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import sys
from pathlib import Path

request = json.load(sys.stdin)
records = sorted((re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(), d.version)
                 for d in importlib.metadata.distributions() if d.metadata.get("Name"))
packages = dict(records)
identity = {"python_path": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "python_version": platform.python_version(), "packages": records}
fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
result = {**identity, "packages": packages, "dependency_fingerprint": fingerprint,
          "platform": platform.system(), "machine": platform.machine(), "torch_version": None,
          "cuda_runtime": None, "hip_runtime": None,
          "duplicates": sorted({name for name, _ in records if sum(n == name for n, _ in records) > 1}),
          "marker_environment": {"implementation_name": sys.implementation.name,
              "implementation_version": platform.python_version(), "os_name": os.name,
              "platform_machine": platform.machine(), "platform_python_implementation": platform.python_implementation(),
              "platform_release": platform.release(), "platform_system": platform.system(),
              "platform_version": platform.version(), "python_full_version": platform.python_version(),
              "python_version": ".".join(platform.python_version_tuple()[:2]), "sys_platform": sys.platform,
              "extra": ""}, "imports": {}, "import_errors": [], "policy_errors": []}
root = Path(request["work_dir"]).resolve()
null_device = os.devnull
null_device_name = os.path.normcase(null_device)

def audit(event, args):
    if event == "subprocess.Popen":
        command = args[1]
        # NumPy's SVE feature query is optional and also runs on hosts without lscpu.
        if (args[0] in (None, "lscpu") and args[2] is None and args[3] is None
                and (command == "lscpu" or isinstance(command, (list, tuple)) and list(command) == ["lscpu"])):
            executable = shutil.which("lscpu")
            if (executable is None or os.name != "nt"
                    and Path(executable).resolve() in {Path("/usr/bin/lscpu"), Path("/bin/lscpu")}):
                return
        if isinstance(command, (list, tuple)) and command:
            name = Path(os.fsdecode(command[0])).name
            # Font discovery and the dynamic-loader cache are read-only import helpers.
            if ((name == "fc-list" and list(command[1:]) in (["--help"], ["--format=%{file}\\n"]))
                    or (name == "ldconfig" and list(command[1:]) == ["-p"])):
                return
    if event in {"socket.connect", "socket.getaddrinfo", "urllib.Request", "subprocess.Popen", "os.system"}:
        result["policy_errors"].append(event)
        raise RuntimeError("Environment inspection cannot access the network or start another process")
    path = None
    if event == "open":
        name, mode, flags = args
        if (isinstance(name, (str, bytes, os.PathLike))
                and os.path.normcase(os.fsdecode(name)) == null_device_name
                and (os.name == "nt" or stat.S_ISCHR(os.stat(null_device).st_mode))):
            return
        if isinstance(name, (str, bytes, os.PathLike)) and ((mode and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            path = name
    elif event in {"os.remove", "os.mkdir", "os.rmdir", "os.chmod", "os.truncate", "os.utime"}:
        path = args[0]
    elif event in {"os.rename", "os.link", "os.symlink"}:
        for name in args[:2]:
            if not Path(os.fsdecode(name)).resolve().is_relative_to(root):
                result["policy_errors"].append(event)
                raise RuntimeError("Environment inspection cannot change files outside its temporary directory")
    if path is not None and not Path(os.fsdecode(path)).resolve().is_relative_to(root):
        result["policy_errors"].append(event)
        raise RuntimeError("Environment inspection cannot change files outside its temporary directory")

sys.addaudithook(audit)
sys.path[:0] = request.get("source_paths", [])
for name in request.get("imports", []):
    try:
        module = importlib.import_module(name)
        if name == "torch":
            result.update(torch_version=str(module.__version__), cuda_runtime=module.version.cuda,
                          hip_runtime=getattr(module.version, "hip", None))
        locations = []
        if getattr(module, "__file__", None):
            locations.append(str(Path(module.__file__).resolve()))
        locations.extend(str(Path(p).resolve()) for p in getattr(module, "__path__", []))
        expected = request.get("source_roots", {}).get(name.split(".")[0])
        if expected is not None:
            expected = Path(expected).resolve()
            # Check parent namespaces too: an imported leaf may conceal a mixed package path.
            for end in range(1, len(name.split("."))):
                parent = sys.modules.get(".".join(name.split(".")[:end]))
                if getattr(parent, "__file__", None):
                    locations.append(str(Path(parent.__file__).resolve()))
                locations.extend(str(Path(p).resolve()) for p in getattr(parent, "__path__", []))
            if not locations or any(not Path(p).is_relative_to(expected) for p in locations):
                raise ValueError("Imported module is outside the selected trainer source: " + name)
        result["imports"][name] = locations
    except Exception as exc:
        result["import_errors"].append({"module": name, "type": type(exc).__name__, "message": str(exc),
                                        "missing": getattr(exc, "name", None) if isinstance(exc, ModuleNotFoundError) else None})
for name, module in list(sys.modules.items()):
    expected = request.get("source_roots", {}).get(name.split(".")[0])
    if expected is None or module is None:
        continue
    try:
        locations = ([getattr(module, "__file__")] if getattr(module, "__file__", None) else [])
        locations.extend(getattr(module, "__path__", []))
        if not locations or any(not Path(p).resolve().is_relative_to(Path(expected).resolve()) for p in locations):
            raise ValueError("Imported namespace is outside the selected trainer source: " + name)
    except Exception as exc:
        result["import_errors"].append({"module": name, "type": "ValueError",
            "message": str(exc), "missing": None})
print("TTS_ENVIRONMENT " + json.dumps(result, ensure_ascii=False))
'''


def _cancelled(cancel) -> bool:
    return bool(cancel() if callable(cancel) else cancel is not None and cancel.is_set())


def _stop(proc) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        proc.kill()


class ProbeCancelled(ValueError):
    pass


def _run(python_path: str, request: dict, *, timeout: float, cancel=None, work_dir: Path | None = None,
         resource_env: dict | None = None) -> dict:
    if _cancelled(cancel):
        raise ProbeCancelled("环境检查已取消。")
    candidate = Path(python_path).expanduser()
    if not candidate.is_absolute() or not candidate.is_file():
        raise ValueError("Python 路径不存在或不是绝对路径。")
    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tts-environment-probe-", dir=work_dir) as folder:
        scratch = Path(folder).resolve()
        env = dict(os.environ)
        env.update(resource_env or {})
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "RANK", "LOCAL_RANK", "WORLD_SIZE",
                    "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
                   PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", GRADIO_ANALYTICS_ENABLED="False", DO_NOT_TRACK="1",
                   TOKENIZERS_PARALLELISM="false", language="en_US")
        for key in ("HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                    "MODELSCOPE_CACHE", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "TORCH_HOME", "XDG_CACHE_HOME",
                    "GRADIO_TEMP_DIR", "TMPDIR", "TEMP", "TMP"):
            env[key] = str(scratch / key.lower())
            Path(env[key]).mkdir()
        payload = {**request, "work_dir": str(scratch)}
        proc = subprocess.Popen([str(candidate), "-s", "-B", "-c", _CHILD], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", env=env,
                                cwd=scratch, start_new_session=os.name != "nt")
        started = time.monotonic()
        data = json.dumps(payload)
        try:
            while True:
                if _cancelled(cancel):
                    raise ProbeCancelled("环境检查已取消。")
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(proc.args, timeout)
                try:
                    stdout, stderr = proc.communicate(data, timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    data = None
        except BaseException:
            _stop(proc)
            proc.communicate(timeout=10)
            raise
        marker = next((line[len(_MARKER):] for line in reversed(stdout.splitlines()) if line.startswith(_MARKER)), None)
        if proc.returncode or marker is None:
            raise ValueError(f"Python 环境检查失败：{(stderr or stdout)[-1500:]}")
        result = json.loads(marker)
        if not isinstance(result, dict) or not isinstance(result.get("packages"), dict):
            raise ValueError("Python 环境检查返回了无效结果。")
        return result


def metadata_fingerprint(python_path: str, *, timeout: float = 30) -> str:
    """Read installed distribution identity without importing the ML stack."""
    return _run(python_path, {}, timeout=timeout)["dependency_fingerprint"]


def _source(engine: str, trainer_path: str) -> tuple[list[str], dict[str, str]]:
    if engine == "voxcpm1.5":
        from .core import validate_upstream

        root = Path(validate_upstream(trainer_path)["path"])
        return [str(root / "src")], {"voxcpm": str(root / "src" / "voxcpm")}
    from .gpt_sovits.core import validate_upstream

    root = Path(validate_upstream(trainer_path)["path"])
    code = root / "GPT_SoVITS"
    return [str(root), str(code), str(code / "BigVGAN")], {
        "module": str(code / "module"), "AR": str(code / "AR"), "process_ckpt": str(code),
        "tools": str(root / "tools"), "BigVGAN": str(code / "BigVGAN"), "f5_tts": str(code / "f5_tts"),
        "text": str(code / "text"), "utils": str(code), "GPT_SoVITS": str(code),
    }


def probe_environment(python_path: str, engine: str, trainer_path: str | None = None, *, timeout: float = 90,
                      cancel=None, work_dir: Path | None = None) -> dict:
    from .environment_recipes import requirements

    started = time.monotonic()
    result = {"python_path": python_path, "prefix": "", "python_version": "", "packages": {},
              "dependency_fingerprint": "", "state": "error", "issues": [], "missing": [], "conflicts": [],
              "imports_checked": False, "source_checked": False, "gpu_state": "unchecked",
              "platform": "", "machine": "", "torch_version": None, "cuda_runtime": None, "hip_runtime": None}

    def issue(code, loc, message, **details):
        result["issues"].append(TtsIssue(code=f"tts.environment.{code}", loc=["environment", loc],
                                             message=message, details=details).model_dump())

    if engine not in _ENGINES:
        issue("engine", "engine", "不支持此语音引擎。")
        return result
    if _cancelled(cancel):
        issue("cancelled", "python_path", "环境检查已取消。")
        return result
    paths, roots, resource_env = [], {}, {}
    resource_error = False
    if trainer_path:
        try:
            paths, roots = _source(engine, trainer_path)
            result["source_checked"] = True
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            issue("source", "trainer_path", str(exc))
            result["conflicts"].append("trainer_path")
        if result["source_checked"]:
            from .environment_resources import resource_environment

            try:
                resource_env = resource_environment(trainer_path)
            except FileNotFoundError as exc:
                issue("resource_missing", "resources", str(exc))
                result["missing"].append("runtime_resources")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issue("resources", "resources", str(exc))
                resource_error = True
    else:
        issue("source_missing", "trainer_path", "训练器源码尚未准备。")
        result["missing"].append("trainer_path")
    imports = [*_BASE_IMPORTS[engine], *(_SOURCE_IMPORTS[engine] if paths else ())]
    try:
        report = _run(python_path, {"imports": imports, "source_paths": paths, "source_roots": roots},
                      timeout=max(0, timeout - (time.monotonic() - started)), cancel=cancel,
                      work_dir=work_dir, resource_env=resource_env)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        code = "cancelled" if isinstance(exc, ProbeCancelled) else "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "probe"
        issue(code, "python_path", "环境检查超时。" if code == "timeout" else str(exc))
        return result
    for key in ("python_path", "prefix", "python_version", "packages", "dependency_fingerprint",
                "platform", "machine", "torch_version", "cuda_runtime", "hip_runtime"):
        result[key] = report[key]
    if not Version("3.10") <= Version(report["python_version"]) < Version("3.13"):
        issue("python_version", "python_path", "需要 Python 3.10 至 3.12。", actual=report["python_version"])
        result["conflicts"].append("python")
    for text in requirements(engine):
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate(report["marker_environment"]):
            continue
        name = canonicalize_name(requirement.name)
        installed = report["packages"].get(name)
        if installed is None:
            result["missing"].append(text)
            issue("dependency_missing", "dependencies", f"尚未安装 {requirement.name}。", requirement=text)
        else:
            try:
                compatible = requirement.specifier.contains(installed, prereleases=True)
            except InvalidVersion:
                compatible = False
            if not compatible:
                result["conflicts"].append(text)
                issue("dependency_version", "dependencies", f"{requirement.name} {installed} 不满足 {requirement.specifier}。",
                      requirement=text, installed=installed)
    for name in report["duplicates"]:
        result["conflicts"].append(name)
        issue("dependency_duplicate", "dependencies", f"发现多份 {name} 安装记录。", package=name)
    for failure in report["import_errors"]:
        module = failure["module"]
        if failure["missing"]:
            result["missing"].append(failure["missing"])
        else:
            result["conflicts"].append(module)
        issue("dependency_import", "dependencies", f"无法加载 {module}：{failure['message']}", failure=failure)
    if report["policy_errors"]:
        issue("import_side_effect", "dependencies", "依赖加载尝试联网、启动其他进程或写入环境目录。",
              operations=sorted(set(report["policy_errors"])))
    result["imports_checked"] = not report["import_errors"] and not report["policy_errors"]
    result["missing"] = sorted(set(result["missing"]))
    result["conflicts"] = sorted(set(result["conflicts"]))
    result["state"] = ("error" if resource_error or report["policy_errors"] else "incompatible" if result["conflicts"]
                       else "missing" if result["missing"] else "ready")
    if result["state"] == "ready":
        from .environment_resources import verify_runtime_resources

        resources = verify_runtime_resources(python_path, engine, trainer_path,
                                             timeout=max(0, timeout - (time.monotonic() - started)),
                                             cancel=cancel, work_dir=work_dir)
        result["resource_check"] = resources
        if resources["state"] != "ready":
            result["state"] = resources["state"]
            result["issues"].extend(TtsIssue.model_validate(item).model_dump() for item in resources["issues"])
            if resources["state"] == "missing":
                result["missing"].append("runtime_resources")
    return result
