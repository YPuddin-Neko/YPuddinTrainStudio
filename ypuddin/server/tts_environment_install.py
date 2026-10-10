"""Prepare a speech runtime inside one operation directory without changing deployment packages."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import re
import shutil
import signal
import stat
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from packaging.version import Version

from ypuddin.package_sources import pypi_sources, run_sources, torch_sources
from ypuddin.tts.environment_probe import probe_environment
from ypuddin.tts.environment_recipes import (
    MACOS_ARM64_FFMPEG,
    MACOS_FFMPEG_ALIASES,
    MACOS_FFMPEG_LIBRARIES,
    RECIPE_REVISION,
    TEXT_RESOURCES,
    WINDOWS_GSV_WHEELS,
    WINDOWS_SHARED_FFMPEG,
    requirements,
    source,
)
from ypuddin.tts.environment_resources import MARKER, _regular, resource_environment

from .download_sources import probe_options
from .network import ProxyPolicy


class PreparationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = f"tts.environment.{code}"


def _cancelled(cancel) -> None:
    if cancel.is_set():
        raise InterruptedError("环境准备已取消。")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stop(proc) -> None:
    if os.name == "nt":
        if proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        proc.kill()


class Runner:
    def __init__(self, context, root: Path):
        self.root = root
        self.policy = ProxyPolicy.from_context(context)
        self.env = self.policy.subprocess_env()
        for name in tuple(self.env):
            if name.startswith(("PIP_", "GIT_")) or name in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "IMAGEIO_FFMPEG_EXE"}:
                self.env.pop(name, None)
        self.env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                        PYTHONIOENCODING="utf-8", PIP_CONFIG_FILE=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                        GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_TERMINAL_PROMPT="0", HF_HUB_DISABLE_TELEMETRY="1",
                        GRADIO_ANALYTICS_ENABLED="False", DO_NOT_TRACK="1")
        self.env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
        for name in ("TMP", "TEMP", "TMPDIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR",
                     "NUMBA_CACHE_DIR", "GRADIO_TEMP_DIR", "UV_CACHE_DIR", "PIP_CACHE_DIR"):
            folder = root / "cache" / name.lower()
            folder.mkdir(parents=True, exist_ok=True)
            self.env[name] = str(folder)

    def run(self, args, log, cancel, *, timeout=3600, extra_env=None) -> str:
        _cancelled(cancel)
        env = dict(self.env)
        env.update(extra_env or {})
        proc = subprocess.Popen([str(arg) for arg in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=self.root, env=env, start_new_session=os.name != "nt")
        messages = queue.Queue(maxsize=256)
        stopped = threading.Event()

        def enqueue(item):
            while not stopped.is_set():
                try:
                    messages.put(item, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def collect():
            try:
                while not stopped.is_set() and (chunk := proc.stdout.readline(16384)):
                    enqueue(chunk.decode("utf-8", "replace"))
            finally:
                enqueue(None)

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        output, size = [], 0
        started = time.monotonic()
        try:
            ended = False
            while not ended:
                _cancelled(cancel)
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("环境准备命令执行超时。")
                try:
                    item = messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    ended = True
                    continue
                text = self.policy.redact(item).rstrip()
                if text:
                    log(text)
                    output.append(text)
                    size += len(text)
                    while size > 512 * 1024 and len(output) > 1:
                        size -= len(output.pop(0))
            proc.wait(timeout=max(1, timeout - (time.monotonic() - started)))
            _cancelled(cancel)
            if proc.returncode:
                raise RuntimeError("环境准备命令失败：" + "\n".join(output)[-4000:])
            return "\n".join(output)
        finally:
            stopped.set()
            _stop(proc)
            proc.wait(timeout=10)
            reader.join(timeout=2)
            proc.stdout.close()

    def pip(self, python: Path, args: list[str]) -> list[str]:
        # --isolated also ignores a user's pip configuration. The cache stays in this operation.
        return [str(python), "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
                "--cache-dir", str(self.root / "cache" / "pip"), *args]


def _python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _source(runner: Runner, root: Path, engine: str, cancel, log) -> Path:
    if shutil.which("git", path=runner.env.get("PATH")) is None:
        raise PreparationError("git_missing", "未找到 Git，无法获取固定版本的训练器源码。")
    definition = source(engine)
    trainer = root / "trainer"
    hooks = root / "empty-hooks"
    hooks.mkdir()
    command = ["git", "-c", "core.autocrlf=false", "-c", f"core.hooksPath={hooks}", "-c", "credential.helper="]
    runner.run([*command, "init", str(trainer)], log, cancel, timeout=60)
    runner.run([*command, "-C", str(trainer), "fetch", "--depth=1", "--no-tags", definition["url"],
                definition["revision"]], log, cancel, timeout=900)
    runner.run([*command, "-C", str(trainer), "checkout", "--detach", "FETCH_HEAD"], log, cancel, timeout=120)
    if engine == "voxcpm1.5":
        from ypuddin.tts.core import validate_upstream
    else:
        from ypuddin.tts.gpt_sovits.core import validate_upstream
    validate_upstream(str(trainer))
    return trainer


class _HttpsRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None:
            raise ValueError("语音资源下载地址无效。")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(runner, item, cancel) -> Path:
    target = runner.root / "cache" / item["name"]
    for attempt in range(3):
        _cancelled(cancel)
        try:
            return _download_once(runner, item, target, cancel)
        except (OSError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt == 2:
                raise
            target.unlink(missing_ok=True)
            cancel.wait(0.25 * (attempt + 1))
    raise RuntimeError("语音资源下载失败。")


def _download_once(runner, item, target, cancel) -> Path:
    digest, count = hashlib.sha256(), 0
    request = urllib.request.Request(item["url"], headers={"User-Agent": "YPuddinTrainStudio"})
    with runner.policy.opener(_HttpsRedirect()).open(request, timeout=15) as response, target.open("xb") as out:
        while True:
            _cancelled(cancel)
            chunk = response.read(256 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > item["size"]:
                raise ValueError("语音资源大小与清单不符。")
            digest.update(chunk)
            out.write(chunk)
    if count != item["size"] or digest.hexdigest() != item["sha256"]:
        raise ValueError("语音资源下载校验失败。")
    return target


def _archive_path(root: Path, name: str) -> Path:
    value = name.rstrip("/")
    if (not value or "\\" in value or ":" in value or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("语音资源压缩包包含无效路径。")
    return root / value


def _extract(archive: Path, destination: Path, kind: str, cancel) -> None:
    budget, names = 128 * 1024 * 1024, set()
    if kind == "zip":
        with zipfile.ZipFile(archive) as stream:
            for entry in stream.infolist():
                _cancelled(cancel)
                target = _archive_path(destination, entry.filename)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode) or (mode and stat.S_IFMT(mode) not in {0, stat.S_IFDIR, stat.S_IFREG}):
                    raise ValueError("语音资源压缩包包含文件重定向。")
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                budget -= entry.file_size
                if budget < 0 or target in names or target.exists():
                    raise ValueError("语音资源压缩包大小或文件重复。")
                names.add(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(entry) as source_file, target.open("xb") as output:
                    _copy(source_file, output, entry.file_size, cancel)
    elif kind == "tar":
        with tarfile.open(archive, "r:gz") as stream:
            for entry in stream:
                _cancelled(cancel)
                target = _archive_path(destination, entry.name)
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                budget -= entry.size
                if not entry.isfile() or budget < 0 or target in names or target.exists():
                    raise ValueError("语音资源压缩包包含无效文件。")
                names.add(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(entry) as source_file, target.open("xb") as output:
                    _copy(source_file, output, entry.size, cancel)
    else:
        raise ValueError("不支持此资源压缩格式。")


def _copy(source_file, output, expected: int, cancel) -> None:
    copied = 0
    while True:
        _cancelled(cancel)
        chunk = source_file.read(min(1024 * 1024, expected - copied + 1))
        if not chunk:
            break
        copied += len(chunk)
        if copied > expected:
            raise ValueError("语音资源解压大小与清单不符。")
        output.write(chunk)
    if copied != expected:
        raise ValueError("语音资源文件不完整。")


def _write_marker(root: Path, trainer: Path, environment: dict) -> None:
    files = []
    for folder in environment.values():
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                              "sha256": _digest(path)})
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        path = trainer / "runtime" / name
        if path.is_file():
            files.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                          "sha256": _digest(path)})
    document = {"schema_version": 1, "recipe_revision": RECIPE_REVISION, "environment": environment,
                "files": files}
    (trainer / MARKER).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", "utf-8")
    resource_environment(trainer)


def _text_resources(runner, trainer, engine, cancel, log) -> dict:
    environment = {}
    if engine == "gpt-sovits-v5":
        folder = runner.root / "resources"
        folder.mkdir()
        for item in TEXT_RESOURCES:
            log(f"准备语言资源：{item['name']}")
            archive = _download(runner, item, cancel)
            _extract(archive, folder, item["kind"], cancel)
        environment = {"NLTK_DATA": "resources/nltk_data", "OPEN_JTALK_DICT_DIR": "resources/open_jtalk_dic_utf_8-1.11"}
    _write_marker(runner.root, trainer, environment)
    return environment


def _ffmpeg(runner, python, trainer, environment, cancel, log) -> None:
    result = runner.run([python, "-s", "-B", "-c",
                         "import imageio_ffmpeg; print('FFMPEG_PATH ' + imageio_ffmpeg.get_ffmpeg_exe())"],
                        lambda _: None, cancel, timeout=30)
    path = next((Path(line.removeprefix("FFMPEG_PATH ")) for line in result.splitlines()
                 if line.startswith("FFMPEG_PATH ")), None)
    if path is None or not path.is_file():
        raise ValueError("FFmpeg 安装文件缺失。")
    folder = runner.root / "bin"
    folder.mkdir(exist_ok=True)
    _regular(folder, directory=True)
    destination = folder / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if destination.exists() or destination.is_symlink():
        _regular(destination)
    shutil.copy2(path, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR)
    # The fixed upstream's speed filter addresses this path directly on every platform.
    runtime = trainer / "runtime"
    runtime.mkdir(exist_ok=True)
    _regular(runtime, directory=True)
    if (runtime / "ffmpeg.exe").exists() or (runtime / "ffmpeg.exe").is_symlink():
        _regular(runtime / "ffmpeg.exe")
    shutil.copy2(destination, runtime / "ffmpeg.exe")
    environment["PATH"] = "bin"
    _write_marker(runner.root, trainer, environment)
    log("FFmpeg 已准备。")


def _shared_ffmpeg(runner, trainer, environment, cancel, log) -> None:
    """Supply the Windows DLL build used by TorchCodec; a static CLI cannot satisfy it."""
    archive = _download(runner, WINDOWS_SHARED_FFMPEG, cancel)
    folder = runner.root / "bin"
    folder.mkdir()
    names = {"ffmpeg.exe", "ffprobe.exe", "avcodec-62.dll", "avdevice-62.dll", "avfilter-11.dll",
             "avformat-62.dll", "avutil-60.dll", "swresample-6.dll", "swscale-9.dll"}
    copied, total = set(), 0
    with zipfile.ZipFile(archive) as stream:
        for entry in stream.infolist():
            _cancelled(cancel)
            parts = PurePosixPath(entry.filename).parts
            _archive_path(runner.root, entry.filename)
            if len(parts) != 3 or parts[:2] != ("ffmpeg-8.0.1-full_build-shared", "bin") or parts[-1] not in names:
                continue
            if stat.S_ISLNK(entry.external_attr >> 16) or parts[-1] in copied:
                raise ValueError("FFmpeg 压缩包包含无效文件。")
            copied.add(parts[-1])
            total += entry.file_size
            if total > 384 * 1024 * 1024:
                raise ValueError("FFmpeg 压缩包大小超出限制。")
            with stream.open(entry) as source_file, (folder / parts[-1]).open("xb") as output:
                _copy(source_file, output, entry.file_size, cancel)
    if copied != names:
        raise ValueError("FFmpeg 共享库文件不完整。")
    environment["PATH"] = "bin"
    _write_marker(runner.root, trainer, environment)
    log("FFmpeg 共享库已准备。")


def _macos_ffmpeg(runner, trainer, environment, metadata, cancel, log) -> None:
    if metadata.get("platform") != "Darwin" or metadata.get("machine", "").lower() not in {"arm64", "aarch64"}:
        return
    version = platform.mac_ver()[0]
    if not version or Version(version) < Version("14"):
        return
    archive = _download(runner, MACOS_ARM64_FFMPEG, cancel)
    folder = runner.root / "resources" / "ffmpeg"
    folder.mkdir(parents=True)
    _regular(folder.parent, directory=True)
    _regular(folder, directory=True)
    copied, total = set(), 0
    with zipfile.ZipFile(archive) as stream:
        for entry in stream.infolist():
            _cancelled(cancel)
            _archive_path(runner.root, entry.filename)
            parts = PurePosixPath(entry.filename).parts
            if len(parts) != 3 or parts[:2] != ("av", ".dylibs") or parts[-1] not in MACOS_FFMPEG_LIBRARIES:
                continue
            if stat.S_ISLNK(entry.external_attr >> 16) or parts[-1] in copied:
                raise ValueError("FFmpeg 共享库压缩包包含无效文件。")
            copied.add(parts[-1])
            total += entry.file_size
            if total > 128 * 1024 * 1024:
                raise ValueError("FFmpeg 共享库大小超出限制。")
            with stream.open(entry) as source_file, (folder / parts[-1]).open("xb") as output:
                _copy(source_file, output, entry.file_size, cancel)
    if copied != MACOS_FFMPEG_LIBRARIES:
        raise ValueError("FFmpeg 共享库文件不完整。")
    for alias, original in MACOS_FFMPEG_ALIASES.items():
        _cancelled(cancel)
        shutil.copy2(folder / original, folder / alias)
    environment["DYLD_LIBRARY_PATH"] = "resources/ffmpeg"
    _write_marker(runner.root, trainer, environment)
    log("FFmpeg 共享库已准备。")


def _candidate_can_recheck(report: dict, engine: str, environment: dict) -> bool:
    if any(name != "trainer_path" for name in report.get("missing", [])):
        return False
    if not report.get("conflicts"):
        return True
    # The initial source-free check cannot yet expose managed macOS FFmpeg libraries.
    if (engine != "voxcpm1.5" or "DYLD_LIBRARY_PATH" not in environment
            or set(report["conflicts"]) != {"torchcodec"}):
        return False
    failures = report.get("issues", [])
    return (any(item.get("code") == "tts.environment.dependency_import" for item in failures)
            and all(item.get("code") == "tts.environment.source_missing"
                    or (item.get("code") == "tts.environment.dependency_import"
                        and item.get("details", {}).get("failure", {}).get("module") == "torchcodec")
                    for item in failures))


def _native_plan(engine: str, metadata: dict) -> tuple[str, list[str], list[str]]:
    if metadata.get("hip_runtime"):
        raise ValueError("当前语音训练器尚未提供 HIP 环境配方。")
    platform = metadata.get("platform")
    machine = metadata.get("machine", "").lower()
    if platform not in {"Windows", "Linux", "Darwin"} or machine not in {"amd64", "x86_64", "aarch64", "arm64"}:
        raise ValueError("当前系统尚未提供语音环境配方。")
    cuda = metadata.get("cuda_runtime")
    installed_torch = metadata.get("packages", {}).get("torch", "")
    if not cuda and not metadata.get("torch_version") and installed_torch:
        match = re.search(r"\+cu(\d{2})(\d+)", installed_torch)
        if match:
            cuda = f"{int(match[1])}.{int(match[2])}"
        elif "+cpu" not in installed_torch and platform != "Darwin":
            raise ValueError("无法确定现有 PyTorch 的运行后端，尚未选择安装包。")
    if cuda:
        version = Version(str(cuda))
        if version >= Version("12.8"):
            backend = "cu128"
        elif version >= Version("12.6"):
            backend = "cu126"
        elif version >= Version("11.8") and engine == "gpt-sovits-v5":
            backend = "cu118"
        else:
            raise ValueError("当前 CUDA 运行时没有兼容的语音依赖组合。")
    else:
        backend = "cpu"
    # The Windows GPT worker uses Gloo even for one GPU. 2.7.1 preserves the tested transport path.
    torch = "2.7.1" if engine == "gpt-sovits-v5" else "2.10.0"
    suffix = "" if platform == "Darwin" else "+" + backend
    packages = [f"torch=={torch}{suffix}", f"torchaudio=={torch}{suffix}"]
    constraints = [*packages]
    if engine == "gpt-sovits-v5":
        constraints += ["numpy==1.26.4", "transformers==4.57.6", "peft==0.17.1", "pydantic==2.10.6",
                        "torchmetrics==1.5.0", "huggingface-hub>=0.34,<1", "gradio==4.44.1"]
    else:
        constraints += ["torchcodec==0.10.0", "numpy>=1.26,<3", "transformers>=4.36.2,<5"]
    return backend, packages, constraints


def _install(runner, python, engine, metadata, context, cancel, log) -> None:
    backend, native, constraints = _native_plan(engine, metadata)
    constraint_file = runner.root / "constraints.txt"
    constraint_file.write_text("\n".join(constraints) + "\n", "utf-8")
    with context._settings_lock:
        sources = dict(context.settings().get("downloads", {}))
    options = probe_options(context)
    fallback = sources.get("fallback", True)
    indexes = pypi_sources(sources.get("pypi", "auto"), fallback, **options)
    common = ["install", "--prefer-binary", "--timeout", "30", "--retries", "2"]
    # Upgrade only the newly created environment's installer; deployment pip is never targeted.
    setuptools = "setuptools>=80,<81" if engine == "gpt-sovits-v5" else "setuptools>=69"
    run_sources(runner, [(url, runner.pip(python, [*common, "--index-url", url, "pip>=25.1", setuptools, "wheel"]))
                         for url in indexes], log, cancel)
    if metadata.get("platform") == "Darwin":
        native_sources = [("index-url", url) for url in indexes]
    else:
        native_sources = torch_sources(backend, sources.get("pytorch", "auto"), fallback, **options)
    commands = []
    for kind, url in native_sources:
        args = [*common, "--only-binary=:all:", "--constraint", str(constraint_file), f"--{kind}", url]
        if kind == "find-links":
            args += ["--index-url", indexes[0]]
        commands.append((url, runner.pip(python, [*args, *native])))
    run_sources(runner, commands, log, cancel)
    if (engine == "gpt-sovits-v5" and metadata.get("platform") == "Windows"
            and metadata.get("machine", "").lower() in {"amd64", "x86_64"}
            and metadata.get("python_version", "").startswith("3.12.")):
        wheels = [str(_download(runner, item, cancel)) for item in WINDOWS_GSV_WHEELS]
        run_sources(runner, [(url, runner.pip(python, [*common, "--index-url", url, "--constraint", str(constraint_file),
                                                       *wheels])) for url in indexes], log, cancel)
    # Constraints keep all transitive dependencies from replacing the selected Torch/audio pair.
    run_sources(runner, [(url, runner.pip(python, [*common, "--index-url", url, "--constraint", str(constraint_file),
                                                  *requirements(engine)])) for url in indexes], log, cancel)


def prepare(context, root: Path, engine: str, candidates: list[dict], cancel, log, phase) -> dict:
    """Return only a completely checked source/interpreter pair; partial work remains unregistered."""
    source(engine)
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    owned = ("trainer", "venv", "resolved.json", "empty-hooks", "bin", "resources", "cache", "probe")
    if root.resolve() != root or any((root / name).exists() or (root / name).is_symlink() for name in owned):
        raise ValueError("环境准备需要新的独立目录。")
    runner = Runner(context, root)
    phase("checking")
    reports = []
    for candidate in candidates:
        _cancelled(cancel)
        report = probe_environment(candidate["python_path"], engine, cancel=cancel, work_dir=root / "probe")
        reports.append((candidate, report))
    usable = [(candidate, report) for candidate, report in reports
              if report.get("python_version") and Version("3.10") <= Version(report["python_version"]) < Version("3.13")]
    if not usable:
        raise ValueError("没有可用的 Python 3.10 至 3.12 环境。")
    phase("source")
    trainer = _source(runner, root, engine, cancel, log)
    environment = _text_resources(runner, trainer, engine, cancel, log)
    if engine == "voxcpm1.5":
        _macos_ffmpeg(runner, trainer, environment, usable[0][1], cancel, log)
    for candidate, report in usable:
        if not _candidate_can_recheck(report, engine, environment):
            continue
        if engine == "gpt-sovits-v5":
            try:
                _ffmpeg(runner, candidate["python_path"], trainer, environment, cancel, log)
            except (OSError, ValueError, RuntimeError):
                continue
        checked = probe_environment(candidate["python_path"], engine, str(trainer), cancel=cancel, work_dir=root / "probe")
        if checked.get("state") != "ready":
            continue
        try:
            runner.run(runner.pip(Path(candidate["python_path"]), ["check"]), log, cancel, timeout=90)
        except RuntimeError:
            continue
        result = {"python_path": candidate["python_path"], "trainer_path": str(trainer),
                  "kind": candidate.get("kind", "deployment"), "probe": checked, "resources": environment}
        break
    else:
        phase("dependencies")
        candidate, metadata = usable[0]
        python = _python(root / "venv")
        runner.run([candidate["python_path"], "-I", "-B", "-m", "venv", str(root / "venv")], log, cancel, timeout=180)
        _install(runner, python, engine, metadata, context, cancel, log)
        if engine == "gpt-sovits-v5":
            _ffmpeg(runner, python, trainer, environment, cancel, log)
        elif metadata.get("platform") == "Windows" and metadata.get("machine", "").lower() in {"amd64", "x86_64"}:
            _shared_ffmpeg(runner, trainer, environment, cancel, log)
        phase("verifying")
        runner.run(runner.pip(python, ["check"]), log, cancel, timeout=90)
        checked = probe_environment(str(python), engine, str(trainer), cancel=cancel, work_dir=root / "probe")
        if checked.get("state") != "ready":
            messages = [item["message"] for item in checked.get("issues", [])]
            raise ValueError("语音环境检查未通过：" + "；".join(messages)[-5000:])
        result = {"python_path": str(python), "trainer_path": str(trainer), "kind": "managed", "probe": checked,
                  "resources": environment}
    _cancelled(cancel)
    phase("verifying")
    record = {"schema_version": 1, "recipe_revision": RECIPE_REVISION, "engine": engine, "source": source(engine),
              "python_path": result["python_path"], "packages": result["probe"]["packages"],
              "dependency_fingerprint": result["probe"]["dependency_fingerprint"], "resources": environment}
    (root / "resolved.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return result
