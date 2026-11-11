"""Private KohakuGBC runtime and a credential-free, cancellable worker connection."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .vlm import VlmError, VlmStopped

GBC_REVISION = "d90f360dbf561f63270522e44fa0e74ebed59c12"
RUNTIME_IDENTITY = "kohakugbc-d90f360-data0.2.0-callback-v1"
MODEL_REVISIONS = {
    "detector": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
    "embedder": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}
_MODELS = {
    "detector": ("google/owlv2-base-patch16-ensemble", (
        "config.json", "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json", "model.safetensors",
    )),
    "embedder": ("sentence-transformers/all-MiniLM-L6-v2", (
        "config.json", "config_sentence_transformers.json", "sentence_bert_config.json", "modules.json",
        "1_Pooling/config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.txt", "model.safetensors",
    )),
}
_PREPARE_LOCK = threading.Lock()
_START_TIMEOUT = 180
_ANALYZE_TIMEOUT = 1800
_MAX_LINE = 8 * 1024 * 1024
_WORKER = Path(__file__).with_name("gbc_worker.py")


@dataclass(frozen=True)
class RuntimePaths:
    python: Path
    detector: Path
    embedder: Path
    identity: str = RUNTIME_IDENTITY


def _cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise VlmStopped("GBC 分析已取消。")


def _device(device: str) -> str:
    if device == "cpu" or re.fullmatch(r"cuda(?::\d+)?", device):
        return device
    raise VlmError("GBC 检测器需要 CPU 或 CUDA 设备。")


def _acquire(lock, cancel):
    while not lock.acquire(timeout=0.1):
        _cancelled(cancel)
    try:
        _cancelled(cancel)
    except BaseException:
        lock.release()
        raise


def _resource_stats(runtime: RuntimePaths) -> dict:
    paths = {"python": runtime.python}
    for kind, (_, names) in _MODELS.items():
        paths.update({f"{kind}/{name}": getattr(runtime, kind) / name for name in names})
    result = {}
    for name, path in paths.items():
        st = path.stat()
        if not path.is_file() or st.st_size <= 0:
            raise ValueError(f"GBC 资源不完整：{name}")
        result[name] = [st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    return result


def prepare(context, *, device: str, progress: Callable[[str], None], cancel: threading.Event) -> RuntimePaths:
    """Prepare only the managed GBC environment; installation never targets the server Python."""
    from ypuddin.package_sources import pypi_sources, run_sources, torch_sources

    from .download_sources import probe_options
    from .tts_environment_install import Runner

    device = _device(device)
    settings = context.settings()
    base = Path(context.data_root) / "environments" / "gbc"
    root = base / RUNTIME_IDENTITY / ("cuda" if device.startswith("cuda") else "cpu")
    models = Path(settings["paths"]["models_dir"]).expanduser() / "gbc"
    runtime = RuntimePaths(
        root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        models / f"owlv2-base-patch16-ensemble-{MODEL_REVISIONS['detector'][:12]}",
        models / f"all-MiniLM-L6-v2-{MODEL_REVISIONS['embedder'][:12]}",
    )
    marker = root / "ready.json"
    _acquire(_PREPARE_LOCK, cancel)
    try:
        try:
            ready = json.loads(marker.read_text("utf-8"))
            if ready.get("identity") == RUNTIME_IDENTITY and ready.get("files") == _resource_stats(runtime):
                return runtime
        except (OSError, ValueError, TypeError):
            pass
        if sys.version_info < (3, 11):
            raise VlmError("GBC 环境准备需要 Python 3.11 或更新版本。")
        root.mkdir(parents=True, exist_ok=True)
        marker.unlink(missing_ok=True)
        runner = Runner(context, root)
        # Public fixed resources need neither the server's VLM keys nor Hugging Face login.
        for name in tuple(runner.env):
            if any(word in name.upper() for word in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")):
                runner.env.pop(name, None)
        runner.env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        progress("正在准备 GBC 运行环境")
        if not runtime.python.is_file():
            runner.run([sys.executable, "-m", "venv", str(root / "venv")], progress, cancel, timeout=120)
        sources = dict(settings.get("downloads", {}))
        options = probe_options(context)
        fallback = sources.get("fallback", True)
        indexes = pypi_sources(sources.get("pypi", "auto"), fallback, **options)
        native = ([("index-url", url) for url in indexes] if sys.platform == "darwin" else
                  torch_sources("cu128" if device.startswith("cuda") else "cpu",
                                sources.get("pytorch", "auto"), fallback, **options))
        common = ["install", "--prefer-binary", "--timeout", "30", "--retries", "2"]
        run_sources(runner, [(url, runner.pip(runtime.python, [*common, "--only-binary=:all:",
                    f"--{kind}", url, *(["--index-url", indexes[0]] if kind == "find-links" else []),
                    "torch==2.15.0", "torchvision==0.29.2"])) for kind, url in native], progress, cancel)
        # These are the package/API versions used by the fixed GBC adapter, in its own venv.
        packages = [
            f"kohakugbc[torch] @ https://github.com/KohakuBlueleaf/KohakuGBC/archive/{GBC_REVISION}.zip",
            "kohakugbc-data==0.2.0", "togeari==0.3.0", "transformers==5.22.1",
            "sentence-transformers==6.2.0", "torch==2.15.0", "torchvision==0.29.2",
        ]
        run_sources(runner, [(url, runner.pip(runtime.python, [*common, "--index-url", url,
                    "--only-binary=kohakugbc-data", *packages])) for url in indexes], progress, cancel)
        progress("正在准备 GBC 检测与嵌入模型")
        manifest = [{"repo": repo, "revision": MODEL_REVISIONS[kind],
                     "path": str(getattr(runtime, kind)), "files": names}
                    for kind, (repo, names) in _MODELS.items()]
        code = (
            "import json,sys; from huggingface_hub import snapshot_download; "
            "[(print(x['repo'],flush=True), snapshot_download(x['repo'],revision=x['revision'],"
            "local_dir=x['path'],allow_patterns=x['files'],token=False)) for x in json.loads(sys.argv[1])]"
        )
        runner.run([runtime.python, "-c", code, json.dumps(manifest)], progress, cancel, timeout=5400)
        runner.run([runtime.python, "-B", _WORKER, "--check"], progress, cancel, timeout=120,
                   extra_env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        _cancelled(cancel)
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": RUNTIME_IDENTITY, "files": _resource_stats(runtime)}), "utf-8")
        temporary.replace(marker)
        progress("GBC 运行环境已就绪")
        return runtime
    except InterruptedError as exc:
        raise VlmStopped("GBC 环境准备已取消。") from exc
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        raise VlmError(f"GBC 环境准备失败：{exc}") from exc
    finally:
        _PREPARE_LOCK.release()


def _worker_env(work: Path, device: str) -> dict[str, str]:
    # Platform/runtime discovery only: no service credentials or network configuration.
    names = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
             "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"}
    env = {key: value for key, value in os.environ.items() if key.upper() in names}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", KOHAKUGBC_PIXEL_CACHE=str(work / "pixels"))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("TMP", "TEMP", "TMPDIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        folder = work / "cache" / name.lower()
        folder.mkdir(parents=True, exist_ok=True)
        env[name] = str(folder)
    return env


class Analyzer:
    """One owned process and one loaded detector/embedder, reused for successive images."""

    def __init__(self, runtime: RuntimePaths, *, device: str, work: Path, cancel: threading.Event):
        self.runtime, self.device = runtime, _device(device)
        self.work, self.cancel = Path(work).resolve(), cancel
        self._lock = threading.Lock()
        self._proc = None
        self._messages = queue.Queue()
        self._errors = deque(maxlen=30)
        self._threads = []
        self._serial = 0
        self._closed = False

    def __enter__(self):
        _cancelled(self.cancel)
        if self._closed:
            raise VlmError("GBC 进程连接已关闭。")
        if self._proc is not None:
            return self
        self.work.mkdir(parents=True, exist_ok=True)
        try:
            self._proc = subprocess.Popen(
                [str(self.runtime.python), "-B", str(_WORKER), "--detector", str(self.runtime.detector),
                 "--embedder", str(self.runtime.embedder), "--device", self.device, "--work", str(self.work)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self.work, env=_worker_env(self.work, self.device), start_new_session=os.name != "nt",
            )
            for stream, output in ((self._proc.stdout, self._messages), (self._proc.stderr, None)):
                thread = threading.Thread(target=self._read, args=(stream, output), daemon=True)
                thread.start()
                self._threads.append(thread)
            message = self._receive(time.monotonic() + _START_TIMEOUT)
            if message.get("type") != "ready":
                raise VlmError("GBC 进程未能完成初始化。")
            (self.work / "worker-ready.json").write_text(json.dumps(message, ensure_ascii=False), "utf-8")
            return self
        except OSError as exc:
            self._shutdown()
            raise VlmError(f"无法启动 GBC 进程：{exc}") from exc
        except BaseException:
            self._shutdown()
            raise

    def _read(self, stream, output):
        try:
            while line := stream.readline(_MAX_LINE + 1):
                if len(line) > _MAX_LINE:
                    if output is not None:
                        output.put({"type": "error", "message": "GBC 返回内容超过大小限制。"})
                    return
                text = line.decode("utf-8", "replace").strip()
                if output is None:
                    self._errors.append(text[-1000:])
                    with (self.work / "worker.stderr.log").open("a", encoding="utf-8") as log:
                        log.write(text + "\n")
                else:
                    try:
                        message = json.loads(text)
                        if not isinstance(message, dict):
                            raise ValueError("expected object")
                        output.put(message)
                    except ValueError:
                        output.put({"type": "error", "message": "GBC 进程返回了无效消息。"})
                        return
        finally:
            if output is not None:
                output.put({"type": "exited"})

    def _receive(self, deadline):
        while True:
            _cancelled(self.cancel)
            if time.monotonic() >= deadline:
                raise VlmError("GBC 分析超时。")
            try:
                message = self._messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.get("type") in {"error", "exited"}:
                detail = message.get("message") or "\n".join(self._errors)[-2000:] or "进程已退出"
                raise VlmError(f"GBC 分析失败：{detail}")
            return message

    def _send(self, message):
        try:
            self._proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (OSError, ValueError, AttributeError) as exc:
            raise VlmError("GBC 进程连接已关闭。") from exc

    def analyze(self, image: Path, *, max_nodes: int, max_depth: int, request: Callable[[Path, str], str]) -> dict:
        _acquire(self._lock, self.cancel)
        try:
            if self._proc is None:
                self.__enter__()
            if not self._proc or self._proc.poll() is not None:
                raise VlmError("GBC 进程尚未启动或已经退出。")
            if not 1 <= max_nodes <= 64 or not 0 <= max_depth <= 5:
                raise VlmError("GBC 节点数或深度超出范围。")
            self._serial += 1
            self._send({"type": "analyze", "id": self._serial, "image": str(Path(image).resolve()),
                        "max_nodes": max_nodes, "max_depth": max_depth})
            deadline = time.monotonic() + _ANALYZE_TIMEOUT
            while True:
                message = self._receive(deadline)
                if message.get("type") == "request":
                    path = Path(message["image"]).resolve()
                    if not path.is_relative_to(self.work) or not path.is_file():
                        raise VlmError("GBC 请求图片路径无效。")
                    # Preserve VlmFatal, request budget failures and VlmStopped exactly.
                    requested_at = time.monotonic()
                    reply = request(path, message["prompt"])
                    deadline += time.monotonic() - requested_at
                    _cancelled(self.cancel)
                    self._send({"type": "reply", "id": message["id"], "text": reply})
                elif message.get("type") == "graph" and message.get("id") == self._serial:
                    graph = message.get("graph")
                    if not isinstance(graph, dict) or not graph.get("vertices"):
                        raise VlmError("GBC 未返回有效图结构。")
                    return graph
                else:
                    raise VlmError("GBC 进程消息顺序无效。")
        except BaseException:
            self._shutdown()
            raise
        finally:
            self._lock.release()

    def close(self):
        self._closed = True
        self._shutdown()

    def _shutdown(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        from .tts_environment_install import _stop

        if proc.poll() is None:
            try:
                proc.stdin.write(b'{"type":"close"}\n')
                proc.stdin.flush()
                proc.wait(timeout=2)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                _stop(proc)
        proc.wait(timeout=10)
        for thread in self._threads:
            thread.join(timeout=2)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()
        self._threads.clear()
        self._messages = queue.Queue()
        self._errors.clear()

    def __exit__(self, *_):
        self.close()
