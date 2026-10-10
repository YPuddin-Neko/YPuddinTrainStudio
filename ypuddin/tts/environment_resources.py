"""Verify and expose resources belonging to a prepared speech environment."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

MARKER = ".ypuddin-tts-environment.json"
_KEYS = {"NLTK_DATA", "OPEN_JTALK_DICT_DIR", "PATH", "DYLD_LIBRARY_PATH"}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, directory: bool = False) -> None:
    info = path.lstat()
    if (not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or getattr(info, "st_file_attributes", 0) & 0x400):
        raise ValueError("语音环境资源包含文件重定向。")


def _relative(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute() or any(p in {"", ".", ".."} for p in value.split("/"))
            or any(ord(c) < 32 for c in value)):
        raise ValueError("语音环境资源路径无效。")
    return value


def _path(root: Path, value: str, *, directory=False) -> Path:
    path = root / _relative(value)
    for parent in reversed(path.parents):
        if parent == root or root in parent.parents:
            _regular(parent, directory=True)
    _regular(path, directory=directory)
    if path.resolve() != path or not path.is_relative_to(root):
        raise ValueError("语音环境资源不在准备目录中。")
    return path


def marker_identity(trainer_path: str | Path) -> str | None:
    marker = Path(trainer_path) / MARKER
    try:
        _regular(marker)
    except FileNotFoundError:
        return None
    if marker.stat().st_size > 1024 * 1024:
        raise ValueError("语音环境资源清单过大。")
    return hashlib.sha256(marker.read_bytes()).hexdigest()


def resource_environment(trainer_path: str | Path, *, expected_sha256: str | None = None) -> dict[str, str]:
    trainer = Path(trainer_path).absolute()
    identity = marker_identity(trainer)
    if expected_sha256 is not None and identity != expected_sha256:
        raise ValueError("语音环境资源清单在检查后改变。")
    if identity is None:
        return {}
    root = trainer.parent
    _regular(root, directory=True)
    _regular(trainer, directory=True)
    if root.resolve() != root or trainer.resolve() != trainer:
        raise ValueError("语音环境准备目录包含文件重定向。")
    raw = (trainer / MARKER).read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity:
        raise ValueError("语音环境资源清单在检查后改变。")
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("schema_version") != 1 or not isinstance(document.get("environment"), dict):
        raise ValueError("语音环境资源清单无效。")
    environment = document["environment"]
    if set(environment) - _KEYS or not isinstance(document.get("files"), list):
        raise ValueError("语音环境资源清单包含未知字段。")
    roots = {}
    for key, value in environment.items():
        if not isinstance(value, str) or (key == "PATH" and value != "bin") or (key != "PATH" and not value.startswith("resources/")):
            raise ValueError("语音环境资源路径无效。")
        if key == "DYLD_LIBRARY_PATH" and value != "resources/ffmpeg":
            raise ValueError("语音共享库路径无效。")
        roots[key] = _path(root, value, directory=True)
    recorded = set()
    aliases = {trainer / "runtime" / name for name in ("ffmpeg.exe", "ffprobe.exe")}
    for item in document["files"]:
        if (not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}
                or not isinstance(item["size"], int) or isinstance(item["size"], bool) or item["size"] < 0
                or not isinstance(item["sha256"], str)):
            raise ValueError("语音环境资源清单无效。")
        path = _path(root, item["path"])
        if ((not any(path.is_relative_to(folder) for folder in roots.values()) and path not in aliases) or item["path"] in recorded
                or path.stat().st_size != item["size"]
                or _digest(path) != item["sha256"]):
            raise ValueError("语音环境资源校验失败。")
        recorded.add(item["path"])
    actual = set()
    for path in aliases:
        if path.exists():
            _regular(path)
            actual.add(path.relative_to(root).as_posix())
    for folder in roots.values():
        for path in folder.rglob("*"):
            if path.is_dir():
                _regular(path, directory=True)
            else:
                _regular(path)
                actual.add(path.relative_to(root).as_posix())
    if actual != recorded:
        raise ValueError("语音环境资源文件集合发生变化。")
    result = {key: str(value) for key, value in roots.items()}
    if "PATH" in result:
        result["PATH"] += os.pathsep + os.environ.get("PATH", "")
    return result


_RESOURCE_CHECK = r'''
import importlib.util
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import wave
from pathlib import Path

root = Path.cwd().resolve()
null_device = os.devnull
null_device_name = os.path.normcase(null_device)
def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "urllib.Request"}:
        raise RuntimeError("Resource checks cannot use the network")
    if event == "open":
        name, mode, flags = args
        if (isinstance(name, (str, bytes, os.PathLike))
                and os.path.normcase(os.fsdecode(name)) == null_device_name
                and (os.name == "nt" or stat.S_ISCHR(os.stat(null_device).st_mode))):
            return
        if isinstance(name, (str, bytes, os.PathLike)) and ((mode and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            if not Path(os.fsdecode(name)).resolve().is_relative_to(root):
                raise RuntimeError("Resource checks cannot change the selected environment")
sys.addaudithook(audit)
result = {"state": "ready", "ffmpeg": None, "text_resources": {}, "issues": []}
def failure(code, message):
    result["state"] = "missing"
    result["issues"].append({"code": "tts.environment." + code, "loc": ["environment", "resources"],
                             "message": message, "severity": "error", "details": {}})
wav = root / "audio.wav"
with wave.open(str(wav), "wb") as stream:
    stream.setnchannels(1)
    stream.setsampwidth(2)
    stream.setframerate(16000)
    stream.writeframes(b"\0\0" * 1600)
if sys.argv[1] == "voxcpm1.5":
    try:
        from torchcodec.decoders import AudioDecoder
        decoded = AudioDecoder(str(wav)).get_all_samples()
        if decoded.sample_rate != 16000 or decoded.data.numel() != 1600:
            raise ValueError("Decoded sample count differs")
        result["ffmpeg"] = "torchcodec_audio_decode"
    except Exception as exc:
        failure("audio_decode", "音频解码检查失败：" + str(exc)[-1200:])
else:
    executable = shutil.which("ffmpeg")
    if executable is None:
        failure("ffmpeg_missing", "尚未准备 FFmpeg。")
    else:
        try:
            decoded = subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(wav),
                                      "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
                                     capture_output=True, check=True, timeout=15)
            if decoded.stdout != b"\0\0" * 1600:
                raise ValueError("Decoded sample count differs")
            result["ffmpeg"] = executable
        except Exception as exc:
            failure("audio_decode", "FFmpeg 音频解码检查失败：" + str(exc)[-800:])
    if (Path(sys.argv[2]) / ".ypuddin-tts-environment.json").is_file():
        executable = Path(sys.argv[2]) / "runtime" / "ffmpeg.exe"
        try:
            decoded = subprocess.run([str(executable), "-nostdin", "-v", "error", "-i", str(wav),
                                      "-af", "atempo=1.0", "-f", "s16le", "pipe:1"],
                                     capture_output=True, check=True, timeout=15)
            if not decoded.stdout:
                raise ValueError("Speed filter returned no audio")
        except Exception as exc:
            failure("speed_filter", "试听变速音频检查失败：" + str(exc)[-800:])
    try:
        import nltk
        for key in ("corpora/cmudict", "taggers/averaged_perceptron_tagger_eng"):
            result["text_resources"][key] = str(nltk.data.find(key))
        from g2p_en import G2p
        if not G2p()("hello"):
            raise ValueError("English phoneme conversion returned no tokens")
    except Exception as exc:
        failure("nltk_missing", "英语词典尚未准备：" + str(exc)[-800:])
    try:
        dictionary = os.environ.get("OPEN_JTALK_DICT_DIR")
        if dictionary:
            folder = Path(dictionary)
        else:
            spec = importlib.util.find_spec("pyopenjtalk")
            folder = Path(spec.origin).parent / "open_jtalk_dic_utf_8-1.11"
        for name in ("sys.dic", "char.bin", "unk.dic", "matrix.bin"):
            if not (folder / name).is_file():
                raise FileNotFoundError(name)
        import pyopenjtalk
        if not pyopenjtalk.g2p("こんにちは"):
            raise ValueError("Japanese phoneme conversion returned no tokens")
        result["text_resources"]["open_jtalk"] = str(folder)
    except Exception as exc:
        failure("japanese_dictionary_missing", "日语词典尚未准备：" + str(exc)[-800:])
print("TTS_RESOURCE_CHECK " + json.dumps(result, ensure_ascii=False))
'''


def verify_runtime_resources(python_path: str, engine: str, trainer_path: str | Path, *, timeout: float = 45,
                             cancel=None, work_dir: Path | None = None) -> dict:
    """Decode a generated PCM clip and inspect text data without downloading or loading weights."""
    import subprocess
    import tempfile
    import time

    from .environment_probe import _cancelled, _stop

    def failed(code, message):
        return {"state": "error", "ffmpeg": None, "text_resources": {}, "issues": [{
            "code": f"tts.environment.{code}", "loc": ["environment", "resources"],
            "message": message, "severity": "error", "details": {},
        }]}

    if engine not in {"voxcpm1.5", "gpt-sovits-v5"}:
        return failed("engine", "不支持此语音引擎。")
    if _cancelled(cancel):
        return failed("cancelled", "环境检查已取消。")
    try:
        additions = resource_environment(trainer_path)
        if work_dir:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tts-resource-check-", dir=work_dir) as folder:
            env = dict(os.environ)
            for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
                env.pop(key, None)
            env.update(additions)
            env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            for key in ("TMP", "TEMP", "TMPDIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR",
                        "NUMBA_CACHE_DIR"):
                env[key] = str(Path(folder) / key.lower())
                Path(env[key]).mkdir()
            proc = subprocess.Popen([python_path, "-s", "-B", "-c", _RESOURCE_CHECK, engine, str(trainer_path)], env=env,
                                    cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    encoding="utf-8", errors="replace",
                                    start_new_session=os.name != "nt")
            started = time.monotonic()
            try:
                while True:
                    if _cancelled(cancel):
                        return failed("cancelled", "环境检查已取消。")
                    if time.monotonic() - started > timeout:
                        return failed("timeout", "语音资源检查超时。")
                    try:
                        stdout, stderr = proc.communicate(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                payload = next((line.removeprefix("TTS_RESOURCE_CHECK ") for line in reversed(stdout.splitlines())
                                if line.startswith("TTS_RESOURCE_CHECK ")), None)
                if proc.returncode or payload is None:
                    return failed("resource_probe", "语音资源检查失败：" + (stderr or stdout)[-1200:])
                return json.loads(payload)
            finally:
                _stop(proc)
                proc.communicate(timeout=10)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return failed("resource_probe", "语音资源检查失败：" + str(exc)[-1200:])
