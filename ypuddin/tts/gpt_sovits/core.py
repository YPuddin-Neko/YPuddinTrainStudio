"""Input identity and output pairing for the fixed GPT-SoVITS release."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import wave
from pathlib import Path

from .config import parse_config
from .upstream import FILES as UPSTREAM_FILES

UPSTREAM_REVISION = "f652b1da5af29a6955f9c3911aa71b7daa6618bc"
SAMPLE_RATE = 48000


def _path(value, label, *, directory=False):
    path = Path(value).expanduser()
    if not str(value).strip() or not path.is_absolute():
        raise ValueError(f"请填写{label}的绝对路径。")
    path = path.resolve(strict=True)
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{label}路径类型错误：{path}")
    return path


def validate_upstream(path):
    root = _path(path, "GPT-SoVITS 源码目录", directory=True)
    for name, expected in UPSTREAM_FILES.items():
        contents = (root / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError(f"GPT-SoVITS 源码不匹配：{name}；需要提交 {UPSTREAM_REVISION}。")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=10)
    if head.returncode or head.stdout.strip() != UPSTREAM_REVISION or dirty.returncode or dirty.stdout.strip():
        raise ValueError(f"GPT-SoVITS 需要干净的官方提交 {UPSTREAM_REVISION}。")
    return {"path": str(root), "revision": UPSTREAM_REVISION}


def asset_paths(config):
    config = parse_config(config)
    root = _path(config.model_path, "GPT-SoVITS 预训练模型目录", directory=True)
    assets = {
        "gpt": config.pretrained_gpt or str(root / "s1v3.ckpt"),
        "sovits": config.pretrained_sovits or str(root / "gsv-v5-pretrained" / f"s2G{config.variant}.pth"),
        "sovits_base": config.pretrained_sovits or str(root / "gsv-v5-pretrained" / f"s2G{config.variant}.pth"),
        "vocoder": str(root / "gsv-v5-pretrained/vocoder.pth"),
        "bert": str(root / "chinese-roberta-wwm-ext-large"),
        "hubert": str(root / "chinese-hubert-base"),
    }
    for name, value in assets.items():
        path = _path(value, name, directory=name in {"bert", "hubert"})
        if path.is_file() and not path.stat().st_size:
            raise ValueError(f"模型文件为空：{path}")
        assets[name] = str(path)
    for name in ("bert", "hubert"):
        folder = Path(assets[name])
        if not (folder / "config.json").is_file() or not any((folder / f).is_file() for f in ("pytorch_model.bin", "model.safetensors")):
            raise ValueError(f"{name} 目录需要 config.json 及模型权重。")
    if not (Path(assets["bert"]) / "vocab.txt").is_file():
        raise ValueError("BERT 目录缺少 vocab.txt。")
    return assets


def text_asset_roots(config):
    config = parse_config(config)
    models = Path(config.model_path).expanduser().resolve()
    trainer = Path(config.trainer_path).expanduser().resolve()
    marker = models / ".ypuddin-tts-package.json"
    try:
        info = marker.lstat()
    except FileNotFoundError:
        managed = False
    else:
        if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError(f"模型包标记必须是普通文件，不能包含文件重定向：{marker}")
        managed = True
    roots = {}
    for key, name, legacy in (("g2pw", "G2PWModel", "GPT_SoVITS/text/G2PWModel"),
                              ("fast_langdetect", "fast_langdetect", "GPT_SoVITS/pretrained_models/fast_langdetect")):
        packaged = models / name
        # Managed packages keep their resource roots even when a directory disappears.
        root = packaged if managed or packaged.exists() or packaged.is_symlink() else trainer / legacy
        if root.is_symlink() or root.resolve() != root:
            raise ValueError(f"模型资源不能包含文件重定向：{root}")
        if managed and not root.is_dir():
            raise ValueError(f"模型包资源目录不存在或不可用：{root}")
        if root.exists() and not root.is_dir():
            raise ValueError(f"模型资源需要目录：{root}")
        roots[key] = root
    return roots


def text_asset_files(root):
    root = Path(root)
    if root.is_symlink() or root.resolve() != root:
        raise ValueError(f"模型资源不能包含文件重定向：{root}")
    files = {}
    if root.is_dir():
        for entry in sorted(root.rglob("*")):
            if entry.is_symlink() or entry.resolve() != entry:
                raise ValueError(f"模型资源不能包含文件重定向：{entry}")
            if entry.is_file():
                files[str(entry.relative_to(root))] = entry
    return files


def model_identity(config):
    config = parse_config(config)
    paths = asset_paths(config)
    assets = {}
    for key, value in paths.items():
        path = Path(value)
        if path.is_dir():
            for entry in sorted(path.rglob("*")):
                if entry.is_file() and entry.suffix in {".json", ".txt", ".bin", ".safetensors", ".model"}:
                    if not entry.resolve().is_relative_to(path):
                        raise ValueError(f"模型文件不能重定向到目录之外：{entry}")
                    assets[f"{key}/{entry.relative_to(path)}"] = str(entry.resolve())
        else:
            assets[key] = str(path)
    for key, root in text_asset_roots(config).items():
        for name, entry in text_asset_files(root).items():
            assets[f"{key}/{name}"] = str(entry)
    return {"engine": config.engine, "variant": config.variant, "directory": str(Path(config.model_path).resolve()),
            "trainer_path": str(Path(config.trainer_path).resolve()), "pretrained_gpt": config.pretrained_gpt,
            "pretrained_sovits": config.pretrained_sovits, "assets": assets}


def require_text_assets(config, languages, *, inference=False):
    languages = set(languages)
    roots = text_asset_roots(config)
    if languages.intersection({"zh", "auto"}):
        root = roots["g2pw"]
        files = text_asset_files(root)
        onnx = next((name for name in ("g2pW.onnx", "g2pw.onnx") if name in files), "g2pW.onnx")
        for name in (onnx, "POLYPHONIC_CHARS.txt", "MONOPHONIC_CHARS.txt",
                     "bopomofo_to_pinyin_wo_tune_dict.json", "char_bopomofo_dict.json"):
            if name not in files or not files[name].stat().st_size:
                raise ValueError(f"中文或自动识别语言需要本地 G2PW 资源：{root / name}")
    if inference and languages.difference({"en"}):
        root = roots["fast_langdetect"]
        files = text_asset_files(root)
        for name in ("lid.176.bin", "lid.176.ftz"):
            if name not in files or not files[name].stat().st_size:
                raise ValueError(f"此语言的试听需要本地语言识别模型：{root / name}")


def validate_model_identity(frozen, model_path):
    current = model_identity({"engine": "gpt-sovits-v5", "variant": frozen["variant"],
                              "model_path": model_path, "trainer_path": frozen["trainer_path"],
                              "pretrained_gpt": frozen.get("pretrained_gpt", ""),
                              "pretrained_sovits": frozen.get("pretrained_sovits", "")})
    if frozen != current:
        raise ValueError("GPT-SoVITS 模型路径或资产集合在检查后改变。")
    return current


def audio_info(path):
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        rate, channels, frames, width = wav.getframerate(), wav.getnchannels(), wav.getnframes(), wav.getsampwidth()
        if channels != 1 or rate not in {32000, 44100, 48000} or frames <= 0 or width not in {1, 2, 3, 4}:
            raise ValueError("GPT-SoVITS 音频需要单声道 32/44.1/48 kHz PCM WAV。")
        count = 0
        while data := wav.readframes(65536):
            count += len(data)
        if count != frames * channels * width:
            raise ValueError("PCM WAV 数据不完整。")
    return {"duration": frames / rate, "sample_rate": rate, "frames": frames, "channels": channels}


def manifest_rows(source, *, allowed=None):
    source = _path(source, "JSONL 数据清单")
    if allowed and not allowed(source):
        raise ValueError("数据清单不在允许目录中。")
    rows = []
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("JSONL 每行需要非空 text。")
        if row.get("language") not in {"zh", "en", "ja", "ko", "yue"}:
            raise ValueError("每条数据必须明确 language：zh/en/ja/ko/yue。")
        speaker = row.get("speaker", "speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("speaker 必须为非空字符串。")
        if any(c in value for value in (row["text"], speaker) for c in "|\r\n\t\0"):
            raise ValueError("转写和说话人名称不能包含竖线、制表符或换行。")
        if not isinstance(row.get("audio"), str) or not row["audio"]:
            raise ValueError("每条数据需要音频路径。")
        audio = Path(row["audio"]).expanduser()
        audio = (audio if audio.is_absolute() else source.parent / audio).resolve(strict=True)
        if allowed and not allowed(audio):
            raise ValueError("音频不在允许目录中。")
        info = audio_info(audio)
        rows.append({"audio": str(audio), "text": row["text"], "language": row["language"],
                     "speaker": speaker, "duration": info["duration"]})
    if not rows:
        raise ValueError("训练集没有有效录音。")
    return rows


def worker_environment(config):
    env = dict(os.environ)
    trainer = Path(config.trainer_path)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[3]), str(trainer),
                                        str(trainer / "GPT_SoVITS"), str(trainer / "GPT_SoVITS/BigVGAN")])
    env.update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", bert_path=asset_paths(config)["bert"])
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    return env


def runtime_probe(config, *, mode="train", gpu_devices=None):
    from .runtime import run_probe
    try:
        details = run_probe(parse_config(config), gpu_devices=gpu_devices, mode=mode)
        errors = [issue["message"] for check in details["checks"].values() for issue in check["issues"]]
        return {"ok": not errors, "errors": errors, "warnings": [], "details": details["details"]}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "details": {}}


def preflight(config, *, check_runtime=True, mode="train", gpu_devices=None, allowed=None):
    config = parse_config(config)
    errors, details = [], {"engine": config.engine, "variant": config.variant, "sample_rate": SAMPLE_RATE}
    for key, check in (("upstream", lambda: validate_upstream(config.trainer_path)),
                       ("model", lambda: model_identity(config)),
                       ("python", lambda: str(_path(config.python_path, "Python")))):
        try:
            selected = {"upstream": config.trainer_path, "model": config.model_path, "python": config.python_path}[key]
            if allowed and not allowed(Path(selected).expanduser().resolve()):
                raise ValueError("路径不在允许访问的目录中。")
            details[key] = check()
            if key == "model" and allowed:
                for path in details[key]["assets"].values():
                    if not allowed(Path(path)):
                        raise ValueError(f"模型文件不在允许访问的目录中：{path}")
            if key == "python" and os.name != "nt" and not os.access(config.python_path, os.X_OK):
                raise ValueError("Python 没有执行权限。")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
    if mode not in {"train", "sample"}:
        errors.append("未知的语音任务类型。")
    if mode == "train":
        try:
            rows = manifest_rows(config.train_manifest, allowed=allowed)
            require_text_assets(config, {row["language"] for row in rows})
            details["dataset"] = {"samples": len(rows), "duration_seconds": sum(r["duration"] for r in rows), "warnings": []}
        except (OSError, ValueError, wave.Error) as exc:
            errors.append(str(exc))
        if config.val_manifest:
            errors.append("此 GPT-SoVITS 训练入口不使用独立验证集，请移除验证集登记。")
    if check_runtime and not errors:
        probe = runtime_probe(config, mode=mode, gpu_devices=gpu_devices)
        errors.extend(probe["errors"])
        details["runtime"] = probe["details"]
    return {"ok": not errors, "errors": errors, "warnings": [], "details": details}


def safe_checkpoint(output_root, path):
    root = Path(output_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate == root or not candidate.is_relative_to(root) or not candidate.is_dir():
        raise ValueError("GPT-SoVITS 检查点不在来源任务目录中。")
    metadata = candidate / "checkpoint.json"
    for name in ("checkpoint.json", "gpt.ckpt", "sovits.pth"):
        file = candidate / name
        if not file.is_file() or not file.stat().st_size or file.resolve().parent != candidate:
            raise ValueError("GPT-SoVITS 检查点必须包含本目录的 checkpoint.json、gpt.ckpt、sovits.pth。")
    value = json.loads(metadata.read_text(encoding="utf-8"))
    if value.get("engine") != "gpt-sovits-v5" or value.get("variant") not in {"v5dev", "v5turbo"}:
        raise ValueError("GPT-SoVITS 检查点模型身份无效。")
    for name in ("gpt.ckpt", "sovits.pth"):
        expected = value.get("files", {}).get(name, {})
        file = candidate / name
        if file.stat().st_size != expected.get("size") or file_digest(file) != expected.get("sha256"):
            raise ValueError(f"GPT-SoVITS 检查点文件已改变：{name}")
    return candidate


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def list_checkpoints(output_root):
    root = Path(output_root)
    items = []
    if not root.is_dir():
        return items
    for metadata in sorted(root.rglob("checkpoint.json")):
        try:
            path = safe_checkpoint(root, metadata)
            value = json.loads(metadata.read_text())
            items.append({"path": str(path), "name": path.name, "step": None,
                          "files": ["checkpoint.json", "gpt.ckpt", "sovits.pth"],
                          "mtime": metadata.stat().st_mtime, "engine": "gpt-sovits-v5",
                          "variant": value["variant"], "stage": value["stage"],
                          "gpt": value["gpt"], "sovits": value["sovits"]})
        except (OSError, ValueError, KeyError):
            continue
    return items
