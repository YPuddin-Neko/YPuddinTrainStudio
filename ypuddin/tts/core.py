"""Validation and filesystem contracts shared by the TTS API and worker."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import TtsConfig
from .execution_config import TtsExecutionConfig, lora_settings, parse_execution_config

SAMPLE_RATE = 44_100
UPSTREAM_REVISION = "f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629"
UPSTREAM_FILES = {
    "scripts/train_voxcpm_finetune.py": "af231830f81bc967e3bc86e35ae1825c2255948dfc33ce6255359d8bb0482ac8",
    "src/voxcpm/core.py": "5116a96ed19e2e2b86a9bffabfb322e276dd33dc0bf4d0b189a1e4fd5d2cc109",
    "src/voxcpm/training/tracker.py": "28001dbdd830cdb5f419f24e46f0e05edbbf60ed3180bfd6025a6977def9ada5",
    "src/voxcpm/model/voxcpm.py": "4d13384ede9e8466a26e7a7808393023088985b11dc78dac5b5476e4a32ed8b2",
}
PATH_FIELDS = ("python_path", "trainer_path", "model_path", "train_manifest", "val_manifest")


def _path(value: str | Path, label: str, *, directory: bool = False) -> Path:
    if not str(value).strip():
        raise ValueError(f"请填写{label}。")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label}须为绝对路径：{path}")
    path = path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{label}不存在：{path}")
    return path


def audio_info(path: Path) -> dict[str, Any]:
    """Read all PCM WAV frames so truncated payloads cannot pass on header metadata alone."""
    if path.suffix.lower() != ".wav":
        raise ValueError(f"音频须为 44.1 kHz PCM WAV：{path}")
    try:
        with wave.open(str(path), "rb") as audio:
            channels, width, rate, frames = (
                audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
            )
            if channels != 1:
                raise ValueError(f"音频须为单声道 WAV，当前为 {channels} 声道：{path}")
            if rate != SAMPLE_RATE:
                raise ValueError(f"音频采样率须为 44100 Hz，当前为 {rate} Hz：{path}")
            if frames <= 0 or width not in {1, 2, 3, 4}:
                raise ValueError(f"音频没有有效 PCM 帧：{path}")
            read = 0
            while block := audio.readframes(65_536):
                read += len(block)
            if read != frames * channels * width:
                raise ValueError(f"音频数据不完整：{path}")
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError(f"无法读取 PCM WAV 音频 {path}：{exc}") from exc
    return {"duration": frames / rate, "sample_rate": rate, "frames": frames, "channels": channels}


def validate_manifest(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    allowed: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    source = _path(path, "JSONL 数据清单")
    if allowed and not allowed(source):
        raise ValueError(f"数据清单不在允许访问的目录中：{source}")
    rows: list[dict[str, Any]] = []
    total = 0.0
    warnings: list[str] = []
    short, long = 0, 0
    try:
        with source.open(encoding="utf-8-sig") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"第 {lineno} 行不是有效 JSON：{exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"第 {lineno} 行须为含 audio、text 的 JSON 对象。")
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ValueError(f"第 {lineno} 行的 text 不能为空。")
                normalized: dict[str, Any] = {"text": row["text"]}
                for key in ("audio", "ref_audio"):
                    value = row.get(key)
                    if key == "ref_audio" and value is None:
                        continue
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"第 {lineno} 行的 {key} 须为音频路径。")
                    audio = Path(value).expanduser()
                    audio = (audio if audio.is_absolute() else source.parent / audio).resolve()
                    if allowed and not allowed(audio):
                        raise ValueError(f"第 {lineno} 行的 {key} 不在允许访问的目录中：{audio}")
                    info = audio_info(audio)
                    normalized[key] = str(audio)
                    normalized["duration" if key == "audio" else "ref_duration"] = info["duration"]
                    if key == "audio":
                        total += info["duration"]
                        short += info["duration"] < 1
                        long += info["duration"] > 30
                dataset_id = row.get("dataset_id", 0)
                if type(dataset_id) is not int or dataset_id < 0:
                    raise ValueError(f"第 {lineno} 行的 dataset_id 须为非负整数。")
                normalized["dataset_id"] = dataset_id
                rows.append(normalized)
    except UnicodeError as exc:
        raise ValueError(f"JSONL 数据清单须使用 UTF-8 编码：{source}") from exc
    if not rows:
        raise ValueError("数据清单没有音频样本。")
    if short:
        warnings.append(f"有 {short} 条录音短于 1 秒，可能难以提供完整的发音上下文。")
    if long:
        warnings.append(f"有 {long} 条录音长于 30 秒，训练时会增加显存占用。")
    if output_path is not None:
        target = Path(output_path)
        if target.resolve() == source:
            raise ValueError("规范化数据清单不能覆盖原始清单。")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "samples": len(rows), "duration_seconds": total, "sample_rate": SAMPLE_RATE,
        "path": str(Path(output_path).resolve() if output_path else source), "warnings": warnings,
    }


def validate_upstream(path: str | Path) -> dict[str, Any]:
    root = _path(path, "VoxCPM 源码目录", directory=True)
    for relative, digest in UPSTREAM_FILES.items():
        source = root / relative
        if not source.is_file():
            raise ValueError(f"VoxCPM 源码缺少 {relative}。")
        contents = source.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(contents).hexdigest() != digest:
            raise ValueError(f"VoxCPM 源码版本不匹配，请切换到提交 {UPSTREAM_REVISION}。")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
    )
    if result.returncode or result.stdout.strip() != UPSTREAM_REVISION:
        raise ValueError(f"VoxCPM 源码须使用 Git 提交 {UPSTREAM_REVISION}。")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "conf"],
        capture_output=True, text=True, timeout=10,
    )
    if dirty.returncode or dirty.stdout.strip():
        raise ValueError("VoxCPM 的 src、scripts 或 conf 中存在未提交改动，请使用独立的原始源码目录。")
    return {"revision": UPSTREAM_REVISION, "path": str(root)}


def validate_model(path: str | Path) -> dict[str, Any]:
    root = _path(path, "VoxCPM 1.5 模型目录", directory=True)
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"模型目录缺少有效的 config.json：{root}") from exc
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("audio_vae_config"), dict)
        or config.get("architecture", "voxcpm") != "voxcpm"
        or config.get("audio_vae_config", {}).get("sample_rate") != SAMPLE_RATE
        or config.get("patch_size") != 4
    ):
        raise ValueError("模型须为 VoxCPM 1.5（voxcpm 架构、44100 Hz AudioVAE、patch_size=4）。")
    if config.get("device", "cuda") not in {"cuda", "cuda:0"}:
        raise ValueError("模型 config.json 的 device 须为 cuda 或 cuda:0，以使用队列分配的显卡。")
    for choices in (
        ("model.safetensors", "pytorch_model.bin"),
        ("audiovae.safetensors", "audiovae.pth"),
        ("tokenizer.json",), ("tokenizer_config.json",),
    ):
        if not any((root / name).is_file() and (root / name).stat().st_size > 0 for name in choices):
            raise ValueError(f"模型目录缺少 {' 或 '.join(choices)}：{root}")
    return {"path": str(root), "architecture": "voxcpm", "sample_rate": SAMPLE_RATE}


def worker_environment(config: TtsConfig | TtsExecutionConfig) -> dict[str, str]:
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(Path(config.trainer_path) / "src")])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def runtime_probe(
    config: TtsConfig | TtsExecutionConfig,
    *,
    mode: str = "train",
    gpu_devices: list[str] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    try:
        if mode not in {"train", "sample"}:
            raise ValueError("未知的 TTS 任务类型。")
        result = subprocess.run(
            [config.python_path, "-m", "ypuddin.tts.bridge", "--probe"],
            input=json.dumps({"trainer_path": config.trainer_path, "mode": mode, "gpu_devices": gpu_devices or []}),
            capture_output=True, text=True, timeout=90, env=worker_environment(config),
        )
        marker = next((line[14:] for line in reversed(result.stdout.splitlines()) if line.startswith("TTS_PREFLIGHT ")), None)
        if marker is None:
            raise ValueError(f"TTS Python 环境检查失败：{(result.stderr or result.stdout).strip()[-1500:]}")
        report = json.loads(marker)
        details = report["details"]
        errors.extend(report["errors"])
        if not details.get("prefix"):
            errors.append("TTS Python 环境检查未返回环境路径。")
        elif Path(details["prefix"]).resolve() == Path(sys.prefix).resolve():
            errors.append("请为 TTS 选择独立的 Python 环境，避免与图像训练依赖互相覆盖。")
        if result.returncode and not report["errors"]:
            errors.append(f"TTS Python 环境检查异常退出（{result.returncode}）。")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        errors.append(str(exc))
    return {"ok": not errors, "errors": errors, "details": details}


def preflight(
    config: TtsConfig | TtsExecutionConfig,
    *,
    check_runtime: bool = True,
    mode: str = "train",
    allowed: Callable[[Path], bool] | None = None,
    gpu_devices: list[str] | None = None,
) -> dict[str, Any]:
    if config.engine == "gpt-sovits-v5":
        from .gpt_sovits.core import preflight as inspect_gpt_sovits

        return inspect_gpt_sovits(config, check_runtime=check_runtime, mode=mode, allowed=allowed, gpu_devices=gpu_devices)
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"engine": config.engine, "sample_rate": SAMPLE_RATE}
    for key, call in (
        ("upstream", lambda: validate_upstream(config.trainer_path)),
        ("model", lambda: validate_model(config.model_path)),
    ):
        try:
            details[key] = call()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
    try:
        python = _path(config.python_path, "TTS Python")
        if os.name != "nt" and not os.access(python, os.X_OK):
            raise ValueError(f"TTS Python 没有执行权限：{python}")
    except ValueError as exc:
        errors.append(str(exc))
    if mode not in {"train", "sample"}:
        errors.append("未知的 TTS 任务类型。")
    if mode == "train":
        for key, path in (("dataset", config.train_manifest), ("validation_dataset", config.val_manifest)):
            if key == "validation_dataset" and not path:
                continue
            try:
                details[key] = validate_manifest(path, allowed=allowed)
                label = "训练集" if key == "dataset" else "验证集"
                warnings.extend(f"{label}：{message}" for message in details[key]["warnings"])
                if key == "dataset" and details[key]["samples"] < config.batch_size:
                    errors.append("训练样本数不能少于批大小；上游数据加载器会丢弃不完整批次。")
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
    if check_runtime and not errors:
        report = runtime_probe(config, mode=mode, gpu_devices=gpu_devices)
        if report["details"]:
            details["runtime"] = report["details"]
        errors.extend(report["errors"])
    details["runtime_checked"] = check_runtime and "runtime" in details
    return {"ok": not errors, "errors": errors, "warnings": warnings, "details": details}


def safe_checkpoint(output_dir: str | Path, checkpoint: str | Path) -> Path:
    root = Path(output_dir).expanduser().resolve()
    candidate = Path(checkpoint).expanduser()
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if candidate.is_file():
        if candidate.name not in {"lora_weights.safetensors", "lora_weights.ckpt"}:
            raise ValueError("请选择 LoRA 检查点目录或 lora_weights 权重文件。")
        candidate = candidate.parent
    if not candidate.is_dir() or not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("LoRA 检查点不在来源任务的输出目录中。")
    for name in ("lora_config.json", "lora_weights.safetensors", "lora_weights.ckpt"):
        asset = candidate / name
        if asset.exists() and not asset.resolve().is_relative_to(root):
            raise ValueError("LoRA 检查点中的文件指向来源任务输出目录之外。")
    if not (candidate / "lora_config.json").is_file() or not any(
        (candidate / name).is_file() and (candidate / name).stat().st_size > 0
        for name in ("lora_weights.safetensors", "lora_weights.ckpt")
    ):
        raise ValueError("LoRA 检查点须包含 lora_config.json 和 lora_weights 权重。")
    return candidate


def training_settings(config: TtsConfig | TtsExecutionConfig, output: Path, train: Path, validation: Path | None) -> dict[str, Any]:
    config = parse_execution_config(config)
    return {
        "pretrained_path": config.model_path, "train_manifest": str(train),
        "val_manifest": str(validation) if validation else "", "sample_rate": SAMPLE_RATE,
        "batch_size": config.batch_size, "grad_accum_steps": config.grad_accum_steps,
        "num_workers": config.num_workers, "preprocessing_num_workers": config.preprocessing_num_workers,
        "num_iters": config.num_iters, "log_interval": config.log_interval,
        "valid_interval": config.valid_interval if config.valid_interval is not None else config.save_interval,
        "save_interval": config.save_interval,
        "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps, "max_steps": config.max_steps or config.num_iters,
        "max_batch_tokens": config.max_batch_tokens, "max_grad_norm": config.max_grad_norm,
        "save_path": str(output), "tensorboard": str(output / "tensorboard"),
        "lambdas": {"loss/diff": config.loss_diff_weight, "loss/stop": config.loss_stop_weight},
        "lora": lora_settings(config),
    }
