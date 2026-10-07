"""Isolated runtime inspection and exact pinned-upstream sample-length statistics."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .core import (
    SAMPLE_RATE,
    UPSTREAM_REVISION,
    _path,
    validate_manifest,
    validate_model,
    validate_upstream,
    worker_environment,
)
from .devices import inspect_devices, selection_issues
from .execution_config import TtsExecutionConfig, lora_settings, parse_execution_config
from .issues import TtsIssue
from .version_config import TtsVersionConfig

CHECKS = ("python", "upstream", "model", "dependencies", "cuda", "bf16", "tokenizer", "lora_targets")
METHOD = "voxcpm_pinned_estimated_sequence_length"
MARKER = "TTS_RUNTIME "
MODEL_ASSETS = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "tokenizer.model", "model.safetensors", "pytorch_model.bin", "audiovae.safetensors", "audiovae.pth",
)


def _issue(code: str, loc: list, message: str, **details: Any) -> dict:
    return TtsIssue(code=code, loc=loc, message=message, details=details).model_dump()


def _check(key: str, state: str = "unchecked", message: str | None = None, code: str | None = None) -> dict:
    return {
        "key": key, "state": state, "checked_at": None if state == "unchecked" else time.time(),
        "issues": [_issue(code or f"tts.environment.{key}", ["environment", key], message)] if message else [],
    }


def _environment(checks: dict[str, dict]) -> dict:
    values = [checks[key] for key in CHECKS]
    states = {item["state"] for item in values}
    state = "unavailable" if "unavailable" in states else "unchecked" if "unchecked" in states else "available"
    return {"state": state, "checked_at": time.time(), "checks": values}


def _filter(config: TtsExecutionConfig, split: str, present: bool) -> dict:
    return {
        "state": "blocked" if split == "train" or present else "not_applicable",
        "method": METHOD, "max_batch_tokens": config.max_batch_tokens,
        "max_sample_tokens": config.max_batch_tokens // config.batch_size
        if split == "train" and config.max_batch_tokens > 0 else None,
        "before_count": None, "kept_count": None, "filtered_count": None,
        "input_fingerprint": None, "issues": [],
    }


def fingerprint_file(path: str | Path, *, allowed: Callable[[Path], bool] | None = None) -> dict:
    source = Path(path).expanduser().resolve(strict=True)
    if allowed is not None and not allowed(source):
        raise ValueError(f"文件不在允许访问的目录中：{source}")
    before = source.stat()
    if not source.is_file():
        raise ValueError(f"训练输入须为普通文件：{source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"文件在检查期间发生变化：{source}")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = source.stat()

    def identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError(f"文件在检查期间发生变化：{source}")
    return {"path": str(source), "size": after.st_size, "sha256": digest.hexdigest()}


def _model_identity(path: str) -> dict:
    model = _path(path, "VoxCPM 1.5 模型目录", directory=True)
    validate_model(model)
    return {
        "directory": str(model),
        "assets": {name: str((model / name).resolve(strict=True)) for name in MODEL_ASSETS if (model / name).exists()},
    }


def verify_input_snapshot(inputs: dict[str, Any], *, manifests: tuple[str, ...] = (), model_path: str | None = None) -> None:
    """Verify the content identities captured at enqueue, before opening the trainer."""
    if not isinstance(inputs, dict) or not isinstance(inputs.get("fingerprints"), list) or not inputs["fingerprints"]:
        raise ValueError("语音任务缺少训练输入内容指纹。")
    seen = set()
    for expected in inputs["fingerprints"]:
        if (
            not isinstance(expected, dict) or not isinstance(expected.get("path"), str)
            or type(expected.get("size")) is not int or expected["size"] < 0
            or not isinstance(expected.get("sha256"), str) or len(expected["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in expected["sha256"])
        ):
            raise ValueError("语音任务的训练输入内容指纹无效。")
        path = Path(expected["path"])
        if not path.is_absolute() or path in seen or path.resolve(strict=True) != path:
            raise ValueError(f"语音训练输入路径已改变或重复：{path}")
        seen.add(path)
        actual = fingerprint_file(path)
        if actual != {key: expected[key] for key in ("path", "size", "sha256")}:
            raise ValueError(f"语音训练输入在任务排队后发生变化：{path}")
    for value in manifests:
        manifest = Path(value)
        if manifest not in seen:
            raise ValueError(f"语音任务的内容指纹未覆盖数据清单：{manifest}")
        for line in manifest.read_text(encoding="utf-8-sig").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"语音数据清单须包含 JSON 对象：{manifest}")
            for key in ("audio", "ref_audio"):
                if row.get(key):
                    audio = Path(row[key]).expanduser()
                    audio = (audio if audio.is_absolute() else manifest.parent / audio).resolve(strict=True)
                    if audio not in seen:
                        raise ValueError(f"语音任务的内容指纹未覆盖音频：{audio}")
    if "model_identity" in inputs and (model_path is None or not isinstance(inputs["model_identity"], dict)):
        raise ValueError("语音任务缺少有效的模型目录与资产身份。")
    if model_path is not None:
        frozen = inputs.get("model_identity", {})
        if frozen.get("engine") == "gpt-sovits-v5":
            from .gpt_sovits.core import validate_model_identity

            current = validate_model_identity(frozen, model_path)
        else:
            current = _model_identity(model_path)
        if "model_identity" in inputs and inputs["model_identity"] != current:
            raise ValueError("语音模型目录或逻辑资产路径在检查后发生变化。")
        for asset in current["assets"].values():
            if Path(asset) not in seen:
                raise ValueError(f"语音模型资产在检查后发生变化或未登记内容指纹：{asset}")


def validate_lora_parameters(model: Any, settings: dict[str, Any]) -> dict:
    """Inspect the initialized model without allocating parameters or consuming random numbers."""
    parameters = dict(model.named_parameters())
    result = {}
    for component, prefixes, targets in (
        ("lm", ("base_lm.", "residual_lm."), settings.get("target_modules_lm", ["q_proj", "v_proj", "k_proj", "o_proj"])),
        ("dit", ("feat_decoder.estimator.",), settings.get("target_modules_dit", ["q_proj", "v_proj", "k_proj", "o_proj"])),
        ("proj", ("",), settings.get("target_proj_modules", ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"])),
    ):
        if not settings.get(f"enable_{component}", False):
            continue
        matched = {}
        for target in targets:
            pairs = {}
            for name, parameter in parameters.items():
                parts = name.rsplit(".", 2)
                if len(parts) < 2 or parts[-2] != target or parts[-1] not in {"lora_A", "lora_B"}:
                    continue
                if component == "proj" and name.rsplit(".", 1)[0] != target:
                    continue
                if not name.startswith(prefixes) or not parameter.requires_grad or parameter.numel() <= 0:
                    continue
                pairs.setdefault(name.rsplit(".", 1)[0], {})[parts[-1]] = parameter.numel()
            complete = [pair for pair in pairs.values() if set(pair) == {"lora_A", "lora_B"}]
            if not complete:
                raise ValueError(f"LoRA {component} 的目标 {target} 未匹配到可训练参数。")
            matched[target] = {"modules": len(complete), "parameters": sum(sum(pair.values()) for pair in complete)}
        if not matched:
            raise ValueError(f"LoRA {component} 没有可训练目标。")
        result[component] = matched
    if not result:
        raise ValueError("至少启用一个可训练的 LoRA 组件。")
    return result


def inspect_lora_structure(model_config: Any, settings: dict[str, Any]) -> dict:
    """Use official component constructors and LoRA injection on meta, without loading weights or KV caches."""
    import torch
    from voxcpm.model.voxcpm import LoRAConfig, VoxCPMModel
    from voxcpm.modules.locdit import VoxCPMLocDiT
    from voxcpm.modules.minicpm4 import MiniCPMModel

    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        model = torch.nn.Module()
        model.lora_config = LoRAConfig(**settings)
        if settings["enable_lm"]:
            model.base_lm = MiniCPMModel(model_config.lm_config)
            residual = model_config.lm_config.model_copy(deep=True)
            residual.num_hidden_layers = model_config.residual_lm_num_layers
            residual.vocab_size = 0
            model.residual_lm = MiniCPMModel(residual)
        if settings["enable_dit"]:
            decoder = model_config.lm_config.model_copy(deep=True)
            decoder.hidden_size = model_config.dit_config.hidden_dim
            decoder.intermediate_size = model_config.dit_config.ffn_dim
            decoder.num_attention_heads = model_config.dit_config.num_heads
            decoder.num_hidden_layers = model_config.dit_config.num_layers
            decoder.kv_channels = model_config.dit_config.kv_channels
            decoder.vocab_size = 0
            model.feat_decoder = torch.nn.Module()
            model.feat_decoder.estimator = VoxCPMLocDiT(decoder, in_channels=model_config.feat_dim)
        if settings["enable_proj"]:
            model.enc_to_lm_proj = torch.nn.Linear(model_config.encoder_config.hidden_dim, model_config.lm_config.hidden_size)
            model.lm_to_dit_proj = torch.nn.Linear(model_config.lm_config.hidden_size, model_config.dit_config.hidden_dim)
            model.res_to_dit_proj = torch.nn.Linear(model_config.lm_config.hidden_size, model_config.dit_config.hidden_dim)
        VoxCPMModel._apply_lora(model)
        return validate_lora_parameters(model, settings)


def exact_token_counts(config: TtsExecutionConfig, manifest: str, tokenizer: Any, model_config: Any) -> dict:
    from voxcpm.training.data import compute_sample_lengths, load_audio_text_datasets

    dataset, _ = load_audio_text_datasets(
        train_manifest=manifest, val_manifest="", sample_rate=SAMPLE_RATE,
        num_proc=config.preprocessing_num_workers, prepare_durations=True,
    )

    def tokenize(batch):
        return {"text_ids": [tokenizer(text) for text in batch["text"]]}

    dataset = dataset.map(
        tokenize, batched=True, remove_columns=["text"], num_proc=config.preprocessing_num_workers,
        desc="Tokenizing training text",
    )
    audio = model_config.audio_vae_config
    hop_length = math.prod(audio.encoder_rates)
    if hop_length <= 0 or model_config.patch_size <= 0:
        raise ValueError("模型音频步长和 patch_size 须为正数。")
    lengths = compute_sample_lengths(
        dataset, audio_vae_fps=audio.sample_rate / hop_length, patch_size=model_config.patch_size,
    )
    if len(lengths) != len(dataset):
        raise ValueError("上游样本长度统计没有覆盖完整训练集。")
    maximum = config.max_batch_tokens // config.batch_size
    kept = sum(length <= maximum for length in lengths)
    return {"before_count": len(dataset), "kept_count": kept, "filtered_count": len(dataset) - kept}


def _child_probe(request: dict) -> dict:
    config = parse_execution_config(request["config"])
    checks = {key: _check(key) for key in CHECKS}
    details = {"python": sys.executable, "python_version": sys.version, "prefix": sys.prefix,
               "upstream_revision": UPSTREAM_REVISION, "packages": {}}
    python_ok = Path(sys.prefix).resolve() != Path(request["service_prefix"]).resolve()
    checks["python"] = _check("python", "available" if python_ok else "unavailable", None if python_ok else "请为 TTS 选择独立的 Python 环境。")
    errors = []
    torch = None
    try:
        torch = importlib.import_module("torch")
        details.update(torch=str(torch.__version__), cuda=torch.version.cuda, hip=getattr(torch.version, "hip", None))
        if tuple(map(int, str(torch.__version__).split("+", 1)[0].split(".")[:2])) < (2, 5):
            errors.append("TTS 环境需要 PyTorch 2.5 或更高版本。")
    except Exception as exc:
        errors.append(f"无法检查 PyTorch：{exc}")
    devices = None
    if torch is not None:
        devices = inspect_devices(torch, request.get("gpu_devices") or [], require_bf16=True)
        details["devices"] = devices
        cuda_available = any(
            item["name"] is not None and not any(issue["code"] == "tts.device.unavailable" for issue in item["issues"])
            for item in devices["checked_devices"]
        )
        checks["cuda"] = _check("cuda", "available" if cuda_available else "unavailable")
        if not cuda_available:
            checks["cuda"]["issues"] = selection_issues(devices)
        else:
            supported = bool(devices["eligible_devices"])
            checks["bf16"] = _check("bf16", "available" if supported else "unavailable")
            checks["bf16"]["issues"] = selection_issues(devices)
    source_valid = False
    for name in ("voxcpm", "torchaudio", "torchcodec", "soundfile", "safetensors", "argbind", "tensorboardX", "transformers", "datasets", "librosa", "matplotlib"):
        try:
            module = importlib.import_module(name)
            if name == "voxcpm":
                location = Path(module.__file__).resolve()
                if not location.is_relative_to(Path(config.trainer_path).resolve() / "src"):
                    raise ValueError("加载的 voxcpm 来自其他源码目录。")
                details["voxcpm_module"] = str(location)
                source_valid = True
            try:
                details["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                details["packages"][name] = str(getattr(module, "__version__", "unpackaged"))
        except Exception as exc:
            errors.append(f"无法加载 {name}：{exc}")
    checks["dependencies"] = _check("dependencies", "unavailable" if errors else "available", "\n".join(errors) if errors else None)
    tokenizer = None
    model_config = None
    if request["model_valid"] and source_valid:
        try:
            from voxcpm.model.voxcpm import VoxCPMConfig

            model_config = VoxCPMConfig.model_validate_json((Path(config.model_path) / "config.json").read_text(encoding="utf-8"))
            checks["model"] = _check("model", "available")
        except Exception as exc:
            checks["model"] = _check("model", "unavailable", str(exc))
        try:
            from transformers import LlamaTokenizerFast
            from voxcpm.model.utils import mask_multichar_chinese_tokens

            tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(config.model_path, local_files_only=True))
            tokenizer("语音 text")
            checks["tokenizer"] = _check("tokenizer", "available")
        except Exception as exc:
            checks["tokenizer"] = _check("tokenizer", "unavailable", str(exc))
        if model_config is not None and torch is not None:
            try:
                details["lora_targets"] = inspect_lora_structure(model_config, lora_settings(config))
                checks["lora_targets"] = _check("lora_targets", "available")
            except Exception as exc:
                checks["lora_targets"] = _check("lora_targets", "unavailable", str(exc))
    statistics = {}
    if request["manifests"].get("train") and config.max_batch_tokens > 0:
        result = _filter(config, "train", True)
        if model_config is not None and tokenizer is not None:
            try:
                result.update(exact_token_counts(config, request["manifests"]["train"], tokenizer, model_config), state="ready")
            except Exception as exc:
                result.update(state="error", issues=[_issue("tts.dataset.token_filter", ["sources", "train"], str(exc))])
        else:
            result["issues"] = [_issue("tts.dataset.token_filter_blocked", ["sources", "train"], "模型配置或分词器尚不可用，无法计算样本长度。")]
        statistics["train"] = result
    return {"checks": checks, "token_filter": statistics, "details": details, "devices": devices}


def _run_probe(config: TtsExecutionConfig, request: dict, cache: Path, timeout: float) -> dict:
    env = worker_environment(config)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env["HF_DATASETS_CACHE"] = str(cache)
    proc = subprocess.Popen(
        [config.python_path, "-m", "ypuddin.tts.runtime", "--probe"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd=config.trainer_path, start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = proc.communicate(json.dumps(request, ensure_ascii=False), timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10)
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.communicate(timeout=10)
        raise
    marker = next((line[len(MARKER):] for line in reversed(stdout.splitlines()) if line.startswith(MARKER)), None)
    if marker is None or proc.returncode:
        raise ValueError(f"TTS 环境检查未完成：{(stderr or stdout)[-1500:]}")
    report = json.loads(marker)
    if not isinstance(report, dict) or not isinstance(report.get("checks"), dict):
        raise ValueError("TTS 环境检查返回了无效结果。")
    return report


def inspect_runtime(
    config: TtsVersionConfig | dict,
    manifests: Mapping[str, str | Path | list[dict] | None],
    *,
    allowed: Callable[[Path], bool] | None = None,
    timeout: float = 90,
    gpu_devices: list[str] | None = None,
) -> dict:
    config = parse_execution_config(config)
    checks = {key: _check(key) for key in CHECKS}
    statistics = {split: _filter(config, split, manifests.get(split) is not None) for split in ("train", "validation")}
    identities: dict[str, dict] = {}
    normalized_hashes = {}
    model_identity = None
    details: dict[str, Any] = {}
    python_valid = False
    for key, value, validator in (
        ("python", config.python_path, lambda path: _path(path, "TTS Python")),
        ("upstream", config.trainer_path, validate_upstream),
        ("model", config.model_path, validate_model),
    ):
        try:
            if allowed is not None and not allowed(Path(value).expanduser().resolve()):
                raise ValueError("路径不在允许访问的目录中。")
            validator(value)
            if key == "python" and os.name != "nt" and not os.access(value, os.X_OK):
                raise ValueError("TTS Python 没有执行权限。")
            checks[key] = _check(key, "available")
            if key == "python":
                python_valid = True
                checks[key] = _check(key)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            checks[key] = _check(key, "unavailable", str(exc))
    with tempfile.TemporaryDirectory(prefix="tts-runtime-") as directory:
        temporary = Path(directory).resolve()
        prepared = {}
        for split in ("train", "validation"):
            source = manifests.get(split)
            if source is None:
                prepared[split] = None
                continue
            try:
                generated = isinstance(source, list)
                if generated:
                    for row in source:
                        for key in ("audio", "ref_audio"):
                            if key in row and row[key] is not None and (not isinstance(row[key], str) or not Path(row[key]).is_absolute()):
                                raise ValueError("规范化音频路径须为绝对路径。")
                    source_path = temporary / f"{split}-input.jsonl"
                    source_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in source), encoding="utf-8")
                else:
                    source_path = Path(source).expanduser()
                    if not source_path.is_absolute():
                        raise ValueError("规范化数据清单须为绝对路径。")
                    source_path = source_path.resolve()
                    identity = fingerprint_file(source_path, allowed=allowed)
                    identities[identity["path"]] = identity
                target = temporary / f"{split}.jsonl"
                manifest_allowed = (lambda path, source_path=source_path: path == source_path or allowed(path)) if generated and allowed else allowed
                summary = validate_manifest(source_path, output_path=target, allowed=manifest_allowed)
                normalized_hashes[split] = hashlib.sha256(target.read_bytes()).hexdigest()
                for line in target.read_text(encoding="utf-8").split("\n"):
                    if not line:
                        continue
                    row = json.loads(line)
                    for key in ("audio", "ref_audio"):
                        if row.get(key):
                            identity = fingerprint_file(row[key], allowed=allowed)
                            identities[identity["path"]] = identity
                prepared[split] = str(target)
                if split == "validation" or config.max_batch_tokens == 0:
                    statistics[split].update(
                        state="not_applicable" if split == "validation" else "disabled",
                        before_count=summary["samples"], kept_count=summary["samples"], filtered_count=0,
                    )
            except (OSError, ValueError, TypeError) as exc:
                prepared[split] = None
                statistics[split].update(state="error", issues=[_issue("tts.dataset.input", ["sources", split], str(exc))])
        if checks["model"]["state"] == "available":
            try:
                model_identity = _model_identity(config.model_path)
                for path in model_identity["assets"].values():
                    identity = fingerprint_file(path, allowed=allowed)
                    identities[identity["path"]] = identity
            except (OSError, ValueError) as exc:
                checks["model"] = _check("model", "unavailable", str(exc))
        if python_valid and checks["upstream"]["state"] == "available":
            request = {"config": config.model_dump(), "manifests": prepared, "service_prefix": sys.prefix, "model_valid": checks["model"]["state"] == "available", "gpu_devices": gpu_devices or []}
            try:
                report = _run_probe(config, request, temporary / "cache", timeout)
                for key in ("python", "dependencies", "cuda", "bf16", "tokenizer", "lora_targets"):
                    checks[key] = report["checks"][key]
                if checks["model"]["state"] == "available":
                    checks["model"] = report["checks"]["model"]
                statistics.update(report["token_filter"])
                details = report["details"]
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                code = "tts.environment.timeout" if isinstance(exc, subprocess.TimeoutExpired) else "tts.environment.probe"
                checks["dependencies"] = _check("dependencies", "unavailable", str(exc), code)
        changed = None
        try:
            if identities:
                captured = {"fingerprints": list(identities.values())}
                if model_identity is not None:
                    captured["model_identity"] = model_identity
                verify_input_snapshot(captured, model_path=config.model_path if model_identity is not None else None)
        except (OSError, ValueError) as exc:
            changed = str(exc)
            for split, value in statistics.items():
                if prepared.get(split):
                    value.update(state="error", before_count=None, kept_count=None, filtered_count=None,
                                 issues=[_issue("tts.dataset.input_changed", ["sources", split], changed)])
        fingerprint = None
        complete_identity = all(checks[key]["state"] == "available" for key in ("python", "upstream", "model", "dependencies", "tokenizer"))
        if not changed and complete_identity and details:
            identity = {"config": config.model_dump(), "files": sorted(identities.values(), key=lambda item: item["path"]), "model_identity": model_identity, "manifests": normalized_hashes, "runtime": details, "upstream_revision": UPSTREAM_REVISION}
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            for split, value in statistics.items():
                if value["before_count"] is not None:
                    value["input_fingerprint"] = hashlib.sha256(f"{fingerprint}:{split}".encode()).hexdigest()
        environment = {**_environment(checks), "devices": details.get("devices")}
        return {"environment": environment, "token_filter": statistics, "input_fingerprint": fingerprint,
                "fingerprints": [] if changed else sorted(identities.values(), key=lambda item: item["path"]),
                "model_identity": None if changed else model_identity, "details": details}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", required=True)
    parser.parse_args()
    result = _child_probe(json.load(sys.stdin))
    print(MARKER + json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
