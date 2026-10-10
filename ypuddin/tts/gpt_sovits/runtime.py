"""Inspect the selected interpreter without importing its ML stack into the server."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import parse_config
from .core import (
    MODEL_REQUIRED_MESSAGE,
    UPSTREAM_REVISION,
    _path,
    asset_paths,
    manifest_rows,
    model_identity,
    require_text_assets,
    validate_upstream,
    worker_environment,
)

CHECKS = ("python", "upstream", "model", "dependencies", "cuda", "precision")
MARKER = "GSV_RUNTIME "


def check(key, state="unchecked", error=None):
    return {"key": key, "state": state, "checked_at": time.time() if state != "unchecked" else None,
            "issues": [] if error is None else [{"code": f"tts.gpt_sovits.environment.{key}",
                                                "loc": ["environment", key], "message": str(error),
                                                "severity": "error", "details": {}}]}


def inspect_weights(config):
    import torch
    from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

    paths = asset_paths(config)
    gpt = torch.load(paths["gpt"], map_location="cpu", weights_only=False)
    if not isinstance(gpt, dict) or not isinstance(gpt.get("weight"), dict) or not isinstance(gpt.get("config"), dict):
        raise ValueError("GPT 权重需要 weight 和 config，训练恢复 checkpoint 不能作为预训练权重。")
    architecture = gpt["config"].get("model", {})
    if architecture.get("vocab_size") != 1025 or architecture.get("embedding_dim") != 512 or architecture.get("n_layer") != 24:
        raise ValueError("GPT 权重架构与 s1v3 不兼容。")
    if not gpt["weight"] or any("lora_" in key for key in gpt["weight"]):
        raise ValueError("GPT 需要完整导出权重。")
    del gpt
    for key in ("sovits", "sovits_base"):
        _, variant, lora = get_sovits_version_from_path_fast(paths[key])
        value = load_sovits_new(paths[key])
        declared = value.get("config", {}).get("model", {}).get("version")
        if variant != config.variant and declared == config.variant and not lora:
            variant = declared
        if variant != config.variant or lora or value.get("lora_rank") or any("lora_" in name for name in value.get("weight", {})):
            raise ValueError("SoVITS 预训练权重必须是所选 v5 变体的完整底模；LoRA 导出权重不能直接继续微调。")
        if not isinstance(value.get("weight"), dict) or not any(k.startswith("cfm.") for k in value["weight"]):
            raise ValueError("SoVITS 权重缺少 CFM 参数。")
        del value
    return {"variant": config.variant, "gpt_architecture": "s1v3", "sovits_mode": "lora"}


def child_probe(request):
    config = parse_config(request["config"])
    checks = {key: check(key) for key in CHECKS}
    details = {"prefix": sys.prefix, "executable": sys.executable, "upstream_revision": UPSTREAM_REVISION}
    checks["python"] = check("python", "available")
    try:
        import torch

        from ..devices import inspect_devices, selection_issues

        for name in ("yaml", "numpy", "scipy", "librosa", "transformers", "peft", "pytorch_lightning",
                     "tensorboard", "soundfile", "module.models", "module.models_v5", "AR.models.t2s_lightning_module"):
            importlib.import_module(name)
        if request.get("mode") == "sample":
            importlib.import_module("resampy")
        checks["dependencies"] = check("dependencies", "available")
        details["torch"] = torch.__version__
        devices = inspect_devices(torch, request.get("gpu_devices", []), require_bf16=False)
        details["devices"] = devices
        errors = selection_issues(devices)
        checks["cuda"] = check("cuda", "unavailable" if errors else "available", errors[0]["message"] if errors else None)
        checks["precision"] = check("precision", "available" if not errors else "unchecked")
        details["precision"] = {"gpt": config.gpt.precision, "sovits": config.sovits.precision}
        if not errors and os.name == "nt" and request.get("mode", "train") == "train" and config.stage in {"gpt", "both"}:
            try:
                if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
                    raise RuntimeError("当前 PyTorch 未提供 Gloo 支持。")
                torch.distributed.ProcessGroupGloo.create_device(hostname="127.0.0.1")
            except Exception as exc:
                raise ValueError(f"GPT 训练需要可用的 Gloo 设备：{exc}") from exc
        if request.get("model_valid", True):
            try:
                details["weights"] = inspect_weights(config)
                checks["model"] = check("model", "available")
            except Exception as exc:
                checks["model"] = check("model", "unavailable", exc)
    except Exception as exc:
        checks["dependencies"] = check("dependencies", "unavailable", exc)
    return {"checks": checks, "details": details}


def run_probe(config, *, gpu_devices=None, mode="train", timeout=90, model_valid=True):
    from ..worker import _stop_process

    request = {"config": config.model_dump(), "gpu_devices": gpu_devices or [], "service_prefix": sys.prefix,
               "mode": mode, "model_valid": model_valid}
    env = worker_environment(config, check_model_assets=model_valid)
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.Popen([config.python_path, "-m", "ypuddin.tts.gpt_sovits.runtime", "--probe"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, cwd=config.trainer_path,
                            start_new_session=os.name != "nt")
    try:
        stdout, stderr = proc.communicate(json.dumps(request, ensure_ascii=False).encode("utf-8"), timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_process(proc)
        proc.communicate(timeout=10)
        raise
    prefix = MARKER.encode("ascii")
    marker = next((line[len(prefix):] for line in reversed(stdout.splitlines()) if line.startswith(prefix)), None)
    if proc.returncode or marker is None:
        diagnostic = (stderr or stdout).decode("utf-8", errors="replace")[-1500:]
        raise ValueError(f"GPT-SoVITS 环境检查失败：{diagnostic}")
    return json.loads(marker.decode("utf-8"))


def inspect_runtime(config, manifests, *, allowed=None, timeout=90, gpu_devices=None):
    from ..runtime import fingerprint_file

    config = parse_config(config)
    checks = {key: check(key) for key in CHECKS}
    details = {"upstream_revision": UPSTREAM_REVISION}
    identities, normalized = {}, {}
    identity = None
    for key, path, validate in (("python", config.python_path, lambda p: _path(p, "Python")),
                                ("upstream", config.trainer_path, validate_upstream),
                                ("model", config.model_path, lambda p: model_identity(config))):
        try:
            if key == "model" and not path.strip():
                raise ValueError(MODEL_REQUIRED_MESSAGE)
            if allowed and not allowed(Path(path).expanduser().resolve()):
                raise ValueError("路径不在允许访问的目录中。")
            result = validate(path)
            if key == "python" and os.name != "nt" and not os.access(path, os.X_OK):
                raise ValueError("Python 没有执行权限。")
            if key == "model":
                model_files = {}
                for value in result["assets"].values():
                    record = fingerprint_file(value, allowed=allowed)
                    model_files[record["path"]] = record
                identities.update(model_files)
                identity = result
            checks[key] = check(key, "available")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            checks[key] = check(key, "unavailable", exc)
    input_errors = []
    for split, source in manifests.items():
        if source is None:
            continue
        try:
            if isinstance(source, list):
                rows = source
            else:
                record = fingerprint_file(source, allowed=allowed)
                identities[record["path"]] = record
                rows = manifest_rows(source, allowed=allowed)
            normalized[split] = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            for row in rows:
                record = fingerprint_file(row["audio"], allowed=allowed)
                identities[record["path"]] = record
            if identity is not None:
                require_text_assets(config, {row.get("language") for row in rows})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            input_errors.append(str(exc))
    if checks["python"]["state"] == checks["upstream"]["state"] == "available":
        try:
            result = run_probe(config, gpu_devices=gpu_devices, timeout=timeout, model_valid=identity is not None)
            for key in ("python", "dependencies", "cuda", "precision"):
                checks[key] = result["checks"][key]
            if identity is not None:
                checks["model"] = result["checks"]["model"]
            details.update(result["details"])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            checks["dependencies"] = check("dependencies", "unavailable", exc)
    changed = False
    for value in identities.values():
        try:
            if fingerprint_file(value["path"], allowed=allowed) != value:
                raise ValueError("模型或音频文件在检查期间改变。")
        except (OSError, ValueError) as exc:
            changed = True
            input_errors.append(str(exc))
    if input_errors:
        input_check = check("model", "unavailable", "\n".join(input_errors))
        if checks["model"]["issues"]:
            checks["model"]["issues"].extend(input_check["issues"])
        else:
            checks["model"] = input_check
    state = "available" if all(value["state"] == "available" for value in checks.values()) else "unavailable"
    files = sorted(identities.values(), key=lambda value: value["path"])
    fingerprint = None
    if state == "available" and not changed:
        fingerprint = hashlib.sha256(json.dumps({"config": config.model_dump(), "files": files,
                                                "model_identity": identity, "manifests": normalized,
                                                "upstream_revision": UPSTREAM_REVISION}, sort_keys=True).encode()).hexdigest()
    return {"environment": {"state": state, "checked_at": time.time(), "checks": list(checks.values()),
                            "devices": details.get("devices")}, "input_fingerprint": fingerprint,
            "fingerprints": [] if changed else files, "model_identity": None if changed else identity, "details": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", required=True)
    parser.parse_args()
    print(MARKER + json.dumps(child_probe(json.load(sys.stdin)), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
