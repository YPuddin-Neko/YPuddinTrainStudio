"""Runs only in the selected VoxCPM Python environment."""

from __future__ import annotations

import argparse
import builtins
import importlib
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

from .core import SAMPLE_RATE, UPSTREAM_REVISION, safe_checkpoint
from .devices import inspect_devices, selection_issues
from .events import Events

log = logging.getLogger("ypuddin.tts")


def probe(trainer_path: str, mode: str, gpu_devices: list[str] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {"python": sys.executable, "prefix": sys.prefix}
    try:
        import torch

        details.update(torch=torch.__version__, cuda=torch.version.cuda, hip=getattr(torch.version, "hip", None))
        details["devices"] = inspect_devices(torch, gpu_devices or [], require_bf16=mode == "train")
        errors.extend(issue["message"] for issue in selection_issues(details["devices"]))
        eligible = details["devices"]["eligible_devices"]
        if eligible:
            details["device"] = next(item["name"] for item in details["devices"]["checked_devices"] if item["device"] == eligible[0])
        parts = str(torch.__version__).split("+", 1)[0].split(".")
        if tuple(map(int, parts[:2])) < (2, 5):
            errors.append("TTS 环境需要 PyTorch 2.5 或更高版本。")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"无法加载 TTS PyTorch：{exc}")
    modules = ["voxcpm", "torchaudio", "torchcodec", "soundfile", "safetensors"]
    if mode == "train":
        modules.extend(["argbind", "tensorboardX", "transformers", "datasets", "librosa", "matplotlib"])
    for name in modules:
        try:
            module = importlib.import_module(name)
            if name == "voxcpm":
                location = Path(module.__file__).resolve()
                if not location.is_relative_to(Path(trainer_path).resolve() / "src"):
                    errors.append("当前加载的 voxcpm 来自其他目录，请检查独立环境与源码路径。")
                details["voxcpm_module"] = str(location)
        except Exception as exc:
            errors.append(f"TTS 环境无法加载 {name}：{exc}")
    return {"ok": not errors, "errors": errors, "details": details}


def load_trainer(root: Path):
    spec = importlib.util.spec_from_file_location("_ypuddin_voxcpm_training", root / "scripts/train_voxcpm_finetune.py")
    if spec is None or spec.loader is None:
        raise ValueError("无法加载 VoxCPM 训练入口。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def train(request: dict[str, Any], events: Events) -> dict[str, Any]:
    module = load_trainer(Path(request["trainer_path"]))
    settings = request["settings"]
    total_steps = settings["num_iters"]
    observed = {"step": 0}
    original_tracker = module.TrainingTracker
    weights = settings.get("lambdas", {})
    original_model = getattr(module, "VoxCPMModel", None)
    if original_model is not None:
        from .runtime import validate_lora_parameters

        class CheckedModel:
            @staticmethod
            def from_local(*args, **kwargs):
                model = original_model.from_local(*args, **kwargs)
                validate_lora_parameters(model, settings["lora"])
                return model

        CheckedModel.__name__ = original_model.__name__
        module.VoxCPMModel = CheckedModel

    def upstream_print(*args, **kwargs):
        if len(args) == 2 and isinstance(args[0], str) and type(args[1]) is bool and kwargs.get("file") is sys.stderr:
            log.debug("%s requires_grad=%s", args[0], args[1])
            return
        builtins.print(*args, **kwargs)

    module.print = upstream_print

    class Tracker(original_tracker):
        def log_metrics(self, metrics, split):
            values = {key: float(value) for key, value in metrics.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("训练指标出现非有限数值，请检查数据与学习率。")
            if self.rank == 0:
                formatted = ", ".join(
                    f"{key}: {value:.6e}" if key == "lr" else f"{key}: {value:.6f}"
                    for key, value in values.items()
                )
                log.info("[%s] step %d: %s", split, self.step + 1, formatted)
                if split == "train":
                    observed["step"] = self.step + 1
                    events.emit(
                        "step", step=self.step + 1, upstream_step=self.step,
                        epoch=values.get("epoch", 0.0),
                        loss=sum(value * weights.get(key, 1.0) for key, value in values.items() if key.startswith("loss/")),
                        lr={"voxcpm": values["lr"]} if "lr" in values else None,
                        grad_norm=values.get("grad_norm"), metrics=values,
                        total_steps=total_steps,
                    )
                elif split == "val":
                    events.emit(
                        "validation", step=self.step + 1, mean=values.get("loss/total"), per_t={}, metrics=values
                    )
            if self.writer is not None:
                for key, value in values.items():
                    self.writer.add_scalar(f"{split}/{key}", value, self.step)

    module.TrainingTracker = Tracker
    original_save = module.save_checkpoint

    def save_checkpoint(*args, **kwargs):
        result = original_save(*args, **kwargs)
        output = Path(args[3] if len(args) > 3 else kwargs["save_dir"])
        step = int(args[4] if len(args) > 4 else kwargs["step"])
        checkpoint = safe_checkpoint(output.parent, output / f"step_{step:07d}")
        weights = checkpoint / "lora_weights.safetensors"
        if not weights.is_file():
            weights = checkpoint / "lora_weights.ckpt"
        events.emit(
            "checkpoint.saved", path=str(weights), kind="tts", step=min(step + 1, total_steps),
            upstream_step=step, engine="voxcpm1.5",
        )
        return result

    module.save_checkpoint = save_checkpoint
    events.emit("run.prepared", total_steps=total_steps)
    events.emit("phase.changed", phase="training")
    log.info("开始 VoxCPM 1.5 LoRA 训练，共 %d 次更新，初始学习率 %.6e", total_steps, settings["learning_rate"])
    try:
        module.train(**settings)
    finally:
        if original_model is not None:
            module.VoxCPMModel = original_model
    if observed["step"] != total_steps:
        raise RuntimeError(f"训练入口提前返回：完成 {observed['step']} / {total_steps} 次更新。")
    checkpoint = safe_checkpoint(Path(settings["save_path"]).parent, Path(settings["save_path"]) / f"step_{total_steps:07d}")
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    if state.get("step") != total_steps:
        raise RuntimeError("最终检查点记录的步数与训练迭代次数不一致。")
    return {"mode": "train", "steps": observed["step"], "checkpoint": str(checkpoint), "revision": UPSTREAM_REVISION}


def sample(request: dict[str, Any], events: Events) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm import LoRAConfig

    options = request["sample"]
    checkpoint = safe_checkpoint(options["source_output_dir"], options["checkpoint"])
    saved = json.loads((checkpoint / "lora_config.json").read_text(encoding="utf-8"))
    base = Path(request["model_path"]).resolve()
    if not saved.get("base_model") or Path(saved["base_model"]).resolve() != base:
        raise ValueError("检查点记录的基础模型与当前任务的本地基础模型不一致。")
    model = VoxCPM(
        voxcpm_model_path=str(base), zipenhancer_model_path=None, enable_denoiser=False,
        optimize=False, device="cuda:0", lora_config=LoRAConfig(**saved["lora_config"]),
    )
    loaded, skipped = model.tts_model.load_lora_weights(str(checkpoint))
    expected = {name for name in model.tts_model.state_dict() if "lora_" in name}
    if not loaded or skipped or set(loaded) != expected:
        raise ValueError("LoRA 权重未完整匹配当前基础模型，无法试听。")
    events.emit("phase.changed", phase="sampling")
    waveform = model.generate(
        text=options["text"], prompt_wav_path=options.get("reference_audio") or None,
        prompt_text=options.get("reference_text") or None, seed=options.get("seed", 42),
        cfg_value=options.get("cfg_value", 2.0), inference_timesteps=options.get("inference_timesteps", 10),
        normalize=False, denoise=False,
    )
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("试听没有生成有效音频。")
    rate = int(model.tts_model.sample_rate)
    if rate != SAMPLE_RATE:
        raise ValueError(f"试听模型输出采样率为 {rate} Hz，与 VoxCPM 1.5 不符。")
    output = Path(request["sample_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    try:
        sf.write(str(temporary), waveform, rate, format="WAV", subtype="PCM_16")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "mode": "sample", "path": str(output), "sample_rate": rate,
        "duration_seconds": waveform.size / rate, "source_job_id": options["source_job_id"],
        "checkpoint": str(checkpoint), "text": options["text"],
        "requested_seed": options.get("seed", 42),
        "seed": getattr(model.tts_model, "last_successful_seed", None),
        "cfg_value": options.get("cfg_value", 2.0),
        "inference_timesteps": options.get("inference_timesteps", 10),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.probe:
        request = json.load(sys.stdin)
        report = probe(request["trainer_path"], request.get("mode", "train"), request.get("gpu_devices"))
        print("TTS_PREFLIGHT " + json.dumps(report, ensure_ascii=False))
        return 0 if report["ok"] else 1
    if args.request is None:
        parser.error("--request is required")
    from ypuddin.worker_log import configure

    configure()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    report = probe(request["trainer_path"], request["mode"], ["cuda:0"])
    if report["errors"]:
        raise RuntimeError("\n".join(report["errors"]))
    events = Events(request["events_path"])
    result = train(request, events) if request["mode"] == "train" else sample(request, events)
    Path(request["receipt_path"]).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
