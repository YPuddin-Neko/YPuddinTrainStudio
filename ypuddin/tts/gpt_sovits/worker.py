"""Supervisor-facing GPT-SoVITS worker with cancellation and completion receipts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..events import Events
from .config import parse_config
from .core import (
    audio_info,
    model_identity,
    preflight,
    require_text_assets,
    safe_checkpoint,
    worker_environment,
)

log = logging.getLogger("ypuddin.tts")


def sample_options(snapshot):
    options = dict(snapshot.get("tts_sample") or {})
    if not isinstance(options.get("text"), str) or not options["text"].strip():
        raise ValueError("请填写试听文本。")
    if not options.get("source_job_id") or not options.get("source_output_dir"):
        raise ValueError("试听任务缺少来源训练任务。")
    options["checkpoint"] = str(safe_checkpoint(options["source_output_dir"], options.get("checkpoint", "")))
    if not options.get("reference_audio") or not isinstance(options.get("reference_text"), str) or not options["reference_text"].strip():
        raise ValueError("GPT-SoVITS 试听需要参考录音及对应转写。")
    reference = Path(options["reference_audio"]).expanduser().resolve(strict=True)
    info = audio_info(reference)
    if not 3 <= info["duration"] <= 10:
        raise ValueError("GPT-SoVITS 参考录音长度须为 3 至 10 秒。")
    options["reference_audio"] = str(reference)
    from ..sample_config import GptSovitsSampleOptions

    variant = snapshot["tts"].get("variant", "v5dev")
    options["gpt_sovits"] = GptSovitsSampleOptions.model_validate(options.get("gpt_sovits", {})).resolved(variant).model_dump()
    require_text_assets(parse_config(snapshot["tts"]), [options["gpt_sovits"][key] for key in ("text_language", "reference_language")])
    return options


def run(snapshot_path, *, mode="train"):
    from ..runtime import verify_input_snapshot
    from ..worker import _stop_process

    snapshot_path = Path(snapshot_path).resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    config = parse_config(snapshot["tts"])
    events_path = Path(snapshot.get("logging", {}).get("events_path") or snapshot_path.parent / "events.jsonl")
    events = Events(events_path)
    control = Path(snapshot.get("control_dir") or snapshot_path.parent / "control")
    stopped = False
    proc = None

    def stop(*_args):
        nonlocal stopped
        stopped = True

    original = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def cancelled():
        return stopped or (control / "stop").exists() or (control / "cancel").exists()

    try:
        events.emit("run.started", engine=config.engine, mode=mode, variant=config.variant)
        if cancelled():
            events.emit("run.stopped")
            return 0
        if snapshot.get("device", "cuda:0") != "cuda:0":
            raise ValueError("GPT-SoVITS 需要由队列映射到单张 CUDA 显卡。")
        if "tts_inputs" not in snapshot:
            raise ValueError("GPT-SoVITS 任务缺少已校验的输入内容指纹。")
        if model_identity(config) != snapshot["tts_inputs"].get("model_identity"):
            raise ValueError("GPT-SoVITS 执行配置与冻结模型身份不一致。")
        manifests = (config.train_manifest,) if mode == "train" else ()
        verify_input_snapshot(snapshot["tts_inputs"], manifests=manifests, model_path=config.model_path)
        events.emit("phase.changed", phase="preflight")
        report = preflight(config, check_runtime=True, mode=mode, gpu_devices=["cuda:0"])
        if not report["ok"]:
            raise ValueError("\n".join(report["errors"]))
        run_dir = snapshot_path.parent / "gpt_sovits"
        run_dir.mkdir()
        request = {"mode": mode, "config": config.model_dump(), "events_path": str(events_path),
                   "receipt_path": str(run_dir / "completion.json"), "run_dir": str(run_dir), "service_prefix": sys.prefix}
        if mode == "train":
            output = Path(snapshot["checkpoint"]["output_dir"]).resolve() / "gpt_sovits"
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ValueError("GPT-SoVITS 输出目录已存在，请创建新任务，避免自动恢复旧状态。")
            request["output"] = str(output)
        else:
            request["sample"] = sample_options(snapshot)
            directory = Path(snapshot["sampling"]["output_dir"]).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            request["sample_path"] = str(directory / "tts-sample.wav")
            if Path(request["sample_path"]).exists():
                raise ValueError("此任务已存在试听文件，请创建新任务。")
        request_path = run_dir / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        verify_input_snapshot(snapshot["tts_inputs"], manifests=manifests, model_path=config.model_path)
        if cancelled():
            events.emit("run.stopped")
            return 0
        proc = subprocess.Popen([config.python_path, "-m", "ypuddin.tts.gpt_sovits.bridge", "--request", str(request_path)],
                                cwd=run_dir, env=worker_environment(config), start_new_session=os.name != "nt")
        while proc.poll() is None:
            if cancelled():
                _stop_process(proc)
                break
            time.sleep(0.1)
        code = proc.wait()
        events.sync()
        if cancelled():
            events.emit("run.stopped")
            return 0
        if code:
            raise RuntimeError(f"GPT-SoVITS 进程异常退出（{code}），请查看前面的错误信息。")
        receipt = Path(request["receipt_path"])
        if not receipt.is_file():
            raise RuntimeError("GPT-SoVITS 没有返回完成记录。")
        result = json.loads(receipt.read_text())
        if result.get("mode") != mode:
            raise RuntimeError("GPT-SoVITS 完成记录与任务类型不一致。")
        if mode == "train":
            safe_checkpoint(snapshot["checkpoint"]["output_dir"], result["checkpoint"])
            required = {"gpt", "sovits"} if config.stage == "both" else {config.stage}
            if set(result.get("stages", {})) != required:
                raise RuntimeError("GPT-SoVITS 未完成指定训练阶段。")
            for stage in required:
                if result["stages"][stage].get("epoch") != getattr(config, stage).epochs:
                    raise RuntimeError("GPT-SoVITS 训练轮数未达到配置要求。")
                if result["stages"][stage].get("global_step", 0) <= 0:
                    raise RuntimeError("GPT-SoVITS 训练阶段没有执行训练步骤。")
        else:
            if result.get("path") != request["sample_path"]:
                raise RuntimeError("试听文件路径与当前任务不一致。")
            audio_info(result["path"])
            events.emit("tts.sample.saved", **{key: value for key, value in result.items() if key != "mode"})
        events.emit("run.finished", **{key: value for key, value in result.items() if key != "mode"})
        return 0
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            _stop_process(proc)
        events.sync()
        if cancelled():
            events.emit("run.stopped")
            return 0
        log.exception("GPT-SoVITS 任务失败：%s", exc)
        events.emit("run.failed", error=str(exc))
        return 1
    finally:
        for sig, previous in original.items():
            signal.signal(sig, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("train", "sample"), default="train")
    args = parser.parse_args()
    from ypuddin.worker_log import configure

    configure()
    return run(args.config, mode=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
