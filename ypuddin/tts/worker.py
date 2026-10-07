"""Supervised TTS worker; the upstream process keeps the supervisor's stdout/stderr pipes."""

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
from typing import Any

from .core import (
    audio_info,
    preflight,
    safe_checkpoint,
    training_settings,
    validate_manifest,
    worker_environment,
)
from .events import Events
from .execution_config import parse_execution_config

log = logging.getLogger("ypuddin.tts")


def _stop_process(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10)
    else:
        # The bridge owns this session, including its DataLoader processes.
        # Looking up every system process is unnecessary and may be restricted.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass
        except ProcessLookupError:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=5)


def _sample_options(snapshot: dict[str, Any]) -> dict[str, Any]:
    options = dict(snapshot.get("tts_sample") or {})
    if not isinstance(options.get("text"), str) or not options["text"].strip():
        raise ValueError("请填写试听文本。")
    if not options.get("source_job_id") or not options.get("source_output_dir"):
        raise ValueError("试听任务缺少来源训练任务。")
    options["checkpoint"] = str(safe_checkpoint(options["source_output_dir"], options.get("checkpoint", "")))
    if bool(options.get("reference_audio")) != bool(options.get("reference_text")):
        raise ValueError("参考音频和对应转写文本须同时填写。")
    if options.get("reference_audio"):
        audio = Path(options["reference_audio"]).expanduser().resolve()
        audio_info(audio)
        options["reference_audio"] = str(audio)
        if not isinstance(options["reference_text"], str) or not options["reference_text"].strip():
            raise ValueError("参考音频的转写文本不能为空。")
    return options


def run(snapshot_path: Path, *, mode: str = "train") -> int:
    snapshot_path = snapshot_path.resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("tts", {}).get("engine") == "gpt-sovits-v5":
        from .gpt_sovits.worker import run as run_gpt_sovits

        return run_gpt_sovits(snapshot_path, mode=mode)
    config = parse_execution_config(snapshot["tts"])
    events_path = Path(snapshot.get("logging", {}).get("events_path") or snapshot_path.parent / "events.jsonl")
    events = Events(events_path)
    control = Path(snapshot.get("control_dir") or snapshot_path.parent / "control")
    stopped = False
    proc: subprocess.Popen | None = None

    def stop_requested(*_args):
        nonlocal stopped
        stopped = True

    original = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        original[sig] = signal.signal(sig, stop_requested)

    def cancelled() -> bool:
        return stopped or (control / "stop").exists() or (control / "cancel").exists()

    try:
        events.emit("run.started", engine=config.engine, mode=mode, run_dir=str(snapshot_path.parent))
        events.emit("phase.changed", phase="preflight")
        if cancelled():
            events.emit("run.stopped")
            return 0
        device = snapshot.get("device", "cuda:0")
        if device != "cuda:0":
            raise ValueError("TTS worker 需要由任务队列映射到单张 CUDA GPU（cuda:0）。")
        if "tts_inputs" in snapshot:
            from .runtime import verify_input_snapshot

            verify_input_snapshot(snapshot["tts_inputs"], manifests=tuple(value for value in (config.train_manifest, config.val_manifest) if value) if mode == "train" else (), model_path=config.model_path)
        report = preflight(config, check_runtime=False, mode=mode)
        if not report["ok"]:
            raise ValueError("\n".join(report["errors"]))
        for warning in report["warnings"]:
            log.warning("%s", warning)
            events.emit("warning", message=warning)
        run_dir = snapshot_path.parent / "tts"
        run_dir.mkdir(parents=True, exist_ok=True)
        receipt = run_dir / "completion.json"
        if receipt.exists():
            raise ValueError("当前任务已存在完成记录，请新建任务或重试为新任务。")
        request: dict[str, Any] = {
            "mode": mode, "trainer_path": config.trainer_path, "model_path": config.model_path,
            "events_path": str(events_path), "receipt_path": str(receipt), "service_prefix": sys.prefix,
        }
        if mode == "train":
            output = Path(snapshot["checkpoint"]["output_dir"]).resolve() / "voxcpm"
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                output.mkdir()
            except FileExistsError as exc:
                raise ValueError("当前任务输出目录已存在 VoxCPM 训练记录，请新建任务，避免自动恢复旧检查点。") from exc
            train_path = run_dir / "train.jsonl"
            validation = run_dir / "validation.jsonl" if config.val_manifest else None
            validate_manifest(config.train_manifest, output_path=train_path)
            if validation:
                validate_manifest(config.val_manifest, output_path=validation)
            request["settings"] = training_settings(config, output, train_path, validation)
            (run_dir / "upstream_config.json").write_text(
                json.dumps(request["settings"], indent=2, ensure_ascii=False), encoding="utf-8"
            )
        else:
            request["sample"] = _sample_options(snapshot)
            sample_dir = Path(snapshot["sampling"]["output_dir"]).resolve()
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / "tts-sample.wav"
            if sample_path.exists():
                raise ValueError("当前试听任务已存在音频，请创建新的试听任务。")
            request["sample_path"] = str(sample_path)
        request_path = run_dir / "bridge_request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        if cancelled():
            events.emit("run.stopped")
            return 0
        if "tts_inputs" in snapshot:
            verify_input_snapshot(snapshot["tts_inputs"], manifests=tuple(value for value in (config.train_manifest, config.val_manifest) if value) if mode == "train" else (), model_path=config.model_path)
        env = worker_environment(config)
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        env["HF_DATASETS_CACHE"] = str(run_dir / "datasets_cache")
        proc = subprocess.Popen(
            [config.python_path, "-m", "ypuddin.tts.bridge", "--request", str(request_path)],
            cwd=config.trainer_path, env=env, start_new_session=os.name != "nt",
        )
        while proc.poll() is None:
            if cancelled():
                _stop_process(proc)
                break
            time.sleep(0.1)
        code = proc.wait()
        events.sync()
        if cancelled():
            log.info("语音任务已取消。")
            events.emit("run.stopped")
            return 0
        if code:
            raise RuntimeError(f"VoxCPM 进程异常退出（退出码 {code}），请查看前面的错误信息。")
        if not receipt.is_file():
            raise RuntimeError("VoxCPM 进程结束，但没有完成训练或试听。")
        result = json.loads(receipt.read_text(encoding="utf-8"))
        if result.get("mode") != mode:
            raise RuntimeError("VoxCPM 完成记录与当前任务不一致。")
        if mode == "train":
            if result.get("steps") != config.num_iters:
                raise RuntimeError("VoxCPM 没有完成设定的训练迭代次数。")
            safe_checkpoint(snapshot["checkpoint"]["output_dir"], result["checkpoint"])
        else:
            if result.get("path") != request["sample_path"]:
                raise RuntimeError("试听音频路径与当前任务不一致。")
            audio_info(Path(result["path"]))
            events.emit("tts.sample.saved", **{key: value for key, value in result.items() if key != "mode"})
        events.emit("run.finished", **{key: value for key, value in result.items() if key != "mode"})
        log.info("语音%s已完成。", "训练" if mode == "train" else "试听")
        return 0
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            _stop_process(proc)
        events.sync()
        if cancelled():
            log.info("语音任务已取消。")
            events.emit("run.stopped")
            return 0
        log.exception("语音任务失败：%s", exc)
        events.emit("run.failed", error=str(exc))
        return 1
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("train", "sample"), default="train")
    args = parser.parse_args()
    from ypuddin.worker_log import configure

    configure()
    return run(args.config, mode=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
