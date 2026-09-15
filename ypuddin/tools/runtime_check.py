"""Two independent real Trainer workers; no downloads or existing training data.

Windows CUDA: python -m ypuddin.tools.runtime_check --device cuda:0 --device cuda:1 --out NEW_DIR
CPU lifecycle: python -m ypuddin.tools.runtime_check --device cpu --device cpu --out NEW_DIR
This checks a tiny toy model, not official model weights, the queue API, or DDP/FSDP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _worker_command(index, out, device, deadline):
    return [
        sys.executable,
        "-m",
        "ypuddin.tools.runtime_check",
        "--worker-index",
        str(index),
        "--out",
        str(out),
        "--device",
        device,
        "--deadline",
        str(deadline),
    ]


def _worker_environment(device):
    from ypuddin.server.supervisor import worker_device_environment

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    if device == "cpu":
        env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    else:
        env = worker_device_environment((device,), env)
    return env


def _cleanup(proc):
    """Only terminate this invocation's owned worker tree, retaining evidence."""
    import psutil

    from ypuddin.server.supervisor import JobSupervisor

    result = {"pid": proc.pid, "descendants": [], "errors": []}
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    except psutil.Error as exc:
        children = []
        result["errors"].append(f"Cannot inspect descendants: {exc}")
    result["descendants"] = [child.pid for child in children]
    try:
        JobSupervisor._kill_process_tree(proc)
        proc.wait(timeout=15)
        _, alive = psutil.wait_procs(children, timeout=3)
        result["remaining_descendants"] = [
            child.pid for child in alive if child.is_running() and child.status() != psutil.STATUS_ZOMBIE
        ]
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as kill_error:
                result["errors"].append(f"Worker cleanup failed: {kill_error}")
    result["exit_code"] = proc.poll()
    return result


def run_runtime_check(devices, out: Path, *, timeout: float = 180) -> dict:
    """Create a fresh isolated output directory and run exactly two workers."""
    import psutil

    if len(devices) != 2 or any(not re.fullmatch(r"cpu|cuda:(0|[1-9]\d*)", d) for d in devices):
        raise ValueError("请指定两个设备，例如 --device cuda:0 --device cuda:1，或两个 --device cpu")
    gpu_devices = [d for d in devices if d != "cpu"]
    if len(gpu_devices) != len(set(gpu_devices)):
        raise ValueError("两个 CUDA 任务必须选择不同的显卡")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("超时秒数必须大于 0")
    out = Path(out).expanduser().absolute()
    # No exist_ok: reruns must never replace previous reports, samples or data.
    out.mkdir(parents=True)
    (out / "barrier").mkdir()
    report = {
        "passed": False,
        "host": {"system": platform.system(), "platform": platform.platform(), "python": sys.version},
        "scope": "Real toy Trainer lifecycle in two independent processes; no queue API or official models.",
        "process_exit_scope": (
            "Launched worker parents plus descendants observed during cleanup; "
            "cleanup errors mean exit is unverified. Toy DataLoader num_workers=0; "
            "not a guarantee of tracking arbitrary detached processes."
        ),
        "windows_cuda_validated": False,
        "not_validated": [
            "official model weights",
            "queue API scheduling",
            "DDP/FSDP",
            "extension installation",
        ],
        "started_unix": time.time(),
        "timeout_seconds": timeout,
        "workers": [
            {"index": i, "requested_device": d, "pid": None, "exit_code": None} for i, d in enumerate(devices)
        ],
        "cleanup": [],
    }
    deadline = time.monotonic() + timeout
    _write_json(
        out / "invocation.json",
        {
            "kind": "ypuddin-runtime-check-v1",
            "parent_pid": os.getpid(),
            "parent_created": psutil.Process().create_time(),
            "out": str(out.resolve()),
            "devices": list(devices),
            "deadline": deadline,
        },
    )
    processes = []
    logs = []
    try:
        environments = [_worker_environment(d) for d in devices]
        masks = [
            e.get("CUDA_VISIBLE_DEVICES") for d, e in zip(devices, environments, strict=True) if d != "cpu"
        ]
        if len(masks) != len(set(masks)):
            raise ValueError("继承的显卡可见范围将两个任务映射到了同一张卡")
        for i, (device, env) in enumerate(zip(devices, environments, strict=True)):
            log = (out / f"worker-{i}.log").open("w", encoding="utf-8")
            logs.append(log)
            proc = subprocess.Popen(
                _worker_command(i, out, device, deadline),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
            )
            processes.append(proc)
            report["workers"][i].update(pid=proc.pid, log=str(out / f"worker-{i}.log"))
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in (None, 0) for proc in processes):
                raise RuntimeError("一个验收任务失败，正在停止本次验收的其他任务；请查看各任务日志")
            if time.monotonic() >= deadline:
                raise TimeoutError("独立任务验收超时，已请求清理本次创建的子进程")
            time.sleep(0.05)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for proc in processes:
            if proc.poll() is None:
                report["cleanup"].append(_cleanup(proc))
        for log in logs:
            log.close()
        for row, proc in zip(report["workers"], processes, strict=False):
            row["exit_code"] = proc.poll()
            worker_report = out / f"worker-{row['index']}" / "report.json"
            if worker_report.is_file():
                try:
                    result = json.loads(worker_report.read_text(encoding="utf-8"))
                    if not isinstance(result, dict) or (
                        "first_step" in result and not isinstance(result["first_step"], dict)
                    ):
                        raise ValueError("验收任务报告必须是对象，且 first_step 必须是对象")
                    row["result"] = result
                except (OSError, ValueError) as exc:
                    row["report_error"] = f"{type(exc).__name__}: {exc}"
        results = [row.get("result", {}) for row in report["workers"]]
        ready = [r.get("first_step", {}).get("completed_unix") for r in results]
        release = [r.get("barrier_released_unix") for r in results]
        overlap = all(type(t) in (int, float) and math.isfinite(t) for t in ready + release) and max(
            ready
        ) <= min(release)
        report["checks"] = {
            "independent_processes": len({p.pid for p in processes}) == 2
            and all(type(r.get("pid")) is int for r in results)
            and len({r.get("pid") for r in results}) == 2,
            "all_exit_codes_zero": len(processes) == 2 and all(p.returncode == 0 for p in processes),
            "all_owned_processes_exited": all(p.poll() is not None for p in processes)
            and all(
                not item.get("errors") and not item.get("remaining_descendants") for item in report["cleanup"]
            ),
            "both_trained_before_either_continued": bool(overlap),
            "all_worker_checks_passed": all(r.get("passed") is True for r in results),
        }
        report["passed"] = "error" not in report and all(report["checks"].values())
        report["windows_cuda_validated"] = (
            report["passed"]
            and platform.system() == "Windows"
            and all(r.get("backend") == "cuda" for r in results)
        )
        if not report["windows_cuda_validated"]:
            report["not_validated"].append("real Windows CUDA")
        report["finished_unix"] = time.time()
        _write_json(out / "report.json", report)
    return report


def _run_worker(index, out, requested_device, deadline):
    import psutil

    # Internal workers may only write under a fresh invocation created by their
    # actual parent process, never an arbitrary existing --out directory.
    invocation = json.loads((out / "invocation.json").read_text(encoding="utf-8"))
    if not isinstance(invocation, dict):
        raise ValueError("验收主进程记录无效")
    # Windows venv python.exe can insert a redirector process. Verify the live
    # ancestor identity, including creation time, rather than requiring direct PPID.
    owned = any(
        ancestor.pid == invocation.get("parent_pid")
        and ancestor.create_time() == invocation.get("parent_created")
        for ancestor in psutil.Process().parents()
    )
    if (
        invocation.get("kind") != "ypuddin-runtime-check-v1"
        or not owned
        or invocation.get("out") != str(out.resolve())
        or not math.isfinite(deadline)
        or invocation.get("deadline") != deadline
        or invocation.get("devices", [])[index : index + 1] != [requested_device]
    ):
        raise ValueError("内部 worker 必须由当前验收主进程在新目录中创建")
    import torch
    from PIL import Image

    from ypuddin.adapters import load_adapter_file
    from ypuddin.config import TrainConfig
    from ypuddin.tools.smoke import smoke_config
    from ypuddin.train import Trainer

    worker = out / f"worker-{index}"
    worker.mkdir()
    report = {
        "passed": False,
        "pid": os.getpid(),
        "requested_device": requested_device,
        "host_system": platform.system(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
        "visibility": {
            k: os.environ.get(k)
            for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        },
    }
    try:
        torch.set_num_threads(1)
        device = torch.device("cpu" if requested_device == "cpu" else "cuda:0")
        if device.type == "cuda":
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("请求的 CUDA 设备不可用或没有被隔离为单卡；禁止回退到 CPU")
            torch.cuda.set_device(0)
            # A real allocation and synchronized operation verifies this device,
            # rather than relying only on inventory metadata.
            assert torch.ones(8, device=device).square().sum().item() == 8
            torch.cuda.synchronize(device)
            report["device_name"] = torch.cuda.get_device_name(0)
            report["backend"] = "hip" if report["hip_version"] else "cuda"
        else:
            report["backend"] = "cpu"
        report["actual_device"] = str(device)

        class CheckedTrainer(Trainer):
            def _optimizer_step(self, group_loss, elapsed):
                super()._optimizer_step(group_loss, elapsed)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                report.setdefault("steps", []).append(
                    {
                        "step": self.progress.step,
                        "completed_unix": time.time(),
                        "loss": float(group_loss),
                    }
                )
                if self.progress.step == 1:
                    report["first_step"] = {"pid": os.getpid(), **report["steps"][-1]}
                    _write_json(out / "barrier" / f"worker-{index}.json", report["first_step"])
                    while not all((out / "barrier" / f"worker-{i}.json").is_file() for i in (0, 1)):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("等待另一个任务完成首步训练超时")
                        time.sleep(0.02)
                    report["barrier_released_unix"] = time.time()

        cfg = smoke_config(
            TrainConfig.model_validate(
                {
                    "model": {"family": "toy", "dtype": "fp32"},
                    "adapter": {"algo": "lokr", "rank": 4, "alpha": 4, "preset": "attn-mlp"},
                    "optimizer": {"type": "adamw", "lr": 0.001},
                    "memory": {"base_precision": "fp32", "offload_text_encoder": False},
                    "loop": {"mixed_precision": "no", "seed": 42 + index, "log_every": 1},
                    "checkpoint": {"save_dtype": "fp32"},
                }
            ),
            out=worker / "training",
            steps=3,
            resolution=64,
            sample_size=64,
            sample_steps=2,
        )
        cfg.dataset.bucket_step = 16
        cfg.sampling.every_steps = 3
        trainer = CheckedTrainer(cfg, device=device)
        trainer.prepare()
        initial, _ = trainer.adapters.export_state()
        initial = {name: value.detach().cpu().clone() for name, value in initial.items()}
        report["model_parameter_devices"] = sorted(
            {str(p.device) for p in trainer.loaded.backbone.parameters()}
        )
        outcome = trainer.run()
        exported = Path(cfg.checkpoint.output_dir) / "smoke-final.safetensors"
        weights, _ = load_adapter_file(exported)
        live, _ = trainer.adapters.export_state()
        changed = [name for name in initial if not torch.equal(initial[name], weights[name])]
        samples = []
        for path in sorted((Path(cfg.checkpoint.output_dir) / "samples").glob("*.png")):
            with Image.open(path) as image:
                image.load()
                samples.append(
                    {
                        "path": str(path),
                        "size": list(image.size),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        report.update(
            outcome=outcome,
            samples=samples,
            export={
                "path": str(exported),
                "sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
                "tensor_count": len(weights),
                "changed_tensor_count": len(changed),
            },
        )
        report["checks"] = {
            "actual_model_on_requested_device": report["model_parameter_devices"] == [str(device)],
            "three_optimizer_steps": trainer.progress.step == 3 and len(report.get("steps", [])) == 3,
            "finite_losses": all(math.isfinite(step["loss"]) for step in report.get("steps", [])),
            "finite_exported_weights": bool(weights)
            and all(torch.isfinite(t).all().item() for t in weights.values()),
            "weights_changed": bool(changed),
            "export_matches_live_weights": weights.keys() == live.keys()
            and all(torch.equal(weights[k], live[k].detach().cpu()) for k in weights),
            "readable_preview": bool(samples) and all(s["size"] == [64, 64] for s in samples),
            "finished": outcome == "finished",
        }
        report["passed"] = all(report["checks"].values())
    except BaseException as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        _write_json(worker / "report.json", report)
    return 0 if report["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--worker-index", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_index is not None:
        if len(args.device) != 1 or args.deadline is None or not math.isfinite(args.deadline):
            parser.error("内部 worker 参数不完整")
        try:
            return _run_worker(args.worker_index, args.out, args.device[0], args.deadline)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    try:
        result = run_runtime_check(args.device, args.out, timeout=args.timeout)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "report": str(args.out.absolute() / "report.json"),
                "windows_cuda_validated": result["windows_cuda_validated"],
            },
            ensure_ascii=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
