"""Run isolated reference/resume jobs using an existing main-model LoRA/LoKr recipe.

No downloads, uploads, environment installation or source-operator replacements.
Use --prepare-only to review configs without starting training. CPU evidence is
never described as CUDA/DTK acceptance. Run only your own trusted checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ypuddin.config import load_config  # noqa: E402


def source_identity():
    paths = sorted((ROOT / "ypuddin").rglob("*.py")) + [
        Path(__file__),
        ROOT / "tests/checkpoint_assertions.py",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_job(command, log, env, timeout):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                import psutil

                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                parent.kill()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Training exited {code}; inspect {log.name}")


def compare(reference, resumed, steps):
    import torch
    from safetensors.torch import load_file

    from tests.checkpoint_assertions import assert_checkpoint_value_exact

    states = [p / f"state-{steps}" for p in (reference, resumed)]
    metadata = [json.loads((p / "state.json").read_text()) for p in states]
    assert metadata[0].get("strategy") == metadata[1].get("strategy"), "checkpoint strategy differs"
    name = "model.safetensors" if metadata[0].get("strategy") == "fsdp2" else "training.safetensors"
    weights = [load_file(p / name) for p in states]
    assert weights[0] and weights[0].keys() == weights[1].keys(), "weight keys"
    for key in weights[0]:
        assert weights[0][key].dtype == weights[1][key].dtype, f"weight dtype differs: {key}"
        assert torch.isfinite(weights[0][key]).all(), f"nonfinite weight: {key}"
        assert torch.equal(weights[0][key], weights[1][key]), f"weight differs: {key}"
    for field in ("progress", "sampler", "sampler_ranks", "adapter_contract"):
        if field not in metadata[0] and field not in metadata[1]:
            continue
        assert_checkpoint_value_exact(metadata[1][field], metadata[0][field], field)
    for component in ("optimizer", "scheduler", "rng"):
        values = [torch.load(p / f"{component}.pt", map_location="cpu", weights_only=False) for p in states]
        assert_checkpoint_value_exact(values[1], values[0], component)
    previews = []
    for directory in (reference, resumed):
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        assert any(e["type"] == "run.finished" for e in events), "training did not finish"
        paths = [Path(e["path"]) for e in events if e["type"] == "sample.saved" and e["step"] == steps]
        assert paths, "missing final-step preview"
        previews.append([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
    assert previews[0] == previews[1], "preview differs"
    return {
        "tensor_count": len(weights[0]),
        "weights_exact": True,
        "states_exact": True,
        "preview_sha256": previews[0],
    }


def main():
    if not __debug__:
        raise RuntimeError("Acceptance assertions require Python without -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, help="New directory; existing directories are refused")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--processes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--strategy", choices=("ddp", "fsdp"), default="ddp")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resume-step", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-text-encoder",
        action="store_true",
        help="Also accept explicit online text-encoder adapter recipes",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.training.mode != "adapter":
        parser.error("Recipe must train adapters")
    if cfg.training.train_text_encoder:
        if not args.allow_text_encoder or cfg.dataset.text_encoding != "online":
            parser.error("Text-encoder acceptance requires --allow-text-encoder and online encoding")
    elif not cfg.training.train_backbone:
        parser.error("Recipe must train at least one component")
    if cfg.adapter.algo not in {"lora", "lokr"} or cfg.checkpoint.resume:
        parser.error("Use LoRA/LoKr and a recipe without an existing full-state resume")
    if not cfg.sampling.prompts or cfg.sampling.prompts_file:
        parser.error("Provide explicit sampling.prompts in the recipe")
    if (
        args.resume_step < 1
        or args.steps <= args.resume_step
        or args.steps % args.resume_step
        or args.timeout < 1
    ):
        parser.error("steps must be a multiple of resume-step and exceed it; timeout must be positive")
    if args.strategy == "fsdp" and (args.processes != 2 or args.device != "cuda"):
        parser.error("FSDP requires two CUDA/HIP devices")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg.loop.max_steps, cfg.loop.epochs = args.steps, None
    cfg.loop.gpu_count, cfg.loop.distributed_strategy = args.processes, args.strategy
    cfg.loop.deterministic = True
    cfg.dataset.cache_dir = str(output / "cache")
    cfg.checkpoint.save_state_every_steps = args.resume_step
    cfg.checkpoint.save_every_epochs = None
    cfg.checkpoint.save_on_finish = True
    cfg.sampling.enabled = True
    cfg.sampling.every_steps, cfg.sampling.every_epochs = args.resume_step, None
    identity = source_identity()
    report = {
        "status": "prepared",
        "strict_passed": False,
        "device_requested": args.device,
        "processes": args.processes,
        "strategy": args.strategy,
        "family": cfg.model.family,
        "algorithm": cfg.adapter.algo,
        "train_backbone": cfg.training.train_backbone,
        "train_text_encoder": cfg.training.train_text_encoder,
        "source_sha256": identity,
        "scope": "Fixed recipe only; CPU is not GPU acceptance",
    }
    commands = []
    for phase in ("reference", "resumed"):
        cfg.checkpoint.output_dir = str(output / phase)
        cfg.checkpoint.resume = (
            str(output / "reference" / f"state-{args.resume_step}") if phase == "resumed" else None
        )
        path = output / f"{phase}.json"
        path.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        command = [sys.executable, "-m", "ypuddin.cli", "train", str(path), "--device", args.device]
        if args.processes == 2:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=127.0.0.1:0",
                "--local_addr=127.0.0.1",
                "--rdzv_conf=is_host=true",
                "--nproc_per_node=2",
                "-m",
                *command[2:],
            ]
        commands.append(command)
    report["commands"] = commands
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(report_path)
        return 0
    try:
        import torch

        report["runtime"] = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda}
        if args.device == "cuda" and (
            not torch.cuda.is_available() or torch.cuda.device_count() < args.processes
        ):
            raise RuntimeError("Requested CUDA/HIP devices unavailable; refusing CPU fallback")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        if args.device == "cpu" and sys.platform == "darwin":
            env["GLOO_SOCKET_IFNAME"] = "lo0"
        for phase, command in zip(("reference", "resumed"), commands, strict=True):
            assert source_identity() == identity, "Source changed during acceptance"
            run_job(command, output / f"{phase}.log", env, args.timeout)
        report["comparison"] = compare(output / "reference", output / "resumed", args.steps)
        assert source_identity() == identity, "Source changed during acceptance"
        report.update(status="passed", strict_passed=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    return 0 if report["strict_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
