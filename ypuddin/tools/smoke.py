"""One-command end-to-end check of a model family on the current machine.

``ypuddin smoke config.toml`` runs the *real* trainer for a handful of steps (synthesising a tiny
dataset when the config has none), generates one preview, saves and re-loads the adapter file and
prints a report: load / step timings, peak VRAM, loss values, exported key format. It exists so that
a new machine (or a new family such as Anima with the official weights) can be validated with one
command whose output is easy to paste into a bug report.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import time
from pathlib import Path
from typing import Any

import torch

from ypuddin.config import TrainConfig
from ypuddin.train.events import Emitter

log = logging.getLogger(__name__)


def synthesize_dataset(root: Path, n: int = 4, size: int = 256) -> Path:
    """A few gradient images with tag captions -- enough to exercise indexing, buckets and caching."""
    import numpy as np
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    tags = ["1girl", "solo", "smile", "red_hair", "blue_eyes", "outdoors", "night", "cat_ears"]
    for i in range(n):
        w, h = (size, size) if i % 2 == 0 else (size, size * 3 // 4)
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = (i * 61) % 255
        arr[..., 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        arr[..., 2] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        Image.fromarray(arr).save(root / f"smoke_{i:02d}.png")
        (root / f"smoke_{i:02d}.txt").write_text(", ".join(tags[i : i + 4]), encoding="utf-8")
    return root


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": p.name, "total_mb": round(p.total_memory / 2**20)})
    return info


def smoke_config(
    cfg: TrainConfig, *, out: Path, steps: int, resolution: int, sample_size: int, sample_steps: int
) -> TrainConfig:
    """The user's config with everything that is not under test dialled down to a few seconds of work."""
    data = cfg.to_dict()
    if not data["dataset"].get("sources"):
        data["dataset"]["sources"] = [{"path": str(synthesize_dataset(out / "data")), "repeats": 1}]
    data["dataset"].update({"resolutions": [resolution], "num_workers": 0, "batch_size": 1})
    data["loop"].update({"max_steps": steps, "epochs": None, "grad_accum": 1})
    data["checkpoint"].update(
        {
            "output_dir": str(out),
            "name": "smoke",
            "save_every_epochs": None,
            "save_every_steps": None,
            "resume": None,
        }
    )
    data["validation"] = {"enabled": False}
    data["sampling"] = {
        **{
            key: data["sampling"][key]
            for key in ("sampler", "scheduler", "er_sde_order", "er_sde_s_noise")
            if key in data["sampling"]
        },
        "enabled": True,
        "every_steps": None,
        "every_epochs": None,
        "at_start": False,
        "prompts": [
            {
                "prompt": "1girl, smile, outdoors",
                "negative": "",
                "seed": 1,
                "width": sample_size,
                "height": sample_size,
                "steps": sample_steps,
            }
        ],
        "width": sample_size,
        "height": sample_size,
    }
    data["logging"] = {**data.get("logging", {}), "events_path": None}
    return TrainConfig.model_validate(data)


def run_smoke(
    cfg: TrainConfig,
    *,
    out: Path,
    steps: int = 3,
    resolution: int = 512,
    sample_size: int = 512,
    sample_steps: int = 8,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    from ypuddin.adapters import load_adapter_file
    from ypuddin.train import Trainer

    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": False,
        "environment": environment(),
        "checks": [],
        "timings_s": {},
        "memory_mb": {},
    }

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if verbose:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)

    scfg = smoke_config(
        cfg, out=out, steps=steps, resolution=resolution, sample_size=sample_size, sample_steps=sample_steps
    )
    report["config_hash"] = __import__("ypuddin.config", fromlist=["config_hash"]).config_hash(scfg)
    losses: list[float] = []
    step_times: list[float] = []

    def on_event(e: dict[str, Any]) -> None:
        t = e["type"]
        if t == "phase.changed" and verbose:
            print(f"  phase: {e['phase']}", flush=True)
        elif t == "cache.progress" and verbose and e.get("done") == e.get("total"):
            print(f"  cached {e['kind']}: {e['done']}/{e['total']}", flush=True)
        elif t == "step":
            losses.append(float(e["loss"]))
            if e.get("it_s"):
                step_times.append(1.0 / float(e["it_s"]))
            if verbose:
                print(
                    f"  step {e['step']}: loss={e['loss']:.4f} grad_norm={e.get('grad_norm', float('nan')):.3f} it/s={e.get('it_s') or 0:.2f} vram={e.get('vram_mb') or 0:.0f}MB",
                    flush=True,
                )
        elif t == "warning" and verbose:
            print(f"  warning: {e.get('message') or e}", flush=True)

    emitter = Emitter(path=out / "events.jsonl", listeners=[on_event])
    cuda = torch.cuda.is_available() and (device is None or device.startswith("cuda"))
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    trainer = Trainer(scfg, device=device, emitter=emitter)
    try:
        t0 = time.perf_counter()
        trainer.prepare()
        report["timings_s"]["prepare"] = round(time.perf_counter() - t0, 2)
        if cuda:
            report["memory_mb"]["after_load"] = round(torch.cuda.memory_allocated(trainer.device) / 2**20)
        loaded = trainer.loaded
        report["device"] = str(trainer.device)
        report["family"] = trainer.family.spec.name
        report["dit_config"] = {
            k: v
            for k, v in (loaded.extra or {}).get("dit_config", {}).items()
            if isinstance(v, (int, float, str, bool))
        }
        report["adapters"] = trainer.adapters.summary()
        report["total_steps"] = trainer.progress.total_steps
        check(
            "model loaded",
            True,
            f"family={report['family']} device={report['device']} trainable={report['adapters'].get('trainable_params')}",
        )
        check(
            "adapters injected",
            len(trainer.adapters.layers) > 0,
            f"{len(trainer.adapters.layers)} layers, by_algo={report['adapters'].get('by_algo')}",
        )

        t0 = time.perf_counter()
        outcome = trainer.run()
        report["timings_s"]["train"] = round(time.perf_counter() - t0, 2)
        report["outcome"] = outcome
        report["losses"] = losses
        check("training finished", outcome == "finished", f"outcome={outcome} steps={len(losses)}")
        check(
            "losses finite",
            bool(losses) and all(math.isfinite(v) for v in losses),
            f"losses={[round(v, 4) for v in losses]}",
        )
        if step_times:
            report["timings_s"]["per_step_median"] = round(sorted(step_times)[len(step_times) // 2], 3)

        t0 = time.perf_counter()
        samples = trainer.sample_images("smoke")
        report["timings_s"]["sample"] = round(time.perf_counter() - t0, 2)
        ok_sample = bool(samples) and all(p.exists() and p.stat().st_size > 0 for p in samples)
        detail = f"{samples[0]}" if samples else "no sample produced"
        if ok_sample:
            from PIL import Image

            im = Image.open(samples[0]).convert("RGB")
            px = list(im.resize((16, 16)).getdata())
            flat = [c for p in px for c in p]
            spread = max(flat) - min(flat)
            report["sample"] = {"path": str(samples[0]), "size": im.size, "value_spread": spread}
            detail += f" size={im.size} value_spread={spread}"
            check("sample is not a flat image", spread > 8, detail)
        check("sample saved", ok_sample, detail)

        path = trainer.save_weights("smoke")
        tensors, meta = load_adapter_file(path)
        prefix = trainer.family.spec.adapter_prefix + "_"
        bad = [k for k in tensors if not k.startswith(prefix)]
        modules = {k.partition(".")[0] for k in tensors}
        report["export"] = {
            "path": str(path),
            "tensors": len(tensors),
            "modules": len(modules),
            "metadata_keys": sorted(meta)[:40],
            "example_keys": sorted(tensors)[:6],
        }
        check(
            "adapter file keys use the family prefix",
            not bad,
            f"{len(tensors)} tensors, {len(modules)} modules, e.g. {sorted(tensors)[:2]}",
        )
        check(
            "adapter file round-trips",
            len(modules) == len(trainer.adapters.layers),
            f"{len(modules)} modules on disk vs {len(trainer.adapters.layers)} injected",
        )
        if cuda:
            report["memory_mb"]["peak"] = round(torch.cuda.max_memory_allocated(trainer.device) / 2**20)
            report["memory_mb"]["peak_reserved"] = round(
                torch.cuda.max_memory_reserved(trainer.device) / 2**20
            )
        report["ok"] = all(c["ok"] for c in report["checks"])
    except Exception as e:  # noqa: BLE001
        import traceback

        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        check("no exception", False, report["error"])
    finally:
        emitter.close()
    (out / "smoke-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return report


def format_report(report: dict[str, Any]) -> str:
    env = report["environment"]
    gpus = ", ".join(f"{g['name']} ({g['total_mb']} MB)" for g in env["gpus"]) or (
        "mps" if env.get("mps_available") else "cpu only"
    )
    lines = [
        f"ypuddin smoke -- {'PASS' if report.get('ok') else 'FAIL'}",
        f"  torch {env['torch']} cuda={env['cuda']} gpus: {gpus}",
        f"  family={report.get('family')} device={report.get('device')} dit={report.get('dit_config') or '-'}",
        f"  timings(s): {report.get('timings_s')}",
        f"  memory(MB): {report.get('memory_mb') or '-'}",
        f"  losses: {[round(v, 4) for v in report.get('losses', [])]}",
    ]
    for c in report["checks"]:
        lines.append(
            f"  [{'ok' if c['ok'] else 'FAIL'}] {c['name']}" + (f" -- {c['detail']}" if c["detail"] else "")
        )
    if report.get("error"):
        lines.append(f"  error: {report['error']}")
    return "\n".join(lines)
