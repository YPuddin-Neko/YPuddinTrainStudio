"""``ypuddin`` command line: train / cache / plan / validate / schema / inspect / convert / serve."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import ypuddin
from ypuddin import worker_log
from ypuddin.config import TrainConfig, dump_toml, load_config

# Named explicitly: workers run this module as ``__main__``.
log = logging.getLogger("ypuddin.cli")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", nargs="?", help="config .toml/.json (optional when using --preset/--set only)")
    p.add_argument("--preset", action="append", default=[], help="preset file(s) applied before the config")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. loop.epochs=5"
    )


def _load(args: argparse.Namespace) -> TrainConfig:
    return load_config(args.config, presets=args.preset, overrides=args.set)


def _worker_config(args: argparse.Namespace) -> TrainConfig:
    worker_log.configure("debug" if args.verbose else "info")
    cfg = _load(args)
    worker_log.configure("debug" if args.verbose else cfg.logging.level)
    return cfg


def cmd_train(args: argparse.Namespace) -> int:
    from ypuddin.runtime_attention import AttentionEnvironmentError
    from ypuddin.train import train

    cfg = _worker_config(args)
    try:
        outcome = train(cfg, device=args.device)
    except AttentionEnvironmentError as exc:
        if args.verbose:
            raise
        if int(os.environ.get("RANK", "0")) == 0:
            log.error("%s", exc)
        return 1
    if int(os.environ.get("RANK", "0")) == 0:
        log.info("training %s", outcome)
    return 0 if outcome in ("finished", "paused") else 1


def cmd_cache(args: argparse.Namespace) -> int:
    from ypuddin.train import cache

    cfg = _worker_config(args)
    outcome = cache(cfg, device=args.device)
    log.info("caching %s", outcome)
    return 0 if outcome == "finished" else 1


def cmd_plan(args: argparse.Namespace) -> int:
    from ypuddin.train.plan import plan

    cfg = _load(args)
    gpu = None
    device = args.device
    try:
        import torch

        if device is None:
            from ypuddin.runtime_profiles import current_profile

            device = (
                "cpu"
                if current_profile().endswith("-cpu")
                else "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        target = torch.device(device)
        if target.type == "cuda" and torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(target.index or 0).total_memory / 2**20
        elif target.type == "mps" and torch.backends.mps.is_available():
            try:
                import psutil

                gpu = psutil.virtual_memory().total / 2**20
            except ImportError:
                gpu = torch.mps.recommended_max_memory() / 2**20
    except Exception:  # noqa: BLE001
        pass
    result = plan(cfg, gpu_total_mb=gpu, device=device or "cpu")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except Exception as e:  # noqa: BLE001
        print(f"invalid: {e}")
        return 1
    print(dump_toml(cfg) if args.dump else "ok")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    print(json.dumps(TrainConfig.json_schema(), indent=2, ensure_ascii=False))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from ypuddin.adapters import detect_algo, group_by_module, load_adapter_file

    tensors, meta = load_adapter_file(args.file)
    groups = group_by_module(tensors)
    total = sum(t.numel() for t in tensors.values())
    print(f"{args.file}: {len(groups)} modules, {total:,} parameters")
    for k, v in sorted(meta.items()):
        if k in ("ypuddin.targets",):
            continue
        print(f"  {k} = {v[:120]}")
    if args.verbose:
        for name, sub in groups.items():
            shapes = ", ".join(f"{s}{tuple(t.shape)}" for s, t in sub.items())
            print(f"  {name} [{detect_algo(set(sub))}] {shapes}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from safetensors.torch import save_file

    from ypuddin.adapters import load_adapter_file
    from ypuddin.adapters.convert import comfy_to_kohya, kohya_to_comfy, lycoris_to_kohya, modernize_text_keys
    from ypuddin.models import get_family

    tensors, meta = modernize_text_keys(*load_adapter_file(args.file))
    if args.to == "comfyui":
        out = kohya_to_comfy(tensors, list(get_family(args.family).adaptable_modules()))
    elif args.to == "kohya":
        out = comfy_to_kohya(lycoris_to_kohya(tensors))
    else:
        raise SystemExit(f"unknown target {args.to}")
    save_file(out, args.output, metadata=meta)
    print(f"wrote {args.output} ({len(out)} tensors)")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from safetensors.torch import load_file, save_file

    from ypuddin.adapters import build_metadata
    from ypuddin.tools import extract_from_state_dicts

    base, tuned = load_file(args.base), load_file(args.tuned)
    rank = "full" if args.rank == "full" else int(args.rank)
    tensors, report = extract_from_state_dicts(
        base,
        tuned,
        algo=args.algo,
        rank=rank,
        factor=args.factor,
        prefix=args.prefix,
        progress=lambda d, t: print(f"\r{d}/{t}", end="", flush=True),
    )
    print()
    worst = sorted(report.items(), key=lambda kv: -kv[1]["residual"])[:5]
    for name, rep in worst:
        print(f"  residual {rep['residual']:.4f}  {name}")
    meta = build_metadata(
        targets=report,
        adapter_cfg={"algo": args.algo, "rank": rank, "alpha": None, "factor": args.factor},
        family=args.family,
        architecture=f"{args.family}/{args.algo}",
        title=Path(args.output).stem,
    )
    save_file({k: v.contiguous() for k, v in tensors.items()}, args.output, metadata=meta)
    print(f"wrote {args.output}: {len(report)} modules")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from ypuddin.tools.smoke import format_report, run_smoke

    cfg = _load(args)
    report = run_smoke(
        cfg,
        out=Path(args.out),
        steps=args.steps,
        resolution=args.resolution,
        sample_size=args.sample_size,
        sample_steps=args.sample_steps,
        device=args.device,
    )
    print(format_report(report))
    print(f"report: {Path(args.out) / 'smoke-report.json'}")
    return 0 if report["ok"] else 1


def cmd_merge(args: argparse.Namespace) -> int:
    from safetensors.torch import load_file, save_file

    from ypuddin.adapters import load_adapter_file
    from ypuddin.tools import merge_into_state_dict

    base = load_file(args.base)
    tensors, meta = load_adapter_file(args.adapter)
    names = None
    if args.family:
        from ypuddin.models import get_family

        names = list(get_family(args.family).adaptable_modules())
    merged, unmatched = merge_into_state_dict(
        base,
        tensors,
        meta,
        prefix=args.prefix,
        strength=args.strength,
        module_names=names,
        requantize_fp8=args.fp8,
    )
    if unmatched:
        print(f"warning: {len(unmatched)} adapter modules did not match the base model, e.g. {unmatched[:3]}")
    save_file({k: v.contiguous() for k, v in merged.items()}, args.output)
    print(f"wrote {args.output}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from ypuddin.server.app import create_app
        from ypuddin.server.http_server import StudioServer
        from ypuddin.server.lifecycle import RESTART_TOKEN_ENV, launch_service
    except ImportError as e:
        print(f"server dependencies missing: {e}. Install with: pip install 'ypuddin[server]'")
        return 1
    if not args.service_worker:
        return launch_service(args.data_root, args.host, args.port)
    app = create_app(data_root=args.data_root)
    settings = app.state.ctx.settings()["server"]
    host, port = args.host or settings["host"], args.port or settings["port"]
    server = StudioServer(
        uvicorn.Config(app, host=host, port=port, log_level="info", timeout_graceful_shutdown=5),
        event_bus=app.state.ctx.bus,
    )
    if args.control_file and args.original_python:
        import os

        app.state.lifecycle.configure(
            control_file=Path(args.control_file),
            host=host,
            port=port,
            shutdown=lambda: setattr(server, "should_exit", True),
            original_python=args.original_python,
            # Keep the capability in this worker, not in later training/probe subprocesses.
            restart_token=os.environ.pop(RESTART_TOKEN_ENV, None),
        )
    try:
        server.run()
    except KeyboardInterrupt:
        # Uvicorn lets the asyncio runner re-raise Ctrl+C after its graceful
        # shutdown logs. Treat an interactive stop as a normal service exit so
        # the launcher never prints a misleading traceback.
        return 0
    return 0 if server.started else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ypuddin", description="YPuddin Train Studio")
    p.add_argument("--version", action="version", version=ypuddin.__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="run a training job")
    _add_config_args(t)
    t.add_argument("--device", default=None)
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("cache", help="pre-encode latent and text caches without training")
    _add_config_args(c)
    c.add_argument("--device", default=None)
    c.set_defaults(fn=cmd_cache)

    pl = sub.add_parser("plan", help="validate and estimate steps / buckets / VRAM without loading weights")
    _add_config_args(pl)
    pl.add_argument(
        "--device",
        default=None,
        help="target device (auto-detect by default; cuda permits offline GPU planning)",
    )
    pl.set_defaults(fn=cmd_plan)

    v = sub.add_parser("validate", help="validate a config")
    _add_config_args(v)
    v.add_argument("--dump", action="store_true", help="print the resolved config as TOML")
    v.set_defaults(fn=cmd_validate)

    s = sub.add_parser("schema", help="print the JSON Schema of TrainConfig (with x-ui hints)")
    s.set_defaults(fn=cmd_schema)

    i = sub.add_parser("inspect", help="show metadata and modules of an adapter file")
    i.add_argument("file")
    i.set_defaults(fn=cmd_inspect)

    cv = sub.add_parser("convert", help="convert adapter key formats")
    cv.add_argument("file")
    cv.add_argument("--to", choices=["comfyui", "kohya"], required=True)
    cv.add_argument("--family", default="anima")
    cv.add_argument("-o", "--output", required=True)
    cv.set_defaults(fn=cmd_convert)

    ex = sub.add_parser("extract", help="extract a LoRA/LoKr from the difference of two full models")
    ex.add_argument("--base", required=True)
    ex.add_argument("--tuned", required=True)
    ex.add_argument("--algo", choices=["lora", "lokr"], default="lokr")
    ex.add_argument("--rank", default="full")
    ex.add_argument("--factor", type=int, default=-1)
    ex.add_argument("--prefix", default="lora_unet")
    ex.add_argument("--family", default="anima")
    ex.add_argument("-o", "--output", required=True)
    ex.set_defaults(fn=cmd_extract)

    sm = sub.add_parser(
        "smoke", help="run a few real training steps + one preview + save/reload and print a report"
    )
    sm.add_argument(
        "config",
        nargs="?",
        help="config file; steps, resolution, batch size, checkpoints, validation and previews are "
        "overridden, and synthetic images are used when no dataset source is set",
    )
    sm.add_argument("--preset", action="append", default=[])
    sm.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    sm.add_argument("--out", default="outputs/smoke")
    sm.add_argument("--steps", type=int, default=3)
    sm.add_argument("--resolution", type=int, default=512, help="training resolution for the smoke dataset")
    sm.add_argument("--sample-size", type=int, default=512)
    sm.add_argument("--sample-steps", type=int, default=8)
    sm.add_argument("--device", default=None, help="cuda / cuda:1 / mps / cpu (default: auto)")
    sm.set_defaults(fn=cmd_smoke)

    mg = sub.add_parser("merge", help="merge an adapter into base weights")
    mg.add_argument("--base", required=True)
    mg.add_argument("--adapter", required=True)
    mg.add_argument("--strength", type=float, default=1.0)
    mg.add_argument("--prefix", default="lora_unet")
    mg.add_argument("--family", default=None, help="model family used to resolve module names (e.g. anima)")
    mg.add_argument(
        "--fp8", choices=["fp8_e4m3", "fp8_e5m2"], default=None, help="re-quantize merged weights to fp8"
    )
    mg.add_argument("-o", "--output", required=True)
    mg.set_defaults(fn=cmd_merge)

    sv = sub.add_parser("serve", help="run the HTTP service for the web UI")
    sv.add_argument("--host", default=None, help="override the saved server host")
    sv.add_argument("--port", type=int, default=None, help="override the saved server port")
    sv.add_argument("--data-root", default="studio_data")
    sv.add_argument("--service-worker", action="store_true", help=argparse.SUPPRESS)
    sv.add_argument("--control-file", help=argparse.SUPPRESS)
    sv.add_argument("--original-python", help=argparse.SUPPRESS)
    sv.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=worker_log.FORMAT)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
