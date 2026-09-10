"""``ypuddin`` command line: train / cache / plan / validate / schema / inspect / convert / serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import ypuddin
from ypuddin.config import TrainConfig, dump_toml, load_config


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", nargs="?", help="config .toml/.json (optional when using --preset/--set only)")
    p.add_argument("--preset", action="append", default=[], help="preset file(s) applied before the config")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. loop.epochs=5"
    )


def _load(args: argparse.Namespace) -> TrainConfig:
    return load_config(args.config, presets=args.preset, overrides=args.set)


def cmd_train(args: argparse.Namespace) -> int:
    from ypuddin.train import train

    cfg = _load(args)
    outcome = train(cfg, device=args.device)
    print(f"training {outcome}")
    return 0 if outcome in ("finished", "paused") else 1


def cmd_cache(args: argparse.Namespace) -> int:
    import torch

    from ypuddin.data import build_data, cache_latents
    from ypuddin.models import get_family

    cfg = _load(args)
    fam = get_family(cfg.model.family)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded = fam.load(
        cfg.model, cfg.memory, device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16
    )
    cache_root = (
        Path(cfg.dataset.cache_dir) if cfg.dataset.cache_dir else Path(cfg.checkpoint.output_dir) / "cache"
    )
    bundle = build_data(cfg, fam.spec.latent, cache_root=cache_root)
    n = cache_latents(
        bundle,
        loaded.latent.encode,
        device=device,
        batch_size=max(1, cfg.dataset.batch_size),
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )
    print(json.dumps({"written": n, **bundle.plan.to_dict()}, indent=2, ensure_ascii=False))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from ypuddin.train.plan import plan

    cfg = _load(args)
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0).total_memory / 2**20
    except Exception:  # noqa: BLE001
        pass
    result = plan(cfg, gpu_total_mb=gpu)
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
    from ypuddin.adapters.convert import comfy_to_kohya, kohya_to_comfy, lycoris_to_kohya
    from ypuddin.models import get_family

    tensors, meta = load_adapter_file(args.file)
    if args.to == "comfyui":
        fam = get_family(args.family)
        names = fam.linear_module_names() if hasattr(fam, "linear_module_names") else []
        out = kohya_to_comfy(tensors, names)
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

        fam = get_family(args.family)
        names = fam.linear_module_names() if hasattr(fam, "linear_module_names") else None
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
    except ImportError as e:
        print(f"server dependencies missing: {e}. Install with: pip install 'ypuddin[server]'")
        return 1

    app = create_app(data_root=args.data_root)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ypuddin", description="YPuddin Train Studio")
    p.add_argument("--version", action="version", version=ypuddin.__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="run a training job")
    _add_config_args(t)
    t.add_argument("--device", default=None)
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("cache", help="pre-encode latents only")
    _add_config_args(c)
    c.add_argument("--device", default=None)
    c.set_defaults(fn=cmd_cache)

    pl = sub.add_parser("plan", help="validate and estimate steps / buckets / VRAM without loading weights")
    _add_config_args(pl)
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
        "config", nargs="?", help="config file; only model/adapter/memory sections matter, defaults otherwise"
    )
    sm.add_argument("--preset", action="append", default=[])
    sm.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    sm.add_argument("--out", default="outputs/smoke")
    sm.add_argument("--steps", type=int, default=3)
    sm.add_argument("--resolution", type=int, default=512, help="training resolution for the smoke dataset")
    sm.add_argument("--sample-size", type=int, default=512)
    sm.add_argument("--sample-steps", type=int, default=8)
    sm.add_argument("--device", default=None, help="cuda / cuda:1 / cpu (default: auto)")
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
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--data-root", default="studio_data")
    sv.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
