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
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. loop.epochs=5")


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
    loaded = fam.load(cfg.model, cfg.memory, device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    cache_root = Path(cfg.dataset.cache_dir) if cfg.dataset.cache_dir else Path(cfg.checkpoint.output_dir) / "cache"
    bundle = build_data(cfg, fam.spec.latent, cache_root=cache_root)
    n = cache_latents(bundle, loaded.latent.encode, device=device, batch_size=max(1, cfg.dataset.batch_size), dtype=torch.float32 if device == "cpu" else torch.bfloat16)
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
    from ypuddin.adapters import group_by_module, load_adapter_file, detect_algo

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

    sv = sub.add_parser("serve", help="run the HTTP service for the web UI")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--data-root", default="studio_data")
    sv.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
