"""Isolated, cancellable XYZ inference using the real family and adapter implementations."""

from __future__ import annotations

import gc
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from ypuddin.models.precision import model_load_precision

from .xyz import AXES, XyzRequest, checkpoint_signature, expand_cells, full_checkpoint_model


def _atomic_json(path: Path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def bind_checkpoint(backbone, path: Path, family: str, prefix: str, *, text=None):
    """Rebuild each exported adapter, requiring complete name/shape compatibility."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    from ypuddin.adapters import AdaptedLinear, FrozenLinear, load_adapter_file, modules_from_tensors
    from ypuddin.adapters.components import TEXT_ADAPTER_PREFIXES

    class ScaledAdapter(AdaptedLinear):
        def forward(self, x):
            if self.dora is None:
                return super().forward(x)
            base = self.base(x)
            if self.multiplier == 0:
                return base
            weight = self.dora.rescale(self.base.dequant(torch.float32) + self.adapter.delta_weight().float())
            bias = None if self.base.bias is None else self.base.bias.to(x.dtype)
            adapted = F.linear(x, weight.to(x.dtype), bias)
            return base + self.multiplier * (adapted - base)

    tensors, metadata = load_adapter_file(path)
    if metadata.get("ypuddin.family", family) != family:
        raise ValueError("Adapter model family differs from the selected sampling model")
    keys = {key.partition(".")[0] for key in tensors}
    component_models = {"backbone": (prefix, backbone)}
    text_components = [
        name
        for name, token in TEXT_ADAPTER_PREFIXES.items()
        if any(key.startswith(token + "_") for key in keys)
    ]
    if text_components:
        if text is None:
            raise ValueError("Checkpoint contains text adapters but no text pipeline was supplied")
        encoders = text.trainable_modules()
        for name in text_components:
            if name not in encoders:
                raise ValueError(f"Checkpoint text component is absent from the sampling model: {name}")
            component_models[name] = (TEXT_ADAPTER_PREFIXES[name], encoders[name])
    plans, matched = [], set()
    for component, (component_prefix, model) in component_models.items():
        modules = modules_from_tensors(tensors, metadata, prefix=component_prefix)
        matched.update(modules)
        names = {
            component_prefix + "_" + name.replace(".", "_"): (name, module)
            for name, module in model.named_modules()
            if isinstance(module, (nn.Linear, FrozenLinear))
        }
        for key, (adapter, dora) in modules.items():
            if key not in names:
                raise ValueError(f"Checkpoint target is absent from the sampling model: {key}")
            name, original = names[key]
            if (original.in_features, original.out_features) != (adapter.in_features, adapter.out_features):
                raise ValueError(f"Checkpoint target shape differs from the sampling model: {name}")
            base = original if isinstance(original, FrozenLinear) else FrozenLinear.from_linear(original)
            wrapper = ScaledAdapter(base, adapter, dora=dora is not None, name=name)
            wrapper.component = component
            if dora is not None:
                wrapper.dora.load_tensor(dora)
            wrapper.requires_grad_(False).eval()
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            # Restore frozen bases without retaining a second complete CPU weight copy.
            plans.append((parent, attr, base, wrapper))
    if not matched or keys != matched:
        raise ValueError("Checkpoint includes unsupported or unmatched adapter modules")
    for parent, attr, _, wrapper in plans:
        setattr(parent, attr, wrapper)
    return plans


def _font():
    from PIL import ImageFont

    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in (
        windows_fonts / "msyh.ttc",
        windows_fonts / "simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "DejaVuSans.ttf",
        "arial.ttf",
    ):
        try:
            return ImageFont.truetype(str(name), 16)
        except OSError:
            pass
    return ImageFont.load_default()


def label_lines(text, font, width, max_lines=2):
    """Wrap Unicode by rendered width, retaining an ellipsis when the label cannot fit."""
    text = " ".join(str(text).split())
    lines, line = [], ""
    for char in text:
        if line and font.getlength(line + char) > width:
            lines.append(line)
            line = ""
        line += char
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and font.getlength(last + "…") > width:
            last = last[:-1]
        lines[-1] = last + "…"
    return lines


def write_grids(output, request, manifest):
    from PIL import Image, ImageDraw

    nx, ny = len(request.x.values), len(request.y.values) if request.y else 1
    axes = manifest["axes"]
    font = _font()
    grids = []
    for zi in range(len(request.z.values) if request.z else 1):
        cells = [cell for cell in manifest["cells"] if cell["z"] == zi]
        if not cells:
            continue
        margin, title = 144, 124
        grid = Image.new("RGB", (margin + nx * request.width, title + ny * (request.height + 24)), "#20242c")
        draw = ImageDraw.Draw(grid)

        def label(value, x, y, width, lines=2, canvas=draw):
            for index, line in enumerate(label_lines(value, font, width, lines)):
                canvas.text((x, y + index * 22), line, fill="white", font=font)

        label(
            f"X: {AXES[request.x.key]} | Y: {AXES[request.y.key] if request.y else '-'} | Z: {AXES[request.z.key] if request.z else '-'}",
            8,
            4,
            grid.width - 16,
        )
        if request.z:
            label(axes["z"]["labels"][zi], 8, 48, grid.width - 16, 1)
        for xi, value in enumerate(axes["x"]["labels"]):
            label(value, margin + xi * request.width + 4, title - 48, request.width - 8)
        for yi in range(ny):
            if request.y:
                label(axes["y"]["labels"][yi], 4, title + yi * (request.height + 24) + 4, margin - 8)
        for cell in cells:
            with Image.open(output / cell["file"]) as image:
                grid.paste(
                    image.convert("RGB"),
                    (margin + cell["x"] * request.width, title + cell["y"] * (request.height + 24)),
                )
        name = f"grid_z{zi:02d}.png"
        temporary = output / (name + ".tmp")
        grid.save(temporary, format="PNG")
        temporary.replace(output / name)
        grid.close()
        grids.append(
            {"z": zi, "z_value": request.z.values[zi] if request.z else None, "file": name, "axes": axes}
        )
    manifest["grids"] = grids


def generate(payload: dict, output: Path, emit, cancelled):
    import torch
    from PIL import Image

    from ypuddin.config import MemoryConfig, ModelConfig, SamplingConfig
    from ypuddin.memory import BlockSwapper
    from ypuddin.models import get_family
    from ypuddin.models.fingerprints import fingerprint_cache

    torch.set_num_threads(min(torch.get_num_threads(), 2))
    request = XyzRequest.model_validate(payload["xyz"]["request"])
    model = ModelConfig.model_validate(payload["model"])
    memory = MemoryConfig.model_validate(payload["memory"])
    family = get_family(model.family)
    cells = expand_cells(request)
    checkpoints = payload["xyz"]["checkpoints"]
    full = payload.get("training", {}).get("mode") == "full"
    if full:
        if (
            request.sampling_model_id
            or request.adapter_scale != 1
            or any(axis and axis.key == "adapter_scale" for axis in (request.x, request.y, request.z))
        ):
            raise ValueError("Full-model XYZ cannot apply adapters or replace the exported backbone")
        if any(
            not cell["checkpoint_id"] or checkpoints.get(cell["checkpoint_id"], {}).get("kind") != "model"
            for cell in cells
        ):
            raise ValueError(
                "Full-model XYZ requires an exported model in every cell; base fallback is disabled"
            )
    elif any(checkpoint.get("kind") == "model" for checkpoint in checkpoints.values()):
        raise ValueError("Full-model checkpoints require full-model XYZ mode")
    device = torch.device(payload["device"])
    dtype = (
        getattr(
            torch,
            {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[
                model_load_precision(model, device.type)
            ],
        )
        if device.type == "cuda"
        else torch.float32
    )
    output.mkdir(parents=True, exist_ok=True)
    axes = {}
    for key in ("x", "y", "z"):
        axis = getattr(request, key)
        if axis:
            axes[key] = axis.model_dump() | {
                "label": AXES[axis.key],
                "labels": [
                    checkpoints[value]["name"] if axis.key == "checkpoint" else str(value)
                    for value in axis.values
                ],
            }
    manifest = {"cells": [], "grids": [], "axes": axes, "complete": False}
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    loaded, swapper, bindings = None, None, []

    def check():
        if cancelled():
            raise InterruptedError("XYZ sampling cancelled")

    def park():
        nonlocal swapper
        if swapper:
            swapper.release_all()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            # Removing inference hooks must not first upload all swapped blocks.
            swapper.device = torch.device("cpu")
            swapper.remove()
            swapper = None
        if loaded and loaded.extra.get("materialized", True):
            loaded.backbone.to("cpu")
            loaded.text.to("cpu")

    conditions = {}
    base_conditions = {}
    needs_uncond = getattr(family, "sampling_needs_uncond", lambda _loaded, cfg: cfg != 1)

    def load_selection(selected_model):
        nonlocal loaded, model, dtype, conditions, base_conditions
        check()
        model = selected_model
        dtype = (
            getattr(
                torch,
                {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[
                    model_load_precision(model, device.type)
                ],
            )
            if device.type == "cuda"
            else torch.float32
        )
        emit("phase.changed", phase="loading")
        with fingerprint_cache(Path(payload["fingerprint_cache"])):
            loaded = family.load(model, memory, device=device, dtype=dtype, backbone_device="cpu")
        loaded.backbone.eval().requires_grad_(False)
        conditions = {}
        prompts = [request.prompt]
        if any(needs_uncond(loaded, cell["cfg"]) for cell in cells):
            prompts.append(request.negative)
        emit("phase.changed", phase="encoding_text")
        loaded.text.to(device)
        for prompt in dict.fromkeys(prompts):
            check()
            conditions[prompt] = loaded.text.encode([prompt], device=device).to("cpu")
        base_conditions = dict(conditions)
        loaded.text.unload()
        check()
        family.materialize_backbone(loaded)
        loaded.backbone.eval().requires_grad_(False)

    try:
        check()
        with torch.inference_mode():
            if not full:
                load_selection(model)
            uninitialized_checkpoint = object()
            previous_checkpoint = uninitialized_checkpoint
            for cell in cells:
                check()
                sampling = SamplingConfig.model_validate(
                    {key: cell[key] for key in ("sampler", "scheduler", "steps", "cfg", "shift", "guidance")}
                )
                errors = family.sampling_errors(sampling)
                if errors:
                    raise ValueError("; ".join(item["msg"] for item in errors))
                if cell["checkpoint_id"] != previous_checkpoint:
                    # First binding keeps the materializer's placement (which
                    # may stage on CUDA before creating pinned swap masters).
                    if previous_checkpoint is not uninitialized_checkpoint:
                        park()
                    for parent, attr, original, _ in bindings:
                        setattr(parent, attr, original)
                    bindings.clear()
                    conditions = dict(base_conditions)
                    if full:
                        if loaded:
                            loaded.text.unload()
                            loaded.latent.unload()
                        loaded = None
                        conditions.clear()
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        checkpoint = checkpoints[cell["checkpoint_id"]]
                        path = Path(checkpoint["path"])
                        if checkpoint_signature(path) != checkpoint["signature"]:
                            raise ValueError("Full-model checkpoint changed after this comparison was queued")
                        selected_model = full_checkpoint_model(path, model.family)
                        if selected_model.model_dump(mode="json") != checkpoint["model"]:
                            raise ValueError("Full-model checkpoint configuration changed after queueing")
                        load_selection(selected_model)
                    elif cell["checkpoint_id"]:
                        checkpoint = checkpoints[cell["checkpoint_id"]]
                        path = Path(checkpoint["path"])
                        if checkpoint_signature(path) != checkpoint["signature"]:
                            raise ValueError("Checkpoint changed after this comparison was queued")
                        loaded.text.to("cpu")
                        bindings = bind_checkpoint(
                            loaded.backbone, path, model.family, family.spec.adapter_prefix, text=loaded.text
                        )
                        if any(wrapper.component != "backbone" for *_, wrapper in bindings):
                            # Materialization may stage the backbone on CUDA. Encode
                            # first without keeping both full base models resident.
                            loaded.backbone.to("cpu")
                    if memory.blocks_to_swap:
                        swapper = BlockSwapper(
                            family.memory_layout(loaded).blocks, memory.blocks_to_swap, device
                        )
                        swapper.set_forward_only(True)
                    previous_checkpoint = cell["checkpoint_id"]
                for *_, wrapper in bindings:
                    wrapper.multiplier = cell["adapter_scale"]
                if any(getattr(wrapper, "component", "backbone") != "backbone" for *_, wrapper in bindings):
                    # Embeddings depend on both checkpoint and scale. Keep the installed
                    # text adapters when parking encoders; unloading would discard them.
                    emit("phase.changed", phase="encoding_text")
                    loaded.text.to(device)
                    conditions = {}
                    for prompt in base_conditions:
                        check()
                        conditions[prompt] = loaded.text.encode([prompt], device=device).to("cpu")
                    loaded.text.to("cpu")
                if swapper:
                    swapper.move_model_to_device(loaded.backbone)
                else:
                    loaded.backbone.to(device)
                loaded.device = device
                defaults = family.sampling_defaults(loaded)
                stride, patch = family.spec.latent.stride, family.spec.latent.patch
                shift = (
                    cell["shift"]
                    if cell["shift"] is not None
                    else family.sampling_shift_for_model(
                        loaded,
                        (request.height // stride // patch) * (request.width // stride // patch),
                        steps=cell["steps"],
                    )
                )
                guidance = cell["guidance"] if cell["guidance"] is not None else defaults.guidance
                cond = conditions[request.prompt].to(device)
                uncond = (
                    conditions[request.negative].to(device) if needs_uncond(loaded, cell["cfg"]) else None
                )
                emit(
                    "xyz.progress",
                    phase="sampling",
                    done=cell["index"],
                    total=len(cells),
                    cell_index=cell["index"],
                    sample_step=0,
                    sample_steps=cell["steps"],
                )

                def predict(x, t, condition=cond, guidance_value=guidance, current_loaded=loaded):
                    check()
                    precision = (
                        torch.autocast("cuda", dtype=dtype)
                        if device.type == "cuda" and dtype != torch.float32
                        else nullcontext()
                    )
                    with precision:
                        return family.forward(
                            current_loaded,
                            x.to(dtype),
                            t.to(device),
                            condition,
                            inference=True,
                            guidance=guidance_value,
                        ).float()

                def step(done, total, cell_index=cell["index"]):
                    check()
                    emit(
                        "xyz.progress",
                        phase="sampling",
                        done=cell_index,
                        total=len(cells),
                        cell_index=cell_index,
                        sample_step=done,
                        sample_steps=total,
                    )

                latents = family.sample_latents(
                    loaded,
                    predict,
                    (1, family.spec.latent.channels, request.height // stride, request.width // stride),
                    steps=cell["steps"],
                    cfg=cell["cfg"],
                    shift=shift,
                    sampler=cell["sampler"],
                    scheduler=cell["scheduler"],
                    predict_uncond=(lambda x, t, condition=uncond, forward=predict: forward(x, t, condition))
                    if uncond is not None
                    else None,
                    generator=torch.Generator().manual_seed(cell["seed"]),
                    device=device,
                    on_step=step,
                )
                if not torch.isfinite(latents).all():
                    raise ValueError("Sampling produced non-finite latents; reduce the comparison settings")
                if swapper:
                    swapper.release_all()
                loaded.backbone.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                check()
                loaded.latent.to(device)
                pixels = loaded.latent.decode(latents).clamp(-1, 1)
                if not torch.isfinite(pixels).all():
                    raise ValueError("VAE decoding produced non-finite pixels")
                array = (
                    ((pixels[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().astype("uint8")
                )
                check()
                name = f"cell_z{cell['z']:02d}_y{cell['y']:02d}_x{cell['x']:02d}.png"
                temporary = output / (name + ".tmp")
                Image.fromarray(array).save(temporary, format="PNG")
                temporary.replace(output / name)
                loaded.latent.to("cpu")
                manifest["cells"].append(
                    cell
                    | {
                        "file": name,
                        "shift": shift,
                        "guidance": guidance,
                        "checkpoint_name": checkpoints[cell["checkpoint_id"]]["name"]
                        if cell["checkpoint_id"]
                        else None,
                    }
                )
                _atomic_json(manifest_path, manifest)
                emit("xyz.progress", phase="sampling", done=len(manifest["cells"]), total=len(cells))
                del latents, pixels, cond, uncond, predict, step
        check()
        manifest["complete"] = True
    finally:
        # Valid cells and partial grids remain reviewable after cancellation/failure.
        if loaded:
            park()
            loaded.text.unload()
            loaded.latent.unload()
        bindings.clear()
        loaded = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        write_grids(output, request, manifest)
        _atomic_json(manifest_path, manifest)


def main():
    request_path = Path(sys.argv[1])
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    run_dir = request_path.parent
    from ypuddin.train.events import Emitter

    emitter = Emitter(path=run_dir / "events.jsonl")
    if payload.get("parent_pid"):
        import threading
        import time

        import psutil

        def watch_parent():
            while True:
                try:
                    parent = psutil.Process(payload["parent_pid"])
                    if parent.create_time() != payload["parent_created"] or not parent.is_running():
                        os._exit(143)
                except psutil.NoSuchProcess:
                    os._exit(143)
                time.sleep(0.5)

        threading.Thread(target=watch_parent, daemon=True).start()
    try:
        generate(
            payload,
            run_dir / "samples",
            emitter.emit,
            lambda: any((run_dir / "control" / name).exists() for name in ("stop", "pause")),
        )
        emitter.emit("run.finished")
        return 0
    except InterruptedError:
        emitter.emit("run.stopped")
        return 130
    except Exception as exc:
        emitter.emit("run.failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        emitter.close()


if __name__ == "__main__":
    raise SystemExit(main())
