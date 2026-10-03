"""Isolated base-model inference for class-prior images. Never constructs an adapter or optimizer."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

from ypuddin.models.precision import model_load_precision


def generate(request: dict, output: Path, emit, cancelled) -> None:
    import torch
    from PIL import Image

    from ypuddin.config import MemoryConfig, ModelConfig, SamplingConfig
    from ypuddin.models import get_family
    from ypuddin.models.fingerprints import fingerprint_cache

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = ModelConfig.model_validate(request["model"])
    sampling = SamplingConfig.model_validate(request.get("sampling", {}))
    family = get_family(model.family)
    errors = family.sampling_errors(sampling)
    if errors:
        raise ValueError("; ".join(f"{item['loc']}: {item['msg']}" for item in errors))
    device = torch.device(request["device"])
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
    width, height = request["width"], request["height"]
    if width % family.spec.latent.align or height % family.spec.latent.align:
        raise ValueError(f"Dimensions must be multiples of {family.spec.latent.align}")
    memory = MemoryConfig(activation_checkpointing="none")

    def check():
        if cancelled():
            raise InterruptedError()

    check()
    emit(phase="loading", done=0, message=f"Loading {model.family} base model without adapters")
    with fingerprint_cache(Path(request["fingerprint_cache"])):
        loaded = family.load(model, memory, device=device, dtype=dtype, backbone_device="cpu")
    loaded.backbone.eval().requires_grad_(False)
    prompts = request["prompts"]
    # Encode each unique prompt once, retain CPU conditioning, then unload the encoder.
    unique = list(dict.fromkeys([*prompts, request["negative"]]))
    conditions = {}
    loaded.text.to(device)
    for prompt in unique:
        check()
        conditions[prompt] = loaded.text.encode([prompt], device=device).to("cpu")
    loaded.text.unload()
    family.materialize_backbone(loaded)
    family.prepare_attention(loaded, model.attention, device=device, dtype=dtype, training=False)
    loaded.backbone.eval().requires_grad_(False)
    loaded.backbone.to(device=device, dtype=dtype)
    loaded.device = device
    stride = family.spec.latent.stride
    patch = family.spec.latent.patch
    defaults = family.sampling_defaults(loaded)
    shift = (
        sampling.shift
        if sampling.shift is not None
        else family.sampling_shift_for_model(
            loaded, (height // stride // patch) * (width // stride // patch), steps=request["steps"]
        )
    )
    guidance = sampling.guidance if sampling.guidance is not None else defaults.guidance
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with torch.inference_mode():
        for index in range(request["count"]):
            check()
            prompt = prompts[index % len(prompts)]
            cond = conditions[prompt].to(device)
            uncond = conditions[request["negative"]].to(device)
            seed = request["seed"] + index

            def predict(x, t, condition=cond):
                check()
                precision = (
                    torch.autocast("cuda", dtype=dtype)
                    if device.type == "cuda" and dtype != torch.float32
                    else nullcontext()
                )
                with precision:
                    return family.forward(
                        loaded,
                        x.to(dtype),
                        t.to(device),
                        condition,
                        inference=True,
                        guidance=guidance,
                    ).float()

            def step(done, total, image_index=index):
                check()
                emit(phase="generating", done=image_index, sample_step=done, sample_steps=total)

            latents = family.sample_latents(
                loaded,
                predict,
                (1, family.spec.latent.channels, height // stride, width // stride),
                sampler=sampling.sampler,
                scheduler=sampling.scheduler,
                er_sde_order=sampling.er_sde_order,
                er_sde_s_noise=sampling.er_sde_s_noise,
                steps=request["steps"],
                shift=shift,
                cfg=request["cfg"],
                predict_uncond=lambda x, t, condition=uncond: predict(x, t, condition),
                generator=torch.Generator().manual_seed(seed),
                device=device,
                on_step=step,
            )
            # Decode with the DiT parked on CPU to avoid a transient DiT/VAE memory peak.
            loaded.backbone.to("cpu")
            loaded.latent.to(device)
            check()
            pixels = loaded.latent.decode(latents).clamp(-1, 1)
            array = ((pixels[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().astype("uint8")
            name = f"prior_{index:04d}_{seed}.png"
            Image.fromarray(array).save(output / name)
            (output / name).with_suffix(".txt").write_text(prompt, encoding="utf-8")
            manifest.append(
                {
                    "file": name,
                    "prompt": prompt,
                    "seed": seed,
                    "model_family": model.family,
                    "sampler": sampling.sampler,
                    "scheduler": sampling.scheduler,
                    "steps": request["steps"],
                    "cfg": request["cfg"],
                    "guidance": guidance,
                    "shift": shift,
                    "er_sde_order": sampling.er_sde_order,
                    "er_sde_s_noise": sampling.er_sde_s_noise,
                    **(
                        {"training_source": request["entries"][index]["training_source"]}
                        if request.get("entries")
                        else {}
                    ),
                }
            )
            loaded.latent.to("cpu")
            if index + 1 < request["count"]:
                loaded.backbone.to(device)
            emit(phase="generating", done=index + 1)
    check()
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    request_path, output, events, stop = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # A hard server exit must not strand a large model process using the GPU.
    # This also covers a crash between Popen and saving the child PID in SQLite.
    if request.get("parent_pid"):
        import os
        import threading
        import time

        import psutil

        def watch_parent():
            while True:
                try:
                    parent = psutil.Process(request["parent_pid"])
                    if parent.create_time() != request["parent_created"] or not parent.is_running():
                        os._exit(143)
                except psutil.NoSuchProcess:
                    os._exit(143)
                time.sleep(0.5)

        threading.Thread(target=watch_parent, daemon=True).start()

    def emit(**fields):
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(fields, ensure_ascii=False) + "\n")

    try:
        generate(request, output, emit, stop.exists)
        emit(phase="generated", done=request["count"])
        return 0
    except InterruptedError:
        emit(phase="cancelled")
        return 130
    except Exception as exc:
        emit(phase="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
