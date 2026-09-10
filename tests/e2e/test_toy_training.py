"""End-to-end: toy family + synthetic images -> cache -> train -> save -> resume bit-exact."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from ypuddin.config import TrainConfig
from ypuddin.train import Trainer
from ypuddin.train.events import Emitter


def _cfg(image_dataset: Path, out: Path, **overrides) -> TrainConfig:
    base = {
        "model": {"family": "toy", "dtype": "fp32"},
        "dataset": {
            "sources": [{"path": str(image_dataset), "repeats": 1}],
            "resolutions": [64],
            "bucket_step": 16,
            "batch_size": 2,
            "num_workers": 0,
            "caption": {"shuffle": True, "tag_dropout": 0.1, "caption_dropout": 0.1},
        },
        "adapter": {"algo": "lokr", "rank": 4, "alpha": 4, "factor": -1, "preset": "attn-mlp"},
        "optimizer": {"type": "adamw", "lr": 1e-3, "grad_clip_norm": 1.0},
        "scheduler": {"type": "cosine", "warmup_steps": 2},
        "loop": {"epochs": 3, "grad_accum": 2, "mixed_precision": "no", "seed": 7},
        "checkpoint": {"output_dir": str(out), "name": "toy", "save_every_epochs": 1},
        "validation": {"enabled": True, "split_ratio": 0.25, "every_epochs": 1, "timesteps": [0.2, 0.5, 0.8]},
        "sampling": {
            "enabled": True,
            "every_epochs": 3,
            "prompts": [{"prompt": "1girl, smile", "width": 64, "height": 64, "steps": 3}],
            "width": 64,
            "height": 64,
        },
    }
    for k, v in overrides.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return TrainConfig.model_validate(base)


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_full_run_produces_artifacts_and_events(image_dataset, tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(image_dataset, out)
    trainer = Trainer(cfg, device="cpu")
    outcome = trainer.run()
    assert outcome == "finished"
    events = _events(out / "events.jsonl")
    types = [e["type"] for e in events]
    assert types[0] == "run.started" and types[-1] == "run.finished"
    assert (
        "data.plan" in types
        and "adapters.injected" in types
        and "validation" in types
        and "sample.saved" in types
    )
    steps = [e for e in events if e["type"] == "step"]
    assert steps and steps[-1]["step"] == trainer.progress.total_steps
    assert all("loss" in s and "lr" in s and "grad_norm" in s for s in steps)
    # artifacts
    finals = list(out.glob("toy-final.safetensors"))
    assert finals, list(out.iterdir())
    assert (out / "samples").exists() and list((out / "samples").glob("*.png"))
    assert (out / "config.toml").exists()
    tensors = load_file(str(finals[0]))
    assert any(k.endswith(".lokr_w1") for k in tensors) and any(k.endswith(".alpha") for k in tensors)
    # loss must actually decrease on this trivial task
    val = [e for e in events if e["type"] == "validation"]
    assert len(val) == 3
    assert val[-1]["mean"] < val[0]["mean"]


def test_pause_and_resume_is_bit_exact(image_dataset, tmp_path):
    # Reference: uninterrupted run of N steps, saving a full state at step 4.
    ref_dir = tmp_path / "ref"
    cfg = _cfg(
        image_dataset,
        ref_dir,
        checkpoint={
            "output_dir": str(ref_dir),
            "name": "toy",
            "save_every_epochs": None,
            "save_state_every_steps": 4,
        },
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    ref = Trainer(cfg, device="cpu")
    ref.run()
    ref_tensors, _ = ref.adapters.export_state()
    total = ref.progress.total_steps
    assert total > 6

    # Resume from the step-4 state in a fresh process-like trainer and train to the end.
    res_dir = tmp_path / "res"
    cfg2 = _cfg(
        image_dataset,
        res_dir,
        checkpoint={
            "output_dir": str(res_dir),
            "name": "toy",
            "save_every_epochs": None,
            "resume": str(ref_dir / "state-4"),
        },
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    # cache dir must be shared for identical latents (content-addressed, so reuse is exact)
    cfg2 = cfg2.model_copy(
        update={"dataset": cfg2.dataset.model_copy(update={"cache_dir": str(ref_dir / "cache")})}
    )
    res = Trainer(cfg2, device="cpu")
    res.prepare()
    assert res.progress.step == 4
    res.run()
    assert res.progress.step == total
    res_tensors, _ = res.adapters.export_state()
    for k in ref_tensors:
        torch.testing.assert_close(res_tensors[k], ref_tensors[k], rtol=0, atol=0)
    # step events after resume must match the reference run exactly
    ref_steps = {e["step"]: e["loss"] for e in _events(ref_dir / "events.jsonl") if e["type"] == "step"}
    res_steps = {e["step"]: e["loss"] for e in _events(res_dir / "events.jsonl") if e["type"] == "step"}
    for s in range(5, total + 1):
        assert res_steps[s] == ref_steps[s], s


def test_control_file_pause(image_dataset, tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(
        image_dataset,
        out,
        sampling={"enabled": False},
        validation={"enabled": False},
        loop={"epochs": 5, "grad_accum": 1, "mixed_precision": "no", "seed": 1},
    )
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    hit = {}

    def on_event(e):
        if e["type"] == "step" and e["step"] == 3 and not hit:
            (out / "control").mkdir(exist_ok=True)
            (out / "control" / "pause").touch()
            hit["at"] = 3

    trainer.emitter.add_listener(on_event)
    outcome = trainer.run()
    assert outcome == "paused"
    assert (out / "state-paused" / "state.json").exists()
    assert trainer.progress.step == 3


def test_lora_and_full_and_ema_variants(image_dataset, tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(
        image_dataset,
        out,
        adapter={
            "algo": "lora",
            "rank": 4,
            "alpha": 4,
            "preset": "attn-mlp",
            "rules": [{"match": "blocks.*.mlp.*", "algo": "full"}],
            "dora": True,
        },
        loop={"epochs": 1, "grad_accum": 1, "mixed_precision": "no", "seed": 3, "ema": True},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    tensors = load_file(str(out / "toy-final.safetensors"))
    assert any(k.endswith(".lora_down.weight") for k in tensors)
    assert any(k.endswith(".diff") for k in tensors)
    assert any(k.endswith(".dora_scale") for k in tensors)
    assert (out / "toy-final-ema.safetensors").exists()


def test_online_latents_without_cache(image_dataset, tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(
        image_dataset,
        out,
        dataset={
            "sources": [{"path": str(image_dataset)}],
            "resolutions": [64],
            "bucket_step": 16,
            "batch_size": 2,
            "num_workers": 0,
            "cache_latents": False,
            "flip": True,
        },
        loop={"epochs": 1, "grad_accum": 1, "mixed_precision": "no"},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    trainer = Trainer(cfg, device="cpu", emitter=Emitter())
    assert trainer.run() == "finished"
    assert trainer.progress.step == trainer.progress.total_steps


def test_block_swap_training_matches_reference(image_dataset, tmp_path):
    """Training with blocks_to_swap must produce exactly the same weights as without."""
    outs = []
    for swap in (0, 2):
        out = tmp_path / f"run{swap}"
        cfg = _cfg(
            image_dataset,
            out,
            memory={"blocks_to_swap": swap},
            loop={"epochs": 1, "grad_accum": 1, "mixed_precision": "no", "seed": 5},
            sampling={"enabled": False},
            validation={"enabled": False},
        )
        cfg = cfg.model_copy(
            update={"dataset": cfg.dataset.model_copy(update={"cache_dir": str(tmp_path / "cache")})}
        )
        tr = Trainer(cfg, device="cpu")
        assert tr.run() == "finished"
        outs.append(tr.adapters.export_state()[0])
    for k in outs[0]:
        torch.testing.assert_close(outs[1][k], outs[0][k], rtol=0, atol=0)


def test_cached_text_mode_with_caption_augmentation(image_dataset, tmp_path):
    """Pre-cached text encodings must cover trigger words, shuffled variants, dropout and sample prompts."""
    out = tmp_path / "run"
    cfg = _cfg(
        image_dataset,
        out,
        dataset={
            "sources": [{"path": str(image_dataset)}],
            "resolutions": [64],
            "bucket_step": 16,
            "batch_size": 2,
            "num_workers": 0,
            "text_encoding": "cached",
            "caption": {
                "trigger_word": "ypd",
                "prefix": "masterpiece",
                "shuffle": True,
                "tag_dropout": 0.2,
                "caption_dropout": 0.2,
                "cache_variants": 4,
            },
        },
        loop={"epochs": 2, "grad_accum": 1, "mixed_precision": "no", "seed": 3},
        sampling={
            "enabled": True,
            "every_epochs": 2,
            "prompts": [
                {"prompt": "ypd, 1girl", "negative": "lowres", "width": 64, "height": 64, "steps": 2}
            ],
            "width": 64,
            "height": 64,
        },
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    events = _events(out / "events.jsonl")
    prepared = next(e for e in events if e["type"] == "run.prepared")
    assert prepared["text_mode"] == "cached"
    text_progress = [e for e in events if e["type"] == "cache.progress" and e["kind"] == "text"]
    assert text_progress and text_progress[-1]["total"] >= 3  # "", prompt, negative + caption variants
    assert [e for e in events if e["type"] == "sample.saved"]
    cached_files = list((out / "cache" / "text").rglob("*.safetensors"))
    assert len(cached_files) == text_progress[-1]["total"]
