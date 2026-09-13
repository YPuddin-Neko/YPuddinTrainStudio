"""End-to-end: toy family + synthetic images -> cache -> train -> save -> resume bit-exact."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

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


@pytest.mark.parametrize("separate_samples", [False, True])
def test_full_run_produces_artifacts_and_events(image_dataset, tmp_path, separate_samples):
    out = tmp_path / "run"
    cfg = _cfg(image_dataset, out)
    samples_dir = tmp_path / "samples" / "job_fixture" if separate_samples else out / "samples"
    if separate_samples:
        cfg.sampling.output_dir = str(samples_dir)
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
    prog = [e for e in events if e["type"] == "sample.progress"]
    assert prog and prog[-1]["done"] == prog[-1]["total"] == 3 and prog[-1]["prompts"] == 1
    steps = [e for e in events if e["type"] == "step"]
    assert steps and steps[-1]["step"] == trainer.progress.total_steps
    assert all("loss" in s and "lr" in s and "grad_norm" in s for s in steps)
    # artifacts
    finals = list(out.glob("toy-final.safetensors"))
    assert finals, list(out.iterdir())
    assert samples_dir.exists() and list(samples_dir.glob("*.png"))
    assert all(Path(e["path"]).parent == samples_dir for e in events if e["type"] == "sample.saved")
    if separate_samples:
        assert not (out / "samples").exists()
    assert (out / "config.toml").exists()
    tensors = load_file(str(finals[0]))
    assert any(k.endswith(".lokr_w1") for k in tensors) and any(k.endswith(".alpha") for k in tensors)
    # loss must actually decrease on this trivial task
    val = [e for e in events if e["type"] == "validation"]
    assert len(val) == 3
    assert val[-1]["mean"] < val[0]["mean"]


@pytest.mark.parametrize("cached", [True, False])
def test_native_mixed_sizes_exact_resume_and_plan(image_dataset, tmp_path, cached):
    from ypuddin.train.plan import plan

    dataset = {
        "resolution_mode": "native",
        "native_max_pixels": 8192,
        "batch_size": 5,
        "masked_loss": True,
        "cache_latents": cached,
        "text_encoding": "cached" if cached else "online",
    }
    cfg = _cfg(
        image_dataset,
        tmp_path / "ref",
        dataset=dataset,
        loop={"epochs": 4, "grad_accum": 2},
        checkpoint={"save_state_every_steps": 2, "save_every_epochs": None},
        validation={"enabled": False},
        sampling={"enabled": False},
    )
    reference = Trainer(cfg, device="cpu")
    assert reference.run() == "finished"
    estimate = plan(cfg, device="cpu")
    assert estimate["steps_per_epoch"] == reference.progress.steps_per_epoch == 2
    assert reference.progress.samples_seen == 48
    assert estimate["native"]["sizes"] == 4
    resumed_cfg = cfg.model_copy(deep=True)
    resumed_cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed_cfg.checkpoint.resume = str(tmp_path / "ref" / "state-2")
    resumed_cfg.dataset.cache_dir = str(tmp_path / "ref" / "cache")
    resumed = Trainer(resumed_cfg, device="cpu")
    assert resumed.run() == "finished"
    assert resumed.progress.samples_seen == reference.progress.samples_seen
    actual, _ = resumed.adapters.export_state()
    expected, _ = reference.adapters.export_state()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "adapter",
    [
        {},
        {"init": "scalar"},
        {"module_dropout": 0.2},
        {"dropout": 0.2, "rank_dropout": 0.2},
        {"init": "scalar", "rs_lora": True, "module_dropout": 0.1},
    ],
)
def test_pause_and_resume_is_bit_exact(image_dataset, tmp_path, adapter):
    # Reference: uninterrupted run of N steps, saving a full state at step 4.
    ref_dir = tmp_path / "ref"
    cfg = _cfg(
        image_dataset,
        ref_dir,
        adapter=adapter,
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
        adapter=adapter,
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


@pytest.mark.parametrize("change", ["contents", "filename", "missing_identity"])
def test_resume_checks_actual_model_assets(image_dataset, tmp_path, change):
    from ypuddin.models.toy import ToyDiT

    weights = tmp_path / "base.safetensors"
    torch.manual_seed(123)
    base = ToyDiT().state_dict()
    save_file(base, weights)
    cfg = _cfg(
        image_dataset,
        tmp_path / "ref",
        model={"dit_path": str(weights)},
        loop={"epochs": 2},
        checkpoint={"save_state_every_steps": 1},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    ref = Trainer(cfg, device="cpu")
    assert ref.run() == "finished"
    checkpoint = tmp_path / "ref" / "state-1"
    meta = json.loads((checkpoint / "state.json").read_text())
    assert meta["format"] == 2 and meta["model_identity"] == ref.model_identity

    if change == "contents":
        base["final_layer.weight"] = base["final_layer.weight"] + 0.25
        save_file(base, weights)  # The path and file size stay the same.
    elif change == "filename":
        renamed = tmp_path / "renamed.safetensors"
        weights.rename(renamed)
        cfg.model.dit_path = str(renamed)
    else:
        del meta["model_identity"]
        (checkpoint / "state.json").write_text(json.dumps(meta))

    cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    cfg.checkpoint.resume = str(checkpoint)
    cfg.dataset.cache_dir = str(tmp_path / "ref" / "cache")
    resumed = Trainer(cfg, device="cpu")
    if change == "contents":
        with pytest.raises(ValueError, match="checkpoint model assets"):
            resumed.run()
    else:
        assert resumed.run() == "finished"
        for key, expected in ref.adapters.training_state_dict().items():
            torch.testing.assert_close(resumed.adapters.training_state_dict()[key], expected, rtol=0, atol=0)
        if change == "missing_identity":
            assert any(
                event["type"] == "warning" and "no model asset identity" in event["message"]
                for event in _events(tmp_path / "resumed" / "events.jsonl")
            )


def test_weight_rotation_keeps_normal_and_ema_as_one_step(image_dataset, tmp_path):
    cfg = _cfg(
        image_dataset,
        tmp_path / "run",
        loop={"epochs": 1, "ema": True},
        checkpoint={"save_every_steps": 1, "keep_last_n": 1, "save_every_epochs": None},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    step = trainer.progress.step
    expected = {f"toy-step{step:06d}.safetensors", f"toy-step{step:06d}-ema.safetensors"}
    assert {path.name for path in trainer.run_dir.glob("toy-step*.safetensors")} == expected
    events = _events(trainer.run_dir / "events.jsonl")
    saved = [event for event in events if event["type"] == "checkpoint.saved" and event["step"] == step]
    assert expected.issubset({Path(event["path"]).name for event in saved})
    assert {event["ema"] for event in saved} == {False, True}
    assert (trainer.run_dir / "toy-final.safetensors").exists()
    assert (trainer.run_dir / "toy-final-ema.safetensors").exists()


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


def test_compile_blocks_wiring_keeps_training_and_export_intact(image_dataset, tmp_path):
    """memory.compile swaps every block for a wrapper after injection; adapters/export must not notice."""
    out = tmp_path / "run"
    cfg = _cfg(
        image_dataset,
        out,
        loop={"epochs": 1, "grad_accum": 1, "mixed_precision": "no"},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    wrapped = []

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner
            wrapped.append(inner)

        def forward(self, *a, **kw):
            return self._orig_mod(*a, **kw)

    n = trainer.compile_blocks(compile_fn=Wrapper)
    blocks = trainer.family.memory_layout(trainer.loaded).blocks
    assert n == len(wrapped) == len(blocks) > 0
    assert all(isinstance(b, Wrapper) for b in blocks)
    before = {k: v.clone() for k, v in trainer.adapters.export_state()[0].items()}
    assert trainer.run() == "finished"
    after = trainer.adapters.export_state()[0]
    assert set(after) == set(before) and any(not torch.equal(after[k], before[k]) for k in after)
    assert not any("_orig_mod" in k for k in after)


def test_compile_is_ignored_off_cuda_and_rejected_with_block_swap(image_dataset, tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(
        image_dataset,
        out,
        memory={"compile": True},
        loop={"epochs": 1, "grad_accum": 1, "mixed_precision": "no"},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    assert any(e["type"] == "warning" and "compile" in e["message"] for e in _events(out / "events.jsonl"))
    bad = _cfg(
        image_dataset,
        tmp_path / "bad",
        memory={"compile": True, "blocks_to_swap": 1},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    import pytest

    with pytest.raises(ValueError, match="compile"):
        Trainer(bad, device="cpu").prepare()


@pytest.mark.parametrize("sampler", ["euler", "heun", "er_sde"])
def test_initial_preview_preserves_training_rng_and_unloads_vae(
    image_dataset, tmp_path, monkeypatch, sampler
):
    outs = []
    for at_start in (False, True):
        cfg = _cfg(
            image_dataset,
            tmp_path / str(at_start),
            adapter={"module_dropout": 0.2},
            loop={"epochs": 1},
            validation={"enabled": False},
            sampling={
                "enabled": True,
                "at_start": at_start,
                "every_epochs": None,
                "every_steps": None,
                "sampler": sampler,
            },
        )
        trainer = Trainer(cfg, device="cpu")
        trainer.prepare()
        unloads = []
        monkeypatch.setattr(trainer.loaded.latent, "unload", lambda calls=unloads: calls.append(True))
        assert trainer.run() == "finished"
        samples = [e for e in _events(trainer.run_dir / "events.jsonl") if e["type"] == "sample.saved"]
        assert len(samples) == int(at_start)
        if at_start:
            assert samples[0]["step"] == 0 and Path(samples[0]["path"]).name.startswith("initial_")
            assert unloads
        outs.append(trainer.adapters.training_state_dict())
    for key in outs[0]:
        torch.testing.assert_close(outs[0][key], outs[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("sampler", ["euler", "er_sde"])
def test_sampling_exception_restores_training_mode_and_unloads_vae(
    image_dataset, tmp_path, monkeypatch, sampler
):
    trainer = Trainer(
        _cfg(image_dataset, tmp_path / "run", memory={"blocks_to_swap": 2}, sampling={"sampler": sampler}),
        device="cpu",
    )
    trainer.prepare()
    trainer.loaded.backbone.train()
    unloaded = []
    monkeypatch.setattr(trainer.loaded.latent, "unload", lambda: unloaded.append(True))

    def fail(_latents):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(trainer.loaded.latent, "decode", fail)
    with pytest.raises(RuntimeError, match="decode failed"):
        trainer.sample_images("broken")
    assert unloaded and trainer.loaded.backbone.training and not trainer.swapper.forward_only


@pytest.mark.parametrize(
    "sampler,scheduler", [("euler", "uniform"), ("heun", "simple"), ("er_sde", "sgm_uniform")]
)
def test_preview_dispatch_uses_resolved_parameters_and_records_them(
    image_dataset, tmp_path, monkeypatch, sampler, scheduler
):
    cfg = _cfg(
        image_dataset,
        tmp_path / "run",
        sampling={
            "sampler": sampler,
            "scheduler": scheduler,
            "steps": 8,
            "cfg": 7,
            "er_sde_order": 2,
            "er_sde_s_noise": 0.25,
            "shift": 2.5,
            "prompts": [{"prompt": "preview", "steps": 2, "cfg": 0, "seed": 19}],
        },
    )
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    calls = []
    dispatch = trainer.family.sample_latents

    def record(loaded, predict, shape, **kwargs):
        assert loaded is trainer.loaded
        calls.append(kwargs)
        return dispatch(loaded, predict, shape, **kwargs)

    monkeypatch.setattr(trainer.family, "sample_latents", record)
    paths = trainer.sample_images("selected")
    assert len(paths) == len(calls) == 1 and paths[0].is_file()
    expected = {
        "sampler": sampler,
        "scheduler": scheduler,
        "steps": 2,
        "cfg": 0.0,
        "shift": 2.5,
        "er_sde_order": 2,
        "er_sde_s_noise": 0.25,
    }
    assert {key: calls[0][key] for key in expected} == expected
    assert calls[0]["generator"].device.type == "cpu" and calls[0]["generator"].initial_seed() == 19
    saved = next(e for e in _events(trainer.run_dir / "events.jsonl") if e["type"] == "sample.saved")
    assert {key: saved[key] for key in expected} == expected
    assert saved["seed"] == 19 and saved["loss"] is None


@pytest.mark.parametrize("sampler", ["euler", "er_sde"])
def test_periodic_preview_and_weights_are_identical_after_resume(image_dataset, tmp_path, sampler):
    cfg = _cfg(
        image_dataset,
        tmp_path / "reference",
        adapter={"module_dropout": 0.2},
        loop={"epochs": None, "max_steps": 4},
        validation={"enabled": False},
        sampling={"sampler": sampler, "at_start": True, "every_steps": 2, "every_epochs": None},
        checkpoint={"save_state_every_steps": 2, "save_every_epochs": None},
    )
    reference = Trainer(cfg, device="cpu")
    assert reference.run() == "finished"
    resumed_cfg = cfg.model_copy(deep=True)
    resumed_cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed_cfg.checkpoint.resume = str(reference.run_dir / "state-2")
    resumed = Trainer(resumed_cfg, device="cpu")
    assert resumed.run() == "finished"
    for key, expected in reference.adapters.training_state_dict().items():
        torch.testing.assert_close(resumed.adapters.training_state_dict()[key], expected, rtol=0, atol=0)
    original_samples = [e for e in _events(reference.run_dir / "events.jsonl") if e["type"] == "sample.saved"]
    resumed_samples = [e for e in _events(resumed.run_dir / "events.jsonl") if e["type"] == "sample.saved"]
    assert [e["step"] for e in original_samples] == [0, 2, 4]
    assert [e["step"] for e in resumed_samples] == [
        4
    ]  # Never regenerate an already completed initial preview.
    assert Path(original_samples[-1]["path"]).read_bytes() == Path(resumed_samples[0]["path"]).read_bytes()


def test_default_preview_matches_legacy_euler_pixels(image_dataset, tmp_path, monkeypatch):
    from ypuddin.sampling.euler import euler_sample

    trainer = Trainer(_cfg(image_dataset, tmp_path / "run"), device="cpu")
    trainer.prepare()
    actual = trainer.sample_images("current")[0].read_bytes()

    def legacy(loaded, predict, shape, **kwargs):
        assert kwargs.pop("sampler") == "euler"
        assert kwargs.pop("scheduler") == "uniform"
        kwargs.pop("er_sde_order")
        kwargs.pop("er_sde_s_noise")
        return euler_sample(predict, shape, **kwargs)

    monkeypatch.setattr(trainer.family, "sample_latents", legacy)
    assert trainer.sample_images("legacy")[0].read_bytes() == actual


@pytest.mark.parametrize("cache_only", [False, True])
def test_preparation_pause_preserves_cache_without_fake_state(image_dataset, tmp_path, cache_only):
    from ypuddin.train import cache

    out = tmp_path / "run"
    cfg = _cfg(
        image_dataset,
        out,
        dataset={"batch_size": 1},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    events = []

    def on_event(e):
        events.append(e)
        if e["type"] == "cache.progress" and e["kind"] == "latents" and e["done"] == 1:
            (out / "control").mkdir(exist_ok=True)
            (out / "control" / "pause").touch()

    em = Emitter(listeners=[on_event])
    outcome = (
        cache(cfg, device="cpu", emitter=em) if cache_only else Trainer(cfg, device="cpu", emitter=em).run()
    )
    assert outcome == "paused"
    assert events[-1]["type"] == "run.paused" and events[-1]["preparing"]
    assert list((out / "cache").rglob("*.safetensors")) and not (out / "state-paused").exists()


def test_training_failure_emits_terminal_event(image_dataset, tmp_path, monkeypatch):
    events = []
    trainer = Trainer(
        _cfg(image_dataset, tmp_path / "run"), device="cpu", emitter=Emitter(listeners=[events.append])
    )

    def fail():
        raise RuntimeError("model rejected")

    monkeypatch.setattr(trainer, "prepare", fail)
    with pytest.raises(RuntimeError, match="model rejected"):
        trainer.run()
    assert [e["type"] for e in events] == ["run.failed"]


def test_legacy_scalar_checkpoint_rejected_and_weights_still_warm_start(image_dataset, tmp_path):
    cfg = _cfg(
        image_dataset,
        tmp_path / "ref",
        adapter={"init": "scalar"},
        loop={"epochs": 1},
        checkpoint={"save_state_every_steps": 1},
        sampling={"enabled": False},
        validation={"enabled": False},
    )
    ref = Trainer(cfg, device="cpu")
    ref.run()
    path = tmp_path / "ref" / "state-1"
    meta = json.loads((path / "state.json").read_text())
    assert meta["format"] == 2 and (path / "training.safetensors").exists()
    meta["format"] = 1
    (path / "state.json").write_text(json.dumps(meta))
    cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    cfg.checkpoint.resume = str(path)
    with pytest.raises(ValueError, match="legacy checkpoints did not save scalar"):
        Trainer(cfg, device="cpu").run()
    cfg.checkpoint.resume = None
    cfg.adapter.resume_weights = str(path / "adapter.safetensors")
    warm = Trainer(cfg, device="cpu")
    warm.prepare()
    assert all(layer.adapter.scalar.item() == 1 for layer in warm.adapters.layers.values())


def test_device_autoselection_prefers_cuda_then_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert Trainer._pick_device().type == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert Trainer._pick_device().type == "cuda"


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
        ),
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
        ),
    ],
)
@pytest.mark.parametrize("swap", [0, 2])
def test_accelerator_train_resume_and_preview(image_dataset, tmp_path, device, swap):
    cfg = _cfg(
        image_dataset,
        tmp_path / "ref",
        adapter={"module_dropout": 0.2, "rank_dropout": 0.1},
        loop={"epochs": 2},
        memory={"blocks_to_swap": swap},
        checkpoint={"save_state_every_steps": 1},
        validation={"enabled": False},
        sampling={"enabled": False},
    )
    ref = Trainer(cfg, device=device)
    ref.run()
    cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    cfg.checkpoint.resume = str(tmp_path / "ref" / "state-1")
    res = Trainer(cfg, device=device)
    res.run()
    for key, expected in ref.adapters.training_state_dict().items():
        torch.testing.assert_close(res.adapters.training_state_dict()[key], expected, rtol=0, atol=0)
    assert res.sample_images("accelerator")


def test_kahan_bf16_resume_keeps_full_precision_moments(image_dataset, tmp_path):
    cfg = _cfg(
        image_dataset,
        tmp_path / "ref",
        adapter={"param_dtype": "bf16"},
        optimizer={"kahan": True},
        loop={"epochs": 2},
        checkpoint={"save_state_every_steps": 1},
        validation={"enabled": False},
        sampling={"enabled": False},
    )
    ref = Trainer(cfg, device="cpu")
    ref.run()
    cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    cfg.checkpoint.resume = str(tmp_path / "ref" / "state-1")
    res = Trainer(cfg, device="cpu")
    res.run()
    for key, expected in ref.adapters.training_state_dict().items():
        torch.testing.assert_close(res.adapters.training_state_dict()[key], expected, rtol=0, atol=0)
    assert all(state["exp_avg"].dtype == torch.float32 for state in res.optimizer.state.values())


def test_optional_log_sinks_receive_metrics_images_and_close(image_dataset, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    seen = {"scalars": [], "images": [], "wandb": [], "closed": [], "init": []}

    class Writer:
        def __init__(self, **kwargs):
            seen["init"].append(kwargs)

        def add_scalar(self, *args, **kwargs):
            seen["scalars"].append(args)

        def add_image(self, *args, **kwargs):
            seen["images"].append(args[0])

        def close(self):
            seen["closed"].append("tensorboard")

    run = SimpleNamespace(
        log=lambda values, **kwargs: seen["wandb"].append(values),
        finish=lambda **kwargs: seen["closed"].append("wandb"),
    )
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=Writer))
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: run, Image=lambda *args, **kwargs: "image"),
    )
    cfg = _cfg(
        image_dataset,
        tmp_path / "run",
        loop={"epochs": 1},
        logging={"tensorboard": True, "wandb": {"project": "test"}},
        sampling={"enabled": True, "at_start": True, "every_epochs": None},
    )
    assert Trainer(cfg, device="cpu").run() == "finished"
    assert {x[0] for x in seen["scalars"]} >= {"train/loss", "validation/mean", "lr/w1"}
    assert seen["images"] and any("samples/0" in x for x in seen["wandb"])
    assert sorted(seen["closed"]) == ["tensorboard", "wandb"]


def test_schedule_free_switches_for_evaluation_export_and_returns_to_train(
    image_dataset, tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    calls = []

    class Fake(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.training = False

        def train(self):
            self.training = True
            calls.append("train")

        def eval(self):
            self.training = False
            calls.append("eval")

        def step(self, closure=None):
            assert self.training
            calls.append("step")
            return super().step(closure)

    monkeypatch.setitem(sys.modules, "audit_schedulefree", SimpleNamespace(Fake=Fake))
    cfg = _cfg(
        image_dataset,
        tmp_path / "run",
        optimizer={"type": "audit_schedulefree.Fake"},
        loop={"epochs": 1},
        sampling={"enabled": True, "at_start": True, "every_epochs": None},
    )
    trainer = Trainer(cfg, device="cpu")
    assert trainer.run() == "finished"
    assert calls[0] == "train" and "step" in calls and calls.count("eval") >= 3
    assert trainer.optimizer.training and trainer.scheduler is None


def test_real_schedule_free_resume_and_tensorboard_event_files(image_dataset, tmp_path):
    pytest.importorskip("schedulefree")
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cfg = _cfg(
        image_dataset,
        tmp_path / "ref",
        optimizer={"type": "adamw_sf"},
        loop={"epochs": 2},
        checkpoint={"save_state_every_steps": 1},
        sampling={"enabled": True, "at_start": True, "every_epochs": 1},
        logging={"tensorboard": True},
    )
    ref = Trainer(cfg, device="cpu")
    ref.run()
    cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    cfg.checkpoint.resume = str(tmp_path / "ref" / "state-1")
    res = Trainer(cfg, device="cpu")
    res.run()
    for key, expected in ref.adapters.training_state_dict().items():
        torch.testing.assert_close(res.adapters.training_state_dict()[key], expected, rtol=0, atol=0)
    events = EventAccumulator(str(tmp_path / "ref" / "tensorboard"))
    events.Reload()
    assert {"train/loss", "validation/mean"} <= set(events.Tags()["scalars"])
    assert "samples/0" in events.Tags()["images"]
    assert all(g["train_mode"] for g in res.optimizer.param_groups)
