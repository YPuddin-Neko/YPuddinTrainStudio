"""Real small native encoders: LoRA gradients, portable sampling and exact continuation."""

import json

import pytest
import torch

from tests.e2e.test_anima_pipeline import tiny_models as anima_models  # noqa: F401
from tests.e2e.test_anima_pipeline import tiny_vae_loader as tiny_qwen_vae  # noqa: F401
from tests.e2e.test_krea2_pipeline import tiny_models as krea_models  # noqa: F401
from tests.unit.test_flux2_family import tiny_root as klein_models  # noqa: F401
from tests.unit.test_sdxl_family import tiny_pipeline as sdxl_models  # noqa: F401
from ypuddin.adapters import load_adapter_file
from ypuddin.config import TrainConfig
from ypuddin.server.xyz_worker import bind_checkpoint
from ypuddin.train import Trainer
from ypuddin.train.plan import plan


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def recipe(request, image_dataset, tmp_path, family, backbone):
    names = {"anima": "anima_models", "krea2": "krea_models", "sdxl": "sdxl_models", "flux2": "klein_models"}
    assets = request.getfixturevalue(names[family])
    model = {"family": family, "dtype": "fp32"}
    if family in {"anima", "krea2"}:
        model.update(
            dit_path=str(assets["dit"]),
            text_encoder_path=str(assets["qwen"] if family == "anima" else assets["qwen_dir"]),
            vae_path=str(assets["vae"]),
        )
    else:
        model["dit_path"] = str(assets)
    if family == "sdxl":
        model["sdxl_max_token_length"] = 150
    return TrainConfig.model_validate(
        {
            "model": model,
            "training": {"mode": "adapter", "train_backbone": backbone, "train_text_encoder": True},
            "adapter": {"algo": "lora", "rank": 2, "alpha": 2},
            "objective": {"timestep_sampling": "uniform"},
            "dataset": {
                "sources": [{"path": str(image_dataset)}],
                "resolutions": [64],
                "bucket_step": 16,
                "batch_size": 1,
                "num_workers": 0,
                "text_encoding": "online",
            },
            "memory": {"activation_checkpointing": "block", "offload_text_encoder": False},
            "loop": {"max_steps": 2, "mixed_precision": "no", "seed": 7},
            "optimizer": {"lr": 0.001},
            "checkpoint": {
                "output_dir": str(tmp_path / "reference"),
                "name": "text",
                "save_dtype": "fp32",
                "save_every_epochs": None,
                "save_state_every_steps": 1,
            },
            "sampling": {
                "enabled": True,
                "every_epochs": None,
                "every_steps": 2,
                "prompts": [{"prompt": "a red cat", "width": 64, "height": 64, "steps": 2, "cfg": 1}],
            },
        }
    )


@pytest.mark.usefixtures("tiny_qwen_vae")
@pytest.mark.parametrize("family", ["anima", "krea2", "sdxl", "flux2"])
@pytest.mark.parametrize("backbone", [False, True])
def test_text_adapter_train_export_sample_exact_resume(
    request, image_dataset, tmp_path, family, backbone, monkeypatch
):
    cfg = recipe(request, image_dataset, tmp_path, family, backbone)
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    original = trainer.adapters.training_state_dict()
    frozen = {
        f"{component}.{key}": value.detach().clone()
        for component, module in trainer.adapters.modules.items()
        for key, value in module.state_dict().items()
        if ".adapter." not in key and ".dora." not in key
    }
    backbone_original = (
        {k: v.detach().clone() for k, v in trainer.loaded.backbone.state_dict().items()}
        if not backbone
        else {}
    )
    planned = plan(cfg, device="cpu")
    assert planned["ok"], planned["errors"]
    assert planned["params"]["trainable"] == trainer.adapters.num_params()
    assert planned["memory"]["peak_mb_estimate"] is None  # TE activations remain explicitly unestimated
    assert trainer.run() == "finished"
    final = trainer.adapters.training_state_dict()
    for component in trainer.adapters.components:
        assert any(
            not torch.equal(original[k], final[k]) for k in original if k.startswith(component + ".")
        ), component
    for component, module in trainer.adapters.modules.items():
        for key, value in module.state_dict().items():
            if (name := f"{component}.{key}") in frozen:
                assert torch.equal(value, frozen[name]), name
    for key, value in backbone_original.items():
        assert torch.equal(trainer.loaded.backbone.state_dict()[key], value), key
    reference_png = list((tmp_path / "reference/samples").glob("*.png"))
    assert reference_png
    artifact = tmp_path / "reference/text-final.safetensors"
    tensors, metadata = load_adapter_file(artifact)
    assert any(k.startswith("lora_te1_") for k in tensors)
    assert any(k.startswith("lora_te2_") for k in tensors) == (family == "sdxl")
    assert json.loads(metadata["ypuddin.components"]) == sorted(trainer.adapters.components)
    loaded = trainer.family.load(
        cfg.model, cfg.memory, device="cpu", dtype=torch.float32, backbone_device="cpu"
    )
    trainer.family.materialize_backbone(loaded)
    original_cond = loaded.text.encode(["a red cat"], "cpu")
    bindings = bind_checkpoint(
        loaded.backbone, artifact, family, trainer.family.spec.adapter_prefix, text=loaded.text
    )
    adapted_cond = loaded.text.encode(["a red cat"], "cpu")
    trainer.adapters.train(False)
    expected_cond = trainer.loaded.text.encode(["a red cat"], "cpu")
    for key, value in adapted_cond.tensors.items():
        torch.testing.assert_close(value, expected_cond[key], rtol=0, atol=0)
    assert any(not torch.equal(adapted_cond[k], original_cond[k]) for k in original_cond.tensors)
    for *_, wrapper in bindings:
        wrapper.multiplier = 0
    for key, value in loaded.text.encode(["a red cat"], "cpu").tensors.items():
        torch.testing.assert_close(value, original_cond[key], rtol=0, atol=0)
    if family == "sdxl" and not backbone:
        verify_xyz_text_scale_and_checkpoint(
            cfg, artifact, metadata, original_cond, expected_cond, tmp_path, monkeypatch
        )
        monkeypatch.undo()
    resumed_cfg = cfg.model_copy(deep=True)
    resumed_cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    resumed_cfg.checkpoint.resume = str(tmp_path / "reference/state-1")
    resumed = Trainer(resumed_cfg, device="cpu")
    assert resumed.run() == "finished"
    for key, value in resumed.adapters.training_state_dict().items():
        torch.testing.assert_close(value, final[key], rtol=0, atol=0)
    resumed_png = list((tmp_path / "resumed/samples").glob("*.png"))
    assert [p.read_bytes() for p in resumed_png] == [p.read_bytes() for p in reference_png]


def verify_xyz_text_scale_and_checkpoint(
    cfg, artifact, metadata, original_cond, expected_cond, tmp_path, monkeypatch
):
    from ypuddin.adapters import save_adapter_file
    from ypuddin.models.sdxl.family import SDXLFamily
    from ypuddin.server.xyz import checkpoint_signature
    from ypuddin.server.xyz_worker import generate

    tensors, _ = load_adapter_file(artifact)
    zero = {
        key: (torch.zeros_like(value) if key.endswith("lora_up.weight") else value)
        for key, value in tensors.items()
    }
    zero_path = save_adapter_file(tmp_path / "zero.safetensors", zero, metadata, dtype="fp32")
    current, seen = {"index": -1}, {}

    class ObservedFamily(SDXLFamily):
        def forward(self, loaded, x, t, cond, **kwargs):
            seen.setdefault(current["index"], []).append({k: v.clone() for k, v in cond.tensors.items()})
            return super().forward(loaded, x, t, cond, **kwargs)

    def emit(event, **data):
        if event == "xyz.progress" and "cell_index" in data:
            current["index"] = data["cell_index"]

    monkeypatch.setattr("ypuddin.models.get_family", lambda _: ObservedFamily())
    payload = {
        "device": "cpu",
        "model": cfg.model.model_dump(),
        "memory": cfg.memory.model_dump(),
        "training": cfg.training.model_dump(),
        "fingerprint_cache": str(tmp_path / "xyz-fingerprints"),
        "xyz": {
            "request": {
                "prompt": "a red cat",
                "width": 64,
                "height": 64,
                "steps": 2,
                "cfg": 1,
                "seed": 42,
                "x": {"key": "checkpoint", "values": ["trained", "zero"]},
                "y": {"key": "adapter_scale", "values": [0, 1]},
            },
            "checkpoints": {
                name: {"name": name, "path": str(path), "signature": checkpoint_signature(path)}
                for name, path in (("trained", artifact), ("zero", zero_path))
            },
        },
    }
    output = tmp_path / "xyz"
    generate(payload, output, emit, lambda: False)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and len(manifest["cells"]) == 4
    for cell in manifest["cells"]:
        expected = (
            expected_cond
            if cell["checkpoint_id"] == "trained" and cell["adapter_scale"] == 1
            else original_cond
        )
        assert seen[cell["index"]]
        for cond in seen[cell["index"]]:
            for key, value in cond.items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("backbone", [False, True])
def test_sdxl_two_rank_text_adapters_and_exact_resume(request, image_dataset, tmp_path, backbone):
    from safetensors.torch import load_file

    from tests.checkpoint_assertions import assert_checkpoint_value_exact
    from tests.e2e.test_distributed_training import _launch

    cfg = recipe(request, image_dataset, tmp_path, "sdxl", backbone)
    cfg.memory.activation_checkpointing = "none"
    cfg.dataset.cache_dir = str(tmp_path / "shared-cache")
    cfg.loop.gpu_count = 2
    cfg.loop.grad_accum = 2
    path = tmp_path / "ddp-config.json"
    path.write_text(json.dumps(cfg.model_dump(mode="json")))
    _launch(path)
    reference = tmp_path / "reference"
    replicas = [load_file(reference / f"rank-{rank}.safetensors") for rank in (0, 1)]
    assert any(key.startswith("text_encoder.") for key in replicas[0])
    assert any(key.startswith("text_encoder_2.") for key in replicas[0])
    for key in replicas[0]:
        torch.testing.assert_close(replicas[0][key], replicas[1][key], rtol=0, atol=0)
    cfg.checkpoint.output_dir = str(tmp_path / "resumed")
    cfg.checkpoint.resume = str(reference / "state-1")
    path.write_text(json.dumps(cfg.model_dump(mode="json")))
    _launch(path)
    state_paths = [reference / "state-2", tmp_path / "resumed/state-2"]
    expected, actual = [load_file(state / "training.safetensors") for state in state_paths]
    assert_checkpoint_value_exact(actual, expected, "adapters")
    for component in ("rng", "optimizer", "scheduler"):
        expected, actual = [
            torch.load(state / f"{component}.pt", weights_only=False, map_location="cpu")
            for state in state_paths
        ]
        assert_checkpoint_value_exact(actual, expected, component)
    expected, actual = [json.loads((state / "state.json").read_text()) for state in state_paths]
    for component in ("progress", "sampler"):
        assert_checkpoint_value_exact(actual[component], expected[component], component)
    assert [p.read_bytes() for p in (reference / "samples").glob("*.png")] == [
        p.read_bytes() for p in (tmp_path / "resumed/samples").glob("*.png")
    ]
