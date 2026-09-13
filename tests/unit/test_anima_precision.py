"""Real Anima mixed-dtype boundaries, using tiny CPU weights rather than models/GPU."""

import pytest
import torch
from safetensors.torch import save_file

from ypuddin.adapters import inject
from ypuddin.config import AdapterConfig
from ypuddin.memory.block_swap import BlockSwapper
from ypuddin.models import TextCond, get_family
from ypuddin.models.anima.family import load_dit
from ypuddin.models.anima.vendor.cosmos_dit import ANIMA_2B_CONFIG, Anima, _adaln_projection
from ypuddin.models.base import LoadedModel

pytestmark = pytest.mark.filterwarnings("ignore:In CPU autocast, but the target dtype is not supported")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("preset", ["attn-only", "full-linear"])
def test_half_weight_family_keeps_fp32_timesteps_with_checkpoint_and_swap(tmp_path, dtype, preset):
    torch.manual_seed(12)
    config = dict(
        ANIMA_2B_CONFIG,
        max_img_h=32,
        max_img_w=32,
        model_channels=128,
        num_blocks=2,
        num_heads=2,
        crossattn_emb_channels=128,
        use_llm_adapter=True,
        llm_adapter_source_dim=128,
        llm_adapter_dim=128,
        llm_adapter_layers=1,
        llm_adapter_heads=2,
        adaln_lora_dim=16,
    )
    path = tmp_path / "tiny.safetensors"
    save_file(Anima(**config).to(dtype).state_dict(), str(path))
    family = get_family("anima")
    results = []
    for checkpoint in (False, True):
        torch.manual_seed(30)
        model, _ = load_dit(path, device="cpu", dtype=dtype)
        adapters = inject(
            model,
            AdapterConfig(algo="lora", rank=4, alpha=4, preset=preset),
            family.presets()[preset],
            base_precision="bf16" if dtype == torch.bfloat16 else "fp16",
        )
        model.train()
        if checkpoint:
            model.enable_gradient_checkpointing()
        swapper = BlockSwapper(model.blocks, 20 if checkpoint else 0, "cpu")
        swapper.move_model_to_device(model)
        identities = {name: (id(value), value.dtype) for name, value in model.named_parameters()}
        modulation_dtypes = []
        handles = [
            module.register_forward_hook(
                lambda _module, _inputs, output, dtypes=modulation_dtypes: dtypes.append(output.dtype)
            )
            for name, module in model.named_modules()
            if name == "final_layer.adaln_modulation"
            or name.endswith(
                ("adaln_modulation_self_attn", "adaln_modulation_cross_attn", "adaln_modulation_mlp")
            )
        ]
        loaded = LoadedModel(backbone=model, text=None, latent=None, device=torch.device("cpu"), dtype=dtype)
        x = torch.randn(1, 16, 4, 4, dtype=dtype, requires_grad=True)
        cond = TextCond(
            {
                "embeds": torch.randn(1, 3, 128, dtype=dtype),
                "attn_mask": torch.ones(1, 3, dtype=torch.bool),
                "t5_ids": torch.ones(1, 3, dtype=torch.long),
                "t5_mask": torch.ones(1, 3, dtype=torch.bool),
            }
        )
        try:
            with torch.autocast("cpu", dtype=dtype):
                output = family.forward(loaded, x, torch.tensor([0.4], dtype=torch.float32), cond)
                loss = output.float().square().mean()
            loss.backward()
            swapper.release_all()
            grads = {
                name: parameter.grad
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            assert torch.isfinite(output).all()
            assert grads and all(grad is not None and torch.isfinite(grad).all() for grad in grads.values())
            assert any(grad.abs().sum() > 0 for grad in grads.values())
            if preset == "full-linear":
                assert any(grad.abs().sum() > 0 for name, grad in grads.items() if "adaln_modulation" in name)
            assert modulation_dtypes and set(modulation_dtypes) == {
                torch.float32 if dtype == torch.float16 else torch.bfloat16
            }
            assert {name: (id(value), value.dtype) for name, value in model.named_parameters()} == identities
            assert all(parameter.dtype == torch.float32 for parameter in adapters.parameters())
            results.append((output.detach(), {name: grad.clone() for name, grad in grads.items()}))
        finally:
            for handle in handles:
                handle.remove()
            swapper.remove()
    torch.testing.assert_close(results[0][0], results[1][0])
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name])


def test_fp16_adaln_protection_computes_above_half_range_without_changing_storage():
    module = torch.nn.Linear(1, 1, bias=False, dtype=torch.float16)
    with torch.no_grad():
        module.weight.fill_(65504)
    weight = module.weight
    value = torch.tensor([[2.0]], requires_grad=True)
    with torch.autocast("cpu", dtype=torch.float16):
        output = _adaln_projection(module, value, use_fp32=True)
    assert output.dtype == torch.float32 and output.item() == 131008
    output.sum().backward()
    assert value.grad.item() == 65504 and module.weight.grad.item() == 2
    assert module.weight is weight and weight.dtype == torch.float16
