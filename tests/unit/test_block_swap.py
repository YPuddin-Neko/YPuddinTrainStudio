import copy

import pytest
import torch

from ypuddin.adapters import TargetPreset, inject
from ypuddin.config import AdapterConfig
from ypuddin.memory import BlockSwapper
from ypuddin.models import get_family

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_block_swap_matches_unswapped_forward_and_grads(device):
    fam = get_family("toy")
    torch.manual_seed(0)
    from ypuddin.config import MemoryConfig, ModelConfig

    loaded = fam.load(ModelConfig(family="toy"), MemoryConfig(), device=device, dtype=torch.float32)
    model = loaded.backbone
    preset = fam.presets()["attn-mlp"]
    aset = inject(model, AdapterConfig(algo="lora", rank=4, alpha=4), preset)
    for layer in aset.layers.values():
        with torch.no_grad():
            for p in layer.adapter.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
    ref = copy.deepcopy(model)

    x = torch.randn(2, 4, 16, 16, device=device)
    t = torch.rand(2, device=device)
    ctx = torch.randn(2, 5, 64, device=device)
    mask = torch.ones(2, 5, dtype=torch.bool, device=device)

    swapper = BlockSwapper(fam.memory_layout(loaded).blocks, num_swap=2, device=device)
    assert swapper.summary()["swapped_blocks"] == 2
    # swapped frozen weights live on the host between calls
    assert model.blocks[1].self_attn.q_proj.base.weight.device.type == "cpu"

    out = model(x, t, ctx, mask)
    out_ref = ref(x, t, ctx, mask)
    torch.testing.assert_close(out, out_ref, rtol=1e-5, atol=1e-5)

    out.square().mean().backward()
    swapper.release_all()
    out_ref.square().mean().backward()
    g = [p.grad for p in model.parameters() if p.requires_grad]
    g_ref = [p.grad for p in ref.parameters() if p.requires_grad]
    assert len(g) == len(g_ref) > 0
    for a, b in zip(g, g_ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    # after backward the blocks are released back to the host
    assert model.blocks[1].self_attn.q_proj.base.weight.device.type == "cpu"

    # inference mode
    swapper.set_forward_only(True)
    with torch.no_grad():
        torch.testing.assert_close(model(x, t, ctx, mask), ref(x, t, ctx, mask), rtol=1e-5, atol=1e-5)
    swapper.remove()
    assert model.blocks[1].self_attn.q_proj.base.weight.device.type == torch.device(device).type
