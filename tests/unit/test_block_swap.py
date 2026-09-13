import copy

import pytest
import torch

from ypuddin.adapters import inject
from ypuddin.config import AdapterConfig
from ypuddin.memory import BlockSwapper
from ypuddin.memory.block_swap import _make_host_master
from ypuddin.models import get_family

DEVICES = (
    ["cpu"]
    + (["mps"] if torch.backends.mps.is_available() else [])
    + (["cuda"] if torch.cuda.is_available() else [])
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
def test_host_master_is_an_independent_copy_with_original_dtype_and_layout(dtype):
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().to(dtype)
    expected = source.float().clone()
    master = _make_host_master(source, pin_memory=False)
    assert master.device.type == "cpu" and master.dtype == dtype
    assert master.shape == source.shape and master.stride() == source.stride()
    assert master.data_ptr() != source.data_ptr()
    assert not master.requires_grad
    source.zero_()
    torch.testing.assert_close(master.float(), expected, rtol=0, atol=0)


def test_pinned_master_allocates_final_buffer_directly_without_intermediate_copy(monkeypatch):
    """Exercise allocation selection on CPU; real CUDA pinning is covered below."""
    source = torch.arange(6, dtype=torch.float32, requires_grad=True)
    empty_like = torch.empty_like
    allocations = []

    def allocate(tensor, **kwargs):
        allocations.append(dict(kwargs))
        # CPU-only test environments have no pinned allocator. Only substitute
        # the allocation; copying and source independence use real tensors.
        return empty_like(tensor, **(kwargs | {"pin_memory": False}))

    def forbidden(*args, **kwargs):
        pytest.fail("A separate CPU copy or pin_memory conversion would recreate the transient allocation")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty_like", allocate)
        patch.setattr(torch.Tensor, "to", forbidden)
        patch.setattr(torch.Tensor, "pin_memory", forbidden)
        master = _make_host_master(source, pin_memory=True)
    assert allocations == [{"device": "cpu", "pin_memory": True}]
    assert master.data_ptr() != source.data_ptr()
    assert not master.requires_grad and master.grad_fn is None
    torch.testing.assert_close(master, source.detach(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for actual pinned memory")
@pytest.mark.parametrize("source_device", ["cpu", "cuda"])
def test_cuda_pinned_master_fetch_release_and_remove_preserve_frozen_values(source_device):
    block = torch.nn.Linear(5, 3, device=source_device, dtype=torch.bfloat16).requires_grad_(False)
    block.register_buffer("scale", torch.tensor(0.125, device=source_device))
    block.register_buffer(
        "quantized", torch.arange(8, device=source_device, dtype=torch.float32).to(torch.float8_e4m3fn)
    )
    block.register_parameter("adapter", torch.nn.Parameter(torch.ones(3, device=source_device)))
    adapter = block.adapter
    frozen = [p for p in block.parameters() if not p.requires_grad] + list(block.buffers())
    expected = [tensor.detach().float().cpu().clone() for tensor in frozen]
    swapper = BlockSwapper([block], num_swap=1, device="cuda", prefetch=False)
    try:
        masters = swapper._masters[0]
        assert len(masters) == len(expected)
        assert all(host.device.type == "cpu" and host.is_pinned() for host in masters)
        assert all(tensor.data_ptr() == host.data_ptr() for tensor, host in zip(frozen, masters, strict=True))
        assert all(tensor is not adapter for tensor in swapper._tensors[0])
        swapper.move_model_to_device(block)
        assert block.adapter is adapter and adapter.device.type == "cuda"
        swapper.ensure(0)
        torch.cuda.synchronize()
        for tensor, value in zip(frozen, expected, strict=True):
            assert tensor.device.type == "cuda"
            torch.testing.assert_close(tensor.float().cpu(), value, rtol=0, atol=0)
        swapper.release_all()
        for tensor, host, value in zip(frozen, masters, expected, strict=True):
            assert tensor.data_ptr() == host.data_ptr()
            torch.testing.assert_close(tensor.float(), value, rtol=0, atol=0)
    finally:
        swapper.remove()
    for tensor, value in zip(frozen, expected, strict=True):
        assert tensor.device.type == "cuda"
        torch.testing.assert_close(tensor.float().cpu(), value, rtol=0, atol=0)


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
    for a, b in zip(g, g_ref, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    # after backward the blocks are released back to the host
    assert model.blocks[1].self_attn.q_proj.base.weight.device.type == "cpu"

    # inference mode
    swapper.set_forward_only(True)
    with torch.no_grad():
        torch.testing.assert_close(model(x, t, ctx, mask), ref(x, t, ctx, mask), rtol=1e-5, atol=1e-5)
    swapper.remove()
    assert model.blocks[1].self_attn.q_proj.base.weight.device.type == torch.device(device).type
