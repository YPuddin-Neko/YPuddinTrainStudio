import copy

import pytest
import torch

from ypuddin.adapters import inject
from ypuddin.config import AdapterConfig
from ypuddin.memory import BlockSwapper
from ypuddin.memory.block_swap import _make_host_masters
from ypuddin.models import get_family

DEVICES = (
    ["cpu"]
    + (["mps"] if torch.backends.mps.is_available() else [])
    + (["cuda"] if torch.cuda.is_available() else [])
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
def test_host_master_is_an_independent_copy_with_original_dtype_shape_and_values(dtype):
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().to(dtype)
    expected = source.float().clone()
    master = _make_host_masters([source], pin_memory=False)[0]
    assert master.device.type == "cpu" and master.dtype == dtype
    assert master.shape == source.shape and master.is_contiguous()
    assert master.data_ptr() != source.data_ptr()
    assert not master.requires_grad
    source.zero_()
    torch.testing.assert_close(master.float(), expected, rtol=0, atol=0)


def test_pinned_master_allocates_final_buffer_directly_without_intermediate_copy(monkeypatch):
    """Exercise allocation selection on CPU; real CUDA pinning is covered below."""
    source = torch.arange(6, dtype=torch.float32, requires_grad=True)
    empty = torch.empty
    allocations = []

    def allocate(size, **kwargs):
        allocations.append({"size": size, **kwargs})
        # CPU-only test environments have no pinned allocator. Only substitute
        # the allocation; copying and source independence use real tensors.
        return empty(size, **(kwargs | {"pin_memory": False}))

    def forbidden(*args, **kwargs):
        pytest.fail("A separate CPU copy or pin_memory conversion would recreate the transient allocation")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", allocate)
        patch.setattr(torch.Tensor, "to", forbidden)
        patch.setattr(torch.Tensor, "pin_memory", forbidden)
        master = _make_host_masters([source], pin_memory=True)[0]
    assert allocations == [{"size": 6, "dtype": torch.float32, "device": "cpu", "pin_memory": True}]
    assert master.data_ptr() != source.data_ptr()
    assert not master.requires_grad and master.grad_fn is None
    torch.testing.assert_close(master, source.detach(), rtol=0, atol=0)


def test_pools_group_mixed_dtypes_without_overlaps_or_source_aliases():
    shared_source = torch.arange(30, dtype=torch.float32)
    sources = [
        shared_source[::2],  # non-dense source; no copying its gaps into the pool
        torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
        torch.tensor(0.125),
        torch.arange(6).to(torch.float8_e4m3fn).reshape(2, 3),
        torch.arange(8).to(torch.float8_e4m3fn)[::2],
        torch.arange(3, dtype=torch.bfloat16),
        torch.tensor([True, False]),
        torch.empty(0, 2),
    ]
    expected = [source.float().clone() for source in sources]
    masters = _make_host_masters(sources, pin_memory=False)
    pools = {}
    for source, host, value in zip(sources, masters, expected, strict=True):
        assert source.shape == host.shape and source.dtype == host.dtype
        assert not host.requires_grad and host.grad_fn is None and host.is_contiguous()
        torch.testing.assert_close(host.float(), value, rtol=0, atol=0)
        storage = host.untyped_storage()
        previous = pools.setdefault(host.dtype, storage.data_ptr())
        assert previous == storage.data_ptr()
        assert storage.nbytes() == sum(s.numel() * s.element_size() for s in sources if s.dtype == host.dtype)
        if source.numel():
            assert source.data_ptr() != host.data_ptr()
    assert len(set(pools.values())) == 4
    for index, host in enumerate(masters):
        if host.numel():
            for other in masters[index + 1 :]:
                if host.dtype == other.dtype and other.numel():
                    assert host.storage_offset() + host.numel() <= other.storage_offset()
    shared_source.zero_()
    torch.testing.assert_close(masters[0], expected[0], rtol=0, atol=0)
    masters[0].zero_()
    for host, value in zip(masters[1:], expected[1:], strict=True):
        torch.testing.assert_close(host.float(), value, rtol=0, atol=0)
    assert _make_host_masters([], pin_memory=False) == []


def test_one_final_allocation_per_dtype_and_no_intermediate_tensor_copies(monkeypatch):
    sources = [torch.ones(9), torch.ones(18), torch.ones(3, dtype=torch.bfloat16)]
    original = torch.empty
    allocations = []

    def allocate(size, **kwargs):
        allocations.append((size, kwargs["dtype"], kwargs["pin_memory"]))
        return original(size, **(kwargs | {"pin_memory": False}))

    def forbidden(*args, **kwargs):
        pytest.fail("Packing must copy directly into its final dtype pools")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", allocate)
        patch.setattr(torch.Tensor, "clone", forbidden)
        patch.setattr(torch.Tensor, "to", forbidden)
        patch.setattr(torch.Tensor, "pin_memory", forbidden)
        masters = _make_host_masters(sources, pin_memory=True)
    assert allocations == [(27, torch.float32, True), (3, torch.bfloat16, True)]
    for host, source in zip(masters, sources, strict=True):
        torch.testing.assert_close(host, source, rtol=0, atol=0)


def test_block_pools_preserve_tensor_identity_and_are_independent_between_blocks(monkeypatch):
    def forbidden_flush():
        pytest.fail("CPU source tensors must not flush CUDA allocations")

    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden_flush)
    blocks = [torch.nn.Linear(4, 3).requires_grad_(False) for _ in range(2)]
    identities = [[id(t) for t in block.parameters()] for block in blocks]
    original = [[t.detach().clone() for t in block.parameters()] for block in blocks]
    swapper = BlockSwapper(blocks, num_swap=2, device="cpu")
    pools = [swapper._masters[i][0].untyped_storage().data_ptr() for i in range(2)]
    assert pools[0] != pools[1]
    for index, block in enumerate(blocks):
        assert [id(t) for t in block.parameters()] == identities[index]
        weight, bias = swapper._masters[index]
        assert weight.untyped_storage().data_ptr() == bias.untyped_storage().data_ptr()
        swapper.ensure(index)
    swapper.release_all()
    swapper.remove()
    assert not swapper._masters and not swapper._handles
    for index, block in enumerate(blocks):
        assert [id(t) for t in block.parameters()] == identities[index]
        for tensor, value in zip(block.parameters(), original[index], strict=True):
            torch.testing.assert_close(tensor, value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for actual pinned memory")
@pytest.mark.parametrize("source_device", ["cpu", "cuda"])
def test_cuda_pinned_master_fetch_release_and_remove_preserve_frozen_values(source_device, monkeypatch):
    block = torch.nn.Linear(5, 3, device=source_device, dtype=torch.bfloat16).requires_grad_(False)
    block.register_buffer("scale", torch.tensor(0.125, device=source_device))
    block.register_buffer(
        "quantized", torch.arange(8, device=source_device, dtype=torch.float32).to(torch.float8_e4m3fn)
    )
    block.register_parameter("adapter", torch.nn.Parameter(torch.ones(3, device=source_device)))
    adapter = block.adapter
    frozen = [p for p in block.parameters() if not p.requires_grad] + list(block.buffers())
    expected = [tensor.detach().float().cpu().clone() for tensor in frozen]
    original_flush, flushes = torch.cuda.empty_cache, []

    def flush():
        # Every source has already been rebound after a completed host copy.
        assert all(tensor.device.type == "cpu" for tensor in frozen)
        flushes.append(True)
        original_flush()

    monkeypatch.setattr(torch.cuda, "empty_cache", flush)
    swapper = BlockSwapper([block], num_swap=1, device="cuda", prefetch=False)
    assert len(flushes) == int(source_device == "cuda")
    try:
        masters = swapper._masters[0]
        assert len(masters) == len(expected)
        assert all(host.device.type == "cpu" and host.is_pinned() for host in masters)
        pools = {}
        for host in masters:
            pointer = host.untyped_storage().data_ptr()
            assert pointer == pools.setdefault(host.dtype, pointer)
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
