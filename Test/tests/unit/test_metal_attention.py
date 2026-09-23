"""CPU contract tests; actual portable Metal kernels require the isolated MPS run."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from ypuddin.models import metal_attention as metal


@pytest.fixture
def compatible_runtime(monkeypatch):
    monkeypatch.setattr(metal.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(metal.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(metal.platform, "mac_ver", lambda: ("15.7.9", (), ""))
    monkeypatch.setattr(metal.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(metal, "current_profile", lambda: "macos-mps")
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(metal.metadata, "version", lambda name: "0.4.1")
    package = SimpleNamespace(
        varlen_attention=lambda *a, **kw: None, _C=SimpleNamespace(varlen_attention_bwd=lambda *a: None)
    )
    monkeypatch.setattr(metal, "import_module", lambda name: package)
    metal._metal_flash_function.cache_clear()
    yield package
    metal._metal_flash_function.cache_clear()


def test_runtime_identity_records_the_verified_stack_without_allocating(compatible_runtime, monkeypatch):
    monkeypatch.setattr(torch, "randn", lambda *a, **k: pytest.fail("preflight allocated a tensor"))
    identity = metal.require_metal_flash("mps")
    assert identity == metal.metal_flash_runtime_identity()
    assert (
        identity.items()
        >= {
            "implementation": "mtlattn-fp32-v1",
            "torch": "2.13.0",
            "mtlattn": "0.4.1",
            "platform": "Darwin",
            "machine": "arm64",
            "macos": "15.7.9",
        }.items()
    )
    for device in ("cpu", "cuda", "cuda:1"):
        with pytest.raises(ValueError, match="requires Apple MPS"):
            metal.require_metal_flash(device)


@pytest.mark.parametrize(
    "condition", ["system", "machine", "macos", "cpu", "wrong_profile", "torch", "cuda", "hip", "mps", "package", "backward"]
)
def test_runtime_rejects_unverified_or_incomplete_stack(compatible_runtime, monkeypatch, condition):
    if condition == "system":
        monkeypatch.setattr(metal.platform, "system", lambda: "Linux")
    elif condition == "machine":
        monkeypatch.setattr(metal.platform, "machine", lambda: "x86_64")
    elif condition == "macos":
        monkeypatch.setattr(metal.platform, "mac_ver", lambda: ("14.7", (), ""))
    elif condition == "cpu":
        monkeypatch.setattr(metal, "current_profile", lambda: "macos-cpu")
    elif condition == "wrong_profile":
        monkeypatch.setattr(metal, "current_profile", lambda: "linux-dtk")
    elif condition == "torch":
        monkeypatch.setattr(torch, "__version__", "2.14.0")
    elif condition in {"cuda", "hip"}:
        monkeypatch.setattr(torch.version, condition, "unexpected-runtime")
    elif condition == "mps":
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    elif condition == "package":
        monkeypatch.setattr(metal.metadata, "version", lambda name: "0.4.2")
    else:
        compatible_runtime._C.varlen_attention_bwd = None
    with pytest.raises((ValueError, RuntimeError)):
        metal.require_metal_flash("mps")


def test_missing_distribution_has_actionable_error(compatible_runtime, monkeypatch):
    def missing(name):
        raise metal.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metal.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="install its matching build"):
        metal.require_metal_flash("mps")


def descriptor(shape=(2, 3, 8, 64), dtype=torch.float32, device="mps"):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device=torch.device(device))


def test_eligibility_is_exact_and_does_not_need_the_optional_package():
    q = descriptor()
    k, v = descriptor((2, 3, 12, 64)), descriptor((2, 3, 12, 64))
    assert metal.metal_flash_eligible(q, k, v)
    for change in [
        descriptor(dtype=torch.float16),
        descriptor(device="cpu"),
        descriptor((2, 3, 8, 32)),
        descriptor((2, 3, 0, 64)),
        descriptor((1, 3, 8, 64)),
        descriptor((2, 1, 8, 64)),
        descriptor((3, 8, 64)),
    ]:
        assert not metal.metal_flash_eligible(change, k, v)
    assert not metal.metal_flash_eligible(q, k, descriptor((2, 3, 13, 64)))
    assert not metal.metal_flash_eligible(q, k, v, attn_mask=object())
    assert not metal.metal_flash_eligible(q, k, v, dropout_p=0.1)
    assert not metal.metal_flash_eligible(q, k, v, is_causal=True)


def test_unsupported_semantics_use_native_sdpa_without_importing_kernel(monkeypatch):
    monkeypatch.setattr(metal, "_metal_flash_function", lambda: pytest.fail("fallback imported kernel"))
    q, k, v = [torch.randn(1, 2, 8, 64, requires_grad=True) for _ in range(3)]
    mask = torch.ones(8, 8, dtype=torch.bool).tril()
    for options in ({"attn_mask": mask}, {"is_causal": True}, {"scale": 0.25}):
        torch.testing.assert_close(
            metal.metal_flash_sdpa(q, k, v, **options), F.scaled_dot_product_attention(q, k, v, **options)
        )


@pytest.mark.parametrize("batch,q_length,kv_length", [(1, 8, 8), (2, 5, 9)])
def test_packed_adapter_preserves_outputs_and_qkv_gradients(monkeypatch, batch, q_length, kv_length):
    calls = []

    def cpu_varlen(q, k, v, cu_q, cu_kv, max_q, *, scale, causal):
        calls.append((cu_q.tolist(), cu_kv.tolist(), max_q))
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert not causal
        return torch.cat(
            [
                F.scaled_dot_product_attention(
                    q[a:b].transpose(0, 1),
                    k[c:d].transpose(0, 1),
                    v[c:d].transpose(0, 1),
                    scale=scale,
                ).transpose(0, 1)
                for a, b, c, d in zip(cu_q[:-1], cu_q[1:], cu_kv[:-1], cu_kv[1:], strict=True)
            ]
        )

    monkeypatch.setattr(metal, "metal_flash_eligible", lambda *a, **k: True)
    monkeypatch.setattr(metal, "_metal_flash_function", lambda: cpu_varlen)
    inputs = [torch.randn(batch, 2, n, 64, requires_grad=True) for n in (q_length, kv_length, kv_length)]
    reference = [x.detach().clone().requires_grad_() for x in inputs]
    out = metal.metal_flash_sdpa(*inputs, scale=0.25)
    expected = F.scaled_dot_product_attention(*reference, scale=0.25)
    gradient = torch.randn_like(out)
    out.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-6)
    for actual, ref in zip(inputs, reference, strict=True):
        torch.testing.assert_close(actual.grad, ref.grad, rtol=2e-5, atol=2e-6)
    assert calls == [
        (
            list(range(0, (batch + 1) * q_length, q_length)),
            list(range(0, (batch + 1) * kv_length, kv_length)),
            q_length,
        )
    ]


def test_kernel_errors_propagate_without_retry(monkeypatch):
    monkeypatch.setattr(metal, "metal_flash_eligible", lambda *a, **k: True)

    def broken(*args, **kwargs):
        raise RuntimeError("Metal command failed")

    monkeypatch.setattr(metal, "_metal_flash_function", lambda: broken)
    q = torch.randn(1, 2, 8, 64)
    with pytest.raises(RuntimeError, match="Metal command failed"):
        metal.metal_flash_sdpa(q, q, q)


@pytest.mark.parametrize("family", ["anima", "krea2", "sdxl", "flux2"])
def test_family_rejects_runtime_before_inspecting_or_loading_weights(monkeypatch, family):
    from ypuddin.models import get_family

    model_family = get_family(family)

    def rejected(device):
        raise ValueError("unverified Metal runtime")

    monkeypatch.setattr(metal, "require_metal_flash", rejected)
    monkeypatch.setattr(
        model_family, "validate_config", lambda cfg: pytest.fail("weights inspected before runtime")
    )
    with pytest.raises(ValueError, match="unverified Metal runtime"):
        model_family.load(SimpleNamespace(attention="metal_flash"), None, device="mps", dtype=torch.float32)


def test_shared_shim_passes_mask_and_gqa_to_model_local_helper(monkeypatch):
    from ypuddin.models.anima.vendor.attention import _sdpa

    calls = []

    def inspect(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, v.shape, kwargs))
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    monkeypatch.setattr(metal, "metal_flash_sdpa", inspect)
    q, k, v = torch.randn(1, 4, 8, 64), torch.randn(1, 2, 8, 64), torch.randn(1, 2, 8, 64)
    mask = torch.ones(8, 8, dtype=torch.bool)
    _sdpa(q, k, v, "metal_flash", attn_mask=mask)
    assert calls[0][0] == calls[0][1] == calls[0][2]
    assert calls[0][3]["attn_mask"] is mask


@pytest.mark.parametrize("checkpointing", [False, True])
def test_flux2_scoped_processors_preserve_math_and_all_gradients_without_global_changes(
    monkeypatch, checkpointing
):
    pytest.importorskip("diffusers")
    from diffusers import Flux2Transformer2DModel
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    model = Flux2Transformer2DModel(
        in_channels=128,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=24,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
        mlp_ratio=2,
        guidance_embeds=False,
    ).train()
    reference = copy.deepcopy(model)
    registry_before = _AttentionBackendRegistry.get_active_backend()
    function_before = F.scaled_dot_product_attention
    calls = []

    def dispatch(*args, **kwargs):
        calls.append(True)
        return F.scaled_dot_product_attention(*args, **kwargs)

    monkeypatch.setattr(metal, "metal_flash_sdpa", dispatch)
    metal.install_metal_flash_processors(model, "flux2")
    metal.validate_metal_flash_processors(model)
    assert _AttentionBackendRegistry.get_active_backend() == registry_before
    assert F.scaled_dot_product_attention is function_before
    if checkpointing:
        model.enable_gradient_checkpointing()
        reference.enable_gradient_checkpointing()
    states, text = torch.randn(1, 4, 128), torch.randn(1, 3, 24)
    outputs, gradients = [], []
    for current in (reference, model):
        x, e = states.clone().requires_grad_(), text.clone().requires_grad_()
        result = current(
            hidden_states=x,
            encoder_hidden_states=e,
            timestep=torch.tensor([0.4]),
            img_ids=torch.arange(16).reshape(4, 4),
            txt_ids=torch.arange(12).reshape(3, 4),
            return_dict=False,
        )[0]
        result.square().mean().backward()
        outputs.append(result.detach())
        gradients.append(
            {**{n: p.grad for n, p in current.named_parameters()}, "input": x.grad, "text": e.grad}
        )
    assert len(calls) == (4 if checkpointing else 2)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for name, expected in gradients[0].items():
        if expected is None:
            assert gradients[1][name] is None
        else:
            torch.testing.assert_close(gradients[1][name], expected, rtol=0, atol=0)
    before = len(calls)
    F.scaled_dot_product_attention(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
    assert len(calls) == before
    metal.restore_metal_flash_processors(model)
    with pytest.raises(ValueError, match="processors changed"):
        metal.validate_metal_flash_processors(model)


def test_sdxl_exact_processor_allowlist_and_exception_cleanup(monkeypatch):
    pytest.importorskip("diffusers")
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    layer = Attention(query_dim=16, heads=2, dim_head=8, processor=AttnProcessor2_0())
    original = layer.processor
    container = SimpleNamespace(attn_processors={"processor": original})
    container.set_attn_processor = lambda processors: setattr(container, "attn_processors", processors)
    metal.install_metal_flash_processors(container, "sdxl")
    layer.set_processor(container.attn_processors["processor"])

    def broken(*args, **kwargs):
        raise RuntimeError("selected kernel failed")

    monkeypatch.setattr(metal, "metal_flash_sdpa", broken)
    with pytest.raises(RuntimeError, match="selected kernel failed"):
        layer(torch.randn(1, 4, 16))
    # The exceptional exit must also restore the thread-local mode stack.
    F.scaled_dot_product_attention(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
    container.attn_processors["unknown"] = object()
    before = dict(container.attn_processors)
    with pytest.raises(ValueError, match="Unsupported"):
        metal.install_metal_flash_processors(container, "sdxl")
    assert container.attn_processors == before
