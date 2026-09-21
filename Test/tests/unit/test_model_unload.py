"""Cache-stage unload must reclaim cyclic modules before releasing device blocks."""

import gc
import weakref

import pytest
import torch
from torch import nn

from ypuddin.models.anima.family import AnimaLatent
from ypuddin.models.anima.text import AnimaText
from ypuddin.models.flux.latent import FluxLatent
from ypuddin.models.flux.text import FluxText
from ypuddin.models.flux2.latent import Flux2Latent
from ypuddin.models.flux2.text import Flux2Text
from ypuddin.models.krea2.text import Krea2Text
from ypuddin.models.sdxl.latent import SDXLLatent
from ypuddin.models.sdxl.text import SDXLText


@pytest.fixture
def manual_gc():
    """Keep automatic collection from accidentally hiding a missing unload GC."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


def cyclic_module():
    model = nn.Linear(2, 2)
    # A callback closure keeps the module and its parameters alive after its
    # owner's reference is dropped, as can happen in transformer modules.
    model.status_callback = lambda: model.training
    return model


PIPELINES = [
    (Krea2Text, "encoder"),
    (AnimaText, "encoder"),
    (SDXLText, "models"),
    (FluxText, "models"),
    (Flux2Text, "model"),
    (AnimaLatent, "vae"),  # Also used by Krea2.
    (SDXLLatent, "vae"),
    (FluxLatent, "vae"),
    (Flux2Latent, "vae"),
]


@pytest.mark.parametrize("pipeline_class,attribute", PIPELINES, ids=[cls.__name__ for cls, _ in PIPELINES])
@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_unload_reclaims_cyclic_weights_before_allocator_release(
    manual_gc, monkeypatch, pipeline_class, attribute, device
):
    # Exercise the actual pipeline unload without loading checkpoints or
    # requiring an accelerator: the tiny weight tensors remain on CPU.
    pipeline = pipeline_class.__new__(pipeline_class)
    pipeline.device = torch.device(device)
    models = [cyclic_module() for _ in range(2 if attribute == "models" else 1)]
    refs = [weakref.ref(obj) for model in models for obj in (model, model.weight, model.bias)]
    setattr(pipeline, attribute, models if attribute == "models" else models[0])
    del models
    assert all(ref() is not None for ref in refs)

    calls = []
    collect = gc.collect

    def collect_cycles():
        calls.append("collect")
        return collect()

    def empty_cache(backend):
        # The order matters: empty_cache alone cannot free live cyclic weights.
        assert all(ref() is None for ref in refs)
        calls.append(backend)

    monkeypatch.setattr(gc, "collect", collect_cycles)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache("cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: empty_cache("mps"))

    pipeline.unload()
    assert all(ref() is None for ref in refs)
    expected = ["collect"] + ([] if device == "cpu" else [torch.device(device).type])
    assert calls == expected

    # Warm caches and deferred backbone loading can request unload again.
    # They must not run another full collection when no weights were loaded.
    pipeline.unload()
    assert calls == expected


def test_unload_preserves_live_external_model_and_cached_tensor(manual_gc):
    pipeline = Krea2Text.__new__(Krea2Text)
    pipeline.device = torch.device("cpu")
    model = cyclic_module()
    pipeline.encoder = model
    with torch.no_grad():
        cached = model(torch.ones(1, 2)).detach().clone()
    expected = cached.clone()

    pipeline.unload()

    assert pipeline.encoder is None
    torch.testing.assert_close(cached, expected, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model(torch.ones(1, 2)), expected, rtol=0, atol=0)
