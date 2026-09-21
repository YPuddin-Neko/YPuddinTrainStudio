"""Turbo identity/admission and real CPU integration; no official weights required."""

import hashlib
import math
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from ypuddin.config import ModelConfig, TrainConfig
from ypuddin.models.krea2 import variants
from ypuddin.models.krea2.family import Krea2Family
from ypuddin.server import create_app
from ypuddin.server.family_config import initial_family_config
from ypuddin.server.model_recommendations import find_recommendation, remember_verified_file


def krea_file(path):
    save_file(
        {
            "first.weight": torch.zeros(2, 64),
            "txtfusion.projector.weight": torch.zeros(1, 12),
            "blocks.0.attn.wq.weight": torch.zeros(2, 2),
        },
        str(path),
    )
    return path


def prove(path, monkeypatch, variant="turbo"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(variants.KNOWN_VARIANTS, digest, variant)
    variants.record_verified_variant(path, digest)
    return digest


def test_variant_names_and_geometry_are_not_evidence(tmp_path):
    path = krea_file(tmp_path / "krea2_turbo_bf16.safetensors")
    assert variants.verified_variant(path) is None
    with pytest.raises(ValueError, match="无法自动区分"):
        variants.resolve_variant(path, "auto")
    assert variants.resolve_variant(path, "raw") == "raw"
    assert variants.resolve_variant(path, "turbo") == "turbo"


def test_verified_identity_binds_current_file_and_invalidates_after_edit(tmp_path, monkeypatch):
    path = krea_file(tmp_path / "arbitrary.safetensors")
    prove(path, monkeypatch)
    assert variants.resolve_variant(path, "auto") == "turbo"
    with pytest.raises(ValueError, match="不符"):
        variants.resolve_variant(path, "raw")
    path.write_bytes(path.read_bytes() + b"changed")
    assert variants.verified_variant(path) is None
    with pytest.raises(ValueError, match="无法自动区分"):
        variants.resolve_variant(path, "auto")


def test_turbo_is_rejected_for_training_before_loading_paths():
    cfg = TrainConfig(model=ModelConfig(family="krea2", krea2_variant="turbo"))
    problems = Krea2Family().training_options_errors(cfg)
    assert any(p["loc"] == "model.krea2_variant" and "仅用于采样" in p["msg"] for p in problems)


def test_turbo_fixed_shift_matches_official_schedule_and_respects_override():
    family = Krea2Family()
    loaded = SimpleNamespace(extra={"variant": "turbo"})
    for tokens in (256, 1024, 4096, 16384):
        assert family.sampling_shift_for_model(loaded, tokens) == pytest.approx(math.exp(1.15))
    for override in (None, 2.0):
        called = []
        family.sample_latents(
            loaded,
            lambda x, t, called=called: called.append(t.clone()) or torch.zeros_like(x),
            (1, 1),
            steps=8,
            cfg=0,
            **({"shift": override} if override is not None else {}),
        )
        base = torch.linspace(1.0, 0.0, 9)[:-1]
        # Krea official timesteps(): exp(mu)/(exp(mu)+(1/t-1)), sigma=1.
        shift = math.exp(1.15) if override is None else override
        expected = shift / (shift + (1 / base - 1))
        torch.testing.assert_close(torch.cat(called), expected)


@pytest.mark.parametrize("sampler", ["euler", "heun"])
@pytest.mark.parametrize("guidance", [0.0, 1.0, 2.5])
def test_turbo_cfg_matches_cond_plus_g_delta_and_zero_skips_negative(sampler, guidance):
    family = Krea2Family()
    loaded = SimpleNamespace(extra={"variant": "turbo"})
    defaults = family.sampling_defaults(loaded)
    assert (defaults.steps, defaults.cfg) == (8, 0.0)
    calls = []

    def uncond(x, t):
        calls.append(1)
        return torch.full_like(x, -1)

    actual = family.sample_latents(
        loaded,
        lambda x, t: torch.full_like(x, 2),
        (1, 1, 2, 2),
        steps=8,
        cfg=guidance,
        sampler=sampler,
        predict_uncond=uncond,
        generator=torch.Generator().manual_seed(73),
    )
    noise = torch.randn((1, 1, 2, 2), generator=torch.Generator().manual_seed(73))
    torch.testing.assert_close(actual, noise - (2 + 3 * guidance))
    assert bool(calls) == family.sampling_needs_uncond(loaded, guidance) == (guidance > 0)


@pytest.mark.parametrize("guidance", [-1.0, float("nan"), float("inf")])
def test_turbo_rejects_invalid_guidance(guidance):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        Krea2Family().sample_latents(
            SimpleNamespace(extra={"variant": "turbo"}), lambda x, t: x, (1, 1), steps=1, cfg=guidance
        )


def test_raw_sampling_defaults_and_cfg_convention_remain_unchanged():
    family = Krea2Family()
    loaded = SimpleNamespace(extra={"variant": "raw"})
    assert (family.sampling_defaults(loaded).steps, family.sampling_defaults(loaded).cfg) == (28, 5.5)
    assert not family.sampling_needs_uncond(loaded, 1)
    assert family.sampling_needs_uncond(loaded, 0)
    result = family.sample_latents(
        loaded,
        lambda x, t: torch.full_like(x, 2),
        (1, 1),
        steps=1,
        cfg=0,
        predict_uncond=lambda x, t: torch.full_like(x, -1),
        generator=torch.Generator().manual_seed(1),
    )
    torch.testing.assert_close(result, torch.randn((1, 1), generator=torch.Generator().manual_seed(1)) + 1)


@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "missing")
    with TestClient(app) as client:
        yield client, app.state.ctx


def test_registry_requires_variant_and_protects_training_default(api, tmp_path):
    client, context = api
    raw = krea_file(tmp_path / "raw.safetensors")
    turbo = krea_file(tmp_path / "renamed.safetensors")
    body = {"family": "krea2", "kind": "dit", "path": str(raw), "is_default": True}
    assert client.post("/api/models", json=body).status_code == 422
    response = client.post("/api/models", json={**body, "variant": "raw"})
    assert response.status_code == 200, response.text
    denied = client.post("/api/models", json={**body, "path": str(turbo), "variant": "turbo"})
    assert denied.status_code == 422 and denied.json()["error"]["code"] == "model.purpose"
    added = client.post(
        "/api/models", json={**body, "path": str(turbo), "variant": "turbo", "is_default": False}
    )
    assert added.status_code == 200, added.text
    assert added.json()["purpose"] == "inference" and added.json()["variant"] == "turbo"
    assert client.patch(f"/api/models/{added.json()['id']}", json={"is_default": True}).status_code == 422
    assert initial_family_config(context, "krea2")["model"]["dit_path"] == str(raw)


def test_unknown_legacy_krea_download_retry_requires_confirmation(api):
    client, _ = api
    client.app.state.model_downloads.tasks["old-krea"] = {
        "family": "krea2",
        "kind": "dit",
        "status": "failed",
    }
    response = client.post("/api/models/downloads/old-krea/retry")
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "download.variant"


def test_verified_turbo_inspection_scan_and_legacy_default_never_select_training(api, tmp_path, monkeypatch):
    client, context = api
    path = krea_file(tmp_path / "unrelated_name.safetensors")
    digest = prove(path, monkeypatch)
    remember_verified_file(context, path, digest)
    inspection = client.post("/api/models/inspect", json={"path": str(path)}).json()
    assert inspection["variant"] == "turbo" and inspection["purpose"] == "inference"
    added = client.post("/api/models", json={"family": "krea2", "kind": "dit", "path": str(path)})
    assert added.status_code == 200, added.text
    mid = added.json()["id"]
    # A stale pre-feature default row must not leak Turbo into a fresh recipe.
    context.db.update("models", mid, {"is_default": 1, "purpose": "training", "variant": None})
    assert not client.get("/api/models").json()[0]["is_default"]
    assert initial_family_config(context, "krea2")["model"]["dit_path"] is None
    assert variants.resolve_variant(path, "auto") == "turbo"


def test_turbo_catalog_is_pinned_and_inference_only():
    for dtype, size in (("bf16", 26283332608), ("fp8", 13141730784)):
        entry = find_recommendation(f"krea2-turbo-{dtype}")
        assert entry.purpose == "inference" and entry.variant == "turbo"
        assert entry.size == size and variants.KNOWN_VARIANTS[entry.sha256] == "turbo"
        assert entry.sources[0].revision == "e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96"
