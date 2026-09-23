"""Klein-only admission retains inspectable legacy assets without changing their data."""

from types import SimpleNamespace

import pytest

from Test.tests.unit import test_sdxl_model_registration
from Test.tests.unit.test_flux_model_inspection import hf_flux_directory, sparse_headers, transformer_shapes
from ypuddin.server.db import now
from ypuddin.server.model_inspection import FLUX1_RETIRED_REASON, FLUX2_DEV_UNSUPPORTED_REASON

api = test_sdxl_model_registration.api


def insert_legacy(context, path, *, family, id_):
    context.db.insert(
        "models",
        {
            "id": id_,
            "family": family,
            "kind": "dit",
            "path": str(path),
            "size": 1,
            "dtype": "bf16",
            "is_default": 1,
            "created_at": now(),
        },
    )


@pytest.mark.parametrize(
    "variant,family,reason",
    [("flux", "flux", FLUX1_RETIRED_REASON), ("dev", "flux2", FLUX2_DEV_UNSUPPORTED_REASON)],
)
def test_retired_pipeline_is_diagnostic_only_and_legacy_rows_survive(api, tmp_path, variant, family, reason):
    client, context = api
    root = hf_flux_directory(tmp_path / "klein_name_is_not_evidence", variant)
    index_before = (root / "model_index.json").read_bytes()
    insert_legacy(context, root, family=family, id_="legacy")
    detected = client.post("/api/models/inspect", json={"path": str(root)})
    assert detected.status_code == 200, detected.text
    assert detected.json()["family"] == family and reason in detected.json()["warnings"]
    response = client.post(
        "/api/models", json={"family": family, "kind": "dit", "path": str(root), "is_default": True}
    )
    assert response.status_code == 422 and reason in response.text
    # Cannot relabel a FLUX.1 pipeline as Klein either.
    response = client.post("/api/models", json={"family": "flux2", "kind": "dit", "path": str(root)})
    assert response.status_code == 422 and reason in response.text
    assert client.patch("/api/models/legacy", json={"is_default": True}).status_code == 422
    assert client.get("/api/models").json()[0]["id"] == "legacy"
    assert client.get("/api/models").json()[0]["is_default"] is True
    assert client.get("/api/models").json()[0]["unsupported_reason"] == reason
    assert "unsupported_reason" not in context.db.fetchone("SELECT * FROM models WHERE id='legacy'")
    assert client.patch("/api/models/legacy", json={"is_default": False}).status_code == 200
    assert (root / "model_index.json").read_bytes() == index_before
    assert (root / "transformer/diffusion_pytorch_model.safetensors").exists()


@pytest.mark.parametrize("native", [False, True])
def test_raw_dev_is_rejected_using_header_and_does_not_clear_klein_default(api, tmp_path, native):
    client, _ = api
    klein = hf_flux_directory(tmp_path / "supported", "klein4")
    existing = client.post(
        "/api/models", json={"family": "flux2", "kind": "dit", "path": str(klein), "is_default": True}
    ).json()
    dev = sparse_headers(
        tmp_path / "klein_base_4b.safetensors", transformer_shapes("flux2", 6144, native=native)
    )
    response = client.post(
        "/api/models", json={"family": "flux2", "kind": "dit", "path": str(dev), "is_default": True}
    )
    assert response.status_code == 422 and FLUX2_DEV_UNSUPPORTED_REASON in response.text
    assert [(row["id"], row["is_default"]) for row in client.get("/api/models").json()] == [
        (existing["id"], True)
    ]
    assert dev.exists()


@pytest.mark.parametrize("variant", ["klein4", "klein9"])
def test_klein_registration_and_scan_are_not_blocked_by_misleading_names(api, tmp_path, variant):
    client, _ = api
    root = hf_flux_directory(tmp_path / "flux1_dev_wrong_name", variant, sharded=True)
    response = client.post(
        "/api/models", json={"family": "flux2", "kind": "dit", "path": str(root), "is_default": True}
    )
    assert response.status_code == 200, response.text
    assert response.json()["family"] == "flux2" and response.json()["path"] == str(root)
    assert response.json()["unsupported_reason"] is None
    assert client.post("/api/models/scan", json={"path": str(root), "family": "flux2"}).json() == []


def test_scan_skips_legacy_flux_and_dev_but_keeps_klein_root(api, tmp_path):
    client, _ = api
    models = tmp_path / "collection"
    models.mkdir()
    hf_flux_directory(models / "old", "flux")
    hf_flux_directory(models / "not_klein", "dev", sharded=True)
    klein = hf_flux_directory(models / "keep", "klein4", sharded=True)
    sparse_headers(models / "dev_bare.safetensors", transformer_shapes("flux2", 6144))
    response = client.post("/api/models/scan", json={"path": str(models), "family": "flux2"})
    assert response.status_code == 200, response.text
    assert [(row["family"], row["path"]) for row in response.json()] == [("flux2", str(klein))]
    assert (models / "dev_bare.safetensors").exists()


def test_flux1_download_and_retry_are_rejected_without_touching_history(api, tmp_path):
    client, _ = api
    downloads = client.app.state.model_downloads
    downloads.opener = SimpleNamespace(
        open=lambda *a, **k: pytest.fail("retired downloads must not use the network")
    )
    response = client.post(
        "/api/models/downloads",
        json={"family": "flux", "kind": "dit", "repo_id": "owner/model", "filename": "weights.safetensors"},
    )
    assert response.status_code == 422 and FLUX1_RETIRED_REASON in response.text
    assert client.get("/api/models/downloads").json() == []
    row = dict(
        id="old-task",
        family="flux",
        kind="dit",
        provider="huggingface",
        mirror="official",
        source_url="https://huggingface.co/owner/repo/resolve/main/model.safetensors",
        filename="model.safetensors",
        target_path=str(tmp_path / "old.safetensors"),
        status="failed",
        downloaded_bytes=0,
        total_bytes=None,
        error="old error",
        model_id=None,
        created_at=1,
        finished_at=2,
        dtype="bf16",
        is_default=True,
    )
    downloads.tasks[row["id"]] = row.copy()
    downloads._persist()
    response = client.post("/api/models/downloads/old-task/retry", json={})
    assert response.status_code == 410 and FLUX1_RETIRED_REASON in response.text
    assert downloads.tasks[row["id"]] == row
    assert len(client.get("/api/models/downloads").json()) == 1


def test_klein_rejects_dev_mistral_encoder_and_bare_qwen_file(api, tmp_path):
    client, _ = api
    dev = hf_flux_directory(tmp_path / "dev", "dev")
    response = client.post(
        "/api/models", json={"family": "flux2", "kind": "text_encoder", "path": str(dev / "text_encoder")}
    )
    assert response.status_code == 422 and FLUX2_DEV_UNSUPPORTED_REASON in response.text
    klein = hf_flux_directory(tmp_path / "klein", "klein4")
    response = client.post(
        "/api/models",
        json={
            "family": "flux2",
            "kind": "text_encoder",
            "path": str(klein / "text_encoder/model.safetensors"),
        },
    )
    assert response.status_code == 200, response.text
    bare = sparse_headers(tmp_path / "bare-qwen.safetensors", {"model.embed_tokens.weight": [151936, 2560]})
    response = client.post(
        "/api/models", json={"family": "flux2", "kind": "text_encoder", "path": str(bare)}
    )
    assert response.status_code == 422 and "MLP geometry mismatch" in response.text
    response = client.post(
        "/api/models", json={"family": "flux2", "kind": "text_encoder", "path": str(klein / "text_encoder")}
    )
    assert response.status_code == 200, response.text


def test_new_klein_defaults_skip_legacy_dev_components_without_mutating_rows(api, tmp_path):
    client, context = api
    dev = hf_flux_directory(tmp_path / "dev", "dev")
    roles = {
        "dit": dev,
        "text_encoder": dev / "text_encoder",
        "tokenizer": dev / "tokenizer",
        "vae": dev / "vae",
    }
    for role, path in roles.items():
        context.db.insert(
            "models",
            {
                "id": role,
                "family": "flux2",
                "kind": role,
                "path": str(path),
                "size": 1,
                "dtype": "bf16",
                "is_default": 1,
                "created_at": now(),
            },
        )
    before = client.get("/api/models").json()
    response = client.get("/api/config/defaults?family=flux2")
    assert response.status_code == 200, response.text
    model = response.json()["model"]
    assert model["dit_path"] is model["text_encoder_path"] is model["tokenizer_path"] is None
    assert model["vae_path"] == str(dev / "vae")  # Shared FLUX.2 VAE remains compatible with Klein.
    assert client.get("/api/models").json() == before
    assert client.get("/api/config/defaults?family=flux").status_code == 422


@pytest.mark.parametrize("suffix", [".ckpt", ".bin", ".pt"])
def test_klein_does_not_accept_uninspectable_legacy_file_formats(api, tmp_path, suffix):
    client, _ = api
    path = tmp_path / ("klein" + suffix)
    path.write_bytes(b"not a safetensors component")
    response = client.post("/api/models", json={"family": "flux2", "kind": "dit", "path": str(path)})
    assert response.status_code == 422 and "safetensors" in response.text
    assert path.read_bytes() == b"not a safetensors component"


def test_scan_does_not_register_bare_klein_encoder_without_hf_directory(api, tmp_path):
    from Test.tests.unit.test_flux_model_inspection import decoder_shapes

    client, _ = api
    root = tmp_path / "encoders"
    root.mkdir()
    path = sparse_headers(root / "qwen.safetensors", decoder_shapes(2560))
    response = client.post("/api/models/scan", json={"path": str(root), "family": "flux2"})
    assert response.status_code == 200 and response.json() == []
    assert path.exists()
