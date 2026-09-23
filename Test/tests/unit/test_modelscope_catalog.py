"""Published source contracts; requests are captured without downloading model weights."""

from pathlib import PurePosixPath
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from ypuddin.server.model_downloads import resolve_source
from ypuddin.server.model_recommendations import find_recommendation
from ypuddin.server.routes_model_recommendations import (
    RecommendationDownloadRequest,
    download_recommendation,
)

VERIFIED_SOURCES = [
    (
        "flux2-klein-base-4b",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "flux-2-klein-base-4b.safetensors",
        "384cd13880a8dd205782035743eead9ec374df5c",
        7751105712,
        "9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92",
        "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/resolve/a3b4f4849157f664bdbc776fd7453c2783562f4d/flux-2-klein-base-4b.safetensors",
    ),
    (
        "flux2-klein-base-9b",
        "black-forest-labs/FLUX.2-klein-base-9B",
        "flux-2-klein-base-9b.safetensors",
        "332e29c86837a0143b3984ebda6f875fd9df76ab",
        18157185168,
        "4a54fad7f5f741b99eee217198daac20b8d8e515e2a1f5b064fd51cf074f95bd",
        "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/resolve/32773329fbe7e81a90ef971740e8ba4b0364ecf3/flux-2-klein-base-9b.safetensors",
    ),
    (
        "flux2-qwen3-4b",
        "Comfy-Org/flux2-klein-4B",
        "split_files/text_encoders/qwen_3_4b.safetensors",
        "305618a79d0d7b2b167266c8a4082ee99e948ca2",
        8044982048,
        "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
        "https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/text_encoders/qwen_3_4b.safetensors",
    ),
    (
        "flux2-qwen3-8b",
        "Comfy-Org/flux2-klein-9B",
        "split_files/text_encoders/qwen_3_8b.safetensors",
        "0bb21a1a5059384ac93399fc1f31e752e605d1ee",
        16381517176,
        "f0ff9239d56269ca1d05e5f86da6a79fac111af464955681f11c7ab0ec5ef6c1",
        "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/3f62d9d8ae1fec33c6e91453d5c712855b096b55/split_files/text_encoders/qwen_3_8b.safetensors",
    ),
    (
        "flux2-vae",
        "Comfy-Org/flux2-klein-4B",
        "split_files/vae/flux2-vae.safetensors",
        "305618a79d0d7b2b167266c8a4082ee99e948ca2",
        336211292,
        "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
        "https://huggingface.co/Comfy-Org/flux2-klein-4B/resolve/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/vae/flux2-vae.safetensors",
    ),
    (
        "sdxl-illustrious-v01",
        "OnomaAIResearch/Illustrious-xl-early-release-v0",
        "Illustrious-XL-v0.1.safetensors",
        "6f7fa36d9cb8aede0e05eeb8790966151a0d21a0",
        6938040760,
        "3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9",
        "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/f08f0826ffe32183ba2d1f4106dd5b32e195a02e/Illustrious-XL-v0.1.safetensors",
    ),
    (
        "krea2-raw-bf16",
        "krea/Krea-2-Raw",
        "raw.safetensors",
        "4c6e0f5dee814bea2d3f5e1fe7264e78a42c4370",
        26283332608,
        "f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7",
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_bf16.safetensors",
    ),
]


def captured_request(model_id, provider):
    def start(body, *, recommendation):
        return body, recommendation

    return download_recommendation(
        model_id,
        RecommendationDownloadRequest(provider=provider),
        SimpleNamespace(start=start),
    )


@pytest.mark.parametrize("model_id,repo,filename,revision,size,sha256,hf_url", VERIFIED_SOURCES)
def test_modelscope_recommendation_uses_verified_source_and_preserves_integrity(
    model_id, repo, filename, revision, size, sha256, hf_url
):
    body, verification = captured_request(model_id, "modelscope")
    url, target_filename = resolve_source(body)
    parsed = urlsplit(url)
    assert parsed.scheme == "https" and parsed.netloc == "modelscope.cn"
    assert parsed.path == f"/api/v1/models/{repo}/repo"
    assert parse_qs(parsed.query) == {"Revision": [revision], "FilePath": [filename]}
    assert target_filename == PurePosixPath(filename).name
    assert verification.id == model_id
    assert verification.size == size and verification.sha256 == sha256
    assert body.purpose == "training" and body.is_default
    assert body.variant == ("raw" if model_id == "krea2-raw-bf16" else None)
    assert len({source.provider for source in verification.sources}) == 2
    assert len(verification.sources) == 2


@pytest.mark.parametrize("model_id,repo,filename,revision,size,sha256,hf_url", VERIFIED_SOURCES)
def test_modelscope_source_override_keeps_original_huggingface_source(
    model_id, repo, filename, revision, size, sha256, hf_url
):
    body, verification = captured_request(model_id, "huggingface")
    assert resolve_source(body)[0] == hf_url
    assert verification.size == size and verification.sha256 == sha256
    assert verification is find_recommendation(model_id)
