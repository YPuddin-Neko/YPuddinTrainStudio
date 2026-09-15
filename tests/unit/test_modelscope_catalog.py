"""Published source contracts; requests are captured without downloading model weights."""

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
    assert target_filename == filename
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
