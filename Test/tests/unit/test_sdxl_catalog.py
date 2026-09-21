from ypuddin.server.model_recommendations import RECOMMENDATIONS


def test_default_sdxl_download_is_pinned_illustrious_v01_complete_checkpoint():
    entries = [entry for entry in RECOMMENDATIONS if entry.family == "sdxl" and entry.recommended]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "dit"
    assert entry.size == 6938040760
    assert entry.sha256 == "3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9"
    assert [(source.provider, source.repo_id, source.filename) for source in entry.sources] == [
        ("huggingface", "OnomaAIResearch/Illustrious-xl-early-release-v0", "Illustrious-XL-v0.1.safetensors"),
        ("modelscope", "OnomaAIResearch/Illustrious-xl-early-release-v0", "Illustrious-XL-v0.1.safetensors"),
    ]
    assert all(source.revision not in {"main", "master"} for source in entry.sources)
