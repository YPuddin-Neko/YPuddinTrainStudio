"""Family caption formats constrain the shared text boundary, not model names."""

import json
from dataclasses import replace

import pytest
from PIL import Image

from ypuddin.config import TrainConfig
from ypuddin.data.dataset import DataConfigError, build_data, prepare_data_layout
from ypuddin.models import get_family
from ypuddin.server import routes_core


def config(root, family="toy", extension="auto"):
    return TrainConfig.model_validate(
        {
            "model": {"family": family},
            "dataset": {
                "sources": [{"path": str(root), "caption_ext": extension}],
                "resolutions": [64],
                "bucket_step": 16,
            },
        }
    )


def files(root, document=None):
    Image.new("RGB", (64, 64), "red").save(root / "portrait.png")
    (root / "portrait.txt").write_text("selected TXT")
    (root / "portrait.json").write_text(json.dumps(document or {"tags": ["smile"], "nl": "A portrait."}))


@pytest.mark.parametrize("name", ["anima", "krea2", "sdxl", "flux2", "toy"])
def test_implemented_families_share_structured_json_to_text(name, tmp_path):
    files(tmp_path)
    family = get_family(name)
    assert family.spec.caption_formats == ("txt", "json")
    data = build_data(config(tmp_path, name), family.spec.latent, cache_root=tmp_path / "cache")
    assert data.train[0]["caption"] == "smile. A portrait."
    assert data.train.use_cached_captions() == ["smile. A portrait."]


def txt_only(monkeypatch):
    family = get_family("toy")
    monkeypatch.setattr(family, "spec", replace(family.spec, caption_formats=("txt",)))
    monkeypatch.setattr(routes_core, "_FAMILY_INFO", {})
    return family


def test_family_api_declares_actual_formats(monkeypatch):
    txt_only(monkeypatch)
    assert routes_core.family_info("toy")["caption_formats"] == ["txt"]


def test_txt_only_auto_skips_json_without_mutating_config(monkeypatch, tmp_path):
    family = txt_only(monkeypatch)
    files(tmp_path, {"tags": 42})  # Ignored sidecars do not become errors or fallback captions.
    cfg = config(tmp_path)
    original = cfg.to_dict()
    data = build_data(cfg, family.spec.latent, cache_root=tmp_path / "cache")
    assert data.train[0]["caption"] == "selected TXT"
    assert data.train.use_cached_captions() == ["selected TXT"]
    assert cfg.to_dict() == original


@pytest.mark.parametrize("extension", [".json", ".JSON"])
def test_unsupported_explicit_json_reports_source_and_file(monkeypatch, tmp_path, extension):
    family = txt_only(monkeypatch)
    files(tmp_path)
    with pytest.raises(DataConfigError, match=r"portrait.json.*does not support JSON") as caught:
        prepare_data_layout(config(tmp_path, extension=extension), family.spec.latent)
    assert caught.value.loc == "dataset.sources.0.caption_ext"


def test_unsupported_validation_json_is_also_rejected(monkeypatch, tmp_path):
    family = txt_only(monkeypatch)
    files(tmp_path)
    cfg = config(tmp_path, extension=".txt")
    cfg.validation.enabled = True
    cfg.validation.sources = [cfg.dataset.sources[0].model_copy(update={"caption_ext": ".json"})]
    with pytest.raises(DataConfigError, match="does not support JSON") as caught:
        prepare_data_layout(cfg, family.spec.latent)
    assert caught.value.loc == "validation.sources.0.caption_ext"
