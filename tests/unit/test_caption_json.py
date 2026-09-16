import hashlib
import json
import random
from dataclasses import replace

import pytest
from PIL import Image

from ypuddin.config import CaptionConfig, DatasetSourceConfig, TrainConfig
from ypuddin.data.caption_json import EDIT_MARKER, StructuredCaption, load_caption_structure, parse_caption
from ypuddin.data.captions import (
    caption_content,
    caption_description,
    caption_variants_for_cache,
    read_caption,
    read_editable_caption,
    read_training_caption,
    transform_caption,
    transform_caption_deterministic,
    write_caption,
)
from ypuddin.data.dataset import DataConfigError, build_data, prepare_data_layout
from ypuddin.data.index import caption_for, caption_target, content_hash, dataset_fingerprint, scan_sources
from ypuddin.models import get_family


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def full_caption():
    return {
        "fixed": {"quality": "newest, safe", "series": "series", "artist": "@artist"},
        "character": {"name": "person", "variant": "adult"},
        "from_path": {"appearance": ["red hair", "hat"], "extra_tags": ["walking"]},
        "ai_output": {
            "count": "1girl",
            "appearance": ["red hair", "eyes"],
            "tags": ["smile", "standing"],
            "environment": ["outdoors", "sky"],
            "nl": "One person smiles, looking at the sky.",
        },
        "meta": {"model": "metadata only", "trigger": "portrait"},
    }


def test_full_document_renders_fixed_fields_and_merged_groups(full_caption, tmp_path):
    path = save_json(tmp_path / "p.JSON", full_caption)
    raw = read_training_caption(path)
    assert isinstance(raw, StructuredCaption)
    assert raw.fixed == ("newest", "safe", "1girl", "person", "adult", "series", "@artist")
    assert raw.appearance == ("red hair", "eyes", "hat")
    assert raw.tags == ("smile", "standing", "walking")
    assert read_caption(path) == (
        "portrait, newest, safe, 1girl, person, adult, series, @artist, red hair, eyes, hat, "
        "smile, standing, walking, outdoors, sky. One person smiles, looking at the sky."
    )


def test_simplified_tags_do_not_override_other_classifications():
    raw = parse_caption(
        {
            "quality": "safe",
            "count": "1boy",
            "character": "person",
            "series": "project",
            "artist": "@x",
            "appearance": ["hat"],
            "tags": ["smile"],
            "environment": ["room"],
            "nl": "A person, indoors.",
        }
    )
    assert raw.text() == "safe, 1boy, person, project, @x, hat, smile, room. A person, indoors."


def test_standard_nested_categories_and_legacy_trigger():
    nested = parse_caption({"tags": {"quality": ["safe"], "tags": ["hat"], "nl": "Prose."}})
    assert nested.text() == "safe, hat. Prose."
    legacy = parse_caption({"tags": ["blue", "legacy", "BLUE", "green"], "meta": {"trigger": "legacy"}})
    assert legacy.text() == "legacy, blue, green"
    out = transform_caption(legacy, CaptionConfig(shuffle=True, tag_dropout=1), random.Random(7))
    assert out == "legacy"


def test_full_editor_override_is_authoritative_even_when_empty(full_caption):
    full_caption["tags"] = ["only edited", "new tag"]
    raw = parse_caption(full_caption)
    assert raw.fixed == () and not raw.trigger and not raw.appearance
    assert raw.text() == "only edited, new tag. One person smiles, looking at the sky."
    full_caption["tags"] = []
    assert parse_caption(full_caption).text() == "One person smiles, looking at the sky."


def test_shuffle_stays_within_groups_and_never_moves_fixed_or_prose(full_caption):
    raw = parse_caption(full_caption)
    cfg = CaptionConfig(shuffle=True, trigger_word="training", keep_tokens=999)
    variants = [transform_caption(raw, cfg, random.Random(seed)) for seed in range(12)]
    assert len(set(variants)) > 1
    for value in variants:
        assert value.endswith(". " + raw.nl)
        tokens = value[: -len(raw.nl) - 2].split(", ")
        assert tokens[:9] == ["training", "portrait", *raw.fixed]
        assert set(tokens[9:12]) == set(raw.appearance)
        assert set(tokens[12:15]) == set(raw.tags)
        assert set(tokens[15:17]) == set(raw.environment)
    assert raw == parse_caption(full_caption)  # no mutable group is shuffled in place


def test_dropout_protects_fixed_fields_and_nl_and_honors_total_dropout(full_caption):
    raw = parse_caption(full_caption)
    cfg = CaptionConfig(tag_dropout=1, shuffle=True, prefix="prefix", suffix="suffix")
    out = transform_caption(raw, cfg, random.Random(4))
    assert (
        out == "prefix, portrait, newest, safe, 1girl, person, adult, series, @artist. " + raw.nl + ", suffix"
    )
    assert transform_caption(raw, CaptionConfig(caption_dropout=1), random.Random(5)) is None


def test_json_cache_variants_are_bounded_repeatable_and_cover_training(tmp_path, full_caption):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64), "red").save(images / "p.png")
    save_json(images / "p.json", full_caption)
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(images)}],
                "resolutions": [64],
                "bucket_step": 16,
                "caption": {"shuffle": True, "tag_dropout": 0.4, "caption_dropout": 0.3, "cache_variants": 8},
            },
        }
    )
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    assert isinstance(bundle.train.raw_caption(bundle.train.items[0]), StructuredCaption)
    cached = bundle.train.use_cached_captions()
    assert 1 < len(cached) <= 8 and all(value.endswith(full_caption["ai_output"]["nl"]) for value in cached)
    again = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache").train
    assert again.use_cached_captions() == cached
    seen = set()
    for epoch in range(40):
        bundle.train.set_epoch(epoch)
        sample = bundle.train[0]
        seen.add(sample["caption"])
        assert sample["caption"] in cached or sample["uncond"] and sample["caption"] == ""
    assert "" in seen and len(seen) > 2


def test_wildcards_and_deterministic_json_variants():
    raw = parse_caption({"quality": "{safe|newest}", "tags": ["{red|blue} hair", "hat"], "nl": "A person."})
    cfg = CaptionConfig(wildcard=True, shuffle=True, tag_dropout=0.2, caption_dropout=0.9)
    assert transform_caption_deterministic(raw, cfg) == transform_caption_deterministic(raw, cfg)
    variants = caption_variants_for_cache(raw, cfg, 12, seed=33)
    assert variants == caption_variants_for_cache(raw, cfg, 12, seed=33)
    assert 1 < len(variants) <= 12 and all(
        "{" not in value and value.endswith("A person.") for value in variants
    )


@pytest.mark.parametrize(
    "document",
    [
        [],
        "raw text",
        {"tags": ["good", 12]},
        {"appearance": {}},
        {"ai_output": "text"},
        {"character": {"name": 3}},
        {"character": {"name": "person", "full": False}},
        {"character": {"name": "person", "full": {}}},
        {"character": {"name": "person", "full": ["valid", 3]}},
        {"nl": []},
        {"meta": {"trigger": 4}},
    ],
)
def test_invalid_json_shapes_fail_with_filename(tmp_path, document):
    path = save_json(tmp_path / "bad.json", document)
    with pytest.raises(ValueError, match="Invalid JSON caption bad.json"):
        read_caption(path, "must not silently fall back")


@pytest.mark.parametrize("text", ["{broken", "", '{"tags": [NaN]}'])
def test_malformed_json_never_becomes_raw_training_text(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="bad.json"):
        read_training_caption(path, "fallback")


@pytest.mark.parametrize("kind", ["full", "simple", "legacy", "standard"])
def test_edit_roundtrip_preserves_metadata_without_resurrecting_tags(tmp_path, full_caption, kind):
    source = {
        "full": full_caption,
        "simple": {"quality": "safe", "appearance": ["old"], "tags": ["delete"], "nl": "Prose, here."},
        "legacy": {"tags": ["old", "delete"], "meta": {"trigger": "old", "custom": 123}},
        "standard": {
            "tags": {"quality": "safe", "tags": ["delete"], "nl": "Prose, here."},
            "meta": {"custom": 123},
        },
    }[kind]
    path = save_json(tmp_path / "p.json", source)
    old_nl = read_training_caption(path).nl
    text = "new, confirmed" + (". " + old_nl if old_nl else "")
    serialized = caption_content(path, text)
    assert json.loads(path.read_text()) == source  # staging serialization does not mutate source
    # Pipeline staging files can be extensionless: content has already been serialized.
    staging = tmp_path / "stage"
    staging.write_text(serialized)
    staging.replace(path)
    saved = json.loads(path.read_text())
    assert EDIT_MARKER not in saved
    expected_meta = dict(source.get("meta", {}))
    if "trigger" in expected_meta:
        expected_meta["trigger"] = ""  # Trigger is a real editable tag, not unrelated metadata.
    assert saved.get("meta", {}) == expected_meta
    if kind == "full":
        assert saved["fixed"] == {"quality": "", "series": "", "artist": ""}
        assert saved["ai_output"]["tags"] == ["new", "confirmed"]
        assert saved["ai_output"]["nl"] == source["ai_output"]["nl"]
    elif kind == "standard":
        assert isinstance(saved["tags"], dict) and saved["tags"]["tags"] == ["new", "confirmed"]
    else:
        assert saved["tags"] == ["new", "confirmed"]
    assert read_caption(path) == text
    write_caption(path, read_caption(path))
    assert read_caption(path) == text  # no duplicate prose on repeated save
    write_caption(path, "")
    assert read_caption(path) == old_nl  # deleted tags and legacy trigger never return
    assert not list(tmp_path.glob(".*.tmp"))


def test_json_write_fails_before_replacing_invalid_original(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{broken")
    with pytest.raises(ValueError):
        write_caption(path, "new")
    assert path.read_text() == "{broken" and len(list(tmp_path.iterdir())) == 1


def test_tag_editor_excludes_natural_language_from_bulk_operations(tmp_path, full_caption):
    path = save_json(tmp_path / "p.json", full_caption)
    tags = read_editable_caption(path).split(", ")
    assert caption_description(path) == full_caption["ai_output"]["nl"]
    assert "One person" not in ", ".join(tags)
    write_caption(path, ", ".join([*tags, "new tag"]))
    after = read_caption(path)
    assert after.count(full_caption["ai_output"]["nl"]) == 1
    assert after.endswith(". " + full_caption["ai_output"]["nl"])
    saved = json.loads(path.read_text())
    assert saved["ai_output"]["tags"] == ["smile", "standing", "new tag"]
    assert saved["fixed"] == full_caption["fixed"]
    assert saved["from_path"] == full_caption["from_path"]


def test_empty_json_is_authoritative_not_class_prompt_fallback(tmp_path):
    path = save_json(tmp_path / "empty.json", {"tags": []})
    assert read_caption(path, "class prompt") == ""


@pytest.mark.parametrize("document", [{"tags": []}, {"tags": {"tags": []}}, {"ai_output": {"tags": []}}])
def test_supported_empty_json_remains_valid_for_training_and_cache(tmp_path, document):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "p.png")
    path = save_json(tmp_path / "p.json", document)
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(tmp_path), "class_prompt": "must not replace empty"}],
                "resolutions": [64],
                "bucket_step": 16,
            },
        }
    )
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    assert bundle.train[0]["caption"] == ""
    assert bundle.train.use_cached_captions() == [""]
    assert json.loads(path.read_text()) == document


@pytest.mark.parametrize("document", [{"caption": "Actual prose."}, {"custom_export": {"tag": "kept"}}, {}])
def test_unknown_json_rejected_by_online_training_and_cache_but_still_viewable(tmp_path, document):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "p.png")
    path = save_json(tmp_path / "p.json", {"tags": ["initially valid"]})
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(tmp_path), "class_prompt": "must not hide unknown JSON"}],
                "resolutions": [64],
                "bucket_step": 16,
            },
        }
    )
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    # A sidecar changed after preflight must also fail when training/cache actually reads it.
    save_json(path, document)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="p.json.*unrecognized caption format"):
        bundle.train[0]
    with pytest.raises(ValueError, match="p.json.*unrecognized caption format"):
        bundle.train.use_cached_captions()
    structure = load_caption_structure(path)
    assert structure["format"] == "unknown" and not structure["editable"]
    assert structure["document"] == document
    assert read_caption(path) == ""  # Inspection never pretends metadata is a caption.
    assert path.read_bytes() == before


def test_txt_read_write_and_transform_remain_unchanged(tmp_path):
    path = tmp_path / "p.TXT"
    write_caption(path, "  first, second\n")
    assert path.read_bytes() == b"first, second\n"
    assert read_training_caption(path) == "first, second"
    assert read_caption(path) == "first, second"
    assert caption_description(path) == ""
    path.write_text("")
    assert read_caption(path, "class fallback") == "class fallback"
    assert read_caption(None, "missing") == "missing"
    # Captured legacy TXT behavior: prefix/trigger/keep_tokens and RNG consumption.
    cfg = CaptionConfig(
        trigger_word="ypd", keep_tokens=1, shuffle=True, tag_dropout=0.5, prefix="masterpiece"
    )
    assert transform_caption("a, b, c, d, e, f", cfg, random.Random(0)) == "masterpiece, ypd, a, b, f, c"


def test_auto_selection_prefers_json_case_insensitively_and_explicit_never_falls_back(tmp_path):
    image = tmp_path / "same.png"
    Image.new("RGB", (64, 64), "red").save(image)
    txt = tmp_path / "same.TXT"
    txt.write_text("text")
    structured = save_json(tmp_path / "same.JSON", {"tags": ["structured"]})
    assert caption_for(image) == str(structured)
    assert caption_target(image) == structured
    assert caption_for(image, ".txt") == str(txt)
    assert caption_for(image, ".missing") is None
    assert caption_target(image, ".missing") == image.with_suffix(".missing")
    structured.unlink()
    assert caption_for(image) == str(txt)
    txt.unlink()
    assert caption_target(image) == image.with_suffix(".txt")
    assert caption_for(image) is None


def test_scan_auto_enumerates_each_caption_directory_once(tmp_path, monkeypatch):
    for i in range(15):
        Image.new("RGB", (64, 64), (i, 0, 0)).save(tmp_path / f"{i}.png")
        (tmp_path / f"{i}.TXT").write_text("same")
    from ypuddin.data import index

    original = index._caption_siblings
    calls = []

    def counted(directory):
        calls.append(directory)
        return original(directory)

    monkeypatch.setattr(index, "_caption_siblings", counted)
    records = scan_sources([DatasetSourceConfig(path=str(tmp_path))])
    assert len(records) == 15 and all(record.caption_path for record in records)
    assert calls == [tmp_path]


@pytest.mark.parametrize("payload", ["{broken", '{"caption":"Unsupported prose field"}', "{}"])
def test_bad_json_preflight_has_source_location_and_no_txt_fallback(tmp_path, payload):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "p.png")
    (tmp_path / "p.json").write_text(payload)
    (tmp_path / "p.txt").write_text("good text")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {"sources": [{"path": str(tmp_path)}], "resolutions": [64], "bucket_step": 16},
        }
    )
    with pytest.raises(DataConfigError, match="p.json") as failure:
        prepare_data_layout(cfg, get_family("toy").spec.latent)
    assert failure.value.loc == "dataset.sources.0.caption_ext"
    from ypuddin.train.plan import plan

    result = plan(cfg)
    assert any(
        error["loc"] == "dataset.sources.0.caption_ext" and "p.json" in error["msg"]
        for error in result["errors"]
    )
    cfg.dataset.sources[0].caption_ext = ".txt"
    assert len(prepare_data_layout(cfg, get_family("toy").spec.latent).items) == 1


@pytest.mark.parametrize("payload", ["{broken", '{"caption":"Unsupported prose field"}'])
def test_bad_validation_caption_has_validation_source_location(tmp_path, payload):
    train, val = tmp_path / "train", tmp_path / "val"
    train.mkdir()
    val.mkdir()
    Image.new("RGB", (64, 64), "red").save(train / "p.png")
    Image.new("RGB", (64, 64), "blue").save(val / "p.png")
    (val / "p.json").write_text(payload)
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {"sources": [{"path": str(train)}], "resolutions": [64], "bucket_step": 16},
            "validation": {"enabled": True, "split_ratio": 0, "sources": [{"path": str(val)}]},
        }
    )
    with pytest.raises(DataConfigError) as failure:
        prepare_data_layout(cfg, get_family("toy").spec.latent)
    assert failure.value.loc == "validation.sources.0.caption_ext"


def test_json_fingerprint_survives_file_move_but_changes_after_tag_edit(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    Image.new("RGB", (64, 64), "red").save(a / "p.png")
    save_json(a / "p.json", {"tags": ["first"], "meta": {"source": "test"}})
    source = DatasetSourceConfig(path=str(a))
    before = dataset_fingerprint(scan_sources([source]), [source])
    a.rename(b)
    (b / "p.png").rename(b / "renamed.png")
    (b / "p.json").rename(b / "renamed.json")
    source = source.model_copy(update={"path": str(b)})
    assert dataset_fingerprint(scan_sources([source]), [source]) == before
    write_caption(b / "renamed.json", "updated")
    assert dataset_fingerprint(scan_sources([source]), [source]) != before


def test_txt_fingerprint_matches_old_algorithm_and_json_interpretation_is_versioned(tmp_path):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "p.png")
    (tmp_path / "p.txt").write_text("a, b")
    source = DatasetSourceConfig(path=str(tmp_path), caption_ext=".txt")
    records = scan_sources([source])
    r = records[0]
    old_payload = {
        "version": 2,
        "sources": [source.model_dump(mode="json", exclude={"path"})],
        "settings": {},
        "records": [(r.source_index, r.content_hash, content_hash(r.caption_path), "")],
    }
    old_digest = hashlib.blake2b(json.dumps(old_payload, sort_keys=True).encode(), digest_size=8).hexdigest()
    assert dataset_fingerprint(records, [source]) == old_digest
    json_path = save_json(tmp_path / "p.json", {"tags": ["new"]})
    as_json = replace(r, caption_path=str(json_path))
    # Same bytes with a different interpretation cannot share a resume fingerprint.
    (tmp_path / "p.txt").write_bytes(json_path.read_bytes())
    assert dataset_fingerprint([as_json], [source]) != dataset_fingerprint(records, [source])
