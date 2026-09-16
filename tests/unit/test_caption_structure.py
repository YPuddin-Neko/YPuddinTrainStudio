"""Editing JSON preserves its data schema and training-time classifications."""

import copy
import json
import random

import pytest

from ypuddin.config import CaptionConfig
from ypuddin.data.caption_json import CaptionConflictError, load_caption_structure
from ypuddin.data.captions import (
    read_editable_caption,
    read_training_caption,
    transform_caption,
    write_caption,
)


@pytest.fixture
def full():
    return {
        "fixed": {"quality": "masterpiece", "artist": "@artist", "custom": {"keep": 1}},
        "character": {"name": "alice", "variant": "adult", "custom": [1, 2]},
        "from_path": {"appearance": ["coat"], "extra_tags": ["walking"]},
        "ai_output": {"count": "1girl", "appearance": ["red hair"], "tags": ["smile"],
                      "environment": ["garden"], "nl": "A person smiles.", "confidence": 0.7},
        "meta": {"trigger": "trigger", "generator": {"unchanged": True}},
        "unrelated": {"nested": [3, "retain"]},
    }


def put_json(tmp_path, data):
    path = tmp_path / "image.json"
    path.write_text(json.dumps(data, ensure_ascii=False))
    return path


@pytest.mark.parametrize("kind", ["full", "nested", "simple"])
@pytest.mark.parametrize(
    "full_name,active_field",
    [
        ("  alice (adult)  ", "full"),
        (["  alice (adult)  "], "full"),
        (" \t\n ", "name"),
        ([" ", "\t"], "name"),
    ],
)
def test_character_full_normalization_matches_editor_and_preserves_metadata(
    tmp_path,
    full,
    kind,
    full_name,
    active_field,
):
    full["character"]["full"] = full_name
    source = (
        full
        if kind == "full"
        else {
            "character": full["character"],
            "nl": "Prose.",
            "custom": {"keep": [1, 2]},
        }
    )
    if kind == "nested":
        source = {"tags": source, "meta": {"generator": "keep"}}
    path = put_json(tmp_path, source)
    original = path.read_bytes()
    structure = load_caption_structure(path)
    caption = read_training_caption(path)
    character_fields = [field for field in structure["fields"] if field["role"].startswith("character_")]
    prefix = ["tags", "character"] if kind == "nested" else ["character"]
    if active_field == "full":
        assert [field["path"] for field in character_fields] == [[*prefix, "full"]]
        assert "alice (adult)" in caption.fixed
        assert "alice" not in caption.fixed and "adult" not in caption.fixed
    else:
        assert [field["path"] for field in character_fields] == [
            [*prefix, "name"],
            [*prefix, "variant"],
        ]
        assert "alice" in caption.fixed and "adult" in caption.fixed
        assert "alice (adult)" not in caption.fixed
    assert path.read_bytes() == original  # Parsing/inspection never normalizes the stored document.

    replacement = ["bob (adult)"] if isinstance(full_name, list) and active_field == "full" else "bob"
    write_caption(
        path, fields=[{"path": [*prefix, active_field], "value": replacement}], revision=structure["revision"]
    )
    expected = copy.deepcopy(source)
    character = expected["tags"]["character"] if kind == "nested" else expected["character"]
    character[active_field] = replacement
    assert json.loads(path.read_text()) == expected
    updated = read_training_caption(path)
    assert ("bob (adult)" if isinstance(replacement, list) else "bob") in updated.fixed
    if active_field == "name":
        assert "adult" in updated.fixed
        assert character["full"] == full_name  # Retain the inactive field verbatim.


@pytest.mark.parametrize("with_flat_caption", [False, True])
def test_description_only_change_preserves_all_tag_ownership_and_dropout(tmp_path, full, with_flat_caption):
    path = put_json(tmp_path, full)
    before = read_training_caption(path)
    write_caption(path, read_editable_caption(path) if with_flat_caption else None, description="New prose.")
    saved = json.loads(path.read_text())
    expected = copy.deepcopy(full)
    expected["ai_output"]["nl"] = "New prose."
    assert saved == expected
    after = read_training_caption(path)
    assert (after.fixed, after.appearance, after.tags, after.environment, after.trigger) == (
        before.fixed, before.appearance, before.tags, before.environment, before.trigger,
    )
    assert transform_caption(after, CaptionConfig(tag_dropout=1), random.Random(1)) == (
        "trigger, masterpiece, 1girl, alice, adult, @artist. New prose."
    )


@pytest.mark.parametrize("kind", ["full", "nested", "simple", "flat"])
@pytest.mark.parametrize("as_string", [False, True])
def test_structured_field_write_keeps_format_types_and_unknown_values(tmp_path, full, kind, as_string):
    tags = "smile, standing" if as_string else ["smile", "standing"]
    source = {
        "full": full,
        "nested": {"tags": {"quality": ["best"], "tags": [], "nl": "Prose.", "custom": 3}, "extra": 9},
        "simple": {"quality": "best", "tags": [], "nl": "Prose.", "extra": 9},
        "flat": {"tags": [], "nl": "Prose.", "extra": 9},
    }[kind]
    field_path = {"full": ["ai_output", "tags"], "nested": ["tags", "tags"]}.get(kind, ["tags"])
    parent = source
    for key in field_path[:-1]:
        parent = parent[key]
    parent[field_path[-1]] = tags
    path = put_json(tmp_path, source)
    structure = load_caption_structure(path)
    assert structure["format"] == kind and structure["document"] == source
    replacement = "new, retained" if as_string else ["tag, with comma", "retained"]
    write_caption(path, fields=[{"path": field_path, "value": replacement}], revision=structure["revision"])
    expected = copy.deepcopy(source)
    parent = expected
    for key in field_path[:-1]:
        parent = parent[key]
    parent[field_path[-1]] = replacement
    assert json.loads(path.read_text()) == expected
    assert "_ypuddin_caption_edit" not in expected


def test_flat_add_remove_changes_real_owners_and_keeps_unedited_values(tmp_path, full):
    path = put_json(tmp_path, full)
    before = read_editable_caption(path)
    write_caption(path, before.replace("red hair, ", "").replace("walking, ", "") + ", new")
    saved = json.loads(path.read_text())
    assert saved["fixed"] == full["fixed"] and saved["character"] == full["character"]
    assert saved["ai_output"]["appearance"] == []
    assert saved["from_path"] == {"appearance": ["coat"], "extra_tags": []}
    assert saved["ai_output"]["tags"] == ["smile", "new"]
    assert saved["ai_output"]["confidence"] == 0.7 and saved["meta"] == full["meta"]


def test_unchanged_flat_view_does_not_split_array_tokens_containing_commas(tmp_path, full):
    full["ai_output"]["appearance"] = ["coat, with patterns"]
    path = put_json(tmp_path, full)
    write_caption(path, read_editable_caption(path), description="New prose.")
    expected = copy.deepcopy(full)
    expected["ai_output"]["nl"] = "New prose."
    assert json.loads(path.read_text()) == expected


def test_flat_case_edit_updates_original_owner(tmp_path, full):
    path = put_json(tmp_path, full)
    write_caption(path, read_editable_caption(path).replace("red hair", "Red Hair"))
    expected = copy.deepcopy(full)
    expected["ai_output"]["appearance"] = ["Red Hair"]
    assert json.loads(path.read_text()) == expected


def test_ambiguous_flat_edit_cannot_destroy_comma_containing_array_tokens(tmp_path, full):
    full["ai_output"]["appearance"] = ["coat, with patterns"]
    path = put_json(tmp_path, full)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="edit their fields"):
        write_caption(path, read_editable_caption(path) + ", added")
    assert path.read_bytes() == original


@pytest.mark.parametrize("patch", [
    {"path": ["unrelated"], "value": "destroy"},
    {"path": ["fixed", "custom"], "value": "destroy"},
    {"path": ["fixed", "quality"], "value": ["wrong type"]},
    {"path": ["ai_output", "tags"], "value": "wrong type"},
])
def test_illegal_structured_edits_preserve_original_bytes(tmp_path, full, patch):
    path = put_json(tmp_path, full)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_caption(path, fields=[patch], revision=load_caption_structure(path)["revision"])
    assert path.read_bytes() == original and not list(tmp_path.glob(".*.tmp"))


def test_stale_revision_does_not_overwrite_external_changes(tmp_path, full):
    path = put_json(tmp_path, full)
    structure = load_caption_structure(path)
    full["ai_output"]["tags"] = ["external edit"]
    path.write_text(json.dumps(full))
    original = path.read_bytes()
    with pytest.raises(CaptionConflictError):
        write_caption(path, fields=[{"path": ["ai_output", "tags"], "value": ["stale"]}],
                      revision=structure["revision"])
    assert path.read_bytes() == original


def test_existing_override_is_labeled_and_never_revives_historical_fields(tmp_path, full):
    source = full | {"tags": ["live tag"], "_ypuddin_caption_edit": 1}
    path = put_json(tmp_path, source)
    structure = load_caption_structure(path)
    assert structure["format"] == "legacy_override" and structure["legacy_override"]
    assert [field["path"] for field in structure["fields"]] == [["tags"], ["nl"]]
    write_caption(path, fields=[{"path": ["tags"], "value": []}, {"path": ["nl"], "value": "Live prose."}],
                  revision=structure["revision"])
    saved = json.loads(path.read_text())
    assert saved == source | {"tags": [], "nl": "Live prose."}
    assert read_training_caption(path).text() == "Live prose."


def test_unknown_json_is_read_only_and_kept_byte_for_byte(tmp_path):
    path = put_json(tmp_path, {"custom_export": {"tag": "do not guess"}})
    before = path.read_bytes()
    structure = load_caption_structure(path)
    assert structure["format"] == "unknown" and not structure["editable"] and not structure["fields"]
    with pytest.raises(ValueError, match="read only"):
        write_caption(path, "replacement")
    assert path.read_bytes() == before
