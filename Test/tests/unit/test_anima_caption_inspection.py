"""Anima advice is explicitly opt-in, typed and non-destructive."""

import json
import random

import pytest

from ypuddin.config import CaptionConfig
from ypuddin.data.anima_caption_inspection import inspect_anima_caption
from ypuddin.data.caption_json import load_caption_structure
from ypuddin.data.captions import read_caption, transform_caption


def inspect_json(tmp_path, document, config=None):
    path = tmp_path / "caption.json"
    payload = (json.dumps(document, ensure_ascii=False, indent=4) + "\n").encode()
    path.write_bytes(payload)
    before = path.stat()
    result = inspect_anima_caption(path, read_caption(path), config or {})
    after = path.stat()
    assert path.read_bytes() == payload
    assert after.st_mtime_ns == before.st_mtime_ns
    return path, result


@pytest.mark.parametrize("kind", ["full", "nested", "simple"])
def test_preview_preserves_owned_fields_types_prose_scores_triggers_and_metadata(tmp_path, kind):
    values = {
        "artist": "Some_Artist, @Other_Artist", "tags": ["Blue_Hair", "red, Blue_Dress", "MY_Trigger"],
        "quality": ["score_7", "Best_Quality"], "nl": "A Girl, wearing Blue_Dress. Do NOT change.",
    }
    if kind == "full":
        document = {"fixed": {key: values[key] for key in ("artist", "quality")},
                    "ai_output": {key: values[key] for key in ("tags", "nl")},
                    "from_path": {"extra_tags": ["Black_Boots"]}}
    else:
        document = {"tags": values} if kind == "nested" else values.copy()
    document["meta"] = {"trigger": "MY_Trigger", "future": {"unrecognized": "Do_NOT_change"}}
    path, result = inspect_json(tmp_path, document)
    suggestions = {field["role"]: field for field in result["suggestions"] if field["path"][0] != "from_path"}
    assert suggestions["artist"]["after"] == "@some artist, @other artist"
    assert suggestions["quality"]["after"] == ["score_7", "best quality"]
    assert suggestions["tags"]["after"] == ["blue hair", "red, blue dress", "MY_Trigger"]
    assert suggestions["artist"]["path"] == (
        ["fixed", "artist"] if kind == "full" else ["tags", "artist"] if kind == "nested" else ["artist"]
    )
    assert not any(field["role"] in {"trigger", "nl"} for field in result["suggestions"])
    assert load_caption_structure(path)["document"] == document
    assert {issue["code"] for issue in result["issues"]} == {
        "anima_artist_prefix", "anima_tag_spacing", "anima_tag_case",
    }
    assert all(issue["severity"] == "warning" for issue in result["issues"])


@pytest.mark.parametrize("document", [
    {"metadata": {"artist": "Some_Artist"}},
    {"_ypuddin_caption_edit": 1, "fixed": {"artist": "Some_Artist"}, "tags": ["Blue_Hair"]},
])
def test_unknown_and_legacy_documents_have_no_speculative_repair(tmp_path, document):
    _, report = inspect_json(tmp_path, document)
    assert report["suggestions"] == [] and report["issues"] == []


def test_custom_trigger_dynamic_tokens_and_score_tokens_are_never_changed(tmp_path):
    _, report = inspect_json(tmp_path, {"tags": ["Custom_Trigger", "score_7", "SCORE_9", "{Blue_Hair|Red_Hair}"]},
                             {"trigger_word": "Custom_Trigger"})
    assert report["suggestions"] == [] and report["issues"] == []


def test_text_warns_about_sentence_fragment_shuffle_but_does_not_repair_or_block_it(tmp_path):
    raw = "1girl, blue hair. A girl sits beside a window, wearing a red dress."
    path = tmp_path / "caption.txt"
    path.write_text(raw)
    report = inspect_anima_caption(path, raw, {"shuffle": True})
    assert report["suggestions"] == []
    assert [issue["code"] for issue in report["issues"]] == ["anima_text_shuffle"]
    assert "may" in report["issues"][0]["message"]
    # The existing generic transform is untouched; this is the risk being disclosed.
    transformed = transform_caption(raw, CaptionConfig(shuffle=True), random.Random(1))
    assert transformed != raw
    assert path.read_text() == raw
    assert not inspect_anima_caption(path, raw, {})["issues"]
    assert not inspect_anima_caption(path, raw, {"shuffle": True, "keep_tokens": 2})["issues"]
    assert not inspect_anima_caption(path, raw, {"shuffle": True, "separator": "|"})["issues"]


def test_ordinary_tag_shuffle_and_structured_nl_do_not_get_sentence_warning(tmp_path):
    path = tmp_path / "caption.txt"
    assert not inspect_anima_caption(path, "1girl, blue_hair, :) looking_at_viewer", {"shuffle": True})["issues"]
    _, report = inspect_json(tmp_path, {"tags": ["1girl", "blue hair"], "nl": "A girl sits near a window, looking outside."}, {"shuffle": True})
    assert not report["issues"]
