"""Dataset-wide tag statistics, live sidecars, precise filters and scoped caption edits."""

import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.server import create_app, routes_work


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    monkeypatch.setattr(routes_work, "gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app, raise_server_exceptions=False)
    project = client.post("/api/projects", json={"name": "Caption statistics", "family": "toy"}).json()
    yield client, app.state.ctx, project
    app.state.regularization.close()
    app.state.dataset_pipeline.close()
    app.state.environment.close()
    app.state.model_downloads.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def png(number=0):
    output = io.BytesIO()
    Image.new("RGB", (32, 32), (number * 20, 50, 100)).save(output, "PNG")
    return output.getvalue()


def upload(api, entries, **fields):
    client, _, project = api
    response = client.post(
        f"/api/projects/{project['id']}/datasets/upload",
        data=fields,
        files=[("files", (name, value, "application/octet-stream")) for name, value in entries],
    )
    assert response.status_code == 200, response.text
    return response.json()["source"]


def stats(api, source):
    response = api[0].get(f"/api/datasets/{source['id']}/caption-stats")
    assert response.status_code == 200, response.text
    return response.json()


def images(api, source, **params):
    response = api[0].get(f"/api/datasets/{source['id']}/images", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def save(api, source, image, caption, **extra):
    return api[0].put(
        f"/api/datasets/{source['id']}/images/{image['hash']}/caption",
        params={"rel_path": image["rel_path"]},
        json={"caption": caption, **extra},
    )


def full_json():
    return {
        "meta": {"trigger": "trigger", "source": "not-a-tag", "other": {"secret": "metadata"}},
        "tags": {
            "quality": ["best"],
            "appearance": ["RED"],
            "tags": ["red", "smile", "smile"],
            "environment": ["garden"],
            "nl": "Natural language, never a tag.",
        },
        "unrelated_metadata": {"model": "not-a-model-tag"},
    }


@pytest.fixture
def mixed(api):
    entries = [(f"concept/{name}.png", png(index)) for index, name in enumerate("abcdefg")]
    entries += [
        ("concept/a.txt", b"red, red, RED, red hair"),
        ("concept/b.txt", b"unused fallback TXT"),
        ("concept/b.json", json.dumps(full_json()).encode()),
        ("concept/d.txt", b"  \n"),
        ("concept/e.json", b'{"tags":["before corruption"]}'),
        ("concept/f.json", b'{"nl":"Only natural language, no tags."}'),
        ("concept/g.json", b"{}"),
    ]
    source = upload(api, entries, class_prompt="training fallback must not count")
    (Path(source["path"]) / "e.json").write_text("{broken")
    return source


def test_statistics_cover_all_images_and_keep_structured_prose_and_metadata_out(api, mixed):
    result = stats(api, mixed)
    assert {key: result[key] for key in ("images", "captioned", "missing", "invalid")} == {
        "images": 7,
        "captioned": 3,
        "missing": 3,
        "invalid": 1,
    }
    assert result["formats"] == {"txt": 2, "json": 4}
    assert result["unique_tags"] == 6
    tags = {item["tag"].casefold(): item["count"] for item in result["tags"]}
    assert tags == {"red": 2, "red hair": 1, "best": 1, "trigger": 1, "smile": 1, "garden": 1}
    assert result["tags"][0] == {"tag": "red", "count": 2}
    one_page = images(api, mixed, page_size=1)
    assert len(one_page["items"]) == 1 and one_page["total"] == 7
    assert stats(api, mixed) == result


def test_exact_tag_status_search_filters_intersect_before_pagination(api, mixed):
    result = images(api, mixed, tag=" RED ", page_size=1, page=2)
    assert result["total"] == 2 and len(result["items"]) == 1
    assert result["items"][0]["rel_path"] == "b.png"
    assert images(api, mixed, tag="re")["total"] == 0
    assert images(api, mixed, tag="red hair")["total"] == 1
    assert images(api, mixed, tag="red", q="a.png")["total"] == 1
    assert images(api, mixed, tag="red", caption_status="missing")["total"] == 0
    for status, count in [("captioned", 3), ("missing", 3), ("invalid", 1)]:
        result = images(api, mixed, caption_status=status)
        assert result["total"] == count
        assert all(item["caption_status"] == status for item in result["items"])
    assert (
        api[0].get(f"/api/datasets/{mixed['id']}/images", params={"caption_status": "unknown"}).status_code
        == 422
    )


def test_caption_saves_update_statistics_and_json_description_without_metadata_loss(api, mixed):
    records = {image["rel_path"]: image for image in images(api, mixed)["items"]}
    assert save(api, mixed, records["a.png"], "blue, blue").status_code == 200
    before = full_json()
    assert (
        save(api, mixed, records["b.png"], "new, NEW", description="Edited prose, not a tag.").status_code
        == 200
    )
    after = json.loads((Path(mixed["path"]) / "b.json").read_text())
    assert after["meta"] == {**before["meta"], "trigger": ""}
    assert after["unrelated_metadata"] == before["unrelated_metadata"]
    assert after["tags"]["nl"] == "Edited prose, not a tag." and after["tags"]["tags"] == ["new"]
    assert stats(api, mixed)["tags"] == [{"tag": "blue", "count": 1}, {"tag": "new", "count": 1}]
    refreshed = {image["rel_path"]: image for image in images(api, mixed)["items"]}
    assert refreshed["b.png"]["caption_description"] == "Edited prose, not a tag."
    assert save(api, mixed, refreshed["b.png"], "new", description="").status_code == 200
    assert images(api, mixed, tag="new")["items"][0]["caption_description"] == ""
    txt = Path(mixed["path"]) / "a.txt"
    before_txt = txt.read_bytes()
    assert save(api, mixed, records["a.png"], "bad", description="not supported on TXT").status_code == 422
    assert txt.read_bytes() == before_txt
    bad = Path(mixed["path"]) / "e.json"
    before_bad = bad.read_bytes()
    assert save(api, mixed, records["e.png"], "replacement").status_code == 422
    assert bad.read_bytes() == before_bad


def test_live_added_removed_auto_sidecars_and_saves_do_not_depend_on_old_index(api):
    source = upload(api, [("set/a.png", png()), ("set/a.txt", b"txt-tag")])
    root = Path(source["path"])
    (root / "a.json").write_text('{"tags":["live-json"],"nl":"Keep prose"}')
    current = images(api, source)["items"][0]
    assert current["caption_format"] == "json" and stats(api, source)["formats"] == {"json": 1}
    assert save(api, source, current, "json-edited").status_code == 200
    assert (root / "a.txt").read_text() == "txt-tag"
    assert json.loads((root / "a.json").read_text())["nl"] == "Keep prose"
    (root / "a.json").unlink()
    assert images(api, source)["items"][0]["caption_format"] == "txt"
    (root / "a.txt").unlink()
    assert stats(api, source)["missing"] == 1
    assert stats(api, source)["formats"] == {} and stats(api, source)["tags"] == []


def test_same_hash_in_different_paths_saves_only_selected_image(api):
    source = upload(api, [("set/one/a.png", png()), ("set/two/a.png", png())])
    first, second = images(api, source)["items"]
    assert first["hash"] == second["hash"]
    assert save(api, source, second, "second-only").status_code == 200
    root = Path(source["path"])
    assert (root / "two/a.txt").read_text().strip() == "second-only"
    assert not (root / "one/a.txt").exists()
    records = routes_work._records(api[1], source["id"])
    assert records[0]["caption_path"] is None
    assert records[1]["caption_path"] == str(root / "two/a.txt")
    assert images(api, source, caption_status="missing")["items"][0]["rel_path"] == "one/a.png"
    url = f"/api/datasets/{source['id']}/images/{second['hash']}/caption"
    assert api[0].get(url, params={"rel_path": "two/a.png"}).json()["caption"] == "second-only"
    assert (
        api[0].put(url, params={"rel_path": "../../wrong/a.png"}, json={"caption": "bad"}).status_code == 404
    )


def test_missing_images_empty_index_unknown_source_and_external_records_are_isolated(api, tmp_path):
    source = upload(api, [("set/a.png", png()), ("set/a.txt", b"local")])
    root = Path(source["path"])
    foreign = tmp_path / "foreign.png"
    foreign.write_bytes(png(2))
    foreign.with_suffix(".txt").write_text("must-not-leak")
    records = routes_work._records(api[1], source["id"])
    records.append({**records[0], "path": str(foreign), "caption_path": str(foreign.with_suffix(".txt"))})
    routes_work._records_path(api[1], source["id"]).write_text(json.dumps(records))
    assert stats(api, source)["images"] == 1
    assert images(api, source, tag="must-not-leak")["total"] == 0
    (root / "a.png").unlink()
    assert stats(api, source) == {
        "images": 0,
        "captioned": 0,
        "missing": 0,
        "invalid": 0,
        "formats": {},
        "unique_tags": 0,
        "tags": [],
    }
    routes_work._records_path(api[1], source["id"]).unlink()
    assert stats(api, source)["images"] == 0
    assert api[0].get("/api/datasets/no_such_dataset/caption-stats").status_code == 404


def test_caption_symlink_cannot_read_or_write_another_source(api, tmp_path):
    source = upload(api, [("set/a.png", png())])
    target = tmp_path / "other-version-caption.txt"
    target.write_text("private-tag")
    (Path(source["path"]) / "a.txt").symlink_to(target)
    result = stats(api, source)
    assert result["invalid"] == 1 and result["tags"] == []
    image = images(api, source)["items"][0]
    assert "private-tag" not in image["caption"]
    assert save(api, source, image, "overwrite").status_code == 400
    assert target.read_text() == "private-tag"


def test_version_and_regularization_sources_remain_separate_after_edit(api):
    client, c, project = api
    source = upload(api, [("concept/a.png", png()), ("concept/a.txt", b"old-version")])
    reg = upload(api, [("concept/a.png", png()), ("concept/a.txt", b"regularization")], is_reg="true")
    response = client.post(
        f"/api/projects/{project['id']}/versions", json={"name": "Copy", "data_mode": "copy"}
    )
    assert response.status_code == 202, response.text
    vid = response.json()["id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        version = client.get(f"/api/projects/{project['id']}/versions/{vid}").json()
        if version["status"] != "copying":
            break
        time.sleep(0.01)
    assert version["status"] == "ready", version
    copied = client.get(f"/api/projects/{project['id']}/datasets", params={"version_id": vid}).json()
    training = next(row["source"] for row in copied if not row["source"]["is_reg"])
    image = images(api, training)["items"][0]
    assert save(api, training, image, "new-version").status_code == 200
    assert stats(api, training)["tags"] == [{"tag": "new-version", "count": 1}]
    assert stats(api, source)["tags"] == [{"tag": "old-version", "count": 1}]
    assert stats(api, reg)["tags"] == [{"tag": "regularization", "count": 1}]
