"""JSON caption semantics across real service imports, edits, pipeline undo and versions."""

import io
import json
import time
import zipfile
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
    project = client.post("/api/projects", json={"name": "JSON captions", "family": "toy"}).json()
    yield client, app.state.ctx, project
    app.state.regularization.close()
    app.state.dataset_pipeline.close()
    app.state.environment.close()
    app.state.model_downloads.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def picture(color="red"):
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(stream, format="PNG")
    return stream.getvalue()


def document():
    return {
        "meta": {"trigger": "trigger_token", "source": "original", "extra": {"preserve": True}},
        "tags": {
            "quality": ["best"],
            "appearance": ["blue_hair"],
            "tags": ["smile"],
            "environment": ["garden"],
            "nl": "A test portrait.",
        },
        "unrelated_metadata": {"score": 0.7},
    }


def contents():
    return [
        ("a.png", picture()),
        ("a.txt", b"fallback TXT"),
        ("a.json", json.dumps(document()).encode()),
        ("b.png", picture("blue")),
        ("b.txt", b"only TXT"),
    ]


def ingest(api, tmp_path, mode="upload", entries=None, **fields):
    client, _, project = api
    entries = contents() if entries is None else entries
    endpoint = f"/api/projects/{project['id']}/datasets"
    if mode == "local":
        source = tmp_path / "external"
        source.mkdir(exist_ok=True)
        for name, data in entries:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return client.post(endpoint, json={"path": str(source), **fields})
    if mode == "zip":
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            for name, data in entries:
                output.writestr(name, data)
        entries = [("images.zip", archive.getvalue())]
    return client.post(
        endpoint + "/upload",
        data=fields,
        files=[("files", (name, data, "application/octet-stream")) for name, data in entries],
    )


def images(api, did):
    response = api[0].get(f"/api/datasets/{did}/images")
    assert response.status_code == 200, response.text
    return {item["rel_path"]: item for item in response.json()["items"]}


def caption_url(did, image):
    return f"/api/datasets/{did}/images/{image['hash']}/caption"


def operation(api, action, **options):
    client, c, project = api
    response = client.post(
        f"/api/projects/{project['id']}/versions/{project['active_version_id']}/pipeline/operations",
        json={"action": action, **options},
    )
    assert response.status_code == 202, response.text
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = client.get(f"/api/dataset-pipeline/operations/{response.json()['id']}").json()
        if (
            result["status"] in {"completed", "failed", "cancelled"}
            and not c.resolve_version(project["id"])["busy"]
        ):
            return result
        time.sleep(0.01)
    pytest.fail(f"pipeline did not finish: {result}")


@pytest.mark.parametrize("mode", ["upload", "zip", "local"])
def test_auto_imports_mixed_formats_and_prefers_json(api, tmp_path, mode):
    result = ingest(api, tmp_path, mode)
    assert result.status_code in {200, 201}, result.text
    source = result.json()["source"]
    assert source["caption_ext"] == "auto"
    rows = images(api, source["id"])
    assert rows["a.png"]["caption_format"] == "json" and rows["a.png"]["caption_error"] is None
    assert rows["a.png"]["caption"] == "trigger_token, best, blue_hair, smile, garden. A test portrait."
    assert "fallback TXT" not in rows["a.png"]["caption"]
    assert rows["b.png"]["caption_format"] == "txt" and rows["b.png"]["caption"] == "only TXT"
    found = api[0].get(f"/api/datasets/{source['id']}/images", params={"q": "garden"}).json()
    assert found["total"] == 1 and found["items"][0]["rel_path"] == "a.png"
    config = api[0].get(f"/api/projects/{api[2]['id']}/config").json()
    assert config["dataset"]["sources"][0]["caption_ext"] == "auto"
    if mode == "local":
        assert Path(source["path"]) != tmp_path / "external"
        assert (tmp_path / "external" / "a.json").read_bytes() == dict(contents())["a.json"]


@pytest.mark.parametrize("mode", ["upload", "zip", "local"])
@pytest.mark.parametrize("bad", [b"{broken JSON", b'{"tags":42}'])
def test_invalid_json_import_is_rejected_without_registering_or_leaving_files(api, tmp_path, mode, bad):
    entries = [("a.png", picture()), ("a.txt", b"must not silently fall back"), ("a.json", bad)]
    result = ingest(api, tmp_path, mode, entries)
    assert result.status_code in {400, 422}, result.text
    client, c, project = api
    assert client.get(f"/api/projects/{project['id']}/datasets").json() == []
    assert client.get(f"/api/projects/{project['id']}/config").json()["dataset"]["sources"] == []
    root = c.dataset_dir(project["id"], project["active_version_id"])
    assert not root.exists() or list(root.iterdir()) == []
    if mode == "local":
        assert (tmp_path / "external" / "a.json").read_bytes() == bad


def test_manual_json_edit_preserves_metadata_and_does_not_restore_removed_tags(api, tmp_path):
    source = ingest(api, tmp_path).json()["source"]
    path = Path(source["path"]) / "a.json"
    row = images(api, source["id"])["a.png"]
    response = api[0].put(caption_url(source["id"], row), json={"caption": "new_tag"})
    assert response.status_code == 200, response.text
    after = json.loads(path.read_text())
    assert after["meta"] == {**document()["meta"], "trigger": ""}
    assert after["unrelated_metadata"] == document()["unrelated_metadata"]
    assert after["tags"]["tags"] == ["new_tag"]
    assert after["tags"]["quality"] == [] and after["tags"]["appearance"] == []
    assert "_ypuddin_caption_edit" not in after
    expected = "new_tag. A test portrait."
    assert api[0].get(caption_url(source["id"], row)).json()["caption"] == expected
    assert images(api, source["id"])["a.png"]["caption"] == expected
    assert api[0].post(f"/api/datasets/{source['id']}/rescan").status_code == 200
    assert images(api, source["id"])["a.png"]["caption"] == expected
    assert (path.parent / "a.txt").read_text() == "fallback TXT"


def test_unknown_json_remains_read_only_in_api_but_blocks_training_plan(api, tmp_path, monkeypatch):
    from ypuddin.server import routes_core

    monkeypatch.setattr(routes_core, "gpu_info", lambda: [])
    original = b'{"caption":"Keep this prose","metadata":{"value":17}}\n'
    response = ingest(api, tmp_path, entries=[("a.png", picture()), ("a.json", original)])
    assert response.status_code == 200, response.text
    source = response.json()["source"]
    row = images(api, source["id"])["a.png"]
    structure = row["caption_structure"]
    assert structure["format"] == "unknown" and not structure["editable"]
    assert structure["document"] == json.loads(original)
    assert api[0].get(caption_url(source["id"], row)).json()["caption_structure"] == structure
    config = {
        "model": {"family": "toy"},
        "dataset": {"sources": [{"path": source["path"]}], "resolutions": [64], "bucket_step": 16},
    }
    response = api[0].post("/api/plan", json={"config": config})
    assert response.status_code == 200, response.text
    result = response.json()
    assert not result["ok"]
    assert any(
        error["loc"] == "dataset.sources.0.caption_ext" and "unrecognized caption format" in error["msg"]
        for error in result["errors"]
    )
    assert (Path(source["path"]) / "a.json").read_bytes() == original


@pytest.mark.parametrize("payload,code,reason", [
    (b'{"caption":"Keep this prose","metadata":{"score":17}}', "caption_format_unsupported", "unrecognized caption format"),
    (b'{"tags":42}', "caption_json_invalid", "tags must be a string or an array of strings"),
    (b'{broken', "caption_json_invalid", "Invalid JSON caption a.json"),
])
def test_inspection_keeps_json_filename_specific_reason_and_original_bytes(api, tmp_path, payload, code, reason):
    source = ingest(api, tmp_path).json()["source"]
    path = Path(source["path"]) / "a.json"
    path.write_bytes(payload)
    result = operation(api, "inspect")
    assert result["status"] == "completed", result
    row = next(row for row in result["result"]["inspection"]["images"] if row["rel_path"] == "a.png")
    errors = [issue for issue in row["issues"] if issue["severity"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == code and errors[0]["path"] == "a.json"
    assert reason in errors[0]["message"]
    assert row["caption"] == ""  # The neighboring TXT is not a silent fallback.
    assert path.read_bytes() == payload


def test_inspection_and_training_share_txt_only_model_auto_selection(api, tmp_path, monkeypatch):
    from dataclasses import replace

    from ypuddin.models import get_family
    from ypuddin.server import routes_core

    source = ingest(api, tmp_path).json()["source"]
    family = get_family("toy")
    monkeypatch.setattr(family, "spec", replace(family.spec, caption_formats=("txt",)))
    monkeypatch.setattr(routes_core, "_FAMILY_INFO", {})
    assert api[0].get("/api/families/toy").json()["caption_formats"] == ["txt"]
    path = Path(source["path"]) / "a.json"
    path.write_text('{"tags":42}')
    result = operation(api, "inspect")
    row = next(row for row in result["result"]["inspection"]["images"] if row["rel_path"] == "a.png")
    assert row["caption"] == "fallback TXT"
    assert not any(issue["severity"] == "error" for issue in row["issues"])
    cfg = api[0].get(f"/api/projects/{api[2]['id']}/config").json()
    cfg["dataset"]["sources"][0]["caption_ext"] = ".json"
    assert api[0].put(f"/api/projects/{api[2]['id']}/config", json=cfg).status_code == 200
    result = operation(api, "inspect")
    row = next(row for row in result["result"]["inspection"]["images"] if row["rel_path"] == "a.png")
    error = next(issue for issue in row["issues"] if issue["severity"] == "error")
    assert error["code"] == "caption_model_unsupported" and error["path"] == "a.json"
    assert "does not support JSON" in error["message"]
    assert path.read_text() == '{"tags":42}'


def test_structured_caption_api_preserves_source_and_rejects_stale_revision(api, tmp_path):
    source = ingest(api, tmp_path).json()["source"]
    path = Path(source["path"]) / "a.json"
    row = images(api, source["id"])["a.png"]
    structure = row["caption_structure"]
    assert structure["format"] == "nested" and structure["editable"]
    assert structure["document"] == document()
    assert api[0].get(caption_url(source["id"], row)).json()["caption_structure"] == structure
    response = api[0].put(caption_url(source["id"], row), json={
        "caption_fields": [{"path": ["tags", "appearance"], "value": ["green_hair"]}],
        "caption_revision": structure["revision"],
    })
    assert response.status_code == 200, response.text
    expected = document()
    expected["tags"]["appearance"] = ["green_hair"]
    assert json.loads(path.read_text()) == expected
    assert response.json()["caption_structure"]["document"] == expected
    before = path.read_bytes()
    stale = api[0].put(caption_url(source["id"], row), json={
        "caption_fields": [{"path": ["tags", "appearance"], "value": ["stale"]}],
        "caption_revision": structure["revision"],
    })
    assert stale.status_code == 409 and path.read_bytes() == before


@pytest.mark.parametrize("patch", [
    {"path": ["unrelated_metadata"], "value": "overwrite"},
    {"path": ["tags", "quality"], "value": "wrong array type"},
    {"path": ["tags", "nl"], "value": ["wrong string type"]},
])
def test_structured_api_rejects_unknown_fields_and_type_changes(api, tmp_path, patch):
    source = ingest(api, tmp_path).json()["source"]
    row = images(api, source["id"])["a.png"]
    path = Path(source["path"]) / "a.json"
    before = path.read_bytes()
    response = api[0].put(caption_url(source["id"], row), json={
        "caption_fields": [patch], "caption_revision": row["caption_structure"]["revision"],
    })
    assert response.status_code == 422, response.text
    assert path.read_bytes() == before


def test_description_only_api_edits_original_nl_field_without_flattening(api, tmp_path):
    source = ingest(api, tmp_path).json()["source"]
    row = images(api, source["id"])["a.png"]
    response = api[0].put(caption_url(source["id"], row), json={"description": "New prose."})
    assert response.status_code == 200, response.text
    expected = document()
    expected["tags"]["nl"] = "New prose."
    assert json.loads((Path(source["path"]) / "a.json").read_text()) == expected


def test_json_batch_edit_and_undo_restore_original_bytes(api, tmp_path):
    source = ingest(api, tmp_path).json()["source"]
    path = Path(source["path"]) / "a.json"
    before = path.read_bytes()
    result = operation(
        api,
        "captions",
        images=[{"dataset_id": source["id"], "rel_path": "a.png"}],
        captions={"mode": "replace", "text": "replacement"},
    )
    assert result["status"] == "completed", result
    assert json.loads(path.read_text())["meta"] == {**document()["meta"], "trigger": ""}
    assert images(api, source["id"])["a.png"]["caption"] == "replacement. A test portrait."
    assert any(Path(change["backup"]).read_bytes() == before for change in result["result"]["changes"])
    restored = operation(api, "restore", restore_operation_id=result["id"])
    assert restored["status"] == "completed", restored
    assert path.read_bytes() == before
    assert "blue_hair" in images(api, source["id"])["a.png"]["caption"]


def test_json_batch_append_keeps_prose_once_and_adds_a_real_tag(api, tmp_path):
    source = ingest(api, tmp_path).json()["source"]
    path = Path(source["path"]) / "a.json"
    before = path.read_bytes()
    result = operation(
        api,
        "captions",
        images=[{"dataset_id": source["id"], "rel_path": "a.png"}],
        captions={"mode": "append", "text": "added_tag"},
    )
    assert result["status"] == "completed", result
    text = images(api, source["id"])["a.png"]["caption"]
    assert text == "trigger_token, best, blue_hair, smile, added_tag, garden. A test portrait."
    after = json.loads(path.read_text())
    assert after["tags"]["tags"] == ["smile", "added_tag"]
    assert after["tags"]["environment"] == ["garden"]
    assert after["meta"] == document()["meta"]
    assert operation(api, "restore", restore_operation_id=result["id"])["status"] == "completed"
    assert path.read_bytes() == before


def test_source_omitting_extension_uses_same_auto_default_for_registry_and_training(api, tmp_path):
    source = ingest(api, tmp_path).json()["source"]
    client, c, project = api
    config = client.get(f"/api/projects/{project['id']}/config").json()
    config["dataset"]["sources"][0].pop("caption_ext")
    saved = client.put(f"/api/projects/{project['id']}/config", json=config)
    assert saved.status_code == 200, saved.text
    assert (
        c.db.fetchone("SELECT caption_ext FROM datasets WHERE id=?", (source["id"],))["caption_ext"] == "auto"
    )
    assert images(api, source["id"])["a.png"]["caption_format"] == "json"


def test_changing_explicit_txt_to_auto_reindexes_registry_and_all_editors(api, tmp_path):
    source = ingest(api, tmp_path, caption_ext=".txt").json()["source"]
    assert images(api, source["id"])["a.png"]["caption"] == "fallback TXT"
    client, c, project = api
    config = client.get(f"/api/projects/{project['id']}/config").json()
    config["dataset"]["sources"][0]["caption_ext"] = "auto"
    saved = client.put(f"/api/projects/{project['id']}/config", json=config)
    assert saved.status_code == 200, saved.text
    assert c.db.fetchone("SELECT caption_ext,index_status FROM datasets WHERE id=?", (source["id"],)) == {
        "caption_ext": "auto",
        "index_status": "ready",
    }
    row = images(api, source["id"])["a.png"]
    assert row["caption_format"] == "json" and "blue_hair" in row["caption"]
    assert client.get(caption_url(source["id"], row)).json()["caption"] == row["caption"]
    inspected = operation(api, "inspect")
    assert inspected["status"] == "completed", inspected
    scanned = next(
        item for item in inspected["result"]["inspection"]["images"] if item["rel_path"] == "a.png"
    )
    assert scanned["caption"] == row["caption"]


def test_version_copy_keeps_independent_json_files_and_edit_scope(api, tmp_path):
    source = ingest(api, tmp_path).json()["source"]
    old_root = Path(source["path"])
    old_json = (old_root / "a.json").read_bytes()
    client, _, project = api
    response = client.post(
        f"/api/projects/{project['id']}/versions",
        json={"name": "JSON fork", "source_version_id": project["active_version_id"], "data_mode": "copy"},
    )
    assert response.status_code == 202, response.text
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        version = client.get(f"/api/projects/{project['id']}/versions/{response.json()['id']}").json()
        if version["status"] != "copying":
            break
        time.sleep(0.01)
    assert version["status"] == "ready", version
    datasets = client.get(
        f"/api/projects/{project['id']}/datasets", params={"version_id": version["id"]}
    ).json()
    copied = datasets[0]["source"]
    new_root = Path(copied["path"])
    assert copied["id"] != source["id"] and new_root != old_root
    assert (new_root / "a.json").read_bytes() == old_json
    image = images(api, copied["id"])["a.png"]
    result = client.put(caption_url(copied["id"], image), json={"caption": "fork_only"})
    assert result.status_code == 200, result.text
    assert (old_root / "a.json").read_bytes() == old_json
    assert images(api, copied["id"])["a.png"]["caption"].startswith("fork_only")
    assert images(api, source["id"])["a.png"]["caption"].startswith("trigger_token")


def test_legacy_batch_rejects_later_bad_json_before_changing_earlier_file(api, tmp_path):
    entries = [
        ("a.png", picture()),
        ("a.json", json.dumps(document()).encode()),
        ("b.png", picture("blue")),
        ("b.json", json.dumps(document()).encode()),
    ]
    source = ingest(api, tmp_path, entries=entries).json()["source"]
    rows = list(images(api, source["id"]).values())
    root = Path(source["path"])
    first = (root / rows[0]["rel_path"]).with_suffix(".json")
    second = (root / rows[1]["rel_path"]).with_suffix(".json")
    original = first.read_bytes()
    second.write_text("{externally damaged")
    response = api[0].post(
        f"/api/datasets/{source['id']}/tags/batch",
        json={"hashes": [row["hash"] for row in rows], "add": ["must_not_partially_apply"]},
    )
    assert first.read_bytes() == original
    assert second.read_text() == "{externally damaged"
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "dataset.caption_invalid"


def test_local_import_preflight_uses_existing_explicit_source_extension(api, tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    (root / "a.png").write_bytes(picture())
    (root / "a.txt").write_text("valid explicit TXT")
    (root / "a.json").write_text("{unused malformed JSON")
    client, _, project = api
    config = client.get(f"/api/projects/{project['id']}/config").json()
    config["dataset"]["sources"] = [{"path": str(root), "caption_ext": ".txt"}]
    assert client.put(f"/api/projects/{project['id']}/config", json=config).status_code == 200
    response = client.post(f"/api/projects/{project['id']}/datasets", json={"path": str(root)})
    assert response.status_code == 201, response.text
    source = response.json()["source"]
    assert source["caption_ext"] == ".txt"
    assert images(api, source["id"])["a.png"]["caption"] == "valid explicit TXT"


@pytest.mark.parametrize("missing_first,fail_at", [(False, 2), (True, 2), (True, 3)])
def test_legacy_batch_io_failure_rolls_back_existing_new_files_and_index(
    api, tmp_path, monkeypatch, missing_first, fail_at
):
    import os

    entries = [("a.png", picture()), ("b.png", picture("blue")), ("b.json", json.dumps(document()).encode())]
    if not missing_first:
        entries.append(("a.json", json.dumps(document()).encode()))
    source = ingest(api, tmp_path, entries=entries).json()["source"]
    root = Path(source["path"])
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    index = routes_work._records_path(api[1], source["id"])
    index_before = index.read_bytes()
    rows = list(images(api, source["id"]).values())
    replace, count = os.replace, 0

    def fail_one(source, target):
        nonlocal count
        count += 1
        if count == fail_at:
            raise OSError("fixture storage full")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", fail_one)
    response = api[0].post(
        f"/api/datasets/{source['id']}/tags/batch",
        json={"hashes": [row["hash"] for row in rows], "add": ["new_tag"]},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "dataset.caption_io"
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert index.read_bytes() == index_before
    monkeypatch.setattr(os, "replace", replace)
    response = api[0].post(
        f"/api/datasets/{source['id']}/tags/batch",
        json={"hashes": [row["hash"] for row in rows], "add": ["new_tag"]},
    )
    assert response.status_code == 200 and response.json()["changed"] == 2
    assert all(row["caption"].count("new_tag") == 1 for row in images(api, source["id"]).values())


def test_bulk_caption_lookups_enumerate_once_per_directory_and_refresh_next_operation(
    api, tmp_path, monkeypatch
):
    from collections import Counter

    from ypuddin.data import index

    calls = Counter()
    siblings = index._caption_siblings

    def count(directory):
        calls[Path(directory)] += 1
        return siblings(directory)

    monkeypatch.setattr(index, "_caption_siblings", count)
    entries = []
    for i in range(8):
        entries.extend([(f"{i}.png", picture((i * 20, 0, 0))), (f"{i}.txt", b"caption")])
    response = ingest(api, tmp_path, "local", entries)
    assert response.status_code == 201, response.text
    source = response.json()["source"]
    root = Path(source["path"])
    assert calls and max(calls.values()) == 1  # Staging preflight and final index each scan once.
    calls.clear()
    result = operation(api, "inspect")
    assert result["status"] == "completed", result
    assert calls[root] == 1
    # A new operation sees a newly added higher-priority JSON; no cache survives the request.
    (root / "0.json").write_text(json.dumps(document()))
    calls.clear()
    result = operation(
        api,
        "captions",
        images=[{"dataset_id": source["id"], "rel_path": f"{i}.png"} for i in range(8)],
        captions={"mode": "append", "text": "added"},
    )
    assert result["status"] == "completed", result
    # Selection, published-file reindex, and refreshed quality report each scan once,
    # independently of the eight selected images; no stale listing crosses publication.
    assert calls[root] == 3
    assert json.loads((root / "0.json").read_text())["tags"]["tags"][-1] == "added"
    assert (root / "0.txt").read_text() == "caption"
