"""Overview uses all indexed files, with strict project/version boundaries."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.server import create_app, routes_work
from ypuddin.server import routes_dataset_overview as overview


def png(index):
    stream = io.BytesIO()
    Image.new("RGB", (64 if index < 18 else 32, 32), (index, 40, 60)).save(stream, "PNG")
    return stream.getvalue()


@pytest.fixture
def dataset_api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Overview", "family": "toy"}).json()
    files = []
    for index in range(20):
        folder = "concept/nested" if index < 18 else "other"
        files.extend(
            [
                ("files", (f"{folder}/{index}.png", png(index), "image/png")),
                ("files", (f"{folder}/{index}.txt", f"Shared, tag{index}".encode(), "text/plain")),
            ]
        )
    response = client.post(f"/api/projects/{project['id']}/datasets/upload", files=files)
    assert response.status_code == 200, response.text
    # Import preserves top-level concepts and can create multiple sources.
    result = response.json()
    row = result["source"]
    if result.get("sources"):
        row = max(result["sources"], key=lambda source: source.get("stats", {}).get("images", 0))
    actual = app.state.ctx.db.fetchone("SELECT * FROM datasets WHERE id=?", (row["id"],))
    yield client, app.state.ctx, dict(actual)
    app.state.dataset_pipeline.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def request(dataset_api, **extra):
    client, _, row = dataset_api
    params = {
        "project_id": row["project_id"],
        **({"version_id": row["version_id"]} if row.get("version_id") else {}),
        **extra,
    }
    return client.get(f"/api/datasets/{row['id']}/overview", params=params)


def test_full_statistics_are_not_truncated_to_the_preview_page(dataset_api):
    response = request(dataset_api, page_size=2)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["images"]["items"]) == 2
    assert data["stats"]["images"] >= 18
    shared = next(tag for tag in data["caption_stats"]["tags"] if tag["tag"] == "Shared")
    assert shared["count"] == data["stats"]["images"]
    assert sum(item["count"] for item in data["stats"]["resolutions"]) == data["stats"]["images"]
    assert "_tokens" not in data["images"]["items"][0]
    searched = request(dataset_api, q="tag17").json()
    assert searched["images"]["total"] == 1
    assert searched["stats"] == data["stats"]
    assert searched["caption_stats"] == data["caption_stats"]


def test_folder_filter_counts_descendants_and_does_not_match_sibling_prefix(dataset_api, monkeypatch):
    source_items = overview._dataset_image_items(dataset_api[1], dataset_api[2])[:3]
    modified = [
        {**item, "rel_path": rel}
        for item, rel in zip(
            source_items, ["concept/nested/a.png", "concept/b.png", "concept-other/c.png"], strict=True
        )
    ]
    monkeypatch.setattr(overview, "_dataset_image_items", lambda c, row: modified)
    data = request(dataset_api, folder="concept").json()
    assert data["stats"]["images"] == 2
    assert {item["path"]: item["count"] for item in data["folders"]} == {
        "concept": 2,
        "concept/nested": 1,
        "concept-other": 1,
    }
    assert all(item["rel_path"].startswith("concept/") for item in data["images"]["items"])
    assert request(dataset_api, folder="concept/nested").json()["stats"]["images"] == 1


def test_windows_relative_paths_are_normalized_for_directory_filtering(dataset_api, monkeypatch):
    item = overview._dataset_image_items(dataset_api[1], dataset_api[2])[0]
    monkeypatch.setattr(
        overview, "_dataset_image_items", lambda c, row: [{**item, "rel_path": "concept\\nested\\a.png"}]
    )
    data = request(dataset_api, folder="concept/nested").json()
    assert data["stats"]["images"] == 1
    assert data["images"]["items"][0]["rel_path"] == "concept/nested/a.png"


@pytest.mark.parametrize("params", [{"project_id": "foreign"}, {"version_id": "foreign"}])
def test_dataset_cannot_be_read_through_another_project_or_version(dataset_api, params):
    response = request(dataset_api, **params)
    assert response.status_code == 404
    assert "images" not in response.json()


def test_unready_index_is_not_a_successful_zero_count(dataset_api):
    _, c, row = dataset_api
    c.db.execute("UPDATE datasets SET index_status='indexing' WHERE id=?", (row["id"],))
    response = request(dataset_api)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "dataset.index_not_ready"


@pytest.mark.parametrize("folder", ["../other", "/absolute", "concept\\nested"])
def test_invalid_folder_is_rejected(dataset_api, folder):
    assert request(dataset_api, folder=folder).status_code == 422


def test_reads_current_caption_contents_not_cached_preview_metadata(dataset_api):
    _, c, row = dataset_api
    item = overview._dataset_image_items(c, row)[0]
    caption = (Path(row["path"]) / item["rel_path"]).with_suffix(".txt")
    caption.write_text("new caption, Changed", encoding="utf-8")
    data = request(dataset_api).json()
    assert {tag["tag"] for tag in data["caption_stats"]["tags"]} >= {"new caption", "Changed"}
