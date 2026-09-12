"""Family changes rebuild recipes while snapshots preserve independent image/caption data."""

import json
import time
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.config import TrainConfig
from ypuddin.models import get_family
from ypuddin.server import create_app, routes_work
from ypuddin.server.errors import ApiError
from ypuddin.server.family_config import change_config_family, initial_family_config, version_family
from ypuddin.server.versions import version_row


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    monkeypatch.setattr(routes_work, "gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    response = client.post("/api/projects", json={"name": "Family tests"})
    assert response.status_code == 201, response.text
    yield client, app.state.ctx, response.json()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


def wait_ready(c, pid, vid):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = c.resolve_version(pid, vid)
        if row["status"] != "copying":
            assert row["status"] == "ready", row
            return row
        time.sleep(0.02)
    pytest.fail("version snapshot did not finish")


@pytest.mark.parametrize("family", ["anima", "krea2", "toy"])
def test_initial_recipe_uses_family_contract_without_cuda_assumptions(api, family):
    _, c, _ = api
    recipe = initial_family_config(c, family)
    spec = get_family(family).spec
    TrainConfig.model_validate(recipe)
    assert recipe["model"]["family"] == family
    assert recipe["adapter"]["preset"] == get_family(family).default_preset()
    assert recipe["sampling"]["steps"] == spec.sampling.steps
    assert recipe["sampling"]["cfg"] == spec.sampling.cfg
    assert recipe["sampling"]["shift"] == spec.sampling.shift
    assert recipe["memory"]["base_precision"] == "auto"
    assert recipe["memory"]["blocks_to_swap"] == 0
    defaults = TrainConfig().to_dict()
    assert recipe["adapter"]["rank"] == defaults["adapter"]["rank"]
    assert recipe["adapter"]["alpha"] == defaults["adapter"]["alpha"]
    assert recipe["adapter"]["factor"] == defaults["adapter"]["factor"]
    assert recipe["scheduler"] == defaults["scheduler"]
    if family == "krea2":
        assert recipe["dataset"]["text_encoding"] == "cached"
        assert recipe["objective"]["timestep_sampling"] == "resolution_shift"
        assert recipe["objective"]["res_shift_tokens"] == [256, 6400]
    if family == "toy":
        assert recipe["model"]["dtype"] == "fp32"
        assert recipe["loop"]["mixed_precision"] == "no"


def test_same_family_is_a_complete_independent_copy(api):
    _, c, _ = api
    original = TrainConfig().to_dict()
    original["model"]["dit_path"] = "/explicit/missing/keep-it.safetensors"
    original["adapter"].update(preset="custom", rules=[{"pattern": "old.*"}], resume_weights="old.bin")
    original["checkpoint"]["resume"] = "/explicit/state"
    original["optimizer"].update(lr=0.75, group_lr={"te": 0.8})
    copy = change_config_family(c, original, "anima")
    assert copy == original
    copy["optimizer"]["group_lr"]["te"] = 1
    assert original["optimizer"]["group_lr"]["te"] == 0.8


@pytest.mark.parametrize("source,target", [("anima", "krea2"), ("krea2", "anima"), ("anima", "toy")])
def test_cross_family_resets_entire_recipe_but_preserves_data(api, source, target):
    _, c, _ = api
    original = initial_family_config(c, source)
    original["model"].update(
        dit_path="old-dit", vae_path="old-vae", tokenizer_path="old-tokenizer", text_encoder_path="old-text"
    )
    original["dataset"].update(
        sources=[{"path": "/data", "repeats": 3, "caption": {"prefix": "kept"}, "is_reg": True}],
        masked_loss=True,
        caption={"trigger_word": "keep", "shuffle": True},
        resolution_mode="native",
        native_max_pixels=500000,
        text_encoding="online",
    )
    original["validation"].update(enabled=True, sources=[{"path": "/validation"}], every_epochs=5)
    original["adapter"].update(preset="not-target", rules=[{"pattern": "old.*"}], resume_weights="old.bin")
    original["checkpoint"].update(resume="old-state", name="old", save_every_steps=17)
    original["optimizer"].update(type="prodigy", lr=1.0, group_lr={"te": 0.9})
    original["memory"].update(base_precision="fp8_e4m3", blocks_to_swap=99)
    original["sampling"].update(
        enabled=True, steps=99, cfg=100, shift=50, prompts=[{"prompt": "old", "steps": 90, "cfg": 100}]
    )
    original["loop"].update(epochs=555, ema=True)
    before = deepcopy(original)
    changed = change_config_family(c, original, target)
    defaults = initial_family_config(c, target)
    assert original == before
    for section in defaults.keys() - {"dataset", "validation"}:
        assert changed[section] == defaults[section], section
    assert changed["validation"] == original["validation"]
    assert changed["dataset"] == {
        **original["dataset"],
        "text_encoding": defaults["dataset"]["text_encoding"],
    }
    changed["dataset"]["sources"][0]["caption"]["prefix"] = "new"
    assert original == before


def test_only_existing_target_family_defaults_with_matching_path_types_are_selected(api, tmp_path):
    _, c, _ = api
    directory = tmp_path / "weights"
    directory.mkdir()
    weights = directory / "weights.safetensors"
    weights.write_bytes(b"registered weight fixture")

    def asset(mid, family, kind, path, default=1):
        c.db.insert(
            "models",
            {
                "id": mid,
                "family": family,
                "kind": kind,
                "path": str(path),
                "is_default": default,
                "created_at": 1,
            },
        )

    asset("m_dit", "krea2", "dit", weights)
    asset("m_text", "krea2", "text_encoder", directory)
    asset("m_wrong_vae", "krea2", "vae", directory)
    asset("m_wrong_tokenizer", "krea2", "tokenizer", weights)
    asset("m_other_family", "anima", "vae", weights)
    asset("m_missing", "krea2", "vae", directory / "gone.safetensors")
    asset("m_nondefault", "krea2", "vae", weights, default=0)
    asset("m_wrong_kind", "krea2", "tagger", weights)
    recipe = initial_family_config(c, "krea2")
    assert recipe["model"]["dit_path"] == str(weights)
    assert recipe["model"]["text_encoder_path"] == str(directory)
    assert recipe["model"]["vae_path"] is None
    assert recipe["model"]["tokenizer_path"] is None
    weights.unlink()
    assert initial_family_config(c, "krea2")["model"]["dit_path"] is None


def test_cross_family_snapshot_copies_sidecars_and_never_rewrites_old_config_or_job(api, tmp_path):
    client, c, p = api
    pid, source_id = p["id"], p["active_version_id"]
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (64, 64), "red").save(source / "one.png")
    Image.new("L", (64, 64), 128).save(source / "one.mask.png")
    (source / "one.txt").write_text("keep this caption")
    config = client.get(f"/api/projects/{pid}/config").json()
    config["dataset"].update(sources=[{"path": str(source), "repeats": 3}], masked_loss=True)
    config["optimizer"]["lr"] = 0.321
    config["checkpoint"]["resume"] = "/old/state"
    assert client.put(f"/api/projects/{pid}/config", json=config).status_code == 200
    config_file = c.config_path(pid, source_id)
    original_bytes = config_file.read_bytes()
    job_config = json.dumps(config)
    c.db.insert(
        "jobs",
        {
            "id": "j_old",
            "type": "train",
            "name": "Old",
            "project_id": pid,
            "version_id": source_id,
            "status": "completed",
            "created_at": 1,
            "run_dir": "/old/run",
            "config_json": job_config,
        },
    )
    created = c.versions.create(pid, "Krea copy", "", source_id, "copy", family="krea2")
    row = wait_ready(c, pid, created["id"])
    changed = json.loads(c.config_path(pid, row["id"]).read_text())
    assert changed["model"]["family"] == "krea2"
    assert changed["dataset"]["masked_loss"] is True
    assert changed["dataset"]["sources"][0]["repeats"] == 3
    copy = Path(changed["dataset"]["sources"][0]["path"])
    assert copy != source
    for name in ("one.png", "one.txt", "one.mask.png"):
        assert (copy / name).read_bytes() == (source / name).read_bytes()
        assert (copy / name).stat().st_ino != (source / name).stat().st_ino
    assert changed["checkpoint"]["resume"] is None
    assert changed["optimizer"]["lr"] == 1e-4
    assert config_file.read_bytes() == original_bytes
    assert c.db.fetchone("SELECT config_json FROM jobs WHERE id='j_old'")["config_json"] == job_config
    assert version_row(c, row)["family"] == "krea2"
    empty = c.versions.create(pid, "Empty toy", "", source_id, "empty", family="toy")
    empty_row = wait_ready(c, pid, empty["id"])
    empty_config = json.loads(c.config_path(pid, empty_row["id"]).read_text())
    assert empty_config["model"]["family"] == "toy"
    assert empty_config["dataset"]["sources"] == []
    assert empty_config["validation"]["sources"] == []


@pytest.mark.parametrize("family", ["flux", "sdxl", "unknown"])
def test_unsupported_family_leaves_no_version_or_busy_source(api, family):
    _, c, p = api
    before = c.db.fetchall("SELECT * FROM project_versions")
    paths = set(c.project_dir(p["id"]).rglob("*"))
    with pytest.raises(ApiError, match="unsupported training family"):
        c.versions.create(p["id"], "Invalid family", "", p["active_version_id"], "empty", family=family)
    assert c.db.fetchall("SELECT * FROM project_versions") == before
    assert set(c.project_dir(p["id"]).rglob("*")) == paths


def test_version_family_handles_legacy_pending_and_malformed_configs(api):
    _, c, p = api
    row = c.resolve_version(p["id"], p["active_version_id"])
    path = c.config_path(p["id"], row["id"])
    path.unlink()
    assert version_family(c, row) == "anima"
    assert version_family(c, {**row, "status": "copying"}) is None
    for content in ("{broken", "[]", '{"model":null}', '{"model":{"family":"sdxl"}}'):
        path.write_text(content)
        assert version_family(c, row) is None
        assert version_row(c, row)["family"] is None
    path.write_text('{"optimizer":{"lr":0.1}}')
    assert version_family(c, row) == "anima"
    path.write_text('{"model":{"family":"toy"}}')
    assert version_family(c, row) == "toy"
