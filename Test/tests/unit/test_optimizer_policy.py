"""The values users see and save must match the optimizer that actually runs."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ypuddin.config import OptimizerConfig, TrainConfig, dump_toml, load_config
from ypuddin.config.optimizer_rules import optimizer_capabilities, optimizer_specific_fields
from ypuddin.server import create_app


@pytest.mark.parametrize(
    "name,lr,schedule",
    [
        ("prodigy", 1, "cosine"),
        ("prodigy_plus_sf", 1, "constant"),
        ("automagic", 1e-6, "constant"),
        ("adamw_sf", 1e-4, "constant"),
    ],
)
def test_managed_values_survive_roundtrip_and_do_not_mutate_input(name, lr, schedule, tmp_path):
    raw = {"optimizer": {"type": name, "lr": 1e-4}, "scheduler": {"type": "cosine", "warmup_steps": 4}}
    original = json.dumps(raw)
    config = TrainConfig.model_validate(raw)
    assert config.optimizer.lr == lr
    assert config.scheduler.type == schedule
    assert config.scheduler.warmup_steps == (4 if name == "prodigy" else 0)
    assert json.dumps(raw) == original
    path = tmp_path / "config.toml"
    path.write_text(dump_toml(config))
    assert load_config(path) == config
    assert TrainConfig.model_validate_json(config.model_dump_json()) == config


def test_ppsf_can_disable_averaging_without_ignoring_scheduler():
    config = TrainConfig.model_validate(
        {
            "optimizer": {"type": "prodigy_plus_sf", "use_schedulefree": False},
            "scheduler": {"type": "linear", "warmup_steps": 5},
        }
    )
    assert config.scheduler.type == "linear"
    assert config.scheduler.warmup_steps == 5


@pytest.mark.parametrize("name", ["prodigy", "prodigy_plus_sf"])
def test_existing_d_coef_is_migrated_and_remains_tunable(name):
    config = OptimizerConfig(type=name, args={"d_coef": 2.5, "d0": 2e-6, "beta3": 0.8})
    assert (config.d_coef, config.d0, config.beta3) == (2.5, 2e-6, 0.8)
    assert config.args == {}
    config.d_coef = 3
    assert config.d_coef == 3
    with pytest.raises(ValidationError, match="conflicts"):
        OptimizerConfig(type=name, d_coef=3, args={"d_coef": 2})


@pytest.mark.parametrize(
    "fragment",
    [
        {"optimizer": {"group_lr": {"attn": 0.001}}},
        {"adapter": {"lr_scale": {"w2": 0.5}}},
        {"adapter": {"rules": [{"match": "*", "lr": 0.001}]}},
        {"optimizer": {"args": {"lr": 0.001}}},
        {"optimizer": {"args": {"fused_back_pass": True}}},
    ],
)
def test_manual_bypasses_are_rejected(fragment):
    fragment.setdefault("optimizer", {})["type"] = "prodigy_plus_sf"
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(fragment)


@pytest.mark.parametrize(
    "field,value",
    [
        ("d_coef", 0),
        ("d_coef", float("inf")),
        ("d0", float("nan")),
        ("beta3", 1),
        ("prodigy_steps", -1),
        ("schedulefree_c", -0.1),
        ("betas", (0, 0.99)),
        ("betas", (0.9, 1)),
        ("lr", float("nan")),
    ],
)
def test_invalid_values_fail_before_optimizer_or_preview(field, value):
    with pytest.raises(ValidationError):
        OptimizerConfig(type="prodigy_plus_sf", **{field: value})


@pytest.mark.parametrize(
    "options",
    [
        {"use_cautious": True, "use_grams": True},
        {"use_focus": True, "factored": True},
        {"use_focus": True, "factored": False, "eps": None, "use_stableadamw": False},
        {"eps": None, "use_stableadamw": True},
    ],
)
def test_ppsf_rejects_combinations_the_library_would_silently_change(options):
    with pytest.raises(ValidationError):
        OptimizerConfig(type="prodigy_plus_sf", **options)


def test_custom_optimizer_args_are_not_lost():
    config = OptimizerConfig(type="torch.optim.AdamW", args={"betas": [0.7, 0.8], "eps": 1e-6})
    assert config.args == {"betas": [0.7, 0.8], "eps": 1e-6}


def test_assignment_migration_is_atomic_and_keeps_managed_values():
    config = OptimizerConfig(type="adamw")
    config.args = {"betas": [0.7, 0.8]}
    assert config.betas == (0.7, 0.8) and config.args == {}
    config.betas = (0.8, 0.9)
    config.type = "prodigy_plus_sf"
    config.lr = 0.2
    assert config.lr == 1
    config.use_cautious = True
    with pytest.raises(ValidationError):
        config.use_grams = True
    assert config.use_cautious and not config.use_grams


@pytest.mark.parametrize("args", [123, {"betas": 0.9}])
def test_bad_legacy_args_raise_a_validation_error_instead_of_server_failure(args):
    with pytest.raises(ValidationError):
        OptimizerConfig(type="prodigy", args=args)


def test_class_aliases_obey_same_policy():
    for name, metadata in optimizer_capabilities().items():
        for alias in metadata.get("aliases", []):
            config = TrainConfig.model_validate({"optimizer": {"type": alias}})
            assert config.optimizer.type == name
            expected = TrainConfig.model_validate({"optimizer": {"type": name}})
            assert config == expected


def test_all_special_parameters_have_typed_visible_controls():
    schema = TrainConfig.json_schema()
    properties = schema["$defs"]["OptimizerConfig"]["properties"]
    for name in ("prodigy", "prodigy_plus_sf", "automagic"):
        for field in optimizer_specific_fields(name):
            assert properties[field]["description"]
            assert not properties[field]["x-ui"].get("advanced")
            assert not properties[field]["x-ui"].get("hidden")
    assert properties["fused_backward"]["x-ui"]["hidden"]
    assert properties["args"]["x-ui"]["advanced"]


def test_save_migrates_only_optimizer_values_and_read_does_not_rewrite_old_presets(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "missing")
    try:
        with TestClient(app) as client:
            body = {
                "name": "adaptive",
                "config": {
                    "optimizer": {"type": "prodigy_plus_sf", "lr": 1e-4, "args": {"d_coef": 2}},
                    "adapter": {"rank": 8},
                },
            }
            saved = client.post("/api/presets", json=body)
            assert saved.status_code == 200, saved.text
            config = saved.json()["config"]
            assert config["optimizer"]["lr"] == 1
            assert config["optimizer"]["d_coef"] == 2
            assert config["optimizer"]["args"] == {}
            assert config["scheduler"]["type"] == "constant"
            assert config["adapter"] == {"rank": 8}
            assert "dataset" not in config and "model" not in config
            file = app.state.ctx.data_root / "presets" / "legacy.json"
            file.write_text(json.dumps({"config": body["config"]}))
            before = file.read_bytes()
            assert client.get("/api/presets/legacy").json()["config"] == body["config"]
            assert file.read_bytes() == before
    finally:
        app.state.ctx.db.close()
