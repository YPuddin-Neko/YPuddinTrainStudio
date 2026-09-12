import json

import pytest
from pydantic import ValidationError

from ypuddin.config import (
    GROUPS,
    ShowWhenError,
    TrainConfig,
    config_hash,
    dump_toml,
    evaluate,
    load_config,
    parse_overrides,
)


def test_defaults_validate_and_roundtrip(tmp_path):
    cfg = TrainConfig()
    assert cfg.adapter.algo == "lokr"
    text = dump_toml(cfg)
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    cfg2 = load_config(p)
    assert cfg2 == cfg
    assert config_hash(cfg) == config_hash(cfg2)


def test_sampling_algorithms_roundtrip_and_keep_legacy_defaults(tmp_path):
    # Omitting the new options must retain the original Euler/linear-grid preview.
    legacy = TrainConfig.model_validate({"sampling": {"steps": 12, "shift": 3}})
    assert legacy.sampling.sampler == "euler"
    assert legacy.sampling.scheduler == "uniform"
    cfg = load_config(
        overrides=[
            "sampling.sampler=er_sde",
            "sampling.scheduler=sgm_uniform",
            "sampling.er_sde_order=2",
            "sampling.er_sde_s_noise=0.4",
        ]
    )
    path = tmp_path / "sampling.toml"
    path.write_text(dump_toml(cfg), encoding="utf-8")
    assert load_config(path) == cfg
    assert TrainConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert cfg.scheduler == legacy.scheduler  # The optimizer LR schedule is a separate setting.


@pytest.mark.parametrize(
    "field,value",
    [
        ("sampler", "unknown"),
        ("scheduler", "cosine"),
        ("er_sde_order", 0),
        ("er_sde_order", 4),
        ("er_sde_s_noise", -0.01),
        ("er_sde_s_noise", 1.01),
        *[
            (field, value)
            for field in ("er_sde_s_noise", "cfg", "shift")
            for value in (float("nan"), float("inf"), -float("inf"))
        ],
    ],
)
def test_invalid_sampling_algorithm_options_rejected(field, value):
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"sampling": {field: value}})


def test_sampling_advanced_options_only_visible_for_enabled_er_sde():
    properties = TrainConfig.json_schema()["$defs"]["SamplingConfig"]["properties"]
    assert properties["sampler"]["enum"] == ["euler", "heun", "er_sde"]
    assert properties["scheduler"]["enum"] == ["uniform", "simple", "sgm_uniform", "normal"]
    for field in ("er_sde_order", "er_sde_s_noise"):
        hints = properties[field]["x-ui"]
        assert hints["advanced"]
        for enabled, sampler, visible in (
            (True, "er_sde", True),
            (False, "er_sde", False),
            (True, "euler", False),
        ):
            assert (
                evaluate(hints["show_when"], {"sampling": {"enabled": enabled, "sampler": sampler}})
                is visible
            )


def test_optimizer_select_lists_actual_registry_and_preserves_custom_paths():
    from ypuddin.config import OptimizerConfig
    from ypuddin.optim.factory import _BUILTIN

    properties = TrainConfig.json_schema()["$defs"]["OptimizerConfig"]["properties"]
    hints = properties["type"]["x-ui"]
    assert set(hints["options"]) == set(_BUILTIN)
    assert hints["control"] == "select" and hints["allow_custom"]
    assert "enum" not in properties["type"]  # Suggestions must not prohibit a valid custom class.
    assert OptimizerConfig(type="torch.optim.SGD").type == "torch.optim.SGD"
    assert not evaluate(properties["betas"]["x-ui"]["show_when"], {"optimizer": {"type": "sgd"}})
    assert evaluate(properties["betas"]["x-ui"]["show_when"], {"optimizer": {"type": "adamw"}})
    assert not evaluate(properties["eps"]["x-ui"]["show_when"], {"optimizer": {"type": "adafactor"}})


def test_caption_auto_is_only_a_default_not_a_legacy_config_migration():
    from ypuddin.config import DatasetSourceConfig

    assert DatasetSourceConfig(path="/dataset").caption_ext == "auto"
    assert DatasetSourceConfig(path="/dataset", caption_ext=".txt").caption_ext == ".txt"
    assert DatasetSourceConfig(path="/dataset", caption_ext=".caption").caption_ext == ".caption"


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"adapter": {"algo": "lokr", "bogus": 1}})


def test_overrides_and_presets(tmp_path):
    preset = tmp_path / "p.toml"
    preset.write_text('[adapter]\nalgo = "lora"\nrank = 8\n', encoding="utf-8")
    cfg = load_config(
        presets=[preset], overrides=["loop.epochs=3", "adapter.alpha=4", "dataset.resolutions=[512,768]"]
    )
    assert cfg.adapter.algo == "lora" and cfg.adapter.rank == 8 and cfg.adapter.alpha == 4
    assert cfg.loop.epochs == 3
    assert cfg.dataset.resolutions == [768, 512]


def test_parse_overrides_types():
    d = parse_overrides(["a.b=true", "a.c=1.5", "a.d=null", "e=full", "f=[1,2]"])
    assert d == {"a": {"b": True, "c": 1.5, "d": None}, "e": "full", "f": [1, 2]}


def test_cross_rules():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"optimizer": {"fused_backward": True}, "loop": {"grad_accum": 2}})
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"adapter": {"algo": "lora", "rank": "full"}})
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"loop": {"epochs": None, "max_steps": None}})
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"sampling": {"enabled": True}})


def test_json_schema_has_ui_hints():
    schema = TrainConfig.json_schema()
    assert schema["x-ui-groups"] == list(GROUPS)
    adapter = schema["$defs"]["AdapterConfig"]["properties"]
    assert adapter["factor"]["x-ui"]["show_when"] == "adapter.algo == 'lokr'"
    assert adapter["algo"]["x-ui"]["group"] == "adapter"
    json.dumps(schema)  # serializable


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("adapter.algo == 'lokr'", True),
        ("adapter.algo != 'lokr'", False),
        ("adapter.algo in ['lora','lokr']", True),
        ("adapter.rank > 8 && loop.epochs >= 10", True),
        ("!(adapter.dora == true)", True),
        ("loop.max_steps == null", True),
        ("missing.path == null", True),
        ("missing.path > 1", False),
        ("(adapter.algo == 'lora' || adapter.algo == 'lokr') && adapter.rs_lora == false", True),
        ("sampling.enabled == true", False),
        ("dataset.resolutions in [1024]", False),
    ],
)
def test_show_when(expr, expected):
    cfg = TrainConfig()
    assert evaluate(expr, cfg.to_dict()) is expected
    assert evaluate(expr, cfg) is expected


def test_show_when_errors():
    with pytest.raises(ShowWhenError):
        evaluate("adapter.algo ==", {})
    with pytest.raises(ShowWhenError):
        evaluate("a.b $ 1", {})


def test_all_show_when_expressions_in_schema_parse():
    from ypuddin.config import show_when

    def walk(node):
        if isinstance(node, dict):
            if "x-ui" in node and "show_when" in node["x-ui"]:
                show_when.parse(node["x-ui"]["show_when"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(TrainConfig.json_schema())


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "..\\escape",
        "/absolute",
        "C:relative",
        "CON",
        "nul.txt",
        "LPT9",
        "COM1.data",
        "trailing.",
        "trailing ",
        "bad?glob",
        "bad\nline",
        "",
    ],
)
def test_checkpoint_prefix_stays_a_portable_filename(name):
    with pytest.raises(ValueError, match="name"):
        TrainConfig.model_validate({"checkpoint": {"name": name}})


def test_checkpoint_prefix_allows_unicode_dots_and_hyphens():
    assert (
        TrainConfig.model_validate({"checkpoint": {"name": "角色.v2-lora"}}).checkpoint.name == "角色.v2-lora"
    )


def test_explicit_null_defaults_survive_toml_file_and_override_merge(tmp_path):
    from ypuddin.config import parse_toml, write_config

    cfg = TrainConfig.model_validate(
        {
            "loop": {"epochs": None, "max_steps": 3},
            "sampling": {"every_epochs": None},
            "validation": {"every_epochs": None},
            "checkpoint": {"save_every_epochs": None},
        }
    )
    text = dump_toml(cfg)
    assert TrainConfig.model_validate(parse_toml(text)) == cfg
    path = tmp_path / "worker.toml"
    write_config(cfg, path)
    loaded = load_config(path, base={"sampling": {"every_epochs": 8, "every_steps": 99}})
    assert loaded == cfg
    assert config_hash(loaded) == config_hash(cfg)
    # A hand-written file without explicit-null metadata retains ordinary schema defaults.
    assert TrainConfig.model_validate(parse_toml("[loop]\nmax_steps=3\n")).loop.epochs == 10


def test_toml_nulls_preserve_nested_lists_and_literal_key_names():
    from ypuddin.config import parse_toml

    cfg = TrainConfig.model_validate(
        {"optimizer": {"args": {"name.with/dots": None, "items": [None, "", {"a": None}]}}}
    )
    assert TrainConfig.model_validate(parse_toml(dump_toml(cfg))) == cfg


def test_explicit_toml_value_overrides_null_metadata():
    from ypuddin.config import parse_toml

    data = parse_toml('__ypuddin_nulls__ = [["loop", "epochs"]]\n[loop]\nepochs=4\n')
    assert data == {"loop": {"epochs": 4}}


@pytest.mark.parametrize(
    "metadata", ['"bad"', "[[]]", '[["loop", 1]]', '[["missing", "epochs"]]', "[[true]]"]
)
def test_invalid_toml_null_metadata_has_clear_parse_error(metadata):
    from ypuddin.config import parse_toml

    with pytest.raises(ValueError, match="null"):
        parse_toml(f"__ypuddin_nulls__ = {metadata}\n[loop]\nmax_steps=3\n")
