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


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"adapter": {"algo": "lokr", "bogus": 1}})


def test_overrides_and_presets(tmp_path):
    preset = tmp_path / "p.toml"
    preset.write_text('[adapter]\nalgo = "lora"\nrank = 8\n', encoding="utf-8")
    cfg = load_config(presets=[preset], overrides=["loop.epochs=3", "adapter.alpha=4", "dataset.resolutions=[512,768]"])
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
