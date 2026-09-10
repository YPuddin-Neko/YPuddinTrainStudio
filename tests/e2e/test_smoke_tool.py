"""`ypuddin smoke` on the toy family: the same path a GPU machine uses to validate Anima."""

import json

from ypuddin.cli import main


def test_smoke_cli_toy_family(tmp_path, capsys):
    out = tmp_path / "smoke"
    rc = main(
        [
            "smoke",
            "--set",
            "model.family=toy",
            "--set",
            "model.dtype=fp32",
            "--set",
            "adapter.rank=4",
            "--set",
            "adapter.alpha=4",
            "--set",
            "loop.mixed_precision=no",
            "--set",
            "dataset.bucket_step=16",
            "--out",
            str(out),
            "--steps",
            "2",
            "--resolution",
            "64",
            "--sample-size",
            "64",
            "--sample-steps",
            "2",
            "--device",
            "cpu",
        ]
    )
    text = capsys.readouterr().out
    assert rc == 0, text
    report = json.loads((out / "smoke-report.json").read_text(encoding="utf-8"))
    assert report["ok"] and report["family"] == "toy" and report["outcome"] == "finished"
    names = {c["name"]: c["ok"] for c in report["checks"]}
    assert names["training finished"] and names["sample saved"] and names["adapter file round-trips"]
    assert len(report["losses"]) == 2 and (out / "data").exists()  # dataset was synthesised
    assert "PASS" in text and "smoke-report.json" in text
    assert report["export"]["tensors"] > 0 and all(
        k.startswith("lora_unet_") for k in report["export"]["example_keys"]
    )
