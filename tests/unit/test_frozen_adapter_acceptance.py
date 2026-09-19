"""The portable acceptance runner must never overwrite results or hide failures."""

import json
import sys

import pytest
import torch
from safetensors.torch import save_file

from scripts import verify_frozen_adapter_resume as acceptance


def recipe(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(
        json.dumps(
            {
                "model": {"family": "toy"},
                "training": {"mode": "adapter", "train_backbone": True, "train_text_encoder": False},
                "adapter": {"algo": "lora"},
                "dataset": {"sources": [{"path": str(tmp_path / "images")}]},
                "sampling": {"prompts": [{"prompt": "cat"}]},
            }
        )
    )
    return path


def test_prepare_preserves_input_and_refuses_existing_output(tmp_path, monkeypatch):
    path = recipe(tmp_path)
    before = path.read_bytes()
    out = tmp_path / "prepared"
    monkeypatch.setattr(
        sys, "argv", ["verify", str(path), str(out), "--device", "cuda", "--processes", "2", "--prepare-only"]
    )
    assert acceptance.main() == 0
    assert path.read_bytes() == before
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "prepared" and report["strict_passed"] is False
    assert report["commands"][0].count("ypuddin.cli") == 1
    config = json.loads((out / "reference.json").read_text())
    assert config["training"]["train_text_encoder"] is False
    assert config["loop"]["gpu_count"] == 2
    saved = (out / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        acceptance.main()
    assert (out / "report.json").read_bytes() == saved


def test_gpu_unavailable_is_failure_without_cpu_fallback(tmp_path, monkeypatch):
    path = recipe(tmp_path)
    out = tmp_path / "unavailable"
    monkeypatch.setattr(sys, "argv", ["verify", str(path), str(out), "--device", "cuda"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(acceptance, "run_job", lambda *args: pytest.fail("must not launch"))
    assert acceptance.main() == 1
    report = json.loads((out / "report.json").read_text())
    assert report["strict_passed"] is False and "refusing CPU fallback" in report["error"]


@pytest.mark.parametrize("corrupt", ["weights", "dtype", "optimizer", "rng", "preview"])
def test_comparison_rejects_corruption(tmp_path, corrupt):
    dirs = [tmp_path / name for name in ("reference", "resumed")]
    for directory in dirs:
        state = directory / "state-4"
        state.mkdir(parents=True)
        save_file({"adapter": torch.ones(2)}, state / "training.safetensors")
        (state / "state.json").write_text(json.dumps({"progress": {"step": 4}, "sampler": {"position": 4}}))
        for name in ("optimizer", "scheduler", "rng"):
            torch.save({"state": torch.tensor([1, 2])}, state / f"{name}.pt")
        preview = directory / "sample.png"
        preview.write_bytes(b"same-preview-bytes")
        (directory / "events.jsonl").write_text(
            json.dumps({"type": "run.finished"})
            + "\n"
            + json.dumps({"type": "sample.saved", "step": 4, "path": str(preview)})
            + "\n"
        )
    assert acceptance.compare(*dirs, 4)["states_exact"]
    state = dirs[1] / "state-4"
    if corrupt in {"weights", "dtype"}:
        value = torch.zeros(2) if corrupt == "weights" else torch.ones(2, dtype=torch.float64)
        save_file({"adapter": value}, state / "training.safetensors")
    elif corrupt == "preview":
        (dirs[1] / "sample.png").write_bytes(b"different")
    else:
        torch.save({"state": torch.tensor([1, 3])}, state / f"{corrupt}.pt")
    with pytest.raises(AssertionError):
        acceptance.compare(*dirs, 4)


def test_text_encoder_acceptance_requires_explicit_opt_in(tmp_path, monkeypatch):
    path = recipe(tmp_path)
    data = json.loads(path.read_text())
    data["model"]["family"] = "sdxl"
    data["training"].update(train_backbone=False, train_text_encoder=True)
    data["dataset"]["text_encoding"] = "online"
    path.write_text(json.dumps(data))
    out = tmp_path / "text-acceptance"
    args = ["verify", str(path), str(out), "--device", "cuda", "--prepare-only"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        acceptance.main()
    assert not out.exists()
    monkeypatch.setattr(sys, "argv", [*args, "--allow-text-encoder"])
    assert acceptance.main() == 0
    report = json.loads((out / "report.json").read_text())
    assert report["train_text_encoder"] and not report["train_backbone"]
    assert not report["strict_passed"]
