"""Real CPU workers and owned-process failure cleanup; never simulate a CUDA pass."""

import json
import sys
from pathlib import Path

import psutil
import pytest

from ypuddin.tools import runtime_check as module


def test_real_two_process_cpu_training_and_artifacts(tmp_path):
    out = tmp_path / "fresh"
    report = module.run_runtime_check(["cpu", "cpu"], out, timeout=60)
    assert report["passed"], report
    assert all(report["checks"].values())
    assert report["windows_cuda_validated"] is False
    assert "real Windows CUDA" in report["not_validated"]
    assert json.loads((out / "report.json").read_text()) == report
    assert len({row["pid"] for row in report["workers"]}) == 2
    latest_first = max(row["result"]["first_step"]["completed_unix"] for row in report["workers"])
    for row in report["workers"]:
        result = row["result"]
        assert row["exit_code"] == 0
        assert result["actual_device"] == result["backend"] == "cpu"
        assert result["steps"][1]["completed_unix"] >= latest_first
        assert len(result["steps"]) == 3
        assert result["export"]["changed_tensor_count"] > 0
        assert Path(result["export"]["path"]).is_file()
        assert result["samples"] and all(Path(s["path"]).is_file() for s in result["samples"])


def test_existing_output_and_invalid_devices_are_not_modified(tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("existing user data")
    with pytest.raises(FileExistsError):
        module.run_runtime_check(["cpu", "cpu"], tmp_path)
    assert keep.read_text() == "existing user data"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]
    for devices in (["cuda:0", "cuda:0"], ["cuda", "cpu"], ["cpu"]):
        with pytest.raises(ValueError):
            module.run_runtime_check(devices, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_requested_cuda_cannot_fall_back_to_cpu(tmp_path, monkeypatch):
    # Deliberately hide all GPUs even on a CUDA host. This tests failure handling,
    # not a fabricated successful CUDA execution.
    environment = module._worker_environment
    monkeypatch.setattr(module, "_worker_environment", lambda _device: environment("cpu"))
    report = module.run_runtime_check(["cuda:0", "cpu"], tmp_path / "unavailable", timeout=60)
    assert not report["passed"] and not report["windows_cuda_validated"]
    assert report["checks"]["all_owned_processes_exited"]
    cuda = report["workers"][0]
    assert cuda["exit_code"] != 0
    assert "禁止回退到 CPU" in cuda["result"]["error"]
    assert cuda["result"]["passed"] is False
    assert "actual_device" not in cuda["result"]


def test_deadline_cleans_only_owned_workers_and_descendants(tmp_path, monkeypatch):
    program = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    monkeypatch.setattr(
        module,
        "_worker_command",
        lambda i, out, *_: [sys.executable, "-c", program, str(out / f"child-{i}.pid")],
    )
    report = module.run_runtime_check(["cpu", "cpu"], tmp_path / "timeout", timeout=3)
    assert not report["passed"] and "TimeoutError" in report["error"]
    assert report["checks"]["all_owned_processes_exited"]
    assert len(report["cleanup"]) == 2
    for row in report["workers"]:
        assert row["exit_code"] is not None and row["exit_code"] != 0
        pid = int((tmp_path / "timeout" / f"child-{row['index']}.pid").read_text())
        try:
            process = psutil.Process(pid)
            assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    assert all(not row.get("errors") and not row.get("remaining_descendants") for row in report["cleanup"])


def test_failed_worker_stops_peer_and_keeps_exit_codes(tmp_path, monkeypatch):
    fail = (
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
        "exec('while not p.exists(): time.sleep(0.01)'); sys.exit(7)"
    )
    wait = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
    monkeypatch.setattr(
        module,
        "_worker_command",
        lambda i, out, *_: [sys.executable, "-c", fail if i == 0 else wait, str(out / "peer-ready")],
    )
    report = module.run_runtime_check(["cpu", "cpu"], tmp_path / "failed", timeout=10)
    assert not report["passed"] and "RuntimeError" in report["error"]
    assert report["workers"][0]["exit_code"] == 7
    assert report["workers"][1]["exit_code"] not in (None, 0)
    assert report["checks"]["all_owned_processes_exited"]


@pytest.mark.parametrize("payload", ["{broken", "null", "[]", '{"first_step":null}', '{"first_step":[]}'])
def test_corrupt_child_report_keeps_parent_failure_report(tmp_path, monkeypatch, payload):
    program = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'report.json').write_text(sys.argv[2])"
    monkeypatch.setattr(
        module,
        "_worker_command",
        lambda i, out, *_: [sys.executable, "-c", program, str(out / f"worker-{i}"), payload],
    )
    out = tmp_path / "broken"
    report = module.run_runtime_check(["cpu", "cpu"], out, timeout=10)
    assert not report["passed"]
    assert all(row["exit_code"] == 0 and "report_error" in row for row in report["workers"])
    assert json.loads((out / "report.json").read_text()) == report


def test_real_forwarding_launcher_allows_windows_venv_redirector_shape(tmp_path, monkeypatch):
    command = module._worker_command
    shim = "import subprocess,sys; raise SystemExit(subprocess.call(sys.argv[1:]))"
    monkeypatch.setattr(
        module,
        "_worker_command",
        lambda *args: [sys.executable, "-c", shim, *command(*args)],
    )
    report = module.run_runtime_check(["cpu", "cpu"], tmp_path / "redirected", timeout=60)
    assert report["passed"], report
    assert all(row["pid"] != row["result"]["pid"] for row in report["workers"])
    assert report["checks"]["all_owned_processes_exited"]


def test_internal_worker_cannot_write_arbitrary_existing_directory(tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("existing user data")
    for deadline in ("nan", "10000000000"):
        with pytest.raises(SystemExit) as error:
            module.main(
                [
                    "--device",
                    "cpu",
                    "--out",
                    str(tmp_path),
                    "--worker-index",
                    "0",
                    "--deadline",
                    deadline,
                ]
            )
        assert error.value.code == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]
