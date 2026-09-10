"""scripts/bootstrap.py: package-source fallback chain and GPU-aware torch flavour selection (no network)."""

import importlib.util
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ypuddin_bootstrap", Path(__file__).resolve().parents[2] / "scripts" / "bootstrap.py"
)
boot = importlib.util.module_from_spec(SPEC)
sys.modules["ypuddin_bootstrap"] = boot
SPEC.loader.exec_module(boot)


def test_index_chains_orders_mirrors_then_official(monkeypatch):
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: True)
    pypi, torch_src = boot.index_chains("cn", "cu128")
    assert pypi[0].startswith("https://mirrors.ustc.edu.cn") and "tuna" in pypi[1] and "aliyun" in pypi[2]
    assert pypi[-1] == boot.PYPI_OFFICIAL
    assert (
        torch_src[0][0] == "find-links" and "aliyun" in torch_src[0][1] and torch_src[0][1].endswith("cu128")
    )
    assert torch_src[-1] == ("index-url", "https://download.pytorch.org/whl/cu128")
    pypi_off, torch_off = boot.index_chains("official", "cu126")
    assert pypi_off[0] == boot.PYPI_OFFICIAL and len(pypi_off) == 4  # mirrors remain as fallback
    assert torch_off == [("index-url", "https://download.pytorch.org/whl/cu126")]
    # auto: mirrors first regardless of whether pypi.org is reachable (official is the last resort)
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: True)
    assert boot.index_chains("auto", "cu128")[0][0].startswith("https://mirrors.ustc.edu.cn")
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: "pypi.org" not in url)
    assert boot.index_chains("auto", "cu128")[0][0].startswith("https://mirrors.ustc.edu.cn")
    # dead mirrors are moved behind the live ones instead of costing a pip timeout each
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: "ustc" not in url and "tuna" not in url)
    chain = boot.index_chains("cn", "cu128")[0]
    assert (
        "aliyun" in chain[0] and chain[1] == boot.PYPI_OFFICIAL and "ustc" in chain[2] and "tuna" in chain[3]
    )


@pytest.mark.parametrize(
    ("driver", "gpus", "expected"),
    [
        (None, [], "cpu"),
        (580, [("NVIDIA GeForce RTX 5090", 12.0)], "cu128"),
        (
            566,
            [("NVIDIA GeForce RTX 5070", 12.0)],
            "cu128",
        ),  # Blackwell always needs cu128 (+ driver warning)
        (566, [("NVIDIA GeForce RTX 4090", 8.9)], "cu126"),
        (552, [("NVIDIA GeForce RTX 4070", 8.9)], "cu124"),
        (535, [("NVIDIA GeForce RTX 3090", 8.6)], "cu118"),
        (440, [("Tesla T4", 7.5)], "cpu"),
    ],
)
def test_pick_torch_tag_by_gpu_and_driver(monkeypatch, driver, gpus, expected):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    monkeypatch.setattr(boot, "nvidia_driver_major", lambda: driver)
    monkeypatch.setattr(boot, "nvidia_gpus", lambda: gpus)
    assert boot.pick_torch_tag("auto") == expected
    assert boot.pick_torch_tag("cu118") == "cu118"  # explicit request always wins


def test_install_falls_back_to_the_next_source_on_failure(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    class Result:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        joined = " ".join(cmd)
        if "aliyun.com/pytorch-wheels" in joined:
            return Result(1)  # torch listing lacks the wheel -> next torch source
        if "[models,server]" in joined and "mirrors.ustc.edu.cn/pypi" in joined:
            return Result(1)  # first mirror lacks a package -> next PyPI index
        return Result(0)

    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: True)
    monkeypatch.setattr(boot, "uv_path", lambda: None)
    monkeypatch.setattr(boot, "VENV", tmp_path / "venv")
    monkeypatch.setattr(boot, "MARKER", tmp_path / "venv" / "marker.json")
    (tmp_path / "venv" / ("Scripts" if boot.WIN else "bin")).mkdir(parents=True)
    boot.venv_python().write_text("")  # pretend the venv exists so no interpreter lookup happens
    boot.ensure_venv("cu128", index_mode="cn", reinstall=False, extras="models,server")
    torch_calls = [c for c in calls if "torch>=2.4" in c]
    # torch: aliyun flat listing (wheel only, no index) failed -> sjtu PEP 503 index succeeded
    assert {"--no-index", "--no-deps", "--find-links"} <= set(torch_calls[0])
    assert "aliyun" in " ".join(torch_calls[0])
    assert "--find-links" not in torch_calls[1]
    assert "sjtu" in torch_calls[1][torch_calls[1].index("--index-url") + 1]
    assert len(torch_calls) == 2
    # ypuddin itself: ustc failed -> tuna succeeded
    pkg_calls = [c for c in calls if any(a.endswith("[models,server]") for a in c)]
    assert [c[c.index("--index-url") + 1] for c in pkg_calls] == list(boot.PYPI_MIRRORS_CN[:2])
    assert boot.MARKER.exists()


def test_install_dies_when_every_source_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.subprocess, "run", lambda cmd, *a, **kw: type("R", (), {"returncode": 1})())
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: True)
    monkeypatch.setattr(boot, "uv_path", lambda: None)
    monkeypatch.setattr(boot, "VENV", tmp_path / "venv")
    monkeypatch.setattr(boot, "MARKER", tmp_path / "venv" / "marker.json")
    (tmp_path / "venv" / ("Scripts" if boot.WIN else "bin")).mkdir(parents=True)
    boot.venv_python().write_text("")
    with pytest.raises(SystemExit):
        boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras="models,server")


def test_flat_listing_success_installs_wheel_then_dependencies(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def ok_run(cmd, *a, **kw):
        calls.append(list(cmd))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(boot.subprocess, "run", ok_run)
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: True)
    monkeypatch.setattr(boot, "uv_path", lambda: None)
    monkeypatch.setattr(boot, "VENV", tmp_path / "venv")
    monkeypatch.setattr(boot, "MARKER", tmp_path / "venv" / "marker.json")
    (tmp_path / "venv" / ("Scripts" if boot.WIN else "bin")).mkdir(parents=True)
    boot.venv_python().write_text("")
    boot.ensure_venv("cu126", index_mode="cn", reinstall=False, extras="models,server")
    torch_calls = [c for c in calls if "torch>=2.4" in c]
    assert len(torch_calls) == 2
    assert "--no-index" in torch_calls[0] and torch_calls[0][-1].endswith("pytorch-wheels/cu126")
    assert (
        "--upgrade" not in torch_calls[1] and "--index-url" in torch_calls[1]
    )  # deps only, keep the cu126 wheel
