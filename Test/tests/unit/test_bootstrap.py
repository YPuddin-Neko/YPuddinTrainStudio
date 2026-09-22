"""scripts/bootstrap.py: package-source fallback chain and GPU-aware torch flavour selection (no network)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("ypuddin_bootstrap", SOURCE_ROOT / "scripts" / "bootstrap.py")
boot = importlib.util.module_from_spec(SPEC)
sys.modules["ypuddin_bootstrap"] = boot
SPEC.loader.exec_module(boot)
REAL_TORCH_NUMPY_BRIDGE = boot.torch_numpy_bridge


@pytest.fixture(autouse=True)
def isolated_bootstrap_root(monkeypatch, tmp_path_factory):
    root = tmp_path_factory.mktemp("bootstrap-project")
    (root / "pyproject.toml").write_bytes((SOURCE_ROOT / "pyproject.toml").read_bytes())
    monkeypatch.setattr(boot, "PACKAGE_CACHE_ROOT", None)
    monkeypatch.setattr(boot, "DOWNLOAD_SETTINGS", None)
    monkeypatch.setattr(boot, "ROOT", root)
    monkeypatch.setattr(boot, "PROFILE", "legacy")
    monkeypatch.setattr(boot, "VENV", root / "venv")
    monkeypatch.setattr(boot, "MARKER", root / "venv" / ".ypuddin-install.json")
    monkeypatch.setattr(boot, "FRONTEND", root / "frontend")
    monkeypatch.setattr(boot, "torch_numpy_bridge", lambda: {"ok": True})


def native_wheel(directory, name, version, tag="cp311-cp311-manylinux_2_28_x86_64", requires=()):
    wheel = directory / f"{name}-{version}-{tag}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires),
        )
        archive.writestr(
            f"{name}-{version}.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n",
        )
        archive.writestr(f"{name}-{version}.dist-info/RECORD", "")
    return wheel


def native_plan(command, selected=boot.DTK_NUMPY1_TORCH):
    if "--report" in command:
        Path(command[command.index("--report") + 1]).write_text(
            json.dumps({"install": [{"metadata": {"name": "torch", "version": selected}}]})
        )
        return True
    if "compile" in command:
        Path(command[command.index("--output-file") + 1]).write_text(f"torch=={selected}\nnumpy==2.4.6\n")
        return True
    return False


def build_metadata():
    metadata = boot.ROOT / "ypuddin.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_text("Name: ypuddin\nVersion: 0.5.9\n")
    return metadata


def protected_files():
    paths = [
        boot.VENV / "Lib/site-packages/ypuddin-0.5.9.dist-info/METADATA",
        boot.VENV / "Lib/site-packages/ypuddin-0.5.9.dist-info/RECORD",
        boot.ROOT / "other.egg-info/PKG-INFO",
        boot.ROOT / "studio_data/studio.db",
        boot.ROOT / "custom-training-data/portrait.png",
        boot.ROOT / "ypuddin/__init__.py",
        boot.ROOT.parent / "external-training-data/caption.txt",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"preserve {path.name}".encode())
    return {path: path.read_bytes() for path in paths}


def fresh_torch(monkeypatch, tag):
    version = "2.5.1" + ("+" + tag if tag != "cpu" else "+cpu")
    snapshots = iter([{}, {"torch": version}, {"torch": version}])
    monkeypatch.setattr(boot, "installed_versions", lambda: next(snapshots))
    monkeypatch.setattr(boot, "dependency_issues", lambda extras: [])
    monkeypatch.setattr(boot, "editable_install_ready", lambda: True)
    monkeypatch.setattr(
        boot,
        "platform",
        type(
            "Platform",
            (),
            {"system": staticmethod(lambda: "Linux"), "machine": staticmethod(lambda: "x86_64")},
        )(),
    )
    monkeypatch.setattr(
        boot,
        "torch_runtime",
        lambda: {"version": version, "cuda": None if tag == "cpu" else f"{tag[2:-1]}.{tag[-1]}"},
    )


def test_index_chains_orders_mirrors_then_official(monkeypatch):
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: True)
    pypi, torch_src = boot.index_chains("cn", "cu128")
    assert pypi[0].startswith("https://mirrors.ustc.edu.cn") and "tuna" in pypi[1] and "aliyun" in pypi[2]
    assert pypi[-1] == boot.PYPI_OFFICIAL
    assert torch_src[0][0] == "index-url" and "sjtu" in torch_src[0][1] and torch_src[0][1].endswith("cu128")
    assert torch_src[-1] == ("index-url", "https://download.pytorch.org/whl/cu128")
    pypi_off, torch_off = boot.index_chains("official", "cu126")
    assert pypi_off[0] == boot.PYPI_OFFICIAL and len(pypi_off) == 4  # mirrors remain as fallback
    assert torch_off == [("index-url", "https://download.pytorch.org/whl/cu126")]
    # auto: mirrors first regardless of whether pypi.org is reachable (official is the last resort)
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: True)
    assert boot.index_chains("auto", "cu128")[0][0].startswith("https://mirrors.ustc.edu.cn")
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: "pypi.org" not in url)
    assert boot.index_chains("auto", "cu128")[0][0].startswith("https://mirrors.ustc.edu.cn")
    # A quick failed probe must never override the configured priority.
    monkeypatch.setattr(boot, "url_ok", lambda url, timeout=4.0: "aliyun" in url)
    assert boot.index_chains("cn", "cu128")[0] == [*boot.PYPI_MIRRORS_CN, boot.PYPI_OFFICIAL]


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
    fresh_torch(monkeypatch, "cu128")
    calls: list[list[str]] = []

    class Result:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        joined = " ".join(cmd)
        if "pytorch-wheels" in joined:
            return Result(1)  # first torch mirror lacks the wheel -> next source
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
    # torch: SJTU is preferred, then the Aliyun flat listing and official fallback.
    assert "--index-url" in torch_calls[0] and "sjtu" in " ".join(torch_calls[0])
    assert "--find-links" in torch_calls[1] and "aliyun" in " ".join(torch_calls[1])
    assert len(torch_calls) == 3
    # ypuddin itself: ustc failed -> tuna succeeded
    pkg_calls = [c for c in calls if any(a.endswith("[models,server]") for a in c)]
    assert [c[c.index("--index-url") + 1] for c in pkg_calls] == list(boot.PYPI_MIRRORS_CN[:2])
    assert boot.MARKER.exists()


def test_install_dies_when_every_source_fails(monkeypatch, tmp_path):
    fresh_torch(monkeypatch, "cpu")
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
    fresh_torch(monkeypatch, "cu126")
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
    assert len(torch_calls) == 1
    assert "--index-url" in torch_calls[0] and torch_calls[0][-1].endswith("pytorch-wheels/cu126")


@pytest.mark.parametrize(
    ("system", "tag", "driver", "nvidia"),
    [
        ("Windows", "cu128", 580, True),
        ("Linux", "cu126", 560, True),
        ("Darwin", "cpu", None, False),
        ("Windows", "cpu", None, False),
        ("Linux", "cpu", None, False),
        ("Windows", "cpu", 580, True),
    ],
)
def test_platform_bootstrap_includes_routine_training_dependencies(monkeypatch, system, tag, driver, nvidia):
    monkeypatch.setattr(boot.platform, "system", lambda: system)
    monkeypatch.setattr(boot, "nvidia_driver_major", lambda: driver)
    extras = set(boot.choose_extras(tag).split(","))
    assert {"models", "server", "optim", "logging"} <= extras
    assert ("nvidia" in extras) is nvidia
    assert not {"cuda", "tagging", "wandb"} & extras


def test_launch_dependency_groups_include_real_optimizer_names_and_no_removed_features():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with (boot.ROOT / "pyproject.toml").open("rb") as source:
        groups = tomllib.load(source)["project"]["optional-dependencies"]
    assert "schedulefree>=1.4" in groups["optim"]
    assert "prodigy-plus-schedule-free>=2.0.1" in groups["optim"]
    assert "tensorboard>=2.16" in groups["logging"]
    assert not any(
        "wandb" in requirement or "onnxruntime" in requirement
        for requirements in groups.values()
        for requirement in requirements
    )
    assert not any("nvidia" in requirement for requirement in groups["server"])


def existing_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(boot, "VENV", tmp_path / "venv")
    monkeypatch.setattr(boot, "MARKER", tmp_path / "venv" / "marker.json")
    boot.venv_python().parent.mkdir(parents=True)
    boot.venv_python().touch()
    monkeypatch.setattr(
        boot,
        "index_chains",
        lambda *args: ([boot.PYPI_OFFICIAL], [("index-url", boot.TORCH_OFFICIAL.format(tag="cu128"))]),
    )
    monkeypatch.setattr(boot, "uv_path", lambda: None)
    versions = {"torch": "2.5.1+cu124", "numpy": "1.26.4", "nvidia-cudnn-cu12": "9.1.0", "triton": "3.1.0"}
    monkeypatch.setattr(boot, "installed_versions", lambda: versions.copy())
    monkeypatch.setattr(boot, "torch_runtime", lambda: {"version": versions["torch"], "cuda": "12.4"})
    monkeypatch.setattr(boot, "dependency_issues", lambda extras: [])
    monkeypatch.setattr(boot, "editable_install_ready", lambda: True)
    return versions


@pytest.mark.parametrize("use_uv", [False, True])
def test_incremental_setup_preserves_native_stack_and_only_adds_required_dependencies(
    monkeypatch, tmp_path, use_uv
):
    versions = existing_environment(monkeypatch, tmp_path)
    if use_uv:
        monkeypatch.setattr(boot, "uv_path", lambda: "uv")
        monkeypatch.setattr(boot, "uv_cache_dir", lambda _: tmp_path)
    commands, pinned = [], []

    def install(command, **kwargs):
        commands.append(command)
        assert "--upgrade" not in command and "torch>=2.4" not in command
        if "--constraint" in command:
            pinned.append(Path(command[command.index("--constraint") + 1]).read_text())
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(boot.subprocess, "run", install)
    boot.ensure_venv("cu128", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE + ",nvidia")
    assert len(pinned) == 1
    assert all(f"{name}=={version}\n" in pinned[0] for name, version in versions.items())
    assert any(f"[{boot.EXTRAS_BASE},nvidia]" in " ".join(command) for command in commands)
    assert json.loads(boot.MARKER.read_text())["torch_version"] == "2.5.1+cu124"


def test_matching_marker_repairs_a_missing_dependency_and_then_skips_network(monkeypatch, tmp_path):
    existing_environment(monkeypatch, tmp_path)
    boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("cpu", boot.EXTRAS_BASE)}))
    checks = iter([["schedulefree is missing"], []])
    monkeypatch.setattr(boot, "dependency_issues", lambda _: next(checks))
    commands = []
    monkeypatch.setattr(
        boot.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command) or type("Result", (), {"returncode": 0})(),
    )
    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert any("-e" in command for command in commands)
    monkeypatch.setattr(boot, "dependency_issues", lambda _: [])
    commands.clear()
    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert commands == []


def test_unexpected_native_dependency_change_does_not_mark_install_success(monkeypatch, tmp_path):
    versions = existing_environment(monkeypatch, tmp_path)
    metadata = build_metadata()
    preserved = protected_files()

    def install(command, **kwargs):
        if "-e" in command:
            versions["torch"] = "9.0.0"
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(boot.subprocess, "run", install)
    with pytest.raises(SystemExit):
        boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert not boot.MARKER.exists()
    assert (metadata / "PKG-INFO").is_file()
    assert all(path.read_bytes() == content for path, content in preserved.items())


@pytest.mark.parametrize("use_uv", [False, True])
def test_first_install_cleans_only_root_build_metadata_after_verification(monkeypatch, tmp_path, use_uv):
    fresh_torch(monkeypatch, "cpu")
    monkeypatch.setattr(boot, "uv_path", lambda: "uv" if use_uv else None)
    monkeypatch.setattr(boot, "uv_cache_dir", lambda _: tmp_path)
    monkeypatch.setattr(boot, "find_base_python", lambda: sys.executable)
    preserved = protected_files()
    commands = []
    installed = False

    def install(command, **kwargs):
        nonlocal installed
        commands.append(command)
        if "venv" in command:
            boot.venv_python().parent.mkdir(parents=True, exist_ok=True)
            boot.venv_python().touch()
        if "-e" in command:
            build_metadata()
            installed = True
        return type("Result", (), {"returncode": 0})()

    def ready():
        assert installed
        assert (boot.ROOT / "ypuddin.egg-info/PKG-INFO").exists()
        assert not boot.MARKER.exists()
        return True

    monkeypatch.setattr(boot.subprocess, "run", install)
    monkeypatch.setattr(boot, "editable_install_ready", ready)
    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert any("venv" in command for command in commands)
    assert any("-e" in command for command in commands)
    assert not (boot.ROOT / "ypuddin.egg-info").exists()
    assert boot.MARKER.exists()
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_healthy_marker_cleans_metadata_without_installing_and_is_repeatable(monkeypatch, tmp_path):
    existing_environment(monkeypatch, tmp_path)
    preserved = protected_files()
    metadata = build_metadata()
    boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("cpu", boot.EXTRAS_BASE)}))
    marker_before = boot.MARKER.read_bytes()
    monkeypatch.setattr(
        boot.subprocess, "run", lambda *args, **kwargs: pytest.fail("healthy environment must not reinstall")
    )
    monkeypatch.setattr(
        boot, "index_chains", lambda *args: pytest.fail("healthy environment must not probe indexes")
    )

    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert not metadata.exists()
    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert boot.MARKER.read_bytes() == marker_before
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_legacy_editable_is_reinstalled_before_its_metadata_is_removed(monkeypatch, tmp_path):
    existing_environment(monkeypatch, tmp_path)
    metadata = build_metadata()
    preserved = protected_files()
    boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("cpu", boot.EXTRAS_BASE)}))
    modern = False
    commands = []

    def install(command, **kwargs):
        nonlocal modern
        commands.append(command)
        assert metadata.exists(), "legacy metadata is still needed until modern installation succeeds"
        if "-e" in command:
            modern = True
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(boot, "editable_install_ready", lambda: modern)
    monkeypatch.setattr(boot.subprocess, "run", install)
    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert any("-e" in command for command in commands)
    assert not metadata.exists()
    assert all(path.read_bytes() == content for path, content in preserved.items())


@pytest.mark.parametrize("failure", ["installer", "dependencies", "readiness"])
def test_failed_setup_keeps_root_metadata_and_user_data(monkeypatch, tmp_path, failure):
    existing_environment(monkeypatch, tmp_path)
    metadata = build_metadata()
    preserved = protected_files()
    monkeypatch.setattr(
        boot.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": int(failure == "installer")})(),
    )
    monkeypatch.setattr(
        boot, "dependency_issues", lambda _: ["missing dependency"] if failure == "dependencies" else []
    )
    monkeypatch.setattr(boot, "editable_install_ready", lambda: failure != "readiness")

    with pytest.raises(SystemExit):
        boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert not boot.MARKER.exists()
    assert (metadata / "PKG-INFO").is_file()
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_metadata_symlink_cannot_delete_its_user_data_target(monkeypatch, tmp_path):
    existing_environment(monkeypatch, tmp_path)
    preserved = protected_files()
    metadata = boot.ROOT / "ypuddin.egg-info"
    try:
        metadata.symlink_to(boot.ROOT / "studio_data", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")
    boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("cpu", boot.EXTRAS_BASE)}))
    monkeypatch.setattr(
        boot.subprocess, "run", lambda *args, **kwargs: pytest.fail("cleanup must not reinstall")
    )

    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert metadata.is_symlink()
    assert all(path.read_bytes() == content for path, content in preserved.items())


@pytest.mark.parametrize("kind", ["junction", "file"])
def test_metadata_cleanup_refuses_windows_reparse_points_and_same_named_files(monkeypatch, tmp_path, kind):
    existing_environment(monkeypatch, tmp_path)
    preserved = protected_files()
    metadata = boot.ROOT / "ypuddin.egg-info"
    if kind == "file":
        metadata.write_text("legacy metadata stored in a file")
    else:
        build_metadata()
        info = type("ReparsePoint", (), {"st_mode": metadata.lstat().st_mode, "st_file_attributes": 0x400})()
        real_lstat = Path.lstat
        monkeypatch.setattr(
            Path,
            "lstat",
            lambda path, *args, **kwargs: info if path == metadata else real_lstat(path, *args, **kwargs),
        )
    boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("cpu", boot.EXTRAS_BASE)}))
    monkeypatch.setattr(
        boot.subprocess, "run", lambda *args, **kwargs: pytest.fail("cleanup must not reinstall")
    )
    monkeypatch.setattr(
        boot.shutil,
        "rmtree",
        lambda *args, **kwargs: pytest.fail("non-ordinary metadata paths must not be removed"),
    )

    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert metadata.exists()
    if kind == "file":
        assert metadata.read_text() == "legacy metadata stored in a file"
    else:
        assert (metadata / "PKG-INFO").is_file()
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_blocked_cleanup_warns_without_reinstalling_and_retries_next_start(monkeypatch, tmp_path, capsys):
    existing_environment(monkeypatch, tmp_path)
    metadata = build_metadata()
    preserved = protected_files()
    boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("cpu", boot.EXTRAS_BASE)}))
    marker_before = boot.MARKER.read_bytes()
    monkeypatch.setattr(
        boot.subprocess, "run", lambda *args, **kwargs: pytest.fail("cleanup failure must not reinstall")
    )
    real_rmtree = boot.shutil.rmtree

    def blocked(path, *args, **kwargs):
        if Path(path) == metadata:
            raise PermissionError("metadata is occupied")
        return real_rmtree(path, *args, **kwargs)

    with monkeypatch.context() as denied:
        denied.setattr(boot.shutil, "rmtree", blocked)
        boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert (metadata / "PKG-INFO").exists()
    output = capsys.readouterr()
    assert "ypuddin.egg-info" in output.out + output.err
    assert "metadata is occupied" in output.out + output.err
    assert boot.MARKER.read_bytes() == marker_before
    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert not metadata.exists()
    assert all(path.read_bytes() == content for path, content in preserved.items())


@pytest.mark.parametrize("system", ["Windows", "Linux", "Darwin"])
def test_first_cpu_install_uses_cpu_channel_except_apple_mps(monkeypatch, tmp_path, system):
    fresh_torch(monkeypatch, "cpu")
    monkeypatch.setattr(boot.platform, "system", lambda: system)
    monkeypatch.setattr(boot, "VENV", tmp_path / "venv")
    monkeypatch.setattr(boot, "MARKER", tmp_path / "venv" / "marker.json")
    boot.venv_python().parent.mkdir(parents=True)
    boot.venv_python().touch()
    monkeypatch.setattr(boot, "uv_path", lambda: None)
    commands = []
    monkeypatch.setattr(
        boot.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command) or type("Result", (), {"returncode": 0})(),
    )
    boot.ensure_venv("cpu", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    torch_command = next(command for command in commands if "torch>=2.4" in command)
    source = torch_command[torch_command.index("--index-url") + 1]
    assert source == (boot.PYPI_OFFICIAL if system == "Darwin" else "https://download.pytorch.org/whl/cpu")


def test_dependency_health_probe_checks_selected_extras_and_transitive_metadata(monkeypatch, tmp_path):
    def distribution(name, requirements):
        root = tmp_path / f"{name}-1.0.dist-info"
        root.mkdir()
        (root / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
            + "".join(f"Requires-Dist: {value}\n" for value in requirements)
        )

    distribution(
        "ypuddin",
        ["pipeline-helper>=1", "missing-optional; extra == 'optim'", "never-required; extra == 'cuda'"],
    )
    distribution("pipeline_helper", ["missing-transitive>=2"])
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    # This fixture intentionally injects synthetic metadata; production probes strip PYTHONPATH.
    monkeypatch.setattr(boot, "_env", lambda: dict(os.environ))
    monkeypatch.setattr(boot, "venv_python", lambda: Path(sys.executable))
    issues = boot.dependency_issues("optim")
    assert any("missing-optional" in issue for issue in issues)
    assert any("missing-transitive" in issue for issue in issues)
    assert not any("never-required" in issue for issue in issues)


def test_needs_copy_link_mode():
    assert boot.needs_copy_link_mode(100, 200) is True  # uv cache and project on different filesystems
    assert boot.needs_copy_link_mode(100, 100) is False
    assert boot.needs_copy_link_mode(None, 100) is False  # unknown -> leave uv's default behaviour
    assert boot.needs_copy_link_mode(100, None) is False


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("v18.20.8", False),
        ("v20.18.9", False),
        ("v20.19.0", True),
        ("v21.9.0", False),
        ("v22.11.0", False),
        ("v22.12.0", True),
        ("v24.0.0", True),
    ],
)
def test_node_runtime_matches_vite(version, supported):
    assert boot.node_supported(version) == supported


def test_saved_server_address_and_explicit_override(tmp_path):
    import json

    (tmp_path / "settings.json").write_text(json.dumps({"server": {"host": "0.0.0.0", "port": 9123}}))
    assert boot.server_address(None, None, str(tmp_path)) == ("0.0.0.0", 9123)
    assert boot.server_address("127.0.0.1", 9133, str(tmp_path)) == ("127.0.0.1", 9133)


def _frontend_manifest(root):
    import hashlib
    import json

    (root / "dist").mkdir(exist_ok=True)
    (root / "dist/index.html").write_text("built")
    inputs = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and "dist" not in p.relative_to(root).parts
    }
    outputs = {"index.html": hashlib.sha256(b"built").hexdigest()}
    (root / "dist/.source-manifest.json").write_text(
        json.dumps({"version": 1, "inputs": inputs, "outputs": outputs})
    )


def test_frontend_fingerprint_survives_timestamps_but_detects_all_build_inputs(monkeypatch, tmp_path):
    import os

    monkeypatch.setattr(boot, "FRONTEND", tmp_path)
    (tmp_path / "src").mkdir()
    for name in ("src/App.tsx", "package-lock.json", "tailwind.config.ts", "tsconfig.json"):
        (tmp_path / name).write_text("before")
    _frontend_manifest(tmp_path)
    assert boot.frontend_stale() is False
    for name in ("src/App.tsx", "package-lock.json", "tailwind.config.ts", "tsconfig.json"):
        p = tmp_path / name
        p.write_text("after")
        os.utime(p, (1, 1))
        assert boot.frontend_stale() is True
        p.write_text("before")
        assert boot.frontend_stale() is False
    (tmp_path / "src/App.tsx").unlink()
    assert boot.frontend_stale() is True


def test_frontend_missing_manifest_or_corrupt_output_requires_build(monkeypatch, tmp_path):
    monkeypatch.setattr(boot, "FRONTEND", tmp_path)
    (tmp_path / "src").mkdir()
    _frontend_manifest(tmp_path)
    (tmp_path / "dist/index.html").write_text("old UI")
    assert boot.frontend_stale() is True
    _frontend_manifest(tmp_path)
    (tmp_path / "dist/.source-manifest.json").unlink()
    assert boot.frontend_stale() is True


def test_verified_frontend_does_not_need_node_to_launch(monkeypatch):
    monkeypatch.setattr(boot, "frontend_stale", lambda: False)
    monkeypatch.setattr(boot.shutil, "which", lambda _: None)
    assert boot.build_frontend() is True


@pytest.mark.parametrize("invalid", ["[]", '{"version": 1, "inputs": {}, "outputs": ["index.html"]}'])
def test_malformed_build_manifest_is_stale(monkeypatch, tmp_path, invalid):
    monkeypatch.setattr(boot, "FRONTEND", tmp_path)
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist/index.html").write_text("built")
    (tmp_path / "dist/.source-manifest.json").write_text(invalid)
    assert boot.frontend_stale() is True


@pytest.mark.parametrize(
    "profile,system,machine,expected",
    [
        ("macos-mps", "Darwin", "arm64", "cpu"),
        ("windows-cuda", "Windows", "AMD64", "cu128"),
        ("linux-cuda", "Linux", "x86_64", "cu128"),
        ("linux-dtk", "Linux", "x86_64", "dtk"),
        ("cpu", "Linux", "x86_64", "cpu"),
    ],
)
def test_platform_profile_selects_matching_install(monkeypatch, profile, system, machine, expected):
    monkeypatch.setattr(boot.platform, "system", lambda: system)
    monkeypatch.setattr(boot.platform, "machine", lambda: machine)
    monkeypatch.setattr(boot, "nvidia_driver_major", lambda: 575)
    monkeypatch.setattr(boot, "nvidia_gpus", lambda: [("test GPU", 12.0)])
    assert boot.platform_torch_tag(profile, "auto") == expected


@pytest.mark.parametrize(
    "profile,requested",
    [
        ("windows-cuda", "auto"),
        ("linux-cuda", "auto"),
        ("linux-dtk", "auto"),
        ("macos-mps", "cu128"),
        ("cpu", "cu128"),
        ("other", "auto"),
    ],
)
def test_wrong_platform_profile_never_starts_install(monkeypatch, profile, requested):
    monkeypatch.setattr(boot.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(boot.platform, "machine", lambda: "arm64")
    with pytest.raises(SystemExit):
        boot.platform_torch_tag(profile, requested)


def test_cuda_profile_rejects_missing_nvidia_driver(monkeypatch):
    monkeypatch.setattr(boot.platform, "system", lambda: "Windows")
    monkeypatch.setattr(boot, "nvidia_driver_major", lambda: None)
    with pytest.raises(SystemExit):
        boot.platform_torch_tag("windows-cuda", "auto")


@pytest.mark.parametrize(
    "system,cuda,cpu",
    [
        ("Windows", "windows-cuda", "windows-cpu"),
        ("Linux", "linux-cuda", "linux-cpu"),
        ("Darwin", "macos-mps", "macos-cpu"),
    ],
)
def test_platform_directories_are_distinct_and_preserve_legacy(monkeypatch, system, cuda, cpu):
    monkeypatch.setattr(boot.platform, "system", lambda: system)
    legacy = boot.ROOT / "venv"
    legacy.mkdir()
    (legacy / "keep.txt").write_text("legacy packages")
    assert boot.select_environment("auto", "cpu") == "legacy"
    assert boot.VENV == legacy
    assert boot.select_environment(cuda, "cu128") == cuda
    cuda_path = boot.VENV
    assert boot.select_environment("cpu", "cpu") == cpu
    assert boot.VENV != cuda_path and legacy not in boot.VENV.parents
    assert boot.VENV == boot.ROOT / "environment" / cpu / "venv"
    assert (legacy / "keep.txt").read_text() == "legacy packages"


def test_two_real_profile_venvs_do_not_share_packages(monkeypatch):
    # Actual stdlib venvs and child interpreters, with no package download or Torch install.
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    paths = []
    for profile, module in [("linux-cuda", "cuda_only_canary"), ("cpu", "cpu_only_canary")]:
        boot.select_environment(profile, "cpu")
        venv.EnvBuilder(with_pip=False).create(boot.VENV)
        py = boot.venv_python()
        site = Path(
            subprocess.check_output(
                [str(py), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
            ).strip()
        )
        (site / (module + ".py")).write_text("VALUE = '" + boot.PROFILE + "'\n")
        paths.append((py, module, boot.PROFILE))
    for py, module, profile in paths:
        other = "cpu_only_canary" if module == "cuda_only_canary" else "cuda_only_canary"
        code = f"import {module}, importlib.util; print({module}.VALUE); assert importlib.util.find_spec('{other}') is None"
        result = subprocess.run([str(py), "-I", "-c", code], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == profile


def test_rebuild_only_changes_selected_profile(monkeypatch):
    fresh_torch(monkeypatch, "cpu")
    other = boot.ROOT / "environment/linux-cuda/venv/keep.txt"
    legacy = boot.ROOT / "venv/keep.txt"
    for file in (other, legacy):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("unchanged")
    boot.select_environment("cpu", "cpu")
    boot.VENV.mkdir(parents=True)
    (boot.VENV / "old.txt").write_text("selected old env")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "venv" in command:
            boot.venv_python().parent.mkdir(parents=True)
            boot.venv_python().touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    monkeypatch.setattr(boot, "find_base_python", lambda: "/base/python")
    monkeypatch.setattr(boot, "uv_path", lambda: None)
    boot.ensure_venv("cpu", index_mode="official", reinstall=True, extras=boot.EXTRAS_BASE)
    assert not (boot.VENV / "old.txt").exists()
    assert json.loads(boot.MARKER.read_text())["profile"] == "linux-cpu"
    assert all(file.read_text() == "unchanged" for file in (other, legacy))
    assert all(command[0] == str(boot.venv_python()) for command in calls if "install" in command)
    assert not any("nvidia" in str(command) for command in calls)


def test_profile_rejects_foreign_marker_and_symlink_before_mutation(monkeypatch):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    boot.select_environment("cpu", "cpu")
    boot.VENV.mkdir(parents=True)
    boot.MARKER.write_text(json.dumps({"profile": "linux-cuda"}))
    with pytest.raises(SystemExit):
        boot.select_environment("cpu", "cpu")
    boot.MARKER.unlink()
    boot.VENV.rmdir()
    target = boot.ROOT / "untouched"
    target.mkdir()
    boot.VENV.symlink_to(target, target_is_directory=True)
    with pytest.raises(SystemExit):
        boot.select_environment("cpu", "cpu")
    assert target.is_dir()


def test_cpu_profile_child_environment_blocks_package_and_device_leak(monkeypatch):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    monkeypatch.setattr(boot, "nvidia_driver_major", lambda: 580)
    monkeypatch.setenv("PYTHONPATH", "/another/environment/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/another/environment")
    boot.select_environment("cpu", "cpu")
    env = boot._env()
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert env["PYTHONNOUSERSITE"] == "1" and env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert env["YPUDDIN_ENV_PROFILE"] == "linux-cpu"
    assert "nvidia" not in boot.choose_extras("cpu")


@pytest.mark.parametrize("requested", ["cpu", "cu128", "cu126"])
def test_dtk_profile_rejects_foreign_torch_channels(monkeypatch, requested):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    with pytest.raises(SystemExit):
        boot.platform_torch_tag("linux-dtk", requested)


def test_dtk_profile_never_uses_nvidia_or_official_torch_sources(monkeypatch):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    monkeypatch.setattr(boot, "nvidia_driver_major", lambda: 580)
    boot.select_environment("linux-dtk", "dtk")
    assert boot.VENV == boot.ROOT / "environment/linux-dtk/venv"
    assert "nvidia" not in boot.choose_extras("dtk")
    assert boot.index_chains("official", "dtk")[1] == []


@pytest.mark.parametrize("reinstall", [False, True])
def test_dtk_missing_vendor_wheels_never_creates_or_deletes_environment(monkeypatch, reinstall):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    boot.select_environment("linux-dtk", "dtk")
    if reinstall:
        boot.venv_python().parent.mkdir(parents=True)
        boot.venv_python().write_text("vendor interpreter")
    monkeypatch.setattr(boot, "run", lambda *a, **k: pytest.fail("installer must not run"))
    with pytest.raises(SystemExit):
        boot.ensure_venv("dtk", index_mode="official", reinstall=reinstall, extras=boot.EXTRAS_BASE)
    assert boot.VENV.exists() == reinstall
    if reinstall:
        assert boot.venv_python().read_text() == "vendor interpreter"


def test_dtk_existing_runtime_is_pinned_while_ordinary_dependencies_are_installed(monkeypatch):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    boot.select_environment("linux-dtk", "dtk")
    boot.venv_python().parent.mkdir(parents=True)
    boot.venv_python().touch()
    versions = {
        "torch": "2.4.1+das.opt1.dtk25041",
        "torchvision": "0.19.1+das.opt1.dtk25041",
        "triton": "3.0.0+das.opt1.dtk25041",
        "hip-runtime": "6.2",
    }
    monkeypatch.setattr(boot, "installed_versions", lambda: versions.copy())
    monkeypatch.setattr(
        boot, "torch_runtime", lambda: {"version": versions["torch"], "cuda": None, "hip": "6.2"}
    )
    monkeypatch.setattr(boot, "dependency_issues", lambda extras: [])
    monkeypatch.setattr(boot, "editable_install_ready", lambda: True)
    monkeypatch.setattr(boot, "uv_path", lambda: None)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if "--constraint" in command:
            pins = Path(command[command.index("--constraint") + 1]).read_text()
            assert all(f"{name}=={version}\n" in pins for name, version in versions.items())
            assert "transformers<5\n" in pins
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(boot.subprocess, "run", execute)
    boot.ensure_venv("dtk", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert all("torch>=2.4" not in command for command in calls)
    assert not any("download.pytorch.org" in str(command) or ",nvidia" in str(command) for command in calls)
    assert json.loads(boot.MARKER.read_text())["profile"] == "linux-dtk"


def test_dtk_matching_marker_repairs_transformers_backend_incompatibility(monkeypatch, tmp_path):
    versions = existing_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    versions.clear()
    versions.update(
        torch="2.4.1+das.opt1.dtk25041",
        torchvision="0.19.1+das.opt1.dtk25041",
        transformers="5.17.0",
        triton="3.0.0+das.opt1.dtk25041",
    )
    monkeypatch.setattr(
        boot, "torch_runtime", lambda: {"version": versions["torch"], "cuda": None, "hip": "5.6"}
    )
    boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("dtk", boot.EXTRAS_BASE)}))
    commands = []

    def install(command, **kwargs):
        commands.append(command)
        if "--constraint" in command:
            constraints = Path(command[command.index("--constraint") + 1]).read_text()
            assert "transformers<5\n" in constraints
            assert "triton==3.0.0+das.opt1.dtk25041\n" in constraints
            versions["transformers"] = "4.57.6"
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(boot.subprocess, "run", install)
    boot.ensure_venv("dtk", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert any("-e" in command for command in commands)
    assert versions["torch"] == "2.4.1+das.opt1.dtk25041"
    commands.clear()
    boot.ensure_venv("dtk", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert commands == []


@pytest.mark.parametrize(
    "profile,torch", [("linux-cuda", "2.4.1+cu124"), ("linux-dtk", "2.5.1+das.opt1.dtk25041")]
)
def test_dtk_torch24_transformers_constraint_does_not_leak_to_other_runtimes(monkeypatch, profile, torch):
    monkeypatch.setattr(boot, "PROFILE", profile)
    assert boot.dtk_compatibility_constraints({"torch": torch}) == []
    assert boot.dtk_compatibility_issues({"torch": torch, "transformers": "5.17.0"}) == []


@pytest.mark.parametrize(
    "runtime", [{"version": "2.5.1", "cuda": "12.8"}, {"version": "2.5.1", "cuda": None}]
)
def test_dtk_rejects_cpu_or_cuda_runtime_before_installing(monkeypatch, runtime):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    boot.select_environment("linux-dtk", "dtk")
    boot.venv_python().parent.mkdir(parents=True)
    boot.venv_python().touch()
    monkeypatch.setattr(
        boot, "installed_versions", lambda: {"torch": runtime["version"], "torchvision": "0.20.1"}
    )
    monkeypatch.setattr(boot, "torch_runtime", lambda: runtime)
    monkeypatch.setattr(boot.subprocess, "run", lambda *a, **k: pytest.fail("installer must not run"))
    with pytest.raises(SystemExit):
        boot.ensure_venv("dtk", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)


@pytest.mark.parametrize("layout", ["dtk25041", "dtk2604"])
@pytest.mark.parametrize("inherit_paths", [False, True])
def test_dtk_launcher_keeps_library_environment_local_and_never_uses_legacy(tmp_path, layout, inherit_paths):
    # Execute the real shell entry with a fake interpreter, without any vendor installation.
    script = tmp_path / "studio-linux-dtk.sh"
    script.write_bytes((SOURCE_ROOT / script.name).read_bytes())
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uname = binaries / "uname"
    uname.write_text("#!/bin/sh\nprintf Linux")
    uname.chmod(0o755)
    dtk = tmp_path / "vendor dtk"
    directories = ["lib", "hip/lib", "lib64", ".hyhal/lib", "bin", "llvm/bin", "hip/bin"]
    if layout == "dtk2604":
        directories += [
            "llvm/lib",
            "dcc/lib",
            "dcc/gcvm/lib",
            "dcc/comgr/lib",
            "dcc/bin",
            "dushmem/lib",
            "opencl/lib",
            ".hyhal/lib64",
            ".hyhal/rocm_smi/lib",
            "include",
            "llvm/include",
            "dcc/gcvm/include",
            "dushmem/include",
            "opencl/include",
        ]
    for directory in directories:
        (dtk / directory).mkdir(parents=True)
    # The launcher reads no vendor shell code and does not alter the caller's files.
    sentinel = tmp_path / "vendor-script-was-sourced"
    (dtk / "env.sh").write_text(f"touch '{sentinel}'\n")
    python = tmp_path / "environment/linux-dtk/venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" "${LD_LIBRARY_PATH:-}" "$PYTHONNOUSERSITE" '
        '"${PYTHONPATH:-cleared}" "$DTK_ROOT" "$DTKROOT" "$ROCM_PATH" "$HIP_PATH" '
        '"${LIBRARY_PATH:-}" "$PATH" "${C_INCLUDE_PATH:-}" "${CPLUS_INCLUDE_PATH:-}"'
    )
    python.chmod(0o755)
    caller_env = {
        **os.environ,
        "PATH": str(binaries) + ":" + os.environ["PATH"],
        "DTK_ROOT": str(dtk),
        "PYTHONPATH": "/foreign/site",
        "ROCM_PATH": "/foreign/runtime",
        "HIP_PATH": "/foreign/runtime/hip",
    }
    for variable in ("LD_LIBRARY_PATH", "LIBRARY_PATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        if inherit_paths:
            caller_env[variable] = f"/existing/{variable.lower()}"
        else:
            caller_env.pop(variable, None)
    before = caller_env.copy()
    result = subprocess.run(
        ["bash", str(script), "doctor"],
        env=caller_env,
        text=True,
        capture_output=True,
        check=True,
    )
    arguments = result.stdout.splitlines()
    assert arguments[:3] == ["scripts/bootstrap.py", "--profile=linux-dtk", "doctor"]
    ld_path, user_site, foreign_path, selected, dtkroot, rocm, hip, linker, path, c_path, cpp_path = (
        arguments[3:]
    )
    assert (user_site, foreign_path) == ("1", "cleared")
    assert (selected, dtkroot, rocm, hip) == (str(dtk), str(dtk), str(dtk), str(dtk / "hip"))
    for variable, value, suffixes in (
        ("LD_LIBRARY_PATH", ld_path, ("/lib", "/lib64", "lib", "lib64")),
        ("LIBRARY_PATH", linker, ("/lib", "/lib64", "lib", "lib64")),
        ("PATH", path, ("/bin", "bin")),
        ("C_INCLUDE_PATH", c_path, ("/include", "include")),
        ("CPLUS_INCLUDE_PATH", cpp_path, ("/include", "include")),
    ):
        entries = value.split(":") if value else []
        assert "" not in entries  # No implicit current-directory search paths.
        for directory in directories:
            if directory.endswith(suffixes):
                assert str(dtk / directory) in entries
        if variable in caller_env:
            assert value.endswith(caller_env[variable])
        for entry in entries:
            if entry.startswith(str(dtk)):
                assert Path(entry).is_dir()
    assert ld_path.split(":")[0] == str(dtk / "lib")
    assert str(dtk / "dcc/lib") in ld_path if layout == "dtk2604" else str(dtk / "dcc/lib") not in ld_path
    assert caller_env == before and not sentinel.exists()


def test_dtk_vendor_wheel_failure_never_falls_back_to_online_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    boot.select_environment("linux-dtk", "dtk")
    boot.venv_python().parent.mkdir(parents=True)
    boot.venv_python().touch()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    native_wheel(wheels, "torch", "2.5.1+das.opt1.dtk25041")
    native_wheel(wheels, "torchvision", "0.20.1+das.opt1.dtk25041")
    monkeypatch.setattr(boot, "installed_versions", lambda: {})
    monkeypatch.setattr(boot, "uv_path", lambda: None)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(boot.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        boot.ensure_venv(
            "dtk", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE, dtk_wheelhouse=str(wheels)
        )
    assert len(calls) == 1
    assert "--no-index" in calls[0] and "--index-url" not in calls[0]
    assert str(wheels) == calls[0][calls[0].index("--find-links") + 1]
    assert not boot.MARKER.exists()


def dtk_wheelhouse(tmp_path, triton_version=None):
    wheels = tmp_path / "vendor-wheels"
    wheels.mkdir()
    native_wheel(wheels, "torch", boot.DTK_NUMPY1_TORCH)
    native_wheel(wheels, "torchvision", "0.22.0+das.opt1.dtk2604.torch271")
    if triton_version:
        wheel = wheels / f"triton-{triton_version}-cp311-cp311-manylinux_2_28_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                f"triton-{triton_version}.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: triton\nVersion: {triton_version}\nRequires-Dist: torch==2.7.1\n",
            )
    return wheels


@pytest.mark.parametrize(
    "profile,version,expected",
    [
        ("linux-dtk", boot.DTK_NUMPY1_TORCH, ["numpy>=1.26,<2"]),
        ("linux-cuda", boot.DTK_NUMPY1_TORCH, []),
        ("linux-dtk", "2.7.1+das.opt1.dtk2604.other", []),
        ("linux-dtk", "2.7.1+das.opt2.dtk2604", []),
        ("linux-dtk", "2.8.0+das.opt1.dtk2604", []),
        ("linux-dtk", "2.7.1", []),
    ],
)
def test_dtk_numpy_constraint_is_limited_to_the_observed_vendor_build(
    monkeypatch, profile, version, expected
):
    monkeypatch.setattr(boot, "PROFILE", profile)
    assert boot.dtk_compatibility_constraints({"torch": version}) == expected
    issues = boot.dtk_compatibility_issues({"torch": version, "numpy": "2.4.6", "transformers": "5.17.0"})
    assert bool(issues) == bool(expected)
    assert all("NumPy" in issue and "transformers" not in issue for issue in issues)


@pytest.mark.parametrize("use_uv", [False, True])
@pytest.mark.parametrize("selected", [boot.DTK_NUMPY1_TORCH, "2.8.0+das.opt1.dtk2604"])
def test_dtk_numpy_constraint_follows_the_resolved_candidate_not_any_wheel(
    monkeypatch, tmp_path, use_uv, selected
):
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    wheels = dtk_wheelhouse(tmp_path)
    native_wheel(wheels, "torch", "2.8.0+das.opt1.dtk2604", "cp312-cp312-manylinux_2_28_x86_64")
    pins = tmp_path / "constraints.txt"
    pins.write_text("")
    commands = []

    def resolve(command, **kwargs):
        commands.append(command)
        assert "--no-index" in command and "--index-url" not in command
        assert command[command.index("--find-links") + 1] == str(wheels)
        assert native_plan(command, selected)
        assert "compile" in command if use_uv else "--dry-run" in command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(boot, "run", resolve)
    result = boot.dtk_native_numpy_constraints(
        wheels,
        {},
        uv="uv" if use_uv else None,
        py="/target/python",
        constraints=pins,
        requirements=["torch>=2.4", "torchvision>=0.19"],
    )
    assert result == [
        f"torch=={selected}",
        *(["numpy>=1.26,<2"] if selected == boot.DTK_NUMPY1_TORCH else []),
    ]
    assert len(commands) == 1


@pytest.mark.parametrize("use_uv", [False, True])
@pytest.mark.parametrize("selected", [boot.DTK_NUMPY1_TORCH, "2.8.0+das.opt1.dtk2604"])
def test_dtk_real_offline_resolver_selects_dependency_compatible_torch(
    monkeypatch, tmp_path, use_uv, selected
):
    uv = shutil.which("uv") if use_uv else None
    if use_uv and not uv:
        pytest.skip("uv is unavailable")
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_NO_MANAGED_PYTHON", "1")
    target = tmp_path / "resolver-venv"
    venv.EnvBuilder(with_pip=not use_uv).create(target)
    py = str(target / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    wheels = tmp_path / "offline-wheels"
    wheels.mkdir()
    for version in [boot.DTK_NUMPY1_TORCH, "2.8.0+das.opt1.dtk2604"]:
        native_wheel(wheels, "torch", version, "py3-none-any")
    native_wheel(wheels, "torchvision", "0.22.0", "py3-none-any", [f"torch=={selected}"])
    pins = tmp_path / "constraints.txt"
    pins.write_text("")
    result = boot.dtk_native_numpy_constraints(
        wheels, {}, uv=uv, py=py, constraints=pins, requirements=["torch>=2.4", "torchvision>=0.19"]
    )
    assert result == [
        f"torch=={selected}",
        *(["numpy>=1.26,<2"] if selected == boot.DTK_NUMPY1_TORCH else []),
    ]
    # Planning must not install even our harmless metadata-only placeholder packages.
    code = "import importlib.util; assert importlib.util.find_spec('torch') is None"
    subprocess.run([py, "-c", code], check=True)


@pytest.mark.parametrize("use_uv", [False, True])
def test_dtk_empty_environment_constrains_numpy_before_first_native_install(monkeypatch, tmp_path, use_uv):
    versions = existing_environment(monkeypatch, tmp_path)
    versions.clear()
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    monkeypatch.setattr(boot, "uv_path", lambda: "uv" if use_uv else None)
    monkeypatch.setattr(boot, "uv_cache_dir", lambda _: tmp_path)
    native = {
        "torch": boot.DTK_NUMPY1_TORCH,
        "torchvision": "0.22.0+das.opt1.dtk2604.torch271",
        "numpy": "1.26.4",
    }
    monkeypatch.setattr(
        boot, "torch_runtime", lambda: {"version": native["torch"], "cuda": None, "hip": "6.3"}
    )
    wheels = dtk_wheelhouse(tmp_path)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if native_plan(command):
            assert not versions
            return subprocess.CompletedProcess(command, 0)
        if "--constraint" in command:
            pins = Path(command[command.index("--constraint") + 1]).read_text()
            assert "numpy>=1.26,<2\n" in pins
            assert f"torch=={boot.DTK_NUMPY1_TORCH}\n" in pins
            if "--no-index" in command:
                assert not versions
                versions.update(native)
            else:
                assert "numpy==1.26.4\n" in pins
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(boot.subprocess, "run", execute)
    boot.ensure_venv(
        "dtk", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE, dtk_wheelhouse=str(wheels)
    )
    assert versions == native and boot.MARKER.exists()
    assert sum("--dry-run" in command or "compile" in command for command in calls) == 1


def test_dtk_selected_affected_wheel_diagnoses_preexisting_numpy2_before_install(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    wheels = dtk_wheelhouse(tmp_path)
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("numpy==2.4.6\n")

    def resolve(command, **kwargs):
        assert native_plan(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(boot, "run", resolve)
    with pytest.raises(SystemExit):
        boot.dtk_native_numpy_constraints(
            wheels,
            {"numpy": "2.4.6"},
            uv=None,
            py="/target/python",
            constraints=constraints,
            requirements=["torch>=2.4", "torchvision>=0.19"],
        )
    assert "NumPy 2.4.6" in capsys.readouterr().err
    assert constraints.read_text() == "numpy==2.4.6\n"


@pytest.mark.parametrize("ready", [False, True])
def test_dtk_existing_numpy2_is_diagnosed_without_package_changes(monkeypatch, tmp_path, ready, capsys):
    versions = existing_environment(monkeypatch, tmp_path)
    versions.clear()
    versions.update(
        torch=boot.DTK_NUMPY1_TORCH, numpy="2.4.6", torchvision="0.22.0+das.opt1.dtk2604.torch271"
    )
    before = versions.copy()
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    if ready:
        boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("dtk", boot.EXTRAS_BASE)}))
    monkeypatch.setattr(
        boot.subprocess,
        "run",
        lambda *a, **k: pytest.fail("must not invoke installer or replace existing packages"),
    )
    with pytest.raises(SystemExit):
        boot.ensure_venv("dtk", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert versions == before
    assert "NumPy 2.4.6" in capsys.readouterr().err
    assert boot.MARKER.exists() == ready


def test_dtk_ready_marker_requires_actual_numpy_bridge_health(monkeypatch, tmp_path, capsys):
    versions = existing_environment(monkeypatch, tmp_path)
    versions.clear()
    versions.update(torch="2.8.0+das.opt1.dtk2604", numpy="2.4.6")
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    monkeypatch.setattr(
        boot, "torch_runtime", lambda: {"version": versions["torch"], "hip": "6.3", "cuda": None}
    )
    monkeypatch.setattr(
        boot, "torch_numpy_bridge", lambda: {"ok": False, "error": "RuntimeError: Numpy is not available"}
    )
    boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("dtk", boot.EXTRAS_BASE)}))
    monkeypatch.setattr(boot.subprocess, "run", lambda *a, **k: pytest.fail("must not install packages"))
    with pytest.raises(SystemExit):
        boot.ensure_venv("dtk", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE)
    assert "CPU 桥接检查失败" in capsys.readouterr().err
    assert versions["numpy"] == "2.4.6"


def test_dtk_bridge_probe_exercises_real_cpu_numpy_round_trip(monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    monkeypatch.setattr(
        boot,
        "venv_json",
        lambda code: json.loads(subprocess.check_output([sys.executable, "-c", code], text=True)),
    )
    assert REAL_TORCH_NUMPY_BRIDGE() == {"ok": True}


def test_dtk_bridge_probe_rejects_import_warning_without_working_numpy_bridge(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    (tmp_path / "torch.py").write_text(
        "import warnings\nwarnings.warn('NumPy ABI unavailable')\ndef from_numpy(value):\n    raise RuntimeError('Numpy is not available')\n"
    )
    monkeypatch.setattr(
        boot,
        "venv_json",
        lambda code: json.loads(
            subprocess.check_output([sys.executable, "-c", code], text=True, cwd=tmp_path)
        ),
    )
    result = REAL_TORCH_NUMPY_BRIDGE()
    assert result == {"ok": False, "error": "RuntimeError: Numpy is not available"}


@pytest.mark.parametrize("use_uv", [False, True])
@pytest.mark.parametrize("initial", ["empty", "torch-ready", "triton-ready"])
def test_dtk_local_triton_is_installed_and_pinned_even_after_ready_marker(
    monkeypatch, tmp_path, use_uv, initial
):
    versions = existing_environment(monkeypatch, tmp_path)
    versions.clear()
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    native = {"torch": "2.7.1+das.opt1.dtk2604", "torchvision": "0.22.0+das.opt1.dtk2604.torch271"}
    supplied_triton = "3.1.0+das.opt1.dtk2604.torch271"
    if initial != "empty":
        versions.update(native)
        versions["numpy"] = "1.26.4"
        boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("dtk", boot.EXTRAS_BASE)}))
    if initial == "triton-ready":
        # A supplied newer wheel must not cause a working native stack to upgrade.
        versions["triton"] = "3.0.0+das.opt1.dtk2604.torch271"
    before = versions.copy()
    wheels = dtk_wheelhouse(tmp_path, supplied_triton)
    monkeypatch.setattr(boot, "uv_path", lambda: "uv" if use_uv else None)
    monkeypatch.setattr(boot, "uv_cache_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        boot, "torch_runtime", lambda: {"version": native["torch"], "cuda": None, "hip": "6.3"}
    )
    commands = []

    def install(command, **kwargs):
        if native_plan(command):
            return subprocess.CompletedProcess(command, 0)
        commands.append(command)
        if "--constraint" in command:
            pins = Path(command[command.index("--constraint") + 1]).read_text()
            assert all(f"{name}=={version}\n" in pins for name, version in before.items())
            if "--no-index" in command:
                assert "triton" in command and "--index-url" not in command
                assert command[command.index("--find-links") + 1] == str(wheels)
                versions.update(native, triton=supplied_triton)
            else:
                assert f"triton=={supplied_triton}\n" in pins
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(boot.subprocess, "run", install)
    options = dict(
        index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE, dtk_wheelhouse=str(wheels)
    )
    boot.ensure_venv("dtk", **options)
    assert sum("--no-index" in command for command in commands) == (initial != "triton-ready")
    assert all(versions[name] == version for name, version in before.items())
    assert versions["triton"] == before.get("triton", supplied_triton)
    assert all("download.pytorch.org" not in " ".join(command) for command in commands)
    # Once the supplement is present, the same invocation needs no installer/network.
    commands.clear()
    boot.ensure_venv("dtk", **options)
    assert commands == []


@pytest.mark.parametrize("torch_present", [False, True])
def test_dtk_without_supplied_triton_keeps_sdpa_environment_optional(monkeypatch, tmp_path, torch_present):
    versions = existing_environment(monkeypatch, tmp_path)
    versions.clear()
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    native = {"torch": "2.7.1+das.opt1.dtk2604", "torchvision": "0.22.0+das.opt1.dtk2604.torch271"}
    if torch_present:
        versions.update(native)
        boot.MARKER.write_text(json.dumps({"signature": boot.install_signature("dtk", boot.EXTRAS_BASE)}))
    wheels = dtk_wheelhouse(tmp_path)
    monkeypatch.setattr(
        boot, "torch_runtime", lambda: {"version": native["torch"], "cuda": None, "hip": "6.3"}
    )
    commands = []

    def install(command, **kwargs):
        if native_plan(command):
            return subprocess.CompletedProcess(command, 0)
        commands.append(command)
        assert "triton" not in command
        if "--no-index" in command:
            versions.update(native)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(boot.subprocess, "run", install)
    boot.ensure_venv(
        "dtk", index_mode="official", reinstall=False, extras=boot.EXTRAS_BASE, dtk_wheelhouse=str(wheels)
    )
    assert "triton" not in versions
    assert sum("--no-index" in command for command in commands) == (not torch_present)
    assert boot.MARKER.exists()


@pytest.mark.parametrize("invalid", ["plain", "renamed", "corrupt"])
def test_dtk_triton_must_have_vendor_metadata_before_any_environment_mutation(monkeypatch, tmp_path, invalid):
    versions = existing_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(boot, "PROFILE", "linux-dtk")
    supplied = "3.1.0" if invalid in {"plain", "renamed"} else "3.1.0+das.opt1.dtk2604"
    wheels = dtk_wheelhouse(tmp_path, supplied)
    wheel = next(wheels.glob("triton-*.whl"))
    if invalid == "renamed":
        wheel.rename(wheel.with_name(wheel.name.replace("3.1.0", "3.1.0+das.opt1.dtk2604")))
    elif invalid == "corrupt":
        wheel.write_bytes(b"not a wheel")
    before = versions.copy()
    monkeypatch.setattr(boot.subprocess, "run", lambda *a, **k: pytest.fail("must fail before installation"))
    with pytest.raises(SystemExit):
        boot.ensure_venv(
            "dtk", index_mode="official", reinstall=True, extras=boot.EXTRAS_BASE, dtk_wheelhouse=str(wheels)
        )
    assert boot.venv_python().exists() and versions == before


def test_dtk_without_ensurepip_targets_only_new_venv_with_local_pip(monkeypatch, tmp_path):
    wheelhouse = tmp_path / "vendor wheels"
    wheelhouse.mkdir()
    (wheelhouse / "pip-25.1-py3-none-any.whl").touch()
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if "ensurepip" in command:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="No module named ensurepip")
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, stdout="--python <python> Run pip in target")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(boot.subprocess, "run", execute)
    boot.create_dtk_venv("/usr/bin/python3", wheelhouse)
    assert calls[2] == ["/usr/bin/python3", "-m", "venv", "--without-pip", str(boot.VENV)]
    seed = calls[3]
    assert seed[:6] == ["/usr/bin/python3", "-m", "pip", "--isolated", "--python", str(boot.venv_python())]
    assert "--no-index" in seed and "--no-deps" in seed and seed[-1] == "pip"
    assert seed[seed.index("--find-links") + 1] == str(wheelhouse)
    assert calls[4] == [str(boot.venv_python()), "-m", "pip", "--version"]
    assert all("--upgrade" not in command and "torch" not in command for command in calls)


@pytest.mark.parametrize("local_wheel,host_support", [(False, True), (True, False)])
def test_dtk_missing_pip_bootstrap_prerequisites_do_not_create_environment(
    monkeypatch, tmp_path, capsys, local_wheel, host_support
):
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    if local_wheel:
        (wheelhouse / "pip-25.1-py3-none-any.whl").touch()
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert "venv" not in command and "install" not in command
        return subprocess.CompletedProcess(
            command,
            1 if "ensurepip" in command else 0,
            stdout="--python" if host_support else "old pip without target option",
            stderr="",
        )

    monkeypatch.setattr(boot.subprocess, "run", execute)
    with pytest.raises(SystemExit):
        boot.create_dtk_venv("/usr/bin/python3", wheelhouse)
    assert not boot.VENV.exists()
    assert "ensurepip" in capsys.readouterr().err
    assert len(calls) == (2 if local_wheel else 1)


def test_dtk_with_working_ensurepip_keeps_normal_venv_creation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        boot.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or subprocess.CompletedProcess(command, 0),
    )
    boot.create_dtk_venv("/python", tmp_path)
    assert calls == [["/python", "-m", "ensurepip", "--version"], ["/python", "-m", "venv", str(boot.VENV)]]


def test_failed_dtk_pip_seed_removes_only_its_own_new_environment(monkeypatch, tmp_path):
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    (wheelhouse / "pip-25.1-py3-none-any.whl").touch()
    other = boot.ROOT / "environment/linux-cuda/venv"
    other.mkdir(parents=True)
    sentinel = other / "keep.txt"
    sentinel.write_text("running environment")

    def execute(command, **kwargs):
        if "ensurepip" in command:
            return subprocess.CompletedProcess(command, 1)
        if "--help" in command:
            return subprocess.CompletedProcess(command, 0, stdout="--python")
        if "venv" in command:
            boot.VENV.mkdir()
            return subprocess.CompletedProcess(command, 0)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(boot.subprocess, "run", execute)
    with pytest.raises(SystemExit):
        boot.create_dtk_venv("/python", wheelhouse)
    assert not boot.VENV.exists() and sentinel.read_text() == "running environment"


def test_dtk_pip_fallback_builds_a_real_isolated_venv_offline(monkeypatch, tmp_path):
    import ensurepip

    bundled = list((Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl"))
    if not bundled:
        pytest.skip("Local interpreter has no bundled pip wheel for this offline integration check")
    # The development venv may itself have been created by uv without pip.
    # Build a temporary host Python instead of modifying that active environment.
    host = tmp_path / "host-python"
    venv.create(host, with_pip=True)
    host_python = str(host / ("Scripts/python.exe" if boot.WIN else "bin/python"))
    host_pip = subprocess.check_output([host_python, "-m", "pip", "--version"], text=True)
    wheelhouse = tmp_path / "local-wheels"
    wheelhouse.mkdir()
    (wheelhouse / bundled[0].name).write_bytes(bundled[0].read_bytes())
    original = subprocess.run

    def simulate_missing_ensurepip(command, **kwargs):
        if command[1:4] == ["-m", "ensurepip", "--version"]:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="No module named ensurepip")
        return original(command, **kwargs)

    monkeypatch.setattr(boot.subprocess, "run", simulate_missing_ensurepip)
    boot.create_dtk_venv(host_python, wheelhouse)
    result = original(
        [str(boot.venv_python()), "-c", "import pip,sys,json; print(json.dumps([sys.prefix,pip.__file__]))"],
        text=True,
        capture_output=True,
        check=True,
    )
    prefix, pip_file = json.loads(result.stdout)
    assert Path(prefix) == boot.VENV and Path(pip_file).is_relative_to(boot.VENV)
    assert subprocess.check_output([host_python, "-m", "pip", "--version"], text=True) == host_pip


@pytest.mark.parametrize(
    "profile",
    ["windows-cuda", "linux-cuda", "linux-dtk"],
)
def test_accelerator_entries_refuse_unvalidated_architectures(monkeypatch, profile):
    """Wheels differ per architecture, so an x86-only entry must say so up front."""
    system = {"windows-cuda": "Windows", "linux-cuda": "Linux", "linux-dtk": "Linux"}[profile]
    monkeypatch.setattr(boot.platform, "system", lambda: system)
    monkeypatch.setattr(boot.platform, "machine", lambda: "aarch64")
    with pytest.raises(SystemExit):
        boot.platform_torch_tag(profile, "auto")


def test_cpu_entry_accepts_both_architectures(monkeypatch):
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    for machine in ("x86_64", "aarch64"):
        monkeypatch.setattr(boot.platform, "machine", lambda machine=machine: machine)
        assert boot.platform_torch_tag("cpu", "auto") == "cpu"


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("x86_64", "x86_64"), ("AMD64", "x86_64"), ("aarch64", "arm64"), ("arm64", "arm64")],
)
def test_host_arch_normalizes_vendor_spellings(monkeypatch, machine, expected):
    monkeypatch.setattr(boot.platform, "machine", lambda: machine)
    assert boot.host_arch() == expected


def test_environment_refuses_reuse_from_another_architecture(monkeypatch, tmp_path):
    """One environment directory must never be shared by two architectures."""
    monkeypatch.setattr(boot, "ROOT", tmp_path)
    monkeypatch.setattr(boot.platform, "system", lambda: "Linux")
    monkeypatch.setattr(boot.platform, "machine", lambda: "x86_64")
    marker = tmp_path / "environment/linux-cpu/venv/.ypuddin-install.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"profile": "linux-cpu", "arch": "arm64"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        boot.select_environment("cpu", "cpu")


def test_saved_download_settings_drive_bootstrap_sources(monkeypatch):
    monkeypatch.setattr(boot, "DOWNLOAD_SETTINGS", {"pypi": "tuna", "pytorch": "sjtu", "fallback": False})
    packages, torch = boot.index_chains("auto", "cu130")
    assert packages == ["https://pypi.tuna.tsinghua.edu.cn/simple"]
    assert torch == [("index-url", "https://mirror.sjtu.edu.cn/pytorch-wheels/cu130")]
    assert boot.index_chains("auto", "dtk")[1] == []


def test_external_environment_root_preserves_default_and_isolates_profiles(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.platform, "system", lambda: "Windows")
    default = boot.ROOT / "venv"
    default.mkdir()
    (default / "keep.txt").write_text("keep")
    external = tmp_path / "different-drive"
    assert boot.select_environment("auto", "cu130", str(external)) == "windows-cuda"
    assert boot.VENV == external / "windows-cuda" / "venv"
    assert not external.exists()
    assert (default / "keep.txt").read_text() == "keep"
    boot.select_environment("cpu", "cpu", str(external))
    assert boot.VENV == external / "windows-cpu" / "venv"


def test_external_environment_refuses_unowned_venv_and_symlink(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.platform, "system", lambda: "Windows")
    external = tmp_path / "custom"
    venv = external / "windows-cpu" / "venv"
    venv.mkdir(parents=True)
    (venv / "user-file").write_text("preserve")
    with pytest.raises(SystemExit):
        boot.select_environment("cpu", "cpu", str(external))
    assert (venv / "user-file").read_text() == "preserve"
    linked = tmp_path / "linked"
    linked.symlink_to(external, target_is_directory=True)
    with pytest.raises(SystemExit):
        boot.select_environment("cpu", "cpu", str(linked))


def test_external_environment_resumes_owned_partial_install(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.platform, "system", lambda: "Windows")
    root = tmp_path / "new-environments"
    venv = root / "windows-cpu" / "venv"
    venv.mkdir(parents=True)
    (venv / ".ypuddin-owner.json").write_text(
        json.dumps({"profile": "windows-cpu", "arch": boot.host_arch()})
    )
    boot.select_environment("cpu", "cpu", str(root))
    assert boot.VENV == venv
    assert not boot.MARKER.exists()  # ownership is not installation success


def test_saved_environment_root_and_explicit_cli_override(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "settings.json").write_text(json.dumps({"paths": {"bootstrap_env_dir": str(tmp_path / "saved")}}))
    monkeypatch.setattr(boot, "platform_torch_tag", lambda *args: "cpu")
    monkeypatch.setattr(boot, "doctor", lambda: 0)
    calls = []
    monkeypatch.setattr(boot, "select_environment", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert boot.main(["doctor", "--data-root", str(data)]) == 0
    assert calls[-1][1]["env_root"] == str(tmp_path / "saved")
    assert boot.main(["doctor", "--data-root", str(data), "--env-root", str(tmp_path / "explicit")]) == 0
    assert calls[-1][1]["env_root"] == str(tmp_path / "explicit")


def test_custom_cache_controls_both_installers_and_keeps_profiles_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(boot, "PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(boot, "PROFILE", "windows-cuda")
    env = boot._env()
    assert env["PIP_CACHE_DIR"] == str(tmp_path / "cache/packages/windows-cuda")
    assert env["UV_CACHE_DIR"] == str(tmp_path / "cache/packages/windows-cuda/uv")
    monkeypatch.setattr(boot, "PROFILE", "windows-cpu")
    assert boot._env()["PIP_CACHE_DIR"] != env["PIP_CACHE_DIR"]


@pytest.mark.parametrize("arguments", [["--env-root"], ["--env-root", " "], ["--env-root="]])
def test_empty_environment_root_is_rejected(arguments):
    with pytest.raises(SystemExit):
        boot.main(["doctor", *arguments])


def test_default_server_port_matches_service(tmp_path):
    from ypuddin.server.context import DEFAULT_SETTINGS
    from ypuddin.server.lifecycle import saved_address

    assert boot.server_address(None, None, str(tmp_path)) == ("127.0.0.1", 8123)
    assert saved_address(tmp_path, None, None) == ("127.0.0.1", 8123)
    assert DEFAULT_SETTINGS["server"]["port"] == 8123
