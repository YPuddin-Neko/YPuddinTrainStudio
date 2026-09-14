"""scripts/bootstrap.py: package-source fallback chain and GPU-aware torch flavour selection (no network)."""

import importlib.util
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ypuddin_bootstrap", SOURCE_ROOT / "scripts" / "bootstrap.py")
boot = importlib.util.module_from_spec(SPEC)
sys.modules["ypuddin_bootstrap"] = boot
SPEC.loader.exec_module(boot)


@pytest.fixture(autouse=True)
def isolated_bootstrap_root(monkeypatch, tmp_path_factory):
    root = tmp_path_factory.mktemp("bootstrap-project")
    (root / "pyproject.toml").write_bytes((SOURCE_ROOT / "pyproject.toml").read_bytes())
    monkeypatch.setattr(boot, "ROOT", root)
    monkeypatch.setattr(boot, "PROFILE", "legacy")
    monkeypatch.setattr(boot, "VENV", root / "venv")
    monkeypatch.setattr(boot, "MARKER", root / "venv" / ".ypuddin-install.json")
    monkeypatch.setattr(boot, "FRONTEND", root / "frontend")


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
    fresh_torch(monkeypatch, "cu128")
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
    assert len(torch_calls) == 2
    assert "--no-index" in torch_calls[0] and torch_calls[0][-1].endswith("pytorch-wheels/cu126")
    assert (
        "--upgrade" not in torch_calls[1] and "--index-url" in torch_calls[1]
    )  # deps only, keep the cu126 wheel


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
    assert boot.VENV == boot.ROOT / "environment/profiles" / cpu / "venv"
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
    other = boot.ROOT / "environment/profiles/linux-cuda/venv/keep.txt"
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
