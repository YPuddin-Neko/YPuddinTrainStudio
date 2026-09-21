"""Check metadata ownership in a real isolated Python, without installing training dependencies."""

import importlib.util
import json
import os
import sys
import venv
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ypuddin_bootstrap_metadata", Path(__file__).resolve().parents[3] / "scripts" / "bootstrap.py"
)
boot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boot)


@pytest.fixture
def editable_environment(monkeypatch, tmp_path):
    source = tmp_path / "源码 with spaces"
    package = source / "ypuddin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.5.9"\n', encoding="utf-8")
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
    monkeypatch.setattr(boot, "ROOT", source)
    monkeypatch.setattr(boot, "VENV", environment)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    site = Path(boot.venv_json("import json,sysconfig; print(json.dumps(sysconfig.get_path('purelib')))"))
    (site / "editable.pth").write_text(
        f"import sys; sys.path.insert(0, {ascii(str(source))})\n", encoding="ascii"
    )
    metadata = site / "ypuddin-0.5.9.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: ypuddin\nVersion: 0.5.9\n")
    (metadata / "entry_points.txt").write_text("[console_scripts]\nypuddin = ypuddin.cli:main\n")
    direct = {"url": source.as_uri(), "dir_info": {"editable": True}}
    (metadata / "direct_url.json").write_text(json.dumps(direct))
    return source, site, metadata, direct


def test_independent_metadata_survives_root_cleanup(editable_environment):
    source, _, metadata, _ = editable_environment
    egg = source / "ypuddin.egg-info"
    egg.mkdir()
    # A stale root copy must not shadow the installed metadata during verification.
    (egg / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: ypuddin\nVersion: 0.1.0\n")
    before = {path.name: path.read_bytes() for path in metadata.iterdir()}
    assert boot.editable_install_ready()
    boot.cleanup_build_metadata()
    assert not egg.exists()
    assert boot.editable_install_ready()
    assert {path.name: path.read_bytes() for path in metadata.iterdir()} == before


@pytest.mark.parametrize(
    "defect",
    [
        "legacy",
        "stale_version",
        "wrong_source",
        "wrong_import",
        "non_editable",
        "wrong_venv",
        "no_entrypoint",
    ],
)
def test_rejects_metadata_that_cannot_replace_root_copy(editable_environment, monkeypatch, tmp_path, defect):
    source, site, metadata, direct = editable_environment
    if defect == "legacy":
        metadata.rename(source / "ypuddin.egg-info")
    elif defect == "stale_version":
        (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: ypuddin\nVersion: 0.1.0\n")
    elif defect == "wrong_source":
        direct["url"] = tmp_path.as_uri()
        (metadata / "direct_url.json").write_text(json.dumps(direct))
    elif defect == "wrong_import":
        other = tmp_path / "other" / "ypuddin"
        other.mkdir(parents=True)
        (other / "__init__.py").write_text('__version__ = "0.5.9"\n')
        (site / "editable.pth").write_text(
            f"import sys; sys.path.insert(0, {ascii(str(other.parent))})\n", encoding="ascii"
        )
    elif defect == "non_editable":
        direct["dir_info"]["editable"] = False
        (metadata / "direct_url.json").write_text(json.dumps(direct))
    elif defect == "wrong_venv":
        selected_python = boot.venv_python()
        monkeypatch.setattr(boot, "venv_python", lambda: selected_python)
        monkeypatch.setattr(boot, "VENV", Path(sys.prefix))
    elif defect == "no_entrypoint":
        (metadata / "entry_points.txt").unlink()
    assert not boot.editable_install_ready()
