"""Exercise Git and package boundaries with disposable local data, never user data."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("studio_package", ROOT / "scripts/package_source.py")
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


def git(root, *args, **kwargs):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True, **kwargs)


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "core.ignoreCase", "false")
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    return tmp_path


@pytest.mark.parametrize(
    "name",
    [
        "studio_data/projects/person/config.json",
        "Studio_Data/Projects/person/train/photo.PNG",
        "copy/STUDIO_DATA/secrets.json",
        "project/p/v1/traindata/photo.png",
        "PROJECTS/p/config.json",
        "DATA/photo.jpg",
        "models/config.json",
        "weights/download.json",
        "output/job/metrics.jsonl",
        "cache/metadata.json",
        "checkpoints/job/state.json",
        "presets/local.json",
        "SETTINGS.JSON",
        ".env.production",
        "frontend/.ENV.LOCAL",
        "frontend/.env.production.example",
        "credentials.json",
        "config/SECRETS.JSON",
        "config/secrets.json.bak",
        "config/.secrets-interrupted-write",
        ".npmrc",
        ".netrc",
        "local/metadata.SQLITE3-WAL",
        "index.db-journal",
        "cache.SQLITE-SHM",
        "download.GGUF",
        "pytorch_model.BIN",
        "state.SAFETENSORS",
        "release.ZIP",
        "backup.tar.gz",
        "download.7Z",
        "report.LOG",
        "events.out.tfevents.123",
        "unfinished.TMP",
        "download.PARTIAL",
        "other/tsconfig.tsbuildinfo",
        ".cache/huggingface/metadata.json",
        ".mypy_cache/result.json",
        "._source.py",
        "frontend/src/._App.tsx",
        "frontend/src/._components/Component.tsx",
        "__MACOSX/source.py",
        "frontend/src/__MACOSX/Component.tsx",
        "nested/__macosx/metadata",
    ],
)
def test_generated_and_private_paths_are_ignored_and_excluded(repository, name):
    result = git(repository, "check-ignore", "--no-index", "--", name)
    assert result.stdout.strip()
    assert pack.excluded_reason(name, built_ui=True), name


@pytest.mark.parametrize(
    "name",
    [
        "ypuddin/data/native.py",
        "ypuddin/models/anima/family.py",
        "ypuddin/models/anima/assets/qwen3_06b/tokenizer.json",
        "ypuddin/models/anima/assets/t5_old/spiece.model",
        "frontend/src/components/projects/ProjectWorkspaceHeader.tsx",
        "frontend/src/components/datasets/CaptionViewer.tsx",
        "docs/images/layout.png",
        "tests/fixtures/caption.txt",
        ".env.example",
        "frontend/.env.sample",
        "examples/.ENV.EXAMPLE",
        "frontend/public/studio.svg",
        "pyproject.toml",
        "frontend/src/MACOSX.ts",
        "docs/__MACOSX-notes.md",
        "docs/note._backup.md",
    ],
)
def test_source_assets_and_templates_are_retained(repository, name):
    result = subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--no-index", "--", name],
        capture_output=True,
    )
    assert result.returncode == 1, (name, result.stdout)
    assert pack.excluded_reason(name, built_ui=True) is None


def test_forced_tracked_runtime_file_blocks_package_without_touching_data(repository, monkeypatch):
    private = repository / "Studio_Data/projects/p/config.json"
    private.parent.mkdir(parents=True)
    private.write_text('{"fixture": "local data must survive"}')
    source = repository / "source.py"
    source.write_text("# source\n")
    git(repository, "add", ".gitignore", "source.py")
    git(repository, "add", "-f", "--", str(private))
    assert pack.tracked_violations(repository) == [
        {"path": "Studio_Data/projects/p/config.json", "reason": "Studio runtime data"}
    ]
    monkeypatch.setattr(pack, "ROOT", repository)
    before = private.read_bytes()
    output = repository / "delivery.zip"
    with pytest.raises(ValueError, match="tracked by Git"):
        pack.package(output)
    assert private.read_bytes() == before
    assert not output.exists()
    git(repository, "rm", "--cached", "--", str(private))
    assert private.read_bytes() == before
    assert pack.tracked_violations(repository) == []


def test_locally_ignored_tracked_paths_are_detected_even_for_custom_data_root(repository):
    with (repository / ".gitignore").open("a") as stream:
        stream.write("\n/my_custom_studio/\n")
    private = repository / "my_custom_studio/config.json"
    private.parent.mkdir()
    private.write_text("{}")
    git(repository, "add", "-f", "--", str(private))
    assert pack.tracked_violations(repository) == [
        {"path": "my_custom_studio/config.json", "reason": "tracked despite Git ignore rules"}
    ]


def test_frontend_build_is_delivery_only_and_private_files_stay_excluded():
    assert pack.excluded_reason("frontend/dist/assets/app.js")
    assert pack.excluded_reason("frontend/dist/assets/app.js", built_ui=True) is None
    assert pack.excluded_reason("frontend/dist/credentials.json", built_ui=True)
    assert pack.excluded_reason("frontend/dist/Studio_Data/photo.png", built_ui=True)
    assert pack.excluded_reason("frontend/dist/nested/node_modules/app.js", built_ui=True)


def test_os_metadata_is_excluded_from_built_ui_without_removing_local_files(repository):
    artifacts = {
        "frontend/dist/index.html": b"<!doctype html>",
        "frontend/dist/assets/app.js": b"export {};",
        "frontend/dist/._index.html": b"AppleDouble metadata",
        "frontend/dist/assets/._app.js": b"AppleDouble metadata",
        "frontend/dist/__MACOSX/index.html": b"macOS metadata",
        "frontend/dist/assets/__MACOSX/app.js": b"macOS metadata",
        "frontend/dist/._assets/app.js": b"macOS metadata",
    }
    for name, content in artifacts.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    assert pack.collect_files(repository) == {
        Path(".gitignore"),
        Path("frontend/dist/index.html"),
        Path("frontend/dist/assets/app.js"),
    }
    for name, content in artifacts.items():
        assert (repository / name).read_bytes() == content


def test_custom_runtime_root_is_excluded_without_manual_ignore(repository):
    root = repository / "my custom studio"
    data = root / "project/person/v1/traindata/photo.png"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"fixture photo")
    data.with_suffix(".txt").write_text("fixture caption")
    (root / "STUDIO.DB").write_bytes(b"fixture database marker, never opened")
    (root / "settings.json").write_text("{}")
    (repository / "source.py").write_text("# keep source\n")
    git(repository, "add", ".gitignore", "source.py")
    # Ignored DB is absent from Git inventory, but its sibling data must still be excluded.
    inventory = git(repository, "ls-files", "--others", "--exclude-standard").stdout.decode()
    assert "photo.png" in inventory and "STUDIO.DB" not in inventory
    assert pack.collect_files(repository) == {Path(".gitignore"), Path("source.py")}
    git(repository, "add", "--", str(data))
    assert pack.tracked_violations(repository) == [
        {
            "path": "my custom studio/project/person/v1/traindata/photo.png",
            "reason": "custom Studio runtime directory (studio.db present)",
        }
    ]
    assert data.read_bytes() == b"fixture photo"


def test_data_root_cannot_overlap_source_for_packaging(repository, monkeypatch):
    database = repository / "studio.db"
    database.write_bytes(b"local database")
    monkeypatch.setattr(pack, "ROOT", repository)
    with pytest.raises(ValueError, match="Source root also contains studio.db"):
        pack.package(repository / "delivery.zip")
    assert database.read_bytes() == b"local database"
    assert not (repository / "delivery.zip").exists()


def test_symlinks_do_not_pull_external_files_into_delivery(repository, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-external"
    outside.mkdir()
    (outside / "notes.txt").write_text("fixture, outside selected source tree")
    try:
        (repository / "linked.txt").symlink_to(outside / "notes.txt")
        (repository / "frontend/dist").parent.mkdir()
        (repository / "frontend/dist").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symbolic links is unavailable on this host")
    assert pack.collect_files(repository) == {Path(".gitignore")}


def test_windows_paths_are_checked_and_paths_cannot_escape_repository():
    assert pack.excluded_reason(r"Studio_Data\Projects\image.png")
    assert pack.excluded_reason(r"local\SECRETS.JSON")
    assert pack.excluded_reason(r"frontend\src\._App.tsx")
    assert pack.excluded_reason(r"frontend\dist\__MACOSX\app.js", built_ui=True)
    assert pack.excluded_reason("../image.png")
    assert pack.excluded_reason("/tmp/image.png")
    assert pack.excluded_reason(r"C:\data\image.png")
