"""Package source and the built UI, preserving local runtime data outside the archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "YPuddinTrainStudio"


def build_identity(root: Path) -> dict:
    """Describe the source included in this archive, including uncommitted changes."""
    root = root.resolve()

    def git(*args: str) -> str | None:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env["GIT_OPTIONAL_LOCKS"] = "0"
        try:
            with tempfile.TemporaryFile() as output:
                result = subprocess.run(
                    ["git", "--no-pager", "-c", "core.fsmonitor=false", "-C", str(root), *args],
                    stdout=output, stderr=subprocess.DEVNULL, timeout=2, env=env, check=False,
                )
                output.seek(0)
                raw = output.read(16_385)
            if result.returncode or len(raw) > 16_384:
                return None
            return raw.decode("utf-8", errors="replace").strip()
        except (OSError, subprocess.TimeoutExpired):
            return None

    version_text = (root / "ypuddin/__init__.py").read_text(encoding="utf-8")
    version = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', version_text)
    build = {
        "version": version.group(1) if version else "unknown",
        "commit": None, "branch": None, "dirty": None,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    top = git("rev-parse", "--show-toplevel")
    commit = git("rev-parse", "HEAD") if top and Path(top).resolve() == root else None
    if commit and re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        changes = git("status", "--porcelain=v1", "--untracked-files=normal")
        build.update(commit=commit.lower(), branch=None if branch == "HEAD" else branch,
                     dirty=None if changes is None else bool(changes))
    return build


def excluded_reason(name: str | Path, *, built_ui: bool = False) -> str | None:
    """Classify repository-relative names without reading potentially private files."""
    normalized = str(name).replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = tuple(part.casefold() for part in path.parts)
    if not parts or path.is_absolute() or ".." in parts or ":" in parts[0]:
        return "invalid repository-relative path"
    if any(part.startswith("._") or part == "__macosx" for part in parts):
        return "local OS metadata"
    filename = parts[-1]
    if parts[0] in {
        "data",
        "models",
        "weights",
        "project",
        "projects",
        "dataset",
        "datasets",
        "output",
        "outputs",
        "runs",
        "cache",
        "caches",
        "checkpoint",
        "checkpoints",
        "logs",
        "samples",
        "thumbs",
        "environment",
        "traindata",
        "reg",
        "presets",
    }:
        return "root runtime directory"
    if "studio_data" in parts:
        return "Studio runtime data"
    if parts[0] in {"test", "tests"}:
        return "local tests and test records"
    if parts[:2] == ("frontend", "screenshots"):
        return "generated browser acceptance screenshots"
    if parts[:2] in {
        ("docs", "validation"), ("docs", "screenshots"), ("docs", "internal"),
        ("docs", "reports"), ("docs", "reference"),
    } or parts in {("handover.md",), ("docs", "design", "03-status.md"), ("docs", "frontend-spec.md")}:
        return "local development records"
    if (
        parts[0] == "docs"
        and filename.endswith(".md")
        and (
            any(
                token in filename
                for token in (
                    "acceptance",
                    "report",
                    "audit",
                    "review",
                    "validation",
                    "comparison",
                    "resume_consistency",
                    "training_completion",
                    "training_test_gaps",
                    "fsdp_adapter_windows",
                    "windows_readiness",
                    "dtk_bf16",
                    "ui_workflow",
                    "anima_full_training_diagnostic_",
                    "ui_and_launcher_changes_",
                    "adapter_form_layout_",
                    "mac_gpu_sensors_",
                    "mac_gpu_telemetry_",
                )
            )
            or re.fullmatch(r"ui_.*_\d{4}-\d{2}-\d{2}\.md", filename)
        )
    ):
        return "local acceptance report"
    build_output = built_ui and parts[:2] == ("frontend", "dist")
    directories = parts[:-1]
    if build_output:
        directories = directories[2:]
    if set(directories) & {
        "venv",
        ".venv",
        "node_modules",
        ".git",
        ".handoff",
        "remote-testing",
        ".windows-auth",
        ".ssh",
        "playwright-report",
        "test-results",
        "blob-report",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".nox",
        ".cache",
        ".vite",
        ".idea",
        ".vscode",
        "build",
        "dist",
        "htmlcov",
        "coverage",
    } or any(part.endswith(".egg-info") for part in directories):
        return "local environment, tooling or build output"
    if filename in {".ds_store", "thumbs.db", "desktop.ini", ".coverage"} or filename.startswith(
        ".coverage."
    ):
        return "local OS or coverage file"
    if filename in {".env", "secrets.json", "credentials.json", ".netrc", "_netrc", ".npmrc", ".pypirc"}:
        return "local credentials or environment"
    if filename.startswith(("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")):
        return "local SSH authentication material"
    if filename.startswith((".secrets-", "secrets.json.", "credentials.json.")):
        return "local credential backup or temporary file"
    if filename.startswith(".env.") and filename not in {".env.example", ".env.sample"}:
        return "local environment override"
    if len(parts) == 1 and filename in {"settings.json", "source_manifest.json"}:
        return "generated local settings or package manifest"
    suffixes = (
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".whl",
        ".npy",
        ".zip",
        ".7z",
        ".rar",
        ".tar",
        ".tgz",
        ".tar.gz",
        ".tar.bz2",
        ".tar.xz",
        ".pyc",
        ".pyo",
        ".pyd",
        ".log",
        ".safetensors",
        ".bin",
        ".gguf",
        ".ckpt",
        ".pt",
        ".pth",
        ".onnx",
        ".npz",
        ".ses",
        ".tsbuildinfo",
        ".swp",
        ".swo",
        ".tmp",
        ".part",
        ".partial",
    )
    if filename.endswith(suffixes) or filename.startswith("events.out.tfevents."):
        return "archive, model, cache or generated file"
    if any(
        filename.endswith(extension + sidecar)
        for extension in (".db", ".sqlite", ".sqlite3")
        for sidecar in ("", "-wal", "-shm", "-journal")
    ):
        return "local database or journal"
    return None


def has_studio_database(directory: Path) -> bool:
    """Identify the service's data root by its database filename, never its contents."""
    if not directory.is_dir() or directory.is_symlink():
        return False
    return any(child.name.casefold() == "studio.db" and child.is_file() for child in directory.iterdir())


def runtime_roots(root: Path, names: list[str]) -> set[tuple[str, ...]]:
    # The DB itself is ignored and will not appear in Git's untracked inventory.
    # Check candidate ancestors once, without recursively scanning environments/data.
    parents = {parent for name in names for parent in Path(name).parents if parent != Path(".")}
    found: set[tuple[str, ...]] = set()
    for parent in sorted(parents, key=lambda p: (len(p.parts), str(p))):
        parts = tuple(part.casefold() for part in parent.parts)
        if any(parts[: len(known)] == known for known in found):
            continue  # Do not inspect a potentially large image directory below a known data root.
        if not any((root / p).is_symlink() for p in (parent, *parent.parents)) and has_studio_database(
            root / parent
        ):
            found.add(parts)
    return found


def runtime_reason(name: str | Path, roots: set[tuple[str, ...]]) -> str | None:
    parts = tuple(part.casefold() for part in PurePosixPath(str(name).replace("\\", "/")).parts)
    if any(parts[: len(parent)] == parent for parent in roots):
        return "custom Studio runtime directory (studio.db present)"
    return None


def tracked_violations(root: Path = ROOT) -> list[dict[str, str]]:
    """Inspect the Git index; ignored tracked files need explicit untracking."""

    def names(*args: str) -> list[str]:
        return [
            name
            for name in subprocess.check_output(["git", "ls-files", "-z", *args], cwd=root)
            .decode("utf-8", errors="surrogateescape")
            .split("\0")
            if name
        ]

    ignored = set(names("--cached", "--ignored", "--exclude-standard"))
    tracked = names("--cached")
    roots = runtime_roots(root, tracked)
    violations = []
    for name in sorted(tracked):
        reason = excluded_reason(name) or runtime_reason(name, roots)
        if reason or name in ignored:
            violations.append({"path": name, "reason": reason or "tracked despite Git ignore rules"})
    return violations


def collect_files(root: Path) -> set[Path]:
    inventory = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
        )
        .decode()
        .split("\0")
    )
    files = {Path(name) for name in inventory if name and (root / name).is_file()}
    files.update(p.relative_to(root) for p in (root / "frontend/dist").rglob("*") if p.is_file())
    roots = runtime_roots(root, [str(p) for p in files])
    return {
        p
        for p in files
        if excluded_reason(p, built_ui=True) is None
        and runtime_reason(p, roots) is None
        and not any((root / parent).is_symlink() for parent in (p, *p.parents))
    }


def package(output: Path) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {output}")
    if has_studio_database(ROOT):
        raise ValueError(
            "Source root also contains studio.db; separate the Studio data root before packaging"
        )
    if violations := tracked_violations(ROOT):
        raise ValueError(
            "Runtime/local files are tracked by Git; remove them from the index while preserving "
            f"local data before packaging: {json.dumps(violations, ensure_ascii=True)}"
        )
    spec = importlib.util.spec_from_file_location("studio_bootstrap", ROOT / "scripts/bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    if bootstrap.frontend_stale():
        raise ValueError("frontend is stale; build the current UI before packaging")
    files = collect_files(ROOT)
    required = {
        Path(name)
        for name in (
            "README.md",
            "LICENSE",
            "CHANGELOG.md",
            "pyproject.toml",
            ".gitignore",
            "studio-windows-cuda.bat",
            "studio-cpu.bat",
            "studio-cpu.sh",
            "studio-linux-cuda.sh",
            "studio-linux-dtk.sh",
            "studio-macos.command",
            "scripts/launch.sh",
            "scripts/launch.bat",
            "scripts/bootstrap.py",
            "scripts/package_source.py",
            "ypuddin/cli.py",
            "ypuddin/sampling/ER_SDE_LICENSE.txt",
            "ypuddin/sampling/ER_SDE_PROVENANCE.md",
            "frontend/package.json",
            "frontend/package-lock.json",
            "frontend/buildFingerprint.ts",
            "frontend/public/licenses/lucide-gpu.txt",
            "frontend/dist/index.html",
            "frontend/dist/.source-manifest.json",
            "frontend/dist/licenses/lucide-gpu.txt",
            "docs/README.md",
            "docs/INSTALLATION.md",
            "docs/RUNTIME_DTK.md",
            "docs/TRAINING.md",
            "docs/DEVELOPMENT.md",
            "docs/api/openapi.json",
        )
    }
    if missing := required - files:
        raise ValueError(f"missing required package entries: {sorted(missing)}")
    frontend_manifest = json.loads((ROOT / "frontend/dist/.source-manifest.json").read_text(encoding="utf-8"))
    build_files = {Path("frontend") / name for name in frontend_manifest["inputs"]}
    build_files.update(Path("frontend/dist") / name for name in frontend_manifest["outputs"])
    if missing := build_files - files:
        raise ValueError(f"build inputs/outputs excluded from package: {sorted(missing)}")
    for name in ("scripts/launch.bat", "studio-windows-cuda.bat", "studio-cpu.bat"):
        bat = (ROOT / name).read_bytes()
        assert bat.isascii() and bat.count(b"\n") == bat.count(b"\r\n"), f"{name} must be ASCII CRLF"
    manifest = {
        "build": build_identity(ROOT),
        "files": {
            p.as_posix(): {
                "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest(),
                "bytes": (ROOT / p).stat().st_size,
            }
            for p in sorted(files)
        },
        "note": "Source + compiled frontend. No model weights, Python environment, runtime data or Git history.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(files):
                archive.write(ROOT / path, f"{PREFIX}/{path.as_posix()}")
            archive.writestr(
                f"{PREFIX}/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False)
            )
        with zipfile.ZipFile(temporary) as archive:
            assert archive.testzip() is None, "archive CRC failed"
            for name, data in manifest["files"].items():
                assert hashlib.sha256(archive.read(f"{PREFIX}/{name}")).hexdigest() == data["sha256"]
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive": str(output),
        "entries": len(files) + 1,
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--check-git", action="store_true", help="Check tracked files without building or packaging"
    )
    args = parser.parse_args()
    if args.check_git:
        violations = tracked_violations(ROOT)
        print(json.dumps({"success": not violations, "violations": violations}, indent=2))
        raise SystemExit(1 if violations else 0)
    if args.output is None:
        parser.error("output is required unless --check-git is used")
    print(json.dumps(package(args.output), indent=2))
