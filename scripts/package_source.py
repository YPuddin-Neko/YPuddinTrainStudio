"""Package source and the built UI, preserving local runtime data outside the archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "YPuddinTrainStudio"


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
    if filename.startswith((".secrets-", "secrets.json.", "credentials.json.")):
        return "local credential backup or temporary file"
    if filename.startswith(".env.") and filename not in {".env.example", ".env.sample"}:
        return "local environment override"
    if len(parts) == 1 and filename in {"settings.json", "source_manifest.json"}:
        return "generated local settings or package manifest"
    suffixes = (
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
            "scripts/launch.sh",
            "scripts/launch.bat",
            "studio-windows-cuda.bat",
            "studio-cpu.bat",
            "studio-cpu.sh",
            "studio-linux-cuda.sh",
            "studio-linux-dtk.sh",
            "studio-macos.command",
            "pyproject.toml",
            ".gitignore",
            "scripts/package_source.py",
            "tests/unit/test_repository_content.py",
            "docs/REPOSITORY_CONTENT_2026-09-12.md",
            "docs/validation/repository-content-2026-09-12.json",
            "ypuddin/sampling/er_sde.py",
            "ypuddin/sampling/dispatch.py",
            "ypuddin/sampling/ER_SDE_LICENSE.txt",
            "ypuddin/sampling/ER_SDE_PROVENANCE.md",
            "ypuddin/data/caption_json.py",
            "ypuddin/server/source_roles.py",
            "ypuddin/server/routes_dataset_paint.py",
            "tests/unit/test_dataset_paint.py",
            "tests/unit/test_image_fit.py",
            "tests/unit/test_training_loss_mean.py",
            "frontend/src/components/BrandMark.tsx",
            "frontend/src/components/masks/ImageEditor.tsx",
            "frontend/src/components/masks/image-editor.css",
            "frontend/src/components/masks/paintApi.ts",
            "frontend/src/components/masks/paintDocument.ts",
            "frontend/src/components/useAnimatedClose.ts",
            "frontend/src/styles/motion.css",
            "frontend/tests/captionPainting.test.tsx",
            "frontend/tests/captionParameterLayout.test.tsx",
            "frontend/tests/imageEditor.test.tsx",
            "frontend/tests/jobSummary.test.tsx",
            "frontend/tests/paintDocument.test.ts",
            "frontend/tests/surfaceMotion.test.tsx",
            "docs/UI_REVIEW_V056_2026-09-12.md",
            "docs/validation/v0.5.6.json",
            "docs/MAC_GPU_TELEMETRY_2026-09-12.md",
            "docs/validation/v0.5.7.json",
            "tests/unit/test_apple_telemetry.py",
            "ypuddin/server/apple_sensors.py",
            "tests/unit/test_apple_sensors.py",
            "docs/MAC_GPU_SENSORS_2026-09-12.md",
            "docs/validation/v0.5.8.json",
            "docs/ADAPTER_FORM_LAYOUT_2026-09-12.md",
            "docs/validation/v0.5.9.json",
            "ypuddin/server/output_binding.py",
            "ypuddin/server/model_inspection.py",
            "tests/unit/test_source_roles.py",
            "tests/unit/test_plan_preview.py",
            "tests/unit/test_output_binding.py",
            "tests/unit/test_model_inspection.py",
            "tests/unit/test_runtime_training_devices.py",
            "frontend/src/pages/Models/LocalModelRegistration.tsx",
            "frontend/tests/sourceRoles.test.tsx",
            "frontend/tests/localModelInspection.test.tsx",
            "tests/unit/test_er_sde.py",
            "tests/unit/test_sampling_schedule_dispatch.py",
            "tests/unit/test_caption_json.py",
            "tests/unit/test_json_caption_workflow.py",
            "tests/unit/test_preset_management.py",
            "docs/TRAINING_PARAMETERS.md",
            "docs/JSON_CAPTIONS.md",
            "docs/UI_PARAMETERS_V055_2026-09-12.md",
            "frontend/tests/lokrParameterMode.test.tsx",
            "frontend/src/components/ConfigHelp.tsx",
            "frontend/src/components/config-help.css",
            "frontend/src/components/CaptionFormatSelect.tsx",
            "frontend/src/components/PresetPreview.tsx",
            "frontend/src/pages/Presets/Presets.tsx",
            "frontend/src/pages/Presets/presets.css",
            "frontend/src/utils/presetEditor.ts",
            "frontend/tests/configHelp.test.tsx",
            "frontend/tests/parameterChoices.test.tsx",
            "frontend/tests/presetManagement.test.tsx",
            "HANDOVER.md",
            "LICENSE",
            "docs/FIX_REPORT_2026-09-11.md",
            "docs/UI_WORKFLOW_2026-09-11.md",
            "docs/UI_REDESIGN_2026-09-11.md",
            "docs/UI_VERSIONS_2026-09-11.md",
            "docs/validation/v0.4.0.json",
            "docs/validation/v0.5.0.json",
            "docs/validation/v0.5.1.json",
            "docs/validation/v0.5.2.json",
            "docs/validation/v0.5.3.json",
            "docs/validation/v0.5.4.json",
            "docs/validation/v0.5.5.json",
            "docs/UI_WORKSPACE_V053_2026-09-12.md",
            "docs/UI_WORKSPACE_V054_2026-09-12.md",
            "docs/USER_REQUIREMENTS_AUDIT_2026-09-12.md",
            "frontend/src/components/datasets/DatasetNavigationGuard.tsx",
            "frontend/src/components/projects/ProjectSidebarContext.ts",
            "frontend/src/styles/project-sidebar.css",
            "frontend/tests/datasetRouterNavigation.test.tsx",
            "docs/UI_DESIGN_REVIEW_2026-09-12.md",
            "ypuddin/server/routes_credentials.py",
            "ypuddin/server/model_recommendations.py",
            "ypuddin/server/routes_model_recommendations.py",
            "frontend/src/pages/Settings/AccessKeys.tsx",
            "frontend/src/pages/Settings/access-keys.css",
            "frontend/src/pages/Models/models.css",
            "frontend/src/pages/Queue/queue.css",
            "frontend/src/pages/TrainConfig/training-workspace.css",
            "frontend/tests/accessKeys.test.tsx",
            "frontend/tests/workspaceRequestIsolation.test.tsx",
            "tests/e2e/test_model_recommendations.py",
            "tests/unit/test_access_credentials.py",
            "tests/unit/test_queue_views.py",
            "docs/UI_SIMPLIFICATION_2026-09-12.md",
            "frontend/src/components/StudioSelect.tsx",
            "frontend/src/components/datasets/CaptionViewer.tsx",
            "frontend/src/components/datasets/caption-viewer.css",
            "frontend/src/components/datasets/RegularizationPanel.tsx",
            "frontend/src/components/datasets/regularization.css",
            "frontend/src/pages/Settings/ServiceInfo.tsx",
            "ypuddin/server/regularization.py",
            "ypuddin/server/regularization_worker.py",
            "ypuddin/server/routes_regularization.py",
            "tests/unit/test_project_layout.py",
            "tests/unit/test_regularization.py",
            "frontend/tests/studioSelect.test.tsx",
            "frontend/tests/regularizationPanel.test.tsx",
            "frontend/tests/preferencesStorage.test.tsx",
            "docs/UI_PIPELINE_2026-09-11.md",
            "docs/native-resolution.md",
            "ypuddin/data/native.py",
            "ypuddin/server/dataset_pipeline.py",
            "ypuddin/server/routes_dataset_pipeline.py",
            "ypuddin/server/model_catalog.py",
            "ypuddin/server/model_credentials.py",
            "frontend/src/components/datasets/DatasetPipelinePanel.tsx",
            "frontend/src/components/datasets/VisualCropEditor.tsx",
            "frontend/src/components/datasets/dataset-pipeline.css",
            "frontend/src/pages/Models/ModelCredentials.tsx",
            "frontend/src/schema/SchemaForm/NumericControl.tsx",
            "frontend/src/utils/trainingPresets.ts",
            "tests/unit/test_native_resolution.py",
            "tests/unit/test_dataset_pipeline.py",
            "frontend/tests/percentageControls.test.tsx",
            "docs/api/openapi.json",
            "frontend/dist/index.html",
            "frontend/dist/.source-manifest.json",
            "frontend/dist/licenses/lucide-gpu.txt",
            "frontend/public/licenses/lucide-gpu.txt",
            "frontend/buildFingerprint.ts",
            "ypuddin/train/logging.py",
            "ypuddin/models/fingerprints.py",
            "ypuddin/server/hardware.py",
            "ypuddin/server/dataset_uploads.py",
            "ypuddin/server/model_downloads.py",
            "ypuddin/server/routes_model_downloads.py",
            "frontend/src/components/ProjectWorkflow.tsx",
            "frontend/src/components/SampleLoss.tsx",
            "frontend/src/components/RouteError.tsx",
            "frontend/src/utils/trainingFamilies.ts",
            "frontend/src/utils/metrics.ts",
            "frontend/tests/metrics.test.ts",
            "frontend/src/pages/JobDetail/metricPresentation.ts",
            "frontend/src/pages/JobDetail/job-metrics.css",
            "frontend/tests/metricPresentation.test.ts",
            "ypuddin/server/sample_events.py",
            "ypuddin/server/project_covers.py",
            "ypuddin/server/family_config.py",
            "tests/unit/test_sample_metadata.py",
            "tests/unit/test_project_presentation.py",
            "tests/unit/test_version_family.py",
            "tests/e2e/test_sample_loss.py",
            "frontend/src/components/sample-loss.css",
            "frontend/src/pages/Projects/ProjectCardMenu.tsx",
            "frontend/src/pages/Projects/ProjectEditor.tsx",
            "frontend/src/pages/Projects/projectGallery.ts",
            "frontend/src/utils/trainingFamilies.ts",
            "frontend/src/components/EnvironmentStatus.tsx",
            "frontend/src/pages/ProjectDetail/ProjectDataImport.tsx",
            "frontend/src/pages/TrainConfig/BucketInspector.tsx",
            "frontend/src/components/masks/MaskEditor.tsx",
            "frontend/src/components/EnvironmentManagerPanel.tsx",
            "ypuddin/server/environment.py",
            "ypuddin/server/routes_environment.py",
            "ypuddin/server/routes_dataset_masks.py",
            "ypuddin/server/versions.py",
            "frontend/src/components/SystemTelemetry.tsx",
            "frontend/src/utils/gpuTelemetry.ts",
            "frontend/src/components/Dialog.tsx",
            "frontend/src/components/SettingsDrawer.tsx",
            "frontend/src/components/projects/ProjectWorkspaceHeader.tsx",
            "frontend/src/components/projects/VersionResults.tsx",
            "frontend/src/components/projects/useProjectVersions.ts",
            "frontend/src/pages/Settings/SettingsSections.tsx",
            "frontend/src/styles/project-workspace.css",
            "frontend/src/styles/project-results.css",
            "frontend/src/styles/settings-drawer.css",
            "frontend/src/styles/settings.css",
            "frontend/src/utils/projectVersions.ts",
            "frontend/src/main.tsx",
            "frontend/src/api/generated.ts",
            "frontend/tests/systemTelemetry.test.tsx",
            "frontend/tests/versionResults.test.tsx",
            "frontend/tests/projectVersions.test.tsx",
            "frontend/tests/settingsDrawer.test.tsx",
            "frontend/tests/settingsNavigation.test.tsx",
            "tests/unit/test_project_versions.py",
            "tests/unit/test_service_controls.py",
            "tests/e2e/test_service_shutdown.py",
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
