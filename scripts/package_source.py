"""Package source and the built UI, preserving local runtime data outside the archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "YPuddinTrainStudio"


def package(output: Path) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {output}")
    spec = importlib.util.spec_from_file_location("studio_bootstrap", ROOT / "scripts/bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    if bootstrap.frontend_stale():
        raise ValueError("frontend is stale; build the current UI before packaging")
    inventory = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    files = {Path(name) for name in inventory if name and (ROOT / name).is_file()}
    files.update(p.relative_to(ROOT) for p in (ROOT / "frontend/dist").rglob("*") if p.is_file())
    forbidden_parts = {
        "venv",
        ".venv",
        "node_modules",
        ".git",
        ".handoff",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "studio_data",
    }
    forbidden_suffixes = {
        ".zip",
        ".pyc",
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
        ".sqlite",
        ".db",
        ".db-wal",
        ".db-shm",
        ".sqlite-wal",
        ".sqlite-shm",
        ".tsbuildinfo",
    }
    files = {
        p
        for p in files
        if not (set(p.parts) & forbidden_parts)
        and p.suffix not in forbidden_suffixes
        and p.name not in {".DS_Store", ".env", "secrets.json", "credentials.json"}
        and not (p.name.startswith(".env.") and p.name not in {".env.example", ".env.sample"})
        and not (ROOT / p).is_symlink()
    }
    required = {
        Path(name)
        for name in (
            "studio.sh",
            "studio.bat",
            "pyproject.toml",
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
    bat = (ROOT / "studio.bat").read_bytes()
    assert bat.isascii() and bat.count(b"\n") == bat.count(b"\r\n"), "studio.bat must be ASCII CRLF"
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
    parser.add_argument("output", type=Path)
    print(json.dumps(package(parser.parse_args().output), indent=2))
