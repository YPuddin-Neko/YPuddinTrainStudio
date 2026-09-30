"""The saved products that are still present on disk."""

from pathlib import Path


def artifact_exists(row: dict) -> bool:
    path = Path(row["path"])
    return path.is_file() or (row["kind"] == "model" and path.is_dir())


def artifact_count(db, **scope: str) -> int:
    if not scope or not set(scope) <= {"project_id", "version_id", "job_id"}:
        raise ValueError("an artifact owner is required")
    rows = db.fetchall(
        "SELECT path,kind FROM artifacts WHERE " + " AND ".join(f"{key}=?" for key in scope),
        tuple(scope.values()),
    )
    return sum(artifact_exists(row) for row in rows)
