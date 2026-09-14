"""SQLite persistence for the service (single writer, WAL, thread-safe via a lock)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, note TEXT DEFAULT '', archived INTEGER DEFAULT 0,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY, project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
  path TEXT NOT NULL, repeats INTEGER DEFAULT 1, caption_ext TEXT DEFAULT '.txt', is_reg INTEGER DEFAULT 0,
  prior_weight REAL DEFAULT 1.0, class_prompt TEXT, created_at REAL NOT NULL,
  index_status TEXT DEFAULT 'indexing', stats_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL, project_id TEXT,
  status TEXT NOT NULL, priority INTEGER DEFAULT 0, scheduled_at REAL,
  created_at REAL NOT NULL, started_at REAL, finished_at REAL,
  run_dir TEXT, config_json TEXT, progress_json TEXT DEFAULT '{}', latest_json TEXT DEFAULT '{}',
  error TEXT, resume_from TEXT, pid INTEGER, exit_code INTEGER
);
CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(status, priority DESC, created_at ASC);
CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY, project_id TEXT, job_id TEXT, name TEXT NOT NULL, path TEXT NOT NULL,
  size INTEGER DEFAULT 0, kind TEXT DEFAULT 'weights', step INTEGER, created_at REAL NOT NULL, meta_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY, family TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL, size INTEGER DEFAULT 0,
  dtype TEXT, is_default INTEGER DEFAULT 0, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Database:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()
        self._migrate_versions()

    def _migrate_versions(self) -> None:
        """Attach old rows without changing their paths or immutable job snapshots."""
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for table, column in (
                    ("projects", "active_version_id"),
                    ("datasets", "version_id"),
                    ("jobs", "version_id"),
                    ("artifacts", "version_id"),
                    ("datasets", "origin_path"),
                ):
                    columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
                    if column not in columns:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
                self.conn.execute("""CREATE TABLE IF NOT EXISTS project_versions (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', parent_version_id TEXT,
                    archived INTEGER NOT NULL DEFAULT 0, legacy_layout INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ready', busy TEXT, progress_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    UNIQUE(project_id, name))""")
                for table, column, declaration in (
                    ("projects", "layout_version", "INTEGER NOT NULL DEFAULT 1"),
                    ("projects", "category", "TEXT"),
                    ("projects", "cover_key", "TEXT"),
                    ("project_versions", "number", "INTEGER"),
                    ("jobs", "samples_dir", "TEXT"),
                    ("jobs", "gpu_devices_json", "TEXT NOT NULL DEFAULT '[]'"),
                    ("models", "purpose", "TEXT NOT NULL DEFAULT 'training'"),
                    ("models", "variant", "TEXT"),
                ):
                    columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
                    if column not in columns:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
                for project in list(
                    self.conn.execute("SELECT * FROM projects WHERE active_version_id IS NULL")
                ):
                    vid = new_id("v")
                    self.conn.execute(
                        "INSERT INTO project_versions (id,project_id,name,legacy_layout,created_at,updated_at) VALUES (?,?,?,1,?,?)",
                        (vid, project["id"], "v1", project["created_at"], project["updated_at"]),
                    )
                    self.conn.execute(
                        "UPDATE projects SET active_version_id=? WHERE id=?", (vid, project["id"])
                    )
                for project in self.conn.execute("SELECT id FROM projects").fetchall():
                    number = self.conn.execute(
                        "SELECT coalesce(max(number),0) FROM project_versions WHERE project_id=?",
                        (project["id"],),
                    ).fetchone()[0]
                    for version in self.conn.execute(
                        "SELECT id FROM project_versions WHERE project_id=? AND number IS NULL ORDER BY created_at,id",
                        (project["id"],),
                    ).fetchall():
                        number += 1
                        self.conn.execute(
                            "UPDATE project_versions SET number=? WHERE id=?", (number, version["id"])
                        )
                self.conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_version_number ON project_versions(project_id,number)"
                )
                for table in ("datasets", "jobs", "artifacts"):
                    self.conn.execute(
                        f"UPDATE {table} SET version_id=(SELECT active_version_id FROM projects WHERE projects.id={table}.project_id) WHERE version_id IS NULL AND project_id IS NOT NULL"
                    )
                self.conn.execute(
                    "UPDATE artifacts SET version_id=(SELECT version_id FROM jobs WHERE jobs.id=artifacts.job_id) WHERE job_id IN (SELECT id FROM jobs WHERE version_id IS NOT NULL)"
                )
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    # ----------------------------------------------------------------- generic helpers
    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        # the cursor must be drained under the lock: another thread's statement on the shared
        # connection would reset it and fetchone() would silently return None
        with self.lock:
            row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        self.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", tuple(row.values()))

    def update(self, table: str, id_: str, fields: dict[str, Any]) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*fields.values(), id_))

    def delete(self, table: str, id_: str) -> None:
        self.execute(f"DELETE FROM {table} WHERE id=?", (id_,))

    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self.fetchone("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_kv(self, key: str, value: Any) -> None:
        self.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    def close(self) -> None:
        self.conn.close()


def now() -> float:
    return time.time()
