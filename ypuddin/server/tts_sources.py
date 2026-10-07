"""Version-owned speech sources, byte identities and asynchronous complete scans."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from ypuddin.tts.issues import TtsIssue
from ypuddin.tts.source_models import (
    SourceSplit,
    TtsRowsResponse,
    TtsSource,
    TtsSourceChanged,
    TtsSourceRow,
    TtsSourcesResponse,
    TtsSourceSummary,
)
from ypuddin.tts.source_scan import (
    SourceFileError,
    absolute_path,
    file_identity,
    identity_changes,
    manifest_references,
    scan_manifest,
)
from ypuddin.tts.version_config import TtsConfigScope

from .db import new_id, now
from .errors import ApiError
from .project_deletion import deleting
from .tts_references import reject_deleting_references
from .versions import assert_version_writable

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tts_sources (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  version_id TEXT NOT NULL REFERENCES project_versions(id) ON DELETE CASCADE,
  split TEXT NOT NULL CHECK(split IN ('train','validation')),
  path TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
  state TEXT NOT NULL DEFAULT 'unchecked', check_id TEXT, snapshot_id TEXT,
  checked_at REAL, summary_json TEXT, issues_json TEXT NOT NULL DEFAULT '[]',
  issues_total INTEGER, issues_truncated INTEGER NOT NULL DEFAULT 0,
  fingerprint TEXT, identities_json TEXT NOT NULL DEFAULT '[]',
  created_at REAL NOT NULL, updated_at REAL NOT NULL,
  UNIQUE(version_id, split)
);
CREATE TABLE IF NOT EXISTS tts_source_snapshots (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES tts_sources(id) ON DELETE CASCADE,
  fingerprint TEXT NOT NULL, created_at REAL NOT NULL,
  engine TEXT NOT NULL DEFAULT 'voxcpm1.5',
  UNIQUE(source_id, fingerprint)
);
CREATE TABLE IF NOT EXISTS tts_source_rows (
  id TEXT NOT NULL,
  snapshot_id TEXT NOT NULL REFERENCES tts_source_snapshots(id) ON DELETE CASCADE,
  line INTEGER NOT NULL, row_json TEXT NOT NULL, normalized_json TEXT,
  assets_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(snapshot_id, id), UNIQUE(snapshot_id, line)
);
CREATE INDEX IF NOT EXISTS idx_tts_sources_owner ON tts_sources(project_id,version_id);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class TtsSources:
    def __init__(self, c: Any):
        self.c = c
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-source")
        self._closed = False
        with c.db.transaction():
            for source in c.db.fetchall("SELECT * FROM tts_sources WHERE state='checking'"):
                self._failure(source, "tts.check.interrupted", "服务停止时数据检查尚未完成，请重新检查。")

    def close(self) -> None:
        with self.c.db.lock:
            self._closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.c.db.transaction():
            for source in self.c.db.fetchall("SELECT * FROM tts_sources WHERE state='checking'"):
                self._failure(source, "tts.check.interrupted", "服务停止时数据检查尚未完成，请重新检查。")

    def _version(self, pid: str, vid: str, *, write: bool = False) -> dict:
        c = self.c
        c.require_project_type(pid, "tts")
        version = c.resolve_version(pid, vid)
        if deleting(c, pid):
            raise ApiError("项目正在删除。", code="project.deleting", status=409)
        if write:
            version = assert_version_writable(c, pid, vid, data=True)
            supervisor = getattr(c, "supervisor", None)
            for job in c.db.fetchall("SELECT id FROM jobs WHERE version_id=?", (vid,)):
                if supervisor and supervisor.is_running(job["id"]):
                    raise ApiError("版本仍有训练或试听进程运行。", code="version.jobs_busy", status=409,
                                   details={"blocker": {"job_id": job["id"]}})
        return version

    def _source(self, pid: str, vid: str, source_id: str) -> dict:
        self._version(pid, vid)
        row = self.c.db.fetchone(
            "SELECT * FROM tts_sources WHERE id=? AND project_id=? AND version_id=?", (source_id, pid, vid)
        )
        if row is None:
            raise ApiError("数据来源不存在或不属于当前版本。", code="tts.source_not_found", status=404)
        return self._refresh_engine(row)

    def _engine(self, pid: str, vid: str) -> str:
        from .tts_projects import _read

        try:
            return _read(self.c, pid, vid).config.engine
        except ApiError as exc:
            if exc.code == "tts.config_missing":
                return "voxcpm1.5"
            raise

    def _refresh_engine(self, source: dict) -> dict:
        if source["state"] in {"valid", "invalid"} and source["snapshot_id"]:
            snapshot = self.c.db.fetchone("SELECT engine FROM tts_source_snapshots WHERE id=?", (source["snapshot_id"],))
            snapshot_engine = snapshot["engine"] if snapshot else "voxcpm1.5"
            if snapshot_engine != self._engine(source["project_id"], source["version_id"]):
                self._stale(source, message="训练引擎已变化，请重新检查数据来源。")
                return self.c.db.fetchone("SELECT * FROM tts_sources WHERE id=?", (source["id"],))
        return source

    def invalidate_engine(self, pid: str, vid: str) -> None:
        """Discard completed and in-flight scans when a saved configuration changes engine."""
        with self.c.db.transaction():
            self._version(pid, vid)
            for source in self.c.db.fetchall("SELECT * FROM tts_sources WHERE version_id=?", (vid,)):
                if source["state"] in {"valid", "invalid", "checking"}:
                    self._stale(source, message="训练引擎已变化，请重新检查数据来源。")

    def _revision(self, vid: str) -> int:
        return self.c.db.fetchone("SELECT data_revision FROM project_versions WHERE id=?", (vid,))["data_revision"]

    def _cas(self, version: dict, expected: int) -> None:
        if type(expected) is not int or expected < 1:
            raise ApiError("数据修订号须为正整数。", code="tts.invalid", status=422)
        if version["data_revision"] != expected:
            raise ApiError("数据来源已更新，请重新读取后操作。", code="tts.data_conflict", status=409,
                           details={"current_data_revision": version["data_revision"]})

    def _bump(self, vid: str) -> int:
        self.c.db.execute(
            "UPDATE project_versions SET data_revision=data_revision+1,updated_at=? WHERE id=?", (now(), vid)
        )
        return self._revision(vid)

    def _mutable(self, pid: str, vid: str, expected: int) -> dict:
        version = self._version(pid, vid, write=True)
        self._cas(version, expected)
        checking = self.c.db.fetchone("SELECT id,check_id FROM tts_sources WHERE version_id=? AND state='checking'", (vid,))
        if checking:
            raise ApiError("数据来源正在检查，请等待检查完成。", code="version.busy", status=409,
                           details={"blocker": {"source_id": checking["id"], "check_id": checking["check_id"]}})
        return version

    def _dto(self, row: dict) -> TtsSource:
        row = self._refresh_engine(row)
        complete = row["state"] in {"valid", "invalid"}
        return TtsSource(
            id=row["id"], scope=TtsConfigScope(project_id=row["project_id"], version_id=row["version_id"]),
            data_revision=self._revision(row["version_id"]), split=row["split"], path=row["path"],
            revision=row["revision"], state=row["state"], check_id=row["check_id"],
            snapshot_id=row["snapshot_id"] if complete else None,
            checked_at=row["checked_at"] if complete else None,
            summary=TtsSourceSummary.model_validate_json(row["summary_json"]) if complete else None,
            issues=json.loads(row["issues_json"]), issues_total=row["issues_total"] if complete else None,
            issues_truncated=bool(row["issues_truncated"]),
        )

    def _event(self, source: dict, reason: str) -> None:
        removed = reason == "removed"
        event = TtsSourceChanged(
            scope=TtsConfigScope(project_id=source["project_id"], version_id=source["version_id"]),
            source_id=source["id"], split=source["split"], source_revision=source["revision"],
            data_revision=self._revision(source["version_id"]),
            check_id=None if removed else source["check_id"],
            snapshot_id=source["snapshot_id"] if not removed and source["state"] in {"valid", "invalid"} else None,
            state=None if removed else source["state"], reason=reason,
        )
        self.c.bus.publish("tts.source.changed", event.model_dump(mode="json"))

    def list(self, pid: str, vid: str) -> TtsSourcesResponse:
        with self.c.db.lock:
            self._version(pid, vid)
            rows = self.c.db.fetchall("SELECT * FROM tts_sources WHERE version_id=? ORDER BY split", (vid,))
            rows = [self._refresh_engine(row) for row in rows]
            return TtsSourcesResponse(scope=TtsConfigScope(project_id=pid, version_id=vid),
                                      data_revision=self._revision(vid), items=[self._dto(row) for row in rows])

    def get(self, pid: str, vid: str, source_id: str) -> TtsSource:
        with self.c.db.lock:
            return self._dto(self._source(pid, vid, source_id))

    def _deletion_admission(self, path: Path) -> None:
        from .job_paths import deleting_jobs

        reject_deleting_references(self.c, [path])
        deleting_project = self.c.db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name='project_deletions'") and self.c.db.fetchone(
            "SELECT project_id FROM project_deletions WHERE state='deleting' LIMIT 1"
        )
        if deleting_jobs or deleting_project:
            try:
                paths = manifest_references(path, allowed=self.c.is_allowed)
            except (OSError, ValueError) as exc:
                raise ApiError("无法确定清单中的录音引用，请修正清单后再登记。",
                               code="tts.references_unresolved", status=409) from exc
            reject_deleting_references(self.c, paths)

    def put(self, pid: str, vid: str, split: SourceSplit, *, expected_data_revision: int, path: str) -> TtsSourcesResponse:
        if split not in {"train", "validation"}:
            raise ApiError("未知的数据用途。", code="tts.invalid", status=422)
        with self.c.db.lock:
            self._mutable(pid, vid, expected_data_revision)
        try:
            if not path.strip():
                raise SourceFileError("tts.path_invalid", "请填写数据清单路径。")
            candidate = absolute_path(path)
            # Registration validates readable path safety; content diagnostics belong to the scan.
            file_identity(candidate, self.c.is_allowed)
        except (OSError, ValueError) as exc:
            code = exc.code if isinstance(exc, SourceFileError) else "tts.source.read_failed"
            raise ApiError(str(exc), code="tts.path_denied" if code == "tts.path_denied" else "tts.invalid",
                           status=403 if code == "tts.path_denied" else 422,
                           details={"issues": [TtsIssue(code=code, loc=["sources", split, "path"], message=str(exc)).model_dump()]}) from exc
        with self.c.db.transaction():
            self._mutable(pid, vid, expected_data_revision)
            self._deletion_admission(candidate)
            previous = self.c.db.fetchone("SELECT * FROM tts_sources WHERE version_id=? AND split=?", (vid, split))
            fields = {
                "path": str(candidate), "revision": previous["revision"] + 1 if previous else 1,
                "state": "unchecked", "check_id": None, "snapshot_id": None, "checked_at": None,
                "summary_json": None, "issues_json": "[]", "issues_total": None, "issues_truncated": 0,
                "fingerprint": None, "identities_json": "[]", "updated_at": now(),
            }
            sid = previous["id"] if previous else new_id("ts")
            if previous:
                self.c.db.update("tts_sources", sid, fields)
                self.c.db.execute("DELETE FROM tts_source_snapshots WHERE source_id=?", (sid,))
            else:
                self.c.db.insert("tts_sources", {"id": sid, "project_id": pid, "version_id": vid,
                                                "split": split, "created_at": now(), **fields})
            self._bump(vid)
            self._event(self._source(pid, vid, sid), "registered")
            return self.list(pid, vid)

    def remove(self, pid: str, vid: str, split: SourceSplit, *, expected_data_revision: int) -> TtsSourcesResponse:
        with self.c.db.transaction():
            self._mutable(pid, vid, expected_data_revision)
            source = self.c.db.fetchone("SELECT * FROM tts_sources WHERE version_id=? AND split=?", (vid, split))
            if source is None:
                raise ApiError("数据来源不存在。", code="tts.source_not_found", status=404)
            self.c.db.delete("tts_sources", source["id"])
            self._bump(vid)
            self._event(source, "removed")
            return self.list(pid, vid)

    def check(self, pid: str, vid: str, source_id: str, *, expected_data_revision: int) -> TtsSource:
        with self.c.db.transaction():
            version = self._version(pid, vid, write=True)
            self._cas(version, expected_data_revision)
            source = self._source(pid, vid, source_id)
            self._cas(self._version(pid, vid, write=True), expected_data_revision)
            if source["state"] == "checking":
                return self._dto(source)
            if self._closed:
                raise ApiError("数据检查服务正在关闭。", code="tts.check.unavailable", status=503)
            check_id = new_id("tc")
            engine = self._engine(pid, vid)
            self.c.db.update("tts_sources", source_id, {
                "state": "checking", "check_id": check_id, "snapshot_id": None, "checked_at": None,
                "summary_json": None, "issues_json": "[]", "issues_total": None, "issues_truncated": 0,
                "updated_at": now(),
            })
            checking = self._source(pid, vid, source_id)
            result = self._dto(checking)
            self._event(checking, "check_started")
            self.executor.submit(self._scan, source, check_id, engine)
            return result

    def _matches(self, source: dict, check_id: str) -> dict | None:
        current = self.c.db.fetchone("SELECT * FROM tts_sources WHERE id=?", (source["id"],))
        if current and current["revision"] == source["revision"] and current["check_id"] == check_id and current["state"] == "checking":
            return current
        return None

    def _failure(self, source: dict, code: str, message: str) -> None:
        issue = TtsIssue(code=code, loc=["sources", source["split"], "path"], message=message)
        self.c.db.update("tts_sources", source["id"], {
            "state": "error", "snapshot_id": None, "summary_json": None, "checked_at": None,
            "issues_json": _json([issue.model_dump()]), "issues_total": None, "issues_truncated": 0,
            "updated_at": now(),
        })
        updated = self.c.db.fetchone("SELECT * FROM tts_sources WHERE id=?", (source["id"],))
        self._event(updated, "check_failed")

    def _scan(self, source: dict, check_id: str, engine: str = "voxcpm1.5") -> None:
        try:
            result = scan_manifest(Path(source["path"]), source["id"], source["split"], allowed=self.c.is_allowed, engine=engine)
            changes = identity_changes(result.identities, allowed=self.c.is_allowed)
            with self.c.db.transaction():
                current = self._matches(source, check_id)
                if current is None:
                    return
                self._version(source["project_id"], source["version_id"], write=True)
                if engine != self._engine(source["project_id"], source["version_id"]):
                    self._stale(current, message="训练引擎已变化，请重新检查数据来源。")
                    return
                if changes:
                    self._stale(current)
                    return
                reject_deleting_references(self.c, [Path(value["path"]) for value in result.identities])
                snapshot = "tss_" + hashlib.sha256((source["id"] + result.fingerprint).encode()).hexdigest()
                if not self.c.db.fetchone("SELECT id FROM tts_source_snapshots WHERE id=?", (snapshot,)):
                    self.c.db.insert("tts_source_snapshots", {"id": snapshot, "source_id": source["id"],
                                                             "fingerprint": result.fingerprint, "created_at": now(),
                                                             "engine": result.engine})
                    for item in result.rows:
                        data = item["row"]
                        row_id = "row_" + str(data["line"])
                        dto = TtsSourceRow(
                            **data, id=row_id, source_id=source["id"], snapshot_id=snapshot,
                            audio_url=self._audio_url(source, snapshot, row_id, "audio") if "audio" in item["assets"] else None,
                            reference_audio_url=self._audio_url(source, snapshot, row_id, "ref_audio") if "ref_audio" in item["assets"] else None,
                            issues=item["issues"],
                        )
                        self.c.db.insert("tts_source_rows", {"id": row_id, "snapshot_id": snapshot, "line": data["line"],
                                                           "row_json": dto.model_dump_json(),
                                                           "normalized_json": _json(item["normalized"]) if item["normalized"] is not None else None,
                                                           "assets_json": _json(item["assets"])})
                if result.fingerprint != source["fingerprint"] and source["state"] != "stale":
                    self._bump(source["version_id"])
                self.c.db.update("tts_sources", source["id"], {
                    "state": "invalid" if any(i.severity == "error" for i in result.issues) or result.summary.invalid_count else "valid",
                    "snapshot_id": snapshot, "checked_at": now(), "summary_json": result.summary.model_dump_json(),
                    "issues_json": _json([issue.model_dump() for issue in result.issues]),
                    "issues_total": result.issues_total, "issues_truncated": int(result.issues_truncated),
                    "fingerprint": result.fingerprint, "identities_json": _json(result.identities), "updated_at": now(),
                })
                self._event(self.c.db.fetchone("SELECT * FROM tts_sources WHERE id=?", (source["id"],)), "check_completed")
        except Exception as exc:
            with self.c.db.transaction():
                current = self._matches(source, check_id)
                if current:
                    code = exc.code if isinstance(exc, (ApiError, SourceFileError)) else "tts.source.read_failed"
                    if code == "tts.source_stale":
                        self._stale(current)
                    else:
                        self._failure(current, code, str(exc) or "无法完成数据来源检查。")

    def _audio_url(self, source: dict, snapshot: str, row_id: str, role: str) -> str:
        return (
            f"/api/tts/projects/{quote(source['project_id'], safe='')}/versions/{quote(source['version_id'], safe='')}"
            f"/sources/{quote(source['id'], safe='')}/rows/{row_id}/audio?"
            + urlencode({"snapshot_id": snapshot, "role": role})
        )

    def _snapshot(self, source: dict, snapshot: str) -> None:
        if source["state"] == "stale" or (source["snapshot_id"] and source["snapshot_id"] != snapshot):
            raise ApiError("数据来源快照已变化，请重新读取。", code="tts.source_stale", status=409)
        if source["state"] not in {"valid", "invalid"} or not source["snapshot_id"]:
            raise ApiError("数据来源尚无完整检查结果。", code="tts.source_not_ready", status=409,
                           details={"state": source["state"]})

    def rows(self, pid: str, vid: str, source_id: str, *, snapshot_id: str, page: int = 1, page_size: int = 50) -> TtsRowsResponse:
        if type(page) is not int or page < 1 or type(page_size) is not int or not 1 <= page_size <= 200:
            raise ApiError("页码须为正整数，每页条数须为 1–200。", code="tts.invalid", status=422)
        with self.c.db.lock:
            source = self._source(pid, vid, source_id)
            self._snapshot(source, snapshot_id)
            total = json.loads(source["summary_json"])["clips_count"]
            offset = (page - 1) * page_size
            items = [] if offset >= total else self.c.db.fetchall(
                "SELECT row_json FROM tts_source_rows WHERE snapshot_id=? ORDER BY line LIMIT ? OFFSET ?",
                (snapshot_id, page_size, offset),
            )
            return TtsRowsResponse(source_id=source_id, snapshot_id=snapshot_id, page=page, page_size=page_size,
                                   total=total,
                                   items=[TtsSourceRow.model_validate_json(item["row_json"]) for item in items])

    def _stale(self, source: dict, *, message: str = "清单或录音内容已变化，请重新检查。") -> None:
        if source["state"] != "stale":
            self._bump(source["version_id"])
        issue = TtsIssue(code="tts.source_stale", loc=["sources", source["split"], "path"],
                         message=message)
        self.c.db.update("tts_sources", source["id"], {
            "state": "stale", "snapshot_id": None, "checked_at": None, "summary_json": None,
            "issues_json": _json([issue.model_dump()]), "issues_total": None, "issues_truncated": 0,
            "updated_at": now(),
        })
        self._event(self.c.db.fetchone("SELECT * FROM tts_sources WHERE id=?", (source["id"],)), "stale")

    def audio_asset(self, pid: str, vid: str, source_id: str, row_id: str, *, snapshot_id: str, role: str = "audio") -> dict:
        if role not in {"audio", "ref_audio"}:
            raise ApiError("未知的录音角色。", code="tts.invalid", status=422)
        with self.c.db.lock:
            source = self._source(pid, vid, source_id)
            self._snapshot(source, snapshot_id)
            row = self.c.db.fetchone("SELECT assets_json FROM tts_source_rows WHERE snapshot_id=? AND id=?", (snapshot_id, row_id))
            asset = json.loads(row["assets_json"]).get(role) if row else None
            if asset is None:
                raise ApiError("这一行没有可预览的录音。", code="tts.audio.not_found", status=404)
            manifest = next(value for value in json.loads(source["identities_json"]) if value["path"] == source["path"])
        try:
            for expected in (manifest, asset):
                actual = file_identity(Path(expected["path"]), self.c.is_allowed)
                if actual != expected:
                    raise SourceFileError("tts.source_stale", "清单或录音内容已变化，请重新检查。")
        except (OSError, SourceFileError) as exc:
            code = exc.code if isinstance(exc, SourceFileError) else (
                "tts.audio.not_found" if isinstance(exc, FileNotFoundError) else "tts.source_stale"
            )
            with self.c.db.transaction():
                current = self._source(pid, vid, source_id)
                if current["snapshot_id"] == snapshot_id:
                    self._stale(current)
            raise ApiError(str(exc), code=code, status=403 if code == "tts.path_denied" else 404 if code == "tts.audio.not_found" else 409) from exc
        with self.c.db.lock:
            self._snapshot(self._source(pid, vid, source_id), snapshot_id)
            return {**asset, "path": Path(asset["path"])}

    def audio(self, pid: str, vid: str, source_id: str, row_id: str, *, snapshot_id: str, role: str = "audio") -> Path:
        return self.audio_asset(pid, vid, source_id, row_id, snapshot_id=snapshot_id, role=role)["path"]

    def snapshot_identity(self, pid: str, vid: str) -> dict:
        with self.c.db.lock:
            self._version(pid, vid)
            sources = self.c.db.fetchall("SELECT * FROM tts_sources WHERE version_id=? ORDER BY split", (vid,))
            sources = [self._refresh_engine(source) for source in sources]
            return {"data_revision": self._revision(vid), "sources": {
                source["split"]: {key: source[key] for key in ("id", "revision", "state", "snapshot_id", "fingerprint")}
                for source in sources
            }}

    def validation_inputs(self, pid: str, vid: str, *, recheck: bool = True) -> dict:
        with self.c.db.lock:
            identity = self.snapshot_identity(pid, vid)
            sources = self.c.db.fetchall("SELECT * FROM tts_sources WHERE version_id=? ORDER BY split", (vid,))
        changed = []
        if recheck:
            for source in sources:
                if source["state"] in {"valid", "invalid"} and identity_changes(json.loads(source["identities_json"]), allowed=self.c.is_allowed):
                    changed.append(source["id"])
        with self.c.db.transaction():
            if identity != self.snapshot_identity(pid, vid):
                raise ApiError("数据来源在校验期间发生变化，请重新校验。", code="tts.data_conflict", status=409,
                               details={"current_data_revision": self._revision(vid)})
            for source in sources:
                if source["id"] in changed:
                    self._stale(source)
            result: dict[str, Any] = {"data_revision": self._revision(vid), "train": None, "validation": None}
            for source in self.c.db.fetchall("SELECT * FROM tts_sources WHERE version_id=? ORDER BY split", (vid,)):
                complete = source["state"] in {"valid", "invalid"}
                normalized = []
                if source["state"] == "valid":
                    normalized = [json.loads(item["normalized_json"]) for item in self.c.db.fetchall(
                        "SELECT normalized_json FROM tts_source_rows WHERE snapshot_id=? ORDER BY line", (source["snapshot_id"],)
                    )]
                result[source["split"]] = {"source": self._dto(source), "rows": normalized,
                                           "fingerprint": source["fingerprint"] if complete else None,
                                           "fingerprints": [item for item in json.loads(source["identities_json"]) if item.get("sha256")]
                                           if complete else []}
            return result

    def audio_stats(self, pid: str, vid: str) -> dict:
        with self.c.db.lock:
            self.c.require_project_type(pid, "tts")
            self.c.resolve_version(pid, vid)
            sources = self.c.db.fetchall("SELECT * FROM tts_sources WHERE version_id=? ORDER BY split", (vid,))
            result: dict[str, Any] = {"train": {"state": "missing", "clips_count": None, "duration_seconds": None}, "validation": None}
            for row in sources:
                source = self._dto(row)
                result[source.split] = {"state": source.state,
                                        "clips_count": source.summary.clips_count if source.summary else None,
                                        "duration_seconds": source.summary.duration_seconds if source.summary else None}
            return result

    def references_to(self, targets: list[Path], *, excluding_project_id: str | None = None) -> list[dict]:
        roots = [absolute_path(path).resolve() for path in targets]
        sources = self.c.db.fetchall("SELECT * FROM tts_sources ORDER BY id")
        references = []
        for source in sources:
            if source["project_id"] == excluding_project_id:
                continue
            path = Path(source["path"])
            try:
                paths = manifest_references(path, allowed=self.c.is_allowed)
                resolved_paths = [(dependency, dependency.resolve()) for dependency in paths]
            except (OSError, ValueError, RuntimeError) as exc:
                # A manifest stored inside the deleted tree is itself an established dependency.
                if any(path == target or path.is_relative_to(target) for target in roots):
                    resolved_paths = [(path, path)]
                else:
                    raise ApiError("无法确定已登记清单的录音引用，请修正或移除登记后重试。",
                                   code="tts.references_unresolved", status=409,
                                   details={"blocker": {"source_id": source["id"], "project_id": source["project_id"], "version_id": source["version_id"]}}) from exc
            for dependency, resolved in resolved_paths:
                if any(resolved == target or resolved.is_relative_to(target) for target in roots):
                    references.append({"source_id": source["id"], "project_id": source["project_id"],
                                       "version_id": source["version_id"], "path": str(dependency)})
        return references
