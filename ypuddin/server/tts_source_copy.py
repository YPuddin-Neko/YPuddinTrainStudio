"""Owned audio copies and scan publication for speech project versions."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ypuddin.tts.source_models import TtsSourceRow
from ypuddin.tts.source_scan import ScanResult, absolute_path, open_source, scan_manifest

from .db import new_id, now
from .errors import ApiError
from .tts_references import reject_deleting_references


def _path_error() -> ApiError:
    return ApiError("版本目录已改变或无法安全访问。", code="version.path", status=409)


def _directory_key(path: Path) -> tuple[int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise _path_error() from exc
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        raise _path_error()
    return info.st_dev, info.st_ino


@dataclass(frozen=True)
class DirectoryGuard:
    path: Path
    chain: tuple[tuple[Path, tuple[int, int]], ...]

    @classmethod
    def capture(cls, path: Path, allowed: Any) -> DirectoryGuard:
        path = absolute_path(path)
        if not allowed(path):
            raise _path_error()
        chain = tuple((item, _directory_key(item)) for item in (*reversed(path.parents), path))
        return cls(path, chain)

    def check(self, allowed: Any) -> None:
        if not allowed(self.path) or any(_directory_key(path) != key for path, key in self.chain):
            raise _path_error()

    @contextmanager
    def opened(self, allowed: Any):
        self.check(allowed)
        fd = None
        try:
            if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
                for path, key in self.chain:
                    child = os.open(path if fd is None else path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    if fd is not None:
                        os.close(fd)
                    fd = child
                    info = os.fstat(fd)
                    if (info.st_dev, info.st_ino) != key:
                        raise _path_error()
            yield fd
            self.check(allowed)
        finally:
            if fd is not None:
                os.close(fd)

    def child(self, name: str, allowed: Any) -> DirectoryGuard:
        with self.opened(allowed) as fd:
            os.mkdir(name if fd is not None else self.path / name, dir_fd=fd)
        return self.capture(self.path / name, allowed)

    @contextmanager
    def new_file(self, name: str, allowed: Any):
        if Path(name).name != name:
            raise _path_error()
        with self.opened(allowed) as fd:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
            stream_fd = os.open(name if fd is not None else self.path / name, flags, 0o600, dir_fd=fd)
            with os.fdopen(stream_fd, "wb") as stream:
                yield stream
                stream.flush()
                os.fsync(stream.fileno())


def prepare(c: Any, pid: str, vid: str) -> dict:
    """Recheck registered sources before admitting a data-copy operation."""
    inputs = c.tts_sources.validation_inputs(pid, vid, recheck=True)
    sources = {}
    identity = {"data_revision": inputs["data_revision"], "sources": {}}
    for split in ("train", "validation"):
        item = inputs[split]
        if item is None:
            continue
        source = item["source"]
        if source.state != "valid":
            code = "tts.source_stale" if source.state == "stale" else "tts.source_not_ready"
            raise ApiError("复制数据需要每份已登记来源均已检查通过且内容未变化。", code=code, status=409,
                           details={"source_id": source.id, "split": split})
        if not source.snapshot_id or not item["fingerprint"] or not item["rows"] or not item["fingerprints"]:
            raise ApiError("数据来源缺少完整检查快照，请重新检查。", code="tts.source_not_ready", status=409)
        identity["sources"][split] = {"id": source.id, "revision": source.revision, "state": source.state,
                                      "snapshot_id": source.snapshot_id, "fingerprint": item["fingerprint"]}
        sources[split] = item
    return {"identity": identity, "sources": sources}


def assert_identity(c: Any, pid: str, vid: str, plan: dict | None) -> None:
    if plan is not None and c.tts_sources.snapshot_identity(pid, vid) != plan["identity"]:
        raise ApiError("数据来源在复制期间发生变化，请重新检查后再复制。", code="tts.source_stale", status=409)


def recheck(c: Any, pid: str, vid: str, plan: dict | None) -> None:
    if plan is not None:
        current = prepare(c, pid, vid)
        if current["identity"] != plan["identity"]:
            raise ApiError("数据来源在复制期间发生变化，请重新检查后再复制。", code="tts.source_stale", status=409)


def dependency_paths(plan: dict | None) -> list[Path]:
    return list(dict.fromkeys(Path(value["path"]) for item in (plan or {}).get("sources", {}).values()
                              for value in item["fingerprints"]))


def copy_data(c: Any, plan: dict | None, staging: DirectoryGuard, final: Path, progress: Any) -> list[dict]:
    """Copy bytes through fixed file handles and preserve exact transcript strings."""
    if not plan:
        return []
    identities = {value["path"]: value for item in plan["sources"].values() for value in item["fingerprints"]}
    names: dict[str, str] = {}
    for item in plan["sources"].values():
        for row in item["rows"]:
            for role in ("audio", "ref_audio"):
                path = row.get(role)
                if path and path not in names:
                    if path not in identities:
                        raise ApiError("录音缺少检查身份，请重新检查。", code="tts.source_not_ready", status=409)
                    names[path] = f"audio-{len(names) + 1:06d}.wav"
    manifests = []
    for split, item in plan["sources"].items():
        rows = [{**row, **{role: str(final / names[row[role]]) for role in ("audio", "ref_audio") if row.get(role)}}
                for row in item["rows"]]
        content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")
        manifests.append({"id": new_id("ts"), "split": split, "name": split + ".jsonl", "content": content})
    progress(files_total=len(names) + len(manifests), bytes_total=sum(identities[path]["size"] for path in names)
             + sum(len(item["content"]) for item in manifests))
    for path, name in names.items():
        expected = identities[path]
        digest, size = hashlib.sha256(), 0
        with open_source(Path(path), c.is_allowed) as source, staging.new_file(name, c.is_allowed) as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                progress(bytes_done=len(chunk))
        if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
            raise ApiError("录音在复制期间发生变化，请重新检查。", code="tts.source_stale", status=409)
        progress(files_done=1)
    for item in manifests:
        with staging.new_file(item["name"], c.is_allowed) as output:
            output.write(item["content"])
        progress(files_done=1, bytes_done=len(item["content"]))
    return [{"id": item["id"], "split": item["split"], "path": final / item["name"]} for item in manifests]


def scan_copies(c: Any, entries: list[dict], *, engine: str = "voxcpm1.5") -> list[dict]:
    result = []
    for entry in entries:
        scan = scan_manifest(entry["path"], entry["id"], entry["split"], allowed=c.is_allowed, engine=engine)
        if scan.summary.invalid_count or any(issue.severity == "error" for issue in scan.issues):
            raise ApiError("复制后的录音或清单未通过检查。", code="tts.source_not_ready", status=409)
        result.append({**entry, "scan": scan})
    return result


def publish(c: Any, pid: str, vid: str, entries: list[dict]) -> None:
    """Called inside the transaction that makes the new version ready."""
    for entry in entries:
        result: ScanResult = entry["scan"]
        reject_deleting_references(c, [Path(value["path"]) for value in result.identities])
        sid = entry["id"]
        snapshot = "tss_" + hashlib.sha256((sid + result.fingerprint).encode()).hexdigest()
        source = {"id": sid, "project_id": pid, "version_id": vid, "split": entry["split"],
                  "path": str(entry["path"]), "revision": 1, "state": "valid", "check_id": new_id("tsc"),
                  "snapshot_id": snapshot, "checked_at": now(), "summary_json": result.summary.model_dump_json(),
                  "issues_json": json.dumps([issue.model_dump() for issue in result.issues], ensure_ascii=False),
                  "issues_total": result.issues_total, "issues_truncated": int(result.issues_truncated),
                  "fingerprint": result.fingerprint, "identities_json": json.dumps(result.identities),
                  "created_at": now(), "updated_at": now()}
        c.db.insert("tts_sources", source)
        c.db.insert("tts_source_snapshots", {"id": snapshot, "source_id": sid,
                                            "fingerprint": result.fingerprint, "created_at": now(), "engine": result.engine})
        for item in result.rows:
            row_id = "row_" + str(item["row"]["line"])
            dto = TtsSourceRow(**item["row"], id=row_id, source_id=sid, snapshot_id=snapshot,
                               audio_url=c.tts_sources._audio_url(source, snapshot, row_id, "audio") if "audio" in item["assets"] else None,
                               reference_audio_url=c.tts_sources._audio_url(source, snapshot, row_id, "ref_audio") if "ref_audio" in item["assets"] else None,
                               issues=item["issues"])
            c.db.insert("tts_source_rows", {"id": row_id, "snapshot_id": snapshot, "line": item["row"]["line"],
                                           "row_json": dto.model_dump_json(),
                                           "normalized_json": json.dumps(item["normalized"], ensure_ascii=False, allow_nan=False),
                                           "assets_json": json.dumps(item["assets"])})


def active_references(c: Any, targets: list[Path], *, excluding_project_id: str | None = None) -> list[dict]:
    """Retain frozen dependencies even if a copying source's JSONL is edited externally."""
    roots = [absolute_path(path).resolve() for path in targets]
    references = []
    for source in c.db.fetchall("""SELECT s.* FROM tts_sources s
        JOIN project_versions parent ON parent.id=s.version_id
        JOIN project_versions target ON target.id=parent.busy AND target.status='copying'"""):
        if source["project_id"] == excluding_project_id:
            continue
        try:
            identities = json.loads(source["identities_json"])
            if not isinstance(identities, list) or not identities:
                raise ValueError("missing frozen source identities")
            for item in identities:
                path = Path(item["path"])
                if not path.is_absolute():
                    raise ValueError("frozen source path is not absolute")
                resolved = path.resolve()
                if any(resolved == root or resolved.is_relative_to(root) for root in roots):
                    references.append({"source_id": source["id"], "project_id": source["project_id"],
                                       "version_id": source["version_id"], "path": str(path)})
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise ApiError(
                "无法确定正在复制的数据来源引用，请等待复制结束并重新检查来源后再删除。",
                code="tts.references_unresolved", status=409,
                details={"blocker": {"source_id": source["id"], "project_id": source["project_id"],
                                     "version_id": source["version_id"], "path": source["path"]}},
            ) from exc
    return references


def new_staging(parent: DirectoryGuard, vid: str, allowed: Any) -> DirectoryGuard:
    return parent.child(f".copy-{vid}-{uuid4().hex}", allowed)
