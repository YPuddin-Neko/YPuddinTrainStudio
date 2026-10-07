"""Persistent speech job request identities and restart recovery."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from .db import new_id, now
from .errors import ApiError


def recover_pending(c: Any) -> None:
    """Called once during app construction, before any request is admitted."""
    c.db.execute("DELETE FROM tts_requests WHERE state='pending'")


def reserve(
    c: Any, pid: str | None, vid: str | None, action: str, key: str, payload: dict, *, target_job_id: str | None = None
) -> tuple[str, dict | None]:
    try:
        normalized_key = str(UUID(key))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ApiError(
            "Idempotency-Key 必须是 UUID。",
            code="tts.invalid",
            status=422,
            details={
                "issues": [
                    {
                        "code": "tts.idempotency_key_invalid",
                        "loc": ["header", "Idempotency-Key"],
                        "message": "请为这次操作提供 UUID 请求标识。",
                        "severity": "error",
                        "details": {},
                    }
                ]
            },
        ) from exc
    scope = json.dumps([pid, vid], separators=(",", ":"))
    identity = json.dumps(
        {"scope": [pid, vid], "action": action, "payload": payload},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    with c.db.transaction():
        row = c.db.fetchone(
            "SELECT * FROM tts_requests WHERE scope=? AND action=? AND request_key=?",
            (scope, action, normalized_key),
        )
        if row is not None:
            if row["request_hash"] != fingerprint:
                raise ApiError("这个请求标识已用于其他内容。", code="tts.idempotency_conflict", status=409)
            if row["state"] == "pending":
                raise ApiError("这个请求正在处理。", code="tts.request_pending", status=409)
            job = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (row["job_id"],))
            if job is None:
                raise ApiError(
                    "该请求创建的任务已被删除。", code="tts.idempotency_result_deleted", status=410
                )
            return row["id"], job
        rid = new_id("tr")
        timestamp = now()
        c.db.insert(
            "tts_requests",
            {
                "id": rid,
                "scope": scope,
                "action": action,
                "request_key": normalized_key,
                "request_hash": fingerprint,
                "state": "pending",
                "job_id": None,
                "target_job_id": target_job_id,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        return rid, None


def complete(c: Any, request_id: str, job_id: str) -> None:
    """The caller holds the transaction that also inserts the job."""
    changed = c.db.execute(
        "UPDATE tts_requests SET state='completed',job_id=?,updated_at=? WHERE id=? AND state='pending'",
        (job_id, now(), request_id),
    )
    if changed.rowcount != 1:
        raise ApiError("请求状态已改变，请重试。", code="tts.request_pending", status=409)


def release(c: Any, request_id: str) -> None:
    c.db.execute("DELETE FROM tts_requests WHERE id=? AND state='pending'", (request_id,))
