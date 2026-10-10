"""Persistent speech environments; installation never targets the running service."""

from __future__ import annotations

import copy
import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ypuddin.tts.environment_models import (
    TtsEnvironmentCheckResult,
    TtsEnvironmentIdentity,
    TtsEnvironmentOperation,
    TtsEnvironmentSnapshot,
    TtsManagedEnvironment,
    TtsPythonCandidate,
)
from ypuddin.tts.issues import TtsIssue

from .db import new_id, now
from .errors import ApiError

STATE_KEY = "tts.environments.v1"
ACTIVE = {"queued", "running"}
ENGINES = ("voxcpm1.5", "gpt-sovits-v5")
MARKER = ".ypuddin-tts-environment.json"


def _error(code, message, status=422, *, field="environment"):
    issue = TtsIssue(code=f"tts.environment.{code}", loc=[field], message=message)
    return ApiError(message, code=issue.code, status=status, details={"issues": [issue.model_dump()]})


def _path_key(value):
    # Resolving the executable symlink would erase the virtual environment entry point.
    return os.path.normcase(os.path.abspath(value))


def _identity_digest(value):
    from ypuddin.tts.environment_binding import identity_digest

    return identity_digest(value)


def _marker_digest(trainer):
    from ypuddin.tts.environment_resources import marker_identity

    return marker_identity(trainer)


def _failure_summary(message, phase):
    if len(message) <= 240 and "\n" not in message and "TTS_ENVIRONMENT " not in message:
        return message
    if match := re.search(r"无法加载 ([A-Za-z0-9_.]+)：", message):
        return f"无法加载语音依赖 {match.group(1)}，请查看操作日志。"
    return {
        "checking": "Python 环境检查失败，请查看操作日志。",
        "source": "训练器源码或资源准备失败，请查看操作日志。",
        "dependencies": "语音环境依赖安装失败，请查看操作日志后重试。",
        "verifying": "语音环境检查未通过，请查看操作日志。",
    }.get(phase, "语音环境准备失败，请查看操作日志后重试。")


def _directory(path):
    for part in (path, *path.parents):
        try:
            value = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode) or (
            getattr(value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise _error("path", "语音环境目录不能使用链接或非目录文件。")


def _source(engine, path):
    if engine == "gpt-sovits-v5":
        from ypuddin.tts.gpt_sovits.core import validate_upstream
    else:
        from ypuddin.tts.core import validate_upstream
    return validate_upstream(path)


class TtsEnvironments:
    def __init__(self, context, torch_environments=None, *, prepare=None, probe=None):
        from ypuddin.tts.environment_probe import probe_environment

        from .tts_environment_install import prepare as install

        self.context = context
        self.torch_environments = torch_environments
        self.prepare = prepare or install
        self.probe = probe or probe_environment
        self.root = context.data_root / "tts" / "environments"
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-environments")
        self.cancels = {}
        self.closed = False
        self.state = context.db.get_kv(STATE_KEY, {
            "environments": {}, "defaults": {}, "operations": {}, "checks": {},
        })
        for key in ("environments", "defaults", "operations", "checks"):
            self.state.setdefault(key, {})
        with context.db.lock:
            for op in self.state["operations"].values():
                if op["status"] in ACTIVE:
                    op.update(status="failed", phase="interrupted", updated_at=now(), cancellable=False,
                              issues=[TtsIssue(code="tts.environment.interrupted", loc=["environment"],
                                               message="环境准备已中断，请重试。").model_dump()])
            self._save()
        if context.background_tasks is not None:
            context.background_tasks.add_source(self.background_tasks, cancel=self.cancel)

    def _save(self):
        self.context.db.set_kv(STATE_KEY, self.state)

    def _replace(self, value):
        self.context.db.set_kv(STATE_KEY, value)
        self.state = value

    def _event(self, op=None):
        self.context.bus.publish("tts.environment.changed", {
            "operation_id": op["id"] if op else None,
            "engine": op["engine"] if op else None,
            "status": op["status"] if op else None,
        })

    def _update(self, id_, **values):
        with self.context.db.lock:
            op = self.state["operations"][id_]
            op.update(**values, updated_at=now())
            op["cancellable"] = op["status"] in ACTIVE
            self._save()
            result = copy.deepcopy(op)
        self._event(result)
        return TtsEnvironmentOperation.model_validate(result)

    def _log(self, id_, message):
        from .network import ProxyPolicy

        message = ProxyPolicy.from_context(self.context).redact(str(message))[-4000:]
        with self.context.db.lock:
            op = self.state["operations"][id_]
            op["logs"] = [*op["logs"], message][-200:]
            op["log_count"] += 1
            op["logs_truncated"] = op["log_count"] > len(op["logs"])
            op["updated_at"] = now()
            self._save()
        return message

    def candidates(self):
        rows = [{"id": "deployment", "kind": "deployment", "python_path": sys.executable}]
        if self.torch_environments is not None:
            for op in self.torch_environments.list():
                if op.status != "completed":
                    continue
                try:
                    python = self.torch_environments.resolve(op.id)
                except (ApiError, OSError, ValueError):
                    continue
                rows.append({"id": f"torch:{op.id}", "kind": "managed", "python_path": python})
        with self.context.db.lock:
            saved = copy.deepcopy(self.state)
        rows.extend({"id": f"tts:{id_}", "kind": row["kind"], "python_path": row["python_path"]}
                    for id_, row in saved["environments"].items())
        for op in saved["operations"].values():
            if op["action"] != "prepare" or op["status"] not in {"failed", "cancelled"}:
                continue
            work = self.root / op["id"]
            owner = work / ".owner.json"
            python = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            try:
                _directory(python.parent)
                if (owner.is_symlink() or not owner.is_file() or owner.stat().st_size > 4096
                        or json.loads(owner.read_text("utf-8")) != {"id": op["id"], "engine": op["engine"]}
                        or not python.is_file()):
                    continue
            except (ApiError, OSError, ValueError):
                continue
            rows.append({"id": f"attempt:{op['id']}", "kind": "managed", "python_path": str(python)})
        result, seen = [], set()
        for row in rows:
            key = _path_key(row["python_path"])
            if key in seen:
                continue
            seen.add(key)
            row["available"] = Path(row["python_path"]).is_file()
            row["checks"] = [value for id_, value in saved["checks"].items()
                             if id_.startswith(row["id"] + "|")]
            result.append(TtsPythonCandidate.model_validate(row))
        return result

    def snapshot(self):
        candidates = self.candidates()
        with self.context.db.lock:
            value = copy.deepcopy(self.state)
        environments = []
        for row in value["environments"].values():
            if not Path(row["python_path"]).is_file() or not Path(row["trainer_path"]).is_dir():
                row["state"] = "missing"
                row["issues"] = [TtsIssue(code="tts.environment.missing", loc=["environment"],
                                           message="环境文件已不存在，请重新准备。").model_dump()]
            environments.append(TtsManagedEnvironment.model_validate(row))
        ops = sorted(value["operations"].values(), key=lambda op: op["created_at"], reverse=True)[:100]
        for op in ops:
            op["logs"] = []
            op["logs_truncated"] = op["log_count"] > 0
        return TtsEnvironmentSnapshot(
            candidates=candidates, environments=environments,
            defaults={engine: value["defaults"].get(engine) for engine in ENGINES}, operations=ops,
        )

    def get(self, id_):
        with self.context.db.lock:
            value = self.state["operations"].get(id_)
            if value is None:
                raise _error("operation_not_found", "环境操作不存在。", 404)
            return TtsEnvironmentOperation.model_validate(copy.deepcopy(value))

    def start(self, action, engine, candidate_id=None, *, make_default=True, retry_of=None):
        candidates = self.candidates()
        if candidate_id is not None and candidate_id not in {row.id for row in candidates}:
            raise _error("candidate_not_found", "Python 环境不存在，请刷新后重试。", 404)
        with self.context.db.lock:
            if self.closed:
                raise _error("closing", "训练器正在关闭。", 409)
            maintenance = self.context.db.get_kv("environment.maintenance", {})
            if maintenance.get("blocked") or maintenance.get("restarting") or maintenance.get("trainer_update"):
                raise _error("busy", "运行环境正在维护，请稍后重试。", 409)
            for op in self.state["operations"].values():
                if op["status"] in ACTIVE and op["engine"] == engine:
                    if (op["action"], op["candidate_id"], op["make_default"]) == (action, candidate_id, make_default):
                        return TtsEnvironmentOperation.model_validate(copy.deepcopy(op))
                    raise _error("busy", "此语音引擎已有环境操作，请等待完成或取消。", 409)
            id_, timestamp = new_id("te"), now()
            op = TtsEnvironmentOperation(
                id=id_, action=action, engine=engine, candidate_id=candidate_id, make_default=make_default,
                status="queued", phase="queued", created_at=timestamp, updated_at=timestamp,
                retry_of=retry_of, cancellable=True,
            )
            staged = copy.deepcopy(self.state)
            staged["operations"][id_] = op.model_dump()
            self._replace(staged)
            cancel = self.cancels[id_] = threading.Event()
            self.context._active_tts_environments += 1
            try:
                self.pool.submit(self._run, id_, [row.model_dump() for row in candidates], cancel)
            except RuntimeError:
                self.context._active_tts_environments -= 1
                self.cancels.pop(id_, None)
                self._update(id_, status="failed", phase="failed", issues=[TtsIssue(
                    code="tts.environment.closing", loc=["environment"], message="训练器正在关闭，请稍后重试。",
                ).model_dump()])
                raise _error("closing", "训练器正在关闭。", 409) from None
        self._event(op.model_dump())
        return op

    def _check_result(self, engine, report):
        fields = set(TtsEnvironmentCheckResult.model_fields) - {"engine", "checked_at"}
        return TtsEnvironmentCheckResult(engine=engine, checked_at=now(), **{
            key: value for key, value in report.items() if key in fields
        })

    def _run_checks(self, op, chosen, work, cancel):
        with self.context.db.lock:
            available = copy.deepcopy([row for row in self.state["environments"].values()
                                       if row["engine"] == op.engine])
        updates, checks, issues = {}, {}, []
        rank = {"ready": 0, "unchecked": 1, "missing": 2, "incompatible": 3, "error": 4}
        for candidate in chosen:
            matching = [row for row in available if _path_key(row["python_path"]) == _path_key(candidate["python_path"])]
            trainers = list(dict.fromkeys(row["trainer_path"] for row in matching)) or [None]
            results = []
            for trainer in trainers:
                if cancel.is_set():
                    raise InterruptedError
                self._update(op.id, phase="checking")
                report = self.probe(candidate["python_path"], op.engine, trainer, cancel=cancel, work_dir=work)
                if cancel.is_set():
                    raise InterruptedError
                check = self._check_result(op.engine, report)
                results.append(check)
                issues.extend(check.issues)
                self._log(op.id, f"{candidate['id']}: {check.state}")
                for issue in check.issues:
                    self._log(op.id, issue.message)
                for row in matching:
                    if row["trainer_path"] != trainer:
                        continue
                    unchanged = (check.dependency_fingerprint == row["dependency_fingerprint"]
                                 and _marker_digest(trainer) == row.get("resources_marker_sha256"))
                    row_issues = [issue.model_dump() for issue in check.issues]
                    if not unchanged:
                        row_issues.append(TtsIssue(code="tts.environment.changed", loc=["environment"],
                                                  message="环境依赖或资源已变化，请重新准备。").model_dump())
                    row.update(state="ready" if check.state == "ready" and unchanged else "changed",
                               checked_at=check.checked_at, check=check.model_dump(), issues=row_issues)
                    updates[row["id"]] = row
            aggregate = max(results, key=lambda item: rank[item.state]).model_copy(deep=True)
            aggregate.issues = [issue for item in results for issue in item.issues]
            aggregate.source_checked = all(item.source_checked for item in results)
            checks[f"{candidate['id']}|{op.engine}"] = aggregate.model_dump()
        with self.context.db.lock:
            if cancel.is_set():
                raise InterruptedError
            staged = copy.deepcopy(self.state)
            staged["environments"].update(updates)
            staged["checks"].update(checks)
            staged["operations"][op.id].update(status="completed", phase="checked", progress=1.0,
                                                updated_at=now(), cancellable=False,
                                                issues=[issue.model_dump() for issue in issues])
            self._replace(staged)
            terminal = copy.deepcopy(staged["operations"][op.id])
        self._event(terminal)

    def _run(self, id_, candidates, cancel):
        try:
            op = self.get(id_)
            if cancel.is_set():
                raise InterruptedError
            self._update(id_, status="running", phase="discovering")
            chosen = [row for row in candidates if op.candidate_id is None or row["id"] == op.candidate_id]
            _directory(self.root)
            self.root.mkdir(parents=True, exist_ok=True)
            work = self.root / id_
            work.mkdir(exist_ok=False)
            (work / ".owner.json").write_text(json.dumps({"id": id_, "engine": op.engine}), encoding="utf-8")
            if op.action == "check":
                self._run_checks(op, chosen, work, cancel)
                self._invalidate()
            else:
                result = self.prepare(
                    self.context, work, op.engine, chosen, cancel,
                    lambda line: self._log(id_, line), lambda phase: self._update(id_, phase=phase),
                )
                if cancel.is_set():
                    raise InterruptedError
                check = self._check_result(op.engine, result["probe"])
                if check.state != "ready":
                    raise _error("not_ready", "环境检查未通过，请查看操作记录。")
                source = _source(op.engine, result["trainer_path"])
                identity = {
                    "id": id_, "engine": op.engine,
                    "python_path": result["python_path"], "trainer_path": result["trainer_path"],
                    "upstream_revision": source["revision"],
                    "dependency_fingerprint": check.dependency_fingerprint,
                    "resources_marker_sha256": _marker_digest(result["trainer_path"]),
                }
                identity["revision"] = _identity_digest(identity)
                row = TtsManagedEnvironment(
                    **identity, kind=result["kind"], state="ready", created_at=now(),
                    checked_at=check.checked_at, check=check,
                )
                with self.context.db.lock:
                    if cancel.is_set():
                        raise InterruptedError
                    staged = copy.deepcopy(self.state)
                    staged["environments"][id_] = row.model_dump()
                    if op.make_default:
                        staged["defaults"][op.engine] = id_
                    staged["operations"][id_].update(status="completed", phase="ready", progress=1.0,
                                                    environment_id=id_, updated_at=now(), cancellable=False)
                    self._replace(staged)
                    terminal = copy.deepcopy(staged["operations"][id_])
                self._event(terminal)
                self._invalidate()
        except InterruptedError:
            self._update(id_, status="cancelled", phase="cancelled")
        except Exception as exc:
            message = self._log(id_, str(exc))
            if cancel.is_set():
                self._update(id_, status="cancelled", phase="cancelled")
            else:
                issue = TtsIssue(code=getattr(exc, "code", "tts.environment.failed"),
                                 loc=["environment"], message=_failure_summary(message, self.get(id_).phase))
                self._update(id_, status="failed", phase="failed", issues=[issue.model_dump()])
        finally:
            with self.context.db.lock:
                self.cancels.pop(id_, None)
                self.context._active_tts_environments -= 1

    def _invalidate(self):
        from .tts_gpu import invalidate

        invalidate()

    def cancel(self, id_):
        with self.context.db.lock:
            op = self.get(id_)
            event = self.cancels.get(id_)
            if op.status not in ACTIVE or event is None:
                raise _error("finished", "这项环境操作已经结束。", 409)
            event.set()
            return self._update(id_, phase="cancelling")

    def retry(self, id_):
        op = self.get(id_)
        if op.status not in {"failed", "cancelled"}:
            raise _error("not_retryable", "只能重试失败或已取消的环境操作。", 409)
        return self.start(op.action, op.engine, op.candidate_id, make_default=op.make_default, retry_of=id_)

    def set_default(self, engine, id_):
        with self.context.db.lock:
            row = copy.deepcopy(self.state["environments"].get(id_))
        if row is None or row["engine"] != engine:
            raise _error("not_found", "此引擎的环境不存在。", 404)
        if row["state"] != "ready":
            raise _error("not_ready", "请先准备并检查语音环境。")
        self.verify(TtsEnvironmentIdentity.model_validate({k: row[k] for k in TtsEnvironmentIdentity.model_fields}).model_dump())
        with self.context.db.lock:
            staged = copy.deepcopy(self.state)
            if staged["environments"][id_]["state"] != "ready":
                raise _error("not_ready", "环境检查状态已变化，请重新检查。", 409)
            staged["defaults"][engine] = id_
            self._replace(staged)
        self._invalidate()
        self._event({"id": None, "engine": engine, "status": None})
        return self.snapshot()

    def resolve(self, config):
        values = config.model_dump()
        engine = values["engine"]
        explicit = bool(values["python_path"] and values["trainer_path"])
        with self.context.db.lock:
            rows = copy.deepcopy(self.state["environments"])
            if explicit:
                row = next((r for r in rows.values() if r["engine"] == engine
                            and _path_key(r["python_path"]) == _path_key(values["python_path"])
                            and _path_key(r["trainer_path"]) == _path_key(values["trainer_path"])), None)
                if row is None:
                    return config.model_copy(deep=True), None
            else:
                row = rows.get(self.state["defaults"].get(engine))
        if row is None or row["state"] != "ready":
            raise _error("not_ready", "请先准备语音环境，或填写已有 Python 和源码路径。")
        resolved = config.model_copy(deep=True)
        for field in ("python_path", "trainer_path"):
            if not values[field]:
                setattr(resolved, field, row[field])
        identity = {key: row.get(key) for key in TtsEnvironmentIdentity.model_fields}
        if (resolved.python_path, resolved.trainer_path) != (row["python_path"], row["trainer_path"]):
            from ypuddin.tts.environment_probe import metadata_fingerprint

            for field in ("python_path", "trainer_path"):
                if values[field] and not self.context.is_allowed(Path(values[field])):
                    raise _error("path_denied", "环境路径不在允许目录内。", 403, field=field)
            try:
                identity.update(python_path=resolved.python_path, trainer_path=resolved.trainer_path,
                                dependency_fingerprint=metadata_fingerprint(resolved.python_path),
                                upstream_revision=_source(engine, resolved.trainer_path)["revision"],
                                resources_marker_sha256=_marker_digest(resolved.trainer_path))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                raise _error("changed", str(exc), 409) from exc
            identity["revision"] = _identity_digest({k: v for k, v in identity.items() if k != "revision"})
        self.verify(identity)
        return resolved, identity

    def verify(self, identity):
        from ypuddin.tts.environment_probe import metadata_fingerprint
        from ypuddin.tts.environment_resources import resource_environment

        if identity is None:
            return
        try:
            value = TtsEnvironmentIdentity.model_validate(identity).model_dump()
            expected = _identity_digest({k: v for k, v in value.items() if k != "revision"})
            if value["revision"] != expected:
                raise ValueError("语音环境身份不匹配。")
            if _source(value["engine"], value["trainer_path"])["revision"] != value["upstream_revision"]:
                raise ValueError("语音源码版本已变化。")
            if metadata_fingerprint(value["python_path"]) != value["dependency_fingerprint"]:
                raise ValueError("语音环境依赖已变化，请重新准备环境。")
            if _marker_digest(value["trainer_path"]) != value["resources_marker_sha256"]:
                raise ValueError("语音环境资源已变化，请重新准备环境。")
            resource_environment(value["trainer_path"], expected_sha256=value["resources_marker_sha256"])
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            raise _error("changed", str(exc), 409) from exc

    def runtime_allowed(self, identity):
        python = set()
        if identity:
            with self.context.db.lock:
                row = self.state["environments"].get(identity["id"])
                if row and _path_key(row["python_path"]) == _path_key(identity["python_path"]):
                    python = {_path_key(identity["python_path"]), _path_key(Path(identity["python_path"]).resolve())}
        return lambda path: self.context.is_allowed(path) or _path_key(path) in python

    def background_tasks(self):
        with self.context.db.lock:
            operations = copy.deepcopy(list(self.state["operations"].values()))
        cutoff = time.time() - 600
        return [{
            "id": op["id"], "kind": "environment", "subject": "GPT-SoVITS" if op["engine"] == "gpt-sovits-v5" else "VoxCPM",
            "state": "running" if op["status"] in ACTIVE else op["status"],
            "detail": "环境检查" if op["action"] == "check" else "环境准备",
            "done": None, "total": None, "unit": None, "link": "/settings/environment?tab=runtime&environment=speech",
            "cancellable": op["cancellable"], "started_at": op["created_at"],
            "finished_at": None if op["status"] in ACTIVE else op["updated_at"],
            "error": op["issues"][0]["message"] if op["status"] == "failed" and op["issues"] else None,
        } for op in operations if op["status"] in ACTIVE or op["updated_at"] >= cutoff]

    def close(self):
        with self.context.db.lock:
            self.closed = True
            for event in self.cancels.values():
                event.set()
        self.pool.shutdown(wait=True)
