"""Version validation without holding the metadata lock during filesystem or runtime checks."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from ypuddin.tts.core import UPSTREAM_REVISION
from ypuddin.tts.gpt_sovits.config import GptSovitsVersionConfig
from ypuddin.tts.issues import TtsIssue
from ypuddin.tts.source_models import TtsSource
from ypuddin.tts.validation_models import (
    TtsBatchingReport,
    TtsDatasetReport,
    TtsEnvironmentReport,
    TtsGptSovitsDatasetReport,
    TtsGptSovitsEnvironmentReport,
    TtsGptSovitsPreparationReport,
    TtsGptSovitsSplitDatasetReport,
    TtsGptSovitsStageReport,
    TtsSplitDatasetReport,
    TtsTokenFilterReport,
    TtsValidationExecutionReport,
    TtsValidationReport,
)
from ypuddin.tts.version_config import TtsConfigResponse, TtsVersionConfig

from .db import new_id, now
from .errors import ApiError
from .project_deletion import deleting
from .tts_projects import get_config
from .versions import assert_version_writable


def inspect_runtime(config: TtsVersionConfig | GptSovitsVersionConfig, manifests: dict, *, allowed: Any, gpu_devices: list[str] | None = None) -> dict:
    if config.engine == "gpt-sovits-v5":
        from ypuddin.tts.gpt_sovits.runtime import inspect_runtime as inspect
    else:
        from ypuddin.tts.runtime import inspect_runtime as inspect

    return inspect(config, manifests, allowed=allowed, timeout=90, gpu_devices=gpu_devices)


def _conflict(code: str, message: str, field: str, value: int) -> ApiError:
    revision_key = "current_revision" if field == "revision" else "current_data_revision"
    return ApiError(
        message,
        code=code,
        status=409,
        details={
            revision_key: value,
            "issues": [TtsIssue(code=code, loc=[field], message=message).model_dump()],
        },
    )


def capture_validation_state(
    c: Any, pid: str, vid: str, revision: int, data_revision: int
) -> TtsConfigResponse:
    """Capture the saved recipe; also safe inside the caller's final enqueue DB lock."""
    with c.db.lock:
        c.require_project_type(pid, "tts")
        if deleting(c, pid):
            raise ApiError("项目正在删除。", code="project.deleting", status=409)
        assert_version_writable(c, pid, vid)
        saved = get_config(c, pid, vid)
        if saved.revision != revision:
            raise _conflict("tts.config_conflict", "配置已更新，请重新检查。", "revision", saved.revision)
        if saved.data_revision != data_revision:
            raise _conflict(
                "tts.data_conflict", "数据已更新，请重新检查。", "data_revision", saved.data_revision
            )
        return saved


def assert_validation_current(c: Any, pid: str, vid: str, captured: TtsConfigResponse) -> TtsConfigResponse:
    with c.db.lock:
        saved = capture_validation_state(c, pid, vid, captured.revision, captured.data_revision)
        if captured.scope != saved.scope or captured.config != saved.config:
            raise _conflict("tts.config_conflict", "配置内容已变化，请重新检查。", "revision", saved.revision)
        return saved


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source(value: dict | None) -> TtsSource | None:
    return TtsSource.model_validate(value["source"]) if value is not None else None


def _source_identity(value: dict | None) -> dict | None:
    if value is None:
        return None
    source = _source(value)
    return {
        "id": source.id,
        "revision": source.revision,
        "state": source.state,
        "snapshot_id": source.snapshot_id,
        "fingerprint": value["fingerprint"],
        "fingerprints": sorted(value["fingerprints"], key=lambda item: item["path"]),
    }


def _input_identity(inputs: dict) -> dict:
    return {
        "data_revision": inputs["data_revision"],
        **{split: _source_identity(inputs[split]) for split in ("train", "validation")},
    }


def _assert_sources(c: Any, pid: str, vid: str, captured: TtsConfigResponse, expected: dict) -> None:
    assert_validation_current(c, pid, vid, captured)
    actual = c.tts_sources.snapshot_identity(pid, vid)
    if actual != expected:
        raise _conflict(
            "tts.data_conflict", "数据检查结果已变化，请重新检查。", "data_revision", actual["data_revision"]
        )


def _source_issues(source: TtsSource, split: str) -> list[TtsIssue]:
    # Row details remain paginated; the report retains source-level causes and one warning summary.
    issues = [issue for issue in source.issues if "rows" not in issue.loc]
    if any(issue.severity == "warning" and "rows" in issue.loc for issue in source.issues):
        issues.append(
            TtsIssue(
                code="tts.dataset.source_warnings",
                loc=["sources", split],
                severity="warning",
                message="数据清单存在录音警告，详情见数据行。",
                details={"source_id": source.id},
            )
        )
    return issues


def _batching(split: str, count: int, batch_size: int) -> TtsBatchingReport:
    full, tail = divmod(count, batch_size)
    training = split == "train"
    partial = 0 if training else int(tail > 0)
    batches = full + partial
    result = TtsBatchingReport(
        state="ready",
        batch_size=batch_size,
        drop_last=training,
        input_samples=count,
        full_batches_per_pass=full,
        tail_samples_per_pass=tail,
        partial_batches_per_pass=partial,
        dropped_tail_samples_per_pass=tail if training else 0,
        yielded_batches_per_pass=batches,
        yielded_samples_per_pass=full * batch_size if training else count,
        can_form_full_batch=count >= batch_size,
        can_yield_batch=batches > 0,
    )
    if training and not result.can_form_full_batch:
        result.issues.append(
            TtsIssue(
                code="tts.dataset.insufficient_batch",
                loc=["sources", "train"],
                message="过滤后样本不足一个完整训练批次。",
                details={"kept_count": count, "batch_size": batch_size, "full_batches_per_pass": full},
            )
        )
    elif not training and not result.can_yield_batch:
        result.issues.append(
            TtsIssue(
                code="tts.dataset.empty_validation",
                loc=["sources", "validation"],
                message="验证清单没有可用样本。",
                details={"kept_count": count},
            )
        )
    return result


def _split_report(
    split: str, value: dict | None, config: TtsVersionConfig, filtered: dict
) -> TtsSplitDatasetReport:
    source = _source(value)
    training = split == "train"
    disabled = source is None and not training
    blocked = "not_applicable" if disabled else "blocked"
    result = TtsSplitDatasetReport(
        split=split,
        state="disabled" if disabled else "unavailable",
        checked_at=None if disabled else now(),
        source_state=source.state if source else "missing",
        source_id=source.id if source else None,
        source_revision=source.revision if source else None,
        snapshot_id=source.snapshot_id if source else None,
        source_summary=source.summary if source and source.state in ("valid", "invalid") else None,
        token_filter=TtsTokenFilterReport(
            state=blocked,
            max_batch_tokens=config.max_batch_tokens,
            max_sample_tokens=config.max_batch_tokens // config.batch_size
            if training and config.max_batch_tokens
            else None,
        ),
        batching=TtsBatchingReport(state=blocked, batch_size=config.batch_size, drop_last=training),
        validation_execution=TtsValidationExecutionReport(
            state="not_applicable" if training or disabled else "blocked",
            max_batches_per_validation=10 if source is not None and not training else None,
        ),
    )
    if source is None:
        if training:
            result.issues.append(
                TtsIssue(code="tts.dataset.missing", loc=["sources", "train"], message="请登记训练数据清单。")
            )
        return result
    result.issues.extend(_source_issues(source, split))
    if source.state != "valid":
        if source.state in ("unchecked", "checking"):
            result.state, result.checked_at = "unchecked", None
        result.issues.append(
            TtsIssue(
                code="tts.source_not_ready"
                if source.state in ("unchecked", "checking")
                else "tts.dataset.source_unavailable",
                loc=["sources", split],
                message="数据清单尚未检查完成。"
                if source.state in ("unchecked", "checking")
                else "数据清单无法用于训练，请查看来源检查结果。",
                details={"source_id": source.id, "state": source.state},
            )
        )
        return result
    token_filter = TtsTokenFilterReport.model_validate(filtered)
    expected_threshold = (
        config.max_batch_tokens // config.batch_size if training and config.max_batch_tokens else None
    )
    if (
        token_filter.max_batch_tokens != config.max_batch_tokens
        or token_filter.max_sample_tokens != expected_threshold
    ):
        raise RuntimeError("Runtime token filtering used a different recipe")
    if token_filter.input_fingerprint and value["fingerprint"]:
        token_filter.input_fingerprint = _digest(
            {
                "runtime": token_filter.input_fingerprint,
                "source": _source_identity(value),
                "batch_size": config.batch_size,
                "max_batch_tokens": config.max_batch_tokens,
            }
        )
    else:
        token_filter.input_fingerprint = None
    result.token_filter = token_filter
    if token_filter.state in ("unchecked", "blocked", "error") or token_filter.kept_count is None:
        result.state = "unavailable" if token_filter.state == "error" else "unchecked"
        result.checked_at = now() if token_filter.state == "error" else None
        return result
    count = token_filter.kept_count
    result.batching = _batching(split, count, config.batch_size)
    if not training:
        result.validation_execution = TtsValidationExecutionReport(
            state="ready",
            max_batches_per_validation=10,
            evaluated_batches_per_validation=min(result.batching.yielded_batches_per_pass, 10),
            evaluated_samples_per_validation=min(count, 10 * config.batch_size),
            not_evaluated_samples_per_validation=max(0, count - 10 * config.batch_size),
        )
    result.state = "unavailable" if result.batching.issues else "available"
    result.checked_at = now()
    return result


def _gpt_sovits_split_report(
    split: str, value: dict | None, config: Any
) -> TtsGptSovitsSplitDatasetReport:
    source = _source(value)
    training = split == "train"
    disabled = source is None and not training
    active = bool(source is not None and source.state == "valid" and training
                  and source.summary is not None and source.summary.valid_clips_count > 0)
    preparation_state = "not_applicable" if disabled or not training else "unchecked" if active else "blocked"
    result = TtsGptSovitsSplitDatasetReport(
        split=split,
        state="disabled" if disabled else "available" if active else "unavailable",
        checked_at=None if disabled else now(),
        source_state=source.state if source else "missing",
        source_id=source.id if source else None,
        source_revision=source.revision if source else None,
        snapshot_id=source.snapshot_id if source else None,
        source_summary=source.summary if source and source.state in ("valid", "invalid") else None,
        preparation=TtsGptSovitsPreparationReport(state=preparation_state),
        stages=[
            TtsGptSovitsStageReport(
                stage=stage,
                state="not_applicable" if not training or config.stage not in ("both", stage)
                else "unchecked" if active else "blocked",
                batch_size=getattr(config, stage).batch_size,
            )
            for stage in ("gpt", "sovits")
        ],
    )
    if source is None:
        if training:
            result.issues.append(
                TtsIssue(code="tts.dataset.missing", loc=["sources", "train"], message="请登记训练数据清单。")
            )
        return result
    result.issues.extend(_source_issues(source, split))
    if not training:
        result.issues.append(TtsIssue(
            code="tts.gpt_sovits.validation_source_unsupported", loc=["sources", "validation"],
            message="GPT-SoVITS v5 训练不使用独立验证清单，请移除验证来源。",
        ))
    if source.state != "valid":
        if source.state in ("unchecked", "checking"):
            result.state, result.checked_at = "unchecked", None
        result.issues.append(TtsIssue(
            code="tts.source_not_ready" if source.state in ("unchecked", "checking") else "tts.dataset.source_unavailable",
            loc=["sources", split],
            message="数据清单尚未检查完成。" if source.state in ("unchecked", "checking")
            else "数据清单无法用于训练，请查看来源检查结果。",
            details={"source_id": source.id, "state": source.state},
        ))
    elif training and (source.summary is None or source.summary.valid_clips_count == 0):
        result.state = "unavailable"
        result.issues.append(TtsIssue(
            code="tts.dataset.empty_training", loc=["sources", "train"], message="训练清单没有可用样本。",
        ))
    return result


def _issues(
    dataset: TtsDatasetReport | TtsGptSovitsDatasetReport,
    environment: TtsEnvironmentReport | TtsGptSovitsEnvironmentReport,
    config: TtsVersionConfig | GptSovitsVersionConfig,
) -> tuple[list[TtsIssue], list[TtsIssue]]:
    all_issues = []
    for split in (dataset.train, dataset.validation):
        all_issues.extend(split.issues)
        reports = (split.preparation, *split.stages) if isinstance(dataset, TtsGptSovitsDatasetReport) else (
            split.token_filter, split.batching, split.validation_execution
        )
        for report in reports:
            all_issues.extend(report.issues)
    for check in environment.checks:
        all_issues.extend(check.issues)
    if config.engine == "voxcpm1.5" and 0 < config.max_steps < config.num_iters:
        all_issues.append(
            TtsIssue(
                code="tts.config.schedule_shorter_than_training",
                loc=["config", "max_steps"],
                severity="warning",
                message="学习率调度步数小于训练迭代次数。",
                details={"max_steps": config.max_steps, "num_iters": config.num_iters},
            )
        )
    unique = {}
    for issue in all_issues:
        key = issue.code, tuple(issue.loc)
        if key not in unique or issue.severity == "error":
            unique[key] = issue
    return (
        [issue for issue in unique.values() if issue.severity == "error"],
        [issue for issue in unique.values() if issue.severity == "warning"],
    )


def validate_version(c: Any, pid: str, vid: str, revision: int, data_revision: int, *, gpu_devices: list[str] | None = None) -> TtsValidationReport:
    from .tts_gpu import invalidate

    invalidate()
    with c.db.lock:
        captured = capture_validation_state(c, pid, vid, revision, data_revision)
        expected_sources = copy.deepcopy(c.tts_sources.snapshot_identity(pid, vid))
    inputs = c.tts_sources.validation_inputs(pid, vid, recheck=True)
    with c.db.lock:
        _assert_sources(c, pid, vid, captured, expected_sources)
        if inputs["data_revision"] != data_revision:
            raise _conflict(
                "tts.data_conflict", "数据已更新，请重新检查。", "data_revision", inputs["data_revision"]
            )
    initial_inputs = _input_identity(inputs)
    manifests = {
        split: inputs[split]["rows"]
        if inputs[split] is not None and _source(inputs[split]).state == "valid"
        else None
        for split in ("train", "validation")
    }
    execution = captured.config.model_copy(deep=True)
    binding = None
    environments = getattr(c, "tts_environments", None)
    if environments is not None:
        execution, binding = environments.resolve(execution)
    allowed = environments.runtime_allowed(binding) if environments is not None and binding is not None else c.is_allowed
    runtime = inspect_runtime(execution, manifests, allowed=allowed, gpu_devices=gpu_devices)
    is_gpt_sovits = captured.config.engine == "gpt-sovits-v5"
    if is_gpt_sovits:
        dataset = TtsGptSovitsDatasetReport(**{
            split: _gpt_sovits_split_report(split, inputs[split], captured.config)
            for split in ("train", "validation")
        })
    else:
        dataset = TtsDatasetReport(**{
            split: _split_report(split, inputs[split], captured.config, runtime["token_filter"][split])
            for split in ("train", "validation")
        })
    environment_model = TtsGptSovitsEnvironmentReport if is_gpt_sovits else TtsEnvironmentReport
    environment = environment_model.model_validate(runtime["environment"])
    refreshed = c.tts_sources.validation_inputs(pid, vid, recheck=True)
    if binding is not None:
        environments.verify(binding)
    with c.db.lock:
        _assert_sources(c, pid, vid, captured, expected_sources)
        if _input_identity(refreshed) != initial_inputs:
            raise _conflict(
                "tts.data_conflict",
                "数据内容已变化，请重新检查。",
                "data_revision",
                refreshed["data_revision"],
            )
    errors, warnings = _issues(dataset, environment, captured.config)
    complete_data = dataset.train.state == "available" and dataset.validation.state in (
        "available",
        "disabled",
    )
    complete_sources = inputs["train"] is not None and all(
        inputs[split] is None
        or (
            _source(inputs[split]).state == "valid"
            and _source(inputs[split]).snapshot_id
            and inputs[split]["fingerprint"]
            and (is_gpt_sovits or getattr(dataset, split).token_filter.before_count is not None)
        )
        for split in ("train", "validation")
    )
    fingerprint = None
    if runtime["input_fingerprint"] and complete_sources and environment.state == "available":
        fingerprint = _digest(
            {
                "scope": captured.scope.model_dump(),
                "revision": revision,
                "data_revision": data_revision,
                "schema_version": 1,
                "upstream_revision": runtime["details"]["upstream_revision"] if is_gpt_sovits else UPSTREAM_REVISION,
                "config": captured.config.model_dump(mode="json"),
                "sources": initial_inputs,
                "runtime": runtime["input_fingerprint"],
                **({"environment_binding": binding} if binding is not None else {}),
            }
        )
    if complete_data and environment.state == "available" and fingerprint is None:
        errors.append(
            TtsIssue(
                code="tts.validation.identity_unavailable",
                loc=[],
                message="无法取得完整的输入身份，请重新检查数据和运行环境。",
            )
        )
    report = TtsValidationReport(
        scope=captured.scope,
        revision=revision,
        data_revision=data_revision,
        validation_id=new_id("validation"),
        input_fingerprint=fingerprint,
        checked_at=now(),
        valid=bool(complete_data and environment.state == "available" and not errors and fingerprint),
        errors=errors,
        warnings=warnings,
        dataset=dataset,
        environment=environment,
        environment_binding=binding,
    )
    report._runtime_fingerprints = copy.deepcopy(runtime.get("fingerprints", []))
    report._runtime_model_identity = copy.deepcopy(runtime.get("model_identity"))
    report._runtime_config = execution.model_copy(deep=True)
    report._runtime_environment = copy.deepcopy(binding)
    return report
