"""Check a task's frozen environment without consulting mutable default selections."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def identity_digest(identity: dict) -> str:
    value = {key: item for key, item in identity.items() if key != "revision"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def snapshot_environment(payload: dict) -> dict | None:
    if "tts_environment" not in payload:
        return None
    identity = payload["tts_environment"]
    recipe = payload.get("tts")
    required = ("id", "revision", "engine", "python_path", "trainer_path", "upstream_revision", "dependency_fingerprint")
    if (not isinstance(identity, dict) or not isinstance(recipe, dict)
            or any(not isinstance(identity.get(key), str) or not identity[key] for key in required)):
        raise ValueError("任务的语音运行环境记录不完整。")
    if any(not Path(identity[key]).expanduser().is_absolute() for key in ("python_path", "trainer_path")):
        raise ValueError("任务的语音运行环境缺少绝对路径。")
    if any(identity[key] != recipe.get(key) for key in ("engine", "python_path", "trainer_path")):
        raise ValueError("任务配置与保存的语音运行环境不一致。")
    if identity["revision"] != identity_digest(identity):
        raise ValueError("任务的语音运行环境身份不匹配。")
    if identity["engine"] == "gpt-sovits-v5":
        from .gpt_sovits.core import UPSTREAM_REVISION
    elif identity["engine"] == "voxcpm1.5":
        from .core import UPSTREAM_REVISION
    else:
        raise ValueError("任务的语音运行环境引擎无效。")
    if identity["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("任务的语音运行环境源码提交不匹配。")
    return identity


def assert_snapshot_environment(payload: dict) -> None:
    identity = snapshot_environment(payload)
    if identity is None:
        return
    from .environment_probe import metadata_fingerprint
    from .environment_resources import marker_identity, resource_environment

    try:
        if "resources_marker_sha256" in identity:
            expected = identity["resources_marker_sha256"]
            if expected is not None and (not isinstance(expected, str) or len(expected) != 64):
                raise ValueError("任务的语音环境资源记录无效。")
            if marker_identity(identity["trainer_path"]) != expected:
                raise ValueError("任务使用的语音环境资源清单已变化。")
            if expected is not None:
                resource_environment(identity["trainer_path"], expected_sha256=expected)
        actual = metadata_fingerprint(identity["python_path"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError(f"无法检查任务的语音运行环境：{exc}") from exc
    if actual != identity["dependency_fingerprint"]:
        raise ValueError("任务使用的 Python 或依赖已变化，请重新检查环境并创建训练任务。")
