"""Local model-source credentials, deliberately separate from public settings/status.

An explicitly cleared provider stays anonymous even if a CLI/environment token exists.
The file is private where POSIX permissions are supported; it is not an encrypted vault.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from .errors import ApiError

Provider = Literal["huggingface", "modelscope"]
PROVIDERS = ("huggingface", "modelscope")


class CredentialState(BaseModel):
    configured: bool


class CredentialStates(BaseModel):
    huggingface: CredentialState
    modelscope: CredentialState


class CredentialUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: SecretStr

    @field_validator("token")
    @classmethod
    def valid_token(cls, value: SecretStr) -> SecretStr:
        token = value.get_secret_value().strip()
        if not token or len(token) > 4096 or any(ord(c) < 33 or ord(c) > 126 or c in ';,"\\' for c in token):
            raise ValueError(
                "token must be 1–4096 printable characters without whitespace or cookie separators"
            )
        return SecretStr(token)


class ModelCredentials:
    def __init__(self, root: Path):
        self.path = root / "secrets.json"
        self.lock = threading.RLock()

    def _read(self) -> dict:
        try:
            if not self.path.exists():
                return {}
            value = json.loads(self.path.read_text("utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("model_sources", {}), dict):
                raise ValueError()
            return value
        except (OSError, ValueError):
            raise ApiError(
                "Cannot read the local credentials file; repair secrets.json before saving credentials.",
                code="credentials.read",
                status=503,
            ) from None

    def token(self, provider: Provider) -> str | None:
        with self.lock:
            stored = self._read().get("model_sources", {})
            if provider in stored:
                value = stored[provider]
                if not isinstance(value, str):
                    raise ApiError("Invalid local credential entry.", code="credentials.read", status=503)
                return value or None
        if provider == "modelscope":
            return os.environ.get("MODELSCOPE_API_TOKEN", "").strip() or None
        try:
            from huggingface_hub import get_token

            return get_token()
        except ImportError:
            return os.environ.get("HF_TOKEN", "").strip() or None

    def state(self) -> dict[str, dict[str, bool]]:
        return {provider: {"configured": bool(self.token(provider))} for provider in PROVIDERS}

    def save(self, provider: Provider, token: str) -> dict[str, bool]:
        with self.lock:
            value = self._read()
            value.setdefault("model_sources", {})[provider] = token
            temporary: str | None = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(prefix=".secrets-", dir=self.path.parent)
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(value, stream, ensure_ascii=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, self.path)
            except OSError:
                raise ApiError(
                    "Cannot save local credentials. Check the service data directory permissions.",
                    code="credentials.write",
                    status=503,
                ) from None
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        return {"configured": bool(token)}
