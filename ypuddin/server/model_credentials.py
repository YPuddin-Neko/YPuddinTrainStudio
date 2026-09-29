"""Local access credentials, deliberately separate from public settings/project data.

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

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from .errors import ApiError

Provider = Literal["huggingface", "modelscope"]
PROVIDERS = ("huggingface", "modelscope")
SiteProvider = Literal["danbooru", "gelbooru", "e621", "rule34"]
SITE_PROVIDERS = ("danbooru", "gelbooru", "e621", "rule34")
AccessProvider = Literal["huggingface", "modelscope", "danbooru", "gelbooru", "e621", "rule34"]
VlmProvider = Literal[
    "openai", "gemini", "openrouter", "siliconflow", "dashscope", "deepseek", "ollama", "lmstudio", "custom"
]


class CredentialState(BaseModel):
    configured: bool


class CredentialStates(BaseModel):
    huggingface: CredentialState
    modelscope: CredentialState


class AccessCredentialStates(CredentialStates):
    danbooru: CredentialState
    gelbooru: CredentialState
    e621: CredentialState
    rule34: CredentialState


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


class SiteCredentialUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr

    @field_validator("api_key")
    @classmethod
    def valid_key(cls, value: SecretStr) -> SecretStr:
        return CredentialUpdate.valid_token(value)


class DanbooruCredentialUpdate(SiteCredentialUpdate):
    username: str = Field(min_length=1, max_length=200)

    @field_validator("username")
    @classmethod
    def valid_username(cls, value: str) -> str:
        username = value.strip()
        if not username or any(char.isspace() or ord(char) < 32 or char == ":" for char in username):
            raise ValueError("username must not contain whitespace, control characters or a colon")
        return username


class GelbooruCredentialUpdate(SiteCredentialUpdate):
    user_id: str = Field(pattern=r"^[0-9]{1,20}$")


class ModelCredentials:
    def __init__(self, root: Path):
        self.path = root / "secrets.json"
        self.lock = threading.RLock()

    def _read(self) -> dict:
        try:
            if not self.path.exists():
                return {}
            value = json.loads(self.path.read_text("utf-8"))
            if not isinstance(value, dict) or any(
                not isinstance(value.get(group, {}), dict)
                for group in ("model_sources", "site_sources", "vlm_services")
            ):
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

    def site(self, provider: SiteProvider) -> tuple[str, str]:
        """Return a private per-operation snapshot; never serialize this tuple in task records."""
        with self.lock:
            entry = self._read().get("site_sources", {}).get(provider, {})
            if entry == {}:
                return "", ""
            try:
                model = (
                    DanbooruCredentialUpdate if provider in {"danbooru", "e621"} else GelbooruCredentialUpdate
                )
                parsed = model.model_validate(entry)
                account = parsed.username if isinstance(parsed, DanbooruCredentialUpdate) else parsed.user_id
                return account, parsed.api_key.get_secret_value()
            except ValidationError:
                raise ApiError(
                    "Invalid local site credential entry.", code="credentials.read", status=503
                ) from None

    def access_state(self) -> dict[str, dict[str, bool]]:
        with self.lock:
            return self.state() | {
                provider: {"configured": bool(self.site(provider)[1])} for provider in SITE_PROVIDERS
            }

    def vlm_key(self, provider: VlmProvider) -> str | None:
        """Private per-operation snapshot of a vision model service key."""
        with self.lock:
            value = self._read().get("vlm_services", {}).get(provider, "")
            if not isinstance(value, str):
                raise ApiError("Invalid local credential entry.", code="credentials.read", status=503)
            return value or None

    def vlm_state(self) -> dict[str, bool]:
        with self.lock:
            stored = self._read().get("vlm_services", {})
            return {provider: bool(key) for provider, key in stored.items() if isinstance(key, str)}

    def save_vlm_key(self, provider: VlmProvider, key: str) -> dict[str, bool]:
        with self.lock:
            value = self._read()
            services = value.setdefault("vlm_services", {})
            if key:
                services[provider] = key
            else:
                services.pop(provider, None)
            self._write(value)
        return {"configured": bool(key)}

    def save(self, provider: Provider, token: str) -> dict[str, bool]:
        with self.lock:
            value = self._read()
            value.setdefault("model_sources", {})[provider] = token
            self._write(value)
        return {"configured": bool(token)}

    def save_site(self, provider: SiteProvider, body: SiteCredentialUpdate) -> dict[str, bool]:
        with self.lock:
            value = self._read()
            value.setdefault("site_sources", {})[provider] = body.model_dump(exclude={"api_key"}) | {
                "api_key": body.api_key.get_secret_value()
            }
            self._write(value)
        return {"configured": True}

    def clear(self, provider: AccessProvider) -> dict[str, bool]:
        if provider in PROVIDERS:
            return self.save(provider, "")
        with self.lock:
            value = self._read()
            value.setdefault("site_sources", {})[provider] = {}
            self._write(value)
        return {"configured": False}

    def _write(self, value: dict) -> None:
        with self.lock:
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
