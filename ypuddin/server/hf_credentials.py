"""Local Hugging Face login state and operation-scoped credential resolution."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network import ProxyPolicy

log = logging.getLogger(__name__)
RESOLVE_TIMEOUT = 45


def _clean(value: str) -> str | None:
    return value.replace("\r", "").replace("\n", "").strip() or None


def local_token() -> str | None:
    token = _clean(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
    if token:
        return token
    try:
        from huggingface_hub import constants
    except ImportError:
        return None
    try:
        return _clean(Path(constants.HF_TOKEN_PATH).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def resolve_token(policy: ProxyPolicy) -> str | None:
    """Resolve a download's login without changing the service's global HF client."""
    from .errors import ApiError

    try:
        from huggingface_hub import constants
    except ImportError:
        return _clean(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
    # OIDC is an explicit HF login mode and takes precedence over environment tokens.
    if not os.environ.get("HF_OIDC_RESOURCE"):
        environment = _clean(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
        if environment:
            return environment
        if local_token() is None:
            return None
    env = policy.subprocess_env()
    env.update(HF_HOME=str(constants.HF_HOME), HF_TOKEN_PATH=str(constants.HF_TOKEN_PATH),
               PYTHONIOENCODING="utf-8")
    if policy.mode == "direct":
        # HTTPX also discovers OS proxies; removing proxy environment variables alone is insufficient.
        env["NO_PROXY"] = env["no_proxy"] = "*"
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--resolve"],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=RESOLVE_TIMEOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode:
            raise ValueError()
        response = json.loads(result.stdout)
        if not isinstance(response, dict) or response.get("error"):
            raise ValueError()
        token = response["token"]
        if token is not None and not isinstance(token, str):
            raise ValueError()
        messages = response.get("warnings", [])
        if not isinstance(messages, list) or any(not isinstance(message, str) for message in messages):
            raise ValueError()
    except subprocess.TimeoutExpired:
        raise ApiError("Hugging Face login refresh timed out; retry the download.",
                       code="credentials.refresh_timeout", status=503) from None
    except (OSError, ValueError, KeyError):
        # Child output is private: failed authentication can contain tokens or proxy credentials.
        raise ApiError("Cannot resolve the local Hugging Face login; check the login and retry the download.",
                       code="credentials.resolve", status=503) from None
    for message in messages:
        log.warning("%s", policy.redact(message))
    return token


def _resolve_worker() -> None:
    """Use HF's own refresh, file locking and token rotation in a private process."""
    import configparser
    import contextlib
    import io
    import re
    import urllib.parse

    from huggingface_hub import constants, get_token

    secrets: set[str] = set()

    def collect_secrets() -> None:
        secrets.update(value for key, value in os.environ.items() if value and
                       ("TOKEN" in key.upper() or "PASSWORD" in key.upper()))
        try:
            secrets.add(Path(constants.HF_TOKEN_PATH).read_text(encoding="utf-8").strip())
            config = configparser.ConfigParser(interpolation=None)
            stored_path = getattr(constants, "HF_STORED_TOKENS_PATH", None)
            if stored_path and Path(stored_path).exists():
                text = Path(stored_path).read_text(encoding="utf-8")
                # Even a malformed INI can be included in the library's parsing warning.
                secrets.update(re.findall(r"(?m)^\s*(?:hf_token|refresh_token)\s*[=:]\s*(.*?)\s*$", text))
                config.read_string(text)
            for section in config.sections():
                secrets.update(config.get(section, key, fallback="") for key in ("hf_token", "refresh_token"))
        except (OSError, UnicodeError, configparser.Error):
            pass
        for key, value in os.environ.items():
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                try:
                    password = urllib.parse.urlsplit(value).password
                    if password:
                        secrets.update((password, urllib.parse.unquote(password)))
                except ValueError:
                    pass

    messages: list[str] = []

    class Warnings(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Warnings(logging.WARNING)
    logger = logging.getLogger("huggingface_hub")
    logger.addHandler(handler)
    collect_secrets()
    response = {"token": None, "warnings": [], "error": None}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            response["token"] = get_token()
    except Exception as error:
        response["error"] = type(error).__name__
    finally:
        logger.removeHandler(handler)
    collect_secrets()
    if response["token"]:
        secrets.add(response["token"])
    for message in messages:
        for secret in sorted(filter(None, secrets), key=len, reverse=True):
            message = message.replace(secret, "***")
        response["warnings"].append(message)
    print(json.dumps(response))


if __name__ == "__main__":
    _resolve_worker()
