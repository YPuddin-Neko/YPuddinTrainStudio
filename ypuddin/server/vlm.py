"""Captions from vision-language models behind an OpenAI-compatible chat-completions API.

Each image travels as a JPEG data URL in one request. API keys stay on the server: they are read per
operation, sent only in the Authorization header and never written to operation records or logs.
Cloud services use the base URL fixed here, so a stored key only ever reaches its own service.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .errors import ApiError

# Cloud services keep their address; local servers and "custom" take the address the user gives.
PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "editable": False},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "editable": False},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "editable": False},
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1", "editable": False},
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "editable": False},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "editable": False},
    "ollama": {"base_url": "http://127.0.0.1:11434/v1", "editable": True},
    "lmstudio": {"base_url": "http://127.0.0.1:1234/v1", "editable": True},
    "custom": {"base_url": "", "editable": True},
}

REFUSALS = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "unable to assist",
    "cannot assist",
    "can't assist",
    "cannot help",
    "can't help",
    "cannot provide",
    "can't provide",
    "against my",
    "content policy",
    "抱歉",
    "无法处理",
    "不能提供",
)
MARKERS = ("count", "appearance", "tags", "environment")
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"^\s*```[\w-]*\s*\n?|\n?\s*```\s*$")
_EMOTICON = re.compile(r"[0-9oOxXuU=^<>@|+.;()\-_]+")


class VlmError(Exception):
    """One image could not be captioned; the run continues with the rest."""


class VlmStopped(VlmError):
    """The run was stopped before this image's request finished."""


class VlmFatal(ApiError):
    """The service refuses every request (address, key or model); the run stops."""


def base_url(provider: str, requested: str | None) -> str:
    """The address a request goes to: fixed for cloud services, validated for the others."""
    entry = PROVIDERS.get(provider)
    if entry is None:
        raise ApiError(f"unknown vision model service: {provider}", code="vlm.provider")
    value = (requested or "").strip() if entry["editable"] else entry["base_url"]
    value = value or entry["base_url"]
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or len(value) > 500
    ):
        raise ApiError(
            "enter the service address as http(s)://host[:port]/path, e.g. http://127.0.0.1:11434/v1",
            code="vlm.base_url",
        )
    return value.rstrip("/")


def _loopback(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def opener_for(url: str, policy: Any) -> urllib.request.OpenerDirector:
    """Loopback servers are always reached directly; everything else follows the proxy settings."""
    if _loopback(url) or policy is None:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return policy.opener()


def image_data_url(path: str | Path, longest: int = 1024) -> str:
    """EXIF-oriented JPEG with transparency over white, the longest side at most ``longest`` pixels."""
    with Image.open(path) as source:
        source.draft("RGB", (longest, longest))
        oriented = ImageOps.exif_transpose(source)
        if oriented.mode in {"RGBA", "LA", "PA"} or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            oriented = Image.alpha_composite(background, rgba)
        image = oriented.convert("RGB")
    if max(image.size) > longest:
        image.thumbnail((longest, longest), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def looks_like_refusal(text: str) -> bool:
    """A short reply built from apology phrases; a tag list with several commas never counts."""
    stripped = text.strip()
    if not stripped or stripped.count(",") >= 3 or stripped.count("，") >= 3:
        return False
    lowered = stripped.lower()
    return any(marker in lowered for marker in REFUSALS)


def clean(text: str) -> str:
    """The answer without reasoning blocks or code fences around it."""
    return _FENCE.sub("", _THINK.sub("", text)).strip()


def readable(tag: str) -> str:
    tag = tag.strip().strip("-*•·").strip().strip("\"'`").strip()
    return tag if _EMOTICON.fullmatch(tag) else tag.replace("_", " ")


def tag_list(text: str) -> list[str]:
    """Tags from a reply: comma or line separated, a leading "Tags:" label ignored, each tag once."""
    lines = clean(text).splitlines()
    if lines and lines[0].rstrip().endswith((":", "：")) and "," not in lines[0]:
        lines = lines[1:]  # a heading such as "Here are the tags:"
    body = re.sub(r"(?im)^\s*(refined\s+tags|tags|标签)\s*[:：]\s*", "", "\n".join(lines))
    tags, seen = [], set()
    for part in re.split(r"[,\n，、]", body):
        tag = readable(part)
        if tag and len(tag) <= 120 and tag.casefold() not in seen:
            seen.add(tag.casefold())
            tags.append(tag)
    return tags


def description(text: str) -> str:
    """Prose from a reply, as one paragraph."""
    body = clean(text)
    body = re.sub(r"(?im)^\s*(nl|description|caption)\s*[:：]\s*", "", body)
    return " ".join(body.split()).strip().strip('"').strip()


def markers(text: str) -> dict[str, Any]:
    """COUNT / APPEARANCE / TAGS / ENVIRONMENT tag lines and an NL paragraph, case-insensitively.

    A group the reply lists but leaves empty maps to an empty list; a group it omits is absent.
    """
    groups: dict[str, Any] = {}
    prose: list[str] = []
    in_nl = False
    for line in clean(text).splitlines():
        stripped = line.strip().lstrip("*#>-`").strip()
        head, colon, rest = stripped.partition(":")
        if not colon:
            head, colon, rest = stripped.partition("：")
        key = head.strip().strip("*").strip().lower()
        if colon and key in MARKERS:
            groups[key] = tag_list(rest.strip().strip("*"))
            in_nl = False
        elif colon and key == "nl":
            prose = [rest.strip().strip("*").strip()]
            in_nl = True
        elif in_nl:
            if not stripped:
                in_nl = False
            else:
                prose.append(stripped)
    nl = " ".join(part for part in prose if part).strip().strip('"').strip()
    if nl:
        groups["nl"] = nl
    return groups


def reject_refusal(text: str) -> None:
    """Inspect prose fields separately so a tag list cannot hide a refusal in NL."""
    groups = markers(text)
    parts = [text]
    if groups.get("nl"):
        parts.append(groups["nl"])
    parts.extend(", ".join(groups[key]) for key in MARKERS if key in groups)
    if any(looks_like_refusal(part) for part in parts):
        raise VlmError("the model refused this image")


class Pacer:
    """Spaces request starts across all workers by ``interval`` seconds."""

    def __init__(self, interval: float):
        self.interval = max(0.0, interval)
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self, cancel: threading.Event | None) -> None:
        if not self.interval:
            return
        with self.lock:
            start = max(time.monotonic(), self.next_start)
            self.next_start = start + self.interval
        while (delay := start - time.monotonic()) > 0:
            if cancel is not None and cancel.wait(min(delay, 0.2)):
                raise VlmStopped()
            if cancel is None:
                time.sleep(min(delay, 0.2))


def _redacted(text: str, key: str | None) -> str:
    """Service error text as recorded: a key echoed back never reaches logs or operation records."""
    return text.replace(key, "***") if key else text


def _error_detail(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read(20000).decode("utf-8", "replace"))
        detail = payload.get("error", payload)
        if isinstance(detail, dict):
            detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
        return str(detail)[:400]
    except (ValueError, OSError, AttributeError):
        return error.reason if isinstance(error.reason, str) else ""


def _request(url: str, key: str | None, body: dict | None, *, policy: Any, timeout: float) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "YPuddinTrainStudio",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if body is not None else "GET"
    )
    with opener_for(url, policy).open(request, timeout=timeout) as response:
        payload = json.loads(response.read(64 * 1024 * 1024).decode("utf-8", "replace"))
    if not isinstance(payload, dict):
        raise VlmError("the service returned an unexpected response")
    return payload


def list_models(provider: str, requested_url: str | None, key: str | None, *, policy: Any) -> list[str]:
    url = base_url(provider, requested_url) + "/models"
    try:
        payload = _request(url, key, None, policy=policy, timeout=20)
    except urllib.error.HTTPError as error:
        raise ApiError(
            f"model list failed with HTTP {error.code}: {_redacted(_error_detail(error), key)}".rstrip(": "),
            code="vlm.models",
            status=502,
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, VlmError) as error:
        raise ApiError(
            f"cannot reach {url}: {getattr(error, 'reason', error)}", code="vlm.unreachable", status=502
        ) from None
    rows = payload.get("data") if isinstance(payload.get("data"), list) else payload.get("models", [])
    names = {
        str(row.get("id") or row.get("name")).removeprefix("models/")
        for row in rows
        if isinstance(row, dict) and (row.get("id") or row.get("name"))
    }
    return sorted(names, key=str.casefold)


def caption(
    image: str | Path,
    prompt: str,
    *,
    provider: str,
    url: str,
    key: str | None,
    model: str,
    policy: Any,
    temperature: float,
    max_tokens: int | None,
    image_size: int,
    detail: str,
    timeout: float,
    retries: int,
    pacer: Pacer,
    cancel: threading.Event | None,
) -> str:
    """One image's reply text. Retries rate limits, server errors and timeouts with backoff."""
    picture: dict[str, Any] = {"url": image_data_url(image, image_size)}
    if detail:
        picture["detail"] = detail
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": picture},
                ],
            }
        ],
    }
    if temperature > 0:
        body["temperature"] = temperature
    if max_tokens:
        body["max_tokens"] = max_tokens
    endpoint = url + "/chat/completions"
    payload: dict = {}
    for attempt in range(retries + 1):
        pacer.wait(cancel)
        if cancel is not None and cancel.is_set():
            raise VlmStopped()
        try:
            payload = _request(endpoint, key, body, policy=policy, timeout=timeout)
            break
        except urllib.error.HTTPError as error:
            detail_text = _redacted(_error_detail(error), key)
            if error.code in {401, 403}:
                raise VlmFatal(
                    f"{provider} rejected the API key (HTTP {error.code}): {detail_text}".rstrip(": "),
                    code="vlm.auth",
                    status=502,
                ) from None
            if error.code == 404:
                raise VlmFatal(
                    f"{endpoint} or model {model} was not found (HTTP 404): {detail_text}".rstrip(": "),
                    code="vlm.not_found",
                    status=502,
                ) from None
            retryable = error.code == 429 or error.code >= 500
            if not retryable or attempt == retries:
                raise VlmError(f"HTTP {error.code}: {detail_text}".rstrip(": ")) from None
            wait = error.headers.get("Retry-After") if error.headers else None
            delay = float(wait) if wait and wait.isdigit() else 2.0 * 2**attempt
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == retries:
                reason = getattr(error, "reason", error)
                raise VlmError(f"cannot reach {url}: {reason}") from None
            delay = 2.0 * 2**attempt
        except ValueError as error:
            raise VlmError(f"the service returned invalid JSON: {error}") from None
        if cancel is not None and cancel.wait(min(delay, 60.0)):
            raise VlmStopped()
        if cancel is None:
            time.sleep(min(delay, 60.0))
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise VlmError("the service returned no answer")
    choice = choices[0]
    message = choice.get("message") or {}
    if message.get("refusal"):
        raise VlmError("the model refused this image")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    text = clean(content or "")
    if choice.get("finish_reason") in {"content_filter", "safety"}:
        raise VlmError("the service's content filter refused this image")
    if not text:
        reason = choice.get("finish_reason")
        raise VlmError("the reply was empty" + (f" (finish reason: {reason})" if reason else ""))
    reject_refusal(text)
    return text
