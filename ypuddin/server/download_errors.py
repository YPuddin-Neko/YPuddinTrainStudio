"""Authentication failures for model downloads, without exposing response contents."""

from __future__ import annotations

import re
from urllib.error import HTTPError
from urllib.parse import urlsplit

_OFFICIAL_HOSTS = {
    "huggingface": {"huggingface.co"},
    "modelscope": {"modelscope.cn", "www.modelscope.cn"},
}


def download_auth_error(
    error: HTTPError, *, provider: str, authenticated: bool,
    gated: bool = False, mirror: bool = False,
) -> str | None:
    """Classify only authentication failures; never read or echo the response body."""
    if error.code not in (401, 403, 407):
        return None
    status = f"（HTTP {error.code}）"
    if error.code == 407:
        return "代理服务器要求身份验证" + status
    if mirror:
        return "镜像下载源拒绝访问" + status
    try:
        source = urlsplit(error.url)
        official = source.scheme == "https" and source.hostname in _OFFICIAL_HOSTS.get(provider, ())
    except ValueError:
        official = False
    if not official:
        return "文件下载服务器拒绝访问" + status

    headers = {key.lower(): str(value)[:4096] for key, value in (error.headers or {}).items()}
    code = headers.get("x-error-code", "").strip().lower()
    detail = " ".join(headers.get("x-error-message", "").lower().split())
    if code == "reponotfound":
        return "找不到模型仓库，或无权访问" + status

    restricted = gated or code == "gatedrepo" or bool(re.search(
        r"\b(?:accept|agree to)\b.{0,60}\b(?:license|licence|terms)\b", detail,
    ))
    if provider == "huggingface" and authenticated:
        expired = code in {"tokenexpired", "expiredtoken"} or bool(re.search(
            r"\btoken (?:has |is )?expired\b|\bexpired (?:access |oauth )?token\b", detail,
        ))
        invalid = code in {"invalidtoken", "invalidcredentials", "invalidoauthtoken"} or bool(re.search(
            r"\binvalid (?:access |oauth |authentication |auth |bearer )?token\b"
            r"|\btoken (?:is |was )?(?:invalid|revoked)\b"
            r"|\binvalid credentials in authorization header\b", detail,
        ))
        if expired or invalid:
            return ("访问令牌已过期" if expired else "访问令牌无效") + status

    if restricted:
        return "没有该模型的下载权限" + status
    if code in {"accessdenied", "insufficientpermissions"} or re.search(r"\baccess (?:is )?denied\b", detail):
        return "没有该模型的下载权限" + status
    return "下载源拒绝访问" + status
