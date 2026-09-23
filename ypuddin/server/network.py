"""Per-request outbound proxy policy; never mutate the service or OS environment."""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
PASSWORD_REVISION = "_network_password_revision"


def validate_proxy_settings(value: dict) -> dict:
    """Validate before Pydantic can include an accidentally pasted credential URL in errors."""
    result = dict(value)
    mode = result.get("proxy_mode", "system")
    if mode not in ("system", "direct", "custom"):
        raise ValueError("network.proxy_mode must be system, direct or custom")
    url = result.get("proxy_url", "")
    username = result.get("proxy_username", "")
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("Proxy address must be an HTTP(S) URL")
    url = url.strip()
    if url:
        try:
            parsed = urllib.parse.urlsplit(url)
            valid = (
                parsed.scheme in ("http", "https")
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
                and parsed.path in ("", "/")
                and not parsed.query
                and not parsed.fragment
                and not any(c.isspace() or ord(c) < 32 for c in url)
                and parsed.port != 0
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Proxy address must be http(s)://host:port, without credentials, path or query")
    if mode == "custom" and not url:
        raise ValueError("Custom proxy requires an HTTP(S) proxy address")
    if not isinstance(username, str) or len(username) > 512 or any(ord(c) < 32 for c in username):
        raise ValueError("Invalid proxy username")
    result.update(proxy_mode=mode, proxy_url=url.rstrip("/"), proxy_username=username)
    return result


class ProxyCredentials:
    """A private file separate from public settings; callers serialize updates with settings."""

    def __init__(self, root: Path):
        self.path = root / "network" / "secrets.json"

    def password(self, revision: str | None = None) -> str:
        try:
            if revision is None:
                settings = self.path.parent.parent / "settings.json"
                revision = json.loads(settings.read_text("utf-8")).get(PASSWORD_REVISION, "") if settings.exists() else ""
            if not self.path.exists():
                if revision:
                    raise ValueError()
                return ""
            document = json.loads(self.path.read_text("utf-8"))
            # Before revisioned commits, this file contained only proxy_password.
            value = document["passwords"][revision] if "passwords" in document else document["proxy_password"] if not revision else None
            if not isinstance(value, str):
                raise ValueError()
            return value
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("Cannot read the private proxy credentials file") from None

    def prepare(self, password: str, current_revision: str, new_revision: str) -> None:
        """Prepare without activating: settings.json's atomic revision switch commits.

        Keep the current credential reachable on failure/process exit between the two
        writes. At most two entries survive; a subsequent preparation drops an orphan.
        """
        if not isinstance(password, str) or len(password) > 4096 or any(ord(c) < 32 for c in password):
            raise ValueError("Invalid proxy password")
        previous = self.password(current_revision)
        self._write({"passwords": {current_revision: previous, new_revision: password}})

    def finish(self, revision: str) -> None:
        if self.path.exists():
            self._write({"passwords": {revision: self.password(revision)}})

    def _write(self, document: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.path.parent, prefix=".secrets-", encoding="utf-8", delete=False
        ) as f:
            temporary = Path(f.name)
            try:
                json.dump(document, f)
                f.flush()
                os.fsync(f.fileno())
                f.close()
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ProxyPolicy:
    mode: str = "system"
    url: str = ""
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)

    @classmethod
    def from_context(cls, context) -> ProxyPolicy:
        # Read public policy and its secret under the same save lock. Existing requests
        # retain their snapshot while the next request sees newly saved preferences.
        with context._settings_lock:
            settings = context.settings()["network"]
            return cls(
                settings["proxy_mode"], settings["proxy_url"], settings["proxy_username"],
                ProxyCredentials(context.data_root).password(),
            )

    def authenticated_url(self) -> str:
        if not self.username and not self.password:
            return self.url
        parsed = urllib.parse.urlsplit(self.url)
        auth = urllib.parse.quote(self.username, safe="") + ":" + urllib.parse.quote(self.password, safe="")
        return urllib.parse.urlunsplit(parsed._replace(netloc=auth + "@" + parsed.netloc))

    def opener(self, *handlers):
        proxies = None if self.mode == "system" else {}
        if self.mode == "custom":
            proxies = dict.fromkeys(("http", "https"), self.authenticated_url())
        # Existing destination/redirect restrictions remain independent handlers.
        # HTTPSHandler retains Python's normal certificate verification.
        return urllib.request.build_opener(urllib.request.ProxyHandler(proxies), *handlers)

    def subprocess_env(self, source: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if source is None else source)
        if self.mode != "system":
            for key in tuple(env):
                if key.lower() in ("http_proxy", "https_proxy", "all_proxy"):
                    env.pop(key, None)
        if self.mode == "custom":
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                env[key] = self.authenticated_url()
        # The application never proxies its own loopback health/control endpoints.
        bypass = ",".join(value for key, value in env.items() if key.lower() == "no_proxy")
        hosts = list(dict.fromkeys(part.strip() for part in bypass.split(",") if part.strip()))
        for host in ("localhost", "127.0.0.1", "::1"):
            if host not in hosts:
                hosts.append(host)
        env["NO_PROXY"] = env["no_proxy"] = ",".join(hosts)
        return env

    def redact(self, value: object) -> str:
        text = str(value)
        # Redact inherited authenticated system proxies too, without recording their URLs.
        urls = [self.authenticated_url()] if self.mode == "custom" else []
        urls.extend(value for key, value in os.environ.items() if key.lower() in ("http_proxy", "https_proxy", "all_proxy"))
        secrets = {self.password, urllib.parse.quote(self.password, safe="")}
        for url in urls:
            try:
                parsed = urllib.parse.urlsplit(url)
                if parsed.password:
                    secrets.update((parsed.password, urllib.parse.unquote(parsed.password)))
            except ValueError:
                pass
        text = re.sub(r"(?i)(https?://)[^\s/]*@", r"\1***@", text)
        for secret in sorted(filter(None, secrets), key=len, reverse=True):
            text = text.replace(secret, "***")
        return text
