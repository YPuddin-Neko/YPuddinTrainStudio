"""Danbooru and Gelbooru for regularization: paced searches, match counts and media downloads.

Every request goes through the service proxy policy (see RegularizationManager._open). API and
media requests are paced separately and slow down when a site answers HTTP 429; busy or failing
responses are retried a few times. Credentials go to the site API only, never to media hosts.
"""

from __future__ import annotations

import base64
import http.client
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

import ypuddin

from .errors import ApiError

API_BYTES = 8 * 1024 * 1024
MEDIA_EXTS = frozenset({"png", "jpg", "jpeg", "webp"})
SAFE_RATINGS = frozenset({"g", "general", "safe"})
ATTEMPTS = 4
# Danbooru meta tags describe the file or its upload, not the picture. Gelbooru does not label tag
# categories, so its best-known meta tags are recognized by name.
META_TAG = re.compile(
    r"^(highres|absurdres|incredibly_absurdres|lowres|tagme|translated|translation_request|commentary"
    r"|commentary_request|revision|variant_set|md5_mismatch|duplicate|bad_id|bad_link"
    r"|bad_[a-z]+_id|[a-z]+_commentary|partial_commentary|[a-z_]+_request|third-party_edit)$"
)


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class Site:
    name: str
    label: str
    base: str
    media_suffix: str
    page_size: int


SITES = {
    "danbooru": Site("danbooru", "Danbooru", "https://danbooru.donmai.us", ".donmai.us", 200),
    "gelbooru": Site("gelbooru", "Gelbooru", "https://gelbooru.com", ".gelbooru.com", 100),
}


def allowed_media(url: str, source: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and not parsed.username
        and not parsed.password
        and port in (None, 443)
        and host.endswith(SITES[source].media_suffix)
    )


class Redirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, source: str, media: bool):
        self.source, self.media = source, media

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not self.media or not allowed_media(newurl, self.source):
            raise ApiError(
                "The provider redirected outside its allowed media hosts", code="regularization.redirect"
            )
        # Media requests never carry API credentials or cookies; the agent and referrer stay.
        kept = {
            name: value for name, value in req.header_items() if name.lower() in ("user-agent", "referer")
        }
        return urllib.request.Request(newurl, headers=kept or {"User-Agent": user_agent()})


def user_agent(account: str = "") -> str:
    agent = f"YPuddinTrainStudio/{ypuddin.__version__}"
    # Danbooru asks API clients to name themselves and the account they act for.
    return f"{agent} (by {account} on Danbooru)" if account else agent


def normalize(tag: str) -> str:
    return " ".join(tag.replace("_", " ").casefold().split())


@dataclass(frozen=True)
class Post:
    id: str
    rating: str
    media: str
    ext: str
    width: int
    height: int
    # Caption tags in the site's order, without meta tags; every tag, normalized, for exclusions.
    tags: tuple[str, ...]
    all_tags: frozenset[str]
    page: str
    source: str

    @property
    def safe(self) -> bool:
        return self.rating in SAFE_RATINGS


class Pace:
    """Spaces requests to at most `rate` per second across threads; halves on request, to a floor."""

    def __init__(self, rate: float, floor: float = 0.2):
        self.rate, self.floor = rate, floor
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self, cancelled: threading.Event) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + 1 / self.rate
        if start > now and cancelled.wait(start - now):
            raise Cancelled()

    def slow_down(self) -> None:
        with self.lock:
            self.rate = max(self.floor, self.rate / 2)


class TooLarge(ApiError):
    """One file over its size limit; a batch skips it and carries on."""

    def __init__(self):
        super().__init__("Regularization download exceeded its byte limit", code="regularization.too_large")


class TagLimitError(ApiError):
    def __init__(self, site: Site, limit: int):
        super().__init__(
            f"{site.label} 当前账号每次最多检索 {limit} 个标签（rating 等条件不计入），请减少检索标签。",
            code="regularization.tag_limit",
            status=422,
            details={"tag_limit": limit},
        )
        self.limit = limit


def _message(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read(64 * 1024) or b"{}")
    except (AttributeError, OSError, ValueError):
        return ""
    return str(body.get("message") or body.get("reason") or "") if isinstance(body, dict) else ""


def read_limited(
    response, limit: int, cancelled: threading.Event, progress: Callable[[int], None] | None = None
):
    chunks, size = [], 0
    while chunk := response.read(256 * 1024):
        if cancelled.is_set():
            raise Cancelled()
        size += len(chunk)
        if size > limit:
            raise TooLarge()
        if progress:
            progress(len(chunk))
        chunks.append(chunk)
    return b"".join(chunks)


class BooruClient:
    """One site, one account, one task: shares its pacing across the task's download threads."""

    def __init__(
        self,
        source: str,
        account: str = "",
        api_key: str = "",
        *,
        opener: Callable[[urllib.request.Request, bool], object],
        cancelled: threading.Event | None = None,
        notify: Callable[[str], None] | None = None,
        max_file_bytes: int = 40 * 1024 * 1024,
    ):
        self.site = SITES[source]
        self.account, self.api_key = account, api_key
        self.opener = opener
        self.cancelled = cancelled or threading.Event()
        self.notify = notify or (lambda message: None)
        self.max_file_bytes = max_file_bytes
        self.api_pace, self.media_pace = Pace(2), Pace(5)
        self._tag_limit: int | None | bool = False  # False: not asked yet; None: unlimited.

    @property
    def authenticated(self) -> bool:
        return bool(self.account and self.api_key)

    def _sleep(self, seconds: float) -> None:
        if self.cancelled.wait(seconds):
            raise Cancelled()

    def _fetch(
        self, url: str, *, media: bool, limit: int, headers: dict | None = None, progress=None
    ) -> bytes:
        account = self.account if self.site.name == "danbooru" else ""
        for attempt in range(1, ATTEMPTS + 1):
            (self.media_pace if media else self.api_pace).wait(self.cancelled)
            request = urllib.request.Request(
                url, headers={"User-Agent": user_agent(account), **(headers or {})}
            )
            try:
                with self.opener(request, media) as response:
                    return read_limited(response, limit, self.cancelled, progress)
            except urllib.error.HTTPError as error:
                code = error.code
                if attempt == ATTEMPTS or code not in (429, 500, 502, 503, 504):
                    raise self._failure(error) from None
                if code == 429:
                    # Asked to slow down: halve both paces and wait as long as the site says.
                    self.api_pace.slow_down()
                    self.media_pace.slow_down()
                    try:
                        delay = min(120.0, max(1.0, float((error.headers or {}).get("Retry-After") or 60)))
                    except (TypeError, ValueError):
                        delay = 60.0
                    self.notify(f"{self.site.label} asked to slow down; retrying in {round(delay)} s")
                else:
                    delay = 15.0 if code == 503 else 5.0 * attempt
                    self.notify(f"{self.site.label} is busy (HTTP {code}); retrying in {round(delay)} s")
                self._sleep(delay)
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                if attempt == ATTEMPTS:
                    raise ApiError(
                        f"Could not read {self.site.name}; check connectivity or retry later",
                        code="regularization.provider",
                    ) from None
                self._sleep(2.0**attempt)
        raise AssertionError("unreachable")

    def _failure(self, error: urllib.error.HTTPError) -> ApiError:
        message = _message(error)
        if (
            self.site.name == "danbooru"
            and error.code == 422
            and (found := re.search(r"more than (\d+) tags", message))
        ):
            return TagLimitError(self.site, int(found.group(1)))
        if error.code in (401, 403):
            return ApiError(
                f"{self.site.name} returned HTTP {error.code}; check site credentials or retry later",
                code="regularization.credentials",
            )
        return ApiError(
            f"{self.site.name} returned HTTP {error.code}; check site credentials or retry later",
            code="regularization.provider",
        )

    def _json(self, path: str, params: dict, *, authenticate: bool = True):
        headers = {"Accept": "application/json"}
        if self.authenticated and authenticate:
            if self.site.name == "danbooru":
                token = base64.b64encode(f"{self.account}:{self.api_key}".encode()).decode()
                headers["Authorization"] = "Basic " + token
            else:
                params = {**params, "user_id": self.account, "api_key": self.api_key}
        raw = self._fetch(
            f"{self.site.base}{path}?{urllib.parse.urlencode(params)}",
            media=False,
            limit=API_BYTES,
            headers=headers,
        )
        if not raw.strip():
            return {}  # Gelbooru answers an empty body when nothing matches.
        try:
            return json.loads(raw)
        except ValueError:
            raise ApiError(
                f"Could not read {self.site.name}; check connectivity or retry later",
                code="regularization.provider",
            ) from None

    def tag_limit(self) -> int | None:
        """Tags one Danbooru search may name (None: no limit). Danbooru does not count `rating:` and
        other size or file metatags; negated tags count."""
        if self.site.name != "danbooru":
            return None
        if self._tag_limit is False:
            level = 0
            if self.authenticated:
                profile = self._json("/profile.json", {})
                level = profile.get("level", 0) if isinstance(profile, dict) else 0
                level = level if isinstance(level, int) else 0
            # Levels: member 20, gold 30, platinum 31 and above.
            self._tag_limit = None if level >= 31 else 6 if level == 30 else 2
        return self._tag_limit

    def query(self, tags: list[str], excluded: list[str]) -> tuple[list[str], list[str]]:
        """The search terms within the account's tag limit, and the exclusions left to check locally."""
        limit = self.tag_limit()
        if limit is not None and len(tags) > limit:
            raise TagLimitError(self.site, limit)
        room = len(excluded) if limit is None else max(0, limit - len(tags))
        negated = [f"-{tag.replace(' ', '_')}" for tag in excluded[:room]]
        return [*tags, *negated, "rating:general"], excluded[room:]

    def count(self, terms: list[str]) -> int | None:
        query = " ".join(terms)
        if self.site.name == "danbooru":
            data = self._json("/counts/posts.json", {"tags": query})
            value = (data.get("counts") or {}).get("posts") if isinstance(data, dict) else None
        else:
            data = self._json(
                "/index.php",
                {"page": "dapi", "s": "post", "q": "index", "json": "1", "tags": query, "limit": 1},
            )
            value = (data.get("@attributes") or {}).get("count") if isinstance(data, dict) else None
        return int(value) if isinstance(value, (int, str)) and str(value).isdigit() else None

    def search(self, terms: list[str], page: int, limit: int | None = None) -> list[Post]:
        size = min(limit or self.site.page_size, self.site.page_size)
        query = " ".join(terms)
        if self.site.name == "danbooru":
            rows = self._json("/posts.json", {"tags": query, "page": page, "limit": size})
        else:
            data = self._json(
                "/index.php",
                {
                    "page": "dapi",
                    "s": "post",
                    "q": "index",
                    "json": "1",
                    "tags": query,
                    "pid": page - 1,
                    "limit": size,
                },
            )
            rows = data.get("post", []) if isinstance(data, dict) else []
            rows = [rows] if isinstance(rows, dict) else rows
        if not isinstance(rows, list):
            raise ApiError(
                f"Could not read {self.site.name}; check connectivity or retry later",
                code="regularization.provider",
            )
        return [post for row in rows if (post := self._post(row)) is not None]

    def _post(self, row) -> Post | None:
        if not isinstance(row, dict):
            return None
        pid, media = str(row.get("id", "")), row.get("file_url")
        if not pid.isdigit() or not isinstance(media, str) or not allowed_media(media, self.site.name):
            return None
        danbooru = self.site.name == "danbooru"
        # Gelbooru leaves out file_ext; the file name says it.
        ext = str(
            row.get("file_ext") or PurePosixPath(urllib.parse.urlsplit(media).path).suffix.lstrip(".")
        ).lower()
        text = row.get("tag_string" if danbooru else "tags", "")
        if not isinstance(text, str):
            return None
        tags = text.split()
        meta = (
            set(row.get("tag_string_meta", "").split())
            if danbooru and isinstance(row.get("tag_string_meta"), str)
            else set()
        )
        caption = tuple(tag for tag in tags if tag not in meta and not META_TAG.match(tag))

        def size(*keys):
            value = next((row.get(key) for key in keys if row.get(key) is not None), 0)
            return int(value) if isinstance(value, (int, str)) and str(value).isdigit() else 0

        return Post(
            id=pid,
            rating=str(row.get("rating", "")),
            media=media,
            ext=ext,
            width=size("image_width", "width"),
            height=size("image_height", "height"),
            tags=caption,
            all_tags=frozenset(normalize(tag) for tag in tags),
            page=f"{self.site.base}/posts/{pid}"
            if danbooru
            else f"{self.site.base}/index.php?page=post&s=view&id={pid}",
            source=str(row.get("source") or ""),
        )

    def download(self, post: Post, progress: Callable[[int], None] | None = None) -> bytes:
        # The site's own origin as referrer: some media hosts refuse requests without one.
        return self._fetch(
            post.media,
            media=True,
            limit=self.max_file_bytes,
            headers={"Referer": self.site.base + "/"},
            progress=progress,
        )
