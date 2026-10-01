"""Standard-library-only package source policy, shared by bootstrap and the server."""

from __future__ import annotations

import copy
import http.client
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait

PYPI = {
    "ustc": "https://mirrors.ustc.edu.cn/pypi/simple",
    "tuna": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple",
    "official": "https://pypi.org/simple",
}
PYTORCH = {
    "sjtu": ("index-url", "https://mirror.sjtu.edu.cn/pytorch-wheels"),
    "aliyun": ("find-links", "https://mirrors.aliyun.com/pytorch-wheels"),
    "official": ("index-url", "https://download.pytorch.org/whl"),
}
_NAMES = {"ustc": "中国科学技术大学", "tuna": "清华大学", "aliyun": "阿里云", "sjtu": "上海交通大学"}
_HOSTS = frozenset(urllib.parse.urlsplit(url).hostname for url in [*PYPI.values(), *(v[1] for v in PYTORCH.values())])
_PROBE_TIMEOUT = 2.0
_PROBE_DEADLINE = 5.0
_PROBE_TTL = 600.0
_PROBE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="package-sources")
_PROBE_LOCK = threading.Lock()
_PROBE_CACHE = OrderedDict()


class _SourceRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if (
            target.scheme != "https" or target.hostname not in _HOSTS
            or target.port not in (None, 443) or target.username is not None or target.password is not None
        ):
            raise ValueError("Package source redirect is outside the allowed sources")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.method = req.get_method()
        return redirected


def _source_candidates():
    return {
        "pypi": [{"id": id_, "name": _NAMES.get(id_, "PyPI 官方"), "url": url} for id_, url in PYPI.items()],
        "pytorch": [
            {"id": id_, "name": _NAMES.get(id_, "PyTorch 官方"), "url": url}
            for id_, (_, url) in PYTORCH.items()
        ],
    }


def _probe_candidate(candidate, opener_factory):
    started = time.monotonic()
    result = dict(candidate, latency_ms=None, available=False)
    try:
        opener = opener_factory(_SourceRedirect())
        request = urllib.request.Request(candidate["url"] + "/", method="HEAD", headers={"User-Agent": "YPuddinTrainStudio"})
        with opener.open(request, timeout=_PROBE_TIMEOUT) as response:
            if 200 <= response.status < 300:
                result.update(available=True, latency_ms=round((time.monotonic() - started) * 1000, 1))
    except (OSError, ValueError, http.client.HTTPException):
        pass
    return result


def probe_sources(*, force=False, opener_factory=None, cache_key=None):
    """Measure fixed source indexes without downloading packages or changing a saved choice."""
    if cache_key is None:
        cache_key = tuple(sorted((k, v) for k, v in os.environ.items() if k.lower().endswith("_proxy")))
    fallback = {"checked_at": time.time(), **{
        group: [dict(item, latency_ms=None, available=False) for item in candidates]
        for group, candidates in _source_candidates().items()
    }}
    requested_at = time.monotonic()
    if not _PROBE_LOCK.acquire(timeout=_PROBE_DEADLINE):
        return fallback
    try:
        cached = _PROBE_CACHE.get(cache_key)
        if cached and time.monotonic() - cached[0] < _PROBE_TTL and (not force or cached[0] >= requested_at):
            return copy.deepcopy(cached[1])
        factory = opener_factory or urllib.request.build_opener
        futures = {
            _PROBE_POOL.submit(_probe_candidate, candidate, factory): (group, index)
            for group, candidates in _source_candidates().items()
            for index, candidate in enumerate(candidates)
        }
        completed, pending = wait(futures, timeout=max(0, _PROBE_DEADLINE - (time.monotonic() - requested_at)))
        for future in pending:
            future.cancel()
        for future in completed:
            group, index = futures[future]
            try:
                fallback[group][index] = future.result()
            except Exception:  # A failed source check must not prevent installation from trying it.
                pass
        _PROBE_CACHE[cache_key] = (time.monotonic(), fallback)
        _PROBE_CACHE.move_to_end(cache_key)
        while len(_PROBE_CACHE) > 16:
            _PROBE_CACHE.popitem(last=False)
        return copy.deepcopy(fallback)
    finally:
        _PROBE_LOCK.release()


def _automatic_order(group, **probe_options):
    rows = probe_sources(**probe_options)[group]
    return [item["id"] for item in sorted(rows, key=lambda item: (
        not item["available"], item["latency_ms"] if item["latency_ms"] is not None else float("inf")
    ))]


def pypi_sources(source="auto", fallback=True, **probe_options):
    if source == "auto":
        ordered = _automatic_order("pypi", **probe_options)
        return [PYPI[id_] for id_ in ordered[:None if fallback else 1]]
    primary = PYPI[source]
    return [primary] + ([url for url in PYPI.values() if url != primary] if fallback else [])


def torch_sources(backend, source="auto", fallback=True, **probe_options):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", backend):
        raise ValueError("Invalid PyTorch backend")
    candidates = {id_: (kind, f"{url}/{backend}") for id_, (kind, url) in PYTORCH.items()}
    if source == "auto":
        ordered = _automatic_order("pytorch", **probe_options)
        return [candidates[id_] for id_ in ordered[:None if fallback else 1]]
    official = candidates["official"]
    mirrors = [candidates["sjtu"], candidates["aliyun"]]
    if source == "mirror":
        return mirrors + ([official] if fallback else [])
    selected = {"sjtu": mirrors[0], "aliyun": mirrors[1], "official": official}[source]
    return [selected] + ([item for item in [*mirrors, official] if item != selected] if fallback else [])


def run_sources(installer, commands, log, cancel, *, timeout=3600):
    """Try separate indexes in order; never merge indexes or swallow cancellation."""
    for index, (url, command) in enumerate(commands):
        if cancel.is_set():
            raise InterruptedError("Installation cancelled")
        log(f"下载源：{url}")
        try:
            installer.run(command, log, cancel, timeout=timeout)
            return
        except InterruptedError:
            raise
        except (RuntimeError, TimeoutError):
            if cancel.is_set():
                raise InterruptedError("Installation cancelled") from None
            if index + 1 == len(commands):
                raise
            log("当前下载源不可用或缺少兼容包，正在尝试下一个来源。")
