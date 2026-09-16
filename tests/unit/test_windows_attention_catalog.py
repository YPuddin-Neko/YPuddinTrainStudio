"""Release discovery, proxy use, integrity and platform gating; no Windows binaries execute."""

import copy
import hashlib
import io
import json
import threading
import urllib.parse
from types import SimpleNamespace

import pytest

from ypuddin.server import windows_attention_catalog as vendor
from ypuddin.server.network import ProxyPolicy


def runtime(**changes):
    return dict(
        platform="Windows",
        machine="AMD64",
        python="3.12.10",
        torch="2.11.0+cu128",
        cuda_runtime="12.8",
        cuda_available=True,
        **changes,
    )


def candidate():
    return next(w for w in vendor.BUNDLED if w.python_tag == "cp312" and w.cuda == "12.8")


def release(tag="v0.9.52", *, torch="2.13", cuda="130", version="2.8.3"):
    filename = f"flash_attn-{version}+cu{cuda}torch{torch}-cp312-cp312-win_amd64.whl"
    asset = dict(vendor.ASSETS["assets"][7], name=filename)
    asset["browser_download_url"] = vendor.DOWNLOAD_PREFIX + tag + "/" + urllib.parse.quote(filename, safe="")
    return {"tag_name": tag, "draft": False, "prerelease": False, "assets": [asset]}


def metadata_opener(monkeypatch, respond):
    calls = []

    def open_url(request, **kwargs):
        calls.append(request.full_url)
        return Response(json.dumps(respond(request.full_url)).encode(), request.full_url)

    monkeypatch.setattr(ProxyPolicy, "opener", lambda *args: SimpleNamespace(open=open_url))
    return calls


class Response(io.BytesIO):
    def __init__(self, data, url, length=None):
        super().__init__(data)
        self.url = url
        self.headers = {"Content-Length": str(len(data) if length is None else length)}

    def geturl(self):
        return self.url


def test_bundled_release_has_real_pinned_assets_without_implying_gpu_validation():
    assert len(vendor.BUNDLED) == 15
    w = candidate()
    assert w.size_bytes == 250730469
    assert w.sha256 == "2a3268e5e50dbf332f21c155a74d1707bd3b7c2e435a460ff9c05aa337cf8f7e"
    assert w.validation == "kernel_probe_required" and not w.compatible
    assert sum(vendor.incompatibility(w, runtime()) is None for w in vendor.BUNDLED) == 1


def test_new_release_matches_torch213_and_keeps_release_specific_download_without_fa3(tmp_path):
    doc = release()
    doc["assets"].append({"name": "flash_attn_3-3.0.0+cu130torch2.13-cp39-abi3-win_amd64.whl"})
    doc["assets"].append({"name": "flash_attn-3.0.0+cu130torch2.13-cp312-cp312-win_amd64.whl"})
    (wheel,) = vendor.parse_assets(doc)
    assert wheel.release == "v0.9.52" and wheel.torch == "2.13"
    assert (
        vendor.incompatibility(wheel, {**runtime(), "torch": "2.13.0+cu130", "cuda_runtime": "13.0"}) is None
    )
    assert vendor.incompatibility(wheel, runtime()) == "torch_version_mismatch"
    assert wheel.source_url.endswith("/releases/tag/v0.9.52")
    assert vendor.permitted_download(wheel.url)
    payload = b"mock wheel bytes, not executable"
    wheel = wheel.model_copy(
        update={"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    )
    urls = []

    def open_url(request, **kwargs):
        urls.append(request.full_url)
        return Response(payload, request.full_url)

    downloaded = vendor.download(wheel, tmp_path, threading.Event(), lambda *a: None, opener=open_url)
    assert downloaded.read_bytes() == payload
    assert urls == [doc["assets"][0]["browser_download_url"]]
    assert "/v0.9.6/" not in urls[0]


def test_discovery_paginates_past_linux_and_deduplicates_identical_assets(monkeypatch):
    monkeypatch.setattr(vendor, "RELEASES_PER_PAGE", 2)
    newest = release("v0.9.53")
    linux = {"tag_name": "v0.9.54", "assets": [{"name": "flash_attn-2.8.3-linux.whl"}]}
    pages = {1: [linux, newest], 2: [release(), release("v0.9.40", version="2.7.4")], 3: []}
    calls = metadata_opener(
        monkeypatch,
        lambda url: pages[int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["page"][0])],
    )
    result = vendor.Catalog().snapshot(
        {**runtime(), "torch": "2.13.0+cu130", "cuda_runtime": "13.0"}, "windows-cuda"
    )
    assert result.origin == "live" and not result.limited
    assert result.release_count == 4 and len(calls) == 3
    assert [w.version for w in result.wheels] == ["2.8.3+cu130torch2.13", "2.7.4+cu130torch2.13"]
    assert all(w.compatible for w in result.wheels)
    assert result.wheels[0].release == "v0.9.53"
    assert calls == [f"{vendor.API_URL}?per_page=2&page={page}" for page in (1, 2, 3)]


def test_discovery_bound_and_unverified_assets_are_explicit(monkeypatch):
    monkeypatch.setattr(vendor, "RELEASES_PER_PAGE", 1)
    monkeypatch.setattr(vendor, "MAX_RELEASE_PAGES", 2)
    doc = release()
    unsigned = release(torch="2.12")["assets"][0]
    unsigned["digest"] = None
    doc["assets"].append(unsigned)
    calls = metadata_opener(monkeypatch, lambda _: [doc])
    result = vendor.Catalog().snapshot(runtime(), "windows-cuda")
    assert len(calls) == 2 and result.limited
    assert result.unverified_assets == 2
    assert len(result.wheels) == 1 and result.wheels[0].torch == "2.13"


def test_manual_refresh_discovers_new_releases_and_partial_failure_keeps_last_complete_catalog(monkeypatch):
    documents = [vendor.ASSETS]
    calls = metadata_opener(monkeypatch, lambda _: documents)
    catalog = vendor.Catalog()
    assert catalog.snapshot(runtime(), "windows-cuda").origin == "live"
    assert catalog.snapshot(runtime(), "windows-cuda").origin == "cached"
    assert len(calls) == 1
    documents = [release(), vendor.ASSETS]
    new_runtime = {**runtime(), "torch": "2.13.0+cu130", "cuda_runtime": "13.0"}
    updated = catalog.snapshot(new_runtime, "windows-cuda", refresh=True)
    assert len(calls) == 2 and updated.origin == "live"
    assert [w.release for w in updated.wheels if w.compatible] == ["v0.9.52"]
    previous = updated.wheels
    monkeypatch.setattr(vendor, "RELEASES_PER_PAGE", 1)

    def partial_failure(url):
        if "page=2" in url:
            raise OSError("offline")
        return [release("v0.9.99", torch="2.14")]

    metadata_opener(monkeypatch, partial_failure)
    fallback = catalog.snapshot(new_runtime, "windows-cuda", refresh=True)
    assert fallback.origin == "cached" and fallback.error
    assert fallback.wheels == previous


@pytest.mark.parametrize("flag", ["draft", "prerelease"])
def test_discovery_ignores_unpublished_or_prerelease_builds(monkeypatch, flag):
    doc = release()
    doc[flag] = True
    metadata_opener(monkeypatch, lambda _: [doc, vendor.ASSETS])
    result = vendor.Catalog().snapshot(runtime(), "windows-cuda")
    assert result.release_count == 1
    assert len(result.wheels) == len(vendor.BUNDLED)
    assert all(w.release == "v0.9.6" for w in result.wheels)


def test_new_release_cannot_use_another_release_download_url(tmp_path):
    wheel = vendor.parse_assets(release())[0]
    wheel = wheel.model_copy(update={"url": wheel.url.replace("/v0.9.52/", "/v0.9.6/")})
    with pytest.raises(ValueError, match="Unverified"):
        vendor.download(
            wheel,
            tmp_path,
            threading.Event(),
            lambda *a: None,
            opener=lambda *a, **k: pytest.fail("must not fetch"),
        )


@pytest.mark.parametrize("malformed", [None, {}, [None], [{"tag_name": "v1", "assets": None}]])
def test_malformed_listing_retains_bundled_fallback(monkeypatch, malformed):
    metadata_opener(monkeypatch, lambda _: malformed)
    result = vendor.Catalog().snapshot(runtime(), "windows-cuda")
    assert result.origin == "bundled" and result.error
    assert len(result.wheels) == 15


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"platform": "Linux"}, "requires_windows_x86_64"),
        ({"machine": "arm64"}, "requires_windows_x86_64"),
        ({"hip_runtime": "6.3"}, "requires_windows_cuda"),
        ({"cuda_available": False}, "cuda_runtime_unavailable"),
        ({"python": "3.11.10"}, "python_abi_mismatch"),
        ({"torch": "2.12.0+cu128"}, "torch_version_mismatch"),
        ({"torch": "2.11.0rc1+cu128"}, "torch_version_mismatch"),
        ({"cuda_runtime": "13.0"}, "cuda_version_mismatch"),
        ({"torch": "2.11.0+das.opt1.dtk2604"}, "cuda_version_mismatch"),
    ],
)
def test_exact_platform_and_build_matching(changes, reason):
    assert vendor.incompatibility(candidate(), {**runtime(), **changes}) == reason
    assert vendor.incompatibility(candidate(), runtime(), profile="linux-dtk") == "requires_windows_cuda"


@pytest.mark.parametrize("change", ["tag", "url", "oversize", "duplicate"])
def test_release_metadata_cannot_authorize_arbitrary_sources_or_unsigned_assets(change):
    doc = copy.deepcopy(vendor.ASSETS)
    if change == "tag":
        doc["tag_name"] = "../outside"
    if change == "url":
        doc["assets"][0]["browser_download_url"] = "https://github.com/attacker/wheels/file.whl"
    if change == "oversize":
        doc["assets"][0]["size"] = 3 * 1024**3
    if change == "duplicate":
        doc["assets"].append(doc["assets"][0])
    with pytest.raises(ValueError):
        vendor.parse_assets(doc)


def test_live_metadata_uses_global_proxy_caches_then_falls_back_with_explicit_reason(monkeypatch):
    calls = []
    fail = False

    def open_url(request, **kwargs):
        calls.append(request.full_url)
        if fail:
            raise OSError("unreachable proxy secret-pass")
        return Response(json.dumps([vendor.ASSETS]).encode(), request.full_url)

    policy = ProxyPolicy("custom", "http://localhost:7890", "user", "secret-pass")
    observed = []

    def opener(self, *handlers):
        observed.append((self, handlers))
        return SimpleNamespace(open=open_url)

    monkeypatch.setattr(ProxyPolicy, "opener", opener)
    catalog = vendor.Catalog()
    live = catalog.snapshot(runtime(), "windows-cuda", proxy=policy)
    assert live.origin == "live" and live.error is None
    assert live.checked_at and len(live.wheels) == 15
    assert observed[0][0] == policy and isinstance(observed[0][1][0], vendor._ReleaseRedirect)
    catalog.snapshot(runtime(), "windows-cuda", proxy=policy)
    assert len(calls) == 1
    fail = True
    cached = catalog.snapshot(runtime(), "windows-cuda", proxy=policy, refresh=True)
    assert cached.origin == "cached" and "secret-pass" not in cached.error and "代理" in cached.error
    assert any(w.compatible for w in cached.wheels)
    bundled = vendor.Catalog().snapshot(runtime(), "windows-cuda", proxy=policy)
    assert bundled.origin == "bundled" and bundled.error and any(w.compatible for w in bundled.wheels)
    no_match = catalog.snapshot({**runtime(), "torch": "2.10.0+cu128"}, "windows-cuda", proxy=policy)
    assert no_match.reason == "no_matching_build"
    with pytest.raises(ValueError):
        catalog.find_wheel("https://untrusted.invalid/file.whl")


def test_non_windows_does_not_query_github(monkeypatch):
    monkeypatch.setattr(ProxyPolicy, "opener", lambda *a: pytest.fail("wrong-platform query"))
    result = vendor.Catalog().snapshot({**runtime(), "platform": "Darwin"}, "macos-mps")
    assert result.reason == "no_matching_build" and not any(w.compatible for w in result.wheels)


def test_download_uses_proxy_and_validates_signed_cdn_bytes_before_publishing(tmp_path, monkeypatch):
    data = b"fake wheel never executed" * 90000
    wheel = candidate().model_copy(
        update={"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    )
    calls, progress = [], []
    policy = ProxyPolicy("custom", "http://localhost:7890")

    def opener(self, *handlers):
        calls.append((self, handlers))
        return SimpleNamespace(
            open=lambda *a, **kw: Response(
                data,
                "https://release-assets.githubusercontent.com/github-production-release-asset/123/token?sig=abc",
            )
        )

    monkeypatch.setattr(ProxyPolicy, "opener", opener)
    result = vendor.download(
        wheel, tmp_path, threading.Event(), lambda *args: progress.append(args), proxy=policy
    )
    assert result.read_bytes() == data and calls[0][0] == policy
    assert isinstance(calls[0][1][0], vendor._AssetRedirect)
    assert progress[0] == (0, len(data), None, None) and progress[-1] == (len(data), len(data), None, 0)
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("mode", ["hash", "short", "oversize", "redirect", "cancel", "network"])
def test_failed_download_never_leaves_an_installable_wheel(tmp_path, mode):
    data = b"safe bytes"
    wheel = candidate().model_copy(
        update={"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    )
    cancel = threading.Event()

    def open_url(*args, **kwargs):
        if mode == "network":
            raise OSError("proxy secret-pass refused")
        received = (
            data[::-1]
            if mode == "hash"
            else data[:-1]
            if mode == "short"
            else data + b"x"
            if mode == "oversize"
            else data
        )
        if mode == "cancel":
            cancel.set()
        return Response(
            received, "https://untrusted.invalid/file.whl" if mode == "redirect" else wheel.url, len(data)
        )

    with pytest.raises(InterruptedError if mode == "cancel" else ValueError) as exc:
        vendor.download(
            wheel,
            tmp_path,
            cancel,
            lambda *a: None,
            opener=open_url,
            proxy=ProxyPolicy("custom", "http://localhost:7890", "u", "secret-pass"),
        )
    assert "secret-pass" not in str(exc.value)
    if mode == "network":
        assert "手动下载" in str(exc.value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/x",
        "https://evil.githubusercontent.com/github-production-release-asset/x",
        "https://github.com/other/release/file.whl",
        "https://user:pass@release-assets.githubusercontent.com/github-production-release-asset/x",
        "https://release-assets.githubusercontent.com:444/github-production-release-asset/x",
    ],
)
def test_redirects_cannot_escape_approved_https_release_hosts(url):
    assert not vendor.permitted_download(url)
    with pytest.raises(ValueError):
        vendor._AssetRedirect().redirect_request(None, None, 302, "", {}, url)
