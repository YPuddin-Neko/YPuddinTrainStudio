"""Vendor inventory, stream integrity, cancellation and platform matching without installing packages."""

import hashlib
import io
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ypuddin.server import dtk_catalog as vendor


def runtime(**changes):
    return {
        "platform": "Linux",
        "machine": "x86_64",
        "python": "3.11.0rc1",
        "torch": "2.4.1+das.opt1.dtk25041",
        "hip_runtime": "5.6",
        "cuda_available": True,
        **changes,
    }


VERSIONS = {
    "torch": "2.4.1+das.opt1.dtk25041",
    "numpy": "1.26.4",
    "einops": "0.8.1",
    "triton": "3.0.0+das.opt1.dtk25041",
    "flash-attn": "2.6.1+das.opt1.dtk25041",
}


def test_catalog_separates_native_build_matching_and_python_adapter_requirements():
    result = vendor.catalog(runtime(), VERSIONS, "linux-dtk")
    flash, xfs = result.wheels[:2]
    assert flash.compatible and flash.binary and flash.dtk == "25.04.1"
    assert not xfs.compatible and not xfs.binary and xfs.torch == ">=2.5"
    assert xfs.declared_torch == ">=2.1.0" and xfs.reason == "requires_torch_runtime_api>=2.5"
    newer = vendor.catalog(runtime(torch="2.5.1+das.opt1.dtk25041"), VERSIONS, "linux-dtk")
    assert newer.wheels[1].compatible and not newer.wheels[0].compatible
    assert all(w.validation == "kernel_probe_required" for w in result.wheels)
    assert result.runtime["dtk"] == "25.04.1"
    assert len(flash.sha256) == 64 and flash.size_bytes == 389084275
    assert result.reason is None


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"platform": "Windows"}, "requires_linux_x86_64"),
        ({"machine": "aarch64"}, "requires_linux_x86_64"),
        ({"hip_runtime": None}, "hip_runtime_unavailable"),
        ({"torch": "2.5.1+das.opt1.dtk25041"}, "torch_version_mismatch"),
        ({"torch": "2.4.1+das.opt1.dtk25042"}, "dtk_version_mismatch"),
        ({"installed_dtk": "26.04"}, "dtk_version_mismatch"),
        ({"python": "3.12.1"}, "python_abi_mismatch"),
    ],
)
def test_native_flash_wheel_rejects_other_framework_runtime_or_python(changes, reason):
    assert vendor.incompatibility(vendor.WHEELS[0], runtime(**changes), VERSIONS, "linux-dtk") == reason


def test_vendor_triton_and_flash_dependencies_cannot_be_filled_by_generic_pypi_builds():
    versions = {**VERSIONS, "triton": "3.0.0"}
    assert (
        vendor.incompatibility(vendor.WHEELS[0], runtime(), versions, "linux-dtk")
        == "requires_package:triton==3.0.0+das.opt1.dtk25041"
    )
    versions.pop("flash-attn")
    assert (
        vendor.incompatibility(
            vendor.WHEELS[1], runtime(torch="2.5.1+das.opt1.dtk25041"), versions, "linux-dtk"
        )
        == "requires_package:flash-attn>=2.6.1"
    )
    assert vendor.catalog(runtime(), VERSIONS, "linux-cuda").reason == "dtk_profile_required"


def test_new_flash_catalog_matches_only_the_reviewed_torch27_dtk26_bundle():
    wheel = vendor.find_wheel("sourcefind-flash-attn-2.8.3-dtk2604-torch271-cp311")
    info = runtime(torch="2.7.1+das.opt1.dtk2604", installed_dtk="26.04")
    versions = {**VERSIONS, "torch": info["torch"], "triton": "3.1.0+das.opt1.dtk2604.torch271"}
    assert vendor.incompatibility(wheel, info, versions, "linux-dtk") is None
    assert vendor.wheel_for_file(Path(wheel.filename)).id == wheel.id
    assert wheel.size_bytes == 658880343
    assert wheel.sha256 == "d2cdd700de8622b2473bbac57328ca6682eb4025c274a6b7f7f50c687a3c554b"
    assert vendor.incompatibility(wheel, runtime(), versions, "linux-dtk") == "torch_version_mismatch"
    assert (
        vendor.incompatibility(wheel, {**info, "installed_dtk": "25.04.1"}, versions, "linux-dtk")
        == "dtk_version_mismatch"
    )
    assert (
        vendor.incompatibility(wheel, info, VERSIONS, "linux-dtk")
        == "requires_package:triton==3.1.0+das.opt1.dtk2604.torch271"
    )


def test_guidance_matches_distribution_architecture_and_python_without_qualifying_beta_driver():
    current = runtime(
        distribution_id="ubuntu",
        distribution="Ubuntu 22.04.5 LTS",
        distribution_version="22.04",
        driver_version="6.3.31-V1.5.3.beta",
        installed_dtk="25.04.1",
        dtk_root="/opt/dtk-25.04.1",
    )
    result = vendor.catalog(current, VERSIONS, "linux-dtk")
    guidance = result.guidance
    assert guidance.driver_version == "6.3.31-V1.5.3.beta"
    assert guidance.driver_verification == "manual_confirmation_required"
    assert guidance.current_stack_reason == "torch24_transformers5_diffusers040_conflict"
    assert guidance.toolkit_source_url.endswith("/1/main") and guidance.driver_source_url.endswith("/6/main")
    candidate = guidance.recommendation
    assert candidate.dtk == "26.04" and candidate.python_tag == "cp311"
    assert candidate.minimum_driver == "6.3.30-V1.4.1a"
    assert candidate.status == "candidate_requires_validation"
    assert candidate.toolkit_checksum_url == candidate.toolkit_url + ".md5"
    assert {item.package for item in candidate.wheels} == {"torch", "torchvision", "triton", "flash-attn"}
    assert result.runtime["installed_dtk"] == "25.04.1" and result.runtime["dtk"] == "25.04.1"


@pytest.mark.parametrize(
    "changes",
    [
        {"distribution_id": "debian"},
        {"distribution_version": "24.04"},
        {"machine": "aarch64"},
        {"python": "3.12.1"},
        {"platform": "Windows"},
    ],
)
def test_guidance_does_not_recommend_ubuntu_cp311_bundle_for_another_system(changes):
    info = runtime(**{"distribution_id": "ubuntu", "distribution_version": "22.04", **changes})
    assert vendor.guidance(info).recommendation is None


def test_system_identity_keeps_installed_toolkit_separate_from_torch_build(tmp_path, monkeypatch):
    toolkit = tmp_path / "dtk-26.04"
    toolkit.mkdir()
    monkeypatch.setenv("DTK_ROOT", str(toolkit))
    monkeypatch.setattr(vendor.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        vendor.platform,
        "freedesktop_os_release",
        lambda: {"ID": "ubuntu", "PRETTY_NAME": "Ubuntu 22.04.5 LTS", "VERSION_ID": "22.04"},
    )
    monkeypatch.setattr(vendor.platform, "libc_ver", lambda: ("glibc", "2.35"))
    info = vendor.system_info()
    assert info["installed_dtk"] == "26.04" and info["dtk_root"] == str(toolkit)
    assert info["distribution_version"] == "22.04" and info["glibc_version"] == "2.35"
    assert vendor.dtk_build_version(runtime()) == "25.04.1"


@pytest.mark.parametrize("failure", [False, True])
def test_driver_identity_parses_vendor_text_with_bounded_read_only_command(tmp_path, monkeypatch, failure):
    tool = tmp_path / "bin/rocm-smi"
    tool.parent.mkdir()
    tool.touch()
    monkeypatch.setenv("DTK_ROOT", str(tmp_path))
    monkeypatch.setattr(vendor.platform, "system", lambda: "Linux")

    def run(command, **kwargs):
        assert command == [str(tool), "--showdriverversion"]
        assert kwargs["timeout"] == 3 and "shell" not in kwargs
        if failure:
            raise subprocess.TimeoutExpired(command, 3)
        return SimpleNamespace(
            returncode=0, stdout="Warning: unsupported format\nDriver Version: 6.3.31-V1.5.3.beta\n"
        )

    monkeypatch.setattr(vendor.subprocess, "run", run)
    assert vendor.driver_version() == (None if failure else "6.3.31-V1.5.3.beta")


class Response(io.BytesIO):
    def __init__(self, value, *, url=vendor.WHEELS[1].url, length=None):
        super().__init__(value)
        self.headers = {"Content-Length": str(len(value) if length is None else length)}
        self.url = url

    def geturl(self):
        return self.url


def entry(value):
    return vendor.WHEELS[1].model_copy(
        update={"size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    )


def test_stream_reports_measured_speed_and_clears_it_at_completion(tmp_path, monkeypatch):
    data = b"reviewed wheel bytes" * 130000
    record = entry(data)
    clock = iter([0.0, 0.5, 1.0, 1.5])
    monkeypatch.setattr(vendor.time, "monotonic", lambda: next(clock))
    updates = []
    target = vendor.download(
        record,
        tmp_path,
        threading.Event(),
        lambda *value: updates.append(value),
        opener=lambda *a, **k: Response(data),
    )
    assert target.read_bytes() == data
    assert updates[0] == (0, len(data), None, None)
    assert any(speed and speed > 0 and eta and eta > 0 for _, _, speed, eta in updates[1:])
    assert updates[-1] == (len(data), len(data), None, 0)
    assert not list(tmp_path.glob("*.partial"))


def test_vendor_download_uses_selected_proxy_without_removing_origin_guard(tmp_path, monkeypatch):
    from ypuddin.server.network import ProxyPolicy

    data = b"verified official content"
    record = entry(data)
    captured = []

    def opener(self, *handlers):
        captured.append((self.mode, handlers))
        return SimpleNamespace(open=lambda *a, **k: Response(data))

    monkeypatch.setattr(ProxyPolicy, "opener", opener)
    result = vendor.download(
        record, tmp_path, threading.Event(), lambda *args: None,
        proxy=ProxyPolicy("custom", "http://localhost:7890"),
    )
    assert result.read_bytes() == data
    assert captured[0][0] == "custom"
    assert isinstance(captured[0][1][0], vendor._VendorRedirect)


def test_vendor_proxy_error_does_not_expose_password(tmp_path, monkeypatch):
    from ypuddin.server.network import ProxyPolicy

    def failure(*args, **kwargs):
        raise OSError("proxy password private-password rejected http://a:private-password@host")

    monkeypatch.setattr(ProxyPolicy, "opener", lambda *_args: SimpleNamespace(open=failure))
    with pytest.raises(ValueError) as exc:
        vendor.download(
            entry(b"reviewed"), tmp_path, threading.Event(), lambda *args: None,
            proxy=ProxyPolicy("custom", "http://localhost:7890", "a", "private-password"),
        )
    assert "private-password" not in str(exc.value)
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("mode", ["digest", "truncated", "oversized", "redirect"])
def test_untrusted_or_incomplete_download_never_becomes_installable(tmp_path, mode):
    data = b"reviewed vendor wheel bytes"
    record = entry(data)
    received = (
        data + b"x"
        if mode == "oversized"
        else data[:-1]
        if mode == "truncated"
        else data[::-1]
        if mode == "digest"
        else data
    )
    url = "https://untrusted.invalid/file.whl" if mode == "redirect" else record.url
    with pytest.raises(ValueError):
        vendor.download(
            record,
            tmp_path,
            threading.Event(),
            lambda *a: None,
            opener=lambda *a, **k: Response(received, url=url, length=len(data)),
        )
    assert not list(tmp_path.glob("*.whl")) and not list(tmp_path.glob("*.partial"))


def test_download_cancellation_removes_partial_bytes(tmp_path):
    data = b"x" * (2 * 1024**2)
    cancel = threading.Event()

    class CancelResponse(Response):
        def read(self, size):
            result = super().read(size)
            cancel.set()
            return result

    with pytest.raises(InterruptedError):
        vendor.download(
            entry(data), tmp_path, cancel, lambda *a: None, opener=lambda *a, **k: CancelResponse(data)
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "url",
    [
        "http://download.sourcefind.cn:65024/file/4/x.whl",
        "https://download.sourcefind.cn/file/4/x.whl",
        "https://download.sourcefind.cn:65024@localhost/file/4/x.whl",
        "https://download.sourcefind.cn:65024/api/private",
    ],
)
def test_redirect_is_rejected_before_following_unapproved_origins(url):
    with pytest.raises(ValueError, match="approved origin"):
        vendor._VendorRedirect().redirect_request(None, None, 302, "redirect", {}, url)
