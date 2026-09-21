import threading

import pytest

from ypuddin.package_sources import pypi_sources, run_sources, torch_sources
from ypuddin.server.models import SettingsDownloads


def test_defaults_and_explicit_source_without_fallback():
    assert SettingsDownloads().pypi == "ustc"
    assert pypi_sources()[0] == "https://mirrors.ustc.edu.cn/pypi/simple"
    assert pypi_sources("official", False) == ["https://pypi.org/simple"]
    assert pypi_sources("tuna")[0] == "https://pypi.tuna.tsinghua.edu.cn/simple"
    with pytest.raises(ValueError):
        SettingsDownloads(pypi="unknown")


def test_cuda_sources_never_use_general_pypi():
    sources = torch_sources("cu130")
    assert sources[0] == ("index-url", "https://mirror.sjtu.edu.cn/pytorch-wheels/cu130")
    assert sources[-1][1] == "https://download.pytorch.org/whl/cu130"
    assert "download.pytorch.org" not in str(torch_sources("cu130", fallback=False))
    assert torch_sources("cu130", "official", False) == [sources[-1]]


def test_failure_falls_back_but_cancellation_does_not():
    class Installer:
        calls = []
        error = RuntimeError("unavailable")

        def run(self, command, *args, **kwargs):
            self.calls.append(command)
            if len(self.calls) == 1:
                raise self.error

    installer = Installer()
    commands = [("mirror", ["mirror"]), ("official", ["official"])]
    run_sources(installer, commands, lambda _: None, threading.Event())
    assert installer.calls == [["mirror"], ["official"]]
    installer.calls = []
    installer.error = InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        run_sources(installer, commands, lambda _: None, threading.Event())
    assert installer.calls == [["mirror"]]
