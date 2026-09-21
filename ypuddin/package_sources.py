"""Standard-library-only package source policy, shared by bootstrap and the server."""

PYPI = {
    "ustc": "https://mirrors.ustc.edu.cn/pypi/simple",
    "tuna": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple",
    "official": "https://pypi.org/simple",
}


def pypi_sources(source="ustc", fallback=True):
    primary = PYPI[source]
    return [primary] + ([url for url in PYPI.values() if url != primary] if fallback else [])


def torch_sources(backend, source="mirror", fallback=True):
    official = ("index-url", f"https://download.pytorch.org/whl/{backend}")
    mirrors = [
        ("index-url", f"https://mirror.sjtu.edu.cn/pytorch-wheels/{backend}"),
        ("find-links", f"https://mirrors.aliyun.com/pytorch-wheels/{backend}"),
    ]
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
