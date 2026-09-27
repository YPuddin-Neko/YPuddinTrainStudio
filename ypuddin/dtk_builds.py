"""Reviewed SourceFind builds of the Hygon (DTK) PyTorch stack; standard library only, shared by
bootstrap and the server.

The files were downloaded from the official DAS1.8 directory on 2026-09-27; their METADATA names and
versions were read and their sizes and SHA-256 recorded here. A new DTK environment whose Python has
no vendor PyTorch downloads the set matching the installed DTK release and the Python version.
"""

from __future__ import annotations

import urllib.parse

SOURCE = "https://download.sourcefind.cn:65024"

# (DTK release, CPython tag) -> (package, version, path on SOURCE, size in bytes, SHA-256)
RUNTIME_SETS: dict[tuple[str, str], tuple[tuple[str, str, str, int, str], ...]] = {
    ("26.04", "cp311"): (
        (
            "torch",
            "2.7.1+das.opt1.dtk2604",
            "/file/4/pytorch/DAS1.8/torch-2.7.1+das.opt1.dtk2604-cp311-cp311-manylinux_2_28_x86_64.whl",
            540265849,
            "2bd30ee0aa9f923c45aef99047fc05c0fb911f8e8eba741918b77545620f7974",
        ),
        (
            "torchvision",
            "0.22.0+das.opt1.dtk2604.torch271",
            "/file/4/vision/DAS1.8/torchvision-0.22.0+das.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl",
            2328953,
            "944cba9026e9f330de294610fb0d19054e293cab0c144152339eaec1577cb6b3",
        ),
        (
            "triton",
            "3.1.0+das.opt1.dtk2604.torch271",
            "/file/4/triton/DAS1.8/triton-3.1.0+das.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl",
            128058626,
            "71cc667b28bc326d888959a9695ddded3ea888ca8ad5fe541fefbf2061d4e4b9",
        ),
    ),
}
# The wheels are manylinux_2_28 x86_64 builds.
MIN_GLIBC = (2, 28)


def file_url(path: str) -> str:
    return SOURCE + urllib.parse.quote(path, safe="/")
