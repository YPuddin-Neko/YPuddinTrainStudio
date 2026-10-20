"""Header validation and identity checks for Comfy-Org's MOSS-Image Base BF16 file."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import RLock

BASE_BF16_SHA256 = "cacd158728a1fb9e57c408a922b54752ec71fceb566b3821269927964e62b41f"
BASE_BF16_SIZE = 10164655808
BASE_CONFIG = dict(dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096,
                   freq_dim=256, latent_channels=128, rope_base=10000.0)

_AUTHENTICATED: OrderedDict[tuple, None] = OrderedDict()
_AUTH_LOCK = RLock()
_AUTH_SCOPE: ContextVar[tuple | None] = ContextVar("moss_fused_authentication", default=None)
_MAX_HEADER = 2 * 1024 * 1024
_QKV = re.compile(r"layers\.[0-9]+\.proj\.qkv\.weight")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"MOSS-Image 权重头包含重复字段：{key}")
        result[key] = value
    return result


def _signature(info):
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev)


def _authentication_key(path, info):
    return (str(path.resolve()), _signature(info), BASE_BF16_SHA256, BASE_BF16_SIZE)


@contextmanager
def authenticated_fused_singlefile(path):
    """Pin a strictly authenticated identity for short registration without rehashing."""
    path = Path(path)
    key = _authentication_key(path, path.stat())
    validate_fused_singlefile(path, authenticate=True)
    if key != _authentication_key(path, path.stat()):
        raise ValueError("MOSS-Image 权重文件在校验期间发生变化，请重试。")
    token = _AUTH_SCOPE.set(key)
    try:
        yield
    finally:
        _AUTH_SCOPE.reset(token)


def _read_header(stream):
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError("MOSS-Image safetensors 权重头不完整。")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= _MAX_HEADER:
        raise ValueError("MOSS-Image safetensors 权重头长度无效。")
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError("MOSS-Image safetensors 权重头不完整。")
    try:
        if not raw.startswith(b"{"):
            raise ValueError("MOSS-Image safetensors 权重头需要以 JSON 对象开头。")
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("MOSS-Image safetensors 权重头无效。") from error
    if not isinstance(header, dict):
        raise ValueError("MOSS-Image safetensors 权重头需要 JSON 对象。")
    return header, 8 + length


def _expected_tensors():
    c = BASE_CONFIG
    dim, ffn, head = c["dim"], c["ffn_dim"], c["dim"] // c["num_heads"]
    expected = {"rope.bands": ("F32", [head // 8])}

    def linear(prefix, rows, cols, bias=True):
        expected[f"{prefix}.weight"] = ("BF16", [rows, cols])
        if bias:
            expected[f"{prefix}.bias"] = ("BF16", [rows])

    linear("video_embedder", dim, c["latent_channels"])
    linear("text_embedder.proj", dim, c["text_dim"])
    linear("time_embedding.0", dim, c["freq_dim"])
    linear("time_embedding.2", dim, dim)
    linear("shared_block_modulation", 6 * dim, dim)
    linear("video_head_modulation", 2 * dim, dim)
    linear("final_linear_video", c["latent_channels"], dim, False)
    expected["final_norm_video.weight"] = ("BF16", [dim])
    for prefix in [*(f"text_embedder.blocks.{i}" for i in range(2)),
                   *(f"layers.{i}" for i in range(c["num_layers"]))]:
        for name, size in (("norm1", dim), ("norm2", dim), ("q_norm", head), ("k_norm", head)):
            expected[f"{prefix}.{name}.weight"] = ("BF16", [size])
        if prefix.startswith("text_embedder."):
            linear(f"{prefix}.qkv", 3 * dim, dim)
            linear(f"{prefix}.proj", dim, dim)
            linear(f"{prefix}.ffn_gate", ffn, dim)
            linear(f"{prefix}.ffn_up", ffn, dim)
            linear(f"{prefix}.ffn_down", dim, ffn)
        else:
            linear(f"{prefix}.proj.qkv", 3 * dim, dim, False)
            linear(f"{prefix}.proj.o", dim, dim, False)
            linear(f"{prefix}.proj.ffn_up_gate", 2 * ffn, dim, False)
            linear(f"{prefix}.proj.ffn_down", dim, ffn, False)
    return expected


def _validate_header(header, payload_start, file_size):
    expected = _expected_tensors()
    if set(header) != set(expected):
        missing, extra = set(expected) - set(header), set(header) - set(expected)
        raise ValueError(f"MOSS-Image BF16 单文件张量不符：missing={sorted(missing)[:3]}, unexpected={sorted(extra)[:3]}")
    ranges = []
    for key, (dtype, shape) in expected.items():
        entry = header[key]
        if (not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}
                or entry["dtype"] != dtype or not isinstance(entry["shape"], list)
                or any(type(n) is not int for n in entry["shape"]) or entry["shape"] != shape):
            raise ValueError(f"MOSS-Image BF16 单文件精度或形状不符：{key}")
        offsets = entry["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
                or offsets[1] - offsets[0] != math.prod(shape) * (4 if dtype == "F32" else 2)):
            raise ValueError(f"MOSS-Image BF16 单文件张量长度不符：{key}")
        ranges.append(tuple(offsets))
    end = 0
    for start, following in sorted(ranges):
        if start != end:
            raise ValueError("MOSS-Image BF16 单文件张量区间重叠或不连续。")
        end = following
    if payload_start + end != file_size:
        raise ValueError("MOSS-Image BF16 单文件长度与权重头不符。")


def is_fused_singlefile(path) -> bool:
    """Recognize the main-block fused projection signature; this does not authenticate weights."""
    path = Path(path)
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            header, _ = _read_header(stream)
    except (OSError, ValueError):
        return False
    return any(_QKV.fullmatch(key) for key in header)


def validate_fused_singlefile(path, *, authenticate=True) -> dict:
    """Validate the full Base layout; authenticate=True also checks its fixed payload SHA."""
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("MOSS-Image BF16 单文件需要 safetensors 格式。")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        signature = _signature(before)
        if not stat.S_ISREG(before.st_mode) or _signature(path.stat()) != signature:
            raise ValueError("MOSS-Image 权重文件已变化，请重新选择。")
        header, payload_start = _read_header(stream)
        _validate_header(header, payload_start, before.st_size)
        if authenticate:
            if before.st_size != BASE_BF16_SIZE:
                raise ValueError("请选择官方 MOSS-Image Base BF16 单文件，文件大小不符。")
            key = _authentication_key(path, before)
            scoped = _AUTH_SCOPE.get()
            if scoped is not None and key != scoped:
                raise ValueError("MOSS-Image 权重文件在校验期间发生变化，请重试。")
            # Share successful checks only while the file identity remains unchanged.
            if scoped is None:
                with _AUTH_LOCK:
                    cached = key in _AUTHENTICATED
                if not cached:
                    stream.seek(0)
                    digest = hashlib.sha256()
                    while chunk := stream.read(8 * 1024 * 1024):
                        digest.update(chunk)
                    if digest.hexdigest() != BASE_BF16_SHA256:
                        raise ValueError("MOSS-Image 单文件与官方 Base BF16 SHA256 不符。")
                if _signature(os.fstat(stream.fileno())) != signature or _signature(path.stat()) != signature:
                    raise ValueError("MOSS-Image 权重文件在校验期间发生变化，请重试。")
                with _AUTH_LOCK:
                    _AUTHENTICATED[key] = None
                    _AUTHENTICATED.move_to_end(key)
                    while len(_AUTHENTICATED) > 8:
                        _AUTHENTICATED.popitem(last=False)
        if _signature(os.fstat(stream.fileno())) != signature or _signature(path.stat()) != signature:
            raise ValueError("MOSS-Image 权重文件在校验期间发生变化，请重试。")
    return dict(BASE_CONFIG)
