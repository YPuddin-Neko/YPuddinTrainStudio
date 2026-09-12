"""Read-only Apple GPU sensors, isolated from the server and training runtime.

Run this file directly with ``python apple_sensors.py --chip 'Apple M4'``.
Only standard-library modules are imported. Private Apple APIs vary by OS/chip;
the caller must impose a subprocess timeout and handle abnormal termination.

Sensor identities were checked against the primary Stats sensor table:
https://github.com/exelban/stats/blob/master/Modules/Sensors/values.swift
These are GPU thermal zones, not individual GPU cores. No SMC writes exist here.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import math
import re
import struct
import subprocess
import sys
import time
from contextlib import ExitStack

_PTR = ct.c_void_p
_U32 = ct.c_uint32
_SIZE = ct.c_size_t
_UTF8 = 0x08000100
_IOKIT = "/System/Library/Frameworks/IOKit.framework/IOKit"
_CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_GPU_KEYS = {
    1: ("Tg05", "Tg0D", "Tg0L", "Tg0T"),
    2: ("Tg0f", "Tg0j"),
    3: ("Tf14", "Tf18", "Tf19", "Tf1A", "Tf24", "Tf28", "Tf29", "Tf2A"),
}
_M4_COMMON = ("Tg0K", "Tg0L", "Tg0d", "Tg0e", "Tg0j", "Tg0k")
_KNOWN_KEYS = frozenset(key for keys in _GPU_KEYS.values() for key in keys) | frozenset(
    ("Tg0G", "Tg0H", "Tg1U", "Tg1k", *_M4_COMMON)
)
_ENERGY_UNITS = {"j": 1.0, "mj": 1e-3, "uj": 1e-6, "µj": 1e-6, "μj": 1e-6, "nj": 1e-9}
_DIE_ROLLUP = re.compile(r"DIE[_ ]\d+ GPU Energy")


def empty_reading() -> dict:
    return {
        "power_w": None,
        "temp_c": None,
        "temp_max_c": None,
        "temp_sensor_count": 0,
        "power_sample_seconds": None,
    }


def gpu_temperature_keys(chip: str) -> tuple[str, ...]:
    match = re.fullmatch(r"Apple\s+M([1-4])(?:\s+(Pro|Max|Ultra))?(?:\s+GPU)?", chip.strip(), re.IGNORECASE)
    if not match:
        return ()
    generation = int(match[1])
    if generation == 4:
        return (("Tg0G", "Tg0H") if match[2] is None else ("Tg1U", "Tg1k")) + _M4_COMMON
    return _GPU_KEYS[generation]


def decode_temperature(dtype: str, raw: bytes) -> float | None:
    if dtype == "flt " and len(raw) == 4:
        value = struct.unpack("<f", raw)[0]
    elif dtype == "sp78" and len(raw) == 2:
        value = int.from_bytes(raw, "big", signed=True) / 256.0
    else:
        return None
    # Zero/negative and implausible sensor sentinel values are unavailable.
    return value if math.isfinite(value) and 0 < value <= 150 else None


def power_from_snapshots(first: dict, second: dict, elapsed: float) -> float | None:
    """Convert matching cumulative GPU energy rollups to watts; never add SRAM again."""
    if not math.isfinite(elapsed) or elapsed <= 0 or not first or first.keys() != second.keys():
        return None
    deltas = []
    for name, (before, first_unit) in first.items():
        after, second_unit = second[name]
        factor = _ENERGY_UNITS.get(first_unit.strip().lower()) if isinstance(first_unit, str) else None
        other = _ENERGY_UNITS.get(second_unit.strip().lower()) if isinstance(second_unit, str) else None
        if factor is None or factor != other:
            return None
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (before, after)):
            return None
        if after < before:
            return None  # A wrap/reset invalidates this entire sampling interval.
        # Subtract integers before float conversion, preserving small deltas of large counters.
        deltas.append((after - before) * factor)
    watts = math.fsum(deltas) / elapsed
    return watts if math.isfinite(watts) and watts >= 0 else None


def select_gpu_rollups(rows: list[tuple[str, str | None, int]]) -> dict:
    candidates: dict[str, tuple[int, str | None]] = {}
    duplicate = set()
    for name, unit, value in rows:
        if name != "GPU Energy" and not _DIE_ROLLUP.fullmatch(name):
            continue
        if name in candidates:
            duplicate.add(name)
        candidates[name] = (value, unit)
    # A global rollup already includes the dies. Never add global and per-die energy.
    names = {"GPU Energy"} if "GPU Energy" in candidates else set(candidates)
    if names & duplicate:
        return {}
    return {name: candidates[name] for name in names}


def _bind(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype, function.argtypes = result, list(arguments)
    return function


class _SMC:
    def __init__(self):
        io = ct.CDLL(_IOKIT)
        system = ct.CDLL("/usr/lib/libSystem.B.dylib")
        self.task = _U32.in_dll(system, "mach_task_self_").value
        self.match = _bind(io, "IOServiceMatching", _PTR, ct.c_char_p)
        self.first = _bind(io, "IOServiceGetMatchingService", _U32, _U32, _PTR)
        self.open = _bind(io, "IOServiceOpen", ct.c_int, _U32, _U32, _U32, ct.POINTER(_U32))
        self.release = _bind(io, "IOObjectRelease", ct.c_int, _U32)
        self.close = _bind(io, "IOServiceClose", ct.c_int, _U32)
        self.call = _bind(
            io, "IOConnectCallStructMethod", ct.c_int,
            _U32, _U32, _PTR, _SIZE, _PTR, ct.POINTER(_SIZE),
        )
        self.connection = _U32()

    def __enter__(self):
        service = self.first(0, self.match(b"AppleSMC"))
        if not service:
            raise OSError("AppleSMC unavailable")
        try:
            if self.open(service, self.task, 0, ct.byref(self.connection)) != 0:
                raise OSError("AppleSMC read connection unavailable")
        finally:
            self.release(service)
        return self

    def __exit__(self, *_):
        if self.connection.value:
            self.close(self.connection)
            self.connection = _U32()

    def _read_command(self, key: str, command: int, size: int = 0) -> bytes | None:
        if key not in _KNOWN_KEYS or command not in (5, 9) or not 0 <= size <= 32:
            return None
        # AppleSMC's 80-byte request ABI: key@0, keyInfo@28, command@42,
        # result/status@40/41 and the 32-byte data area@48. Only read commands.
        packet = bytearray(80)
        struct.pack_into("=I", packet, 0, int.from_bytes(key.encode("ascii"), "big"))
        struct.pack_into("=I", packet, 28, size)
        packet[42] = command
        output = ct.create_string_buffer(80)
        length = _SIZE(80)
        buffer = (ct.c_ubyte * 80).from_buffer(packet)
        rc = self.call(self.connection, 2, buffer, 80, output, ct.byref(length))
        if rc != 0 or length.value != 80 or output.raw[40] != 0 or output.raw[41] != 0:
            return None
        return output.raw

    def read(self, key: str) -> float | None:
        info = self._read_command(key, 9)
        if info is None:
            return None
        size, code = struct.unpack_from("=II", info, 28)
        dtype = code.to_bytes(4, "big").decode("ascii", errors="replace")
        if (dtype, size) not in (("flt ", 4), ("sp78", 2)):
            return None
        data = self._read_command(key, 5, size)
        return decode_temperature(dtype, data[48:48 + size]) if data is not None else None


def _read_temperatures(chip: str) -> list[float]:
    keys = gpu_temperature_keys(chip)
    if not keys:
        return []
    values = []
    with _SMC() as smc:
        for key in keys:
            try:
                value = smc.read(key)
            except Exception:
                continue
            if value is not None and math.isfinite(value) and 0 < value <= 150:
                values.append(value)
    return values


class _IOReport:
    def __init__(self):
        cf = ct.CDLL(_CORE_FOUNDATION)
        io = ct.CDLL("/usr/lib/libIOReport.dylib")
        self.release = _bind(cf, "CFRelease", None, _PTR)
        self.string = _bind(cf, "CFStringCreateWithCString", _PTR, _PTR, ct.c_char_p, _U32)
        self.text = _bind(cf, "CFStringGetCString", ct.c_bool, _PTR, _PTR, ct.c_long, _U32)
        self.type_id = _bind(cf, "CFGetTypeID", ct.c_ulong, _PTR)
        self.dict_type = _bind(cf, "CFDictionaryGetTypeID", ct.c_ulong)()
        self.array_type = _bind(cf, "CFArrayGetTypeID", ct.c_ulong)()
        self.string_type = _bind(cf, "CFStringGetTypeID", ct.c_ulong)()
        self.dict_get = _bind(cf, "CFDictionaryGetValue", _PTR, _PTR, _PTR)
        self.dict_copy = _bind(cf, "CFDictionaryCreateMutableCopy", _PTR, _PTR, ct.c_long, _PTR)
        self.array_count = _bind(cf, "CFArrayGetCount", ct.c_long, _PTR)
        self.array_at = _bind(cf, "CFArrayGetValueAtIndex", _PTR, _PTR, ct.c_long)
        self.channels = _bind(
            io, "IOReportCopyChannelsInGroup", _PTR, _PTR, _PTR, ct.c_uint64, ct.c_uint64, ct.c_uint64,
        )
        self.subscribe = _bind(
            io, "IOReportCreateSubscription", _PTR, _PTR, _PTR, ct.POINTER(_PTR), ct.c_uint64, _PTR,
        )
        self.sample = _bind(io, "IOReportCreateSamples", _PTR, _PTR, _PTR, _PTR)
        self.name = _bind(io, "IOReportChannelGetChannelName", _PTR, _PTR)
        self.unit = _bind(io, "IOReportChannelGetUnitLabel", _PTR, _PTR)
        self.value = _bind(io, "IOReportSimpleGetIntegerValue", ct.c_int64, _PTR, ct.c_int32)
        self.resources = ExitStack()

    def _own(self, pointer):
        if not pointer:
            raise OSError("IOReport resource unavailable")
        self.resources.callback(self.release, pointer)
        return pointer

    def __enter__(self):
        try:
            group = self._own(self.string(None, b"Energy Model", _UTF8))
            self.key = self._own(self.string(None, b"IOReportChannels", _UTF8))
            raw = self._own(self.channels(group, None, 0, 0, 0))
            if self.type_id(raw) != self.dict_type:
                raise ValueError("Invalid IOReport channels")
            self.channel_set = self._own(self.dict_copy(None, 0, raw))
            actual = _PTR()
            subscription = self.subscribe(None, self.channel_set, ct.byref(actual), 0, None)
            if actual.value:
                self._own(actual.value)
            self.subscription = self._own(subscription)
            return self
        except Exception:
            self.resources.close()
            raise

    def __exit__(self, *_):
        self.resources.close()

    def _text(self, pointer) -> str | None:
        if not pointer or self.type_id(pointer) != self.string_type:
            return None
        buffer = ct.create_string_buffer(1024)
        return buffer.value.decode("utf-8") if self.text(pointer, buffer, len(buffer), _UTF8) else None

    def snapshot(self) -> tuple[dict, float]:
        sample = self.sample(self.subscription, self.channel_set, None)
        timestamp = time.monotonic()
        if not sample:
            return {}, timestamp
        try:
            if self.type_id(sample) != self.dict_type:
                return {}, timestamp
            array = self.dict_get(sample, self.key)
            if not array or self.type_id(array) != self.array_type:
                return {}, timestamp
            count = self.array_count(array)
            if not 0 < count <= 4096:
                return {}, timestamp
            rows = []
            for index in range(count):
                channel = self.array_at(array, index)
                if not channel or self.type_id(channel) != self.dict_type:
                    continue
                name = self._text(self.name(channel))
                if name == "GPU Energy" or (name and _DIE_ROLLUP.fullmatch(name)):
                    rows.append((name, self._text(self.unit(channel)), self.value(channel, 0)))
            return select_gpu_rollups(rows), timestamp
        finally:
            self.release(sample)


def _read_power(sample_seconds: float) -> tuple[float | None, float | None]:
    with _IOReport() as reader:
        first, before = reader.snapshot()
        if not first:
            return None, None
        time.sleep(sample_seconds)
        second, after = reader.snapshot()
        elapsed = after - before
        watts = power_from_snapshots(first, second, elapsed)
        return watts, elapsed if watts is not None else None


def collect(chip: str, sample_seconds: float = 0.25) -> dict:
    result = empty_reading()
    if sys.platform != "darwin":
        return result
    try:
        values = _read_temperatures(chip)
        if values:
            result.update(temp_c=math.fsum(values) / len(values), temp_max_c=max(values), temp_sensor_count=len(values))
    except Exception:
        pass
    try:
        if math.isfinite(sample_seconds) and 0 < sample_seconds <= 2:
            result["power_w"], result["power_sample_seconds"] = _read_power(sample_seconds)
    except Exception:
        pass
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chip_name", nargs="?")
    parser.add_argument("--chip")
    args = parser.parse_args(argv)
    chip = args.chip or args.chip_name
    if not chip and sys.platform == "darwin":
        try:
            chip = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], timeout=1, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass
    print(json.dumps(collect(chip or ""), allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
