"""Read-only Apple sensor contracts without loading platform libraries in unit tests."""

import ctypes as ct
import json
import struct
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock

import pytest

from ypuddin.server import apple_sensors as sensors


@pytest.mark.parametrize("unit,factor", [("J", 1), ("mJ", 1e-3), ("uJ", 1e-6), ("µJ", 1e-6), ("μJ", 1e-6), ("nJ", 1e-9)])
def test_power_is_counter_delta_in_actual_monotonic_interval(unit, factor):
    first = {"GPU Energy": (10**18, unit)}
    second = {"GPU Energy": (10**18 + 25, unit)}
    assert sensors.power_from_snapshots(first, second, 0.5) == pytest.approx(50 * factor)


@pytest.mark.parametrize("first,second,elapsed", [
    ({}, {}, 0.25),
    ({"GPU Energy": (10, "J")}, {}, 0.25),
    ({"GPU Energy": (10, "J")}, {"GPU Energy": (9, "J")}, 0.25),
    ({"GPU Energy": (-1, "J")}, {"GPU Energy": (2, "J")}, 0.25),
    ({"GPU Energy": (True, "J")}, {"GPU Energy": (2, "J")}, 0.25),
    ({"GPU Energy": (1.5, "J")}, {"GPU Energy": (2, "J")}, 0.25),
    ({"GPU Energy": (1, "W")}, {"GPU Energy": (2, "W")}, 0.25),
    ({"GPU Energy": (1, None)}, {"GPU Energy": (2, None)}, 0.25),
    ({"GPU Energy": (1, "J")}, {"GPU Energy": (2, "mJ")}, 0.25),
    ({"GPU Energy": (1, "J")}, {"GPU Energy": (2, "J")}, 0),
    ({"GPU Energy": (1, "J")}, {"GPU Energy": (2, "J")}, -1),
    ({"GPU Energy": (1, "J")}, {"GPU Energy": (2, "J")}, float("nan")),
    ({"GPU Energy": (1, "J")}, {"GPU Energy": (2, "J")}, float("inf")),
])
def test_invalid_or_insufficient_power_snapshots_are_unknown(first, second, elapsed):
    assert sensors.power_from_snapshots(first, second, elapsed) is None


def test_two_valid_unchanged_snapshots_can_report_real_zero_watts():
    sample = {"GPU Energy": (1800, "nJ")}
    assert sensors.power_from_snapshots(sample, sample, 0.25) == 0


def test_gpu_rollup_excludes_sram_subcomponents_and_does_not_double_count_dies():
    rows = [("GPU Energy", "nJ", 200), ("GPU", "mJ", 1), ("GPU SRAM", "mJ", 1),
            ("GPU SRAM Energy", "nJ", 20), ("DIE_0 GPU Energy", "nJ", 100),
            ("DIE_1 GPU Energy", "nJ", 100), ("CPU Energy", "nJ", 1000)]
    assert sensors.select_gpu_rollups(rows) == {"GPU Energy": (200, "nJ")}
    dies = sensors.select_gpu_rollups(rows[1:])
    assert dies == {"DIE_0 GPU Energy": (100, "nJ"), "DIE_1 GPU Energy": (100, "nJ")}
    assert sensors.power_from_snapshots(dies, {key: (150, "nJ") for key in dies}, 0.25) == pytest.approx(4e-7)
    assert sensors.select_gpu_rollups(rows + [("GPU Energy", "nJ", 201)]) == {}
    assert sensors.select_gpu_rollups([("GPU SRAM", "mJ", 5)]) == {}


@pytest.mark.parametrize("chip", ["Apple M4", "Apple M4 GPU", " Apple M4 GPU ", "apple m4 gpu"])
def test_m4_system_display_name_uses_only_known_gpu_thermal_zones(chip):
    assert sensors.gpu_temperature_keys(chip) == ("Tg0G", "Tg0H", "Tg0K", "Tg0L", "Tg0d", "Tg0e", "Tg0j", "Tg0k")


@pytest.mark.parametrize("generation,keys", [
    (1, ("Tg05", "Tg0D", "Tg0L", "Tg0T")),
    (2, ("Tg0f", "Tg0j")),
    (3, ("Tf14", "Tf18", "Tf19", "Tf1A", "Tf24", "Tf28", "Tf29", "Tf2A")),
])
def test_curated_generation_maps_cover_base_and_pro_max_ultra(generation, keys):
    for variant in ("", " Pro", " Max", " Ultra"):
        assert sensors.gpu_temperature_keys(f"Apple M{generation}{variant} GPU") == keys


@pytest.mark.parametrize("chip", ["Apple M4 Pro GPU", "Apple M4 Max", "Apple M4 Ultra"])
def test_m4_larger_variants_do_not_borrow_base_chip_keys(chip):
    assert sensors.gpu_temperature_keys(chip)[:2] == ("Tg1U", "Tg1k")
    assert "Tg0G" not in sensors.gpu_temperature_keys(chip)


@pytest.mark.parametrize("chip", ["Apple M5", "Apple Test GPU", "Intel GPU", "M4", "Apple M4 Pro Max", "Apple M4 GPU 2"])
def test_unknown_chip_does_not_open_smc(chip, monkeypatch):
    native = Mock(side_effect=AssertionError("Unknown chip must not probe arbitrary sensors"))
    monkeypatch.setattr(sensors, "_SMC", native)
    assert sensors._read_temperatures(chip) == []
    native.assert_not_called()


@pytest.mark.parametrize("dtype,raw,expected", [
    ("flt ", struct.pack("<f", 49.5), 49.5),
    ("sp78", bytes.fromhex("3280"), 50.5),
    ("flt ", struct.pack("<f", 0), None),
    ("sp78", bytes.fromhex("ff80"), None),
    ("flt ", struct.pack("<f", float("nan")), None),
    ("flt ", struct.pack("<f", float("inf")), None),
    ("flt ", struct.pack("<f", 255), None),
    ("flt ", b"123", None),
    ("sp78", b"1234", None),
    ("ui32", b"1234", None),
])
def test_temperature_decoder_rejects_malformed_or_non_temperature_values(dtype, raw, expected):
    assert sensors.decode_temperature(dtype, raw) == expected


def test_temperature_average_and_max_only_include_successful_known_keys(monkeypatch):
    reader = Mock()
    keys = sensors.gpu_temperature_keys("Apple M4 GPU")
    reader.read.side_effect = [40.0, None, OSError("absent"), 0.0, float("nan"), 50.0, 60.0, -1]
    resource = Mock()
    resource.__enter__ = Mock(return_value=reader)
    resource.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(sensors, "_SMC", lambda: resource)
    monkeypatch.setattr(sensors.sys, "platform", "darwin")
    monkeypatch.setattr(sensors, "_read_power", Mock(side_effect=OSError("no IOReport")))
    result = sensors.collect("Apple M4 GPU")
    assert result == {"power_w": None, "power_sample_seconds": None,
                      "temp_c": 50, "temp_max_c": 60, "temp_sensor_count": 3}
    assert [call.args[0] for call in reader.read.call_args_list] == list(keys)
    resource.__exit__.assert_called_once()


def _smc_packet(size=4, dtype="flt ", value=42.25):
    packet = bytearray(80)
    struct.pack_into("=II", packet, 28, size, int.from_bytes(dtype.encode(), "big"))
    packet[48:52] = struct.pack("<f", value)
    return packet


@pytest.mark.parametrize("length,result,status,rc", [(79, 0, 0, 0), (81, 0, 0, 0), (80, 132, 0, 0), (80, 0, 1, 0), (80, 0, 0, -1)])
def test_smc_read_rejects_native_struct_error_before_decoding(length, result, status, rc):
    reader = sensors._SMC.__new__(sensors._SMC)
    reader.connection = ct.c_uint32(7)
    packet = _smc_packet()
    packet[40], packet[41] = result, status
    def call(_connection, selector, input_data, input_size, output, output_size):
        assert selector == 2 and input_size == 80 and bytes(input_data)[42] == 9
        ct.memmove(output, bytes(packet), 80)
        ct.cast(output_size, ct.POINTER(ct.c_size_t))[0] = length
        return rc
    reader.call = call
    assert reader.read("Tg0G") is None


@pytest.mark.parametrize("size,dtype", [(0, "flt "), (33, "flt "), (3, "flt "), (4, "ui32"), (4, "sp78")])
def test_smc_never_reads_payload_with_invalid_size_or_type(size, dtype):
    reader = sensors._SMC.__new__(sensors._SMC)
    reader._read_command = Mock(return_value=bytes(_smc_packet(size, dtype)))
    assert reader.read("Tg0G") is None
    reader._read_command.assert_called_once_with("Tg0G", 9)


def test_smc_only_sends_read_commands_and_closes_owned_connection():
    reader = sensors._SMC.__new__(sensors._SMC)
    reader.connection = ct.c_uint32(7)
    calls = []
    def call(_connection, selector, input_data, input_size, output, output_size):
        packet = bytes(input_data)
        calls.append((selector, packet[42], struct.unpack_from("=I", packet, 28)[0]))
        assert input_size == 80
        ct.memmove(output, bytes(_smc_packet()), 80)
        return 0
    reader.call, reader.close = call, Mock()
    assert reader.read("Tg0G") == 42.25
    assert reader._read_command("Tg0G", 6) is None
    assert reader._read_command("F0Tg", 5, 4) is None
    assert calls == [(2, 9, 0), (2, 5, 4)]
    reader.__exit__()
    reader.close.assert_called_once()
    assert reader.connection.value == 0


def test_smc_failed_open_releases_service_and_does_not_acquire_connection():
    reader = sensors._SMC.__new__(sensors._SMC)
    reader.match, reader.first, reader.open = Mock(return_value=1), Mock(return_value=2), Mock(return_value=-1)
    reader.task, reader.connection, reader.release = 3, ct.c_uint32(), Mock()
    with pytest.raises(OSError):
        reader.__enter__()
    reader.release.assert_called_once_with(2)


def test_ioreport_failed_subscription_releases_every_successful_cf_allocation():
    reader = sensors._IOReport.__new__(sensors._IOReport)
    reader.resources = ExitStack()
    reader.release = Mock()
    reader.string = Mock(side_effect=[1, 2])
    reader.channels, reader.dict_copy = Mock(return_value=3), Mock(return_value=4)
    reader.type_id, reader.dict_type = Mock(return_value=10), 10
    def subscribe(_allocator, _channels, actual, _options, _context):
        ct.cast(actual, ct.POINTER(ct.c_void_p))[0] = 5
        return None
    reader.subscribe = subscribe
    with pytest.raises(OSError):
        reader.__enter__()
    assert [call.args[0] for call in reader.release.call_args_list] == [5, 4, 3, 2, 1]


def test_ioreport_releases_sample_even_when_channel_parser_fails():
    reader = sensors._IOReport.__new__(sensors._IOReport)
    reader.subscription, reader.channel_set, reader.key = 1, 2, 3
    reader.sample, reader.release = Mock(return_value=4), Mock()
    reader.type_id, reader.dict_type = Mock(return_value=10), 10
    reader.dict_get = Mock(side_effect=ValueError("invalid native snapshot"))
    with pytest.raises(ValueError):
        reader.snapshot()
    reader.release.assert_called_once_with(4)


def test_power_sampling_uses_actual_snapshot_time_not_requested_sleep(monkeypatch):
    reader = Mock()
    reader.snapshot.side_effect = [({"GPU Energy": (10, "J")}, 1.0), ({"GPU Energy": (12, "J")}, 1.4)]
    resource = Mock()
    resource.__enter__, resource.__exit__ = Mock(return_value=reader), Mock(return_value=False)
    monkeypatch.setattr(sensors, "_IOReport", lambda: resource)
    sleep = Mock()
    monkeypatch.setattr(sensors.time, "sleep", sleep)
    watts, seconds = sensors._read_power(0.25)
    assert watts == pytest.approx(5.0) and seconds == pytest.approx(0.4)
    sleep.assert_called_once_with(0.25)
    resource.__exit__.assert_called_once()


def test_missing_first_snapshot_does_not_invent_initial_zero_or_sleep(monkeypatch):
    reader = Mock()
    reader.snapshot.return_value = ({}, 1.0)
    resource = Mock()
    resource.__enter__, resource.__exit__ = Mock(return_value=reader), Mock(return_value=False)
    monkeypatch.setattr(sensors, "_IOReport", lambda: resource)
    sleep = Mock()
    monkeypatch.setattr(sensors.time, "sleep", sleep)
    assert sensors._read_power(0.25) == (None, None)
    sleep.assert_not_called()
    resource.__exit__.assert_called_once()


def test_missing_smc_symbols_does_not_hide_valid_power(monkeypatch):
    monkeypatch.setattr(sensors.sys, "platform", "darwin")
    monkeypatch.setattr(sensors, "_read_temperatures", Mock(side_effect=AttributeError("missing native symbol")))
    monkeypatch.setattr(sensors, "_read_power", lambda _: (1.25, 0.255))
    assert sensors.collect("Apple M4 GPU") == {"power_w": 1.25, "power_sample_seconds": 0.255,
                                             "temp_c": None, "temp_max_c": None, "temp_sensor_count": 0}


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_other_platforms_never_load_native_libraries(platform, monkeypatch):
    monkeypatch.setattr(sensors.sys, "platform", platform)
    native = Mock(side_effect=AssertionError("Must not load Darwin libraries"))
    monkeypatch.setattr(sensors.ct, "CDLL", native)
    assert sensors.collect("Apple M4") == sensors.empty_reading()
    native.assert_not_called()


def test_script_runs_without_site_packages_project_import_or_working_directory(tmp_path):
    script = Path(sensors.__file__).resolve()
    # Run the real entrypoint using only the stdlib; avoid native sensors on the test host.
    wrapper = (
        "import runpy,sys;sys.platform='linux';"
        f"sys.argv=[{str(script)!r},'--chip','Apple M4 GPU'];"
        f"runpy.run_path({str(script)!r},run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-S", "-c", wrapper], cwd=tmp_path,
                            text=True, capture_output=True, check=True, timeout=5)
    assert json.loads(result.stdout) == sensors.empty_reading()
    assert len(result.stdout.splitlines()) == 1 and result.stderr == ""


def test_cli_passes_exact_system_chip_name_and_emits_single_json(monkeypatch, capsys):
    read = Mock(return_value=sensors.empty_reading())
    monkeypatch.setattr(sensors, "collect", read)
    assert sensors.main(["--chip", "Apple M4 GPU"]) == 0
    read.assert_called_once_with("Apple M4 GPU")
    assert json.loads(capsys.readouterr().out) == sensors.empty_reading()
