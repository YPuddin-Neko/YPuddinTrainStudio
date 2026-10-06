"""How many layers each model family's presets train, measured outside the service process.

Counting them builds every backbone on the meta device. That imports diffusers, transformers and
torch._dynamo, and importing diffusers asks torch whether a GPU is available, which starts the CUDA or
HIP runtime in that process (on Windows the CUDA context stays for the life of the process). A
short-lived child process with every GPU hidden takes the measurement instead. Its result is stored
in the database under a fingerprint of the code and libraries it was measured with, so later starts
read it back without a child.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KV_KEY = "families.geometry"
MARKER = "YPUDDIN_FAMILY_GEOMETRY "
TIMEOUT = 180.0  # seconds a measurement may take before the child is stopped
RETRY_SECONDS = 300.0  # after a failed measurement, requests answer at once until a retry is due
# The child builds models on the meta device only; no GPU is needed or touched.
HIDDEN_GPUS = {"CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1"}
_PACKAGE = Path(__file__).resolve().parents[1]
_MODULE_SOURCE = Path(__file__).resolve()
# Some GPU extensions query the current device when imported, even for meta models.
_GPU_EXTENSIONS = (
    "flash_attn",
    "flash_attn_interface",
    "flash_attn_3",
    "flash_attn_2_cuda",
    "flash_attn_3_cuda",
    "sageattention",
    "xformers",
)
# The code that decides the layers: model definitions and their configs, the adapter rules, the config
# defaults, plus the libraries whose model classes the families build.
_SOURCES = ("models", "adapters", "config")
_CONFIG_BYTES = 64 * 1024  # JSON configs count; tokenizer vocabularies (megabytes) cannot change layers
_LIBRARIES = ("torch", "diffusers", "transformers")

_lock = threading.Lock()
_measured: dict[str, dict[str, Any]] = {}  # family -> geometry; the same for every service in a process


def measure(name: str) -> dict[str, Any]:
    """Layer counts of one family's presets on its official geometry, built on the meta device here.

    A family whose backbone cannot be built counts no layers, and ``failure`` says why.
    """
    from ypuddin.adapters.rules import resolve_targets
    from ypuddin.config import AdapterConfig
    from ypuddin.models import get_family

    family = get_family(name)
    failure = None
    try:
        modules = family.adaptable_modules()
    except Exception as error:  # noqa: BLE001
        modules, failure = {}, f"{type(error).__name__}: {error}"
    probe = AdapterConfig(algo="lora", rank=4, alpha=4)
    probe_conv = AdapterConfig(algo="lora", rank=4, alpha=4, layer_types="linear_conv")
    presets = {}
    for preset_name, preset in family.presets().items():
        with_conv = resolve_targets(modules, probe_conv, preset) if preset.conv else []
        presets[preset_name] = {
            "layers": len(resolve_targets(modules, probe, preset)),
            "layers_with_conv": len(with_conv),
            "conv_layers": sum(bool(modules[target.name]) for target in with_conv),
        }
    result: dict[str, Any] = {
        "presets": presets,
        "linear_modules": sum(not kernel for kernel in modules.values()),
    }
    if failure:
        result["failure"] = failure
    return result


def unmeasured(name: str, reason: str) -> dict[str, Any]:
    """Zero counts while no measurement is available; ``transient`` asks callers not to keep them."""
    from ypuddin.models import get_family

    zero = {"layers": 0, "layers_with_conv": 0, "conv_layers": 0}
    presets = {preset: dict(zero) for preset in get_family(name).presets()}
    return {"presets": presets, "linear_modules": 0, "failure": reason, "transient": True}


def fingerprint() -> str:
    """Changes whenever the code or a library that decides the layer counts changes."""
    digest = hashlib.sha256()
    digest.update(_MODULE_SOURCE.read_bytes() + b"\0")
    for folder in _SOURCES:
        for path in sorted((_PACKAGE / folder).rglob("*")):
            if path.suffix not in {".py", ".json"} or "__pycache__" in path.parts or not path.is_file():
                continue
            if path.suffix == ".json" and path.stat().st_size > _CONFIG_BYTES:
                continue
            digest.update(path.relative_to(_PACKAGE).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    for library in _LIBRARIES:
        try:
            version = metadata.version(library)
        except metadata.PackageNotFoundError:
            version = ""
        digest.update(f"{library}={version}\0".encode())
    return digest.hexdigest()


def _valid(geometry: Any) -> bool:
    return (
        isinstance(geometry, dict)
        and "failure" not in geometry
        and isinstance(geometry.get("linear_modules"), int)
        and isinstance(geometry.get("presets"), dict)
        and all(
            isinstance(counts, dict)
            and all(isinstance(counts.get(key), int) for key in ("layers", "layers_with_conv", "conv_layers"))
            for counts in geometry["presets"].values()
        )
    )


class FamilyGeometry:
    """Measures every family once per code version, in a child process, and keeps the result."""

    def __init__(self, db: Any, *, timeout: float = TIMEOUT) -> None:
        self.db = db
        self.timeout = timeout
        self._lock = threading.Lock()
        self._running: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._failed_at: float | None = None
        self._failure = ""
        self._family_failures: dict[str, str] = {}
        self._closed = False

    def start(self) -> threading.Event:
        """Begin measuring in the background unless every family is known; the event is set when done."""
        from ypuddin.models import available

        with self._lock:
            if self._running is not None:
                return self._running
            done = threading.Event()
            retry_due = self._failed_at is None or time.monotonic() - self._failed_at >= RETRY_SECONDS
            if self._closed or not retry_due or all(name in _measured for name in available()):
                done.set()
                return done
            self._running = done
            self._thread = threading.Thread(
                target=self._measure, args=(done,), name="family-geometry", daemon=True
            )
            self._thread.start()
            return done

    def layers(self, name: str) -> dict[str, Any]:
        """Wait for the initial measurement; retry failed measurements in the background."""
        if (known := _measured.get(name)) is not None:
            return known
        retry = self._failed_at is not None
        done = self.start()
        if not retry:
            done.wait(self.timeout + 30)  # the child is stopped at its timeout
        if (known := _measured.get(name)) is not None:
            return known
        return unmeasured(name, self._family_failures.get(name) or self._failure or "not measured")

    def close(self, timeout: float = 10.0) -> None:
        with self._lock:
            self._closed = True
            proc, thread = self._proc, self._thread
        if proc is not None and proc.poll() is None:
            proc.kill()
        if thread is not None:
            thread.join(timeout)

    # ----------------------------------------------------------------------------------- internals
    def _measure(self, done: threading.Event) -> None:
        from ypuddin.models import available

        try:
            key = fingerprint()
            if self._load(key):
                return
            started = time.monotonic()
            families = self._run_child()
            log.info(
                "Measured model family layers in a separate process in %.1f s", time.monotonic() - started
            )
            stored = {}
            failures = []
            family_failures = {}
            with _lock:
                for name in available():
                    geometry = families.get(name)
                    if not _valid(geometry):
                        reason = (
                            geometry.get("failure") or geometry.get("error") or "invalid layer counts"
                            if isinstance(geometry, dict)
                            else "missing layer counts"
                        )
                        failures.append(f"{name}: {reason}")
                        family_failures[name] = reason
                        log.warning("Model family %s has no layer counts: %s", name, reason)
                        continue
                    _measured[name] = geometry
                    stored[name] = geometry
            with self._lock:
                if self._closed:
                    return
            self.db.set_kv(KV_KEY, {"fingerprint": key, "families": stored})
            with self._lock:
                self._failed_at = time.monotonic() if failures else None
                self._failure = "; ".join(failures)
                self._family_failures = family_failures
        except Exception as error:  # noqa: BLE001
            with self._lock:
                closed = self._closed
                self._failed_at = time.monotonic()
                self._failure = str(error) or type(error).__name__
                self._family_failures = {}
            if not closed:
                log.warning("Could not measure model family layers: %s", error)
        finally:
            with self._lock:
                self._running = None
            done.set()

    def _load(self, key: str) -> bool:
        from ypuddin.models import available

        saved = self.db.get_kv(KV_KEY, {})
        families = (
            saved.get("families") if isinstance(saved, dict) and saved.get("fingerprint") == key else None
        )
        if not isinstance(families, dict) or not all(_valid(families.get(name)) for name in available()):
            return False
        with _lock:
            for name in available():
                _measured.setdefault(name, families[name])
        return True

    def _run_child(self) -> dict[str, Any]:
        root = _PACKAGE.parent
        code = (
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            f"for name in {_GPU_EXTENSIONS!r}:\n"
            "    sys.modules[name] = None\n"
            "from ypuddin.server.family_geometry import child_main\n"
            "child_main()\n"
        )
        options: dict[str, Any] = {}
        if os.name == "nt":  # pragma: no cover - Windows only
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        with self._lock:
            if self._closed:
                raise RuntimeError("service is stopping")
            self._proc = proc = subprocess.Popen(
                [sys.executable, "-c", code, str(root)],
                cwd=str(root),
                env={**os.environ, **HIDDEN_GPUS},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **options,
            )
        try:
            try:
                out, err = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise RuntimeError(f"no result after {self.timeout:.0f} s") from None
        finally:
            with self._lock:
                self._proc = None
        for line in reversed(out.splitlines()):
            if line.startswith(MARKER):
                return json.loads(line[len(MARKER) :])
        detail = err.strip()[-2000:] or f"exit code {proc.returncode}"
        raise RuntimeError(detail)


def child_main() -> None:
    """Entry point of the measuring child: one JSON line with every family's geometry."""
    from ypuddin.models import available

    families: dict[str, Any] = {}
    for name in available():
        try:
            families[name] = measure(name)
        except Exception as error:  # noqa: BLE001
            families[name] = {"error": f"{type(error).__name__}: {error}"}
    sys.stdout.write("\n" + MARKER + json.dumps(families) + "\n")
    sys.stdout.flush()
