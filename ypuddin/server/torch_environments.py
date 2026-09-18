"""Isolated, reversible framework installation. Never replace loaded Torch/DLLs.

Builds use the exact stable combinations documented at
https://pytorch.org/get-started/previous-versions/ (checked 2026-09-14).
Optional compiled attention extensions are deliberately not copied across Torch ABIs.
"""

from __future__ import annotations

import json
import ntpath
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ypuddin.package_sources import pypi_sources, run_sources, torch_sources
from ypuddin.runtime_profiles import current_profile, selected_key

from .db import new_id, now
from .environment import EnvironmentError, Installer, protected

SOURCE_ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_EXTENSIONS = {"xformers", "flash-attn", "sageattention", "bitsandbytes"}
VERSIONS = {
    "2.13.0": ("0.28.0", ("cu126", "cu130", "cpu", "mps")),
    "2.11.0": ("0.26.0", ("cu126", "cu128", "cu130", "cpu", "mps")),
    "2.10.0": ("0.25.0", ("cu126", "cu128", "cu130", "cpu", "mps")),
}
# Conservative native driver floors, not the broader minor-version compatibility claim.
DRIVER_FLOORS = {"cu126": 560, "cu128": 570, "cu130": 580}
ACTIVE = {"planning", "installing", "verifying"}


class TorchBuild(BaseModel):
    id: str
    torch: str
    torchvision: str
    backend: str
    label: str
    index_url: str
    supported: bool
    reason: str | None = None
    recommended: bool = False
    validation: str = "official_build"


class TorchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    build_id: str


class TorchOperation(BaseModel):
    environment_profile: str = "legacy"
    id: str
    build_id: str
    status: Literal["planning", "ready", "installing", "verifying", "completed", "failed", "cancelled"]
    created_at: float
    updated_at: float
    plan: list[dict[str, Any]] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    error: str | None = None
    phase: str = "preflight"
    environment_id: str | None = None
    dismissed_at: float | None = None
    progress: float | None = None


class TorchSnapshot(BaseModel):
    environment_profile: str = "legacy"
    environment_root: str = ""
    unavailable_backends: list[dict[str, str]] = Field(
        default_factory=lambda: [
            {"id": "dtk", "label": "海光 DTK 在线环境切换", "reason": "vendor_runtime_required"}
        ]
    )
    builds: list[TorchBuild]
    operations: list[TorchOperation]
    current_python: str
    selected_environment: str | None
    environments: list[dict[str, Any]]
    disk_free_bytes: int
    minimum_free_bytes: int
    source_url: str = "https://pytorch.org/get-started/previous-versions/"
    optional_extensions: list[str]


def python_in(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def same_interpreter(left: str | None, right: str | None) -> bool:
    """Compare interpreter locations, not the common binary a venv may link to."""
    if not isinstance(left, str) or not left or not isinstance(right, str) or not right:
        return False

    def identity(value: str) -> str:
        if sys.platform != "win32":
            return os.path.normcase(os.path.abspath(value))
        value = ntpath.normcase(ntpath.abspath(value))
        if value.startswith("\\\\?\\unc\\"):
            return "\\\\" + value[8:]
        return value[4:] if value.startswith("\\\\?\\") else value

    return identity(left) == identity(right)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def nvidia_driver_major() -> int | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        result = subprocess.run(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return int(result.stdout.strip().split(".")[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def build_catalog(runtime: dict, driver: int | None, profile: str = "legacy") -> list[TorchBuild]:
    result = []
    system = runtime["platform"]
    for version, (vision, backends) in VERSIONS.items():
        for backend in backends:
            reason = None
            if backend == "mps" and (
                system != "Darwin" or runtime["machine"].lower() not in ("arm64", "aarch64")
            ):
                reason = "requires_apple_silicon"
            elif (
                system == "Darwin" and backend != "mps" and not (profile == "macos-cpu" and backend == "cpu")
            ):
                reason = "use_macos_mps_build"
            elif backend.startswith("cu"):
                if system not in ("Windows", "Linux"):
                    reason = "requires_windows_or_linux_nvidia"
                elif driver is None:
                    reason = "nvidia_driver_not_detected"
                elif driver < DRIVER_FLOORS[backend]:
                    reason = f"requires_driver_{DRIVER_FLOORS[backend]}"
                elif (runtime.get("gpu_capability") or [0])[0] >= 12 and backend == "cu126":
                    reason = "blackwell_requires_cu128_or_newer"
            if profile != "legacy":
                matches = (
                    (profile.endswith("-cpu") and backend == "cpu")
                    or (profile.endswith("-cuda") and backend.startswith("cu"))
                    or (profile == "macos-mps" and backend == "mps")
                )
                if not matches:
                    reason = "different_deployment_profile"
            tested = version == "2.11.0" and backend == "cu128"
            result.append(
                TorchBuild(
                    id=f"{version}-{backend}",
                    torch=version,
                    torchvision=vision,
                    backend=backend,
                    label=f"PyTorch {version} · " + ("Apple MPS" if backend == "mps" else backend.upper()),
                    index_url="https://pypi.org/simple"
                    if backend == "mps" or (backend == "cpu" and system == "Darwin")
                    else f"https://download.pytorch.org/whl/{backend}",
                    supported=reason is None,
                    reason=reason,
                    recommended=reason is None and (tested or (version == "2.13.0" and backend == "mps")),
                    validation="tiny_cuda_training_verified" if tested else "official_build",
                )
            )
    return result


VERIFY = r"""
import json, sys, torch, torchvision
from pathlib import Path
from ypuddin.server.app import create_app
expected, backend, report = sys.argv[1:]
assert torch.__version__.split('+')[0] == expected, (torch.__version__, expected)
device = 'cuda' if backend.startswith('cu') else ('mps' if backend == 'mps' else 'cpu')
if device == 'cuda':
    assert torch.cuda.is_available(), 'CUDA wheel installed, but driver/GPU is unavailable'
if device == 'mps':
    assert torch.backends.mps.is_available(), 'Apple MPS is unavailable'
x = torch.randn(4, 4, device=device, requires_grad=True)
y = (x @ x).square().mean(); y.backward()
assert torch.isfinite(x.grad).all(), 'Framework forward/backward verification failed'
Path(report).write_text(json.dumps({'torch':str(torch.__version__), 'torchvision':str(torchvision.__version__), 'device':device, 'forward_backward':True, 'python':sys.executable}), encoding='utf-8')
"""


class TorchEnvironments:
    def __init__(self, context, environment, *, installer=None, driver=nvidia_driver_major):
        self.context, self.environment = context, environment
        self.profile = getattr(environment, "profile", current_profile())
        self.selected_key = selected_key(self.profile)
        self.root = environment.root / "runtimes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.installer = installer or Installer(environment.root, context=context)
        self.driver = driver
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="torch-environment")
        self.lock = threading.RLock()
        self.cancel_events: dict[str, threading.Event] = {}
        for op in self.list():
            if op.status in ACTIVE:
                self._update(
                    op.id,
                    status="failed",
                    error="Service stopped while preparing this environment. The previous environment is unchanged.",
                )

    def list(self) -> list[TorchOperation]:
        rows = self.context.db.fetchall("SELECT value FROM kv WHERE key LIKE 'torch.operation.%'")
        return sorted(
            (
                TorchOperation.model_validate(value)
                for r in rows
                if (value := json.loads(r["value"])).get("environment_profile", "legacy") == self.profile
            ),
            key=lambda o: o.created_at,
            reverse=True,
        )

    def get(self, id_: str) -> TorchOperation:
        value = self.context.db.get_kv("torch.operation." + id_)
        if not value or value.get("environment_profile", "legacy") != self.profile:
            raise EnvironmentError(404, "PyTorch operation not found")
        return TorchOperation.model_validate(value)

    def _update(self, id_: str, **changes) -> TorchOperation:
        with self.lock:
            value = self.get(id_).model_dump()
            value.update(changes, updated_at=now())
            self.context.db.set_kv("torch.operation." + id_, value)
            return TorchOperation.model_validate(value)

    def _log(self, id_: str, message: str):
        with self.lock:
            self._update(id_, logs=[*self.get(id_).logs, message[-4000:]][-300:])

    def current_environment(self) -> str | None:
        for op in self.list():
            if op.status != "completed" or not op.environment_id:
                continue
            record = self.context.db.get_kv("torch.environment." + op.environment_id, {})
            if record.get("environment_profile", "legacy") != self.profile:
                continue
            if same_interpreter(record.get("python"), sys.executable):
                return op.environment_id
        return None

    def status(self):
        versions = self.environment.versions()
        return TorchSnapshot(
            environment_profile=self.profile,
            environment_root=str(self.root),
            builds=build_catalog(self.environment.runtime(), self.driver(), self.profile),
            operations=self.list(),
            current_python=sys.executable,
            selected_environment=self.current_environment(),
            environments=[
                self.context.db.get_kv("torch.environment." + o.environment_id)
                for o in self.list()
                if o.status == "completed" and o.environment_id
            ],
            disk_free_bytes=shutil.disk_usage(self.root).free,
            minimum_free_bytes=8 * 1024**3,
            optional_extensions=sorted(OPTIONAL_EXTENSIONS.intersection(versions)),
        )

    def _idle(self):
        self.environment._idle()
        if any(op.status in ACTIVE for op in self.list()):
            raise EnvironmentError(409, "Another PyTorch operation is in progress")
        if any(op.status in ("planning", "installing", "verifying") for op in self.environment.list()):
            raise EnvironmentError(409, "An extension operation is in progress")

    def start(self, request: TorchRequest):
        with self.environment.lock, self.lock, self.context.db.lock:
            self._idle()
            build = next(
                (
                    b
                    for b in build_catalog(self.environment.runtime(), self.driver(), self.profile)
                    if b.id == request.build_id
                ),
                None,
            )
            if not build or not build.supported:
                raise EnvironmentError(422, build.reason if build else "Unknown PyTorch build")
            required = (12 if build.backend.startswith("cu") else 8) * 1024**3
            if shutil.disk_usage(self.root).free < required:
                raise EnvironmentError(
                    422,
                    f"At least {required // 1024**3} GiB of free space is required to keep the existing environment and prepare a replacement",
                )
            op = TorchOperation(
                environment_profile=self.profile,
                id=new_id("torch"),
                build_id=build.id,
                status="ready",
                created_at=now(),
                updated_at=now(),
                phase="review",
                plan=[
                    {
                        "name": "torch",
                        "from_version": self.environment.versions().get("torch"),
                        "version": build.torch,
                        "index_url": build.index_url,
                    },
                    {
                        "name": "torchvision",
                        "from_version": self.environment.versions().get("torchvision"),
                        "version": build.torchvision,
                    },
                    {
                        "name": "isolated_environment",
                        "minimum_free_bytes": required,
                        "keeps_current_environment": True,
                    },
                    {
                        "name": "optional_extensions",
                        "not_copied": sorted(OPTIONAL_EXTENSIONS.intersection(self.environment.versions())),
                    },
                ],
            )
            self.context.db.set_kv("torch.operation." + op.id, op.model_dump())
            return op

    def apply(self, id_: str):
        with self.environment.lock, self.lock, self.context.db.lock:
            self._idle()
            if self.get(id_).status != "ready":
                raise EnvironmentError(409, "Create a new plan before preparing this environment")
            previous = self.context.db.get_kv("environment.maintenance", {})
            self.context.db.set_kv("torch.prior_maintenance." + id_, previous)
            self.context.db.set_kv(
                "environment.maintenance", {**previous, "blocked": True, "torch_operation": id_}
            )
            self.cancel_events[id_] = threading.Event()
            op = self._update(id_, status="installing", phase="creating_environment", progress=None)
            self.pool.submit(self._install, id_)
            return op

    def _install(self, id_: str):
        cancel = self.cancel_events[id_]

        def log(message):
            self._log(id_, message)

        work = self.root / id_
        try:
            build = next(
                b
                for b in build_catalog(self.environment.runtime(), self.driver(), self.profile)
                if b.id == self.get(id_).build_id
            )
            if not build.supported:
                raise ValueError(build.reason)
            # All writes stay inside this operation directory. Failed environments are never selected.
            self.installer.run([sys.executable, "-m", "venv", str(work)], log, cancel, timeout=180)
            py = str(python_in(work))
            pip = [
                py,
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "--cache-dir",
                str(self.environment.root / "cache"),
            ]
            # Bind the backend as well as the release so dependency resolution cannot
            # replace a CUDA wheel with a CPU build from a general package index.
            torch_version = build.torch + (f"+{build.backend}" if build.backend.startswith("cu") else "")
            constraints = work / "constraints.txt"
            versions = self.environment.versions()
            constraints.write_text(
                "".join(
                    f"{name}==={version}\n"
                    for name, version in sorted(versions.items())
                    if (not protected(name) or name == "numpy")
                    and name not in OPTIONAL_EXTENSIONS
                    and name != "ypuddin"
                )
                + f"torch=={torch_version}\ntorchvision=={build.torchvision}\n",
                encoding="utf-8",
            )
            self._update(id_, phase="installing_pytorch")
            sources = self.context.settings().get("downloads", {})
            fallback = sources.get("fallback", True)
            indexes = pypi_sources(sources.get("pypi", "ustc"), fallback)
            torch_indexes = (
                [("index-url", url) for url in indexes]
                if build.index_url == "https://pypi.org/simple"
                else torch_sources(build.backend, sources.get("pytorch", "mirror"), fallback)
            )
            run_sources(
                self.installer,
                [
                    (
                        url,
                        [
                            *pip,
                            "install",
                            "--only-binary=:all:",
                            "--no-input",
                            "--no-deps",
                            *(
                                ["--no-index", "--find-links", url]
                                if kind == "find-links"
                                else ["--index-url", url]
                            ),
                            f"torch=={torch_version}",
                            f"torchvision=={build.torchvision}",
                        ],
                    )
                    for kind, url in torch_indexes
                ],
                log,
                cancel,
            )
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            project = tomllib.loads((SOURCE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
            extras = ["models", "server", "optim", "logging"] + (
                ["nvidia"] if build.backend.startswith("cu") else []
            )
            requirements = project["dependencies"] + [
                r for extra in extras for r in project["optional-dependencies"][extra]
            ]
            self._update(id_, phase="installing_dependencies")
            run_sources(
                self.installer,
                [
                    (
                        url,
                        [
                            *pip,
                            "install",
                            "--only-binary=:all:",
                            "--no-input",
                            "--index-url",
                            url,
                            "--constraint",
                            str(constraints),
                            f"torch=={torch_version}",
                            f"torchvision=={build.torchvision}",
                            *requirements,
                        ],
                    )
                    for url in indexes
                ],
                log,
                cancel,
            )
            # Source linkage gives every environment the same checked-out code without creating egg-info.
            site = work / (
                "Lib/site-packages"
                if os.name == "nt"
                else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            )
            (site / "ypuddin-source.pth").write_text(str(SOURCE_ROOT) + "\n", encoding="utf-8")
            self._update(id_, status="verifying", phase="verifying")
            self.installer.run([*pip, "check"], log, cancel)
            self.installer.run(
                [py, "-c", VERIFY, build.torch, build.backend, str(work / "verification.json")],
                log,
                cancel,
                timeout=120,
            )
            verified = json.loads((work / "verification.json").read_text(encoding="utf-8"))
            record = {
                "environment_profile": self.profile,
                "id": id_,
                "build_id": build.id,
                "python": py,
                "verified_at": now(),
                "verification": verified,
            }
            self.context.db.set_kv("torch.environment." + id_, record)
            self._update(id_, status="completed", phase="ready_to_restart", environment_id=id_, progress=1.0)
            log(
                "Environment verified. Choose Restart and switch to activate it; the previous environment is preserved."
            )
        except InterruptedError:
            self._update(id_, status="cancelled", phase="cancelled")
        except Exception as exc:
            self._log(id_, str(exc))
            self._update(id_, status="failed", error=str(exc), phase="failed")
        finally:
            self.context.db.set_kv(
                "environment.maintenance", self.context.db.get_kv("torch.prior_maintenance." + id_, {})
            )

    def resolve(self, id_: str) -> str:
        record = self.context.db.get_kv("torch.environment." + id_)
        if (
            not record
            or record.get("environment_profile", "legacy") != self.profile
            or self.get(id_).status != "completed"
        ):
            raise EnvironmentError(422, "Only a verified environment can be activated")
        expected = python_in(self.root / id_)
        if record["python"] != str(expected) or not expected.is_file() or (self.root / id_).is_symlink():
            raise EnvironmentError(422, "Prepared environment no longer exists at its verified location")
        return str(expected)

    def cancel(self, id_: str):
        op = self.get(id_)
        if op.status == "ready":
            return self._update(id_, status="cancelled")
        if op.status in ACTIVE and id_ in self.cancel_events:
            self.cancel_events[id_].set()
            return op
        raise EnvironmentError(409, "Operation is already finished")

    def dismiss(self, id_: str):
        if self.get(id_).status not in ("completed", "failed", "cancelled"):
            raise EnvironmentError(409, "Only a finished notification can be dismissed")
        return self._update(id_, dismissed_at=now())

    def close(self):
        for event in self.cancel_events.values():
            event.set()
        self.pool.shutdown(wait=True)
