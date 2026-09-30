"""YPuddin Train Studio 一键引导脚本（只用标准库，裸系统 Python 即可运行）。

由各环境的启动脚本经 ``scripts/launch.sh`` / ``scripts/launch.bat`` 调用
（``studio-linux-dtk.sh`` 直接调用本脚本）::

    python scripts/bootstrap.py [全局参数] [命令] [命令参数]

命令
  run     （默认）创建/更新 venv、按需构建前端、启动服务、打开浏览器
  dev     同时启动后端与 Vite 热更新前端
  build   只构建前端
  test    运行本机 Test/ 目录里的测试（测试不随仓库发布）
  smoke   转发到 ``ypuddin smoke``（使用指定权重执行短程训练自检）
  doctor  打印本机情况：Python、torch/CUDA/显卡、可选依赖、Node、前端构建状态
  shell   打印如何激活 venv

全局参数
  --profile=<auto|legacy|windows-cuda|linux-cuda|linux-dtk|macos-mps|cpu>  独立平台环境；auto 保留已有 legacy venv
  --torch=<cu128|cu126|cu124|cu118|cpu|auto>  首次安装/重建的 PyTorch 类型（默认 auto）
  --index=<auto|cn|official>  包源；未指定时使用界面保存的下载源设置，没有保存时为 auto。
                  auto / cn：中科大 -> 清华 -> 阿里 -> 官方，失败后依次换源；official：普通依赖官方优先、镜像兜底，PyTorch 仅用官方源
  --mirror        等价于 --index=cn
  --env-root=<目录>  基础环境根目录，按平台隔离；默认源码目录下的 environment
  --reinstall     只重建选中平台的环境（其他环境及 studio_data/ 不受影响）
  --no-browser    服务起来后不自动打开浏览器
  --no-frontend   跳过前端构建（只要 API）
  --dtk-wheelhouse=<目录>  Linux DTK 首次安装使用的本地厂商 wheel 集合（离线或未核对的 DTK 版本）；不从普通包源安装 Torch
  --host/--port/--data-root 原样传给 ``ypuddin serve``
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ypuddin import dtk_builds  # noqa: E402
from ypuddin.package_sources import (  # noqa: E402
    pypi_sources,
)
from ypuddin.package_sources import (  # noqa: E402
    torch_sources as configured_torch_sources,
)

DOWNLOAD_SETTINGS = None
PACKAGE_CACHE_ROOT = None
VENV = ROOT / "venv"
FRONTEND = ROOT / "frontend"
MARKER = VENV / ".ypuddin-install.json"
PROFILE = "legacy"
WIN = os.name == "nt"
PYPI_OFFICIAL = "https://pypi.org/simple"
# mainland-China PyPI mirrors in the order they are tried; the official index is always the last resort
PYPI_MIRRORS_CN = (
    "https://mirrors.ustc.edu.cn/pypi/simple",
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
)
TORCH_OFFICIAL = "https://download.pytorch.org/whl/{tag}"
# mirrors of download.pytorch.org tried before the official index: Aliyun serves one flat wheel listing per
# CUDA tag (--find-links), SJTU mirrors the PEP 503 layout (--index-url)
TORCH_MIRRORS_CN = (
    ("index-url", "https://mirror.sjtu.edu.cn/pytorch-wheels/{tag}"),
    ("find-links", "https://mirrors.aliyun.com/pytorch-wheels/{tag}"),
)
# minimum NVIDIA driver (major) able to run each CUDA wheel flavour, newest first
CUDA_TAGS = (("cu128", 570), ("cu126", 560), ("cu124", 550), ("cu118", 450))
# GPUs with compute capability >= 12.0 (RTX 50 series / Blackwell) only have kernels in the cu128+ wheels
BLACKWELL_CC = 12.0
EXTRAS_BASE = "models,server,optim,logging"
DTK_NUMPY1_TORCH = "2.7.1+das.opt1.dtk2604"


def log(msg: str) -> None:
    print(f"[studio] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[studio] 错误：{msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([str(c) for c in cmd], check=True, **kw)
    except subprocess.CalledProcessError:
        # Callers may clean up partial environments before stopping.
        log("  命令失败：" + " ".join(str(c) for c in cmd))
        raise


# --------------------------------------------------------------------------- environment probes
def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if WIN else "bin/python")


def venv_bin(name: str) -> Path:
    return VENV / ("Scripts" if WIN else "bin") / (name + (".exe" if WIN else ""))


def uv_path() -> str | None:
    return shutil.which("uv")


def uv_cache_dir(uv: str) -> Path | None:
    try:
        out = subprocess.run([uv, "cache", "dir"], capture_output=True, text=True, timeout=15).stdout.strip()
        return Path(out) if out else None
    except Exception:  # noqa: BLE001
        return None


def st_dev_of(path: Path) -> int | None:
    """st_dev of the nearest existing ancestor (the path itself may not exist yet)."""
    p = Path(path)
    while True:
        try:
            return os.stat(p).st_dev
        except OSError:
            if p.parent == p:
                return None
            p = p.parent


def needs_copy_link_mode(cache_dev: int | None, root_dev: int | None) -> bool:
    """uv hard-links packages from its cache; across filesystems it warns and falls back to copying."""
    return cache_dev is not None and root_dev is not None and cache_dev != root_dev


def _nvidia_smi(query: str) -> list[str]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def nvidia_driver_major() -> int | None:
    for line in _nvidia_smi("driver_version"):
        m = re.search(r"(\d+)\.", line)
        if m:
            return int(m.group(1))
    return None


def nvidia_gpus() -> list[tuple[str, float | None]]:
    """[(name, compute capability)] e.g. [("NVIDIA GeForce RTX 5090", 12.0)]."""
    gpus = []
    for line in _nvidia_smi("name,compute_cap"):
        name, _, cc = line.rpartition(",")
        try:
            gpus.append((name.strip() or line, float(cc.strip())))
        except ValueError:
            gpus.append((line, None))
    return gpus


def pick_torch_tag(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin":
        return "cpu"  # PyPI wheels carry MPS support
    driver = nvidia_driver_major()
    if driver is None:
        return "cpu"
    gpus = nvidia_gpus()
    if any(cc is not None and cc >= BLACKWELL_CC for _, cc in gpus):
        # RTX 50 series: older CUDA builds fail with "no kernel image is available for execution on the device"
        if driver < 570:
            log(
                f"警告：检测到 RTX 50 系显卡，但驱动版本 {driver} 过旧（需要 >= 570），请先升级显卡驱动，否则 CUDA 无法使用"
            )
        return "cu128"
    for tag, min_driver in CUDA_TAGS:
        if driver >= min_driver:
            return tag
    return "cpu"


def host_arch() -> str:
    """Normalized CPU architecture; wheels differ across it even on the same OS."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine or "unknown"


# Entries whose PyTorch source has only ever been installed and validated on x86_64.
X86_ONLY_PROFILES = ("windows-cuda", "linux-cuda", "linux-dtk")


def platform_torch_tag(profile: str, requested: str) -> str:
    """Validate an explicit deployment profile before creating or modifying its venv."""
    expected = {"windows-cuda": "Windows", "linux-cuda": "Linux", "linux-dtk": "Linux", "macos-mps": "Darwin"}
    if profile not in ("auto", "legacy", "cpu", *expected):
        die("未知部署类型；请选择 auto/legacy/windows-cuda/linux-cuda/linux-dtk/macos-mps/cpu")
    if profile in expected and platform.system() != expected[profile]:
        die(f"{profile} 启动入口只适用于 {expected[profile]}，当前为 {platform.system()}")
    if profile in X86_ONLY_PROFILES and host_arch() != "x86_64":
        die(f"{profile} 启动入口只支持 x86_64，当前架构是 {host_arch()}；请使用 CPU 启动入口。")
    if profile == "linux-dtk":
        if requested not in ("auto", "dtk"):
            die("DTK 部署入口只能使用匹配海光运行时的厂商 PyTorch，不能安装 CUDA/CPU wheel")
        return "dtk"
    if requested == "dtk":
        die("DTK PyTorch 只能通过 studio-linux-dtk.sh 独立启动入口使用")
    if profile == "macos-mps":
        if platform.machine().lower() not in ("arm64", "aarch64"):
            die("Apple MPS 部署入口需要 Apple Silicon Mac")
        if requested not in ("auto", "cpu"):
            die("Apple MPS 使用 macOS PyTorch，不安装 CUDA wheel")
        return "cpu"  # On macOS this selects the universal PyPI build with MPS.
    if profile == "cpu":
        if requested not in ("auto", "cpu"):
            die("CPU 部署入口不能选择 CUDA wheel")
        return "cpu"
    tag = pick_torch_tag(requested)
    if profile.endswith("-cuda") and (not tag.startswith("cu") or nvidia_driver_major() is None):
        die("CUDA 部署入口未检测到可用 NVIDIA 驱动。请先安装驱动，或使用 CPU 启动入口。")
    return tag


def select_environment(profile: str, torch_tag: str, env_root: str | None = None) -> str:
    """Select a directory before all probes/mutations. Never migrate a pre-existing venv."""
    global VENV, MARKER, PROFILE
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system())
    if not env_root and (profile == "legacy" or (profile == "auto" and (ROOT / "venv").exists())):
        PROFILE, VENV = "legacy", ROOT / "venv"
    else:
        if not system:
            die("此系统没有独立部署环境入口")
        if profile == "auto":
            profile = (
                "macos-mps"
                if system == "macos" and platform.machine().lower() in ("arm64", "aarch64")
                else (system + "-cuda" if torch_tag.startswith("cu") else "cpu")
            )
        PROFILE = system + "-cpu" if profile == "cpu" else profile
        base = Path(os.path.abspath(Path(env_root).expanduser())) if env_root else ROOT / "environment"
        VENV = base / PROFILE / "venv"
        # Old environment/profiles/ layout: report it instead of deleting gigabytes on the user's behalf.
        stale = ROOT / "environment" / "profiles"
        if not env_root and stale.is_dir():
            log(f"旧版环境目录 {stale} 已不再使用；确认 {VENV} 可用后可以删除。")
    MARKER = VENV / ".ypuddin-install.json"
    # Refuse links before either installation or explicit rebuild can affect another directory.
    path = VENV
    while path != ROOT and path.parent != path:
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            die(f"环境目录不允许链接或目录联接，未修改：{path}")
        path = path.parent
    owner_marker = MARKER if MARKER.exists() else VENV / ".ypuddin-owner.json"
    if VENV.exists() and not VENV.is_dir():
        die(f"环境路径不是目录，未修改：{VENV}")
    if env_root and VENV.exists() and any(VENV.iterdir()) and not owner_marker.exists():
        die(f"指定目录已有未登记的环境，未修改：{VENV}；请选择新的环境根目录")
    if owner_marker.exists():
        try:
            marker = json.loads(owner_marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            die("环境安装标记损坏，未修改环境；请先检查环境目录")
        owner = marker.get("profile", "legacy")
        if owner != PROFILE:
            die(f"环境属于 {owner}，不能作为 {PROFILE} 更新或重建")
        installed_arch = marker.get("arch")
        if installed_arch and installed_arch != host_arch():
            die(
                f"该环境是在 {installed_arch} 上安装的，不能在 {host_arch()} 上更新或重建；"
                "请为本架构使用独立的环境目录。"
            )
    return PROFILE


def url_ok(url: str, timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def index_chains(mode: str, torch_tag: str) -> tuple[list[str], list[tuple[str, str]]]:
    """(PyPI index urls in order, torch sources as (kind, url) with kind in {find-links, index-url}).

    auto/cn: mirrors first (USTC -> Tsinghua -> Aliyun), official PyPI as the last resort.
    official: official first, mirrors as the fallback.
    """
    if DOWNLOAD_SETTINGS is not None:
        sources = DOWNLOAD_SETTINGS
        return (
            pypi_sources(sources.get("pypi", "ustc"), sources.get("fallback", True)),
            []
            if torch_tag == "dtk"
            else configured_torch_sources(
                torch_tag, sources.get("pytorch", "mirror"), sources.get("fallback", True)
            ),
        )
    if mode == "official":
        pypi = [PYPI_OFFICIAL, *PYPI_MIRRORS_CN]  # mirrors still serve as a fallback
        torch_src = [("index-url", TORCH_OFFICIAL.format(tag=torch_tag))]
    else:
        pypi = pypi_sources()
        torch_src = [(kind, url.format(tag=torch_tag)) for kind, url in TORCH_MIRRORS_CN] + [
            ("index-url", TORCH_OFFICIAL.format(tag=torch_tag))
        ]
    return pypi, [] if torch_tag == "dtk" else torch_src


def python_ok(exe: str) -> bool:
    try:
        v = subprocess.run(
            [exe, "-c", "import sys;print(sys.version_info[:2])"], capture_output=True, text=True
        ).stdout
        major, minor = (int(x) for x in re.findall(r"\d+", v)[:2])
        return (major, minor) >= (3, 10) and (major, minor) < (3, 13)
    except Exception:  # noqa: BLE001
        return False


def find_base_python() -> str:
    """A CPython 3.10–3.12 to create the venv with (the interpreter running this script is tried first)."""
    candidates = [sys.executable]
    for name in ("python3.12", "python3.11", "python3.10", "python3", "python"):
        exe = shutil.which(name)
        if exe:
            candidates.append(exe)
    for exe in candidates:
        if python_ok(exe):
            return exe
    die("没有找到 Python 3.10 - 3.12。请先安装（python.org，或 `uv python install 3.12`）再运行。")
    return ""  # unreachable


# --------------------------------------------------------------------------- venv + dependencies
def install_signature(torch_tag: str, extras: str) -> str:
    h = hashlib.sha256((ROOT / "pyproject.toml").read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update(
        f"{PROFILE}|{torch_tag}|{extras}|{sys.platform}|{platform.machine()}|{sys.version_info[:2]}".encode()
    )
    return h.hexdigest()[:16]


def venv_json(code: str, *args: str):
    """Probe the selected venv, never the Python that happened to launch this script."""
    with tempfile.TemporaryDirectory(prefix="ypuddin-bootstrap-probe-") as directory:
        result = subprocess.run(
            [str(venv_python()), "-c", code, *args],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=45,
            env=_env(),
        )
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or "environment probe failed")[-1600:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def installed_versions() -> dict[str, str]:
    return venv_json("""
import importlib.metadata as m, json, re
versions = {}
for distribution in m.distributions():
    name = distribution.metadata['Name']
    if name:
        versions.setdefault(re.sub(r'[-_.]+', '-', name).lower(), distribution.version)
print(json.dumps(versions))
""")


def editable_install_ready() -> bool:
    """Require independent PEP 660 metadata before removing setuptools' source-tree copy."""
    code = """
import importlib.metadata as metadata
import json, sys, sysconfig
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

root, environment = (Path(value).resolve() for value in sys.argv[1:])
ready = False
if Path(sys.prefix).resolve() == environment:
    import ypuddin
    source_matches = Path(ypuddin.__file__).resolve() == (root / 'ypuddin' / '__init__.py').resolve()
    for location in {sysconfig.get_path('purelib'), sysconfig.get_path('platlib')}:
        site = Path(location).resolve()
        if not site.is_relative_to(environment):
            continue
        for directory in site.glob('ypuddin-*.dist-info'):
            if not directory.resolve().is_relative_to(environment):
                continue
            dist = metadata.Distribution.at(directory)
            direct = json.loads(dist.read_text('direct_url.json') or '{}')
            url = urlsplit(direct.get('url', ''))
            source = Path(url2pathname(('//' + url.netloc if url.netloc else '') + url.path)).resolve()
            entrypoint = any(
                entry.group == 'console_scripts' and entry.name == 'ypuddin'
                and entry.value == 'ypuddin.cli:main' for entry in dist.entry_points
            )
            if (dist.metadata['Name'] == 'ypuddin' and dist.version == ypuddin.__version__
                and source_matches and entrypoint and url.scheme == 'file' and source == root
                and direct.get('dir_info', {}).get('editable') is True):
                ready = True
print(json.dumps(ready))
"""
    try:
        return bool(venv_json(code, str(ROOT), str(VENV)))
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
        return False


def cleanup_build_metadata() -> None:
    """Called only after verifying venv metadata; never remove runtime dist-info or link targets."""
    directory = ROOT / "ypuddin.egg-info"
    try:
        info = directory.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return  # retried on the next start
    try:
        # lstat also detects Windows junctions on supported Python versions.
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            return  # never follow links or remove a same-named file
        shutil.rmtree(directory)
    except OSError:
        # An occupied Windows file must not trigger another dependency install;
        # the next start retries.
        pass


def dependency_issues(extras: str) -> list[str]:
    """Verify installed requirement metadata (including transitive dependencies) without imports."""
    code = """
import importlib.metadata as metadata, json, sys
try:
    from packaging.requirements import Requirement
    from packaging.markers import default_environment
    from packaging.utils import canonicalize_name
except ImportError:
    print(json.dumps(['packaging is missing']))
    sys.exit(0)
issues, seen = [], set()
def visit(name, extras):
    key = (canonicalize_name(name), tuple(sorted(extras)))
    if key in seen:
        return
    seen.add(key)
    try:
        package = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        issues.append(name + ' is missing')
        return
    for text in package.requires or []:
        req = Requirement(text)
        if req.marker and not any(req.marker.evaluate(dict(default_environment(), extra=extra)) for extra in ['', *extras]):
            continue
        try:
            version = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            issues.append(str(req) + ' is missing')
            continue
        if req.specifier and not req.specifier.contains(version, prereleases=True):
            issues.append(str(req) + ' (installed ' + version + ')')
        visit(req.name, req.extras)
visit('ypuddin', sys.argv[1].split(','))
print(json.dumps(sorted(set(issues))))
"""
    try:
        return venv_json(code, extras)
    except Exception as exc:
        return [f"dependency verification failed: {exc}"]


def protected_versions(versions: dict[str, str]) -> dict[str, str]:
    """Keep the installed native training stack intact while adding ordinary dependencies."""
    return {
        name: version
        for name, version in versions.items()
        if name in {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton", "numpy"}
        or name.startswith(("nvidia-", "cuda-", "pytorch-", "dtk-", "dcu-", "hygon-", "hip-", "rocm-"))
        and name != "nvidia-ml-py"
    }


def torch_runtime() -> dict:
    return venv_json(
        "import torch,json; print(json.dumps({'version':torch.__version__,'cuda':torch.version.cuda,"
        "'hip':getattr(torch.version,'hip',None)}))"
    )


def model_runtime() -> dict:
    return venv_json("""
import ast, importlib.metadata as metadata, json
out = {'ok': False, 'stage': 'torchvision'}
for name in ('torch', 'torchvision', 'transformers'):
    try: out[name] = metadata.version(name)
    except metadata.PackageNotFoundError: out[name] = None
try:
    version_file = metadata.distribution('torchvision').locate_file('torchvision/version.py')
    for node in ast.parse(version_file.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets):
            out['torchvision_build'] = ast.literal_eval(node.value)
except (OSError, SyntaxError, ValueError, metadata.PackageNotFoundError):
    pass
try:
    import torch
    out.update(cuda=torch.version.cuda, hip=getattr(torch.version, 'hip', None))
    import torchvision
    torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5)
    if torch.version.cuda:
        vision_cuda = torch.ops.torchvision._cuda_version()
        major, minor = (int(v) for v in torch.version.cuda.split('.')[:2])
        if vision_cuda < 0 or (vision_cuda // 1000, (vision_cuda % 1000) // 10) != (major, minor):
            raise RuntimeError(f'Torch CUDA {torch.version.cuda} / TorchVision CUDA {vision_cuda} do not match')
    out['stage'] = 'qwen3'
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    out['ok'] = True
except Exception as exc:
    out['error'] = f'{type(exc).__name__}: {exc}'
print(json.dumps(out))
""")


def ensure_model_runtime(extras: str, index_mode: str) -> None:
    if "models" not in extras.split(","):
        return
    report = model_runtime()
    if report.get("ok"):
        return
    # A PyPI TorchVision can satisfy metadata requirements while carrying different
    # native CUDA operators. Replace only that wheel, keeping Torch and its dependencies.
    if (
        report.get("stage") == "torchvision"
        and report.get("cuda")
        and not report.get("hip")
        and report.get("torchvision")
    ):
        before = installed_versions()
        tag = "cu" + report["cuda"].replace(".", "")
        _, sources = index_chains(index_mode, tag)
        requirement = "torchvision==" + report["torchvision"].split("+")[0]
        uv, py = uv_path(), str(venv_python())
        command = (
            [uv, "pip", "install", "--python", py, "--reinstall-package", "torchvision"]
            if uv
            else [py, "-m", "pip", "install", "--force-reinstall"]
        )
        log(f"修复 TorchVision 原生算子（{tag}），保留 PyTorch {report.get('torch')}")
        for kind, url in sources:
            source_args = ["--index-url", url] if kind == "index-url" else ["--no-index", "--find-links", url]
            result = subprocess.run([*command, "--no-deps", requirement, *source_args], env=_env())
            if result.returncode == 0:
                after = installed_versions()
                changed = [
                    name
                    for name, version in before.items()
                    if name != "torchvision" and after.get(name) != version
                ]
                if changed:
                    die("TorchVision 修复改变了其他依赖，停止启动：" + ", ".join(changed))
                report = model_runtime()
                if report.get("ok"):
                    return
    display = report | {"torchvision": report.get("torchvision_build") or report.get("torchvision")}
    versions = " / ".join(
        f"{name} {display.get(name) or '未安装'}" for name in ("torch", "torchvision", "transformers")
    )
    die(f"模型依赖加载失败（{versions}）：{report.get('error', '未知错误')}。请运行 doctor 检查。")


def validate_dtk_runtime(current: dict) -> None:
    if not current.get("hip") or current.get("cuda"):
        die("DTK 环境需要厂商 HIP PyTorch；当前解释器不是 HIP 构建，未继续安装普通依赖。")


def dtk_compatibility_constraints(versions: dict[str, str]) -> list[str]:
    """Observed vendor-build incompatibilities missing from package metadata."""
    constraints = []
    if PROFILE == "linux-dtk" and "torch" in versions:
        torch_version = tuple(int(n) for n in re.findall(r"\d+", versions["torch"])[:2])
        if torch_version < (2, 5):
            constraints.append("transformers<5")
        if versions["torch"] == DTK_NUMPY1_TORCH:
            constraints.append("numpy>=1.26,<2")
    return constraints


def dtk_compatibility_issues(versions: dict[str, str]) -> list[str]:
    constraints = dtk_compatibility_constraints(versions)
    issues = []
    if "transformers<5" in constraints:
        transformers = versions.get("transformers", "")
        if transformers and int(re.match(r"\d+", transformers)[0]) >= 5:
            issues.append(
                f"transformers<5 is required by DTK Torch {versions['torch']} (installed {transformers})"
            )
    numpy = versions.get("numpy", "")
    if "numpy>=1.26,<2" in constraints and numpy and int(re.match(r"\d+", numpy)[0]) >= 2:
        issues.append(
            f"DTK Torch {versions['torch']} 与 NumPy {numpy} 的桥接 ABI 不兼容，需要 numpy>=1.26,<2"
        )
    return issues


def torch_numpy_bridge() -> dict:
    """Exercise both CPU bridges; importing Torch can merely warn about a broken ABI."""
    return venv_json("""
import json
try:
    import numpy as np
    import torch
    values = np.array([1.25, -2.5], dtype=np.float32)
    tensor = torch.from_numpy(values)
    tensor[0] = 3.5
    restored = tensor.numpy()
    assert tensor.device.type == 'cpu'
    assert values[0] == 3.5 and np.array_equal(restored, values)
    print(json.dumps({'ok': True}))
except Exception as exc:
    print(json.dumps({'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}))
""")


def validate_dtk_numpy_version(versions: dict[str, str]) -> None:
    issue = next((item for item in dtk_compatibility_issues(versions) if "NumPy" in item), None)
    if issue:
        die(issue + "。请准备匹配的依赖后重试，或使用 --reinstall 配合完整的厂商 wheel 集合重建。")


def validate_dtk_numpy_bridge(versions: dict[str, str]) -> None:
    validate_dtk_numpy_version(versions)
    try:
        result = torch_numpy_bridge()
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    if result.get("ok") is not True:
        die(
            "DTK 的 NumPy ↔ PyTorch CPU 桥接检查失败："
            + str(result.get("error", "unknown error"))
            + "。请检查 NumPy 与厂商 PyTorch 是否匹配后重试。"
        )


def dtk_native_numpy_constraints(
    wheelhouse: Path,
    versions: dict[str, str],
    *,
    uv: str | None,
    py: str,
    constraints: Path,
    requirements: list[str],
) -> list[str]:
    """Resolve the actual local Torch candidate before applying its ABI constraint.

    A wheelhouse may contain several builds or Python tags. Let the same installer
    select a compatible dependency solution, then pin that exact Torch version so
    the second resolution cannot switch builds underneath the NumPy constraint.
    """
    if "torch" in versions:
        return [value for value in dtk_compatibility_constraints(versions) if value.startswith("numpy")]
    candidates = set()
    for wheel in sorted(wheelhouse.glob("torch-*.whl")):
        try:
            with zipfile.ZipFile(wheel) as archive:
                entries = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if len(entries) != 1:
                    raise ValueError("wheel 需要唯一的 METADATA")
                metadata = BytesParser().parsebytes(archive.read(entries[0]))
            version = metadata.get("Version", "")
            if metadata.get("Name", "").lower() != "torch" or wheel.name.split("-")[1] != version:
                raise ValueError("Torch wheel 文件名与 METADATA 不一致")
            candidates.add(version)
        except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
            die(f"无法核验厂商 Torch wheel {wheel.name}：{exc}；未安装原生包")
    if DTK_NUMPY1_TORCH not in candidates:
        return []
    common = ["--python", py] if uv else []
    common += [
        "--no-index",
        "--only-binary=:all:",
        "--find-links",
        str(wheelhouse),
        "--constraint",
        str(constraints),
    ]
    if uv:
        request = constraints.parent / "native-request.txt"
        request.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        plan = constraints.parent / "native-plan.txt"
        run(
            [
                uv,
                "pip",
                "compile",
                str(request),
                *common,
                "--no-header",
                "--no-annotate",
                "--output-file",
                str(plan),
            ],
            env=_env(),
        )
        selected = re.findall(r"(?m)^torch==([^\s;\\]+)\s*$", plan.read_text(encoding="utf-8"))
    else:
        plan = constraints.parent / "native-plan.json"
        run(
            [py, "-m", "pip", "install", *common, "--dry-run", "--report", str(plan), *requirements],
            env=_env(),
        )
        selected = [
            item["metadata"]["version"]
            for item in json.loads(plan.read_text(encoding="utf-8"))["install"]
            if item.get("metadata", {}).get("name", "").lower() == "torch"
        ]
    if len(selected) != 1 or selected[0] not in candidates:
        die("无法确认安装器实际选择的本地厂商 Torch 构建，未安装原生包。")
    validate_dtk_numpy_version({**versions, "torch": selected[0]})
    return [
        f"torch=={selected[0]}",
        *[
            value
            for value in dtk_compatibility_constraints({"torch": selected[0]})
            if value.startswith("numpy")
        ],
    ]


def dtk_wheelhouse_has_triton(wheelhouse: Path) -> bool:
    """Recognize explicitly supplied vendor Triton without importing native code."""
    wheels = sorted(wheelhouse.glob("triton-*.whl"))
    for wheel in wheels:
        try:
            with zipfile.ZipFile(wheel) as archive:
                entries = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if len(entries) != 1:
                    raise ValueError("wheel 需要唯一的 METADATA")
                metadata = BytesParser().parsebytes(archive.read(entries[0]))
            version = metadata.get("Version", "")
            if (
                metadata.get("Name", "").lower() != "triton"
                or not re.search(r"\+[^\s]*\bdtk\d+\b", version)
                or wheel.name.split("-")[1] != version
            ):
                raise ValueError("需要文件名与 METADATA 一致的 DTK 厂商 Triton 构建")
        except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
            die(f"DTK wheel 目录中的 {wheel.name} 无法作为厂商 Triton 安装：{exc}；未修改环境")
    return bool(wheels)


def create_dtk_venv(
    base: str, wheelhouse: Path | None, *, system_site_packages: bool = False, index_url: str | None = None
) -> None:
    """Create only the selected, new DTK venv; never install pip into the host Python.

    ``system_site_packages`` lets the venv see the host's packages (its vendor PyTorch). Without
    ensurepip, pip comes from the wheel directory, else from ``index_url``.
    """
    env = _env()
    share = ["--system-site-packages"] if system_site_packages else []
    try:
        bundled_pip = (
            subprocess.run(
                [base, "-m", "ensurepip", "--version"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                env=env,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        bundled_pip = False
    if bundled_pip:
        run([base, "-m", "venv", *share, str(VENV)], env=env)
        return
    # Debian/Ubuntu vendor Python may omit ensurepip even though system pip works.
    # Check every bootstrap prerequisite before creating a partial environment.
    local_pip = wheelhouse is not None and any(wheelhouse.glob("pip-*.whl"))
    if not local_pip and not index_url:
        die(
            "当前 Python 缺少 ensurepip。请把与本机 Python 匹配的 pip-*.whl 放入 --dtk-wheelhouse 目录后重试，"
            "或改用 uv / 自带 ensurepip 的 Python。"
        )
    try:
        host_pip = subprocess.run(
            [base, "-m", "pip", "--isolated", "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=env,
        )
        can_target = host_pip.returncode == 0 and "--python" in host_pip.stdout
    except (OSError, subprocess.TimeoutExpired):
        can_target = False
    if not can_target:
        die(
            "当前 Python 缺少 ensurepip，且系统 pip 不支持 --python（需要 pip 22.3 或更新）。"
            "请安装 uv，或改用自带 ensurepip 的 Python。"
        )
    owned = not VENV.exists()
    source = ["--no-index", "--find-links", str(wheelhouse)] if local_pip else ["--index-url", str(index_url)]
    try:
        log(
            "[2/5] Python 未提供 ensurepip，使用"
            + ("本地 pip wheel" if local_pip else "包源中的 pip")
            + "引导独立 DTK 环境"
        )
        run([base, "-m", "venv", "--without-pip", *share, str(VENV)], env=env)
        run(
            [
                base,
                "-m",
                "pip",
                "--isolated",
                "--python",
                str(venv_python()),
                "install",
                "--no-cache-dir",
                "--disable-pip-version-check",
                "--no-deps",
                "--only-binary=:all:",
                *source,
                "pip",
            ],
            env=env,
        )
        run([str(venv_python()), "-m", "pip", "--version"], env=env)
    except (OSError, subprocess.CalledProcessError):
        # This directory did not exist before this call. Remove only our failed
        # bootstrap so the next attempt cannot mistake it for a ready environment.
        if owned and VENV.is_dir() and not VENV.is_symlink():
            shutil.rmtree(VENV)
        die("独立 DTK 环境的 pip 引导失败，请核对 pip 来源与 Python 版本后重试。")


# Runs in the Python a DTK environment would be built from. Importing Torch there needs the DTK
# library paths the launcher exports; "share" says how a new venv can see that PyTorch: through its
# system site-packages, or, when it lives inside another venv, through a .pth link to that venv.
DTK_PYTHON_PROBE = r"""
import importlib.metadata as metadata, json, platform, sys
from pathlib import Path
libc, libc_version = platform.libc_ver()
out = {
    "python": "cp%d%d" % sys.version_info[:2],
    "glibc": [int(part) for part in libc_version.split(".")[:2]] if libc == "glibc" and libc_version else None,
}
try:
    import torch, torchvision
    location = Path(torch.__file__).resolve().parent.parent
    in_venv = sys.prefix != sys.base_prefix and location.is_relative_to(Path(sys.prefix).resolve())
    # Vendor module versions may omit build suffixes; compare and pin package metadata in both environments.
    out.update(
        torch=metadata.version("torch"), torchvision=metadata.version("torchvision"), hip=getattr(torch.version, "hip", None),
        cuda=torch.version.cuda, location=str(location), share="link" if in_venv else "system",
    )
except Exception as exc:
    out["error"] = (type(exc).__name__ + ": " + str(exc))[-500:]
print(json.dumps(out))
"""


def probe_dtk_python(python: str) -> dict:
    """Python tag, glibc and any vendor PyTorch of an interpreter; empty when it cannot be run."""
    try:
        with tempfile.TemporaryDirectory(prefix="ypuddin-dtk-probe-") as directory:
            result = subprocess.run(
                [python, "-c", DTK_PYTHON_PROBE],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=300,
                env=_env(),
            )
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return {}


def vendor_stack(python: str, info: dict) -> dict | None:
    """The Hygon PyTorch a new environment can share from ``python``: a HIP build with TorchVision."""
    if not info.get("torch") or not info.get("hip") or info.get("cuda") or not info.get("torchvision"):
        return None
    return {
        "python": python,
        "share": info["share"],
        "location": info["location"],
        "torch": info["torch"],
        "torchvision": info["torchvision"],
    }


def recorded_vendor_stack() -> dict | None:
    """What the current environment shares, as recorded when it was created."""
    try:
        return json.loads((VENV / ".ypuddin-owner.json").read_text(encoding="utf-8")).get("vendor_stack")
    except (OSError, ValueError, AttributeError):
        return None


def installed_dtk() -> str | None:
    """DTK release of the runtime directory, e.g. 26.04 for /opt/dtk -> /opt/dtk-26.04."""
    root = Path(os.environ.get("DTK_ROOT", "/opt/dtk"))
    try:
        resolved = str(root.resolve()) if root.is_dir() else ""
    except OSError:
        resolved = ""
    match = re.search(r"dtk[-_](\d{2}\.\d{2}(?:\.\d+)?)$", resolved, re.IGNORECASE)
    return match[1] if match else None


def reviewed_dtk_set(info: dict):
    """The reviewed vendor PyTorch wheels for this DTK release and Python, if there is a set."""
    glibc = tuple(info.get("glibc") or ())
    if host_arch() != "x86_64" or not glibc or glibc < dtk_builds.MIN_GLIBC:
        return None
    return dtk_builds.RUNTIME_SETS.get((installed_dtk() or "", str(info.get("python", ""))))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, target: Path, size: int, digest: str) -> None:
    """Download to a partial file and keep it only when its size and SHA-256 match the review."""
    part = target.with_name(target.name + ".part")
    sha = hashlib.sha256()
    received, shown = 0, -1
    log(f"[3/5] 下载 {target.name}（{size / 2**20:.0f} MB）")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as stream:  # noqa: S310
            while chunk := response.read(1 << 20):
                received += len(chunk)
                if received > size:
                    break
                sha.update(chunk)
                stream.write(chunk)
                if received * 10 // size != shown:
                    shown = received * 10 // size
                    log(f"  {received * 100 // size}%（{received / 2**20:.0f} / {size / 2**20:.0f} MB）")
    except OSError as exc:
        part.unlink(missing_ok=True)
        die(f"下载 {target.name} 失败：{exc}。请检查网络后重试，或用 --dtk-wheelhouse 指定本地 wheel 目录。")
    if received != size or sha.hexdigest() != digest:
        part.unlink(missing_ok=True)
        die(f"下载的 {target.name} 与核对过的官方文件不一致（大小或 SHA-256），已删除；请重试。")
    os.replace(part, target)


def fetch_dtk_wheels(entries, directory: Path) -> list[Path]:
    """The reviewed vendor wheels, downloaded once; a kept file is used again only if it still matches."""
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for _, _, path, size, digest in entries:
        target = directory / path.rsplit("/", 1)[1]
        if not (target.is_file() and target.stat().st_size == size and file_sha256(target) == digest):
            download_verified(dtk_builds.file_url(path), target, size, digest)
        files.append(target)
    return files


def link_vendor_stack(location: str) -> None:
    """Put another venv's site-packages after ours, so its vendor PyTorch is importable here."""
    site = venv_json("import json, sysconfig; print(json.dumps(sysconfig.get_path('purelib')))")
    Path(site, "ypuddin-vendor-stack.pth").write_text(location + "\n", encoding="utf-8")


def dtk_base_python() -> str:
    """The Python a DTK environment is built from: the one a shared environment recorded, else the
    interpreter behind the current one; never a path inside the environment being rebuilt."""
    candidates = [(recorded_vendor_stack() or {}).get("python"), sys.executable]
    candidates += [
        shutil.which(name) for name in ("python3", "python3.12", "python3.11", "python3.10", "python")
    ]
    roots = {VENV.absolute(), VENV.resolve()}

    def inside(path: Path) -> bool:
        return any(path.is_relative_to(root) for root in roots)

    for candidate in filter(None, candidates):
        path = Path(candidate).absolute()
        if inside(path):
            # The launcher runs an existing environment's own Python; its link leads to the base.
            path = path.resolve()
            if inside(path):
                continue
        if python_ok(str(path)):
            return str(path)
    die("没有找到 Python 3.10 - 3.12；请设置 YPUDDIN_DTK_PYTHON=/path/to/python。")
    return ""  # unreachable


def ensure_venv(
    torch_tag: str, *, index_mode: str, reinstall: bool, extras: str, dtk_wheelhouse: str | None = None
) -> None:
    wheelhouse = None
    vendor_triton = False
    base = None
    shared = None  # the vendor PyTorch of another Python this environment uses
    downloads = None  # reviewed vendor wheels to download when there is nothing to share
    if dtk_wheelhouse:
        if PROFILE != "linux-dtk":
            die("--dtk-wheelhouse 只能用于 linux-dtk 环境")
        wheelhouse = Path(dtk_wheelhouse).expanduser().resolve()
        if not wheelhouse.is_dir() or not all(
            any(wheelhouse.glob(name + "-*.whl")) for name in ("torch", "torchvision")
        ):
            die("DTK wheel 目录需要包含匹配的 torch 与 torchvision wheel")
        vendor_triton = dtk_wheelhouse_has_triton(wheelhouse)
    if PROFILE == "linux-dtk" and (reinstall or not venv_python().exists()) and not wheelhouse:
        # Settle where the vendor PyTorch comes from before anything is created or deleted: the
        # chosen Python's own installation, else the reviewed download for this DTK and Python.
        base = dtk_base_python()
        info = probe_dtk_python(base)
        shared = vendor_stack(base, info)
        downloads = None if shared else reviewed_dtk_set(info)
        if not shared and not downloads:
            die(
                f"{base} 里没有海光版 PyTorch，也没有与本机 DTK {installed_dtk() or '未知版本'} / "
                f"Python {info.get('python', '未知版本')} 对应的已核对下载。请设置 YPUDDIN_DTK_PYTHON 指向"
                "已装好海光版 PyTorch 的 Python，或用 --dtk-wheelhouse=目录 提供匹配的厂商 wheel。"
            )
    if reinstall and VENV.exists():
        log(f"--reinstall：删除旧的虚拟环境 {VENV}（studio_data/ 不受影响）")
        shutil.rmtree(VENV)
    sig = install_signature(torch_tag, extras)
    ready = False
    if venv_python().exists() and MARKER.exists():
        try:
            if json.loads(MARKER.read_text()).get("signature") == sig:
                issues = dependency_issues(extras)
                if PROFILE == "linux-dtk":
                    current_versions = installed_versions()
                    issues += dtk_compatibility_issues(current_versions)
                    if vendor_triton and "triton" not in current_versions:
                        issues.append("本地厂商 wheel 集合中的 Triton 尚未安装")
                if not issues:
                    ready = editable_install_ready()
                    if not ready:
                        log("[2/5] 更新训练器安装信息")
                else:
                    log("[2/5] 检测到缺失或不满足版本要求的依赖，自动补齐: " + "; ".join(issues[:8]))
        except Exception:  # noqa: BLE001
            pass
    if ready:
        if PROFILE == "linux-dtk":
            validate_dtk_runtime(torch_runtime())
            validate_dtk_numpy_bridge(installed_versions())
        ensure_model_runtime(extras, index_mode)
        cleanup_build_metadata()
        log("[2/5] 常规依赖已齐全，跳过安装")
        return
    created = not venv_python().exists()
    if PROFILE == "linux-dtk" and not created:
        shared = recorded_vendor_stack()
    # pip is the installer that leaves packages it can see outside the venv alone, so a shared
    # environment uses it even when uv is available.
    uv = None if shared else uv_path()
    if created:
        VENV.parent.mkdir(parents=True, exist_ok=True)
        base = base or find_base_python()
        log(f"[2/5] 创建虚拟环境（{base}）")
        if shared:
            create_dtk_venv(
                base,
                wheelhouse,
                system_site_packages=shared["share"] == "system",
                index_url=index_chains(index_mode, torch_tag)[0][0],
            )
        elif uv:
            run([uv, "venv", "--quiet", "--python", base, str(VENV)])
        elif PROFILE == "linux-dtk":
            # An offline wheel directory brings its own pip; otherwise pip may come from the index.
            create_dtk_venv(
                base, wheelhouse, index_url=None if wheelhouse else index_chains(index_mode, torch_tag)[0][0]
            )
        else:
            run([base, "-m", "venv", str(VENV)])
        # Record ownership before package downloads so an interrupted installation
        # can resume without accepting arbitrary pre-existing environments.
        (VENV / ".ypuddin-owner.json").write_text(
            json.dumps(
                {"profile": PROFILE, "arch": host_arch(), **({"vendor_stack": shared} if shared else {})}
            ),
            encoding="utf-8",
        )
        if shared and shared["share"] == "link":
            link_vendor_stack(shared["location"])
    else:
        log(f"[2/5] 虚拟环境 {VENV.name}/ 已存在，更新依赖")
    py = str(venv_python())
    versions = installed_versions()
    if shared:
        current = versions.get("torch")
        if created and current != shared["torch"]:
            shutil.rmtree(VENV)
            die(
                f"新环境里没有读到 {shared['python']} 的海光版 PyTorch {shared['torch']}"
                f"（读到 {current or '无'}）；请检查该 Python 的安装，或用 --dtk-wheelhouse 提供 wheel。"
            )
        if not current:
            die(f"共用的 {shared['python']} 里已经没有 PyTorch；请恢复它，或用 --reinstall 重建环境。")
        if created:
            log(f"[3/5] 使用 {shared['python']} 已装的海光版 PyTorch {current}（{shared['location']}）")
        elif current != shared["torch"]:
            log(
                f"[3/5] {shared['python']} 里的 PyTorch 已从 {shared['torch']} 变为 {current}，按现有版本继续"
            )
    if PROFILE == "linux-dtk" and "torch" in versions and "numpy" in versions:
        validate_dtk_numpy_bridge(versions)
    preserved = protected_versions(versions)
    if PROFILE == "linux-dtk" and not wheelhouse and not shared and "torch" not in versions and not downloads:
        # A first install that stopped before PyTorch continues with the same reviewed download.
        downloads = reviewed_dtk_set(probe_dtk_python(py))
    required_native = ["torch", "torchvision"] + (["triton"] if vendor_triton or downloads else [])
    if PROFILE == "linux-dtk" and downloads and not all(name in versions for name in required_native):
        files = fetch_dtk_wheels(downloads, VENV.parent / "vendor-wheels")
        command = [uv, "pip", "install", "--python", py] if uv else [py, "-m", "pip", "install"]
        # Only these reviewed files: their ordinary dependencies come from the package index together
        # with the trainer's, which pins these builds.
        run([*command, "--no-index", "--no-deps", *map(str, files)], env=_env())
        versions = installed_versions()
        changed = [name for name, version in preserved.items() if versions.get(name) != version]
        if changed:
            die("厂商 wheel 安装意外改变了受保护的原生依赖：" + ", ".join(changed) + "；停止安装")
        preserved = protected_versions(versions)
        missing = [name for name in required_native if name not in versions]
        if missing:
            die("厂商 wheel 安装后仍缺少 " + " / ".join(missing) + "；停止安装")
    elif PROFILE == "linux-dtk" and not all(name in versions for name in required_native):
        if not wheelhouse:
            die("DTK 独立环境缺少厂商 torch / torchvision，请用 --dtk-wheelhouse 指定本地 wheel 目录")
        # Native vendor packages and their dependencies must all come from this exact, explicit collection.
        # No CUDA/CPU fallback and no system package or driver installation.
        command = [uv, "pip", "install", "--python", py] if uv else [py, "-m", "pip", "install"]
        command += ["--no-index", "--only-binary=:all:", "--find-links", str(wheelhouse)]
        with tempfile.TemporaryDirectory(prefix="ypuddin-dtk-constraints-") as directory:
            constraints = Path(directory) / "native-stack.txt"
            constraints.write_text(
                "".join(f"{name}=={version}\n" for name, version in sorted(preserved.items()))
            )
            requirements = ["torch>=2.4", "torchvision>=0.19"] + (["triton"] if vendor_triton else [])
            compatibility = dtk_native_numpy_constraints(
                wheelhouse, versions, uv=uv, py=py, constraints=constraints, requirements=requirements
            )
            with constraints.open("a", encoding="utf-8") as stream:
                stream.write("".join(value + "\n" for value in compatibility))
            run([*command, "--constraint", str(constraints), *requirements], env=_env())
        versions = installed_versions()
        changed = [name for name, version in preserved.items() if versions.get(name) != version]
        if changed:
            die("厂商 wheel 安装意外改变了受保护的原生依赖：" + ", ".join(changed) + "；停止安装")
        preserved = protected_versions(versions)
        missing = [name for name in required_native if name not in versions]
        if missing:
            die("厂商 wheel 安装后仍缺少 " + " / ".join(missing) + "；停止安装")
    if "torch" in versions:
        # Updating the launcher/dependencies must not replace a working CUDA/MPS
        # build merely because auto-detection now selects a newer wheel channel.
        try:
            current = torch_runtime()
        except Exception as exc:
            die(f"已有 PyTorch 无法加载：{exc}。请运行 doctor 检查，需要重建时使用 --reinstall。")
        if tuple(int(n) for n in re.findall(r"\d+", current["version"])[:2]) < (2, 4):
            die("已有 PyTorch 低于 2.4，请使用 --reinstall 重建环境。")
        if PROFILE == "linux-dtk":
            validate_dtk_runtime(current)
        elif PROFILE != "legacy" and (
            bool(current["cuda"]) != PROFILE.endswith("-cuda") or current.get("hip")
        ):
            die(f"已有 PyTorch 与 {PROFILE} 平台不符，请检查是否手动混装过包。")
        backend = (
            f"HIP {current['hip']}" if current.get("hip") else f"CUDA {current['cuda'] or '无（CPU/MPS）'}"
        )
        log(f"[3/5] 保留现有 PyTorch {current['version']} / {backend}")
    pypi_chain, torch_sources = index_chains(index_mode, torch_tag)

    env = _env()
    if uv and needs_copy_link_mode(st_dev_of(uv_cache_dir(uv) or VENV), st_dev_of(ROOT)):
        # uv cache and project are on different filesystems: copy instead of hardlinking, no warning
        env["UV_LINK_MODE"] = "copy"

    def attempt(cmd: list[str]) -> bool:
        if subprocess.run(cmd, env=env).returncode == 0:
            return True
        log("  命令失败：" + " ".join(cmd))
        return False

    def pip_cmd(
        args: list[str], index_url: str | None, *, upgrade: bool = False, extra: list[str] | None = None
    ) -> list[str]:
        if uv:
            cmd = [uv, "pip", "install", "--python", py]
        else:
            cmd = [py, "-m", "pip", "install"] + (["--upgrade"] if upgrade else [])
        cmd += [*args]
        if index_url:
            cmd += ["--index-url", index_url]
        return cmd + (extra or [])

    def pip_install(args: list[str], what: str) -> None:
        for i, index_url in enumerate(pypi_chain):
            if attempt(pip_cmd(args, index_url)):
                return
            log(
                f"  {what}：包源 {index_url} 失败（缺包或网络错误）"
                + ("，换下一个源重试" if i + 1 < len(pypi_chain) else "")
            )
        die(f"所有包源都无法安装 {what}，请检查网络后重试（可加 --index=cn 或 --index=official）")

    def torch_install() -> None:
        for i, (kind, url) in enumerate(torch_sources):
            if kind == "index-url":
                ok = attempt(pip_cmd(["torch>=2.4", "torchvision>=0.19"], url))
            else:
                # flat wheel listing: take the CUDA wheel from the listing only (PyPI's newer plain build would
                # win otherwise -- CPU-only on Windows), then let the PyPI chain fill in its dependencies
                ok = attempt(
                    pip_cmd(
                        ["torch>=2.4", "torchvision>=0.19"],
                        None,
                        extra=["--no-index", "--no-deps", "--find-links", url],
                    )
                ) and attempt(pip_cmd(["torch>=2.4", "torchvision>=0.19"], pypi_chain[0], upgrade=False))
            if ok:
                return
            log(f"  PyTorch：来源 {url} 失败" + ("，换下一个来源重试" if i + 1 < len(torch_sources) else ""))
        die("所有来源都无法安装 PyTorch，请检查网络或用 --torch= 指定版本后重试")

    if not uv:
        log("[3/5] 检查 pip / wheel")
        pip_install(["pip", "wheel"], "pip/wheel")
    if "torch" not in versions:
        if platform.system() == "Darwin":
            log("[3/5] 安装 PyTorch（macOS）")
            pip_install(["torch>=2.4", "torchvision>=0.19"], "torch / torchvision")
        else:
            log(f"[3/5] 安装 PyTorch（{torch_tag}），首次下载可能需要几分钟")
            torch_install()
        current = torch_runtime()
        expected_cuda = None if torch_tag == "cpu" else f"{torch_tag[2:-1]}.{torch_tag[-1]}"
        if current["cuda"] != expected_cuda:
            die(
                f"新装 PyTorch CUDA 类型不匹配（期望 {expected_cuda}，实际 {current['cuda']}），请检查包源后重建。"
            )
        preserved = protected_versions(installed_versions())
    if "torch" not in preserved:
        die("无法确认已安装 PyTorch 的版本，停止安装。")
    numpy_release = tuple(int(n) for n in re.findall(r"\d+", versions.get("numpy", ""))[:2])
    upgrade_shared_numpy = PROFILE == "linux-dtk" and bool(shared) and (1, 0) <= numpy_release < (1, 26)
    compatibility = dtk_compatibility_constraints(versions)
    if upgrade_shared_numpy:
        # pip shadows an external package inside this venv without uninstalling the host copy.
        preserved.pop("numpy", None)
        if "numpy>=1.26,<2" not in compatibility:
            compatibility.append("numpy>=1.26,<2")
        log(f"[4/5] 在项目虚拟环境中更新 NumPy {versions['numpy']} 至 >=1.26,<2，保留宿主安装")
    log("[4/5] 安装训练器依赖")
    # Exact constraints apply to the whole dependency resolution, not just the
    # explicitly requested torch package. Conflicting accelerators fail before
    # pip can silently swap the existing CUDA stack for a different build.
    with tempfile.TemporaryDirectory(prefix="ypuddin-bootstrap-constraints-") as directory:
        constraints = Path(directory) / "native-stack.txt"
        constraints.write_text(
            "".join(f"{name}=={version}\n" for name, version in sorted(preserved.items()))
            + "".join(value + "\n" for value in compatibility),
            encoding="utf-8",
        )
        pip_install(
            ["--constraint", str(constraints), "-e", f"{ROOT}[{extras}]"],
            f"ypuddin[{extras}]",
        )
    after = installed_versions()
    changed = [name for name, version in preserved.items() if after.get(name) != version]
    if changed:
        die("安装器意外改变了受保护的原生依赖：" + ", ".join(changed) + "，请检查环境。")
    if upgrade_shared_numpy:
        numpy_release = tuple(int(n) for n in re.findall(r"\d+", after.get("numpy", ""))[:2])
        if not (1, 26) <= numpy_release < (2, 0):
            die(f"项目虚拟环境需要 NumPy >=1.26,<2，安装后读到 {after.get('numpy') or '未安装'}；停止启动。")
    issues = dependency_issues(extras) + dtk_compatibility_issues(after)
    if issues:
        die("安装后依赖仍不完整：" + "; ".join(issues[:12]) + "；下次启动会重试补齐。")
    if PROFILE == "linux-dtk":
        validate_dtk_numpy_bridge(after)
    ensure_model_runtime(extras, index_mode)
    if not editable_install_ready():
        die("训练器安装校验失败（venv 中的可编辑安装信息不完整），下次启动会重试。")
    cleanup_build_metadata()
    MARKER.write_text(
        json.dumps(
            {
                "signature": sig,
                "profile": PROFILE,
                "arch": host_arch(),
                "torch": torch_tag,
                "torch_version": after["torch"],
                "extras": extras,
                "time": time.time(),
            }
        )
    )


def choose_extras(torch_tag: str) -> str:
    extras = EXTRAS_BASE
    if PROFILE == "linux-dtk":
        extras += ",dtk"
    if (
        not PROFILE.endswith("-cpu")
        and PROFILE != "linux-dtk"
        and platform.system() in {"Windows", "Linux"}
        and (torch_tag != "cpu" or nvidia_driver_major() is not None)
    ):
        extras += ",nvidia"
    return extras


# --------------------------------------------------------------------------- frontend
def frontend_stale() -> bool:
    dist = FRONTEND / "dist" / "index.html"
    if not dist.exists():
        return True
    try:
        manifest = json.loads((dist.parent / ".source-manifest.json").read_text(encoding="utf-8"))
        inputs = [
            p for directory in ("src", "public") for p in (FRONTEND / directory).rglob("*") if p.is_file()
        ]
        pattern = r"^(package(?:-lock)?\.json|index\.html|buildFingerprint\.ts|(?:vite|tailwind|postcss)\.config\.[^/]+|tsconfig[^/]*\.json|\.env(?:\..*)?)$"
        inputs.extend(p for p in FRONTEND.iterdir() if p.is_file() and re.match(pattern, p.name))
        current = {
            p.relative_to(FRONTEND).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        }
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != 1
            or current != manifest.get("inputs")
        ):
            return True
        outputs = manifest.get("outputs", {})
        if not isinstance(outputs, dict) or "index.html" not in outputs:
            return True
        for name, digest in outputs.items():
            output = (dist.parent / name).resolve()
            if (
                dist.parent.resolve() not in output.parents
                or hashlib.sha256(output.read_bytes()).hexdigest() != digest
            ):
                return True
        return False
    except (OSError, ValueError, TypeError):
        return True


def build_frontend(force: bool = False) -> bool:
    if not force and not frontend_stale():
        log("[5/5] 前端构建已是最新，跳过")
        return True
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        if (FRONTEND / "dist" / "index.html").exists():
            die(
                "前端与当前源码不一致。请安装 Node.js 20.19+ / 22.12+ 后重启，或使用包含最新前端的完整发布包。"
            )
        log("[5/5] 没有找到 Node.js / npm，跳过前端构建（需要 Node.js 20.19+ 或 22.12+）")
        return False
    node = shutil.which("node")
    if not node:
        die("前端构建需要 Node.js 20.19+ 或 22.12+，请安装后重试")
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if not node_supported(version):
        die(f"Node.js {version} 不支持当前前端；需要 20.19+ 或 22.12+")
    lock = FRONTEND / "package-lock.json"
    log("[5/5] 构建前端界面")
    run([npm, "ci" if lock.exists() else "install", "--no-audit", "--no-fund"], cwd=FRONTEND)
    run([npm, "run", "build"], cwd=FRONTEND)
    return True


def node_supported(version: str) -> bool:
    match = re.match(r"v?(\d+)\.(\d+)", version)
    if not match:
        return False
    major, minor = map(int, match.groups())
    return (major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major >= 23


def server_address(host: str | None, port: int | None, data_root: str) -> tuple[str, int]:
    """Explicit flags win; otherwise use the settings saved by the UI."""
    saved = {}
    path = Path(data_root).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    try:
        saved = json.loads((path / "settings.json").read_text(encoding="utf-8")).get("server", {})
    except (OSError, ValueError):
        pass
    return host or saved.get("host", "127.0.0.1"), int(port or saved.get("port", 8123))


def configured_data_root(data_root: str) -> str:
    """Use a data root saved by the UI on the next launcher start."""
    path = Path(data_root).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    try:
        value = json.loads((path / "settings.json").read_text(encoding="utf-8")).get("_pending_data_root")
    except (OSError, ValueError, TypeError):
        value = None
    return str(Path(value).expanduser()) if isinstance(value, str) and value.strip() else data_root


# --------------------------------------------------------------------------- service
def wait_for(url: str, timeout: float = 60) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def serve(host: str | None, port: int | None, data_root: str, open_browser: bool) -> int:
    data_root = configured_data_root(data_root)
    host, port = server_address(host, port, data_root)
    ypuddin = venv_bin("ypuddin")
    cmd = [str(ypuddin), "serve", "--host", host, "--port", str(port), "--data-root", data_root]
    log(f"启动服务（数据目录 {data_root}）")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=_env())
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    if wait_for(url + "api/health"):
        log(f"服务已就绪：{url}（按 Ctrl+C 停止）")
        if not (FRONTEND / "dist" / "index.html").exists():
            log("前端未构建，当前只提供 API")
        if open_browser:
            webbrowser.open(url)
    else:
        log("服务 60 秒内没有响应，请查看上面的输出")
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()


def dev(host: str | None, port: int | None, data_root: str, fe_port: int, open_browser: bool) -> int:
    data_root = configured_data_root(data_root)
    host, port = server_address(host, port, data_root)
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        die("dev 模式需要 Node.js / npm")
    if not (FRONTEND / "node_modules").exists():
        run([npm, "ci", "--no-audit"], cwd=FRONTEND)
    backend = subprocess.Popen(
        [str(venv_bin("ypuddin")), "serve", "--host", host, "--port", str(port), "--data-root", data_root],
        cwd=ROOT,
        env=_env(),
    )
    env = dict(_env(), VITE_USE_MOCK="false", VITE_BACKEND_URL=f"http://127.0.0.1:{port}")
    fe = subprocess.Popen([npm, "run", "dev", "--", "--port", str(fe_port)], cwd=FRONTEND, env=env)
    if open_browser and wait_for(f"http://127.0.0.1:{fe_port}/", 60):
        webbrowser.open(f"http://127.0.0.1:{fe_port}/")
    try:
        while backend.poll() is None and fe.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    for p in (fe, backend):
        if p.poll() is None:
            p.terminate()
    return 0


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if PACKAGE_CACHE_ROOT:
        package_cache = Path(PACKAGE_CACHE_ROOT).expanduser().absolute() / "packages" / PROFILE
        env["PIP_CACHE_DIR"] = str(package_cache)
        env["UV_CACHE_DIR"] = str(package_cache / "uv")
    env["YPUDDIN_ENV_PROFILE"] = PROFILE
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    if PROFILE.endswith("-cpu"):
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def doctor() -> int:
    py = venv_python()
    print(f"项目目录   : {ROOT}")
    print(f"部署环境   : {PROFILE}")
    print(f"系统       : {platform.platform()}")
    print(f"虚拟环境   : {'已创建' if py.exists() else '未创建'} ({VENV})")
    if MARKER.exists():
        print(f"上次安装   : {MARKER.read_text().strip()}")
    nvidia = PROFILE != "linux-dtk" and not PROFILE.endswith("-cpu") and platform.system() != "Darwin"
    driver = nvidia_driver_major() if nvidia else None
    gpus = nvidia_gpus() if nvidia else []
    if PROFILE == "linux-dtk":
        shared = recorded_vendor_stack()
        print(
            f"DTK        : 共用 {shared['python']} 已装的厂商 HIP PyTorch（{shared['location']}）"
            if shared
            else "DTK        : 使用独立环境中的厂商 HIP PyTorch"
        )
    elif nvidia:
        print(
            f"NVIDIA     : {'驱动 ' + str(driver) if driver else '未找到 nvidia-smi'} -> 自动选择的 PyTorch 类型 {pick_torch_tag('auto')}"
        )
    for name, cc in gpus:
        print(f"  显卡       : {name}（计算能力 {cc}）")
    if py.exists():
        code = (
            "import torch,json;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'hip':getattr(torch.version,'hip',None),"
            "'cuda_available':torch.cuda.is_available(),"
            "'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
            "'arch_list':torch.cuda.get_arch_list() if torch.cuda.is_available() else [],"
            "'mps':bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())}))"
        )
        r = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
        print(f"PyTorch    : {r.stdout.strip() or r.stderr.strip()[-300:]}")
        try:
            archs = set(json.loads(r.stdout).get("arch_list", []))
            for name, cc in gpus:
                if cc is not None and archs and f"sm_{int(round(cc * 10))}" not in archs:
                    print(
                        f"  警告       : 已安装的 PyTorch 不含 {name}（sm_{int(round(cc * 10))}）的内核，"
                        "请用本环境的启动脚本运行 --reinstall --torch=cu128"
                    )
        except Exception:  # noqa: BLE001
            pass
        versions = installed_versions()
        for name in (
            "transformers",
            "fastapi",
            "schedulefree",
            "lion-pytorch",
            "prodigyopt",
            "prodigy-plus-schedule-free",
            "pytorch-optimizer",
            "tensorboard",
        ):
            print(f"  {name:<28}: {versions.get(name, '缺失，下次启动自动补齐')}")
        if platform.system() in {"Windows", "Linux"} and driver is not None:
            print(f"  {'nvidia-ml-py':<28}: {versions.get('nvidia-ml-py', '缺失，下次启动自动补齐')}")
        if PROFILE == "macos-mps":
            optional = ("mtlattn",)
        elif PROFILE == "linux-dtk":
            optional = ("xformers", "flash-attn", "bitsandbytes")
        elif driver is not None:
            optional = ("xformers", "flash-attn", "sageattention", "bitsandbytes")
        else:
            optional = ()
        for name in optional:
            print(f"  {name:<28}: {versions.get(name, '未安装（可选）')}")
        issues = dependency_issues(choose_extras(pick_torch_tag("auto")))
        print("依赖完整性 : " + ("通过" if not issues else "; ".join(issues[:12])))
        try:
            native = model_runtime()
            print(
                "模型依赖   : "
                + ("TorchVision 原生算子、Qwen3 导入通过" if native.get("ok") else str(native))
            )
        except Exception as exc:
            print(f"模型依赖   : {exc}")
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    print(f"Node/npm   : {shutil.which('node') or '未安装'} / {npm or '未安装'}")
    print(
        f"前端构建   : {'已构建' if (FRONTEND / 'dist' / 'index.html').exists() else '未构建'}"
        + (
            "（源码有更新，下次 run 会重建）"
            if (FRONTEND / "dist" / "index.html").exists() and frontend_stale()
            else ""
        )
    )
    return 0


# --------------------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    global DOWNLOAD_SETTINGS, PACKAGE_CACHE_ROOT
    DOWNLOAD_SETTINGS = None
    opts = {
        "torch": "auto",
        "profile": "auto",
        "index": "auto",
        "reinstall": False,
        "browser": True,
        "frontend": True,
        "host": None,
        "port": None,
        "data_root": "studio_data",
        "fe_port": "3000",
        "dtk_wheelhouse": None,
        "env_root": None,
    }
    rest: list[str] = []
    it = iter(argv)
    for a in it:
        if a.startswith("--torch="):
            opts["torch"] = a.split("=", 1)[1]
        elif a.startswith("--profile="):
            opts["profile"] = a.split("=", 1)[1]
        elif a.startswith("--env-root="):
            opts["env_root"] = a.split("=", 1)[1]
            if not opts["env_root"].strip():
                die("--env-root 不能为空")
        elif a.startswith("--dtk-wheelhouse="):
            opts["dtk_wheelhouse"] = a.split("=", 1)[1]
        elif a == "--mirror":
            opts["index"] = "cn"
        elif a.startswith("--index="):
            opts["index"] = a.split("=", 1)[1]
        elif a == "--reinstall":
            opts["reinstall"] = True
        elif a == "--no-browser":
            opts["browser"] = False
        elif a == "--no-frontend":
            opts["frontend"] = False
        elif a in ("--host", "--port", "--data-root", "--fe-port", "--env-root"):
            opts[a[2:].replace("-", "_")] = next(it, opts[a[2:].replace("-", "_")])
        else:
            rest.append(a)
    command = rest[0] if rest and not rest[0].startswith("-") else "run"
    passthrough = rest[1:] if rest and not rest[0].startswith("-") else rest
    if opts["torch"] not in ("auto", "cpu", "dtk", *[t for t, _ in CUDA_TAGS]):
        die(f"--torch 取值无效：{opts['torch']!r}（可选 auto/cpu/dtk/cu128/cu126/cu124/cu118）")
    if opts["index"] not in ("auto", "cn", "official"):
        die(f"--index 取值无效：{opts['index']!r}（可选 auto/cn/official）")

    if any(a == "--env-root" for a in argv) and not (opts["env_root"] or "").strip():
        die("--env-root 不能为空")
    settings_file = Path(opts["data_root"]).expanduser() / "settings.json"
    saved_settings = json.loads(settings_file.read_text(encoding="utf-8")) if settings_file.exists() else {}
    opts["browser"] = opts["browser"] and saved_settings.get("server", {}).get("open_browser", True)
    if not any(a == "--mirror" or a.startswith("--index=") for a in argv):
        DOWNLOAD_SETTINGS = saved_settings.get("downloads")
    configured_cache = saved_settings.get("paths", {}).get("cache_dir")
    default_cache = Path(opts["data_root"]).expanduser().absolute() / "cache"
    PACKAGE_CACHE_ROOT = (
        configured_cache
        if configured_cache and Path(configured_cache).expanduser().absolute() != default_cache
        else None
    )
    env_root = opts["env_root"] or saved_settings.get("paths", {}).get("bootstrap_env_dir")
    torch_tag = platform_torch_tag(opts["profile"], opts["torch"])
    if env_root:
        select_environment(opts["profile"], torch_tag, env_root=env_root)
    else:
        select_environment(opts["profile"], torch_tag)
    if command == "doctor":
        return doctor()
    if command == "test" and not (ROOT / "Test").is_dir():
        die("未找到本地测试目录 Test/；测试只保存在开发机上，不随仓库发布。")
    if opts["reinstall"] and Path(sys.prefix).resolve() == VENV.resolve():
        # In Windows the interpreter executing this file cannot delete itself.
        base = getattr(sys, "_base_executable", None)
        if not base or Path(base).resolve().is_relative_to(VENV.resolve()):
            die("请用系统 Python 运行 --reinstall：当前解释器位于待重建的环境内。")
        return subprocess.run([base, str(Path(__file__).resolve()), *argv], cwd=ROOT, env=_env()).returncode
    extras = choose_extras(torch_tag) + (",dev" if command == "test" else "")
    if PROFILE == "linux-dtk":
        gpu_desc = "厂商 DTK / HIP 环境"
    elif PROFILE == "macos-mps":
        gpu_desc = "Apple GPU（MPS）"
    elif PROFILE.endswith("-cpu"):
        gpu_desc = "不使用"
    else:
        gpus = nvidia_gpus()
        gpu_desc = "、".join(n for n, _ in gpus) if gpus else "未检测到 NVIDIA 显卡"
    torch_label = "MPS" if PROFILE == "macos-mps" else torch_tag
    log(
        f"[1/5] 环境检查：Python {platform.python_version()} · 显卡：{gpu_desc} · PyTorch：{torch_label}"
    )
    ensure_venv(
        torch_tag,
        index_mode=opts["index"],
        reinstall=opts["reinstall"],
        extras=extras,
        **({"dtk_wheelhouse": opts["dtk_wheelhouse"]} if opts["dtk_wheelhouse"] else {}),
    )

    if command == "build":
        return 0 if build_frontend(force=True) else 1
    if command == "test":
        rc = subprocess.run([str(venv_bin("pytest")), "-q"], cwd=ROOT).returncode
        npx = shutil.which("npx") or shutil.which("npx.cmd")
        if npx and (FRONTEND / "node_modules").exists():
            config = ROOT / "Test/frontend/vitest.config.ts"
            rc |= subprocess.run(
                [npx, "--no-install", "vitest", "run", "--config", str(config)], cwd=FRONTEND
            ).returncode
        return rc
    if command == "smoke":
        return subprocess.run(
            [str(venv_bin("ypuddin")), "smoke", *passthrough], cwd=ROOT, env=_env()
        ).returncode
    if command == "shell":
        act = VENV / ("Scripts/activate" if WIN else "bin/activate")
        print(f"激活虚拟环境：{act}" if WIN else f"激活虚拟环境：source {act}")
        return 0
    if command == "dev":
        return dev(
            opts["host"],
            int(opts["port"]) if opts["port"] else None,
            opts["data_root"],
            int(opts["fe_port"]),
            opts["browser"],
        )
    if command == "run":
        if opts["frontend"]:
            build_frontend()
        return serve(
            opts["host"], int(opts["port"]) if opts["port"] else None, opts["data_root"], opts["browser"]
        )
    die(f"未知命令 {command!r}（可选 run/dev/build/test/smoke/doctor/shell）")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
