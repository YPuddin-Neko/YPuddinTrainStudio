"""YPuddin Train Studio 一键引导脚本（只用标准库，裸系统 Python 即可运行）。

由 ``studio.sh`` / ``studio.bat`` 调用::

    python scripts/bootstrap.py [全局参数] [命令] [命令参数]

命令
  run     （默认）创建/更新 venv、按需构建前端、启动服务、打开浏览器
  dev     同时启动后端与 Vite 热更新前端
  build   只构建前端
  test    跑 pytest（有 Node 时再跑 vitest）
  smoke   转发到 ``ypuddin smoke``（在本机用真实权重跑几步训练自检，见 docs/deploy.md）
  doctor  打印本机情况：Python、torch/CUDA/显卡、可选依赖、Node、前端构建状态
  shell   打印如何激活 venv

全局参数
  --torch=<cu128|cu126|cu124|cu118|cpu|auto>  首次安装/重建的 PyTorch 类型（默认 auto）
  --index=<auto|cn|official>  包源。auto / cn（默认）：国内镜像优先，中科大 -> 清华 -> 阿里 -> 官方兜底，
                  探测不通的源自动排后，逐个尝试直到成功；official：官方源优先（镜像兜底）
  --mirror        等价于 --index=cn
  --reinstall     删掉 venv 重装（studio_data/ 不受影响）
  --no-browser    服务起来后不自动打开浏览器
  --no-frontend   跳过前端构建（只要 API）
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "venv"
FRONTEND = ROOT / "frontend"
MARKER = VENV / ".ypuddin-install.json"
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
    ("find-links", "https://mirrors.aliyun.com/pytorch-wheels/{tag}"),
    ("index-url", "https://mirror.sjtu.edu.cn/pytorch-wheels/{tag}"),
)
# minimum NVIDIA driver (major) able to run each CUDA wheel flavour, newest first
CUDA_TAGS = (("cu128", 570), ("cu126", 560), ("cu124", 550), ("cu118", 450))
# GPUs with compute capability >= 12.0 (RTX 50 series / Blackwell) only have kernels in the cu128+ wheels
BLACKWELL_CC = 12.0
EXTRAS_BASE = "models,server,optim,logging"


def log(msg: str) -> None:
    print(f"[studio] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[studio] 错误：{msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("  执行: " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


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


def url_ok(url: str, timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def order_by_reachability(indexes: list[str]) -> list[str]:
    """Indexes that answer a quick probe first (original order kept within each group), so a dead
    mirror costs one 4 s probe instead of a full pip timeout per package."""
    alive = [u for u in indexes if url_ok(u.rstrip("/") + "/pip/")]
    dead = [u for u in indexes if u not in alive]
    if dead:
        log("以下包源探测不通，排到最后再试: " + ", ".join(dead))
    return alive + dead


def index_chains(mode: str, torch_tag: str) -> tuple[list[str], list[tuple[str, str]]]:
    """(PyPI index urls in order, torch sources as (kind, url) with kind in {find-links, index-url}).

    auto/cn: mirrors first (USTC -> Tsinghua -> Aliyun), official PyPI as the last resort.
    official: official first, mirrors as the fallback.
    """
    if mode == "official":
        pypi = [PYPI_OFFICIAL, *PYPI_MIRRORS_CN]  # mirrors still serve as a fallback
        torch_src = [("index-url", TORCH_OFFICIAL.format(tag=torch_tag))]
    else:
        pypi = order_by_reachability([*PYPI_MIRRORS_CN, PYPI_OFFICIAL])
        torch_src = [(kind, url.format(tag=torch_tag)) for kind, url in TORCH_MIRRORS_CN] + [
            ("index-url", TORCH_OFFICIAL.format(tag=torch_tag))
        ]
    if mode == "auto":
        log("包源选择: 国内镜像优先（中科大 → 清华 → 阿里 → 官方兜底；探测不通的自动排后）")
    return pypi, torch_src


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
    h.update(f"{torch_tag}|{extras}|{sys.platform}|{platform.machine()}|{sys.version_info[:2]}".encode())
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
        )
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or "environment probe failed")[-1600:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def installed_versions() -> dict[str, str]:
    return venv_json(
        "import importlib.metadata as m,json,re; "
        "print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version "
        "for d in m.distributions() if d.metadata['Name']}))"
    )


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
    except OSError as exc:
        log(f"  未能检查 ypuddin.egg-info（{exc}），下次启动会重试。")
        return
    try:
        # lstat also detects Windows junctions on supported Python versions.
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            log("  未清理 ypuddin.egg-info：此路径不是普通目录，请检查链接或同名文件。")
            return
        shutil.rmtree(directory)
        log("  已清理根目录 ypuddin.egg-info；运行所需安装信息保留在 venv 中。")
    except OSError as exc:
        # An occupied Windows file must not trigger another dependency install.
        log(f"  未能清理 ypuddin.egg-info（{exc}），下次启动会重试；训练环境已保留。")


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
        or name.startswith("nvidia-")
        and name != "nvidia-ml-py"
    }


def torch_runtime() -> dict:
    return venv_json(
        "import torch,json; print(json.dumps({'version':torch.__version__,'cuda':torch.version.cuda}))"
    )


def ensure_venv(torch_tag: str, *, index_mode: str, reinstall: bool, extras: str) -> None:
    if reinstall and VENV.exists():
        log(f"--reinstall：删除旧的虚拟环境 {VENV}（studio_data/ 不受影响）")
        shutil.rmtree(VENV)
    sig = install_signature(torch_tag, extras)
    ready = False
    if venv_python().exists() and MARKER.exists():
        try:
            if json.loads(MARKER.read_text()).get("signature") == sig:
                issues = dependency_issues(extras)
                if not issues:
                    ready = editable_install_ready()
                    if not ready:
                        log("[2/5] 更新训练器安装信息（保留现有环境和依赖）")
                else:
                    log("[2/5] 检测到缺失或不满足版本要求的依赖，自动补齐: " + "; ".join(issues[:8]))
        except Exception:  # noqa: BLE001
            pass
    if ready:
        cleanup_build_metadata()
        log("[2/5] 常规依赖已齐全，跳过安装")
        return
    uv = uv_path()
    if not venv_python().exists():
        base = find_base_python()
        log(f"[2/5] 创建虚拟环境 {VENV.name}/（基于 {base}，用 {'uv' if uv else 'python -m venv'}）")
        if uv:
            run([uv, "venv", "--python", base, str(VENV)])
        else:
            run([base, "-m", "venv", str(VENV)])
    else:
        log(f"[2/5] 虚拟环境 {VENV.name}/ 已存在，更新依赖")
    py = str(venv_python())
    versions = installed_versions()
    preserved = protected_versions(versions)
    if "torch" in versions:
        # Updating the launcher/dependencies must not replace a working CUDA/MPS
        # build merely because auto-detection now selects a newer wheel channel.
        try:
            current = torch_runtime()
        except Exception as exc:
            die(f"已有 PyTorch 无法加载，未修改环境：{exc}。请运行 doctor 检查；需要重建时使用 --reinstall。")
        if tuple(int(n) for n in re.findall(r"\d+", current["version"])[:2]) < (2, 4):
            die("已有 PyTorch 低于 2.4，自动补依赖不会替换它；请使用 --reinstall 重建环境。")
        log(f"[3/5] 保留现有 PyTorch {current['version']} / CUDA {current['cuda'] or '无（CPU/MPS）'}")
    pypi_chain, torch_sources = index_chains(index_mode, torch_tag)

    env = dict(os.environ)
    if uv and needs_copy_link_mode(st_dev_of(uv_cache_dir(uv) or VENV), st_dev_of(ROOT)):
        # uv cache and project are on different filesystems: copy instead of hardlinking, no warning
        env["UV_LINK_MODE"] = "copy"
        log("  uv 缓存与项目不在同一文件系统，用复制模式安装（UV_LINK_MODE=copy）")

    def attempt(cmd: list[str]) -> bool:
        log("  执行: " + " ".join(cmd))
        return subprocess.run(cmd, env=env).returncode == 0

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
                ok = attempt(pip_cmd(["torch>=2.4"], url))
            else:
                # flat wheel listing: take the CUDA wheel from the listing only (PyPI's newer plain build would
                # win otherwise -- CPU-only on Windows), then let the PyPI chain fill in its dependencies
                ok = attempt(
                    pip_cmd(["torch>=2.4"], None, extra=["--no-index", "--no-deps", "--find-links", url])
                ) and attempt(pip_cmd(["torch>=2.4"], pypi_chain[0], upgrade=False))
            if ok:
                return
            log(f"  PyTorch：来源 {url} 失败" + ("，换下一个来源重试" if i + 1 < len(torch_sources) else ""))
        die("所有来源都无法安装 PyTorch，请检查网络或用 --torch= 指定版本后重试")

    if not uv:
        log("[3/5] 检查 pip / wheel")
        pip_install(["pip", "wheel"], "pip/wheel")
    if "torch" not in versions:
        if platform.system() == "Darwin":
            log("[3/5] 安装 Apple PyTorch（PyPI wheel 包含 MPS 支持）")
            pip_install(["torch>=2.4"], "torch")
        else:
            log(f"[3/5] 安装 PyTorch（{torch_tag}，CUDA 版本下载较大，请耐心等待）")
            torch_install()
        current = torch_runtime()
        expected_cuda = None if torch_tag == "cpu" else f"{torch_tag[2:-1]}.{torch_tag[-1]}"
        if current["cuda"] != expected_cuda:
            die(
                f"新装 PyTorch CUDA 类型不匹配（期望 {expected_cuda}，实际 {current['cuda']}），请检查包源后重建。"
            )
        preserved = protected_versions(installed_versions())
    if "torch" not in preserved:
        die("无法确认已安装 PyTorch 的版本，停止安装以保护训练环境。")
    log(f"[4/5] 安装训练器 ypuddin 及其依赖 [{extras}]")
    # Exact constraints apply to the whole dependency resolution, not just the
    # explicitly requested torch package. Conflicting accelerators fail before
    # pip can silently swap the existing CUDA stack for a different build.
    with tempfile.TemporaryDirectory(prefix="ypuddin-bootstrap-constraints-") as directory:
        constraints = Path(directory) / "native-stack.txt"
        constraints.write_text(
            "".join(f"{name}=={version}\n" for name, version in sorted(preserved.items())), encoding="utf-8"
        )
        pip_install(
            ["--constraint", str(constraints), "-e", f"{ROOT}[{extras}]"],
            f"ypuddin[{extras}]（保留现有 PyTorch/CUDA）",
        )
    after = installed_versions()
    changed = [name for name, version in preserved.items() if after.get(name) != version]
    if changed:
        die("安装器意外改变了受保护的原生依赖：" + ", ".join(changed) + "；未写入成功标记，请检查环境。")
    issues = dependency_issues(extras)
    if issues:
        die("安装后依赖仍不完整：" + "; ".join(issues[:12]) + "；下次启动会重试补齐。")
    if not editable_install_ready():
        die("未能验证 venv 中训练器的独立安装信息；保留根目录元数据，未写入成功标记。")
    cleanup_build_metadata()
    MARKER.write_text(
        json.dumps(
            {
                "signature": sig,
                "torch": torch_tag,
                "torch_version": after["torch"],
                "extras": extras,
                "time": time.time(),
            }
        )
    )


def choose_extras(torch_tag: str) -> str:
    extras = EXTRAS_BASE
    if platform.system() in {"Windows", "Linux"} and (
        torch_tag != "cpu" or nvidia_driver_major() is not None
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
    log("[5/5] 构建前端：安装 npm 依赖 ...")
    run([npm, "ci" if lock.exists() else "install", "--no-audit"], cwd=FRONTEND)
    log("[5/5] 构建前端：编译打包（tsc + vite build）...")
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
    return host or saved.get("host", "127.0.0.1"), int(port or saved.get("port", 8765))


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
    host, port = server_address(host, port, data_root)
    ypuddin = venv_bin("ypuddin")
    cmd = [str(ypuddin), "serve", "--host", host, "--port", str(port), "--data-root", data_root]
    log(f"启动服务（数据目录 {data_root}）...")
    log("  执行: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=ROOT, env=_env())
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    if wait_for(url + "api/health"):
        log(f"服务已就绪：{url}   （API 文档：{url}api/docs；按 Ctrl+C 停止）")
        if not (FRONTEND / "dist" / "index.html").exists():
            log("提示：前端未构建，当前只提供 API")
        if open_browser:
            log("正在打开浏览器 ...")
            webbrowser.open(url)
    else:
        log("服务 60 秒内没有响应，请查看上面的输出")
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()


def dev(host: str | None, port: int | None, data_root: str, fe_port: int, open_browser: bool) -> int:
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
    return env


def doctor() -> int:
    py = venv_python()
    print(f"项目目录   : {ROOT}")
    print(f"系统       : {platform.platform()}")
    print(f"虚拟环境   : {'已创建' if py.exists() else '未创建'} ({VENV})")
    if MARKER.exists():
        print(f"上次安装   : {MARKER.read_text().strip()}")
    driver = nvidia_driver_major()
    gpus = nvidia_gpus()
    print(
        f"NVIDIA     : {'驱动 ' + str(driver) if driver else '未找到 nvidia-smi'} -> 自动选择的 PyTorch 版本 {pick_torch_tag('auto')}"
    )
    for name, cc in gpus:
        print(f"  显卡       : {name}（计算能力 {cc}）")
    if py.exists():
        code = (
            "import torch,json;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
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
                        "请运行 ./studio.sh --reinstall --torch=cu128"
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
        for name in ("xformers", "flash-attn", "sageattention", "bitsandbytes"):
            print(f"  {name:<28}: {versions.get(name, '未安装（可选，启动器不会自动安装）')}")
        issues = dependency_issues(choose_extras(pick_torch_tag("auto")))
        print("依赖完整性 : " + ("通过" if not issues else "; ".join(issues[:12])))
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
    opts = {
        "torch": "auto",
        "index": "auto",
        "reinstall": False,
        "browser": True,
        "frontend": True,
        "host": None,
        "port": None,
        "data_root": "studio_data",
        "fe_port": "3000",
    }
    rest: list[str] = []
    it = iter(argv)
    for a in it:
        if a.startswith("--torch="):
            opts["torch"] = a.split("=", 1)[1]
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
        elif a in ("--host", "--port", "--data-root", "--fe-port"):
            opts[a[2:].replace("-", "_")] = next(it, opts[a[2:].replace("-", "_")])
        else:
            rest.append(a)
    command = rest[0] if rest and not rest[0].startswith("-") else "run"
    passthrough = rest[1:] if rest and not rest[0].startswith("-") else rest
    if opts["torch"] not in ("auto", "cpu", *[t for t, _ in CUDA_TAGS]):
        die(f"--torch 取值无效：{opts['torch']!r}（可选 auto/cpu/cu128/cu126/cu124/cu118）")
    if opts["index"] not in ("auto", "cn", "official"):
        die(f"--index 取值无效：{opts['index']!r}（可选 auto/cn/official）")

    if command == "doctor":
        return doctor()

    torch_tag = pick_torch_tag(opts["torch"])
    extras = choose_extras(torch_tag) + (",dev" if command == "test" else "")
    gpus = nvidia_gpus()
    gpu_desc = "、".join(n for n, _ in gpus) if gpus else "未检测到 NVIDIA 显卡"
    log(
        f"[1/5] 环境检查：Python {platform.python_version()} · 显卡：{gpu_desc} · PyTorch 版本：{torch_tag} · 安装工具：{'uv' if uv_path() else 'pip'}"
    )
    ensure_venv(torch_tag, index_mode=opts["index"], reinstall=opts["reinstall"], extras=extras)

    if command == "build":
        return 0 if build_frontend(force=True) else 1
    if command == "test":
        rc = subprocess.run([str(venv_bin("pytest")), "-q"], cwd=ROOT).returncode
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if npm and (FRONTEND / "node_modules").exists():
            rc |= subprocess.run([npm, "run", "test"], cwd=FRONTEND).returncode
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
