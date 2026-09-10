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
  --torch=<cu128|cu126|cu124|cu118|cpu|auto>  PyTorch 版本（默认 auto：按显卡计算能力与驱动版本选）
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
import subprocess
import sys
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
EXTRAS_BASE = "models,server"


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
    h.update(f"{torch_tag}|{extras}|{sys.platform}".encode())
    return h.hexdigest()[:16]


def ensure_venv(torch_tag: str, *, index_mode: str, reinstall: bool, extras: str) -> None:
    if reinstall and VENV.exists():
        log(f"--reinstall：删除旧的虚拟环境 {VENV}（studio_data/ 不受影响）")
        shutil.rmtree(VENV)
    sig = install_signature(torch_tag, extras)
    if venv_python().exists() and MARKER.exists():
        try:
            if json.loads(MARKER.read_text()).get("signature") == sig:
                log("[2/5] 依赖已是最新（pyproject.toml 未变化），跳过安装")
                return
        except Exception:  # noqa: BLE001
            pass
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
        args: list[str], index_url: str | None, *, upgrade: bool = True, extra: list[str] | None = None
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
        log("[3/5] 升级 pip / wheel")
        pip_install(["pip", "wheel"], "pip/wheel")
    if torch_tag != "cpu":
        log(f"[3/5] 安装 PyTorch（CUDA 版本 {torch_tag}，约 2.5 GB，耐心等待）")
        torch_install()
    else:
        log("[3/5] 安装 PyTorch（CPU / Apple MPS 版）")
        pip_install(["torch>=2.4"], "torch")
    log(f"[4/5] 安装训练器 ypuddin 及其依赖 [{extras}]")
    pip_install(["-e", f"{ROOT}[{extras}]"], f"ypuddin[{extras}]")
    MARKER.write_text(
        json.dumps({"signature": sig, "torch": torch_tag, "extras": extras, "time": time.time()})
    )


def choose_extras(torch_tag: str) -> str:
    extras = EXTRAS_BASE
    if torch_tag != "cpu" and platform.system() == "Linux":
        extras += ",cuda,optim"  # bitsandbytes wheels are reliable on Linux; Windows users opt in manually
    return extras


# --------------------------------------------------------------------------- frontend
def frontend_stale() -> bool:
    dist = FRONTEND / "dist" / "index.html"
    if not dist.exists():
        return True
    built = dist.stat().st_mtime
    for p in (FRONTEND / "src").rglob("*"):
        if p.is_file() and p.stat().st_mtime > built:
            return True
    for name in ("package.json", "vite.config.ts", "index.html", "tailwind.config.js"):
        f = FRONTEND / name
        if f.exists() and f.stat().st_mtime > built:
            return True
    return False


def build_frontend(force: bool = False) -> bool:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        log(
            "[5/5] 没有找到 Node.js / npm，跳过前端构建（API 仍可用；要用网页界面请安装 Node.js >= 18 后重跑）"
        )
        return False
    if not force and not frontend_stale():
        log("[5/5] 前端构建已是最新，跳过")
        return True
    lock = FRONTEND / "package-lock.json"
    log("[5/5] 构建前端：安装 npm 依赖 ...")
    run([npm, "ci" if lock.exists() else "install"], cwd=FRONTEND)
    log("[5/5] 构建前端：编译打包（tsc + vite build）...")
    run([npm, "run", "build"], cwd=FRONTEND)
    return True


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


def serve(host: str, port: int, data_root: str, open_browser: bool) -> int:
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


def dev(host: str, port: int, data_root: str, fe_port: int, open_browser: bool) -> int:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        die("dev 模式需要 Node.js / npm")
    if not (FRONTEND / "node_modules").exists():
        run([npm, "ci"], cwd=FRONTEND)
    backend = subprocess.Popen(
        [str(venv_bin("ypuddin")), "serve", "--host", host, "--port", str(port), "--data-root", data_root],
        cwd=ROOT,
        env=_env(),
    )
    if port != 8765:
        log("提示：Vite 代理固定指向 127.0.0.1:8765（frontend/vite.config.ts），dev 模式请用默认端口")
    env = dict(_env(), VITE_USE_MOCK="false")
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
        for mod in ("bitsandbytes", "sageattention", "prodigyopt", "transformers", "fastapi"):
            r = subprocess.run(
                [str(py), "-c", f"import {mod};print(getattr({mod},'__version__','ok'))"],
                capture_output=True,
                text=True,
            )
            print(f"  {mod:<14}: {r.stdout.strip() or '未安装'}")
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
        "host": "127.0.0.1",
        "port": "8765",
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
        return dev(opts["host"], int(opts["port"]), opts["data_root"], int(opts["fe_port"]), opts["browser"])
    if command == "run":
        if opts["frontend"]:
            build_frontend()
        return serve(opts["host"], int(opts["port"]), opts["data_root"], opts["browser"])
    die(f"未知命令 {command!r}（可选 run/dev/build/test/smoke/doctor/shell）")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
