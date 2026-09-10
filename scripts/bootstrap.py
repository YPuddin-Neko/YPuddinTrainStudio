"""One-click bootstrap for YPuddin Train Studio (no third-party imports: runs on a bare system Python).

Called by ``studio.sh`` / ``studio.bat``::

    python scripts/bootstrap.py [global flags] [command] [command flags]

Commands
  run     (default) create/refresh the venv, build the frontend if needed, start the service, open the browser
  dev     start the backend and the Vite dev server (hot reload) side by side
  build   build the frontend only
  test    run pytest (+ vitest when node is available)
  smoke   forward to ``ypuddin smoke`` (real training steps on this machine, see docs/deploy.md)
  doctor  print what this machine has: python, torch, CUDA, GPUs, node, frontend build state
  shell   print how to activate the venv

Global flags
  --torch=<cu128|cu126|cu124|cu118|cpu|auto>   PyTorch wheel flavour (default auto: from the NVIDIA driver)
  --index=<auto|cn|official>  package sources. auto (default): probe pypi.org; unreachable/slow -> mainland-China
                  mirror chain USTC -> Tsinghua -> Aliyun -> official, each tried in turn until one succeeds.
                  cn: mirror chain first even if pypi.org answers. official: pypi.org / download.pytorch.org only.
  --mirror        alias for --index=cn
  --reinstall     delete .venv and start over (studio_data/ is never touched)
  --no-browser    do not open the browser after the service is up
  --no-frontend   skip the frontend build (API only)
  --host/--port/--data-root are forwarded to ``ypuddin serve``
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
VENV = ROOT / ".venv"
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
    print(f"[studio] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


# --------------------------------------------------------------------------- environment probes
def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if WIN else "bin/python")


def venv_bin(name: str) -> Path:
    return VENV / ("Scripts" if WIN else "bin") / (name + (".exe" if WIN else ""))


def uv_path() -> str | None:
    return shutil.which("uv")


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
                f"WARNING: driver {driver} is too old for an RTX 50-series GPU; update to >= 570 or CUDA will not work"
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


def pypi_reachable(timeout: float = 4.0) -> bool:
    return url_ok(PYPI_OFFICIAL + "/pip/", timeout)


def order_by_reachability(indexes: list[str]) -> list[str]:
    """Indexes that answer a quick probe first (original order kept within each group), so a dead
    mirror costs one 4 s probe instead of a full pip timeout per package."""
    alive = [u for u in indexes if url_ok(u.rstrip("/") + "/pip/")]
    dead = [u for u in indexes if u not in alive]
    if dead:
        log("unreachable package sources skipped for now: " + ", ".join(dead))
    return alive + dead


def index_chains(mode: str, torch_tag: str) -> tuple[list[str], list[tuple[str, str]]]:
    """(PyPI index urls in order, torch sources as (kind, url) with kind in {find-links, index-url})."""
    if mode == "auto":
        mode = "official" if pypi_reachable() else "cn"
        log(
            f"package sources: {mode} ({'pypi.org answered' if mode == 'official' else 'pypi.org unreachable -> mirrors first'})"
        )
    if mode == "official":
        pypi = [PYPI_OFFICIAL, *PYPI_MIRRORS_CN]  # mirrors still serve as a fallback
        torch_src = [("index-url", TORCH_OFFICIAL.format(tag=torch_tag))]
    else:
        pypi = order_by_reachability([*PYPI_MIRRORS_CN, PYPI_OFFICIAL])
        torch_src = [(kind, url.format(tag=torch_tag)) for kind, url in TORCH_MIRRORS_CN] + [
            ("index-url", TORCH_OFFICIAL.format(tag=torch_tag))
        ]
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
    die("no Python 3.10–3.12 found. Install one (python.org, or `uv python install 3.12`) and re-run.")
    return ""  # unreachable


# --------------------------------------------------------------------------- venv + dependencies
def install_signature(torch_tag: str, extras: str) -> str:
    h = hashlib.sha256((ROOT / "pyproject.toml").read_bytes())
    h.update(f"{torch_tag}|{extras}|{sys.platform}".encode())
    return h.hexdigest()[:16]


def ensure_venv(torch_tag: str, *, index_mode: str, reinstall: bool, extras: str) -> None:
    if reinstall and VENV.exists():
        log(f"--reinstall: removing {VENV} (studio_data/ is kept)")
        shutil.rmtree(VENV)
    sig = install_signature(torch_tag, extras)
    if venv_python().exists() and MARKER.exists():
        try:
            if json.loads(MARKER.read_text()).get("signature") == sig:
                log("dependencies up to date")
                return
        except Exception:  # noqa: BLE001
            pass
    uv = uv_path()
    if not venv_python().exists():
        base = find_base_python()
        if uv:
            run([uv, "venv", "--python", base, str(VENV)])
        else:
            run([base, "-m", "venv", str(VENV)])
    py = str(venv_python())
    pypi_chain, torch_sources = index_chains(index_mode, torch_tag)

    def attempt(cmd: list[str]) -> bool:
        log("$ " + " ".join(cmd))
        return subprocess.run(cmd).returncode == 0

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
                f"{what}: source {index_url} failed"
                + (", trying the next one" if i + 1 < len(pypi_chain) else "")
            )
        die(f"could not install {what} from any package source")

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
            log(
                f"torch: source {url} failed"
                + (", trying the next one" if i + 1 < len(torch_sources) else "")
            )
        die("could not install PyTorch from any source")

    if not uv:
        pip_install(["pip", "wheel"], "pip/wheel")
    if torch_tag != "cpu":
        log(f"installing PyTorch ({torch_tag})")
        torch_install()
    else:
        log("installing PyTorch (cpu / mps)")
        pip_install(["torch>=2.4"], "torch")
    log(f"installing ypuddin[{extras}]")
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
            "node/npm not found: skipping the frontend build (the API still works; install Node.js >= 18 for the UI)"
        )
        return False
    if not force and not frontend_stale():
        log("frontend build is up to date")
        return True
    lock = FRONTEND / "package-lock.json"
    run([npm, "ci" if lock.exists() else "install"], cwd=FRONTEND)
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
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=ROOT, env=_env())
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    if wait_for(url + "api/health"):
        log(f"service is up: {url}  (API docs: {url}api/docs)")
        if not (FRONTEND / "dist" / "index.html").exists():
            log("no frontend build: only the API is served")
        if open_browser:
            webbrowser.open(url)
    else:
        log("service did not answer within 60 s; see the output above")
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()


def dev(host: str, port: int, data_root: str, fe_port: int, open_browser: bool) -> int:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        die("dev mode needs Node.js/npm")
    if not (FRONTEND / "node_modules").exists():
        run([npm, "ci"], cwd=FRONTEND)
    backend = subprocess.Popen(
        [str(venv_bin("ypuddin")), "serve", "--host", host, "--port", str(port), "--data-root", data_root],
        cwd=ROOT,
        env=_env(),
    )
    if port != 8765:
        log(
            "note: the Vite proxy targets 127.0.0.1:8765 (frontend/vite.config.ts); dev mode expects the default port"
        )
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
    print(f"root      : {ROOT}")
    print(f"platform  : {platform.platform()}")
    print(f"venv      : {'present' if py.exists() else 'missing'} ({VENV})")
    if MARKER.exists():
        print(f"install   : {MARKER.read_text().strip()}")
    driver = nvidia_driver_major()
    gpus = nvidia_gpus()
    print(
        f"nvidia    : {'driver ' + str(driver) if driver else 'no nvidia-smi'} -> torch tag {pick_torch_tag('auto')}"
    )
    for name, cc in gpus:
        print(f"  gpu         : {name} (compute capability {cc})")
    if py.exists():
        code = (
            "import torch,json;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),"
            "'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
            "'arch_list':torch.cuda.get_arch_list() if torch.cuda.is_available() else [],"
            "'mps':bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())}))"
        )
        r = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
        print(f"torch     : {r.stdout.strip() or r.stderr.strip()[-300:]}")
        try:
            archs = set(json.loads(r.stdout).get("arch_list", []))
            for name, cc in gpus:
                if cc is not None and archs and f"sm_{int(round(cc * 10))}" not in archs:
                    print(
                        f"  WARNING     : the installed torch has no kernels for {name} (sm_{int(round(cc * 10))});"
                        " run ./studio.sh --reinstall --torch=cu128"
                    )
        except Exception:  # noqa: BLE001
            pass
        for mod in ("bitsandbytes", "sageattention", "prodigyopt", "transformers", "fastapi"):
            r = subprocess.run(
                [str(py), "-c", f"import {mod};print(getattr({mod},'__version__','ok'))"],
                capture_output=True,
                text=True,
            )
            print(f"  {mod:<14}: {r.stdout.strip() or 'not installed'}")
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    print(f"node/npm  : {shutil.which('node') or 'missing'} / {npm or 'missing'}")
    print(
        f"frontend  : {'built' if (FRONTEND / 'dist' / 'index.html').exists() else 'not built'}"
        + (" (stale)" if (FRONTEND / "dist" / "index.html").exists() and frontend_stale() else "")
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
        die(f"unknown --torch value {opts['torch']!r}")
    if opts["index"] not in ("auto", "cn", "official"):
        die(f"unknown --index value {opts['index']!r}")

    if command == "doctor":
        return doctor()

    torch_tag = pick_torch_tag(opts["torch"])
    extras = choose_extras(torch_tag) + (",dev" if command == "test" else "")
    log(
        f"python {platform.python_version()} · torch flavour {torch_tag} · {'uv' if uv_path() else 'pip'} · extras [{extras}]"
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
        print(f"activate with: {act}" if WIN else f"source {act}")
        return 0
    if command == "dev":
        return dev(opts["host"], int(opts["port"]), opts["data_root"], int(opts["fe_port"]), opts["browser"])
    if command == "run":
        if opts["frontend"]:
            build_frontend()
        return serve(opts["host"], int(opts["port"]), opts["data_root"], opts["browser"])
    die(f"unknown command {command!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
