#!/usr/bin/env bash
# YPuddin Train Studio — one-click launcher for Linux / macOS.
#
#   ./studio.sh                    first run: create .venv, install deps (CUDA torch picked from the
#                                  NVIDIA driver), build the frontend, start the service, open the browser
#   ./studio.sh --port 8800        other port (default 8765; also --host, --data-root)
#   ./studio.sh --torch=cu126      force a PyTorch flavour: cu128 cu126 cu124 cu118 cpu
#   ./studio.sh --mirror           PyPI mirror in mainland China (PyTorch wheels still come from pytorch.org)
#   ./studio.sh --reinstall        rebuild .venv from scratch (studio_data/ is kept)
#   ./studio.sh dev                backend + Vite dev server with hot reload
#   ./studio.sh build | test | doctor | shell
#   ./studio.sh smoke --set model.dit_path=... --set model.text_encoder_path=... --set model.vae_path=...
#
# Everything non-trivial lives in scripts/bootstrap.py (stdlib only); this file only finds a Python.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

pause_on_error() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 130 ] && [ -t 0 ] && [ -t 1 ]; then
    printf '[studio] exited with code %s — press Enter to close\n' "$rc"
    read -r _ || true
  fi
}
trap pause_on_error EXIT

# Prefer the venv's own interpreter once it exists (no dependency on a system Python afterwards).
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python scripts/bootstrap.py "$@"
fi

for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)' 2>/dev/null; then
    exec "$cand" scripts/bootstrap.py "$@"
  fi
done

if command -v uv >/dev/null 2>&1; then
  echo "[studio] no Python 3.10-3.12 on PATH; letting uv install 3.12"
  uv python install 3.12
  exec "$(uv python find 3.12)" scripts/bootstrap.py "$@"
fi

echo "[studio] ERROR: Python 3.10-3.12 is required. Install it (https://www.python.org or https://docs.astral.sh/uv/) and re-run." >&2
exit 1
