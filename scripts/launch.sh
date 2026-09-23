#!/usr/bin/env bash
# Linux / macOS 启动的共用阶段：由 studio-linux-cuda.sh、studio-cpu.sh、studio-macos.command
# 选定 --profile 后调用。本文件只负责找到可用的 Python，其余逻辑在 scripts/bootstrap.py。

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

pause_on_error() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 130 ] && [ -t 0 ] && [ -t 1 ]; then
    printf '[studio] 已退出（错误码 %s），原因见上方输出。按回车关闭。\n' "$rc"
    read -r _ || true
  fi
}
trap pause_on_error EXIT

# 优先使用已创建环境里的 Python
if [ -n "${YPUDDIN_BOOTSTRAP_VENV:-}" ] && [ -x "$YPUDDIN_BOOTSTRAP_VENV/bin/python" ]; then
  exec "$YPUDDIN_BOOTSTRAP_VENV/bin/python" scripts/bootstrap.py "$@"
fi
if [ -x venv/bin/python ]; then
  exec venv/bin/python scripts/bootstrap.py "$@"
fi

echo "[studio] 首次运行：正在查找 Python 3.10 - 3.12 ..."
for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)' 2>/dev/null; then
    echo "[studio] 使用 $(command -v "$cand")（$("$cand" -c 'import platform;print(platform.python_version())')）"
    exec "$cand" scripts/bootstrap.py "$@"
  fi
done

if command -v uv >/dev/null 2>&1; then
  echo "[studio] 系统里没有 Python 3.10 - 3.12，用 uv 下载安装 Python 3.12 ..."
  uv python install 3.12
  exec "$(uv python find 3.12)" scripts/bootstrap.py "$@"
fi

echo "[studio] 错误：需要 Python 3.10 - 3.12。请先安装（https://www.python.org 或 https://docs.astral.sh/uv/）再运行。" >&2
exit 1
