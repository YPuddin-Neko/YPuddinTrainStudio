#!/usr/bin/env bash
# YPuddin Train Studio —— Linux / macOS 启动脚本的共用阶段
#
# 这不是入口。每种环境在仓库根目录都有自己的启动脚本，由它选定环境后调用本文件：
#   ./studio-linux-cuda.sh         Linux + NVIDIA CUDA
#   ./studio-cpu.sh                Linux / macOS，仅 CPU
#   ./studio-macos.command         macOS Apple 芯片（MPS）
# 海光 DTK 用 ./studio-linux-dtk.sh，它直接调用 scripts/bootstrap.py。
# 请运行上面这些脚本，不要直接运行本文件：不带 --profile 启动会装进根目录的
# legacy venv。可用参数见 README.md 和 docs/deploy.md。
#
# 所有逻辑都在 scripts/bootstrap.py（只用标准库）；本文件只负责找到一个可用的 Python。

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

pause_on_error() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 130 ] && [ -t 0 ] && [ -t 1 ]; then
    printf '[studio] 脚本以错误码 %s 结束（上面有原因）—— 按回车关闭\n' "$rc"
    read -r _ || true
  fi
}
trap pause_on_error EXIT

# venv 已存在：直接用它自己的 Python，不再依赖系统 Python
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
