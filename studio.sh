#!/usr/bin/env bash
# YPuddin Train Studio —— Linux / macOS 一键启动脚本
#
#   ./studio.sh                    首次运行：创建 venv、按显卡驱动安装 CUDA 版 PyTorch、安装依赖、
#                                  构建前端、启动服务并打开浏览器；之后再运行只做增量检查
#   ./studio.sh --port 8800        换端口（默认 8765；还有 --host、--data-root）
#   ./studio.sh --torch=cu126      强制 PyTorch 版本：cu128 cu126 cu124 cu118 cpu（默认按显卡/驱动自动选）
#   ./studio.sh --index=cn         强制国内镜像优先（中科大 → 清华 → 阿里 → 官方；默认自动探测）
#   ./studio.sh --reinstall        删掉 venv 重装（studio_data/ 里的项目和权重不受影响）
#   ./studio.sh dev                后端 + Vite 热更新前端（前端开发用）
#   ./studio.sh build | test | doctor | shell
#   ./studio.sh smoke --set model.dit_path=... --set model.text_encoder_path=... --set model.vae_path=...
#
# 所有逻辑都在 scripts/bootstrap.py（只用标准库）；本文件只负责找到一个可用的 Python。

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
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
