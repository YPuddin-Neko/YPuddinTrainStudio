#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
case "$(uname -s)" in
  Darwin) export YPUDDIN_BOOTSTRAP_VENV="environment/macos-cpu/venv" ;;
  Linux) export YPUDDIN_BOOTSTRAP_VENV="environment/linux-cpu/venv" ;;
  *) echo "[studio] 此 CPU 启动入口只适用于 Linux 或 macOS；Windows 请使用 studio-cpu.bat。" >&2; exit 1 ;;
esac
exec scripts/launch.sh --profile=cpu "$@"
