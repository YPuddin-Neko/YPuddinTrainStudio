#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
case "$(uname -s)" in
  Darwin) export YPUDDIN_BOOTSTRAP_VENV="environment/macos-cpu/venv" ;;
  Linux) export YPUDDIN_BOOTSTRAP_VENV="environment/linux-cpu/venv" ;;
  *) echo "[studio] This CPU launcher supports Linux or macOS." >&2; exit 1 ;;
esac
exec scripts/launch.sh --profile=cpu "$@"
