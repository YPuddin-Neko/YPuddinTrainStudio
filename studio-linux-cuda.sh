#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export YPUDDIN_BOOTSTRAP_VENV="environment/profiles/linux-cuda/venv"
exec scripts/launch.sh --profile=linux-cuda "$@"
