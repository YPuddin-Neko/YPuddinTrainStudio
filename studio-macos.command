#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export YPUDDIN_BOOTSTRAP_VENV="environment/profiles/macos-mps/venv"
exec ./studio.sh --profile=macos-mps "$@"
