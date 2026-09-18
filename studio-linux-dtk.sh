#!/usr/bin/env bash
# 海光 DTK 独立入口；不安装驱动，不从 CUDA / CPU 包源获取 PyTorch。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "$(uname -s)" != Linux ]; then
  echo '[studio] DTK 启动入口只适用于 Linux。' >&2
  exit 1
fi
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export DTKROOT="$DTK_ROOT"
export ROCM_PATH="$DTK_ROOT" HIP_PATH="$DTK_ROOT/hip"
if [ ! -d "$DTK_ROOT" ]; then
  echo '[studio] 未找到 DTK 运行时目录，请设置 DTK_ROOT=/path/to/dtk。' >&2
  exit 1
fi
# Process-local vendor search paths. DTK 26.04 keeps OpenMP, GCVM and COMGR
# under dcc; older releases may expose these through llvm or top-level links.
# Prepend only existing directories, preserving caller paths after this runtime.
for studio_dtk_lib in \
  /opt/hyhal/lib64 /opt/hyhal/lib \
  "$DTK_ROOT/.hyhal/lib64" "$DTK_ROOT/.hyhal/lib" \
  "$DTK_ROOT/.hyhal/rocm_smi/lib" \
  "$DTK_ROOT/opencl/lib" "$DTK_ROOT/dushmem/lib" "$DTK_ROOT/lib64" \
  "$DTK_ROOT/comgr/lib" "$DTK_ROOT/gcvm/lib" \
  "$DTK_ROOT/dcc/comgr/lib" "$DTK_ROOT/dcc/gcvm/lib" "$DTK_ROOT/dcc/lib" \
  "$DTK_ROOT/llvm/lib" "$DTK_ROOT/hip/lib" "$DTK_ROOT/lib"; do
  if [ -d "$studio_dtk_lib" ]; then
    export LD_LIBRARY_PATH="$studio_dtk_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="$studio_dtk_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  fi
done
for studio_dtk_bin in \
  "$DTK_ROOT/opencl/bin" /opt/hyhal/bin \
  "$DTK_ROOT/.hyhal/bin" "$DTK_ROOT/.hyhal/rocm_smi/bin" \
  "$DTK_ROOT/hip/bin/hipify" "$DTK_ROOT/hip/bin" \
  "$DTK_ROOT/dcc/bin" "$DTK_ROOT/llvm/bin" "$DTK_ROOT/bin"; do
  if [ -d "$studio_dtk_bin" ]; then
    export PATH="$studio_dtk_bin:$PATH"
  fi
done
for studio_dtk_include in \
  "$DTK_ROOT/.hyhal/include" /opt/hyhal/include \
  "$DTK_ROOT/opencl/include" "$DTK_ROOT/dushmem/include" \
  "$DTK_ROOT/dcc/gcvm/include" "$DTK_ROOT/llvm/include" "$DTK_ROOT/include"; do
  if [ -d "$studio_dtk_include" ]; then
    export C_INCLUDE_PATH="$studio_dtk_include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
    export CPLUS_INCLUDE_PATH="$studio_dtk_include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  fi
done
studio_dtk_python=environment/linux-dtk/venv/bin/python
if [ -x "$studio_dtk_python" ]; then
  exec "$studio_dtk_python" scripts/bootstrap.py --profile=linux-dtk "$@"
fi
if [ -n "${YPUDDIN_DTK_PYTHON:-}" ]; then
  if ! "$YPUDDIN_DTK_PYTHON" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)' 2>/dev/null; then
    echo '[studio] YPUDDIN_DTK_PYTHON 必须指向与厂商 wheel 匹配的 Python 3.10–3.12。' >&2
    exit 1
  fi
  exec "$YPUDDIN_DTK_PYTHON" scripts/bootstrap.py --profile=linux-dtk "$@"
fi
# Use the Python selected by the vendor environment first; never borrow the legacy venv.
for studio_dtk_candidate in python3 python3.12 python3.11 python3.10 python; do
  if command -v "$studio_dtk_candidate" >/dev/null 2>&1 && "$studio_dtk_candidate" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)' 2>/dev/null; then
    exec "$studio_dtk_candidate" scripts/bootstrap.py --profile=linux-dtk "$@"
  fi
done
echo '[studio] 需要与厂商 wheel 匹配的 Python 3.10–3.12。请设置 YPUDDIN_DTK_PYTHON=/path/to/python。' >&2
exit 1
