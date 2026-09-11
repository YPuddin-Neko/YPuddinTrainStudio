@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ============================================================================
REM YPuddin Train Studio - Windows launcher
REM
REM This file must stay PURE ASCII and use CRLF line endings (enforced by
REM .gitattributes). Do not add Chinese characters here and do not use
REM "chcp 65001": switching the codepage mid-script makes cmd.exe resume
REM reading the file at a wrong byte offset (garbled "is not recognized"
REM errors). All Chinese output is printed by scripts\bootstrap.py, which
REM writes Unicode to the console correctly on its own.
REM
REM Usage:
REM   studio.bat                 first run: create venv, install the CUDA PyTorch
REM                              matching your GPU, install deps, build frontend,
REM                              start the server and open the browser
REM                              Includes optimizers, local logs and NVIDIA monitoring.
REM                              Later starts repair missing deps and preserve Torch/CUDA.
REM   studio.bat --port 8800     change port (default 8765; also --host, --data-root)
REM   studio.bat --torch=cu126   first install/reinstall: cu128 cu126 cu124 cu118 cpu
REM                              (default: auto by GPU compute capability + driver)
REM   studio.bat --index=official  packages: official PyPI first (default: China
REM                              mirrors first - USTC, Tsinghua, Aliyun, official last)
REM   studio.bat --reinstall     delete venv and reinstall (studio_data\ is kept)
REM   studio.bat dev             backend + Vite dev server (frontend development)
REM   studio.bat build / test / doctor / shell
REM   studio.bat smoke --set model.dit_path=... --set model.text_encoder_path=... --set model.vae_path=...
REM
REM All logic lives in scripts\bootstrap.py; this file only finds a usable Python.
REM ============================================================================

echo [studio] YPuddin Train Studio

if exist "venv\Scripts\python.exe" goto :run_venv

echo [studio] First run: looking for Python 3.10 - 3.12 ...

py -3.12 -c "import sys" >nul 2>&1
if %errorlevel%==0 (set "PY=py -3.12" & goto :run_py)
py -3.11 -c "import sys" >nul 2>&1
if %errorlevel%==0 (set "PY=py -3.11" & goto :run_py)
py -3.10 -c "import sys" >nul 2>&1
if %errorlevel%==0 (set "PY=py -3.10" & goto :run_py)

python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>&1
if %errorlevel%==0 (set "PY=python" & goto :run_py)

where uv >nul 2>&1
if not errorlevel 1 goto :run_uv

echo [studio] ERROR: Python 3.10 - 3.12 not found. Install it from https://www.python.org (tick "Add python.exe to PATH") and run this again.
goto :fail

:run_venv
"venv\Scripts\python.exe" scripts\bootstrap.py %*
goto :done

:run_py
echo [studio] Using %PY%
%PY% scripts\bootstrap.py %*
goto :done

:run_uv
echo [studio] No Python 3.10 - 3.12 found; installing Python 3.12 via uv ...
uv python install 3.12
if errorlevel 1 goto :fail
for /f "delims=" %%P in ('uv python find 3.12') do "%%P" scripts\bootstrap.py %*
goto :done

:fail
set "RC=1"
goto :pause

:done
set "RC=%errorlevel%"

:pause
if not "%RC%"=="0" (
  echo [studio] Exited with code %RC% - scroll up for the reason.
  pause
)
endlocal & exit /b %RC%
