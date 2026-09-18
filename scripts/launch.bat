@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ============================================================================
REM YPuddin Train Studio - shared Windows launcher stage
REM
REM Not an entry point. Every environment has its own start script in the
REM repository root; it selects the environment and then calls this file:
REM   studio-windows-cuda.bat    Windows + NVIDIA CUDA
REM   studio-cpu.bat             Windows, CPU only
REM Linux and macOS use studio-linux-cuda.sh, studio-linux-dtk.sh,
REM studio-cpu.sh or studio-macos.command. Run those, not this file:
REM launching without a profile would install into the legacy root venv.
REM Accepted options are documented in README.md and docs/deploy.md.
REM
REM This file must stay PURE ASCII and use CRLF line endings (enforced by
REM .gitattributes). Do not add Chinese characters here and do not use
REM "chcp 65001": switching the codepage mid-script makes cmd.exe resume
REM reading the file at a wrong byte offset (garbled "is not recognized"
REM errors). All Chinese output is printed by scripts\bootstrap.py, which
REM writes Unicode to the console correctly on its own.
REM
REM All logic lives in scripts\bootstrap.py; this file only finds a usable Python.
REM ============================================================================

echo [studio] YPuddin Train Studio

if defined YPUDDIN_BOOTSTRAP_VENV if exist "%YPUDDIN_BOOTSTRAP_VENV%\Scripts\python.exe" goto :run_profile
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

:run_profile
"%YPUDDIN_BOOTSTRAP_VENV%\Scripts\python.exe" scripts\bootstrap.py %*
goto :done

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
