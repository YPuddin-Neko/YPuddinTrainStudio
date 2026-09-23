@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ============================================================================
REM Shared Windows launcher stage, called by studio-windows-cuda.bat and
REM studio-cpu.bat after they select a profile. It only finds a usable Python;
REM all logic lives in scripts\bootstrap.py.
REM
REM Keep this file pure ASCII with CRLF line endings (enforced by .gitattributes).
REM Do not use "chcp 65001": switching the code page mid-script makes cmd.exe
REM resume reading at a wrong byte offset. Chinese output comes from bootstrap.py.
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
