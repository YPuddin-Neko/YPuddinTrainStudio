@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM Shared Windows launcher stage: finds Python 3.10 - 3.12 for scripts\bootstrap.py.
REM Pure ASCII, CRLF; never "chcp 65001" (cmd.exe would misread byte offsets).
REM An update can replace this file while the trainer runs, and cmd.exe reads each
REM next line at a saved byte offset: the trainer starts on the last line read (:run).
echo [studio] YPuddin Train Studio
set "PY="
if defined YPUDDIN_BOOTSTRAP_VENV if exist "%YPUDDIN_BOOTSTRAP_VENV%\Scripts\python.exe" set "PY="%YPUDDIN_BOOTSTRAP_VENV%\Scripts\python.exe""
if not defined PY if exist "venv\Scripts\python.exe" set "PY="venv\Scripts\python.exe""
for %%V in (3.12 3.11 3.10) do if not defined PY py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
if not defined PY python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY where uv >nul 2>&1 && goto :uv
if defined PY goto :run
echo [studio] ERROR: Python 3.10 - 3.12 not found. Install it from https://www.python.org (tick "Add python.exe to PATH") and run this again.
goto :fail
:uv
echo [studio] No Python 3.10 - 3.12 found; installing Python 3.12 via uv ...
uv python install 3.12 || goto :fail
for /f "delims=" %%P in ('uv python find 3.12') do set "PY="%%P""
if defined PY goto :run
:fail
echo [studio] Exited with code 1 - scroll up for the reason.
pause
endlocal & exit /b 1
REM Launchers from before this layout resume below once the trainer exits.
                                                                                                                                                                                                                                                                                                                                                                                                                                                  (if not errorlevel 1 if errorlevel 0 exit /b 0) & call echo [studio] Exited with code %%errorlevel%% - scroll up for the reason.& pause & exit /b 1
:run
%PY% scripts\bootstrap.py %* & (if not errorlevel 1 if errorlevel 0 exit /b 0) & call echo [studio] Exited with code %%errorlevel%% - scroll up for the reason.& pause & exit /b 1
