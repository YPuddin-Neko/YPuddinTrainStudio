@echo off
REM YPuddin Train Studio - one-click launcher for Windows.
REM
REM   studio.bat                    first run: create .venv, install deps (CUDA torch picked from the
REM                                 NVIDIA driver), build the frontend, start the service, open the browser
REM   studio.bat --port 8800        other port (default 8765; also --host, --data-root)
REM   studio.bat --torch=cu126      force a PyTorch flavour: cu128 cu126 cu124 cu118 cpu
REM   studio.bat --mirror           PyPI mirror in mainland China (PyTorch wheels still come from pytorch.org)
REM   studio.bat --reinstall        rebuild .venv from scratch (studio_data\ is kept)
REM   studio.bat dev                backend + Vite dev server with hot reload
REM   studio.bat build | test | doctor | shell
REM   studio.bat smoke --set model.dit_path=... --set model.text_encoder_path=... --set model.vae_path=...
REM
REM This file must stay pure ASCII (cmd.exe parses it with the ANSI codepage). Logic is in scripts\bootstrap.py.

setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\bootstrap.py %*
  goto :done
)

for %%V in (3.12 3.11 3.10) do (
  py -%%V -c "import sys" >nul 2>&1 && (
    py -%%V scripts\bootstrap.py %*
    goto :done
  )
)

python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>&1 && (
  python scripts\bootstrap.py %*
  goto :done
)

where uv >nul 2>&1 && (
  echo [studio] no Python 3.10-3.12 found; letting uv install 3.12
  uv python install 3.12
  for /f "delims=" %%P in ('uv python find 3.12') do (
    "%%P" scripts\bootstrap.py %*
    goto :done
  )
)

echo [studio] ERROR: Python 3.10-3.12 is required. Install it from https://www.python.org (tick "Add to PATH") and re-run.
set ERRORLEVEL_SAVE=1
goto :pause

:done
set ERRORLEVEL_SAVE=%ERRORLEVEL%

:pause
if not "%ERRORLEVEL_SAVE%"=="0" (
  echo [studio] exited with code %ERRORLEVEL_SAVE%
  pause
)
endlocal & exit /b %ERRORLEVEL_SAVE%
