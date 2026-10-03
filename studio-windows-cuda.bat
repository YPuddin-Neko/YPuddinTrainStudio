@echo off
setlocal
cd /d "%~dp0"
set "YPUDDIN_BOOTSTRAP_VENV=environment\windows-cuda\venv"
call scripts\launch.bat --profile=windows-cuda %* & if errorlevel 1 (exit /b 1) else exit /b 0
