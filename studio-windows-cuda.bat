@echo off
setlocal
cd /d "%~dp0"
set "YPUDDIN_BOOTSTRAP_VENV=environment\windows-cuda\venv"
call scripts\launch.bat --profile=windows-cuda %*
exit /b %errorlevel%
