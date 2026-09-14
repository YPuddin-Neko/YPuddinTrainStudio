@echo off
setlocal
cd /d "%~dp0"
set "YPUDDIN_BOOTSTRAP_VENV=environment\profiles\windows-cuda\venv"
call studio.bat --profile=windows-cuda %*
exit /b %errorlevel%
