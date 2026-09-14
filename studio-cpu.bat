@echo off
setlocal
cd /d "%~dp0"
set "YPUDDIN_BOOTSTRAP_VENV=environment\profiles\windows-cpu\venv"
call studio.bat --profile=cpu %*
exit /b %errorlevel%
