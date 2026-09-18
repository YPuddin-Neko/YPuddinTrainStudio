@echo off
setlocal
cd /d "%~dp0"
set "YPUDDIN_BOOTSTRAP_VENV=environment\windows-cpu\venv"
call scripts\launch.bat --profile=cpu %*
exit /b %errorlevel%
