@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo [studio] YPuddin Train Studio

REM 用法：
REM   studio.bat                    首次运行：创建 venv、按显卡安装 CUDA 版 PyTorch、装依赖、构建前端、启动服务
REM   studio.bat --port 8800        换端口（默认 8765；还有 --host、--data-root）
REM   studio.bat --torch=cu126      强制 PyTorch 版本：cu128 cu126 cu124 cu118 cpu（默认按显卡/驱动自动选）
REM   studio.bat --index=cn         强制国内镜像优先（中科大 - 清华 - 阿里 - 官方；默认自动探测）
REM   studio.bat --reinstall        删掉 venv 重装（studio_data\ 里的项目和权重不受影响）
REM   studio.bat dev                后端 + Vite 热更新前端（前端开发用）
REM   studio.bat build / test / doctor / shell
REM   studio.bat smoke --set model.dit_path=... --set model.text_encoder_path=... --set model.vae_path=...
REM
REM 所有逻辑都在 scripts\bootstrap.py；本文件只负责找到一个可用的 Python。
REM 本文件必须是 UTF-8（无 BOM）+ CRLF 换行（仓库 .gitattributes 已强制），请勿改动前两行顺序。

if exist "venv\Scripts\python.exe" goto :run_venv

echo [studio] 首次运行：正在查找 Python 3.10 - 3.12 ...

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

echo [studio] 错误：需要 Python 3.10 - 3.12。请到 https://www.python.org 安装（勾选 Add python.exe to PATH）后重新运行。
goto :fail

:run_venv
"venv\Scripts\python.exe" scripts\bootstrap.py %*
goto :done

:run_py
echo [studio] 使用 %PY%
%PY% scripts\bootstrap.py %*
goto :done

:run_uv
echo [studio] 系统里没有 Python 3.10 - 3.12，用 uv 下载安装 Python 3.12 ...
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
  echo [studio] 脚本以错误码 %RC% 结束（上面有原因）
  pause
)
endlocal & exit /b %RC%
