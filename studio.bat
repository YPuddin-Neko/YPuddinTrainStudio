@echo off
chcp 65001 >nul
REM YPuddin Train Studio - Windows 一键启动脚本
REM
REM   studio.bat                    首次运行：创建 venv、按显卡驱动安装 CUDA 版 PyTorch、安装依赖、
REM                                 构建前端、启动服务并打开浏览器；之后再运行只做增量检查
REM   studio.bat --port 8800        换端口（默认 8765；还有 --host、--data-root）
REM   studio.bat --torch=cu126      强制 PyTorch 版本：cu128 cu126 cu124 cu118 cpu（默认按显卡/驱动自动选）
REM   studio.bat --index=cn         强制国内镜像优先（中科大 - 清华 - 阿里 - 官方；默认自动探测）
REM   studio.bat --reinstall        删掉 venv 重装（studio_data\ 里的项目和权重不受影响）
REM   studio.bat dev                后端 + Vite 热更新前端（前端开发用）
REM   studio.bat build | test | doctor | shell
REM   studio.bat smoke --set model.dit_path=... --set model.text_encoder_path=... --set model.vae_path=...
REM
REM 所有逻辑都在 scripts\bootstrap.py；本文件只负责找到一个可用的 Python。
REM 文件按 UTF-8（无 BOM）保存，第二行 chcp 65001 之后的中文才能正确显示，请勿改动顺序。

setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM venv 已存在：直接用它自己的 Python
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" scripts\bootstrap.py %*
  goto :done
)

echo [studio] 首次运行：正在查找 Python 3.10 - 3.12 ...
for %%V in (3.12 3.11 3.10) do (
  py -%%V -c "import sys" >nul 2>&1 && (
    echo [studio] 使用 Python %%V（py 启动器）
    py -%%V scripts\bootstrap.py %*
    goto :done
  )
)

python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>&1 && (
  echo [studio] 使用 PATH 里的 python
  python scripts\bootstrap.py %*
  goto :done
)

where uv >nul 2>&1 && (
  echo [studio] 系统里没有 Python 3.10 - 3.12，用 uv 下载安装 Python 3.12 ...
  uv python install 3.12
  for /f "delims=" %%P in ('uv python find 3.12') do (
    "%%P" scripts\bootstrap.py %*
    goto :done
  )
)

echo [studio] 错误：需要 Python 3.10 - 3.12。请到 https://www.python.org 安装（勾选 "Add python.exe to PATH"）后重新运行。
set ERRORLEVEL_SAVE=1
goto :pause

:done
set ERRORLEVEL_SAVE=%ERRORLEVEL%

:pause
if not "%ERRORLEVEL_SAVE%"=="0" (
  echo [studio] 脚本以错误码 %ERRORLEVEL_SAVE% 结束（上面有原因）
  pause
)
endlocal & exit /b %ERRORLEVEL_SAVE%
