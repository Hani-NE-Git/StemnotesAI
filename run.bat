@echo off
cd /d "%~dp0"
set STEMNOTES_HOME=%~dp0
if not exist python\python.exe (
    echo Python not found. Run setup_windows.bat first.
    pause
    exit /b 1
)
python\python.exe -m stemnotes %*
if errorlevel 1 pause
