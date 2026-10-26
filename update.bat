@echo off
cd /d "%~dp0"
set STEMNOTES_HOME=%~dp0
if not exist python\python.exe (
    echo Python not found. Run setup_windows.bat first.
    pause
    exit /b 1
)
echo Updating StemNotes code from GitHub (models, Python, SoundFont and projects are kept)...
python\python.exe -m stemnotes.updater %*
pause
