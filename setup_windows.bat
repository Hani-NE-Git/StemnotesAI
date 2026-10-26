@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo ================================================================
echo  StemNotes - portable setup (all files stay in this folder)
echo ================================================================
set PYVER=3.12.10
set PYZIP=python-%PYVER%-embed-amd64.zip
set PYURL=https://www.python.org/ftp/python/%PYVER%/%PYZIP%

if not exist python\python.exe (
    echo [1/5] Downloading embeddable Python %PYVER% ...
    curl -L -o "%PYZIP%" "%PYURL%" || goto :fail
    mkdir python
    tar -xf "%PYZIP%" -C python || goto :fail
    del "%PYZIP%"
    rem enable site-packages for the embeddable distribution
    (
        echo python312.zip
        echo .
        echo Lib\site-packages
        echo ..
        echo import site
    ) > python\python312._pth
) else (
    echo [1/5] Python already present.
)

if not exist python\Scripts\pip.exe (
    echo [2/5] Installing pip ...
    curl -L -o get-pip.py https://bootstrap.pypa.io/get-pip.py || goto :fail
    python\python.exe get-pip.py --no-warn-script-location || goto :fail
    del get-pip.py
) else (
    echo [2/5] pip already present.
)

echo [3/5] Installing core packages ...
python\python.exe -m pip install --no-warn-script-location -U -r requirements-base.txt || goto :fail
python\python.exe -m pip install --no-warn-script-location --no-deps tinysoundfont || goto :fail

echo [4/5] Downloading SoundFont (GeneralUser GS) ...
if not exist soundfonts mkdir soundfonts
if not exist soundfonts\GeneralUser-GS.sf2 (
    curl -L -o soundfonts\GeneralUser-GS.sf2 https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2
)

set AI=%1
if "%AI%"=="" (
    set /p AI="[5/5] Install AI module (PyTorch CUDA 12.8 + Demucs + Basic Pitch, ~3.5 GB download)? [Y/n]: "
)
if /i "%AI%"=="n" goto :done
if /i "%AI%"=="--no-ai" goto :done
echo [5/5] Installing PyTorch (CUDA 12.8 build, works with RTX 50xx / Blackwell) ...
python\python.exe -m pip install --no-warn-script-location -U torch --index-url https://download.pytorch.org/whl/cu128 || goto :fail
echo       Installing Demucs, Basic Pitch and friends ...
python\python.exe -m pip install --no-warn-script-location -U -r requirements-ai.txt || goto :fail
python\python.exe -m pip install --no-warn-script-location --no-deps basic-pitch || goto :fail
python\python.exe -c "import torch;print('CUDA available:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

:done
echo.
echo Setup finished. Start the app with run.bat
pause
exit /b 0

:fail
echo.
echo *** Setup failed. Check your internet connection and run setup_windows.bat again. ***
pause
exit /b 1
