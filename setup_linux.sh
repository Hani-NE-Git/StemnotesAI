#!/usr/bin/env bash
# Portable setup: creates ./venv next to the app, downloads the SoundFont, optionally installs the AI module.
set -e
cd "$(dirname "$0")"
echo "=== StemNotes setup (Linux) ==="
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
if [ ! -x venv/bin/python ]; then
  $PY -m venv venv
fi
venv/bin/pip install -U pip wheel
venv/bin/pip install -U -r requirements-base.txt
venv/bin/pip install --no-deps tinysoundfont
mkdir -p soundfonts
if [ ! -f soundfonts/GeneralUser-GS.sf2 ]; then
  curl -L -o soundfonts/GeneralUser-GS.sf2 https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2
fi
AI="$1"
if [ -z "$AI" ]; then
  read -r -p "Install AI module (PyTorch + Demucs + Basic Pitch)? [Y/n] " AI
fi
case "$AI" in
  n|N|--no-ai) ;;
  *)
    if command -v nvidia-smi >/dev/null 2>&1; then
      venv/bin/pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    else
      venv/bin/pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    venv/bin/pip install -U -r requirements-ai.txt
    venv/bin/pip install --no-deps basic-pitch
    ;;
esac
echo "Done. Start with ./run.sh"
