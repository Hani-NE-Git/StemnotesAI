"""Portable folder layout. Everything (models, soundfonts, projects) lives next to the app."""
import os
import sys
from pathlib import Path


def app_home() -> Path:
    env = os.environ.get("STEMNOTES_HOME")
    if env:
        return Path(env).resolve()
    # repo layout: <home>/stemnotes/paths.py -> <home>
    return Path(__file__).resolve().parent.parent


HOME = app_home()
MODELS_DIR = HOME / "models"
SOUNDFONTS_DIR = HOME / "soundfonts"
PROJECTS_DIR = HOME / "projects"
CACHE_DIR = HOME / "cache"
SETTINGS_FILE = HOME / "settings.json"

for d in (MODELS_DIR, SOUNDFONTS_DIR, PROJECTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def setup_model_env() -> None:
    """Point every ML framework cache into the app folder. Must run before importing torch/librosa."""
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "huggingface" / "hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))
    os.environ.setdefault("LIBROSA_CACHE_DIR", str(CACHE_DIR / "librosa"))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "mpl"))
    for key in ("TORCH_HOME", "HF_HOME", "HF_HUB_CACHE", "NUMBA_CACHE_DIR", "LIBROSA_CACHE_DIR", "MPLCONFIGDIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


setup_model_env()

SOUNDFONT_URL = "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2"
SOUNDFONT_NAME = "GeneralUser-GS.sf2"


def find_soundfont() -> Path | None:
    preferred = SOUNDFONTS_DIR / SOUNDFONT_NAME
    if preferred.exists():
        return preferred
    for p in sorted(SOUNDFONTS_DIR.glob("*.sf2")):
        return p
    for p in sorted(SOUNDFONTS_DIR.glob("*.SF2")):
        return p
    return None


def is_windows() -> bool:
    return sys.platform.startswith("win")
