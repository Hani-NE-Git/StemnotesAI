"""StemNotes: lightweight note-transcription DAW (stems + ghost notes + piano roll)."""
from pathlib import Path

try:
    __version__ = (Path(__file__).resolve().parent.parent / "VERSION").read_text(encoding="utf-8").strip()
except Exception:
    __version__ = "0.0.0"
