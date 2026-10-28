"""Project data model and JSON persistence."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PITCH_LOW = 21   # A0
PITCH_HIGH = 108  # C8
N_PITCHES = PITCH_HIGH - PITCH_LOW + 1

TRACK_COLORS = ["#4fc3f7", "#ff8a65", "#aed581", "#ba68c8", "#ffd54f", "#4db6ac", "#f06292", "#90a4ae", "#7986cb", "#a1887f"]


@dataclass
class Note:
    start: float      # beats from grid origin
    length: float     # beats
    pitch: int
    velocity: int = 100

    @property
    def end(self) -> float:
        return self.start + self.length

    def to_json(self) -> list:
        return [round(self.start, 6), round(self.length, 6), int(self.pitch), int(self.velocity)]

    @staticmethod
    def from_json(v: list) -> "Note":
        return Note(float(v[0]), float(v[1]), int(v[2]), int(v[3]) if len(v) > 3 else 100)


@dataclass
class GhostNote:
    start: float   # seconds (song time)
    end: float
    pitch: int
    conf: float    # 0..1


@dataclass
class PitchImage:
    """A time x pitch matrix (uint8 0..255) aligned to piano keys. rows = pitches ascending, cols = time."""
    data: np.ndarray           # shape (rows, cols)
    t0: float                  # time of column 0 (seconds)
    dt: float                  # seconds per column
    pitch_low: float           # midi pitch at bottom edge of row 0
    rows_per_semitone: int     # 1 or 3

    @property
    def duration(self) -> float:
        return self.data.shape[1] * self.dt


@dataclass
class Track:
    name: str
    kind: str                       # "midi" | "audio"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    color: str = TRACK_COLORS[0]
    mute: bool = False
    solo: bool = False
    volume: float = 0.8
    # midi
    instrument: dict = field(default_factory=lambda: {"type": "sf2", "bank": 0, "program": 0, "drums": False})
    notes: list[Note] = field(default_factory=list)
    ghost_source: str | None = None   # key into analysis ("bass", "drums", ...)
    # audio
    path: str | None = None           # relative to project dir or absolute
    role: str | None = None           # "mix" | stem name

    @property
    def is_drums(self) -> bool:
        return self.kind == "midi" and bool(self.instrument.get("drums"))

    def to_json(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id, "name": self.name, "kind": self.kind, "color": self.color,
            "mute": self.mute, "solo": self.solo, "volume": self.volume,
        }
        if self.kind == "midi":
            d["instrument"] = self.instrument
            d["notes"] = [n.to_json() for n in self.notes]
            d["ghost_source"] = self.ghost_source
        else:
            d["path"] = self.path
            d["role"] = self.role
        return d

    @staticmethod
    def from_json(d: dict) -> "Track":
        t = Track(name=d["name"], kind=d["kind"], id=d.get("id", uuid.uuid4().hex[:8]), color=d.get("color", TRACK_COLORS[0]),
                  mute=d.get("mute", False), solo=d.get("solo", False), volume=d.get("volume", 0.8))
        if t.kind == "midi":
            t.instrument = d.get("instrument", t.instrument)
            t.notes = [Note.from_json(n) for n in d.get("notes", [])]
            t.ghost_source = d.get("ghost_source")
        else:
            t.path = d.get("path")
            t.role = d.get("role")
        return t


class Analysis:
    """Analysis results stored in <project>/analysis. Lazily loaded."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.ghosts: dict[str, list[GhostNote]] = {}
        self.images: dict[str, PitchImage] = {}   # key like "spec:bass", "heat:bass"
        self.beats: list[float] = []
        self.sources: list[str] = []              # roles that have data (mix, bass, drums, ...)
        self.warnings: list[str] = []             # last analysis run problems (not saved)
        self.summary: list[str] = []              # last analysis run: notes per engine/stem (not saved)

    def load(self) -> None:
        meta = self.dir / "analysis.json"
        if not meta.exists():
            return
        d = json.loads(meta.read_text(encoding="utf-8"))
        self.beats = d.get("beats", [])
        self.sources = d.get("sources", [])
        self.ghosts = {self._migrate_ghost_key(k): [GhostNote(*g) for g in v] for k, v in d.get("ghosts", {}).items()}
        for key, info in d.get("images", {}).items():
            f = self.dir / info["file"]
            if f.exists():
                arr = np.load(f)["data"]
                self.images[self._migrate_image_key(key)] = PitchImage(arr, info["t0"], info["dt"], info["pitch_low"], info["rows_per_semitone"])

    @staticmethod
    def _migrate_ghost_key(key: str) -> str:
        """v0.2 stored ghosts per role ("bass"); v0.3 stores per role and engine ("bass:basic_pitch")."""
        if ":" in key:
            return key
        return f"{key}:drums_heuristic" if key == "drums" else f"{key}:basic_pitch"

    @staticmethod
    def _migrate_image_key(key: str) -> str:
        parts = key.split(":")
        if parts[0] == "heat" and len(parts) == 2:
            return f"heat:{parts[1]}:basic_pitch"
        return key

    def ghosts_for(self, role: str, engines: set[str] | None = None) -> list[tuple[str, list[GhostNote]]]:
        """[(engine_id, ghosts)] for a stem, optionally limited to the given engines."""
        out = []
        for key, g in self.ghosts.items():
            r, _, eng = key.partition(":")
            if r == role and (engines is None or eng in engines):
                out.append((eng, g))
        return out

    def engines_for(self, role: str | None = None) -> list[str]:
        ids = []
        for key in self.ghosts:
            r, _, eng = key.partition(":")
            if (role is None or r == role) and eng not in ids:
                ids.append(eng)
        return ids

    def heat_for(self, role: str, engine: str | None) -> PitchImage | None:
        if engine and f"heat:{role}:{engine}" in self.images:
            return self.images[f"heat:{role}:{engine}"]
        for key, img in self.images.items():
            if key.startswith(f"heat:{role}:"):
                return img
        return None

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        images = {}
        for key, img in self.images.items():
            fname = key.replace(":", "_") + ".npz"
            np.savez_compressed(self.dir / fname, data=img.data)
            images[key] = {"file": fname, "t0": img.t0, "dt": img.dt, "pitch_low": img.pitch_low,
                           "rows_per_semitone": img.rows_per_semitone}
        d = {
            "beats": self.beats,
            "sources": self.sources,
            "ghosts": {k: [[round(g.start, 4), round(g.end, 4), g.pitch, round(g.conf, 3)] for g in v]
                       for k, v in self.ghosts.items()},
            "images": images,
        }
        (self.dir / "analysis.json").write_text(json.dumps(d), encoding="utf-8")


class Project:
    VERSION = 1

    def __init__(self, directory: Path, name: str = "Untitled"):
        self.dir = Path(directory)
        self.name = name
        self.bpm = 120.0
        self.offset = 0.0          # seconds: song time of beat 0
        self.beats_per_bar = 4
        self.song_path: str | None = None
        self.tracks: list[Track] = []
        self.analysis = Analysis(self.dir / "analysis")
        self.dirty = False

    # ---- time conversion
    def beat_to_sec(self, beat: float) -> float:
        return self.offset + beat * 60.0 / self.bpm

    def sec_to_beat(self, sec: float) -> float:
        return (sec - self.offset) * self.bpm / 60.0

    # ---- paths
    @property
    def file(self) -> Path:
        return self.dir / "project.json"

    def resolve(self, p: str | None) -> Path | None:
        if not p:
            return None
        path = Path(p)
        return path if path.is_absolute() else (self.dir / path)

    def relativize(self, p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(self.dir.resolve()))
        except ValueError:
            return str(p)

    # ---- tracks
    def add_track(self, track: Track) -> Track:
        used = {t.color for t in self.tracks}
        if track.color in used:
            for c in TRACK_COLORS:
                if c not in used:
                    track.color = c
                    break
        self.tracks.append(track)
        self.dirty = True
        return track

    def midi_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.kind == "midi"]

    def audio_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.kind == "audio"]

    def track_by_id(self, tid: str) -> Track | None:
        for t in self.tracks:
            if t.id == tid:
                return t
        return None

    def is_audible(self, track: Track) -> bool:
        if track.mute:
            return False
        if any(t.solo for t in self.tracks):
            return track.solo
        return True

    # ---- persistence
    def to_json(self) -> dict:
        return {
            "version": self.VERSION, "name": self.name, "bpm": self.bpm, "offset": self.offset,
            "beats_per_bar": self.beats_per_bar, "song_path": self.song_path,
            "tracks": [t.to_json() for t in self.tracks],
        }

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        self.dirty = False

    @classmethod
    def load(cls, directory: Path) -> "Project":
        directory = Path(directory)
        if directory.is_file():
            directory = directory.parent
        d = json.loads((directory / "project.json").read_text(encoding="utf-8"))
        p = cls(directory, d.get("name", directory.name))
        p.bpm = float(d.get("bpm", 120.0))
        p.offset = float(d.get("offset", 0.0))
        p.beats_per_bar = int(d.get("beats_per_bar", 4))
        p.song_path = d.get("song_path")
        p.tracks = [Track.from_json(t) for t in d.get("tracks", [])]
        p.analysis.load()
        return p
