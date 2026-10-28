"""Full analysis pipeline: load song -> tempo -> stems -> spectrograms -> ghost notes. Runs in a worker thread."""
from __future__ import annotations

import shutil
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..project import Project, Track, TRACK_COLORS
from ..audio.loader import load_audio, clear_cache
from . import features

STEM_ORDER = ["drums", "bass", "guitar", "piano", "vocals", "other"]
STEM_LABELS = {"mix": "Оригинал", "drums": "Ударные", "bass": "Бас", "guitar": "Гитара", "piano": "Пианино",
               "vocals": "Вокал", "other": "Остальное"}
# default MIDI track per stem: (name, program, drums)
DEFAULT_MIDI = {
    "bass": ("Бас", 33, False),
    "drums": ("Ударные", 0, True),
    "guitar": ("Гитара", 30, False),
    "piano": ("Пианино", 0, False),
    "vocals": ("Вокал (мелодия)", 73, False),
    "other": ("Остальное", 48, False),
}


@dataclass
class AnalysisOptions:
    tempo: bool = True
    separate: bool = True
    spectrograms: bool = True
    transcribe: bool = True
    model: str = "htdemucs_6s"
    engines: list[str] | None = None     # engine ids; None = all available engines that are on by default


class CancelledError(Exception):
    pass


class AnalysisJob:
    def __init__(self, project: Project, options: AnalysisOptions,
                 progress: Callable[[float, str], None], done: Callable[[Exception | None], None]):
        self.project = project
        self.options = options
        self.progress = progress
        self.done = done
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _check(self) -> None:
        if self.cancel.is_set():
            raise CancelledError()

    def _run(self) -> None:
        try:
            run_analysis(self.project, self.options, self.progress, self._check)
            self.done(None)
        except CancelledError:
            self.done(CancelledError("Отменено"))
        except Exception as e:
            traceback.print_exc()
            self.done(e)


def ensure_song_in_project(project: Project, song: Path) -> Path:
    """Copy the song into the project folder (self-contained project)."""
    project.dir.mkdir(parents=True, exist_ok=True)
    dest = project.dir / ("song" + song.suffix.lower())
    if not dest.exists() or dest.stat().st_size != song.stat().st_size:
        shutil.copy2(song, dest)
    project.song_path = project.relativize(dest)
    return dest


def ensure_audio_track(project: Project, role: str, path: Path, name: str) -> Track:
    for t in project.audio_tracks():
        if t.role == role:
            t.path = project.relativize(path)
            return t
    t = Track(name=name, kind="audio", path=project.relativize(path), role=role, volume=0.8)
    t.color = "#9e9e9e" if role == "mix" else TRACK_COLORS[(STEM_ORDER.index(role) + 1) % len(TRACK_COLORS)] if role in STEM_ORDER else "#9e9e9e"
    project.add_track(t)
    return t


def ensure_default_midi_tracks(project: Project, stems: list[str]) -> None:
    if project.midi_tracks():
        return
    for role in STEM_ORDER:
        if role not in stems or role not in DEFAULT_MIDI:
            continue
        name, program, drums = DEFAULT_MIDI[role]
        t = Track(name=name, kind="midi", instrument={"type": "sf2", "bank": 0, "program": program, "drums": drums},
                  ghost_source=role)
        project.add_track(t)


def run_analysis(project: Project, opt: AnalysisOptions, progress: Callable[[float, str], None],
                 check: Callable[[], None] = lambda: None) -> None:
    song = project.resolve(project.song_path)
    if song is None or not song.exists():
        raise FileNotFoundError("Файл песни не найден")
    an = project.analysis
    an.dir.mkdir(parents=True, exist_ok=True)

    progress(0.0, "Загрузка песни…")
    audio = load_audio(song, 44100)                         # (n,2)
    mono22 = _mono_resampled(audio, 44100, features.ANALYSIS_SR)
    ensure_audio_track(project, "mix", song, STEM_LABELS["mix"])
    sources: dict[str, np.ndarray] = {"mix": mono22}
    check()

    # ---- tempo
    if opt.tempo:
        progress(0.03, "Определение темпа…")
        try:
            bpm, beats = features.detect_tempo(mono22, features.ANALYSIS_SR)
            bpm, offset = features.refine_bpm(beats, bpm)
            project.bpm = round(bpm, 2)
            project.offset = round(offset, 4)
            an.beats = [round(b, 4) for b in beats]
        except Exception:
            traceback.print_exc()
        check()

    # ---- stems
    stems: dict[str, Path] = {}
    stems_dir = project.dir / "stems"
    if opt.separate:
        from .separate import separate

        def sp(frac, msg):
            progress(0.05 + 0.45 * frac, msg)
        stems = separate(audio, 44100, stems_dir, model=opt.model, progress=sp)
    elif stems_dir.exists():
        stems = {p.stem: p for p in stems_dir.glob("*.wav")}
    check()
    for role in STEM_ORDER:
        if role in stems:
            ensure_audio_track(project, role, stems[role], STEM_LABELS.get(role, role))
    if stems:
        for t in project.audio_tracks():
            if t.role == "mix":
                t.mute = True      # stems add up to the original; avoid playing both
    clear_cache()
    for role, p in stems.items():
        try:
            a = load_audio(p, 44100)
            sources[role] = _mono_resampled(a, 44100, features.ANALYSIS_SR)
        except Exception:
            traceback.print_exc()
    an.sources = ["mix"] + [r for r in STEM_ORDER if r in sources]

    # ---- spectrograms
    if opt.spectrograms:
        n = len(sources)
        for i, (role, mono) in enumerate(sources.items()):
            check()
            progress(0.5 + 0.15 * i / n, f"Спектрограмма: {STEM_LABELS.get(role, role)}…")
            try:
                an.images[f"spec:{role}"] = features.cqt_image(mono, features.ANALYSIS_SR)
            except Exception:
                traceback.print_exc()

    # ---- transcription with the selected engines
    if opt.transcribe:
        from .engines import ENGINES, run_engine
        chosen = [e for e in ENGINES if e.available() and (opt.engines is None and e.default_on or opt.engines is not None and e.id in opt.engines)]
        jobs = [(e, role) for e in chosen for role in sources if role in e.roles and not e.multi_role]
        an.warnings = []
        multi = [e for e in chosen if e.multi_role]
        for eng in multi:
            check()
            progress(0.65, f"{eng.name}: вся песня… (первый запуск скачивает модель)")
            result = run_engine(eng, song, sources["mix"], features.ANALYSIS_SR, "mix", an.warnings)
            if isinstance(result, tuple):
                result = {}
            from .engines import stem_energy_confidence
            for role, (ghosts, heat) in result.items():
                if role != "mix" and role in sources and role in stems:
                    stem_energy_confidence(ghosts, sources[role], features.ANALYSIS_SR)
                an.ghosts[f"{role}:{eng.id}"] = ghosts
                if heat is not None:
                    an.images[f"heat:{role}:{eng.id}"] = heat
        for i, (eng, role) in enumerate(jobs):
            check()
            progress(0.65 + 0.33 * i / max(1, len(jobs)), f"{eng.name}: {STEM_LABELS.get(role, role)}…")
            path = song if role == "mix" else stems.get(role)
            if path is None:
                continue
            ghosts, heat = run_engine(eng, path, sources[role], features.ANALYSIS_SR, role, an.warnings)
            an.ghosts[f"{role}:{eng.id}"] = ghosts
            if heat is not None:
                an.images[f"heat:{role}:{eng.id}"] = heat
            elif f"heat:{role}:{eng.id}" in an.images:
                del an.images[f"heat:{role}:{eng.id}"]

    ensure_default_midi_tracks(project, list(sources.keys()))
    progress(0.99, "Сохранение…")
    an.save()
    project.save()
    progress(1.0, "Готово")


def _mono_resampled(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    mono = audio.mean(axis=1).astype(np.float32)
    if sr == target:
        return mono
    import soxr
    return soxr.resample(mono, sr, target).astype(np.float32)
