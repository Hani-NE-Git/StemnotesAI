"""Pluggable ghost-note engines. Several engines can run on the same stem; the user picks which to show.

Each engine returns (ghost notes, optional heat image). Engines are only used if their libraries are installed.
"""
from __future__ import annotations

import importlib.util
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .. import paths
from ..project import GhostNote, PitchImage, PITCH_LOW, PITCH_HIGH, N_PITCHES

PITCHED_ROLES = ["mix", "bass", "guitar", "piano", "vocals", "other"]


@dataclass
class Engine:
    id: str
    name: str
    kind: str                      # "pitched" | "drums"
    roles: list[str]               # default stems it runs on
    color: tuple[int, int, int]    # ghost colour in the piano roll
    description: str
    needs: list[str] = field(default_factory=list)     # importable modules
    run: Callable[..., tuple[list[GhostNote], PitchImage | None]] | None = None
    default_on: bool = True
    multi_role: bool = False       # True: runs once on the whole song and returns {role: (ghosts, heat)}

    def available(self) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in self.needs)


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------- helpers
def contour_to_notes(times: np.ndarray, midi: np.ndarray, conf: np.ndarray, voiced: np.ndarray,
                     min_len: float = 0.06, max_gap: float = 0.03) -> list[GhostNote]:
    """Turn a frame-wise pitch contour into note segments (same rounded pitch, consecutive voiced frames)."""
    notes: list[GhostNote] = []
    n = len(times)
    i = 0
    while i < n:
        if not voiced[i] or not np.isfinite(midi[i]):
            i += 1
            continue
        p = int(round(midi[i]))
        j = i
        last_voiced = i
        confs = []
        while j < n:
            if voiced[j] and np.isfinite(midi[j]) and int(round(midi[j])) == p:
                last_voiced = j
                confs.append(conf[j])
            elif times[j] - times[last_voiced] > max_gap:
                break
            elif voiced[j] and np.isfinite(midi[j]) and int(round(midi[j])) != p:
                break
            j += 1
        start, end = float(times[i]), float(times[last_voiced] + (times[1] - times[0] if n > 1 else 0.01))
        if end - start >= min_len and PITCH_LOW <= p <= PITCH_HIGH:
            notes.append(GhostNote(start, end, p, float(np.clip(np.mean(confs) if confs else 0.3, 0.05, 1.0))))
        i = max(j, i + 1)
    return notes


def contour_image(times: np.ndarray, midi: np.ndarray, conf: np.ndarray, voiced: np.ndarray) -> PitchImage:
    n = len(times)
    dt = float(times[1] - times[0]) if n > 1 else 0.01
    img = np.zeros((N_PITCHES, max(1, n)), dtype=np.uint8)
    for k in range(n):
        if voiced[k] and np.isfinite(midi[k]):
            r = int(round(midi[k])) - PITCH_LOW
            if 0 <= r < N_PITCHES:
                img[r, k] = int(np.clip(conf[k], 0, 1) * 255)
    return PitchImage(img, float(times[0]) if n else 0.0, dt, PITCH_LOW - 0.5, 1)


ROLE_RANGE = {"bass": (28.0, 450.0), "vocals": (70.0, 1200.0), "guitar": (70.0, 1400.0), "piano": (27.5, 4200.0),
              "other": (50.0, 2000.0), "mix": (50.0, 2000.0)}


# ---------------------------------------------------------------- engines
def run_basic_pitch(path: Path, mono: np.ndarray, sr: int, role: str):
    from .transcribe import transcribe_pitched
    return transcribe_pitched(path)


def run_pyin(path: Path, mono: np.ndarray, sr: int, role: str):
    import librosa
    fmin, fmax = ROLE_RANGE.get(role, (50.0, 2000.0))
    hop = 256
    f0, vflag, vprob = librosa.pyin(mono, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    midi = librosa.hz_to_midi(np.where(np.isfinite(f0) & (f0 > 0), f0, np.nan))
    voiced = np.asarray(vflag, dtype=bool) & (np.asarray(vprob) > 0.3)
    conf = np.asarray(vprob, dtype=np.float32)
    return contour_to_notes(times, midi, conf, voiced), contour_image(times, midi, conf, voiced)


def _ensure_torchaudio_stub() -> None:
    """torchcrepe imports torchaudio only to read files (we feed arrays). If torchaudio is missing or its
    binary does not match torch, register a stub so torchcrepe still imports."""
    import sys
    import types
    try:
        import torchaudio  # noqa: F401
    except Exception:
        stub = types.ModuleType("torchaudio")
        stub.load = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("torchaudio stub"))
        sys.modules["torchaudio"] = stub


def run_crepe(path: Path, mono: np.ndarray, sr: int, role: str):
    import torch
    _ensure_torchaudio_stub()
    import torchcrepe
    import soxr
    fmin, fmax = ROLE_RANGE.get(role, (50.0, 2000.0))
    audio16 = soxr.resample(mono.astype(np.float32), sr, 16000).astype(np.float32)
    x = torch.from_numpy(audio16)[None, :]
    hop = 160  # 10 ms
    dev = _device()
    pitch, periodicity = torchcrepe.predict(x, 16000, hop, fmin, fmax, "full", return_periodicity=True,
                                            batch_size=512, device=dev, decoder=torchcrepe.decode.viterbi)
    periodicity = torchcrepe.filter.median(periodicity, 3)
    pitch = torchcrepe.filter.mean(pitch, 3)
    hz = pitch[0].cpu().numpy()
    per = periodicity[0].cpu().numpy()
    times = np.arange(len(hz)) * hop / 16000.0
    midi = 69 + 12 * np.log2(np.maximum(hz, 1e-6) / 440.0)
    voiced = per > 0.25
    return contour_to_notes(times, midi, per.astype(np.float32), voiced), contour_image(times, midi, per, voiced)


def run_piano_bytedance(path: Path, mono: np.ndarray, sr: int, role: str):
    import soxr
    from piano_transcription_inference import PianoTranscription, sample_rate
    ckpt_dir = paths.MODELS_DIR / "piano_transcription"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
    audio = soxr.resample(mono.astype(np.float32), sr, sample_rate).astype(np.float32) if sr != sample_rate else mono.astype(np.float32)
    tr = PianoTranscription(device=_device(), checkpoint_path=str(ckpt))
    res = tr.transcribe(audio, None)
    frame = np.asarray(res["output_dict"]["frame_output"], dtype=np.float32)  # (frames, 88) at 100 fps
    fps = 100.0
    ghosts: list[GhostNote] = []
    for ev in res["est_note_events"]:
        s, e, p = float(ev["onset_time"]), float(ev["offset_time"]), int(ev["midi_note"])
        a, b = int(s * fps), max(int(s * fps) + 1, int(e * fps))
        r = p - PITCH_LOW
        c = float(frame[a:b, r].mean()) if 0 <= r < 88 and a < len(frame) else 0.5
        ghosts.append(GhostNote(s, max(e, s + 0.05), p, float(np.clip(0.3 + 0.7 * c, 0.05, 1.0))))
    img = (frame.T * 255).clip(0, 255).astype(np.uint8)
    return ghosts, PitchImage(img, 0.0, 1.0 / fps, PITCH_LOW - 0.5, 1)


# ---------------------------------------------------------------- MuScriptor (Kyutai / Mirelo)
# MT3_FULL_PLUS instrument groups -> our stem roles
MUSC_GROUP_TO_ROLE = {
    "acoustic_bass": "bass", "electric_bass": "bass", "contrabass": "bass",
    "drums": "drums",
    "acoustic_piano": "piano", "electric_piano": "piano", "organ": "piano", "chromatic_percussion": "piano",
    "acoustic_guitar": "guitar", "clean_electric_guitar": "guitar", "distorted_electric_guitar": "guitar",
    "voice": "vocals",
}
MUSC_ROLE_TO_GROUPS = {
    "bass": ["acoustic_bass", "electric_bass", "contrabass"],
    "drums": ["drums"],
    "piano": ["acoustic_piano", "electric_piano", "organ", "chromatic_percussion"],
    "guitar": ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
    "vocals": ["voice"],
}
NO_CONF = -1.0   # engine gives decisions, not probabilities


def _settings() -> dict:
    import json
    try:
        return json.loads(paths.SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


_musc_cache: dict = {}


def _muscriptor_model():
    import os
    st = _settings()
    token = st.get("hf_token")
    if token:
        os.environ["HF_TOKEN"] = token
    size = st.get("muscriptor_model") or "medium"
    key = (size, _device())
    if key not in _musc_cache:
        from muscriptor import TranscriptionModel
        _musc_cache.clear()
        _musc_cache[key] = TranscriptionModel.load_model(size, device=_device())
    return _musc_cache[key]


def muscriptor_events_to_ghosts(events) -> dict[str, list[GhostNote]]:
    """Consume the MuScriptor event stream; returns {instrument_group: ghosts}."""
    from muscriptor.events import NoteStartEvent, NoteEndEvent
    by_group: dict[str, list[GhostNote]] = {}
    for ev in events:
        if isinstance(ev, NoteEndEvent):
            st = ev.start_event
            start = float(st.start_time)
            end = float(ev.end_time)
            if st.instrument == "drums" or end <= start:
                end = start + 0.1
            if PITCH_LOW <= int(st.pitch) <= PITCH_HIGH or st.instrument == "drums":
                by_group.setdefault(st.instrument, []).append(GhostNote(start, end, int(st.pitch), NO_CONF))
        elif isinstance(ev, NoteStartEvent):
            pass
    for g in by_group.values():
        g.sort(key=lambda x: x.start)
    return by_group


def run_muscriptor_mix(path: Path, mono: np.ndarray, sr: int, role: str) -> dict[str, tuple[list[GhostNote], PitchImage | None]]:
    """One pass over the whole song; notes are routed to stems by the instrument the model assigned."""
    model = _muscriptor_model()
    by_group = muscriptor_events_to_ghosts(model.transcribe(str(path)))
    out: dict[str, list[GhostNote]] = {"mix": []}
    for group, ghosts in by_group.items():
        out["mix"].extend(ghosts)
        role_ = MUSC_GROUP_TO_ROLE.get(group, "other")
        out.setdefault(role_, []).extend(ghosts)
    for g in out.values():
        g.sort(key=lambda x: x.start)
    return {r: (g, None) for r, g in out.items()}


def run_muscriptor_stem(path: Path, mono: np.ndarray, sr: int, role: str):
    """Transcribe one separated stem, constrained to that stem's instrument groups."""
    model = _muscriptor_model()
    groups = MUSC_ROLE_TO_GROUPS.get(role)
    by_group = muscriptor_events_to_ghosts(model.transcribe(str(path), instruments=groups))
    ghosts = [g for lst in by_group.values() for g in lst]
    ghosts.sort(key=lambda x: x.start)
    return ghosts, None


def run_drums_heuristic(path: Path, mono: np.ndarray, sr: int, role: str):
    from .transcribe import transcribe_drums
    return transcribe_drums(mono, sr), None


ENGINES: list[Engine] = [
    Engine("basic_pitch", "Basic Pitch (Spotify)", "pitched", list(PITCHED_ROLES), (255, 200, 60),
           "Полифония, любой инструмент. Даёт несколько кандидатов и карту вероятностей.", ["basic_pitch", "onnxruntime"], run_basic_pitch),
    Engine("crepe", "CREPE (монофония)", "pitched", ["bass", "vocals"], (120, 220, 255),
           "Нейросеть для одноголосных партий: бас, вокал. Точная высота, одна нота в момент времени.", ["torchcrepe", "torch"], run_crepe),
    Engine("pyin", "pYIN (монофония, без нейросети)", "pitched", ["bass", "vocals"], (120, 255, 160),
           "Классический алгоритм для одноголосных партий. Быстрый, без моделей.", ["librosa"], run_pyin, default_on=False),
    Engine("piano_bytedance", "Piano Transcription (ByteDance)", "pitched", ["piano"], (255, 120, 90),
           "Лучшая модель для фортепиано. Первый запуск скачивает ~170 МБ в models.", ["piano_transcription_inference", "torch"], run_piano_bytedance),
    Engine("muscriptor", "MuScriptor (вся песня)", "pitched", ["mix", "bass", "drums", "guitar", "piano", "vocals", "other"], (255, 110, 220),
           "Kyutai/Mirelo: один проход по оригиналу, ноты раскладываются по инструментам, включая ударные. "
           "Нужен HF-токен и принятая лицензия модели (некоммерческая).", ["muscriptor", "torch"], run_muscriptor_mix, multi_role=True),
    Engine("muscriptor_stem", "MuScriptor (по стемам)", "pitched", ["bass", "drums", "guitar", "piano", "vocals"], (200, 120, 255),
           "MuScriptor отдельно на каждом стеме с ограничением по инструменту. Точнее на плотных миксах, дольше.",
           ["muscriptor", "torch"], run_muscriptor_stem, default_on=False),
    Engine("drums_heuristic", "Ударные: детектор ударов", "drums", ["drums"], (255, 200, 60),
           "Поиск ударов + классификация бочка/малый/хэт/тарелка по спектру.", ["librosa"], run_drums_heuristic),
]
ENGINE_BY_ID = {e.id: e for e in ENGINES}


def engine_color(engine_id: str) -> tuple[int, int, int]:
    e = ENGINE_BY_ID.get(engine_id)
    return e.color if e else (255, 200, 60)


def run_engine(engine: Engine, path: Path, mono: np.ndarray, sr: int, role: str):
    try:
        return engine.run(path, mono, sr, role)
    except Exception:
        traceback.print_exc()
        return ({}, None) if engine.multi_role else ([], None)
