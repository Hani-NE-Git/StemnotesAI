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
from ..log import get as _get_log
from ..project import GhostNote, PitchImage, PITCH_LOW, PITCH_HIGH, N_PITCHES

log = _get_log("engines")

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
    fmin = max(fmin, 33.0)   # CREPE's lowest bin is ~32.7 Hz; a lower fmin masks every bin and yields NaN
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
_musc_error: str | None = None


def _muscriptor_model():
    """Load (once) the MuScriptor model; a failed load is remembered so every stem does not retry the download."""
    global _musc_error
    import os
    st = _settings()
    token = (st.get("hf_token") or "").strip()
    if token:
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            os.environ[var] = token
    size = st.get("muscriptor_model") or ("large" if _device() == "cuda" else "medium")
    key = (size, _device(), token[-6:])
    if key in _musc_cache:
        return _musc_cache[key]
    if _musc_error is not None and _musc_error[0] == key:
        raise RuntimeError(_musc_error[1])
    try:
        from muscriptor import TranscriptionModel
        _musc_cache.clear()
        _musc_cache[key] = TranscriptionModel.load_model(size, device=_device())
    except Exception as e:
        msg = str(e).strip()
        if not token:
            msg = ("HF-токен не задан. Примите лицензию на странице модели и вставьте read-токен в поле «HF token» "
                   "окна анализа.\n" + msg)
        elif "403" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            msg = (f"Hugging Face отказал в доступе к MuScriptor/muscriptor-{size}. Проверьте, что лицензия принята "
                   f"именно на странице https://huggingface.co/MuScriptor/muscriptor-{size} и токен действителен.\n" + msg)
        _musc_error = (key, msg)
        raise RuntimeError(msg) from e
    _musc_error = None
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


def gate_silence(mono: np.ndarray, sr: int, floor_db: float = -45.0) -> tuple[np.ndarray, float]:
    """Zero out parts of a stem that are practically silent (bleed from other instruments), so a model that is
    forced to output this instrument does not invent notes there. Returns (gated audio, fraction kept)."""
    x = mono.astype(np.float32)
    win = max(1, int(0.02 * sr))
    n = len(x) // win
    if n == 0:
        return x, 1.0
    rms = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
    ref = float(np.percentile(rms, 95)) + 1e-9
    keep = rms > ref * (10 ** (floor_db / 20.0))
    # dilate by 100 ms so attacks and tails survive
    k = np.convolve(keep.astype(np.float32), np.ones(11), mode="same") > 0
    mask = np.repeat(k, win)
    out = x.copy()
    out[:len(mask)] *= mask
    if len(mask) < len(out):
        out[len(mask):] = 0
    return out, float(k.mean())


def run_muscriptor_mix(path: Path, mono: np.ndarray, sr: int, role: str) -> dict[str, tuple[list[GhostNote], PitchImage | None]]:
    """One pass over the whole song; notes are routed to stems by the instrument the model assigned."""
    model = _muscriptor_model()
    log.info("MuScriptor (whole song): transcribing %s", path)
    by_group = muscriptor_events_to_ghosts(model.transcribe(str(path)))
    log.info("MuScriptor (whole song): notes per instrument group: %s", {k: len(v) for k, v in by_group.items()})
    out: dict[str, list[GhostNote]] = {"mix": []}
    for group, ghosts in by_group.items():
        out["mix"].extend(ghosts)
        role_ = MUSC_GROUP_TO_ROLE.get(group, "other")
        out.setdefault(role_, []).extend(ghosts)
    for g in out.values():
        g.sort(key=lambda x: x.start)
    return {r: (g, None) for r, g in out.items()}


def stem_energy_confidence(ghosts: list[GhostNote], stem_mono: np.ndarray, sr: int) -> None:
    """For notes routed to a stem by instrument label, measure how loud that stem is under the note.
    conf 1.0 = stem at its normal level, ~0.05 = stem silent (the note probably belongs to another instrument)."""
    if not ghosts or len(stem_mono) == 0:
        return
    x = stem_mono.astype(np.float32)
    win = max(1, int(0.02 * sr))
    n = len(x) // win
    if n == 0:
        return
    rms = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
    ref = float(np.percentile(rms, 95)) + 1e-9
    for g in ghosts:
        a = int(g.start / 0.02)
        b = max(a + 1, int(min(g.end, g.start + 0.5) / 0.02))
        seg = rms[a:b]
        if len(seg) == 0:
            continue
        db = 20 * np.log10(float(seg.max()) / ref + 1e-9)
        g.conf = float(np.clip((db + 40.0) / 30.0, 0.05, 1.0))   # -10 dB → 1.0, -40 dB → ~0.05


def run_muscriptor_stem(path: Path, mono: np.ndarray, sr: int, role: str):
    """Transcribe ONE separated stem file, constrained to that stem's instrument groups.
    Silent parts of the stem are zeroed first so the model cannot invent notes from bleed."""
    import torch
    model = _muscriptor_model()
    groups = MUSC_ROLE_TO_GROUPS.get(role)
    gated, kept = gate_silence(mono, sr)
    log.info("MuScriptor (stem) role=%s file=%s instruments=%s audible=%.0f%%", role, path, groups, kept * 100)
    audio = (torch.from_numpy(gated)[None, :], sr)      # the stem only, never the full mix
    by_group = muscriptor_events_to_ghosts(model.transcribe(audio, instruments=groups))
    ghosts = [g for lst in by_group.values() for g in lst]
    ghosts.sort(key=lambda x: x.start)
    stem_energy_confidence(ghosts, mono, sr)
    log.info("MuScriptor (stem) role=%s: %d notes, groups %s", role, len(ghosts), {k: len(v) for k, v in by_group.items()})
    return ghosts, None


def run_drums_heuristic(path: Path, mono: np.ndarray, sr: int, role: str):
    from .transcribe import transcribe_drums
    return transcribe_drums(mono, sr), None


ENGINES: list[Engine] = [
    Engine("muscriptor", "MuScriptor (вся песня)", "pitched", ["mix", "bass", "drums", "guitar", "piano", "vocals", "other"], (255, 110, 220),
           "Kyutai/Mirelo: один проход по оригиналу, ноты раскладываются по инструментам, как их опознала модель. "
           "Процент = насколько стем звучит под нотой (тихий стем → нота, скорее всего, чужая). "
           "Нужен HF-токен и принятая лицензия модели (некоммерческая).", ["muscriptor", "torch"], run_muscriptor_mix, multi_role=True),
    Engine("muscriptor_stem", "MuScriptor (по стемам)", "pitched", ["bass", "drums", "guitar", "piano", "vocals"], (200, 120, 255),
           "MuScriptor отдельно на каждом стеме Demucs, модели запрещено выдавать чужие инструменты. "
           "Надёжнее для баса и пианино, дольше.",
           ["muscriptor", "torch"], run_muscriptor_stem, default_on=True),
    Engine("basic_pitch", "Basic Pitch (Spotify)", "pitched", list(PITCHED_ROLES), (255, 200, 60),
           "Полифония, любой инструмент. Даёт несколько кандидатов и карту вероятностей.", ["basic_pitch", "onnxruntime"], run_basic_pitch),
    Engine("crepe", "CREPE (монофония)", "pitched", ["bass", "vocals"], (120, 220, 255),
           "Нейросеть для одноголосных партий: бас, вокал. Точная высота, одна нота в момент времени.", ["torchcrepe", "torch"], run_crepe),
    Engine("pyin", "pYIN (монофония, без нейросети)", "pitched", ["bass", "vocals"], (120, 255, 160),
           "Классический алгоритм для одноголосных партий. Быстрый, без моделей.", ["librosa"], run_pyin, default_on=False),
    Engine("piano_bytedance", "Piano Transcription (ByteDance)", "pitched", ["piano"], (255, 120, 90),
           "Лучшая модель для фортепиано. Первый запуск скачивает ~170 МБ в models.", ["piano_transcription_inference", "torch"], run_piano_bytedance),
    Engine("drums_heuristic", "Ударные: детектор ударов", "drums", ["drums"], (255, 200, 60),
           "Поиск ударов + классификация бочка/малый/хэт/тарелка по спектру.", ["librosa"], run_drums_heuristic),
]
ENGINE_BY_ID = {e.id: e for e in ENGINES}


def engine_color(engine_id: str) -> tuple[int, int, int]:
    e = ENGINE_BY_ID.get(engine_id)
    return e.color if e else (255, 200, 60)


def run_engine(engine: Engine, path: Path, mono: np.ndarray, sr: int, role: str, errors: list[str] | None = None):
    import time
    t0 = time.time()
    try:
        log.info("engine %s start: role=%s file=%s", engine.id, role, path)
        result = engine.run(path, mono, sr, role)
        if engine.multi_role:
            log.info("engine %s done in %.1fs: %s", engine.id, time.time() - t0, {r: len(v[0]) for r, v in result.items()})
        else:
            log.info("engine %s done in %.1fs: role=%s notes=%d", engine.id, time.time() - t0, role, len(result[0]))
        return result
    except Exception as e:
        log.exception("engine %s failed (role=%s)", engine.id, role)
        traceback.print_exc()
        if errors is not None:
            msg = str(e).strip().splitlines()
            errors.append(f"{engine.name}: " + (msg[0] if msg else type(e).__name__) + ("\n   " + "\n   ".join(msg[1:8]) if len(msg) > 1 else ""))
        return ({}, None) if engine.multi_role else ([], None)
