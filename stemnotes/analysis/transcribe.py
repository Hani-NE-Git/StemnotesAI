"""Ghost-note candidates: Basic Pitch for pitched stems (several candidates with confidence),
simple onset/band-energy classifier for drums."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..project import GhostNote, PitchImage, PITCH_LOW, N_PITCHES


def transcribe_pitched(path: Path, min_conf: float = 0.15) -> tuple[list[GhostNote], PitchImage]:
    """Returns (ghost notes, note-activation heatmap). Low thresholds on purpose: we want candidates, the user decides."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict, Model
    from basic_pitch.note_creation import model_frames_to_time

    model = Model(ICASSP_2022_MODEL_PATH)
    model_output, _midi, note_events = predict(
        str(path), model,
        onset_threshold=0.3, frame_threshold=0.2, minimum_note_length=60.0,
        minimum_frequency=None, maximum_frequency=None, multiple_pitch_bends=False, melodia_trick=True,
    )
    ghosts: list[GhostNote] = []
    for ev in note_events:
        start, end, pitch, amp = float(ev[0]), float(ev[1]), int(ev[2]), float(ev[3])
        conf = float(np.clip(amp, 0.0, 1.0))
        if conf < min_conf:
            continue
        ghosts.append(GhostNote(start, end, pitch, conf))
    ghosts.sort(key=lambda g: g.start)

    note_act = np.asarray(model_output["note"], dtype=np.float32)  # (frames, 88)
    n_frames = note_act.shape[0]
    times = model_frames_to_time(n_frames)
    dt = float(times[1] - times[0]) if n_frames > 1 else 0.0116
    # resample to a uniform grid (tiny per-window offsets in basic pitch)
    grid = np.arange(0.0, float(times[-1]) if n_frames else 0.0, dt)
    idx = np.clip(np.searchsorted(times, grid), 0, max(0, n_frames - 1))
    uniform = note_act[idx] if n_frames else np.zeros((1, 88), dtype=np.float32)
    img = (uniform.T * 255.0).clip(0, 255).astype(np.uint8)  # rows = pitch 21..108
    if img.shape[0] != N_PITCHES:
        fixed = np.zeros((N_PITCHES, img.shape[1]), dtype=np.uint8)
        m = min(N_PITCHES, img.shape[0])
        fixed[:m] = img[:m]
        img = fixed
    return ghosts, PitchImage(img, 0.0, dt, PITCH_LOW - 0.5, 1)


KICK, SNARE, HAT_CLOSED, HAT_OPEN, CRASH, RIDE = 36, 38, 42, 46, 49, 51


def transcribe_drums(mono: np.ndarray, sr: int) -> list[GhostNote]:
    """Onset detection + spectral band energies -> kick / snare / hi-hat / crash candidates with confidences."""
    import librosa
    hop = 256
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="frames",
                                        backtrack=False, pre_max=3, post_max=3, pre_avg=10, post_avg=10,
                                        delta=0.04, wait=2)
    if len(onsets) == 0:
        return []
    env_max = float(env.max()) if env.max() > 0 else 1.0
    win = int(0.06 * sr)
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = lambda lo, hi: (freqs >= lo) & (freqs < hi)
    b_kick, b_snare, b_hat, b_cymb = band(40, 130), band(150, 900), band(5000, 11000), band(2500, 5000)
    ghosts: list[GhostNote] = []
    onset_times = [float(f * hop / sr) for f in onsets]
    for k, f in enumerate(onsets):
        t = onset_times[k]
        t_next = onset_times[k + 1] if k + 1 < len(onset_times) else t + 1.0
        a = int(t * sr)
        seg = mono[a:a + win]
        if len(seg) < 256:
            continue
        seg = seg * np.hanning(len(seg))
        spec = np.abs(np.fft.rfft(seg, n_fft)) ** 2
        total = spec.sum() + 1e-9
        e_kick, e_snare, e_hat, e_cymb = spec[b_kick].sum() / total, spec[b_snare].sum() / total, spec[b_hat].sum() / total, spec[b_cymb].sum() / total
        strength = float(np.sqrt(np.clip(env[f] / env_max, 0.03, 1.0)))
        # decay length distinguishes crash (long) from hat (short); stop before the next hit
        long_seg = mono[a:a + int(min(0.4, max(0.1, t_next - t - 0.02)) * sr)]
        decay = 0.0
        if len(long_seg) > win * 2:
            tail = np.abs(long_seg[win:]).mean() + 1e-9
            head = np.abs(long_seg[:win]).mean() + 1e-9
            decay = float(tail / head)
        cands: list[tuple[int, float]] = []
        if e_kick > 0.25:
            cands.append((KICK, min(1.0, e_kick * 1.6) * strength))
        if e_snare > 0.25:
            cands.append((SNARE, min(1.0, e_snare * 1.6) * strength))
        if e_hat + e_cymb > 0.3:
            hi = e_hat + e_cymb
            if decay > 0.35:
                cands.append((CRASH, min(1.0, hi * 1.3) * strength))
            else:
                cands.append((HAT_CLOSED, min(1.0, hi * 1.3) * strength))
        if not cands:
            # weak unknown hit: pick the dominant band at low confidence
            best = max([(KICK, e_kick), (SNARE, e_snare), (HAT_CLOSED, e_hat + e_cymb)], key=lambda x: x[1])
            cands.append((best[0], 0.2 * strength))
        for pitch, conf in cands:
            ghosts.append(GhostNote(t, t + 0.1, pitch, float(np.clip(conf, 0.05, 1.0))))
    return ghosts
