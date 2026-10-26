"""Signal features that help to see notes: tempo/beats, pitch-aligned CQT spectrogram."""
from __future__ import annotations

import numpy as np

from ..project import PitchImage, PITCH_LOW, N_PITCHES

ANALYSIS_SR = 22050


def detect_tempo(mono: np.ndarray, sr: int) -> tuple[float, list[float]]:
    """Returns (bpm, beat_times_sec)."""
    import librosa
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, units="time")
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 120.0
    if bpm <= 0:
        bpm = 120.0
    beat_times = [float(b) for b in np.atleast_1d(beats)]
    return bpm, beat_times


def refine_bpm(beat_times: list[float], bpm_guess: float) -> tuple[float, float]:
    """Fit a straight line through beat indices -> (bpm, offset_sec of beat 0)."""
    if len(beat_times) < 4:
        return bpm_guess, (beat_times[0] if beat_times else 0.0)
    t = np.asarray(beat_times)
    idx = np.arange(len(t))
    a, b = np.polyfit(idx, t, 1)  # t = a*i + b
    if a <= 0:
        return bpm_guess, float(t[0])
    bpm = 60.0 / a
    offset = float(b)
    # move offset into [0, period)
    period = 60.0 / bpm
    while offset < 0:
        offset += period
    while offset >= period:
        offset -= period
    return float(bpm), offset


def cqt_image(mono: np.ndarray, sr: int, rows_per_semitone: int = 3, hop: int = 512) -> PitchImage:
    """Log-frequency spectrogram with rows aligned to piano keys (A0..C8). uint8 dB-scaled."""
    import librosa
    bins_per_octave = 12 * rows_per_semitone
    n_bins = N_PITCHES * rows_per_semitone
    # centre of row 0 must be at PITCH_LOW: fmin is lower edge -> shift by half a semitone-step
    fmin = librosa.midi_to_hz(PITCH_LOW - 0.5 + 0.5 / rows_per_semitone)
    try:
        c = np.abs(librosa.cqt(mono, sr=sr, hop_length=hop, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave))
    except Exception:
        # too few samples etc.
        return PitchImage(np.zeros((n_bins, 1), dtype=np.uint8), 0.0, hop / sr, PITCH_LOW - 0.5, rows_per_semitone)
    db = librosa.amplitude_to_db(c, ref=np.max, top_db=70.0)
    img = ((db + 70.0) / 70.0 * 255.0).clip(0, 255).astype(np.uint8)
    return PitchImage(img, 0.0, hop / sr, PITCH_LOW - 0.5, rows_per_semitone)
