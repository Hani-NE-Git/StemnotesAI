"""Audio file loading with caching (mp3/wav/flac/ogg/m4a via pedalboard)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_cache: dict[tuple[str, int], np.ndarray] = {}


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Returns float32 array of shape (n, 2) resampled to sample_rate."""
    key = (str(Path(path).resolve()), int(sample_rate))
    if key in _cache:
        return _cache[key]
    from pedalboard.io import AudioFile
    with AudioFile(str(path)) as f:
        with f.resampled_to(sample_rate) as r:
            data = r.read(int(r.frames))  # (channels, n)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] == 1:
        data = np.repeat(data, 2, axis=0)
    elif data.shape[0] > 2:
        data = data[:2]
    out = np.ascontiguousarray(data.T)
    _cache[key] = out
    return out


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    st = load_audio(path, sample_rate)
    return st.mean(axis=1).astype(np.float32)


def clear_cache() -> None:
    _cache.clear()


def waveform_peaks(audio: np.ndarray, sample_rate: int, px_per_sec: float, t_start: float, n_px: int) -> np.ndarray:
    """Min/max peaks per pixel column. Returns (n_px, 2) array in [-1, 1]."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    out = np.zeros((n_px, 2), dtype=np.float32)
    spp = sample_rate / px_per_sec  # samples per pixel
    s0 = int(t_start * sample_rate)
    n = len(mono)
    if spp < 1:
        spp = 1
    for i in range(n_px):
        a = int(s0 + i * spp)
        b = int(s0 + (i + 1) * spp)
        if b <= 0 or a >= n:
            continue
        a = max(0, a)
        b = min(n, max(b, a + 1))
        seg = mono[a:b]
        out[i, 0] = seg.min()
        out[i, 1] = seg.max()
    return out
