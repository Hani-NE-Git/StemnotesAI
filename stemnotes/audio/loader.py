"""Audio file loading with caching (mp3/wav/flac/ogg/m4a via pedalboard)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_cache: dict[tuple[str, int], np.ndarray] = {}


def decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode at native sample rate: (n, channels) float32, sr. wav/flac via soundfile, others via pedalboard."""
    path = Path(path)
    if path.suffix.lower() in (".wav", ".flac", ".ogg"):
        try:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return np.ascontiguousarray(data), int(sr)
        except Exception:
            pass
    from pedalboard.io import AudioFile
    with AudioFile(str(path)) as f:
        data = f.read(int(f.frames))      # (channels, n)
        sr = int(f.samplerate)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    return np.ascontiguousarray(data.T), sr


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Returns float32 array of shape (n, 2) resampled to sample_rate (soxr, fast)."""
    key = (str(Path(path).resolve()), int(sample_rate))
    if key in _cache:
        return _cache[key]
    data, sr = decode(path)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    if sr != sample_rate:
        import soxr
        data = soxr.resample(data, sr, sample_rate).astype(np.float32)
    out = np.ascontiguousarray(data, dtype=np.float32)
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


def load_peaks(path: Path, sample_rate: int = 0, chunk: int = 64) -> tuple[np.ndarray, np.ndarray, int]:
    """Min/max per chunk of the mono signal at the file's native rate (no resampling, nothing cached)."""
    data, sr = decode(path)
    mono = data.mean(axis=1)
    n = (len(mono) // chunk) * chunk
    if n == 0:
        mins, maxs = np.zeros(1, np.float32), np.zeros(1, np.float32)
    else:
        m = mono[:n].reshape(-1, chunk)
        mins, maxs = m.min(axis=1), m.max(axis=1)
    return mins.astype(np.float32), maxs.astype(np.float32), int(sr)
