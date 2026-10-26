"""Stem separation with Demucs (htdemucs_6s: drums, bass, other, vocals, guitar, piano). Models cached in app folder."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .. import paths  # noqa: F401  (sets TORCH_HOME before torch import)

DEFAULT_MODEL = "htdemucs_6s"


def gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        return "CPU (CUDA недоступна)"
    except Exception as e:
        return f"torch не установлен ({e})"


def separate(audio: np.ndarray, sr: int, out_dir: Path, model: str = DEFAULT_MODEL,
             progress: Callable[[float, str], None] | None = None) -> dict[str, Path]:
    """audio: (n, 2) float32. Writes <out_dir>/<stem>.wav and returns {stem: path}."""
    import torch
    import soundfile as sf
    from demucs.api import Separator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if progress:
        progress(0.0, f"Загрузка модели {model} ({device})…")
    sep = Separator(model=model, device=device, shifts=1, overlap=0.25, split=True, progress=False)

    def cb(d):
        if progress and d.get("state") == "end":
            seg = d.get("segment_offset", 0)
            total = d.get("audio_length", 1) or 1
            progress(min(1.0, seg / total), "Разделение на дорожки…")
    sep.update_parameter(callback=cb)

    wav = torch.from_numpy(np.ascontiguousarray(audio.T.astype(np.float32)))  # (2, n)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    origin, stems = sep.separate_tensor(wav, sr)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name, tensor in stems.items():
        data = tensor.cpu().numpy().T  # (n, 2) at sep.samplerate
        p = out_dir / f"{name}.wav"
        sf.write(str(p), data, int(sep.samplerate), subtype="PCM_16")
        result[name] = p
    if progress:
        progress(1.0, "Стемы готовы")
    return result
