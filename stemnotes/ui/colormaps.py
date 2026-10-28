"""Small colormaps and QImage builders for pitch-aligned images."""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

from ..project import PitchImage


def _lut(stops: list[tuple[float, tuple[int, int, int]]], alpha_curve: str = "linear") -> np.ndarray:
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=np.float32)
    t = np.linspace(0, 1, 256)
    rgb = np.stack([np.interp(t, xs, cols[:, i]) for i in range(3)], axis=1)
    if alpha_curve == "linear":
        a = t
    else:
        a = np.sqrt(t)
    a = (a * 255).clip(0, 255)
    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[:, 0] = rgb[:, 2]  # B (QImage Format_ARGB32 is BGRA in memory on little-endian)
    lut[:, 1] = rgb[:, 1]
    lut[:, 2] = rgb[:, 0]
    lut[:, 3] = a
    return lut


SPECTRO_LUT = _lut([(0.0, (20, 30, 60)), (0.35, (40, 90, 160)), (0.6, (60, 180, 200)), (0.8, (230, 220, 80)), (1.0, (255, 250, 230))], "sqrt")
HEAT_LUT = _lut([(0.0, (120, 40, 20)), (0.4, (220, 90, 30)), (0.75, (255, 170, 60)), (1.0, (255, 240, 180))], "linear")


class ImageCache:
    def __init__(self):
        self._cache: dict[int, QImage] = {}
        self._keep: dict[int, np.ndarray] = {}

    def get(self, img: PitchImage, lut: np.ndarray) -> QImage:
        key = id(img) ^ id(lut)
        q = self._cache.get(key)
        if q is not None:
            return q
        data = img.data
        if data.ndim != 2 or data.size == 0:
            data = np.zeros((1, 1), dtype=np.uint8)
        flipped = data[::-1]                      # row 0 = highest pitch
        rgba = lut[flipped]                        # (rows, cols, 4)
        rgba = np.ascontiguousarray(rgba)
        h, w = rgba.shape[:2]
        q = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_ARGB32)
        self._keep[key] = rgba                     # keep buffer alive
        self._cache[key] = q
        return q

    def get_level(self, img: PitchImage, lut: np.ndarray, level: int) -> tuple[QImage, int]:
        """Down-sampled (max over 2**level columns) QImage for zoomed-out views. Returns (image, factor)."""
        factor = 1 << level
        if factor == 1:
            return self.get(img, lut), 1
        key = (id(img) ^ id(lut), level)
        q = self._cache.get(key)
        if q is not None:
            return q, factor
        data = img.data
        if data.ndim != 2 or data.size == 0:
            data = np.zeros((1, 1), dtype=np.uint8)
        rows, cols = data.shape
        n = (cols // factor) * factor
        if n == 0:
            small = data[:, :1]
        else:
            small = data[:, :n].reshape(rows, n // factor, factor).max(axis=2)
        rgba = np.ascontiguousarray(lut[small[::-1]])
        h, w = rgba.shape[:2]
        q = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_ARGB32)
        self._keep[key] = rgba
        self._cache[key] = q
        return q, factor

    def clear(self) -> None:
        self._cache.clear()
        self._keep.clear()
