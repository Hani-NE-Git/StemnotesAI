"""VST3 instrument hosting through Pedalboard: offline rendering of MIDI tracks and short previews."""
from __future__ import annotations

import base64
import threading
from pathlib import Path

import numpy as np


class VSTInstrument:
    def __init__(self, path: str, plugin_name: str | None = None):
        from pedalboard import load_plugin
        self.path = path
        self.lock = threading.Lock()
        self.plugin = load_plugin(path, plugin_name=plugin_name) if plugin_name else load_plugin(path)
        if not getattr(self.plugin, "is_instrument", True):
            raise ValueError("Этот плагин не инструмент (это эффект)")

    @staticmethod
    def plugin_names(path: str) -> list[str]:
        from pedalboard import VST3Plugin
        try:
            return list(VST3Plugin.get_plugin_names_for_file(path))
        except Exception:
            return []

    @property
    def name(self) -> str:
        return getattr(self.plugin, "name", Path(self.path).stem)

    def get_state(self) -> str | None:
        try:
            return base64.b64encode(bytes(self.plugin.raw_state)).decode("ascii")
        except Exception:
            return None

    def set_state(self, state_b64: str | None) -> None:
        if not state_b64:
            return
        try:
            self.plugin.raw_state = base64.b64decode(state_b64)
        except Exception:
            pass

    def show_editor(self) -> None:
        """Blocks until the plugin window is closed."""
        with self.lock:
            self.plugin.show_editor()

    def render(self, events: list[tuple[float, float, int, int]], sample_rate: int, tail: float = 2.0) -> np.ndarray:
        """events: (start_sec, end_sec, pitch, velocity) with start_sec >= 0. Returns (n, 2) float32."""
        if not events:
            return np.zeros((0, 2), dtype=np.float32)
        msgs: list[tuple[bytes, float]] = []
        end = 0.0
        for s, e, p, v in events:
            s = max(0.0, s)
            e = max(s + 0.01, e)
            msgs.append((bytes([0x90, int(p) & 0x7F, int(max(1, min(127, v)))]), s))
            msgs.append((bytes([0x80, int(p) & 0x7F, 0]), e))
            end = max(end, e)
        msgs.sort(key=lambda m: m[1])
        duration = end + tail
        with self.lock:
            out = self.plugin(msgs, duration, float(sample_rate), num_channels=2, reset=True)
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 1:
            out = out[None, :]
        if out.shape[0] == 1:
            out = np.repeat(out, 2, axis=0)
        return np.ascontiguousarray(out[:2].T)
