"""SoundFont synthesizer wrapper (TinySoundFont, renders into numpy buffers, no own audio driver)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class SF2Synth:
    def __init__(self, soundfont: Path, sample_rate: int):
        import tinysoundfont
        self.sample_rate = int(sample_rate)
        self.synth = tinysoundfont.Synth(gain=-6.0, samplerate=self.sample_rate)
        self.sfid = self.synth.sfload(str(soundfont))
        self.path = Path(soundfont)
        self.channel_programs: dict[int, tuple[int, int, bool]] = {}
        self._buf = bytearray()

    def set_program(self, channel: int, bank: int, program: int, drums: bool) -> None:
        key = (bank, program, drums)
        if self.channel_programs.get(channel) == key:
            return
        try:
            self.synth.program_select(channel, self.sfid, bank, program, is_drums=drums)
        except Exception:
            # fall back to bank 0 / program 0 if preset missing
            try:
                self.synth.program_select(channel, self.sfid, 0, 0 if not drums else 0, is_drums=drums)
            except Exception:
                pass
        self.channel_programs[channel] = key

    def set_volume(self, channel: int, volume: float) -> None:
        v = int(max(0.0, min(1.0, volume)) * 127)
        try:
            self.synth.control_change(channel, 7, v)
        except Exception:
            pass

    def note_on(self, channel: int, pitch: int, velocity: int) -> None:
        try:
            self.synth.noteon(channel, int(pitch), int(max(1, min(127, velocity))))
        except Exception:
            pass

    def note_off(self, channel: int, pitch: int) -> None:
        try:
            self.synth.noteoff(channel, int(pitch))
        except Exception:
            pass

    def all_notes_off(self, channel: int | None = None) -> None:
        try:
            self.synth.notes_off(channel)
        except Exception:
            pass

    def render(self, frames: int) -> np.ndarray:
        """Returns (frames, 2) float32."""
        need = frames * 2 * 4
        if len(self._buf) != need:
            self._buf = bytearray(need)
        mv = memoryview(self._buf)
        self.synth.generate_simple(frames, buffer=mv)
        return np.frombuffer(self._buf, dtype=np.float32).reshape(frames, 2)

    def preset_name(self, bank: int, program: int) -> str | None:
        try:
            return self.synth.sfpreset_name(self.sfid, bank, program)
        except Exception:
            return None
