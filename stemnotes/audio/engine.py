"""Playback engine: PortAudio shared-mode output (WASAPI shared on Windows, no ASIO), mixes
audio tracks, SoundFont MIDI tracks (sample-accurate events) and pre-rendered VST3 tracks."""
from __future__ import annotations

import bisect
import threading
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from ..log import get as _get_log
from ..project import Project, Track
from .loader import load_audio
from .sf2 import SF2Synth
from .vst import VSTInstrument

log = _get_log("engine")

DEFAULT_BLOCK = 1024


def list_output_devices() -> list[dict]:
    """Returns [{index, name, hostapi, default}] for output-capable devices."""
    out = []
    try:
        import sounddevice as sd
        apis = sd.query_hostapis()
        devs = sd.query_devices()
    except Exception:
        return out
    for i, d in enumerate(devs):
        if d.get("max_output_channels", 0) <= 0:
            continue
        api = apis[d["hostapi"]]["name"] if d["hostapi"] < len(apis) else "?"
        out.append({"index": i, "name": d["name"], "hostapi": api,
                    "default_sr": int(d.get("default_samplerate", 44100))})
    return out


def pick_default_device() -> int | None:
    """Prefer WASAPI default output on Windows (shared mode, low latency, system audio keeps working)."""
    try:
        import sounddevice as sd
        apis = sd.query_hostapis()
    except Exception:
        return None
    for api in apis:
        if "WASAPI" in api["name"] and api.get("default_output_device", -1) >= 0:
            return int(api["default_output_device"])
    try:
        d = sd.default.device[1]
        return int(d) if d is not None and d >= 0 else None
    except Exception:
        return None


class Engine:
    def __init__(self, shared=None):
        self.shared = shared                  # optional: object with pos/playing/rendering/end multiprocessing Values
        self.events_out: list[tuple[str, dict]] = []   # async events for the UI (vst_rendered, stopped, error)
        self.project: Project | None = None
        self.sample_rate = 44100
        self.block = DEFAULT_BLOCK
        self.device: int | None = None
        self.stream = None
        self.available = False
        self.error: str | None = None
        self.master = 0.9

        self.lock = threading.Lock()
        self.playing = False
        self.pos = 0                   # transport position, samples (song time 0 = sample 0)
        self.clock = 0                 # total samples rendered (monotonic)
        self.loop_enabled = False
        self.loop_start = 0
        self.loop_end = 0
        self.end_sample = 0

        self.audio: dict[str, np.ndarray] = {}         # track id -> (n,2)
        self.synth: SF2Synth | None = None
        self.soundfont: Path | None = None
        self.sf2_channels: dict[str, int] = {}         # track id -> channel
        self.events: list[tuple[int, int, int, int, int]] = []   # (sample, on(1)/off(0), channel, pitch, vel)
        self.event_times: list[int] = []
        self.ev_idx = 0
        self.immediate: list[tuple[int, int, int, int]] = []     # (on/off, channel, pitch, vel)
        self.preview_offs: list[tuple[int, int, int]] = []       # (clock_sample, channel, pitch)

        self.vst: dict[str, VSTInstrument] = {}
        self.vst_buffers: dict[str, np.ndarray] = {}
        self.vst_dirty: set[str] = set()
        self.vst_rendering: set[str] = set()
        self.oneshots: list[list] = []                            # [buffer, position]
        self._vst_thread: threading.Thread | None = None
        self.on_vst_rendered: Callable[[str], None] | None = None
        self.on_stopped: Callable[[], None] | None = None

    # ------------------------------------------------------------------ device
    def start(self, device: int | None = None, block: int | None = None) -> bool:
        self.stop_stream()
        if block:
            self.block = int(block)
        try:
            import sounddevice as sd
        except Exception as e:  # PortAudio missing (Linux: sudo apt install libportaudio2)
            self.available = False
            self.error = f"PortAudio не найден: {e}"
            if self.soundfont is not None:
                self.load_soundfont(self.soundfont)
            return False
        try:
            self.device = device if device is not None else pick_default_device()
            info = sd.query_devices(self.device, "output") if self.device is not None else sd.query_devices(kind="output")
            self.sample_rate = int(info.get("default_samplerate", 44100)) or 44100
            extra = None
            try:
                apis = sd.query_hostapis()
                if "WASAPI" in apis[info["hostapi"]]["name"]:
                    extra = sd.WasapiSettings(exclusive=False)
            except Exception:
                extra = None
            # Python callbacks cannot hit "low" latency reliably: ask for ~2.5 blocks of latency instead.
            latency = max(0.02, 2.5 * self.block / self.sample_rate)
            self.stream = sd.OutputStream(samplerate=self.sample_rate, blocksize=self.block, device=self.device,
                                          channels=2, dtype="float32", callback=self._callback,
                                          latency=latency, extra_settings=extra)
            self.stream.start()
            self.available = True
            self.error = None
        except Exception as e:  # no device (headless) or driver error
            self.available = False
            self.error = str(e)
            self.stream = None
            log.error("audio device open failed (device=%s block=%s): %s", self.device, self.block, e)
        if self.available:
            log.info("audio stream started: device=%s sr=%d block=%d latency=%s", self.device, self.sample_rate, self.block,
                     getattr(self.stream, "latency", "?"))
        if self.soundfont is not None:
            self.load_soundfont(self.soundfont)
        if self.project is not None:
            self.set_project(self.project)
        return self.available

    def stop_stream(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None
        self.available = False

    def load_soundfont(self, path: Path) -> None:
        self.soundfont = Path(path)
        with self.lock:
            try:
                self.synth = SF2Synth(self.soundfont, self.sample_rate)
            except Exception as e:
                self.synth = None
                self.error = f"SoundFont: {e}"
                log.error("soundfont load failed %s: %s", path, e)
            self.sf2_channels.clear()
        if self.project is not None:
            self.refresh_instruments()

    # ------------------------------------------------------------------ project
    def set_project(self, project: Project) -> None:
        self.stop()
        with self.lock:
            self.project = project
            self.audio.clear()
            self.vst_buffers.clear()
            self.vst_dirty.clear()
            self.sf2_channels.clear()
        self.reload_audio()
        self.refresh_instruments()
        self.rebuild_events()

    def update_project(self, project: Project, reload_audio: bool = False, dirty_vst: set[str] | None = None) -> None:
        """Replace project data without stopping playback (edits, mix changes, instrument changes)."""
        with self.lock:
            self.project = project
        if reload_audio:
            self.reload_audio()
        self.refresh_instruments()
        self.rebuild_events()
        if dirty_vst:
            with self.lock:
                self.vst_dirty.update(t for t in dirty_vst if t in self.vst)
            self.kick_vst_render()
        self.update_mix()

    def reload_audio(self) -> None:
        if self.project is None:
            return
        new: dict[str, np.ndarray] = {}
        for t in self.project.audio_tracks():
            p = self.project.resolve(t.path)
            if p and p.exists():
                try:
                    new[t.id] = load_audio(p, self.sample_rate)
                except Exception:
                    traceback.print_exc()
        with self.lock:
            self.audio = new
        self._update_end()

    def refresh_instruments(self) -> None:
        """Assign SF2 channels / VST instances for current midi tracks."""
        if self.project is None:
            return
        with self.lock:
            used = set(self.sf2_channels.values())
            for t in self.project.midi_tracks():
                inst = t.instrument
                if inst.get("type", "sf2") == "sf2":
                    if t.id not in self.sf2_channels:
                        for ch in range(16):
                            if ch not in used:
                                self.sf2_channels[t.id] = ch
                                used.add(ch)
                                break
                    ch = self.sf2_channels.get(t.id)
                    if self.synth is not None and ch is not None:
                        self.synth.set_program(ch, int(inst.get("bank", 0)), int(inst.get("program", 0)), bool(inst.get("drums")))
                        self.synth.set_volume(ch, t.volume if self.project.is_audible(t) else 0.0)
                else:
                    if t.id in self.sf2_channels:
                        used.discard(self.sf2_channels.pop(t.id))
                    if t.id not in self.vst and inst.get("path"):
                        try:
                            log.info("loading VST3 for track %s: %s (%s)", t.name, inst["path"], inst.get("plugin_name"))
                            v = VSTInstrument(inst["path"], inst.get("plugin_name"))
                            v.set_state(inst.get("state"))
                            self.vst[t.id] = v
                            self.vst_dirty.add(t.id)
                            log.info("VST3 loaded: %s", v.name)
                        except Exception as e:
                            log.exception("VST3 load failed for track %s", t.name)
                            self.error = f"VST3: {e}"
                            self.events_out.append(("error", {"message": f"VST3 «{inst.get('path')}»: {e}"}))
            # drop stale
            ids = {t.id for t in self.project.tracks}
            for tid in list(self.sf2_channels):
                if tid not in ids:
                    self.sf2_channels.pop(tid)
            for tid in list(self.vst):
                if tid not in ids or self.project.track_by_id(tid).instrument.get("type") == "sf2":
                    self.vst.pop(tid, None)
                    self.vst_buffers.pop(tid, None)
        self.update_mix()
        self.kick_vst_render()

    def replace_vst(self, track: Track, path: str, plugin_name: str | None) -> VSTInstrument:
        log.info("replace_vst track=%s path=%s plugin=%s", track.name, path, plugin_name)
        v = VSTInstrument(path, plugin_name)
        log.info("VST3 loaded: %s (instrument=%s)", v.name, getattr(v.plugin, "is_instrument", "?"))
        track.instrument = {"type": "vst3", "path": path, "plugin_name": plugin_name, "state": None, "name": v.name}
        with self.lock:
            self.vst[track.id] = v
            self.vst_buffers.pop(track.id, None)
            self.vst_dirty.add(track.id)
            self.sf2_channels.pop(track.id, None)
        self.rebuild_events()
        self.kick_vst_render()
        return v

    def update_mix(self) -> None:
        """Apply mute/solo/volume (cheap; call on any change)."""
        if self.project is None:
            return
        with self.lock:
            for t in self.project.midi_tracks():
                ch = self.sf2_channels.get(t.id)
                if ch is not None and self.synth is not None:
                    audible = self.project.is_audible(t)
                    self.synth.set_volume(ch, t.volume if audible else 0.0)
                    if not audible:
                        self.synth.all_notes_off(ch)

    def _update_end(self) -> None:
        end = 0
        for a in self.audio.values():
            end = max(end, len(a))
        if self.project is not None:
            for t in self.project.midi_tracks():
                for n in t.notes:
                    end = max(end, int((self.project.beat_to_sec(n.end) + 2.0) * self.sample_rate))
        self.end_sample = max(end, self.sample_rate * 10)
        if self.shared is not None:
            self.shared.end.value = self.end_sample / self.sample_rate

    def rebuild_events(self) -> None:
        """Rebuild the SF2 event list from project notes (call after any note edit)."""
        if self.project is None:
            return
        ev: list[tuple[int, int, int, int, int]] = []
        sr = self.sample_rate
        for t in self.project.midi_tracks():
            ch = self.sf2_channels.get(t.id)
            if ch is None:
                continue
            for n in t.notes:
                s = int(self.project.beat_to_sec(n.start) * sr)
                e = int(self.project.beat_to_sec(n.end) * sr)
                if e <= s:
                    e = s + 1
                ev.append((s, 1, ch, n.pitch, n.velocity))
                ev.append((e, 0, ch, n.pitch, 0))
        ev.sort(key=lambda x: (x[0], x[1]))
        with self.lock:
            self.events = ev
            self.event_times = [x[0] for x in ev]
            self.ev_idx = bisect.bisect_left(self.event_times, self.pos)
        self._update_end()

    def mark_vst_dirty(self, track_id: str) -> None:
        with self.lock:
            if track_id in self.vst:
                self.vst_dirty.add(track_id)

    def mark_all_vst_dirty(self) -> None:
        with self.lock:
            self.vst_dirty.update(self.vst.keys())

    # ------------------------------------------------------------------ VST rendering (background)
    def kick_vst_render(self) -> None:
        if self._vst_thread is not None and self._vst_thread.is_alive():
            return
        with self.lock:
            if not self.vst_dirty:
                return
        self._vst_thread = threading.Thread(target=self._vst_worker, daemon=True)
        self._vst_thread.start()

    def _vst_worker(self) -> None:
        while True:
            with self.lock:
                if not self.vst_dirty or self.project is None:
                    return
                tid = self.vst_dirty.pop()
                self.vst_rendering.add(tid)
                inst = self.vst.get(tid)
                track = self.project.track_by_id(tid)
                proj = self.project
            if inst is None or track is None:
                with self.lock:
                    self.vst_rendering.discard(tid)
                continue
            if self.shared is not None:
                self.shared.rendering.value = len(self.vst_rendering)
            try:
                import time
                t0 = time.time()
                events = [(proj.beat_to_sec(n.start), proj.beat_to_sec(n.end), n.pitch, n.velocity) for n in track.notes]
                events = [e for e in events if e[1] > 0]
                buf = inst.render(events, self.sample_rate) if events else np.zeros((0, 2), dtype=np.float32)
                peak = float(np.abs(buf).max()) if len(buf) else 0.0
                log.info("VST3 render track=%s notes=%d -> %.1fs audio, peak=%.3f, took %.1fs", track.name, len(events),
                         len(buf) / self.sample_rate, peak, time.time() - t0)
                if events and peak < 1e-4:
                    self.events_out.append(("warning", {"message": f"VST3 «{inst.name}» на дорожке «{track.name}» вернул тишину. "
                                                        "Откройте окно плагина и выберите пресет/звук."}))
            except Exception as e:
                log.exception("VST3 render failed for track %s", track.name)
                self.events_out.append(("error", {"message": f"Рендер VST3 «{track.name}»: {e}"}))
                buf = np.zeros((0, 2), dtype=np.float32)
            with self.lock:
                self.vst_buffers[tid] = buf
                self.vst_rendering.discard(tid)
            if self.shared is not None:
                self.shared.rendering.value = len(self.vst_rendering)
            self.events_out.append(("vst_rendered", {"track_id": tid}))
            if self.on_vst_rendered:
                try:
                    self.on_vst_rendered(tid)
                except Exception:
                    pass

    # ------------------------------------------------------------------ transport
    @property
    def position(self) -> float:
        return self.pos / self.sample_rate

    def play(self) -> None:
        if self.project is None:
            return
        self.kick_vst_render()
        with self.lock:
            self.ev_idx = bisect.bisect_left(self.event_times, self.pos)
            self.playing = True
        if self.shared is not None:
            self.shared.playing.value = True

    def stop(self) -> None:
        with self.lock:
            self.playing = False
            if self.synth is not None:
                self.synth.all_notes_off()
        if self.shared is not None:
            self.shared.playing.value = False

    def toggle(self) -> None:
        if self.playing:
            self.stop()
        else:
            self.play()

    def seek(self, sec: float) -> None:
        with self.lock:
            self.pos = max(0, int(sec * self.sample_rate))
            self.ev_idx = bisect.bisect_left(self.event_times, self.pos)
            if self.synth is not None:
                self.synth.all_notes_off()
        if self.shared is not None:
            self.shared.pos.value = self.pos / self.sample_rate

    def set_loop(self, enabled: bool, start_sec: float | None = None, end_sec: float | None = None) -> None:
        with self.lock:
            self.loop_enabled = enabled
            if start_sec is not None:
                self.loop_start = max(0, int(start_sec * self.sample_rate))
            if end_sec is not None:
                self.loop_end = max(0, int(end_sec * self.sample_rate))
            if self.loop_end <= self.loop_start:
                self.loop_enabled = False

    # ------------------------------------------------------------------ preview
    def preview_note(self, track: Track, pitch: int, velocity: int = 100, duration: float = 0.4) -> None:
        if track.kind != "midi":
            return
        if track.instrument.get("type", "sf2") == "sf2":
            ch = self.sf2_channels.get(track.id)
            if ch is None or self.synth is None:
                return
            with self.lock:
                self.immediate.append((1, ch, pitch, velocity))
                self.preview_offs.append((self.clock + int(duration * self.sample_rate), ch, pitch))
        else:
            inst = self.vst.get(track.id)
            if inst is None:
                return

            def work():
                try:
                    buf = inst.render([(0.0, duration, pitch, velocity)], self.sample_rate, tail=1.0)
                    with self.lock:
                        self.oneshots.append([buf * float(track.volume), 0])
                except Exception as e:
                    log.exception("VST3 preview failed")
                    self.events_out.append(("error", {"message": f"VST3 предпрослушивание: {e}"}))
            threading.Thread(target=work, daemon=True).start()

    def preview_off(self, track: Track, pitch: int) -> None:
        ch = self.sf2_channels.get(track.id)
        if ch is not None:
            with self.lock:
                self.immediate.append((0, ch, pitch, 0))

    def test_vst(self, track: Track) -> dict:
        """Render one note through the track's VST3 and report what came back (for diagnostics)."""
        inst = self.vst.get(track.id)
        if inst is None:
            return {"ok": False, "error": "на дорожке нет VST3-инструмента"}
        try:
            import time
            t0 = time.time()
            buf = inst.render([(0.0, 0.5, 60, 100)], self.sample_rate, tail=1.0)
            peak = float(np.abs(buf).max()) if len(buf) else 0.0
            log.info("VST3 test %s: %d samples, peak %.4f, %.2fs", inst.name, len(buf), peak, time.time() - t0)
            return {"ok": True, "name": inst.name, "samples": int(len(buf)), "peak": peak, "seconds": time.time() - t0,
                    "params": len(getattr(inst.plugin, "parameters", {}) or {})}
        except Exception as e:
            log.exception("VST3 test failed")
            return {"ok": False, "name": inst.name, "error": str(e)}

    # ------------------------------------------------------------------ callback
    def _callback(self, outdata, frames, time_info, status) -> None:
        try:
            out = self._render(frames)
            outdata[:] = out
        except Exception:
            traceback.print_exc()
            outdata.fill(0)

    def _render(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, 2), dtype=np.float32)
        with self.lock:
            proj = self.project
            # immediate (preview) events
            if self.synth is not None:
                for on, ch, p, v in self.immediate:
                    if on:
                        self.synth.note_on(ch, p, v)
                    else:
                        self.synth.note_off(ch, p)
                self.immediate.clear()
            stopped = False
            if self.playing and proj is not None:
                done = 0
                while done < frames:
                    seg = frames - done
                    if self.loop_enabled and self.loop_end > self.pos:
                        seg = min(seg, self.loop_end - self.pos)
                    self._render_segment(proj, self.pos, seg, out[done:done + seg])
                    self.pos += seg
                    done += seg
                    if self.loop_enabled and self.pos >= self.loop_end:
                        self.pos = self.loop_start
                        if self.synth is not None:
                            self.synth.all_notes_off()
                        self.ev_idx = bisect.bisect_left(self.event_times, self.pos)
                    elif self.pos >= self.end_sample:
                        self.playing = False
                        stopped = True
                        if self.synth is not None:
                            self.synth.all_notes_off()
                        break
            else:
                if self.synth is not None:
                    self._synth_into(out, 0, frames)
            # preview note-offs (absolute clock)
            if self.preview_offs and self.synth is not None:
                keep = []
                for c, ch, p in self.preview_offs:
                    if c <= self.clock + frames:
                        self.synth.note_off(ch, p)
                    else:
                        keep.append((c, ch, p))
                self.preview_offs = keep
            # one-shots
            if self.oneshots:
                keep = []
                for item in self.oneshots:
                    buf, p = item
                    n = min(frames, len(buf) - p)
                    if n > 0:
                        out[:n] += buf[p:p + n]
                        item[1] = p + n
                        if item[1] < len(buf):
                            keep.append(item)
                self.oneshots = keep
            self.clock += frames
        if self.shared is not None:
            self.shared.pos.value = self.pos / self.sample_rate
            if stopped:
                self.shared.playing.value = False
        if stopped:
            self.events_out.append(("stopped", {}))
        np.multiply(out, self.master, out=out)
        peak = float(np.abs(out).max()) if frames else 0.0
        if peak > 0.98:               # gentle per-block limiter: no harsh clipping when many stems are summed
            gain = 0.98 / peak
            self._limiter_gain = min(getattr(self, "_limiter_gain", 1.0), gain)
        else:
            self._limiter_gain = min(1.0, getattr(self, "_limiter_gain", 1.0) * 1.02)
        if self._limiter_gain < 1.0:
            np.multiply(out, np.float32(self._limiter_gain), out=out)
        np.clip(out, -1.0, 1.0, out=out)
        if stopped and self.on_stopped:
            try:
                self.on_stopped()
            except Exception:
                pass
        return out

    def _synth_into(self, out: np.ndarray, offset: int, n: int) -> None:
        if n <= 0 or self.synth is None:
            return
        out[offset:offset + n] += self.synth.render(n)

    def _render_segment(self, proj: Project, pos: int, n: int, out: np.ndarray) -> None:
        # audio tracks
        for t in proj.audio_tracks():
            if not proj.is_audible(t):
                continue
            a = self.audio.get(t.id)
            if a is None or pos >= len(a):
                continue
            m = min(n, len(a) - pos)
            out[:m] += a[pos:pos + m] * np.float32(t.volume)
        # vst tracks
        for t in proj.midi_tracks():
            if t.id in self.vst_buffers and proj.is_audible(t):
                b = self.vst_buffers[t.id]
                if pos < len(b):
                    m = min(n, len(b) - pos)
                    out[:m] += b[pos:pos + m] * np.float32(t.volume)
        # soundfont tracks with sample-accurate events
        if self.synth is None:
            return
        ev = self.events
        cur = 0
        end = pos + n
        while self.ev_idx < len(ev) and ev[self.ev_idx][0] < end:
            s, on, ch, p, v = ev[self.ev_idx]
            local = max(0, s - pos)
            if local > cur:
                self._synth_into(out, cur, local - cur)
                cur = local
            if on:
                self.synth.note_on(ch, p, v)
            else:
                self.synth.note_off(ch, p)
            self.ev_idx += 1
        if cur < n:
            self._synth_into(out, cur, n - cur)
