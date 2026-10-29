"""Audio server: owns the Engine (audio device, SoundFont, VST3 instruments). Runs in its own process so the
user interface can never starve the audio callback. The UI talks to it through AudioHost."""
from __future__ import annotations

import base64
import hashlib
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from ..log import get as _get_log
from ..project import Project
from .engine import Engine, list_output_devices

log = _get_log("audio-server")


def make_shared(ctx=None):
    """Shared transport state (works with or without a child process)."""
    import multiprocessing
    ctx = ctx or multiprocessing
    return SimpleNamespace(pos=ctx.Value("d", 0.0), playing=ctx.Value("b", False),
                           rendering=ctx.Value("i", 0), end=ctx.Value("d", 10.0))


def _notes_hash(track: dict) -> str:
    return hashlib.md5(json.dumps(track.get("notes", []), separators=(",", ":")).encode()).hexdigest()


class AudioServer:
    def __init__(self, shared):
        self.engine = Engine(shared=shared)
        self.project: Project | None = None
        self._notes_hash: dict[str, str] = {}
        self._audio_paths: dict[str, str] = {}
        self._tempo: tuple[float, float] = (0.0, 0.0)
        self.editor_open = False

    # ------------------------------------------------------------------ dispatch
    def handle(self, cmd: dict):
        op = cmd.get("op")
        fn = getattr(self, "op_" + str(op), None)
        if fn is None:
            raise ValueError(f"unknown op {op}")
        return fn(**{k: v for k, v in cmd.items() if k != "op"})

    # ------------------------------------------------------------------ ops
    def op_start(self, device=None, block=None):
        self.engine.start(device, block)
        return {"available": self.engine.available, "error": self.engine.error,
                "sample_rate": self.engine.sample_rate, "block": self.engine.block}

    def op_stop_stream(self):
        self.engine.stop_stream()
        return True

    def op_devices(self):
        return list_output_devices()

    def op_load_soundfont(self, path):
        self.engine.load_soundfont(Path(path))
        return {"error": self.engine.error}

    def op_master(self, value):
        self.engine.master = float(value)
        return True

    def op_set_project(self, directory, data):
        self.project = Project.from_dict(Path(directory), data)
        self._notes_hash = {t["id"]: _notes_hash(t) for t in data.get("tracks", []) if t.get("kind") == "midi"}
        self._audio_paths = {t["id"]: t.get("path") or "" for t in data.get("tracks", []) if t.get("kind") == "audio"}
        self._tempo = (self.project.bpm, self.project.offset)
        self.engine.set_project(self.project)
        return self._vst_info()

    def op_sync(self, directory, data):
        """Apply an edited project: reloads audio only for new/changed audio files, re-renders only VST tracks
        whose notes changed, re-applies instruments and mix."""
        project = Project.from_dict(Path(directory), data)
        notes_hash = {t["id"]: _notes_hash(t) for t in data.get("tracks", []) if t.get("kind") == "midi"}
        audio_paths = {t["id"]: t.get("path") or "" for t in data.get("tracks", []) if t.get("kind") == "audio"}
        tempo = (project.bpm, project.offset)
        dirty = {tid for tid, h in notes_hash.items() if self._notes_hash.get(tid) != h}
        if tempo != self._tempo:
            dirty = set(notes_hash)
        reload_audio = audio_paths != self._audio_paths
        self.project = project
        self._notes_hash, self._audio_paths, self._tempo = notes_hash, audio_paths, tempo
        self.engine.update_project(project, reload_audio=reload_audio, dirty_vst=dirty)
        return self._vst_info()

    def _vst_info(self) -> dict:
        return {"vst_tracks": {tid: v.name for tid, v in self.engine.vst.items()},
                "sf2_tracks": list(self.engine.sf2_channels.keys()), "error": self.engine.error}

    def op_play(self):
        self.engine.play()
        return True

    def op_stop(self):
        self.engine.stop()
        return True

    def op_seek(self, sec):
        self.engine.seek(float(sec))
        return True

    def op_set_loop(self, enabled, start=None, end=None):
        self.engine.set_loop(bool(enabled), start, end)
        return True

    def op_preview(self, track_id, pitch, velocity=100, duration=0.4):
        t = self.project.track_by_id(track_id) if self.project else None
        if t is not None:
            self.engine.preview_note(t, int(pitch), int(velocity), float(duration))
        return True

    def op_preview_off(self, track_id, pitch):
        t = self.project.track_by_id(track_id) if self.project else None
        if t is not None:
            self.engine.preview_off(t, int(pitch))
        return True

    def op_replace_vst(self, track_id, path, plugin_name=None):
        t = self.project.track_by_id(track_id) if self.project else None
        if t is None:
            return {"ok": False, "error": "дорожка не найдена"}
        try:
            v = self.engine.replace_vst(t, path, plugin_name)
            return {"ok": True, "name": v.name, "instrument": t.instrument}
        except Exception as e:
            log.exception("replace_vst failed")
            return {"ok": False, "error": str(e)}

    def op_vst_names(self, path):
        from .vst import VSTInstrument
        return VSTInstrument.plugin_names(path)

    def op_vst_state(self, track_id):
        v = self.engine.vst.get(track_id)
        return v.get_state() if v else None

    def op_show_editor(self, track_id):
        v = self.engine.vst.get(track_id)
        if v is None:
            return {"ok": False, "error": "на дорожке нет VST3-инструмента"}
        self.engine.stop()
        self.editor_open = True
        try:
            log.info("opening VST3 editor for %s", v.name)
            v.show_editor()
            log.info("VST3 editor closed")
        except Exception as e:
            log.exception("show_editor failed")
            return {"ok": False, "error": str(e)}
        finally:
            self.editor_open = False
        self.engine.mark_vst_dirty(track_id)
        self.engine.kick_vst_render()
        return {"ok": True, "state": v.get_state()}

    def op_test_vst(self, track_id):
        t = self.project.track_by_id(track_id) if self.project else None
        if t is None:
            return {"ok": False, "error": "дорожка не найдена"}
        return self.engine.test_vst(t)

    def op_close_editor(self):
        """Ask every top-level window of this process to close (the Pedalboard editor has no API for that)."""
        import os
        import sys
        if not sys.platform.startswith("win"):
            return {"ok": False, "error": "только Windows"}
        import ctypes
        import ctypes.wintypes as wt
        user32 = ctypes.windll.user32
        WM_CLOSE = 0x0010
        pid = os.getpid()
        closed = []
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)

        def cb(hwnd, lparam):
            wpid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
            if wpid.value == pid and user32.IsWindowVisible(hwnd):
                user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
                closed.append(int(hwnd))
            return True
        user32.EnumWindows(EnumWindowsProc(cb), 0)
        log.info("close_editor: WM_CLOSE sent to %d window(s)", len(closed))
        return {"ok": True, "windows": len(closed)}

    def op_editor_state(self):
        return {"open": self.editor_open}

    def op_ping(self):
        return "pong"

    def op_quit(self):
        self.engine.stop()
        self.engine.stop_stream()
        return True

    def drain_events(self) -> list[tuple[str, dict]]:
        ev = self.engine.events_out
        if not ev:
            return []
        self.engine.events_out = []
        return ev


PLUGIN_OPS = {"replace_vst", "show_editor", "test_vst", "vst_names", "set_project", "sync", "load_soundfont"}


def run_server(conn, shared) -> None:
    """Child-process entry point. Commands that touch plugins run on the main thread (Pedalboard requires it);
    transport, preview and queries run on the dispatcher thread, so the UI never waits behind a plugin."""
    import queue
    import threading
    from .. import log as _log
    _log.setup()
    log.info("audio server process started")
    server = AudioServer(shared)
    send_lock = threading.Lock()
    plugin_queue: queue.Queue = queue.Queue()
    stop = threading.Event()

    def send(msg):
        with send_lock:
            try:
                conn.send(msg)
            except Exception:
                pass

    def execute(cid, cmd):
        try:
            result = server.handle(cmd)
            send(("reply", cid, {"ok": True, "result": result}))
        except Exception as e:
            log.exception("command failed: %s", cmd.get("op"))
            send(("reply", cid, {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}))

    def dispatcher():
        while not stop.is_set():
            try:
                if conn.poll(0.05):
                    kind, cid, cmd = conn.recv()
                    if cmd.get("op") == "quit":
                        plugin_queue.put((cid, cmd))
                        stop.set()
                        break
                    if cmd.get("op") in PLUGIN_OPS:
                        if server.editor_open:
                            send(("reply", cid, {"ok": False, "error": "editor_open"}))
                        else:
                            plugin_queue.put((cid, cmd))
                    else:
                        execute(cid, cmd)
                for name, payload in server.drain_events():
                    send(("event", name, payload))
            except (EOFError, BrokenPipeError, OSError):
                log.info("audio server: pipe closed, exiting")
                stop.set()
                plugin_queue.put(None)
                break
            except Exception:
                log.exception("audio server dispatcher error")

    threading.Thread(target=dispatcher, daemon=True, name="dispatcher").start()
    while not stop.is_set() or not plugin_queue.empty():
        try:
            item = plugin_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            break
        cid, cmd = item
        execute(cid, cmd)
        if cmd.get("op") == "quit":
            break
    try:
        server.engine.stop_stream()
    except Exception:
        pass
    log.info("audio server process exiting")
