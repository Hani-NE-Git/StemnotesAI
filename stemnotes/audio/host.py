"""AudioHost: the UI-side handle to the audio server (separate process by default, in-process fallback)."""
from __future__ import annotations

import itertools
import queue
import threading
from pathlib import Path

from ..log import get as _get_log
from ..project import Project, Track
from .server import AudioServer, make_shared, run_server

log = _get_log("audio-host")


class AudioHost:
    def __init__(self, inprocess: bool = False):
        self.inprocess = inprocess
        self.available = False
        self.error: str | None = None
        self.sample_rate = 44100
        self.block = 1024
        self.master = 0.9
        self.vst_tracks: dict[str, str] = {}      # track id -> plugin name (mirrors the server)
        self.project: Project | None = None
        self._events: queue.Queue = queue.Queue()
        self._ids = itertools.count(1)
        self._pending: dict[int, dict] = {}
        self._pending_lock = threading.Lock()
        self._proc = None
        self._conn = None
        self._server: AudioServer | None = None
        if inprocess:
            self.shared = make_shared()
            self._server = AudioServer(self.shared)
        else:
            self._spawn()

    # ------------------------------------------------------------------ process
    def _spawn(self) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.shared = make_shared(ctx)
        parent, child = ctx.Pipe()
        self._proc = ctx.Process(target=run_server, args=(child, self.shared), name="stemnotes-audio", daemon=True)
        self._proc.start()
        self._conn = parent
        threading.Thread(target=self._reader, daemon=True).start()
        try:
            # wait for the child to answer, but give up as soon as it dies
            cid = next(self._ids)
            slot = {"event": threading.Event(), "payload": None}
            with self._pending_lock:
                self._pending[cid] = slot
            self._conn.send(("call", cid, {"op": "ping"}))
            deadline = 30.0
            while not slot["event"].wait(0.1):
                deadline -= 0.1
                if not self._proc.is_alive() or deadline <= 0:
                    raise RuntimeError("процесс завершился" if not self._proc.is_alive() else "нет ответа 30 с")
            log.info("audio server process pid=%s ready", self._proc.pid)
        except Exception as e:
            log.error("audio server did not start (%s); falling back to in-process audio", e)
            self._kill()
            self.inprocess = True
            self.shared = make_shared()
            self._server = AudioServer(self.shared)

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None
        self._conn = None

    def _reader(self) -> None:
        conn = self._conn
        while conn is not None:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                self._events.put(("server_gone", {}))
                break
            kind = msg[0]
            if kind == "reply":
                _, cid, payload = msg
                with self._pending_lock:
                    slot = self._pending.get(cid)
                if slot is not None:
                    slot["payload"] = payload
                    slot["event"].set()
            elif kind == "event":
                self._events.put((msg[1], msg[2]))

    # ------------------------------------------------------------------ calls
    def call(self, op: str, wait: bool = True, timeout: float | None = 120.0, **kw):
        cmd = {"op": op, **kw}
        if self.inprocess:
            try:
                return self._server.handle(cmd)
            finally:
                for ev in self._server.drain_events():
                    self._events.put(ev)
        cid = next(self._ids)
        slot = {"event": threading.Event(), "payload": None}
        with self._pending_lock:
            self._pending[cid] = slot
        try:
            self._conn.send(("call", cid, cmd))
        except Exception as e:
            with self._pending_lock:
                self._pending.pop(cid, None)
            raise RuntimeError(f"аудиопроцесс недоступен: {e}")
        if not wait:
            return None
        if not slot["event"].wait(timeout):
            with self._pending_lock:
                self._pending.pop(cid, None)
            raise TimeoutError(f"аудиопроцесс не ответил на {op}")
        with self._pending_lock:
            self._pending.pop(cid, None)
        p = slot["payload"]
        if not p.get("ok"):
            log.error("audio op %s failed: %s\n%s", op, p.get("error"), p.get("trace", ""))
            raise RuntimeError(p.get("error"))
        return p.get("result")

    def call_async(self, op: str, done, **kw) -> None:
        """Run a (possibly slow) call in a thread; `done(result, error)` is delivered as a 'callback' event
        so the UI picks it up in its timer without blocking."""
        def work():
            try:
                r = self.call(op, timeout=None, **kw)
                self._events.put(("callback", {"fn": done, "result": r, "error": None}))
            except Exception as e:
                self._events.put(("callback", {"fn": done, "result": None, "error": str(e)}))
        threading.Thread(target=work, daemon=True).start()

    def poll_events(self) -> list[tuple[str, dict]]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    # ------------------------------------------------------------------ API used by the UI
    def start(self, device=None, block=None) -> bool:
        r = self.call("start", device=device, block=block, timeout=60)
        self.available, self.error, self.sample_rate, self.block = r["available"], r["error"], r["sample_rate"], r["block"]
        return self.available

    def stop_stream(self) -> None:
        try:
            self.call("stop_stream", timeout=10)
        except Exception:
            pass

    def devices(self) -> list[dict]:
        try:
            return self.call("devices", timeout=30)
        except Exception:
            return []

    def load_soundfont(self, path: Path) -> None:
        r = self.call("load_soundfont", path=str(path), timeout=120)
        if r.get("error"):
            self.error = r["error"]

    def set_master(self, v: float) -> None:
        self.master = v
        self.call("master", value=v, wait=False)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        if project is None:
            return
        r = self.call("set_project", directory=str(project.dir), data=project.to_json(), timeout=600)
        self.vst_tracks = r.get("vst_tracks", {})

    def sync(self) -> None:
        """Push the current project state (notes, mix, instruments, tempo) to the audio process, without waiting."""
        if self.project is None:
            return

        def done(r, err):
            if err:
                self._events.put(("error", {"message": err}))
            elif r:
                self.vst_tracks = r.get("vst_tracks", {})
        self.call_async("sync", done, directory=str(self.project.dir), data=self.project.to_json())

    # transport
    @property
    def playing(self) -> bool:
        return bool(self.shared.playing.value)

    @property
    def position(self) -> float:
        return float(self.shared.pos.value)

    @property
    def end_sec(self) -> float:
        return float(self.shared.end.value)

    @property
    def vst_rendering(self) -> int:
        return int(self.shared.rendering.value)

    def play(self) -> None:
        self.call("play", wait=False)

    def stop(self) -> None:
        self.call("stop", wait=False)
        self.shared.playing.value = False

    def seek(self, sec: float) -> None:
        self.shared.pos.value = float(sec)
        self.call("seek", sec=float(sec), wait=False)

    def set_loop(self, enabled: bool, start=None, end=None) -> None:
        self.call("set_loop", enabled=enabled, start=start, end=end, wait=False)

    def preview_note(self, track: Track, pitch: int, velocity: int = 100, duration: float = 0.4) -> None:
        self.call("preview", track_id=track.id, pitch=pitch, velocity=velocity, duration=duration, wait=False)

    def preview_off(self, track: Track, pitch: int) -> None:
        self.call("preview_off", track_id=track.id, pitch=pitch, wait=False)

    # VST
    def has_vst(self, track_id: str) -> bool:
        return track_id in self.vst_tracks

    def vst_names(self, path: str) -> list[str]:
        return self.call("vst_names", path=path, timeout=120) or []

    def replace_vst(self, track: Track, path: str, plugin_name: str | None) -> dict:
        r = self.call("replace_vst", track_id=track.id, path=path, plugin_name=plugin_name, timeout=300)
        if r.get("ok"):
            track.instrument = r["instrument"]
            self.vst_tracks[track.id] = r["name"]
        return r

    def vst_state(self, track_id: str):
        try:
            return self.call("vst_state", track_id=track_id, timeout=30)
        except Exception:
            return None

    def show_editor(self, track_id: str) -> None:
        """Non-blocking: the server replies with an 'editor_closed'-style result through the pending call;
        we turn it into an event so the UI stays responsive while the plugin window is open."""
        if self.inprocess:
            r = self.call("show_editor", track_id=track_id, timeout=None)
            self._events.put(("editor_closed", {"track_id": track_id, **r}))
            return

        def work():
            try:
                r = self.call("show_editor", track_id=track_id, timeout=None)
            except Exception as e:
                r = {"ok": False, "error": str(e)}
            self._events.put(("editor_closed", {"track_id": track_id, **r}))
        threading.Thread(target=work, daemon=True).start()

    def test_vst(self, track_id: str) -> dict:
        return self.call("test_vst", track_id=track_id, timeout=300)

    def quit(self) -> None:
        try:
            self.call("quit", timeout=10)
        except Exception:
            pass
        if self._proc is not None:
            self._proc.join(timeout=5)
            self._kill()
