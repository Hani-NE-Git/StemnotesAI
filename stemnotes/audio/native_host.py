"""NativeHost: drives the C++ `stemnotes-host` process (JUCE) over a JSON-line pipe. Same API as AudioHost."""
from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from .. import paths
from ..log import get as _get_log, LOG_DIR
from ..project import Project, Track

log = _get_log("native-host")

BIN_DIR = paths.HOME / "bin"
EXE_NAME = "stemnotes-host.exe" if sys.platform.startswith("win") else "stemnotes-host"
HOST_VERSION_FILE = paths.HOME / "host" / "VERSION"


def host_exe() -> Path | None:
    p = BIN_DIR / EXE_NAME
    return p if p.exists() else None


def expected_host_version() -> str:
    try:
        return HOST_VERSION_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"


class _Shared:
    """Mirrors the transport state published by the host (position events)."""
    def __init__(self):
        self.pos = 0.0
        self.playing = False
        self.end = 10.0
        self.editor_open = False


class NativeHost:
    inprocess = False

    def __init__(self, exe: Path | None = None):
        self.exe = exe or host_exe()
        if self.exe is None:
            raise FileNotFoundError("stemnotes-host не найден в папке bin")
        self.available = False
        self.error: str | None = None
        self.sample_rate = 44100
        self.block = 1024
        self.master = 0.9
        self.device_name = ""
        self.vst_tracks: dict[str, str] = {}
        self.project: Project | None = None
        self.sync_pending = False
        self.editor_open = False
        self.shared = _Shared()
        self._events: queue.Queue = queue.Queue()
        self._ids = itertools.count(1)
        self._pending: dict[int, dict] = {}
        self._plock = threading.Lock()
        self._wlock = threading.Lock()
        self._alive = True
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._stderr = open(LOG_DIR / "stemnotes-host.log", "a", encoding="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen([str(self.exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                      cwd=str(paths.HOME), bufsize=1, encoding="utf-8", creationflags=flags)
        threading.Thread(target=self._reader, daemon=True).start()
        v = self.call("version", timeout=30)
        log.info("stemnotes-host %s started (pid %s) from %s", v, self._proc.pid, self.exe)
        self.host_version = str(v)

    # ------------------------------------------------------------------ transport layer
    def _reader(self) -> None:
        for line in self._proc.stdout:
            try:
                m = json.loads(line)
            except Exception:
                continue
            if "event" in m:
                name, data = m["event"], m.get("data") or {}
                if name == "position":
                    self.shared.pos = float(data.get("sec", 0.0))
                    self.shared.playing = bool(data.get("playing"))
                    self.shared.end = float(data.get("end", 10.0))
                    self.editor_open = bool(data.get("editor_open"))
                    continue
                if name == "vst_loaded":
                    if data.get("ok"):
                        self.vst_tracks[data["track_id"]] = data.get("name") or "VST3"
                        self._events.put(("warning", {"message": f"VST3 загружен: {data.get('name')}"}))
                    else:
                        self.vst_tracks.pop(data.get("track_id"), None)
                        self._events.put(("error", {"message": f"VST3 не загрузился: {data.get('error')}"}))
                    continue
                self._events.put((name, data))
            else:
                with self._plock:
                    slot = self._pending.get(m.get("id"))
                if slot is not None:
                    slot["msg"] = m
                    slot["event"].set()
        self._alive = False
        self._events.put(("server_gone", {}))

    def call(self, op: str, wait: bool = True, timeout: float | None = 120.0, **kw):
        cid = next(self._ids)
        slot = {"event": threading.Event(), "msg": None}
        with self._plock:
            self._pending[cid] = slot
        try:
            with self._wlock:
                self._proc.stdin.write(json.dumps({"id": cid, "op": op, **kw}, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
        except Exception as e:
            with self._plock:
                self._pending.pop(cid, None)
            raise RuntimeError(f"аудиохост недоступен: {e}")
        if not wait:
            return None
        if not slot["event"].wait(timeout):
            with self._plock:
                self._pending.pop(cid, None)
            raise TimeoutError(f"аудиохост не ответил на {op}")
        with self._plock:
            self._pending.pop(cid, None)
        m = slot["msg"]
        if not m.get("ok"):
            log.error("host op %s failed: %s", op, m.get("error"))
            raise RuntimeError(str(m.get("error")))
        return m.get("result")

    def call_async(self, op: str, done, **kw) -> None:
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

    # ------------------------------------------------------------------ API (same as AudioHost)
    def start(self, device=None, block=None) -> bool:
        dev = device if isinstance(device, str) else ""
        r = self.call("start", device=dev, block=int(block or 1024), timeout=60)
        self.available = bool(r.get("available"))
        self.error = r.get("error")
        self.sample_rate = int(r.get("sample_rate") or 44100)
        self.block = int(r.get("block") or 1024)
        self.device_name = r.get("device") or ""
        return self.available

    def stop_stream(self) -> None:
        try:
            self.call("stop_stream", timeout=10)
        except Exception:
            pass

    def devices(self) -> list[dict]:
        try:
            return [{"index": d["id"], "name": d["name"], "hostapi": d["type"], "default_sr": 0} for d in self.call("devices", timeout=30)]
        except Exception:
            return []

    def load_soundfont(self, path: Path) -> None:
        r = self.call("load_soundfont", path=str(path), timeout=120)
        if r.get("error"):
            self.error = r["error"]

    def set_master(self, v: float) -> None:
        self.master = v
        self.call("master", value=float(v), wait=False)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        if project is None:
            return
        r = self.call("set_project", directory=str(project.dir), data=project.to_json(), timeout=900)
        self.vst_tracks = dict(r.get("vst_tracks", {}))

    def sync(self) -> None:
        if self.project is None:
            return

        def done(r, err):
            if err:
                self._events.put(("error", {"message": err}))
            elif r:
                for k, v in r.get("vst_tracks", {}).items():
                    self.vst_tracks[k] = v
        self.call_async("sync", done, directory=str(self.project.dir), data=self.project.to_json())

    @property
    def playing(self) -> bool:
        return self.shared.playing

    @property
    def position(self) -> float:
        return self.shared.pos

    @property
    def end_sec(self) -> float:
        return self.shared.end

    @property
    def vst_rendering(self) -> int:
        return 0

    def play(self) -> None:
        self.shared.playing = True
        self.call("play", wait=False)

    def stop(self) -> None:
        self.shared.playing = False
        self.call("stop", wait=False)

    def seek(self, sec: float) -> None:
        self.shared.pos = float(sec)
        self.call("seek", sec=float(sec), wait=False)

    def set_loop(self, enabled: bool, start=None, end=None) -> None:
        self.call("set_loop", enabled=bool(enabled), start=float(start or 0.0), end=float(end or 0.0), wait=False)

    def preview_note(self, track: Track, pitch: int, velocity: int = 100, duration: float = 0.4) -> None:
        self.call("preview", track_id=track.id, pitch=int(pitch), velocity=int(velocity), duration=float(duration), wait=False)

    def preview_off(self, track: Track, pitch: int) -> None:
        self.call("preview_off", track_id=track.id, pitch=int(pitch), wait=False)

    def has_vst(self, track_id: str) -> bool:
        return track_id in self.vst_tracks

    def vst_names(self, path: str) -> list[str]:
        return list(self.call("vst_names", path=path, timeout=300) or [])

    def replace_vst(self, track: Track, path: str, plugin_name: str | None) -> dict:
        r = self.call("replace_vst", track_id=track.id, path=path, plugin_name=plugin_name or "", timeout=600)
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
        r = self.call("show_editor", track_id=track_id, timeout=60)
        if not r.get("ok"):
            self._events.put(("editor_closed", {"track_id": track_id, "ok": False, "error": r.get("error")}))
        else:
            self.editor_open = True

    def close_editor(self) -> None:
        self.call("close_editor", wait=False)

    def test_vst(self, track_id: str) -> dict:
        return self.call("test_vst", track_id=track_id, timeout=60)

    def quit(self) -> None:
        try:
            self.call("quit", timeout=5)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        try:
            self._stderr.close()
        except Exception:
            pass
