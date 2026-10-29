"""Smoke test of stemnotes-host with real VST3 instruments, no audio device needed (offline render).

For every plugin: load it on a MIDI track, play a few notes, report the output peak, the size of the saved state and
any non-protocol text the host process wrote to stdout (must be none). Optionally opens each editor for N seconds.

    python host/tools/vst_smoke_test.py "C:/Program Files/Common Files/VST3/EZdrummer 3.vst3=36,38,42" ...
    python host/tools/vst_smoke_test.py --exe bin/stemnotes-host.exe --editor 3 "path/to/Plugin.vst3"

A plugin argument is PATH or PATH=PITCH,PITCH,... (default pitches 48,52,55). Exit code 1 if any plugin is silent.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "bin" / ("stemnotes-host.exe" if sys.platform.startswith("win") else "stemnotes-host")


class Host:
    def __init__(self, exe: Path, log: Path):
        self.log = open(log, "w", encoding="utf-8")
        self.proc = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     bufsize=1, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.replies: dict[int, dict] = {}
        self.events: queue.Queue = queue.Queue()
        self.foreign: list[str] = []          # stdout lines that are not protocol JSON
        self.cv = threading.Condition()
        self.next_id = 1
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                self.foreign.append(line.rstrip())
                continue
            if "event" in m:
                if m["event"] != "position":
                    self.events.put(m)
            else:
                with self.cv:
                    self.replies[m["id"]] = m
                    self.cv.notify_all()

    def call(self, op: str, timeout: float = 120, **kw):
        cid, self.next_id = self.next_id, self.next_id + 1
        self.proc.stdin.write(json.dumps({"id": cid, "op": op, **kw}) + "\n")
        self.proc.stdin.flush()
        with self.cv:
            if not self.cv.wait_for(lambda: cid in self.replies, timeout):
                raise TimeoutError(f"host did not answer {op}")
            m = self.replies.pop(cid)
        if not m.get("ok"):
            raise RuntimeError(f"{op}: {m.get('error')}")
        return m.get("result")

    def wait_event(self, name: str, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            try:
                e = self.events.get(timeout=0.5)
            except queue.Empty:
                continue
            if e["event"] == name:
                return e.get("data") or {}
        return None

    def quit(self):
        try:
            self.call("quit", timeout=30)
            self.proc.wait(timeout=30)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.log.close()


def check(exe: Path, plugin: str, pitches: list[int], settle: float, editor: float, workdir: Path) -> bool:
    name = Path(plugin).stem
    log = workdir / f"{name}.log"
    h = Host(exe, log)
    t0 = time.time()
    ok = False
    try:
        notes = [[i * 0.5, 0.45, p, 110] for i, p in enumerate(pitches * 4)]
        track = {"id": "t1", "name": name, "kind": "midi", "mute": False, "solo": False, "volume": 0.8,
                 "instrument": {"type": "vst3", "path": plugin, "plugin_name": None, "state": None}, "notes": notes}
        h.call("set_project", directory=str(workdir), data={"bpm": 120, "offset": 0.0, "tracks": [track]}, timeout=600)
        ev = h.wait_event("vst_loaded", 300)
        if not ev or not ev.get("ok"):
            print(f"FAIL  {name}: not loaded: {ev and ev.get('error')}")
            return False
        loaded = time.time() - t0
        time.sleep(settle)   # sample-based instruments stream their samples in the background
        peaks = []
        for _ in range(2):
            h.call("seek", sec=0.0)
            h.call("play")
            peaks.append(h.call("offline_render", seconds=4.0)["peak"])
            h.call("stop")
        state = h.call("vst_state", track_id="t1") or ""
        if editor > 0:
            if not h.call("show_editor", track_id="t1").get("ok"):
                print(f"      {name}: editor did not open")
            time.sleep(editor)
            h.call("close_editor")
            if h.wait_event("editor_closed", 15) is None:
                print(f"      {name}: no editor_closed event")
        ok = max(peaks) > 1e-4
        print(f"{'OK  ' if ok else 'FAIL'}  {ev.get('name')}: peak {max(peaks):.3f}, loaded in {loaded:.1f}s, "
              f"state {len(state)} b64 chars")
    finally:
        h.quit()
        ready = [l.strip() for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if "plugin ready" in l]
        if ready:
            print("      " + ready[-1])
        if h.foreign:
            ok = False
            print(f"FAIL  {name}: {len(h.foreign)} foreign line(s) on the protocol channel, e.g. {h.foreign[0]!r}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plugins", nargs="+", help="PATH[=PITCH,PITCH,...]")
    ap.add_argument("--exe", type=Path, default=EXE)
    ap.add_argument("--settle", type=float, default=8.0, help="seconds to wait after loading (default 8)")
    ap.add_argument("--editor", type=float, default=0.0, help="open each plugin window for N seconds")
    a = ap.parse_args()
    if not a.exe.exists():
        print(f"host not found: {a.exe}")
        return 2
    with tempfile.TemporaryDirectory(prefix="stemnotes-smoke-") as tmp:
        results = []
        for spec in a.plugins:
            path, _, pits = spec.partition("=")
            pitches = [int(p) for p in pits.split(",")] if pits else [48, 52, 55]
            results.append(check(a.exe, path, pitches, a.settle, a.editor, Path(tmp)))
    print(f"{sum(results)}/{len(results)} plugins passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
