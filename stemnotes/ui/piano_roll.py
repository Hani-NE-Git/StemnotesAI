"""Piano roll: tiled, cached background (rows, waveform, spectrogram, AI heatmap, grid, ghost candidates, other
tracks) plus a cheap dynamic layer (active track notes, selection, loop, playhead). Thousands of ghost notes are
culled with numpy and drawn in batches, so panning and zooming stay smooth."""
from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget

from ..analysis.engines import engine_color
from ..audio.loader import load_peaks
from ..gm import GM_DRUMS, is_black, note_name
from ..log import get as _get_log
from ..project import Note, Project, Track, PITCH_HIGH, PITCH_LOW, N_PITCHES
from .colormaps import HEAT_LUT, SPECTRO_LUT, ImageCache

log = _get_log("pianoroll")

KEY_W_NOTES = 84
KEY_W_DRUMS = 136
RULER_H = 36
MIN_KEY_H, MAX_KEY_H = 6, 40
MIN_PPS, MAX_PPS = 8.0, 2000.0
TILE = 512
MAX_TILES = 160


@dataclass
class Layer:
    on: bool = True
    alpha: float = 0.8


@dataclass
class LayerSettings:
    waveform: Layer = field(default_factory=lambda: Layer(True, 0.35))
    spectrogram: Layer = field(default_factory=lambda: Layer(False, 0.7))
    heatmap: Layer = field(default_factory=lambda: Layer(True, 0.8))
    ghosts: Layer = field(default_factory=lambda: Layer(True, 0.9))
    others: Layer = field(default_factory=lambda: Layer(False, 0.35))
    grid: Layer = field(default_factory=lambda: Layer(True, 1.0))
    beats: Layer = field(default_factory=lambda: Layer(True, 1.0))
    source: str = "auto"
    ghost_min_conf: float = 0.2
    ghost_labels: bool = True
    ghost_engines: set[str] | None = None
    heat_engine: str | None = None


class PeakStore:
    """Waveform peaks per audio file (min/max per 64 samples), loaded once, audio itself is not kept."""

    def __init__(self):
        self._peaks: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self._failed: set[str] = set()

    def get(self, path) -> tuple[np.ndarray, np.ndarray, int] | None:
        key = str(path)
        if key in self._peaks:
            return self._peaks[key]
        if key in self._failed:
            return None
        try:
            self._peaks[key] = load_peaks(path, 22050, 64)
        except Exception:
            log.exception("peaks failed for %s", path)
            self._failed.add(key)
            return None
        return self._peaks[key]

    def clear(self) -> None:
        self._peaks.clear()
        self._failed.clear()


class GhostArrays:
    """numpy view of ghost note lists for fast culling. Cached by the identity of the list object."""

    def __init__(self):
        self._cache: dict[int, tuple] = {}

    def get(self, ghosts: list):
        key = id(ghosts)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is ghosts:
            return hit[1]
        if ghosts:
            arr = np.array([(g.start, g.end, g.pitch, g.conf) for g in ghosts], dtype=np.float64)
        else:
            arr = np.zeros((0, 4), dtype=np.float64)
        self._cache[key] = (ghosts, arr)
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        return arr


class PianoRoll(QWidget):
    notes_changed = Signal(str)
    seek_requested = Signal(float)
    loop_changed = Signal(bool, float, float)
    status = Signal(str)
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.project: Project | None = None
        self.host = None
        self.track: Track | None = None
        self.layers = LayerSettings()
        self.images = ImageCache()
        self.peaks = PeakStore()
        self.ghost_arrays = GhostArrays()

        self.key_w = KEY_W_NOTES
        self.pps = 120.0
        self.key_h = 16
        self.t0 = 0.0
        self.scroll_y = (PITCH_HIGH - 72) * self.key_h
        self.snap = 0.25
        self.default_len = 1.0
        self.playhead = 0.0
        self.loop_on = False
        self.loop_a = 0.0
        self.loop_b = 0.0
        self.follow = True

        self.selected: set[int] = set()
        self._mode: str | None = None
        self._press = QPointF()
        self._drag_orig: list[Note] = []
        self._drag_idx: list[int] = []
        self._create_idx: int | None = None
        self._rubber: QRectF | None = None
        self._preview_pitch: int | None = None
        self._pan_start = (0.0, 0.0)
        self._undo: dict[str, list[list[Note]]] = {}
        self._redo: dict[str, list[list[Note]]] = {}
        self._clipboard: list[Note] = []
        self._moved = False

        self._gen = 0                                   # static background generation
        self._notes_cache: tuple[int, int, np.ndarray] | None = None   # (id(notes), rev, array)
        self._notes_rev = 0
        self._tiles: OrderedDict[tuple, QPixmap] = OrderedDict()
        self._last_ph_x: int | None = None

    # ------------------------------------------------------------------ setup / invalidation
    def set_project(self, project: Project | None, host) -> None:
        self.project = project
        self.host = host
        self.track = None
        self.selected.clear()
        self.images.clear()
        self.peaks.clear()
        self.t0 = 0.0
        self.invalidate()

    def set_track(self, track: Track | None) -> None:
        if track is not self.track:
            self.selected.clear()
        self.track = track if (track is not None and track.kind == "midi") else None
        self._notes_rev += 1
        self.key_w = KEY_W_DRUMS if (self.track is not None and self.track.is_drums) else KEY_W_NOTES
        self.invalidate()

    def set_snap(self, beats: float) -> None:
        self.snap = beats
        self.invalidate()

    def invalidate(self) -> None:
        """Static background changed (layers, analysis, zoom, grid, source): drop cached tiles and repaint."""
        self._gen += 1
        self._tiles.clear()
        self.update()

    def set_playhead(self, sec: float) -> None:
        self.playhead = sec
        if self.follow and self.host is not None and self.host.playing:
            w = self.grid_width() / self.pps
            if sec > self.t0 + w * 0.95 or sec < self.t0:
                self.t0 = max(0.0, sec - w * 0.1)
                self.view_changed.emit()
                self.update()
                return
        x = int(self.x_of(sec))
        if x != self._last_ph_x:
            self._last_ph_x = x
            self.update()

    def set_loop(self, on: bool, a: float, b: float) -> None:
        self.loop_on, self.loop_a, self.loop_b = on, a, b
        self.update()

    # ------------------------------------------------------------------ geometry
    def grid_width(self) -> int:
        return max(1, self.width() - self.key_w)

    def grid_height(self) -> int:
        return max(1, self.height() - RULER_H)

    def total_height(self) -> int:
        return N_PITCHES * self.key_h

    def total_seconds(self) -> float:
        end = 60.0
        if self.host is not None:
            end = max(end, self.host.end_sec + 20.0)
        return end

    def x_of(self, sec: float) -> float:
        return self.key_w + (sec - self.t0) * self.pps

    def sec_of(self, x: float) -> float:
        return self.t0 + (x - self.key_w) / self.pps

    def y_of_pitch_top(self, pitch: float) -> float:
        return RULER_H + (PITCH_HIGH - pitch) * self.key_h - self.scroll_y

    def pitch_of_y(self, y: float) -> int:
        p = PITCH_HIGH - int((y - RULER_H + self.scroll_y) // self.key_h)
        return max(PITCH_LOW, min(PITCH_HIGH, p))

    def beat_x(self, beat: float) -> float:
        return self.x_of(self.project.beat_to_sec(beat))

    def snap_beat(self, beat: float, floor: bool = False) -> float:
        if self.snap <= 0:
            return beat
        q = beat / self.snap
        return float((np.floor(q) if floor else np.round(q)) * self.snap)

    def clamp_scroll(self) -> None:
        self.scroll_y = max(0, min(self.total_height() - self.grid_height(), self.scroll_y))
        self.t0 = max(0.0, self.t0)

    def visible_pitch_range(self) -> tuple[int, int]:
        return self.pitch_of_y(self.height()), self.pitch_of_y(RULER_H)

    def set_zoom(self, pps: float | None = None, key_h: int | None = None) -> None:
        if pps is not None:
            self.pps = max(MIN_PPS, min(MAX_PPS, pps))
        if key_h is not None:
            self.key_h = max(MIN_KEY_H, min(MAX_KEY_H, int(key_h)))
        self.invalidate()

    # ------------------------------------------------------------------ data helpers
    def active_source(self) -> str | None:
        if self.layers.source != "auto":
            return self.layers.source
        if self.track is not None and self.track.ghost_source:
            return self.track.ghost_source
        return "mix"

    def source_path(self, role: str | None):
        if self.project is None or role is None:
            return None
        for t in self.project.audio_tracks():
            if t.role == role:
                return self.project.resolve(t.path)
        return None

    def push_undo(self) -> None:
        if self.track is None:
            return
        st = self._undo.setdefault(self.track.id, [])
        st.append(copy.deepcopy(self.track.notes))
        if len(st) > 100:
            del st[0]
        self._redo.pop(self.track.id, None)

    def undo(self) -> None:
        if self.track is None:
            return
        st = self._undo.get(self.track.id, [])
        if not st:
            return
        self._redo.setdefault(self.track.id, []).append(copy.deepcopy(self.track.notes))
        self.track.notes = st.pop()
        self.selected.clear()
        self._commit()

    def redo(self) -> None:
        if self.track is None:
            return
        st = self._redo.get(self.track.id, [])
        if not st:
            return
        self._undo.setdefault(self.track.id, []).append(copy.deepcopy(self.track.notes))
        self.track.notes = st.pop()
        self.selected.clear()
        self._commit()

    def import_ghosts(self, engine_ids: set[str] | None, only_loop: bool = False, snap: bool = True) -> int:
        if self.track is None or self.project is None:
            return 0
        role = self.active_source()
        groups = self.project.analysis.ghosts_for(role or "", engine_ids)
        if not groups:
            return 0
        self.push_undo()
        base = len(self.track.notes)
        a, b = (self.loop_a, self.loop_b) if (only_loop and self.loop_b > self.loop_a) else (None, None)
        seen: set[tuple[float, int]] = set()
        for _eng, ghosts in groups:
            for g in ghosts:
                if 0 <= g.conf < self.layers.ghost_min_conf:
                    continue
                if a is not None and (g.end <= a or g.start >= b):
                    continue
                start = self.project.sec_to_beat(g.start)
                end = self.project.sec_to_beat(g.end)
                if snap and self.snap > 0:
                    start = self.snap_beat(start)
                    end = max(start + self.snap, self.snap_beat(end))
                key = (round(start, 3), g.pitch)
                if key in seen:
                    continue
                seen.add(key)
                vel = 100 if g.conf < 0 else int(60 + 67 * min(1.0, g.conf))
                self.track.notes.append(Note(float(start), float(max(0.05, end - start)), int(g.pitch), vel))
        added = len(self.track.notes) - base
        self.selected = set(range(base, base + added))
        self._commit()
        return added

    def _commit(self) -> None:
        if self.track is None:
            return
        self._notes_rev += 1
        self.project.dirty = True
        self.notes_changed.emit(self.track.id)
        self.update()

    # ------------------------------------------------------------------ painting: entry
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        if self.project is None:
            p.fillRect(self.rect(), QColor("#1b1d22"))
            p.setPen(QColor("#777"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Откройте песню (Ctrl+O) или проект")
            p.end()
            return
        self.clamp_scroll()
        grid = QRect(self.key_w, RULER_H, self.grid_width(), self.grid_height())
        p.save()
        p.setClipRect(grid)
        self._paint_tiles(p, grid)
        self._paint_loop(p, grid)
        self._paint_notes(p, grid)
        if self._rubber is not None:
            p.setPen(QPen(QColor(255, 255, 255, 160), 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor(255, 255, 255, 30))
            p.drawRect(self._rubber)
        p.restore()
        self._paint_keyboard(p)
        self._paint_ruler(p)
        self._paint_playhead(p, grid)
        p.end()

    # ------------------------------------------------------------------ painting: tiles (absolute pixel space)
    # abs_x = sec * pps ; abs_y = (PITCH_HIGH - pitch) * key_h   (row 0 = highest pitch)
    def _paint_tiles(self, p: QPainter, grid: QRect) -> None:
        ax0 = self.t0 * self.pps
        ay0 = float(self.scroll_y)
        tx0, tx1 = int(ax0 // TILE), int((ax0 + grid.width()) // TILE)
        ty0, ty1 = int(ay0 // TILE), int(min(ay0 + grid.height(), self.total_height()) // TILE)
        dpr = self.devicePixelRatioF()
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                key = (self._gen, tx, ty)
                pix = self._tiles.get(key)
                if pix is None:
                    pix = self._render_tile(tx, ty, dpr)
                    self._tiles[key] = pix
                    if len(self._tiles) > MAX_TILES:
                        self._tiles.popitem(last=False)
                else:
                    self._tiles.move_to_end(key)
                x = grid.left() + tx * TILE - ax0
                y = grid.top() + ty * TILE - ay0
                p.drawPixmap(QPointF(x, y), pix)

    def _render_tile(self, tx: int, ty: int, dpr: float) -> QPixmap:
        pix = QPixmap(int(TILE * dpr), int(TILE * dpr))
        pix.setDevicePixelRatio(dpr)
        pix.fill(QColor("#1b1d22"))
        p = QPainter(pix)
        p.translate(-tx * TILE, -ty * TILE)
        ax_a, ax_b = tx * TILE, (tx + 1) * TILE
        ay_a, ay_b = ty * TILE, (ty + 1) * TILE
        t_a, t_b = ax_a / self.pps, ax_b / self.pps
        # pitch range covered by this tile (row = PITCH_HIGH - pitch)
        row_a, row_b = int(ay_a // self.key_h), int(ay_b // self.key_h) + 1
        p_hi, p_lo = PITCH_HIGH - row_a, max(PITCH_LOW, PITCH_HIGH - row_b)
        self._tile_rows(p, ax_a, ax_b, p_lo, p_hi)
        self._tile_waveform(p, ax_a, ax_b, ay_a, ay_b, t_a, t_b)
        self._tile_images(p, ax_a, ax_b, ay_a, ay_b, t_a, t_b)
        self._tile_grid(p, ax_a, ax_b, ay_a, ay_b, t_a, t_b)
        self._tile_ghosts(p, t_a, t_b, p_lo, p_hi)
        self._tile_other_notes(p, t_a, t_b, p_lo, p_hi)
        p.end()
        return pix

    def _abs_y(self, pitch: int) -> float:
        return (PITCH_HIGH - pitch) * self.key_h

    def _tile_rows(self, p: QPainter, ax_a: float, ax_b: float, p_lo: int, p_hi: int) -> None:
        w = ax_b - ax_a
        for pitch in range(p_lo, p_hi + 1):
            y = self._abs_y(pitch)
            p.fillRect(QRectF(ax_a, y, w, self.key_h), QColor("#16181c") if is_black(pitch) else QColor("#20232a"))
            if pitch % 12 == 0:
                p.fillRect(QRectF(ax_a, y + self.key_h - 1, w, 1), QColor("#3a3f4a"))
            elif self.key_h >= 10:
                p.fillRect(QRectF(ax_a, y + self.key_h - 1, w, 1), QColor("#262a31"))

    def _tile_waveform(self, p: QPainter, ax_a, ax_b, ay_a, ay_b, t_a, t_b) -> None:
        L = self.layers.waveform
        if not L.on:
            return
        peaks = self.peaks.get(self.source_path(self.active_source()))
        if peaks is None:
            return
        mins, maxs, sr = peaks
        chunk = 64
        w = int(ax_b - ax_a)
        t_px = np.arange(w + 1) / self.pps + t_a
        idx = np.clip((t_px * sr / chunk).astype(np.int64), 0, len(mins))
        starts, ends = idx[:-1], idx[1:]
        valid = ends > starts
        if not valid.any():
            return
        s_valid = starts[valid]
        mn = np.minimum.reduceat(mins, s_valid)
        mx = np.maximum.reduceat(maxs, s_valid)
        total_h = self.total_height()
        cy = total_h / 2
        half = total_h / 2 * 0.95
        y1 = cy - mx * half
        y2 = np.maximum(cy - mn * half, y1 + 1)
        # keep only the part of each line inside this tile
        y1c, y2c = np.clip(y1, ay_a, ay_b), np.clip(y2, ay_a, ay_b)
        keep = y2c > y1c
        if not keep.any():
            return
        xs = np.nonzero(valid)[0][keep] + ax_a
        col = QColor("#8fd3ff")
        col.setAlphaF(L.alpha)
        p.setPen(QPen(col, 1))
        p.drawLines([QLineF(float(x), float(a), float(x), float(b)) for x, a, b in zip(xs, y1c[keep], y2c[keep])])

    def _tile_images(self, p: QPainter, ax_a, ax_b, ay_a, ay_b, t_a, t_b) -> None:
        role = self.active_source()
        an = self.project.analysis
        for key, L, lut in (("spec", self.layers.spectrogram, SPECTRO_LUT), ("heat", self.layers.heatmap, HEAT_LUT)):
            if not L.on:
                continue
            img = an.images.get(f"spec:{role}") if key == "spec" else an.heat_for(role or "", self.layers.heat_engine)
            if img is None:
                continue
            cols_per_px = 1.0 / (img.dt * self.pps)
            level = int(max(0, min(8, math.floor(math.log2(max(1.0, cols_per_px))))))
            q, factor = self.images.get_level(img, lut, level)
            rows, cols = img.data.shape
            rps = img.rows_per_semitone
            c_a = (t_a - img.t0) / img.dt / factor
            c_b = (t_b - img.t0) / img.dt / factor
            top_pitch = img.pitch_low + rows / rps
            pitch_top = PITCH_HIGH + 1 - ay_a / self.key_h
            pitch_bot = PITCH_HIGH + 1 - ay_b / self.key_h
            r_a = (top_pitch - pitch_top) * rps
            r_b = (top_pitch - pitch_bot) * rps
            p.save()
            p.setOpacity(L.alpha)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, cols_per_px > 1.0 / 4)
            p.drawImage(QRectF(ax_a, ay_a, ax_b - ax_a, ay_b - ay_a), q, QRectF(c_a, r_a, c_b - c_a, r_b - r_a))
            p.restore()

    def _tile_grid(self, p: QPainter, ax_a, ax_b, ay_a, ay_b, t_a, t_b) -> None:
        if not self.layers.grid.on:
            return
        proj = self.project
        beat_px = self.pps * 60.0 / proj.bpm
        b0 = int(math.floor(proj.sec_to_beat(t_a))) - 1
        b1 = int(math.ceil(proj.sec_to_beat(t_b))) + 1
        sub, beats, bars = [], [], []
        steps = int(round(1 / self.snap)) if (0 < self.snap < 1 and beat_px * self.snap >= 5) else 0
        for b in range(b0, b1):
            x = proj.beat_to_sec(b) * self.pps
            if b % proj.beats_per_bar == 0:
                bars.append(x)
            elif beat_px >= 4:
                beats.append(x)
            for s_ in range(1, steps):
                sub.append(proj.beat_to_sec(b + s_ * self.snap) * self.pps)
        for xs, color in ((sub, "#2b3038"), (beats, "#3c424d"), (bars, "#6b7280")):
            if xs:
                p.setPen(QPen(QColor(color), 1))
                p.drawLines([QLineF(x, ay_a, x, ay_b) for x in xs if ax_a - 1 <= x <= ax_b + 1])
        if self.layers.beats.on and proj.analysis.beats and ay_a <= 14:
            p.setPen(QPen(QColor(255, 120, 200, 110), 1, Qt.PenStyle.DotLine))
            p.drawLines([QLineF(bt * self.pps, 0, bt * self.pps, 14) for bt in proj.analysis.beats if t_a <= bt <= t_b])

    def _tile_ghosts(self, p: QPainter, t_a, t_b, p_lo, p_hi) -> None:
        L = self.layers.ghosts
        if not L.on:
            return
        role = self.active_source()
        groups = self.project.analysis.ghosts_for(role or "", self.layers.ghost_engines)
        if not groups:
            return
        multi = len(groups) > 1
        inner_h = max(2.0, self.key_h - 4)
        bar_h = inner_h / len(groups) if multi else inner_h
        font = QFont(p.font())
        font.setPixelSize(max(8, min(11, self.key_h - 4)))
        p.setFont(font)
        for gi, (eng, ghosts) in enumerate(groups):
            arr = self.ghost_arrays.get(ghosts)
            if len(arr) == 0:
                continue
            start, end, pitch, conf = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            mask = (end >= t_a) & (start <= t_b) & (pitch >= p_lo) & (pitch <= p_hi)
            mask &= (conf < 0) | (conf >= self.layers.ghost_min_conf)
            if not mask.any():
                continue
            sel = arr[mask]
            cr, cg, cb = engine_color(eng)
            y_off = 2 + gi * bar_h
            h = max(1.5, bar_h - (1 if multi else 0))
            conf_v = np.where(sel[:, 3] < 0, 0.75, sel[:, 3])
            bucket = np.clip((conf_v * 4).astype(int), 0, 3)      # 4 alpha levels -> 4 batched draw calls
            x1 = sel[:, 0] * self.pps
            w = np.maximum(3.0, (sel[:, 1] - sel[:, 0]) * self.pps)
            y = (PITCH_HIGH - sel[:, 2]) * self.key_h + y_off
            for b in range(4):
                m = bucket == b
                if not m.any():
                    continue
                c = (b + 1) / 4.0
                fill = QColor(cr, cg, cb)
                fill.setAlphaF(min(1.0, (0.12 + 0.55 * c) * L.alpha))
                border = QColor(min(255, cr + 30), min(255, cg + 30), min(255, cb + 30))
                border.setAlphaF(min(1.0, (0.3 + 0.6 * c) * L.alpha))
                p.setPen(QPen(border, 1, Qt.PenStyle.DashLine if self.key_h >= 9 else Qt.PenStyle.SolidLine))
                p.setBrush(fill)
                p.drawRects([QRectF(float(a), float(yy), float(ww), h) for a, yy, ww in zip(x1[m], y[m], w[m])])
            if self.layers.ghost_labels and bar_h >= 9 and len(sel) <= 400:
                p.setPen(QColor(255, 255, 255, int(220 * L.alpha)))
                for a, yy, ww, cv, raw in zip(x1, y, w, conf_v, sel[:, 3]):
                    if raw >= 0 and ww > 26:
                        p.drawText(QRectF(a + 2, yy, ww - 4, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"{int(raw * 100)}%")

    def _tile_other_notes(self, p: QPainter, t_a, t_b, p_lo, p_hi) -> None:
        L = self.layers.others
        if not L.on:
            return
        proj = self.project
        p.setPen(Qt.PenStyle.NoPen)
        for t in proj.midi_tracks():
            if t is self.track or not t.notes:
                continue
            col = QColor(t.color)
            col.setAlphaF(L.alpha)
            p.setBrush(col)
            rects = []
            for n in t.notes:
                if n.pitch < p_lo or n.pitch > p_hi:
                    continue
                s_, e_ = proj.beat_to_sec(n.start), proj.beat_to_sec(n.end)
                if e_ < t_a or s_ > t_b:
                    continue
                rects.append(QRectF(s_ * self.pps, self._abs_y(n.pitch) + 4, max(3.0, (e_ - s_) * self.pps), self.key_h - 8))
            if rects:
                p.drawRects(rects)

    # ------------------------------------------------------------------ painting: dynamic layer (widget coords)
    def _note_rect(self, n: Note) -> QRectF:
        x1 = self.beat_x(n.start)
        x2 = self.beat_x(n.end)
        y = self.y_of_pitch_top(n.pitch)
        return QRectF(x1, y + 1, max(3.0, x2 - x1), self.key_h - 2)

    def _paint_loop(self, p: QPainter, grid: QRect) -> None:
        if not self.loop_on or self.loop_b <= self.loop_a:
            return
        xa, xb = self.x_of(self.loop_a), self.x_of(self.loop_b)
        p.fillRect(QRectF(xa, grid.top(), xb - xa, grid.height()), QColor(255, 255, 255, 12))

    def _paint_notes(self, p: QPainter, grid: QRect) -> None:
        if self.track is None or not self.track.notes:
            return
        proj = self.project
        base = QColor(self.track.color)
        t_a, t_b = self.t0, self.t0 + grid.width() / self.pps
        lo, hi = self.visible_pitch_range()
        notes = self.track.notes
        c = self._notes_cache
        if c is not None and c[0] == id(notes) and c[1] == self._notes_rev and len(c[2]) == len(notes):
            arr = c[2]
        else:
            arr = np.array([(n.start, n.length, n.pitch, n.velocity) for n in notes], dtype=np.float64)
            self._notes_cache = (id(notes), self._notes_rev, arr)
        s_sec = proj.beat_to_sec(arr[:, 0])
        e_sec = proj.beat_to_sec(arr[:, 0] + arr[:, 1])
        vis = np.nonzero((e_sec >= t_a) & (s_sec <= t_b) & (arr[:, 2] >= lo) & (arr[:, 2] <= hi))[0]
        if len(vis) == 0:
            return
        font = QFont(p.font())
        font.setPixelSize(max(8, min(11, self.key_h - 4)))
        p.setFont(font)
        # batch by velocity shade (4 levels) and selection
        rects_by: dict[tuple[int, bool], list[QRectF]] = {}
        for i in vis:
            i = int(i)
            r = self._note_rect(notes[i])
            shade = min(3, int(arr[i, 3] / 32))
            rects_by.setdefault((shade, i in self.selected), []).append(r)
        for (shade, sel), rects in rects_by.items():
            c = QColor(base).lighter(70 + 20 * shade)
            p.setBrush(c)
            p.setPen(QPen(QColor("white"), 2) if sel else QPen(base.darker(160), 1))
            p.drawRects(rects)
        if self.key_h >= 11 and len(vis) <= 500:
            p.setPen(QColor(0, 0, 0, 200))
            drums = self.track.is_drums
            for i in vis:
                n = notes[int(i)]
                r = self._note_rect(n)
                if r.width() > 24:
                    label = GM_DRUMS.get(n.pitch, note_name(n.pitch)) if drums else note_name(n.pitch)
                    p.drawText(r.adjusted(3, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)

    def _paint_playhead(self, p: QPainter, grid: QRect) -> None:
        x = self.x_of(self.playhead)
        if grid.left() <= x <= grid.right():
            p.setPen(QPen(QColor("#ff5252"), 2))
            p.drawLine(QPointF(x, 0), QPointF(x, grid.bottom()))

    def _paint_keyboard(self, p: QPainter) -> None:
        p.save()
        p.setClipRect(QRect(0, RULER_H, self.key_w, self.grid_height()))
        p.fillRect(QRect(0, RULER_H, self.key_w, self.grid_height()), QColor("#111"))
        lo, hi = self.visible_pitch_range()
        drums = self.track is not None and self.track.is_drums
        font = QFont(p.font())
        font.setPixelSize(max(7, min(11, self.key_h - 3)))
        p.setFont(font)
        for pitch in range(lo, hi + 1):
            y = self.y_of_pitch_top(pitch)
            r = QRectF(0, y, self.key_w - 2, self.key_h - 1)
            if drums:
                name = GM_DRUMS.get(pitch)
                p.fillRect(r, QColor("#3a2f2a") if name else QColor("#1a1a1a"))
                if name and self.key_h >= 8:
                    p.setPen(QColor("#e8d5c0"))
                    p.drawText(r.adjusted(3, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"{pitch} {name}")
            else:
                black = is_black(pitch)
                p.fillRect(r, QColor("#2a2a2a") if black else QColor("#e8e8e8"))
                if pitch == self._preview_pitch:
                    p.fillRect(r, QColor(100, 180, 255, 150))
                if (pitch % 12 == 0 or self.key_h >= 14) and not black and self.key_h >= 8:
                    p.setPen(QColor("#333"))
                    p.drawText(r.adjusted(0, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, note_name(pitch))
        p.restore()
        p.fillRect(QRect(0, 0, self.key_w, RULER_H), QColor("#111"))

    def _paint_ruler(self, p: QPainter) -> None:
        proj = self.project
        p.save()
        p.setClipRect(QRect(self.key_w, 0, self.grid_width(), RULER_H))
        p.fillRect(QRect(self.key_w, 0, self.grid_width(), RULER_H), QColor("#14161a"))
        if self.loop_on and self.loop_b > self.loop_a:
            p.fillRect(QRectF(self.x_of(self.loop_a), 0, self.x_of(self.loop_b) - self.x_of(self.loop_a), RULER_H), QColor(120, 200, 120, 70))
        font = QFont(p.font())
        font.setPixelSize(10)
        p.setFont(font)
        bar_beats = proj.beats_per_bar
        beat_px = self.pps * 60.0 / proj.bpm
        bar_px = beat_px * bar_beats
        step = 1
        while bar_px * step < 40:
            step *= 2
        b0 = int(math.floor(proj.sec_to_beat(self.t0) / bar_beats)) - 1
        b1 = int(math.ceil(proj.sec_to_beat(self.t0 + self.grid_width() / self.pps) / bar_beats)) + 1
        p.setPen(QColor("#9aa3b2"))
        for bar in range(b0, b1):
            if bar % step:
                continue
            x = self.beat_x(bar * bar_beats)
            p.drawLine(QPointF(x, 0), QPointF(x, 16))
            p.drawText(QPointF(x + 3, 12), str(bar + 1))
        sec_step = 1.0
        while self.pps * sec_step < 60:
            sec_step *= 2
        while self.pps * sec_step > 240 and sec_step > 0.25:
            sec_step /= 2
        s0 = int(math.floor(self.t0 / sec_step))
        s1 = int(math.ceil((self.t0 + self.grid_width() / self.pps) / sec_step)) + 1
        p.setPen(QColor("#6f7a8a"))
        for k in range(s0, s1):
            t = k * sec_step
            x = self.x_of(t)
            p.drawLine(QPointF(x, RULER_H - 8), QPointF(x, RULER_H))
            m, s = divmod(t, 60)
            label = f"{int(m)}:{s:05.2f}" if sec_step < 1 else f"{int(m)}:{int(s):02d}"
            p.drawText(QPointF(x + 3, RULER_H - 9), label)
        p.restore()

    # ------------------------------------------------------------------ hit testing
    def _note_at(self, pos: QPointF) -> tuple[int | None, bool]:
        if self.track is None:
            return None, False
        for i in range(len(self.track.notes) - 1, -1, -1):
            r = self._note_rect(self.track.notes[i])
            if r.contains(pos):
                return i, (r.right() - pos.x()) <= max(6.0, min(10.0, r.width() * 0.3))
        return None, False

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.setFocus()
        if self.project is None:
            return
        pos = e.position()
        self._press = pos
        if pos.y() < RULER_H and pos.x() >= self.key_w:
            if e.button() == Qt.MouseButton.LeftButton:
                self._mode = "ruler"
                self._ruler_drag_start = max(0.0, self.sec_of(pos.x()))
                self.seek_requested.emit(self._ruler_drag_start)
            elif e.button() == Qt.MouseButton.RightButton:
                self.loop_on = False
                self.loop_changed.emit(False, self.loop_a, self.loop_b)
                self.update()
            return
        if pos.x() < self.key_w and pos.y() >= RULER_H:
            if e.button() == Qt.MouseButton.LeftButton and self.track is not None:
                self._mode = "keys"
                self._key_preview(self.pitch_of_y(pos.y()))
            return
        if e.button() == Qt.MouseButton.MiddleButton:
            self._mode = "pan"
            self._pan_start = (self.t0, self.scroll_y)
            return
        if self.track is None:
            self.status.emit("Выберите MIDI-дорожку слева, чтобы писать ноты")
            return
        idx, edge = self._note_at(pos)
        if e.button() == Qt.MouseButton.RightButton:
            self.push_undo()
            self._mode = "erase"
            if idx is not None:
                self._erase_at(pos)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        mods = e.modifiers()
        if idx is not None:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                if idx in self.selected:
                    self.selected.discard(idx)
                else:
                    self.selected.add(idx)
            elif idx not in self.selected:
                self.selected = {idx}
            self.push_undo()
            self._drag_idx = sorted(self.selected)
            self._drag_orig = [copy.copy(self.track.notes[i]) for i in self._drag_idx]
            self._mode = "resize" if edge else "move"
            self._moved = False
            if mods & Qt.KeyboardModifier.ControlModifier and self._mode == "move":
                new_notes = [copy.copy(n) for n in self._drag_orig]
                base = len(self.track.notes)
                self.track.notes.extend(new_notes)
                self.selected = set(range(base, base + len(new_notes)))
                self._drag_idx = sorted(self.selected)
            self._key_preview(self.track.notes[idx].pitch)
            return
        if mods & Qt.KeyboardModifier.ControlModifier:
            self._mode = "rubber"
            self._rubber = QRectF(pos, pos)
            if not (mods & Qt.KeyboardModifier.ShiftModifier):
                self.selected.clear()
            return
        beat = self.snap_beat(self.project.sec_to_beat(self.sec_of(pos.x())), floor=True)
        pitch = self.pitch_of_y(pos.y())
        self.push_undo()
        n = Note(beat, self.default_len if self.snap <= 0 else max(self.snap, self.default_len), pitch, 100)
        self.track.notes.append(n)
        self._create_idx = len(self.track.notes) - 1
        self.selected = {self._create_idx}
        self._mode = "create"
        self._key_preview(pitch)
        self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self.project is None:
            return
        pos = e.position()
        mode = self._mode
        if mode is None:
            idx, edge = self._note_at(pos) if pos.x() >= self.key_w and pos.y() >= RULER_H else (None, False)
            self.setCursor(Qt.CursorShape.SizeHorCursor if edge else Qt.CursorShape.ArrowCursor)
            if pos.x() >= self.key_w and pos.y() >= RULER_H:
                pitch = self.pitch_of_y(pos.y())
                sec = self.sec_of(pos.x())
                beat = self.project.sec_to_beat(sec)
                bar = int(math.floor(beat / self.project.beats_per_bar)) + 1
                bb = beat - (bar - 1) * self.project.beats_per_bar
                name = GM_DRUMS.get(pitch, note_name(pitch)) if (self.track and self.track.is_drums) else note_name(pitch)
                self.status.emit(f"{name} ({pitch})   такт {bar}, доля {bb + 1:.2f}   {sec:6.2f} с")
            return
        if mode == "ruler":
            t = max(0.0, self.sec_of(pos.x()))
            a, b = sorted((self._ruler_drag_start, t))
            if abs(pos.x() - self._press.x()) > 4:
                self.loop_a, self.loop_b, self.loop_on = a, b, True
                self.loop_changed.emit(True, a, b)
            self.update()
            return
        if mode == "keys":
            pitch = self.pitch_of_y(pos.y())
            if pitch != self._preview_pitch:
                self._key_preview(pitch)
            return
        if mode == "pan":
            dx = (pos.x() - self._press.x()) / self.pps
            self.t0 = max(0.0, self._pan_start[0] - dx)
            self.scroll_y = self._pan_start[1] - (pos.y() - self._press.y())
            self.clamp_scroll()
            self.view_changed.emit()
            self.update()
            return
        if mode == "erase":
            self._erase_at(pos)
            return
        if mode == "rubber":
            self._rubber = QRectF(self._press, pos).normalized()
            self.update()
            return
        if self.track is None:
            return
        self._notes_rev += 1
        if mode == "create" and self._create_idx is not None:
            n = self.track.notes[self._create_idx]
            beat = self.project.sec_to_beat(self.sec_of(pos.x()))
            end = self.snap_beat(beat) if self.snap > 0 else beat
            n.length = max(self.snap if self.snap > 0 else 0.05, end - n.start)
            pitch = self.pitch_of_y(pos.y())
            if pitch != n.pitch:
                n.pitch = pitch
                self._key_preview(pitch)
            self.update()
            return
        if mode in ("move", "resize"):
            d_beat = self.project.sec_to_beat(self.sec_of(pos.x())) - self.project.sec_to_beat(self.sec_of(self._press.x()))
            if self.snap > 0:
                d_beat = round(d_beat / self.snap) * self.snap
            d_pitch = self.pitch_of_y(pos.y()) - self.pitch_of_y(self._press.y())
            if abs(d_beat) > 0 or d_pitch != 0:
                self._moved = True
            for i, orig in zip(self._drag_idx, self._drag_orig):
                n = self.track.notes[i]
                if mode == "move":
                    n.start = orig.start + d_beat
                    n.pitch = max(PITCH_LOW, min(PITCH_HIGH, orig.pitch + d_pitch))
                else:
                    n.length = max(self.snap if self.snap > 0 else 0.05, orig.length + d_beat)
            if mode == "move" and d_pitch != 0 and self._drag_idx:
                pitch = self.track.notes[self._drag_idx[0]].pitch
                if pitch != self._preview_pitch:
                    self._key_preview(pitch)
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        mode = self._mode
        self._mode = None
        self._key_release()
        if mode == "rubber" and self._rubber is not None and self.track is not None:
            for i, n in enumerate(self.track.notes):
                if self._rubber.intersects(self._note_rect(n)):
                    self.selected.add(i)
            self._rubber = None
            self.update()
            return
        if mode in ("create", "move", "resize", "erase") and self.track is not None:
            if mode == "create" and self._create_idx is not None:
                self.default_len = self.track.notes[self._create_idx].length
            self._create_idx = None
            if mode == "move" and not self._moved:
                st = self._undo.get(self.track.id)
                if st:
                    st.pop()
                self.update()
                return
            self._commit()
        self.update()

    def _erase_at(self, pos: QPointF) -> None:
        idx, _ = self._note_at(pos)
        if idx is not None and self.track is not None:
            self._notes_rev += 1
            del self.track.notes[idx]
            self.selected = {i - 1 if i > idx else i for i in self.selected if i != idx}
            self.update()

    def _key_preview(self, pitch: int) -> None:
        self._key_release()
        self._preview_pitch = pitch
        if self.host is not None and self.track is not None:
            self.host.preview_note(self.track, pitch, 100, duration=0.6)
        self.update()

    def _key_release(self) -> None:
        if self._preview_pitch is not None and self.host is not None and self.track is not None:
            self.host.preview_off(self.track, self._preview_pitch)
        self._preview_pitch = None

    def wheelEvent(self, e: QWheelEvent) -> None:
        mods = e.modifiers()
        delta = e.angleDelta().y() or e.angleDelta().x()
        pos = e.position()
        if mods & Qt.KeyboardModifier.ControlModifier:
            sec = self.sec_of(pos.x())
            factor = 1.15 if delta > 0 else 1 / 1.15
            self.pps = max(MIN_PPS, min(MAX_PPS, self.pps * factor))
            self.t0 = max(0.0, sec - (pos.x() - self.key_w) / self.pps)
            self.invalidate()
        elif mods & Qt.KeyboardModifier.AltModifier:
            pitch_at = PITCH_HIGH - (pos.y() - RULER_H + self.scroll_y) / self.key_h
            self.key_h = max(MIN_KEY_H, min(MAX_KEY_H, self.key_h + (1 if delta > 0 else -1)))
            self.scroll_y = (PITCH_HIGH - pitch_at) * self.key_h - (pos.y() - RULER_H)
            self.invalidate()
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            self.t0 -= delta / 120.0 * (self.grid_width() / self.pps) * 0.15
        else:
            self.scroll_y -= delta / 120.0 * self.key_h * 3
        self.clamp_scroll()
        self.view_changed.emit()
        self.update()

    # ------------------------------------------------------------------ keyboard
    def keyPressEvent(self, e: QKeyEvent) -> None:
        k = e.key()
        mods = e.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if self.track is None:
            e.ignore()
            return
        if k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.selected:
                self.push_undo()
                self.track.notes = [n for i, n in enumerate(self.track.notes) if i not in self.selected]
                self.selected.clear()
                self._commit()
            return
        if ctrl and k == Qt.Key.Key_A:
            self.selected = set(range(len(self.track.notes)))
            self.update()
            return
        if ctrl and k == Qt.Key.Key_Z:
            self.undo()
            return
        if ctrl and k == Qt.Key.Key_Y:
            self.redo()
            return
        if ctrl and k == Qt.Key.Key_C:
            self._clipboard = [copy.copy(self.track.notes[i]) for i in sorted(self.selected)]
            self.status.emit(f"Скопировано нот: {len(self._clipboard)}")
            return
        if ctrl and k == Qt.Key.Key_X:
            self._clipboard = [copy.copy(self.track.notes[i]) for i in sorted(self.selected)]
            self.push_undo()
            self.track.notes = [n for i, n in enumerate(self.track.notes) if i not in self.selected]
            self.selected.clear()
            self._commit()
            return
        if ctrl and k == Qt.Key.Key_V and self._clipboard:
            self.push_undo()
            start = min(n.start for n in self._clipboard)
            at = self.snap_beat(self.project.sec_to_beat(self.playhead))
            base = len(self.track.notes)
            for n in self._clipboard:
                c = copy.copy(n)
                c.start = n.start - start + at
                self.track.notes.append(c)
            self.selected = set(range(base, base + len(self._clipboard)))
            self._commit()
            return
        if ctrl and k == Qt.Key.Key_D and self.selected:
            self.push_undo()
            sel = [self.track.notes[i] for i in sorted(self.selected)]
            span = max(n.end for n in sel) - min(n.start for n in sel)
            if self.snap > 0:
                span = max(self.snap, math.ceil(span / self.snap) * self.snap)
            base = len(self.track.notes)
            for n in sel:
                c = copy.copy(n)
                c.start += span
                self.track.notes.append(c)
            self.selected = set(range(base, base + len(sel)))
            self._commit()
            return
        if self.selected and k in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            self.push_undo()
            step = self.snap if self.snap > 0 else 0.25
            for i in self.selected:
                n = self.track.notes[i]
                if k == Qt.Key.Key_Left:
                    n.start -= step
                elif k == Qt.Key.Key_Right:
                    n.start += step
                elif k == Qt.Key.Key_Up:
                    n.pitch = min(PITCH_HIGH, n.pitch + (12 if ctrl else 1))
                else:
                    n.pitch = max(PITCH_LOW, n.pitch - (12 if ctrl else 1))
            self._commit()
            return
        if self.selected and k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus):
            self.push_undo()
            d = 10 if k != Qt.Key.Key_Minus else -10
            for i in self.selected:
                n = self.track.notes[i]
                n.velocity = max(1, min(127, n.velocity + d))
            self.status.emit("Velocity: " + ", ".join(str(self.track.notes[i].velocity) for i in sorted(self.selected)[:8]))
            self._commit()
            return
        e.ignore()
