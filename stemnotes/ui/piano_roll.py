"""Piano roll with layered background: waveform, CQT spectrogram, AI activation heatmap, ghost-note candidates,
other tracks' notes, grid; plus note editing for the active track."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget

from ..gm import GM_DRUMS, is_black, note_name
from ..project import Note, Project, Track, PITCH_HIGH, PITCH_LOW, N_PITCHES
from .colormaps import HEAT_LUT, SPECTRO_LUT, ImageCache

KEY_W_NOTES = 84
KEY_W_DRUMS = 136
RULER_H = 36
MIN_KEY_H, MAX_KEY_H = 6, 40
MIN_PPS, MAX_PPS = 8.0, 2000.0


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
    source: str = "auto"          # "auto" = follow the active track's ghost source, otherwise a role
    ghost_min_conf: float = 0.2
    ghost_labels: bool = True


class PeakCache:
    """Min/max per chunk for fast waveform drawing."""
    CHUNK = 64

    def __init__(self):
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = id(audio)
        if key in self._cache:
            return self._cache[key]
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        n = (len(mono) // self.CHUNK) * self.CHUNK
        if n == 0:
            r = (np.zeros(1, np.float32), np.zeros(1, np.float32))
        else:
            m = mono[:n].reshape(-1, self.CHUNK)
            r = (m.min(axis=1), m.max(axis=1))
        self._cache[key] = r
        return r


class PianoRoll(QWidget):
    notes_changed = Signal(str)        # track id
    seek_requested = Signal(float)     # seconds
    loop_changed = Signal(bool, float, float)
    status = Signal(str)
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.project: Project | None = None
        self.engine = None
        self.track: Track | None = None
        self.layers = LayerSettings()
        self.images = ImageCache()
        self.peaks = PeakCache()

        self.key_w = KEY_W_NOTES
        self.pps = 120.0           # pixels per second
        self.key_h = 16
        self.t0 = 0.0              # seconds at left edge of grid
        self.scroll_y = (PITCH_HIGH - 72) * self.key_h   # start around C5 at top
        self.snap = 0.25           # beats (0 = off)
        self.default_len = 1.0     # beats
        self.playhead = 0.0
        self.loop_on = False
        self.loop_a = 0.0
        self.loop_b = 0.0
        self.follow = True

        self.selected: set[int] = set()   # indices into track.notes
        self._mode: str | None = None
        self._press = QPointF()
        self._drag_orig: list[Note] = []
        self._drag_idx: list[int] = []
        self._create_idx: int | None = None
        self._rubber: QRectF | None = None
        self._preview_pitch: int | None = None
        self._hover_pitch: int | None = None
        self._pan_start = (0.0, 0.0)
        self._undo: dict[str, list[list[Note]]] = {}
        self._redo: dict[str, list[list[Note]]] = {}
        self._clipboard: list[Note] = []
        self._rev = 0
        self._bg: QPixmap | None = None
        self._bg_key = None
        self._last_ph_x: int | None = None

    def invalidate(self) -> None:
        """Content changed: redraw the cached background on next paint."""
        self._rev += 1
        self.update()

    # ------------------------------------------------------------------ setup
    def set_project(self, project: Project | None, engine) -> None:
        self.project = project
        self.engine = engine
        self.track = None
        self.selected.clear()
        self.images.clear()
        self.t0 = 0.0
        self.invalidate()

    def set_track(self, track: Track | None) -> None:
        if track is not self.track:
            self.selected.clear()
        self.track = track if (track is not None and track.kind == "midi") else None
        self.key_w = KEY_W_DRUMS if (self.track is not None and self.track.is_drums) else KEY_W_NOTES
        self.invalidate()

    def set_snap(self, beats: float) -> None:
        self.snap = beats
        self.invalidate()

    def set_playhead(self, sec: float) -> None:
        self.playhead = sec
        if self.follow and self.engine is not None and self.engine.playing:
            w = self.grid_width() / self.pps
            if sec > self.t0 + w * 0.95 or sec < self.t0:
                self.t0 = max(0.0, sec - w * 0.1)
                self.view_changed.emit()
                self.invalidate()
                return
        x = int(self.x_of(sec))
        if x != self._last_ph_x:
            self._last_ph_x = x
            self.update()

    def set_loop(self, on: bool, a: float, b: float) -> None:
        self.loop_on, self.loop_a, self.loop_b = on, a, b
        self.invalidate()

    # ------------------------------------------------------------------ geometry
    def grid_width(self) -> int:
        return max(1, self.width() - self.key_w)

    def grid_height(self) -> int:
        return max(1, self.height() - RULER_H)

    def total_height(self) -> int:
        return N_PITCHES * self.key_h

    def total_seconds(self) -> float:
        end = 60.0
        if self.engine is not None and self.engine.sample_rate:
            end = max(end, self.engine.end_sample / self.engine.sample_rate + 20.0)
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
        return (np.floor(q) if floor else np.round(q)) * self.snap

    def clamp_scroll(self) -> None:
        self.scroll_y = max(0, min(self.total_height() - self.grid_height(), self.scroll_y))
        self.t0 = max(0.0, self.t0)

    def visible_pitch_range(self) -> tuple[int, int]:
        top = self.pitch_of_y(RULER_H)
        bottom = self.pitch_of_y(self.height())
        return bottom, top

    # ------------------------------------------------------------------ data helpers
    def active_source(self) -> str | None:
        if self.layers.source != "auto":
            return self.layers.source
        if self.track is not None and self.track.ghost_source:
            return self.track.ghost_source
        return "mix"

    def source_audio(self, role: str | None) -> np.ndarray | None:
        if self.project is None or self.engine is None or role is None:
            return None
        for t in self.project.audio_tracks():
            if t.role == role:
                return self.engine.audio.get(t.id)
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

    def _commit(self) -> None:
        if self.track is None:
            return
        self.project.dirty = True
        self.notes_changed.emit(self.track.id)
        self.invalidate()

    # ------------------------------------------------------------------ painting
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        if self.project is None:
            p.fillRect(self.rect(), QColor("#1b1d22"))
            p.setPen(QColor("#777"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Откройте песню (Ctrl+O) или проект")
            p.end()
            return
        self.clamp_scroll()
        key = (self._rev, round(self.t0, 5), self.scroll_y, self.pps, self.key_h, self.key_w, self.width(), self.height(),
               id(self.track), self.loop_on, self.loop_a, self.loop_b, self._preview_pitch, self.snap,
               self._rubber.getCoords() if self._rubber is not None else None)
        if self._bg is None or key != self._bg_key:
            dpr = self.devicePixelRatioF()
            pix = QPixmap(int(self.width() * dpr), int(self.height() * dpr))
            pix.setDevicePixelRatio(dpr)
            bp = QPainter(pix)
            self._paint_background(bp)
            bp.end()
            self._bg = pix
            self._bg_key = key
        p.drawPixmap(0, 0, self._bg)
        grid = QRect(self.key_w, RULER_H, self.grid_width(), self.grid_height())
        self._paint_playhead(p, grid)
        p.end()

    def _paint_background(self, p: QPainter) -> None:
        p.fillRect(self.rect(), QColor("#1b1d22"))
        grid = QRect(self.key_w, RULER_H, self.grid_width(), self.grid_height())
        p.save()
        p.setClipRect(grid)
        self._paint_rows(p, grid)
        self._paint_waveform(p, grid)
        self._paint_images(p, grid)
        self._paint_grid(p, grid)
        self._paint_loop(p, grid)
        self._paint_ghosts(p, grid)
        self._paint_other_notes(p, grid)
        self._paint_notes(p, grid)
        if self._rubber is not None:
            p.setPen(QPen(QColor(255, 255, 255, 160), 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor(255, 255, 255, 30))
            p.drawRect(self._rubber)
        p.restore()
        self._paint_keyboard(p)
        self._paint_ruler(p)

    def _paint_rows(self, p: QPainter, grid: QRect) -> None:
        lo, hi = self.visible_pitch_range()
        for pitch in range(lo, hi + 1):
            y = self.y_of_pitch_top(pitch)
            if is_black(pitch):
                p.fillRect(QRectF(grid.left(), y, grid.width(), self.key_h), QColor("#16181c"))
            else:
                p.fillRect(QRectF(grid.left(), y, grid.width(), self.key_h), QColor("#20232a"))
            if pitch % 12 == 0:
                p.fillRect(QRectF(grid.left(), y + self.key_h - 1, grid.width(), 1), QColor("#3a3f4a"))
            elif self.key_h >= 10:
                p.fillRect(QRectF(grid.left(), y + self.key_h - 1, grid.width(), 1), QColor("#262a31"))

    def _paint_waveform(self, p: QPainter, grid: QRect) -> None:
        L = self.layers.waveform
        if not L.on:
            return
        audio = self.source_audio(self.active_source())
        if audio is None or self.engine is None:
            return
        mins, maxs = self.peaks.get(audio)
        sr = self.engine.sample_rate
        chunk = PeakCache.CHUNK
        w = grid.width()
        t_px = np.arange(w + 1) / self.pps + self.t0
        idx = (t_px * sr / chunk).astype(np.int64)
        idx = np.clip(idx, 0, len(mins))
        starts, ends = idx[:-1], idx[1:]
        valid = ends > starts
        if not valid.any():
            return
        s_valid = starts[valid]
        mn = np.minimum.reduceat(mins, s_valid) if len(s_valid) else np.zeros(0)
        mx = np.maximum.reduceat(maxs, s_valid) if len(s_valid) else np.zeros(0)
        # reduceat includes until next start; approximate fine for display
        cy = grid.top() + grid.height() / 2
        half = grid.height() / 2 * 0.95
        col = QColor("#8fd3ff")
        col.setAlphaF(L.alpha)
        p.setPen(QPen(col, 1))
        xs = np.nonzero(valid)[0]
        y1 = cy - mx * half
        y2 = np.maximum(cy - mn * half, y1 + 1)
        from PySide6.QtCore import QLineF
        gl = grid.left()
        p.drawLines([QLineF(gl + float(x), float(a), gl + float(x), float(b)) for x, a, b in zip(xs, y1, y2)])

    def _paint_images(self, p: QPainter, grid: QRect) -> None:
        role = self.active_source()
        an = self.project.analysis
        for key, L, lut in (("spec", self.layers.spectrogram, SPECTRO_LUT), ("heat", self.layers.heatmap, HEAT_LUT)):
            if not L.on:
                continue
            img = an.images.get(f"{key}:{role}")
            if img is None:
                continue
            q = self.images.get(img, lut)
            rows, cols = img.data.shape
            rps = img.rows_per_semitone
            # visible time range -> image columns
            t_a = self.t0
            t_b = self.t0 + grid.width() / self.pps
            c_a = (t_a - img.t0) / img.dt
            c_b = (t_b - img.t0) / img.dt
            # pitch range -> image rows (q row 0 = highest pitch = pitch_low + rows/rps)
            top_pitch = img.pitch_low + rows / rps
            pitch_top_view = PITCH_HIGH + 1 - (self.scroll_y / self.key_h)         # pitch at grid top edge (upper edge of top row)
            pitch_bot_view = pitch_top_view - grid.height() / self.key_h
            r_a = (top_pitch - pitch_top_view) * rps
            r_b = (top_pitch - pitch_bot_view) * rps
            src = QRectF(c_a, r_a, c_b - c_a, r_b - r_a)
            p.save()
            p.setOpacity(L.alpha)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.pps < 400)
            p.drawImage(QRectF(grid), q, src)
            p.restore()

    def _paint_grid(self, p: QPainter, grid: QRect) -> None:
        if not self.layers.grid.on:
            return
        proj = self.project
        beat_px = self.pps * 60.0 / proj.bpm
        b0 = int(np.floor(proj.sec_to_beat(self.t0))) - 1
        b1 = int(np.ceil(proj.sec_to_beat(self.t0 + grid.width() / self.pps))) + 1
        bar_pen = QPen(QColor("#6b7280"), 1)
        beat_pen = QPen(QColor("#3c424d"), 1)
        sub_pen = QPen(QColor("#2b3038"), 1)
        if self.snap > 0 and self.snap < 1 and beat_px * self.snap >= 5:
            steps = int(round(1 / self.snap))
            for b in range(b0, b1):
                for s in range(1, steps):
                    x = self.beat_x(b + s * self.snap)
                    p.setPen(sub_pen)
                    p.drawLine(QPointF(x, grid.top()), QPointF(x, grid.bottom()))
        for b in range(b0, b1):
            x = self.beat_x(b)
            if b % proj.beats_per_bar == 0:
                p.setPen(bar_pen)
            else:
                if beat_px < 4:
                    continue
                p.setPen(beat_pen)
            p.drawLine(QPointF(x, grid.top()), QPointF(x, grid.bottom()))
        if self.layers.beats.on and proj.analysis.beats:
            pen = QPen(QColor(255, 120, 200, 110), 1, Qt.PenStyle.DotLine)
            p.setPen(pen)
            for bt in proj.analysis.beats:
                if self.t0 <= bt <= self.t0 + grid.width() / self.pps:
                    x = self.x_of(bt)
                    p.drawLine(QPointF(x, grid.top()), QPointF(x, grid.top() + 14))

    def _paint_loop(self, p: QPainter, grid: QRect) -> None:
        if not self.loop_on or self.loop_b <= self.loop_a:
            return
        xa, xb = self.x_of(self.loop_a), self.x_of(self.loop_b)
        p.fillRect(QRectF(xa, grid.top(), xb - xa, grid.height()), QColor(255, 255, 255, 12))

    def _paint_ghosts(self, p: QPainter, grid: QRect) -> None:
        L = self.layers.ghosts
        if not L.on:
            return
        role = self.active_source()
        ghosts = self.project.analysis.ghosts.get(role or "", [])
        if not ghosts:
            return
        t_a, t_b = self.t0, self.t0 + grid.width() / self.pps
        lo, hi = self.visible_pitch_range()
        font = QFont(p.font())
        font.setPixelSize(max(8, min(11, self.key_h - 4)))
        p.setFont(font)
        for g in ghosts:
            if g.end < t_a or g.start > t_b or g.conf < self.layers.ghost_min_conf or not (lo <= g.pitch <= hi):
                continue
            x1, x2 = self.x_of(g.start), self.x_of(g.end)
            w = max(3.0, x2 - x1)
            y = self.y_of_pitch_top(g.pitch)
            fill = QColor(255, 200, 60)
            fill.setAlphaF(min(1.0, (0.12 + 0.55 * g.conf) * L.alpha))
            border = QColor(255, 220, 120)
            border.setAlphaF(min(1.0, (0.3 + 0.6 * g.conf) * L.alpha))
            p.setPen(QPen(border, 1, Qt.PenStyle.DashLine))
            p.setBrush(fill)
            r = QRectF(x1, y + 2, w, max(2.0, self.key_h - 4))
            p.drawRect(r)
            if self.layers.ghost_labels and w > 26 and self.key_h >= 11:
                p.setPen(QColor(255, 240, 200, int(230 * L.alpha)))
                p.drawText(r.adjusted(2, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"{int(g.conf * 100)}%")

    def _paint_other_notes(self, p: QPainter, grid: QRect) -> None:
        L = self.layers.others
        if not L.on:
            return
        for t in self.project.midi_tracks():
            if t is self.track:
                continue
            col = QColor(t.color)
            col.setAlphaF(L.alpha)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            for n in t.notes:
                r = self._note_rect(n)
                if r.right() < grid.left() or r.left() > grid.right():
                    continue
                p.drawRect(r.adjusted(0, 3, 0, -3))

    def _note_rect(self, n: Note) -> QRectF:
        x1 = self.beat_x(n.start)
        x2 = self.beat_x(n.end)
        y = self.y_of_pitch_top(n.pitch)
        return QRectF(x1, y + 1, max(3.0, x2 - x1), self.key_h - 2)

    def _paint_notes(self, p: QPainter, grid: QRect) -> None:
        if self.track is None:
            return
        base = QColor(self.track.color)
        font = QFont(p.font())
        font.setPixelSize(max(8, min(11, self.key_h - 4)))
        p.setFont(font)
        for i, n in enumerate(self.track.notes):
            r = self._note_rect(n)
            if r.right() < grid.left() or r.left() > grid.right() or r.bottom() < grid.top() or r.top() > grid.bottom():
                continue
            c = QColor(base)
            c = c.lighter(70 + int(60 * n.velocity / 127))
            p.setBrush(c)
            if i in self.selected:
                p.setPen(QPen(QColor("white"), 2))
            else:
                p.setPen(QPen(base.darker(160), 1))
            p.drawRoundedRect(r, 2, 2)
            if r.width() > 24 and self.key_h >= 11:
                p.setPen(QColor(0, 0, 0, 200))
                label = GM_DRUMS.get(n.pitch, note_name(n.pitch)) if self.track.is_drums else note_name(n.pitch)
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
        # bars
        bar_beats = proj.beats_per_bar
        beat_px = self.pps * 60.0 / proj.bpm
        bar_px = beat_px * bar_beats
        step = 1
        while bar_px * step < 40:
            step *= 2
        b0 = int(np.floor(proj.sec_to_beat(self.t0) / bar_beats)) - 1
        b1 = int(np.ceil(proj.sec_to_beat(self.t0 + self.grid_width() / self.pps) / bar_beats)) + 1
        for bar in range(b0, b1):
            if bar % step:
                continue
            x = self.beat_x(bar * bar_beats)
            p.setPen(QColor("#9aa3b2"))
            p.drawLine(QPointF(x, 0), QPointF(x, 16))
            p.drawText(QPointF(x + 3, 12), str(bar + 1))
        # seconds
        sec_step = 1.0
        while self.pps * sec_step < 60:
            sec_step *= 2
        while self.pps * sec_step > 240 and sec_step > 0.25:
            sec_step /= 2
        s0 = int(np.floor(self.t0 / sec_step))
        s1 = int(np.ceil((self.t0 + self.grid_width() / self.pps) / sec_step)) + 1
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
        """Returns (index, near_right_edge)."""
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
                self.invalidate()
            return
        if pos.x() < self.key_w and pos.y() >= RULER_H:
            if e.button() == Qt.MouseButton.LeftButton and self.track is not None:
                pitch = self.pitch_of_y(pos.y())
                self._mode = "keys"
                self._key_preview(pitch)
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
                # ctrl-drag = duplicate
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
        # create note
        beat = self.snap_beat(self.project.sec_to_beat(self.sec_of(pos.x())), floor=True)
        pitch = self.pitch_of_y(pos.y())
        self.push_undo()
        n = Note(beat, self.default_len if self.snap <= 0 else max(self.snap, self.default_len), pitch, 100)
        self.track.notes.append(n)
        self._create_idx = len(self.track.notes) - 1
        self.selected = {self._create_idx}
        self._mode = "create"
        self._key_preview(pitch)
        self.invalidate()

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
                bar = int(np.floor(beat / self.project.beats_per_bar)) + 1
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
            self.invalidate()
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
            self.invalidate()
            return
        if mode == "erase":
            self._erase_at(pos)
            return
        if mode == "rubber":
            self._rubber = QRectF(self._press, pos).normalized()
            self.invalidate()
            return
        if self.track is None:
            return
        if mode == "create" and self._create_idx is not None:
            n = self.track.notes[self._create_idx]
            beat = self.project.sec_to_beat(self.sec_of(pos.x()))
            end = self.snap_beat(beat) if self.snap > 0 else beat
            n.length = max(self.snap if self.snap > 0 else 0.05, end - n.start)
            pitch = self.pitch_of_y(pos.y())
            if pitch != n.pitch:
                n.pitch = pitch
                self._key_preview(pitch)
            self.invalidate()
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
            self.invalidate()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        mode = self._mode
        self._mode = None
        self._key_release()
        if mode == "rubber" and self._rubber is not None and self.track is not None:
            for i, n in enumerate(self.track.notes):
                if self._rubber.intersects(self._note_rect(n)):
                    self.selected.add(i)
            self._rubber = None
            self.invalidate()
            return
        if mode in ("create", "move", "resize", "erase") and self.track is not None:
            if mode == "create" and self._create_idx is not None:
                self.default_len = self.track.notes[self._create_idx].length
            self._create_idx = None
            if mode == "move" and not getattr(self, "_moved", True):
                # plain click on a note: nothing changed
                st = self._undo.get(self.track.id)
                if st:
                    st.pop()
                self.invalidate()
                return
            self._commit()
        self.invalidate()

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        pass

    def _erase_at(self, pos: QPointF) -> None:
        idx, _ = self._note_at(pos)
        if idx is not None and self.track is not None:
            del self.track.notes[idx]
            self.selected = {i - 1 if i > idx else i for i in self.selected if i != idx}
            self.invalidate()

    def _key_preview(self, pitch: int) -> None:
        self._key_release()
        self._preview_pitch = pitch
        if self.engine is not None and self.track is not None:
            self.engine.preview_note(self.track, pitch, 100, duration=0.6)
        self.invalidate()

    def _key_release(self) -> None:
        if self._preview_pitch is not None and self.engine is not None and self.track is not None:
            self.engine.preview_off(self.track, self._preview_pitch)
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
        elif mods & Qt.KeyboardModifier.AltModifier:
            pitch_at = PITCH_HIGH - (pos.y() - RULER_H + self.scroll_y) / self.key_h
            self.key_h = max(MIN_KEY_H, min(MAX_KEY_H, self.key_h + (1 if delta > 0 else -1)))
            self.scroll_y = (PITCH_HIGH - pitch_at) * self.key_h - (pos.y() - RULER_H)
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            self.t0 -= delta / 120.0 * (self.grid_width() / self.pps) * 0.15
        else:
            self.scroll_y -= delta / 120.0 * self.key_h * 3
        self.clamp_scroll()
        self.view_changed.emit()
        self.invalidate()

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
            self.invalidate()
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
            # duplicate selection right after itself
            self.push_undo()
            sel = [self.track.notes[i] for i in sorted(self.selected)]
            span = max(n.end for n in sel) - min(n.start for n in sel)
            if self.snap > 0:
                span = max(self.snap, np.ceil(span / self.snap) * self.snap)
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
