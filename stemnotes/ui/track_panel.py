"""Track list: colour, name, mute/solo, volume, instrument, ghost source."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QInputDialog, QLabel, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget)

from ..gm import GM_PROGRAMS
from ..project import Project, Track

SOURCE_LABELS = {"mix": "Оригинал", "drums": "Ударные", "bass": "Бас", "guitar": "Гитара", "piano": "Пианино",
                 "vocals": "Вокал", "other": "Остальное"}


def instrument_label(track: Track) -> str:
    inst = track.instrument
    if inst.get("type", "sf2") == "sf2":
        if inst.get("drums"):
            return "Ударные (GM)"
        return GM_PROGRAMS[int(inst.get("program", 0)) % 128]
    return "VST3: " + (inst.get("name") or inst.get("path", "?").split("/")[-1].split("\\")[-1])


class TrackRow(QFrame):
    def __init__(self, panel: "TrackPanel", track: Track):
        super().__init__(panel)
        self.panel = panel
        self.track = track
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("trackRow")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)

        top = QHBoxLayout()
        self.swatch = QLabel()
        self.swatch.setFixedSize(10, 22)
        self.swatch.setStyleSheet(f"background:{track.color}; border-radius:2px;")
        top.addWidget(self.swatch)
        self.name = QLabel(track.name)
        self.name.setStyleSheet("font-weight:600;")
        self.name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top.addWidget(self.name)
        kind = QLabel("MIDI" if track.kind == "midi" else "АУДИО")
        kind.setStyleSheet("color:#8a93a5; font-size:10px;")
        top.addWidget(kind)
        self.mute = QToolButton()
        self.mute.setText("M")
        self.mute.setCheckable(True)
        self.mute.setChecked(track.mute)
        self.mute.setToolTip("Mute (выключить дорожку)")
        self.mute.toggled.connect(self._on_mute)
        self.solo = QToolButton()
        self.solo.setText("S")
        self.solo.setCheckable(True)
        self.solo.setChecked(track.solo)
        self.solo.setToolTip("Solo (только эта дорожка)")
        self.solo.toggled.connect(self._on_solo)
        top.addWidget(self.mute)
        top.addWidget(self.solo)
        lay.addLayout(top)

        bottom = QHBoxLayout()
        self.vol = QSlider(Qt.Orientation.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(int(track.volume * 100))
        self.vol.setFixedWidth(70)
        self.vol.setToolTip("Громкость")
        self.vol.valueChanged.connect(self._on_vol)
        bottom.addWidget(self.vol)
        if track.kind == "midi":
            self.inst = QPushButton(instrument_label(track))
            self.inst.setToolTip("Инструмент (SoundFont или VST3)")
            self.inst.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.inst.clicked.connect(lambda: panel.instrument_requested.emit(track.id))
            bottom.addWidget(self.inst)
            src_lbl = QLabel("Фон:")
            src_lbl.setStyleSheet("color:#8a93a5;")
            bottom.addWidget(src_lbl)
            self.ghost = QComboBox()
            self.ghost.setToolTip("Какой стем показывать для этой дорожки: его волна, спектрограмма, ИИ-вероятности и призрачные ноты.\n"
                                  "Стемы появляются после «ИИ-анализ» с разделением на дорожки.")
            self.ghost.setMinimumWidth(100)
            bottom.addWidget(self.ghost)
            self.refresh_sources()
            self.ghost.currentIndexChanged.connect(self._on_ghost)
        else:
            lbl = QLabel(SOURCE_LABELS.get(track.role or "", track.role or ""))
            lbl.setStyleSheet("color:#8a93a5;")
            bottom.addWidget(lbl, 1)
        lay.addLayout(bottom)
        self.set_active(False)

    def refresh_sources(self) -> None:
        if self.track.kind != "midi":
            return
        self.ghost.blockSignals(True)
        self.ghost.clear()
        self.ghost.addItem("— нет —", None)
        sources = self.panel.project.analysis.sources if self.panel.project else []
        roles = list(sources) if sources else list(SOURCE_LABELS)
        for r in roles:
            self.ghost.addItem(SOURCE_LABELS.get(r, r), r)
        idx = self.ghost.findData(self.track.ghost_source)
        self.ghost.setCurrentIndex(max(0, idx))
        self.ghost.blockSignals(False)

    def refresh(self) -> None:
        self.name.setText(self.track.name)
        self.mute.blockSignals(True); self.mute.setChecked(self.track.mute); self.mute.blockSignals(False)
        self.solo.blockSignals(True); self.solo.setChecked(self.track.solo); self.solo.blockSignals(False)
        self.vol.blockSignals(True); self.vol.setValue(int(self.track.volume * 100)); self.vol.blockSignals(False)
        if self.track.kind == "midi":
            self.inst.setText(instrument_label(self.track))
        self.swatch.setStyleSheet(f"background:{self.track.color}; border-radius:2px;")

    def set_active(self, active: bool) -> None:
        self.setStyleSheet("#trackRow{background:%s; border:1px solid %s; border-radius:4px;}" %
                           (("#2d3340", "#6fa8ff") if active else ("#22252c", "#2f333c")))

    def _on_mute(self, v: bool) -> None:
        self.track.mute = v
        self.panel.mix_changed.emit()

    def _on_solo(self, v: bool) -> None:
        self.track.solo = v
        self.panel.mix_changed.emit()

    def _on_vol(self, v: int) -> None:
        self.track.volume = v / 100.0
        self.panel.mix_changed.emit()

    def _on_ghost(self, _i: int) -> None:
        self.track.ghost_source = self.ghost.currentData()
        self.panel.ghost_changed.emit()

    def mousePressEvent(self, e) -> None:
        self.panel.select(self.track.id)
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        name, ok = QInputDialog.getText(self, "Имя дорожки", "Название:", text=self.track.name)
        if ok and name.strip():
            self.track.name = name.strip()
            self.name.setText(self.track.name)
            self.panel.project.dirty = True


class TrackPanel(QWidget):
    track_selected = Signal(str)
    mix_changed = Signal()
    ghost_changed = Signal()
    instrument_requested = Signal(str)
    add_midi_requested = Signal()
    add_audio_requested = Signal()
    remove_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self.rows: dict[str, TrackRow] = {}
        self.active_id: str | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inner = QWidget()
        self.inner_lay = QVBoxLayout(self.inner)
        self.inner_lay.setContentsMargins(0, 0, 0, 0)
        self.inner_lay.setSpacing(4)
        self.inner_lay.addStretch(1)
        self.scroll.setWidget(self.inner)
        lay.addWidget(self.scroll, 1)

        btns = QHBoxLayout()
        b_midi = QPushButton("+ MIDI")
        b_midi.setToolTip("Добавить MIDI-дорожку")
        b_midi.clicked.connect(self.add_midi_requested)
        b_audio = QPushButton("+ Аудио")
        b_audio.setToolTip("Добавить аудиофайл как дорожку")
        b_audio.clicked.connect(self.add_audio_requested)
        b_del = QPushButton("Удалить")
        b_del.clicked.connect(lambda: self.active_id and self.remove_requested.emit(self.active_id))
        btns.addWidget(b_midi)
        btns.addWidget(b_audio)
        btns.addWidget(b_del)
        lay.addLayout(btns)
        btns2 = QHBoxLayout()
        b_all_off = QPushButton("Все выкл")
        b_all_off.setToolTip("Заглушить все дорожки")
        b_all_off.clicked.connect(lambda: self.set_all_mute(True))
        b_all_on = QPushButton("Все вкл")
        b_all_on.setToolTip("Снять mute и solo со всех дорожек")
        b_all_on.clicked.connect(lambda: self.set_all_mute(False))
        b_only = QPushButton("Только активная")
        b_only.setToolTip("Слышать только выбранную дорожку (solo на ней, остальное без solo)")
        b_only.clicked.connect(self.solo_active)
        btns2.addWidget(b_all_off)
        btns2.addWidget(b_all_on)
        btns2.addWidget(b_only)
        lay.addLayout(btns2)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.rebuild()

    def rebuild(self) -> None:
        for r in self.rows.values():
            self.inner_lay.removeWidget(r)
            r.deleteLater()
        self.rows.clear()
        if self.project is None:
            return
        for i, t in enumerate(self.project.tracks):
            row = TrackRow(self, t)
            self.rows[t.id] = row
            self.inner_lay.insertWidget(i, row)
        if self.active_id not in self.rows:
            midi = self.project.midi_tracks()
            self.active_id = midi[0].id if midi else None
        for tid, row in self.rows.items():
            row.set_active(tid == self.active_id)

    def refresh(self) -> None:
        for row in self.rows.values():
            row.refresh()

    def refresh_sources(self) -> None:
        for row in self.rows.values():
            row.refresh_sources()

    def select(self, tid: str) -> None:
        self.active_id = tid
        for k, row in self.rows.items():
            row.set_active(k == tid)
        self.track_selected.emit(tid)

    def set_all_mute(self, mute: bool) -> None:
        if self.project is None:
            return
        for t in self.project.tracks:
            t.mute = mute
            if not mute:
                t.solo = False
        self.refresh()
        self.mix_changed.emit()

    def solo_active(self) -> None:
        if self.project is None or self.active_id is None:
            return
        for t in self.project.tracks:
            t.solo = (t.id == self.active_id)
            if t.solo:
                t.mute = False
        self.refresh()
        self.mix_changed.emit()
