"""Main window: toolbar/transport, track panel, piano roll, layers panel."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QGridLayout, QLineEdit,
                               QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QScrollBar, QSlider,
                               QToolBar, QToolButton, QWidget, QInputDialog)

from .. import paths
from ..audio.engine import Engine
from ..midi_export import export_midi
from ..project import Project, Track, N_PITCHES
from .dialogs import AnalysisDialog, InstrumentDialog, SettingsDialog, UpdateDialog
from .. import __version__
from .layers_panel import LayersPanel
from .piano_roll import PianoRoll
from .track_panel import TrackPanel

SNAP_OPTIONS = [("1 такт", 4.0), ("1/2", 2.0), ("1/4", 1.0), ("1/8", 0.5), ("1/8 T", 1 / 3), ("1/16", 0.25),
                ("1/16 T", 1 / 6), ("1/32", 0.125), ("Без привязки", 0.0)]
AUDIO_FILTER = "Аудио (*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.wma *.aiff);;Все файлы (*)"


def load_settings() -> dict:
    try:
        return json.loads(paths.SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(s: dict) -> None:
    try:
        paths.SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


class MainWindow(QMainWindow):
    def __init__(self, engine: Engine):
        super().__init__()
        self.engine = engine
        self.project: Project | None = None
        self.settings = load_settings()
        self.setWindowTitle(f"StemNotes {__version__}")
        self.resize(1500, 900)

        # ---- centre: piano roll + scrollbars
        centre = QWidget()
        g = QGridLayout(centre)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(0)
        self.roll = PianoRoll()
        self.hbar = QScrollBar(Qt.Orientation.Horizontal)
        self.vbar = QScrollBar(Qt.Orientation.Vertical)
        g.addWidget(self.roll, 0, 0)
        g.addWidget(self.vbar, 0, 1)
        g.addWidget(self.hbar, 1, 0)
        self.setCentralWidget(centre)
        self.hbar.valueChanged.connect(self._hbar_moved)
        self.vbar.valueChanged.connect(self._vbar_moved)
        self.roll.view_changed.connect(self._sync_bars)
        self.roll.notes_changed.connect(self._notes_changed)
        self.roll.seek_requested.connect(self._seek)
        self.roll.loop_changed.connect(self._loop_changed)
        self.roll.status.connect(self.statusBar().showMessage)

        # ---- docks
        self.tracks = TrackPanel()
        dock_l = QDockWidget("Дорожки", self)
        dock_l.setWidget(self.tracks)
        dock_l.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock_l.setMinimumWidth(390)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_l)
        self.tracks.track_selected.connect(self._track_selected)
        self.tracks.mix_changed.connect(self._mix_changed)
        self.tracks.ghost_changed.connect(self.roll.invalidate)
        self.tracks.instrument_requested.connect(self._choose_instrument)
        self.tracks.add_midi_requested.connect(self._add_midi_track)
        self.tracks.add_audio_requested.connect(self._add_audio_track)
        self.tracks.remove_requested.connect(self._remove_track)

        self.layers = LayersPanel(self.roll.layers)
        dock_r = QDockWidget("Слои", self)
        dock_r.setWidget(self.layers)
        dock_r.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock_r.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_r)
        self.layers.changed.connect(self.roll.invalidate)
        self.layers.import_requested.connect(self._import_ghosts)

        self._build_toolbar()
        self.statusBar().showMessage("Готово")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self.engine.on_vst_rendered = lambda tid: None
        self.engine.on_stopped = lambda: None
        QApplication.instance().installEventFilter(self)   # Space / Home work wherever the focus is

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() == QEvent.Type.KeyPress and not ev.isAutoRepeat():
            fw = QApplication.focusWidget()
            typing = isinstance(fw, (QLineEdit, QAbstractSpinBox)) or (fw is not None and fw.inherits("QTextEdit"))
            if ev.key() == Qt.Key.Key_Space and not typing and ev.modifiers() in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier):
                if self.isActiveWindow() or (fw is not None and fw.window() is self):
                    self.toggle_play()
                    return True
            if ev.key() == Qt.Key.Key_Home and not typing and self.isActiveWindow():
                self.stop_to_start()
                return True
            if ev.key() == Qt.Key.Key_Escape and not typing and self.isActiveWindow() and self.engine.playing:
                self.toggle_play()
                return True
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------------ toolbar
    def _build_toolbar(self) -> None:
        tb = QToolBar("Главная")
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip + (f"  ({shortcut})" if shortcut else ""))
            tb.addAction(a)
            return a

        act("Открыть песню", self.open_song, "Ctrl+O", "Выбрать аудиофайл: создаётся проект и предлагается анализ")
        act("Открыть проект", self.open_project, "Ctrl+Shift+O")
        act("Сохранить", self.save_project, "Ctrl+S")
        tb.addSeparator()
        act("ИИ-анализ", self.run_analysis, "Ctrl+I", "Разделить на стемы, найти темп и призрачные ноты")
        act("Экспорт MIDI", self.export_midi, "Ctrl+E", "Сохранить все MIDI-дорожки в один .mid для FL Studio")
        tb.addSeparator()

        self.play_btn = QToolButton()
        self.play_btn.setText("▶ Играть")
        self.play_btn.setToolTip("Играть / Пауза (Пробел). Home — в начало")
        self.play_btn.clicked.connect(self.toggle_play)
        tb.addWidget(self.play_btn)
        stop = QToolButton()
        stop.setText("■ В начало")
        stop.setToolTip("Стоп и в начало (Home)")
        stop.clicked.connect(self.stop_to_start)
        tb.addWidget(stop)
        self.loop_btn = QToolButton()
        self.loop_btn.setText("⟲ Цикл")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setToolTip("Повторять выделенный на линейке участок (L)")
        self.loop_btn.setShortcut(QKeySequence("L"))
        self.loop_btn.toggled.connect(self._loop_toggled)
        tb.addWidget(self.loop_btn)
        self.pos_lbl = QLabel("  0:00.00  |  такт 1  ")
        self.pos_lbl.setStyleSheet("font-family:monospace; font-size:13px;")
        tb.addWidget(self.pos_lbl)
        tb2 = QToolBar("Сетка и звук")
        tb2.setMovable(False)
        self.addToolBarBreak()
        self.addToolBar(tb2)
        tb = tb2

        tb.addWidget(QLabel(" BPM "))
        self.bpm = QDoubleSpinBox()
        self.bpm.setRange(20.0, 300.0)
        self.bpm.setDecimals(2)
        self.bpm.setValue(120.0)
        self.bpm.setToolTip("Темп сетки. Ноты хранятся в долях, поэтому смена BPM сдвигает их относительно аудио")
        self.bpm.valueChanged.connect(self._bpm_changed)
        self.bpm.editingFinished.connect(self.roll.setFocus)
        tb.addWidget(self.bpm)
        tb.addWidget(QLabel(" Сдвиг сетки, с "))
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-10.0, 600.0)
        self.offset.setDecimals(3)
        self.offset.setSingleStep(0.01)
        self.offset.setToolTip("Время (с) первой доли первого такта в песне")
        self.offset.valueChanged.connect(self._offset_changed)
        self.offset.editingFinished.connect(self.roll.setFocus)
        tb.addWidget(self.offset)
        half = QToolButton()
        half.setText("½ BPM")
        half.setToolTip("Поделить BPM пополам (если детектор нашёл удвоенный темп)")
        half.clicked.connect(lambda: self.bpm.setValue(self.bpm.value() / 2))
        dbl = QToolButton()
        dbl.setText("×2 BPM")
        dbl.clicked.connect(lambda: self.bpm.setValue(self.bpm.value() * 2))
        tb.addWidget(half)
        tb.addWidget(dbl)
        tb.addSeparator()
        tb.addWidget(QLabel(" Привязка "))
        self.snap = QComboBox()
        for label, val in SNAP_OPTIONS:
            self.snap.addItem(label, val)
        self.snap.setCurrentIndex(5)
        self.snap.currentIndexChanged.connect(lambda i: (self.roll.set_snap(self.snap.currentData()), self.roll.setFocus()))
        self.snap.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.roll.set_snap(self.snap.currentData())
        tb.addWidget(self.snap)
        tb.addSeparator()
        tb.addWidget(QLabel(" Громкость "))
        vol = QSlider(Qt.Orientation.Horizontal)
        vol.setRange(0, 100)
        vol.setValue(int(self.engine.master * 100))
        vol.setFixedWidth(90)
        vol.valueChanged.connect(lambda v: setattr(self.engine, "master", v / 100.0))
        tb.addWidget(vol)
        act("Звук…", self.audio_settings, None, "Выбор устройства вывода и размера буфера")
        act("Обновить", self.check_updates, None, "Обновить код программы из GitHub (модели и настройки сохраняются)")
        self.vst_btn = QPushButton("Окно VST")
        self.vst_btn.setToolTip("Открыть интерфейс VST3-плагина активной дорожки")
        self.vst_btn.clicked.connect(self.open_vst_editor)
        tb.addWidget(self.vst_btn)

    # ------------------------------------------------------------------ project lifecycle
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.engine.set_project(project) if project else None
        self.roll.set_project(project, self.engine)
        self.tracks.set_project(project)
        if project is not None:
            self.bpm.blockSignals(True); self.bpm.setValue(project.bpm); self.bpm.blockSignals(False)
            self.offset.blockSignals(True); self.offset.setValue(project.offset); self.offset.blockSignals(False)
            self.layers.set_sources(project.analysis.sources or [t.role for t in project.audio_tracks() if t.role])
            self.layers.set_engines(project.analysis.engines_for())
            self.setWindowTitle(f"StemNotes {__version__} — {project.name}")
            if self.tracks.active_id:
                self._track_selected(self.tracks.active_id)
        self._sync_bars()
        self.roll.invalidate()

    def open_song(self) -> None:
        start = self.settings.get("last_dir", str(Path.home()))
        f, _ = QFileDialog.getOpenFileName(self, "Открыть песню", start, AUDIO_FILTER)
        if not f:
            return
        song = Path(f)
        self.settings["last_dir"] = str(song.parent)
        save_settings(self.settings)
        name = song.stem
        pdir = paths.PROJECTS_DIR / name
        if (pdir / "project.json").exists():
            r = QMessageBox.question(self, "Проект существует", f"Проект «{name}» уже есть. Открыть его?\n(Нет — создать новый с суффиксом)")
            if r == QMessageBox.StandardButton.Yes:
                self._load_project(pdir)
                return
            i = 2
            while (paths.PROJECTS_DIR / f"{name}_{i}").exists():
                i += 1
            pdir = paths.PROJECTS_DIR / f"{name}_{i}"
        from ..analysis.pipeline import ensure_song_in_project, ensure_audio_track
        project = Project(pdir, name)
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            ensure_song_in_project(project, song)
            ensure_audio_track(project, "mix", project.resolve(project.song_path), "Оригинал")
            project.add_track(Track(name="Мелодия", kind="midi", instrument={"type": "sf2", "bank": 0, "program": 0, "drums": False}, ghost_source="mix"))
            project.save()
        finally:
            QApplication.restoreOverrideCursor()
        self.set_project(project)
        r = QMessageBox.question(self, "ИИ-анализ", "Запустить анализ песни (темп, стемы, призрачные ноты)?\n"
                                 "Это можно сделать позже кнопкой «ИИ-анализ».")
        if r == QMessageBox.StandardButton.Yes:
            self.run_analysis()

    def open_project(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Открыть проект", str(paths.PROJECTS_DIR), "Проект StemNotes (project.json)")
        if f:
            self._load_project(Path(f).parent)

    def _load_project(self, pdir: Path) -> None:
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            project = Project.load(pdir)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть проект:\n{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.set_project(project)

    def save_project(self) -> None:
        if self.project is None:
            return
        for t in self.project.midi_tracks():
            if t.instrument.get("type") == "vst3" and t.id in self.engine.vst:
                st = self.engine.vst[t.id].get_state()
                if st:
                    t.instrument["state"] = st
        self.project.save()
        self.statusBar().showMessage(f"Сохранено: {self.project.file}", 4000)

    def closeEvent(self, e) -> None:
        if self.project is not None and self.project.dirty:
            r = QMessageBox.question(self, "Выход", "Сохранить проект перед выходом?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Cancel:
                e.ignore()
                return
            if r == QMessageBox.StandardButton.Yes:
                self.save_project()
        self.engine.stop()
        self.engine.stop_stream()
        e.accept()

    # ------------------------------------------------------------------ analysis / export
    def run_analysis(self) -> None:
        if self.project is None:
            QMessageBox.information(self, "Анализ", "Сначала откройте песню (Ctrl+O).")
            return
        try:
            import torch  # noqa: F401
            import basic_pitch  # noqa: F401
        except Exception as e:
            QMessageBox.warning(self, "ИИ-модуль", "Библиотеки анализа не установлены.\n"
                                "Запустите setup_windows.bat (или setup_linux.sh) с установкой ИИ-модуля.\n\n" + str(e))
            return
        self.engine.stop()
        first = not (self.project.dir / "stems").exists()
        dlg = AnalysisDialog(self.project, self, first_time=first, settings=self.settings)
        dlg.finished_ok.connect(self._analysis_done)
        dlg.exec()
        save_settings(self.settings)

    def _analysis_done(self) -> None:
        self.project.analysis.load()
        self.set_project(self.project)
        self.statusBar().showMessage("Анализ завершён", 5000)

    def export_midi(self) -> None:
        if self.project is None:
            return
        default = str(self.project.dir / f"{self.project.name}.mid")
        f, _ = QFileDialog.getSaveFileName(self, "Экспорт MIDI", default, "MIDI (*.mid)")
        if not f:
            return
        r = QMessageBox.question(self, "Экспорт MIDI", "Выровнять по началу песни?\n\n"
                                 "Да — время 0 в MIDI = начало аудиофайла (удобно класть рядом с оригиналом в FL Studio).\n"
                                 "Нет — время 0 = первая доля первого такта сетки.")
        try:
            n = export_midi(self.project, Path(f), align_to_song_start=(r == QMessageBox.StandardButton.Yes))
            self.statusBar().showMessage(f"Экспортировано нот: {n} → {f}", 6000)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Экспорт MIDI", str(e))

    # ------------------------------------------------------------------ transport
    def toggle_play(self) -> None:
        if self.project is None:
            return
        if self.engine.playing:
            self.engine.stop()
        else:
            if self.roll.loop_on and self.roll.loop_b > self.roll.loop_a and not (self.roll.loop_a <= self.engine.position < self.roll.loop_b):
                self.engine.seek(self.roll.loop_a)
            self.engine.play()
        self._update_play_btn()

    def stop_to_start(self) -> None:
        self.engine.stop()
        self.engine.seek(self.roll.loop_a if self.roll.loop_on else 0.0)
        self.roll.t0 = max(0.0, self.engine.position - 1.0) if self.roll.loop_on else 0.0
        self._sync_bars()
        self._update_play_btn()

    def _seek(self, sec: float) -> None:
        self.engine.seek(sec)
        self.roll.set_playhead(sec)

    def _loop_changed(self, on: bool, a: float, b: float) -> None:
        self.engine.set_loop(on, a, b)
        self.loop_btn.blockSignals(True)
        self.loop_btn.setChecked(on)
        self.loop_btn.blockSignals(False)

    def _loop_toggled(self, on: bool) -> None:
        self.roll.loop_on = on and self.roll.loop_b > self.roll.loop_a
        self.engine.set_loop(self.roll.loop_on, self.roll.loop_a, self.roll.loop_b)
        if on and not self.roll.loop_on:
            self.statusBar().showMessage("Сначала выделите участок на линейке (тянуть мышью)", 4000)
            self.loop_btn.blockSignals(True); self.loop_btn.setChecked(False); self.loop_btn.blockSignals(False)
        self.roll.invalidate()

    def _update_play_btn(self) -> None:
        self.play_btn.setText("■ Стоп" if self.engine.playing else "▶ Играть")

    def _tick(self) -> None:
        if self.project is None:
            return
        pos = self.engine.position
        self.roll.set_playhead(pos)
        beat = self.project.sec_to_beat(pos)
        m, s = divmod(pos, 60)
        if beat >= 0:
            bar = int(beat // self.project.beats_per_bar) + 1
            bb = beat - (bar - 1) * self.project.beats_per_bar
            self.pos_lbl.setText(f"  {int(m)}:{s:05.2f}  |  такт {bar}  доля {int(bb) + 1}  ")
        else:
            self.pos_lbl.setText(f"  {int(m)}:{s:05.2f}  |  до начала сетки  ")
        if not self.engine.playing and self.play_btn.text().startswith("■"):
            self._update_play_btn()
        if self.engine.vst_rendering:
            self.statusBar().showMessage("Рендер VST3-дорожки…", 500)

    # ------------------------------------------------------------------ edits
    def _bpm_changed(self, v: float) -> None:
        if self.project is None:
            return
        self.project.bpm = v
        self.project.dirty = True
        self.engine.rebuild_events()
        self.engine.mark_all_vst_dirty()
        self.roll.invalidate()

    def _offset_changed(self, v: float) -> None:
        if self.project is None:
            return
        self.project.offset = v
        self.project.dirty = True
        self.engine.rebuild_events()
        self.engine.mark_all_vst_dirty()
        self.roll.invalidate()

    def _notes_changed(self, tid: str) -> None:
        self.engine.rebuild_events()
        self.engine.mark_vst_dirty(tid)
        self.engine.kick_vst_render()

    def _mix_changed(self) -> None:
        if self.project:
            self.project.dirty = True
        self.engine.update_mix()

    def _track_selected(self, tid: str) -> None:
        if self.project is None:
            return
        t = self.project.track_by_id(tid)
        self.roll.set_track(t)
        self.vst_btn.setEnabled(t is not None and t.kind == "midi" and t.instrument.get("type") == "vst3")

    def _import_ghosts(self, engines, only_loop: bool, snap: bool) -> None:
        if self.project is None or self.roll.track is None:
            self.statusBar().showMessage("Выберите MIDI-дорожку, в которую скопировать ноты", 4000)
            return
        n = self.roll.import_ghosts(engines, only_loop, snap)
        self.statusBar().showMessage(f"Скопировано нот: {n} (Ctrl+Z — отменить)", 5000)

    def _choose_instrument(self, tid: str) -> None:
        t = self.project.track_by_id(tid) if self.project else None
        if t is None:
            return
        dlg = InstrumentDialog(t, self)
        if dlg.exec() != InstrumentDialog.DialogCode.Accepted or not dlg.result_instrument:
            return
        inst = dlg.result_instrument
        try:
            if inst["type"] == "vst3":
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                names = []
                try:
                    from ..audio.vst import VSTInstrument
                    names = VSTInstrument.plugin_names(inst["path"])
                finally:
                    QApplication.restoreOverrideCursor()
                plugin_name = None
                if len(names) > 1:
                    plugin_name, ok = QInputDialog.getItem(self, "VST3", "В файле несколько плагинов, выберите:", names, 0, False)
                    if not ok:
                        return
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    self.engine.replace_vst(t, inst["path"], plugin_name)
                finally:
                    QApplication.restoreOverrideCursor()
            else:
                t.instrument = inst
                self.engine.refresh_instruments()
                self.engine.rebuild_events()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Инструмент", f"Не удалось загрузить:\n{e}")
            return
        self.project.dirty = True
        self.tracks.refresh()
        self._track_selected(tid)
        self.roll.invalidate()

    def open_vst_editor(self) -> None:
        tid = self.tracks.active_id
        inst = self.engine.vst.get(tid) if tid else None
        if inst is None:
            self.statusBar().showMessage("У активной дорожки нет VST3-инструмента", 3000)
            return
        self.engine.stop()
        self.statusBar().showMessage("Окно плагина открыто. Закройте его, чтобы вернуться.", 3000)
        try:
            inst.show_editor()
        except Exception as e:
            QMessageBox.warning(self, "VST3", f"Не удалось открыть окно плагина:\n{e}")
        t = self.project.track_by_id(tid)
        if t is not None:
            t.instrument["state"] = inst.get_state()
            self.project.dirty = True
            self.engine.mark_vst_dirty(tid)
            self.engine.kick_vst_render()

    def _add_midi_track(self) -> None:
        if self.project is None:
            return
        n = len(self.project.midi_tracks()) + 1
        t = Track(name=f"MIDI {n}", kind="midi", instrument={"type": "sf2", "bank": 0, "program": 0, "drums": False},
                  ghost_source=self.roll.active_source())
        self.project.add_track(t)
        self.engine.refresh_instruments()
        self.tracks.rebuild()
        self.tracks.select(t.id)

    def _add_audio_track(self) -> None:
        if self.project is None:
            return
        f, _ = QFileDialog.getOpenFileName(self, "Аудиофайл", self.settings.get("last_dir", ""), AUDIO_FILTER)
        if not f:
            return
        t = Track(name=Path(f).stem, kind="audio", path=f, role=None)
        self.project.add_track(t)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.engine.reload_audio()
        finally:
            QApplication.restoreOverrideCursor()
        self.tracks.rebuild()

    def _remove_track(self, tid: str) -> None:
        if self.project is None:
            return
        t = self.project.track_by_id(tid)
        if t is None:
            return
        if QMessageBox.question(self, "Удалить дорожку", f"Удалить «{t.name}»?") != QMessageBox.StandardButton.Yes:
            return
        self.project.tracks.remove(t)
        self.project.dirty = True
        self.engine.refresh_instruments()
        self.engine.rebuild_events()
        self.tracks.active_id = None
        self.tracks.rebuild()
        if self.tracks.active_id:
            self._track_selected(self.tracks.active_id)
        else:
            self.roll.set_track(None)

    def check_updates(self) -> None:
        UpdateDialog(self).exec()

    def audio_settings(self) -> None:
        dlg = SettingsDialog(self.engine, self.settings, self)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            self.settings["device"] = dlg.device()
            self.settings["block"] = dlg.block()
            save_settings(self.settings)
            self.engine.start(dlg.device(), dlg.block())
            if not self.engine.available:
                QMessageBox.warning(self, "Звук", f"Не удалось открыть устройство:\n{self.engine.error}")

    # ------------------------------------------------------------------ scrollbars
    def _sync_bars(self) -> None:
        self.hbar.blockSignals(True)
        self.vbar.blockSignals(True)
        total_ms = int(self.roll.total_seconds() * 1000)
        page_ms = int(self.roll.grid_width() / self.roll.pps * 1000)
        self.hbar.setRange(0, max(0, total_ms - page_ms))
        self.hbar.setPageStep(page_ms)
        self.hbar.setValue(int(self.roll.t0 * 1000))
        self.vbar.setRange(0, max(0, self.roll.total_height() - self.roll.grid_height()))
        self.vbar.setPageStep(self.roll.grid_height())
        self.vbar.setValue(int(self.roll.scroll_y))
        self.hbar.blockSignals(False)
        self.vbar.blockSignals(False)

    def _hbar_moved(self, v: int) -> None:
        self.roll.t0 = v / 1000.0
        self.roll.invalidate()

    def _vbar_moved(self, v: int) -> None:
        self.roll.scroll_y = v
        self.roll.invalidate()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._sync_bars()
