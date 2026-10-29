"""Background layer controls for the piano roll."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from .piano_roll import Layer, LayerSettings
from .track_panel import SOURCE_LABELS
from ..analysis.engines import ENGINE_BY_ID, engine_color, engine_label, engine_description


class LayerRow(QWidget):
    def __init__(self, title: str, layer: Layer, on_change, tip: str = ""):
        super().__init__()
        self.layer = layer
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.cb = QCheckBox(title)
        self.cb.setChecked(layer.on)
        self.cb.setToolTip(tip)
        self.sl = QSlider(Qt.Orientation.Horizontal)
        self.sl.setRange(0, 100)
        self.sl.setValue(int(layer.alpha * 100))
        self.sl.setFixedWidth(80)
        self.sl.setToolTip("Прозрачность")
        lay.addWidget(self.cb, 1)
        lay.addWidget(self.sl)

        def cb_changed(v):
            layer.on = bool(v)
            on_change()

        def sl_changed(v):
            layer.alpha = v / 100.0
            on_change()
        self.cb.toggled.connect(cb_changed)
        self.sl.valueChanged.connect(sl_changed)


class LayersPanel(QWidget):
    changed = Signal()
    import_requested = Signal(object, bool, bool)   # engine ids (set or None), only loop region, snap to grid

    def __init__(self, settings: LayerSettings, parent=None):
        super().__init__(parent)
        self.s = settings
        from PySide6.QtWidgets import QScrollArea
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)

        src_box = QGroupBox("Источник фона")
        f = QFormLayout(src_box)
        self.source = QComboBox()
        self.source.setToolTip("Какой стем показывать как волну/спектрограмму/призрачные ноты")
        self.source.addItem("Как у дорожки", "auto")
        self.source.currentIndexChanged.connect(self._on_source)
        f.addRow(self.source)
        lay.addWidget(src_box)

        box = QGroupBox("Слои (вкл / прозрачность)")
        v = QVBoxLayout(box)
        v.addWidget(LayerRow("Волна", self.s.waveform, self._emit, "Форма волны стема: где звук, где тишина"))
        v.addWidget(LayerRow("Спектрограмма", self.s.spectrogram, self._emit, "Спектр по нотам (CQT): яркость = энергия на этой высоте"))
        v.addWidget(LayerRow("ИИ-вероятности", self.s.heatmap, self._emit, "Тепловая карта нейросети: насколько вероятна нота в каждый момент"))
        v.addWidget(LayerRow("Призрачные ноты", self.s.ghosts, self._emit, "Кандидаты нот с процентом уверенности"))
        v.addWidget(LayerRow("Другие дорожки", self.s.others, self._emit, "Ноты остальных MIDI-дорожек бледным цветом"))
        v.addWidget(LayerRow("Сетка", self.s.grid, self._emit))
        v.addWidget(LayerRow("Найденные доли", self.s.beats, self._emit, "Метки долей, найденных при анализе темпа (розовые чёрточки у линейки)"))
        lay.addWidget(box)

        gbox = QGroupBox("Призрачные ноты")
        g = QFormLayout(gbox)
        self.thr = QSlider(Qt.Orientation.Horizontal)
        self.thr.setRange(0, 95)
        self.thr.setValue(int(self.s.ghost_min_conf * 100))
        self.thr_lbl = QLabel(f"{self.thr.value()}%")
        self.thr.valueChanged.connect(self._on_thr)
        row = QHBoxLayout()
        row.addWidget(self.thr, 1)
        row.addWidget(self.thr_lbl)
        g.addRow("Скрывать ниже:", row)
        self.labels = QCheckBox("Показывать проценты")
        self.labels.setChecked(self.s.ghost_labels)
        self.labels.toggled.connect(self._on_labels)
        g.addRow(self.labels)
        lay.addWidget(gbox)

        self.eng_box = QGroupBox("Движки подбора (цвет = движок)")
        self.eng_lay = QVBoxLayout(self.eng_box)
        self.eng_checks: dict[str, QCheckBox] = {}
        self.heat_combo = QComboBox()
        self.heat_combo.setToolTip("Чью карту вероятностей рисовать в слое «ИИ-вероятности»")
        self.heat_combo.currentIndexChanged.connect(self._on_heat_engine)
        hrow = QHBoxLayout()
        hrow.addWidget(QLabel("Карта:"))
        hrow.addWidget(self.heat_combo, 1)
        self.eng_lay.addLayout(hrow)
        imp = QGroupBox("Перенести в дорожку")
        il = QVBoxLayout(imp)
        self.imp_engine = QComboBox()
        self.imp_engine.setToolTip("Чьи призрачные ноты скопировать в активную MIDI-дорожку как обычные ноты (потом правите руками)")
        il.addWidget(self.imp_engine)
        self.imp_loop = QCheckBox("Только участок цикла")
        self.imp_snap = QCheckBox("Привязать к сетке")
        self.imp_snap.setChecked(True)
        il.addWidget(self.imp_loop)
        il.addWidget(self.imp_snap)
        self.imp_btn = QPushButton("Скопировать ноты в дорожку")
        self.imp_btn.clicked.connect(self._on_import)
        il.addWidget(self.imp_btn)
        self.eng_lay.addWidget(imp)
        self.eng_hint = QLabel("Запустите «ИИ-анализ», чтобы появились кандидаты нот.")
        self.eng_hint.setStyleSheet("color:#8a93a5;")
        self.eng_hint.setWordWrap(True)
        self.eng_lay.addWidget(self.eng_hint)
        lay.addWidget(self.eng_box)

        help_lbl = QLabel(
            "<b>Управление</b><br>"
            "ЛКМ по пустому — новая нота, тяни — длина<br>"
            "ЛКМ по ноте — двигать, за правый край — длина<br>"
            "ПКМ — удалить (можно вести), Shift+ЛКМ — выделять<br>"
            "Ctrl+ЛКМ рамка — выделение, Ctrl+тянуть — копия<br>"
            "Колесо — вверх/вниз, Shift — влево/вправо<br>"
            "Ctrl+колесо — зум по времени, Alt+колесо — по высоте<br>"
            "Линейка: клик — позиция, тянуть — цикл, ПКМ — убрать цикл<br>"
            "Пробел — играть/стоп, Del — удалить, Ctrl+Z/Y — отмена<br>"
            "Стрелки — сдвиг выделенного, +/− — velocity, Ctrl+D — дубль"
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color:#aab; font-size:11px;")
        lay.addWidget(help_lbl)
        lay.addStretch(1)

    def set_sources(self, roles: list[str]) -> None:
        cur = self.source.currentData()
        self.source.blockSignals(True)
        self.source.clear()
        self.source.addItem("Как у дорожки", "auto")
        for r in roles:
            self.source.addItem(SOURCE_LABELS.get(r, r), r)
        idx = self.source.findData(cur)
        self.source.setCurrentIndex(max(0, idx))
        self.source.blockSignals(False)

    def set_engines(self, engine_ids: list[str]) -> None:
        """Rebuild the per-engine checkboxes from the engines present in the analysis."""
        for cb in self.eng_checks.values():
            self.eng_lay.removeWidget(cb)
            cb.hide()
            cb.deleteLater()
        self.eng_checks.clear()
        self.eng_hint.setVisible(not engine_ids)
        for eid in engine_ids:
            r, g, b = engine_color(eid)
            cb = QCheckBox(("    " if "/" in eid else "") + engine_label(eid))
            cb.setStyleSheet(f"QCheckBox{{color:rgb({r},{g},{b});}}")
            cb.setToolTip(engine_description(eid))
            # group variants start off (the combined set is on), so nothing is drawn twice
            cb.setChecked((self.s.ghost_engines is None and "/" not in eid) or (self.s.ghost_engines is not None and eid in self.s.ghost_engines))
            cb.toggled.connect(self._on_engines)
            self.eng_checks[eid] = cb
            self.eng_lay.addWidget(cb)
        self.imp_engine.clear()
        self.imp_engine.addItem("Все включённые движки", None)
        for eid in engine_ids:
            self.imp_engine.addItem(engine_label(eid), eid)
        pref = self.imp_engine.findData("muscriptor")
        if pref > 0:
            self.imp_engine.setCurrentIndex(pref)
        self.heat_combo.blockSignals(True)
        self.heat_combo.clear()
        self.heat_combo.addItem("Авто", None)
        for eid in engine_ids:
            if "/" not in eid:
                self.heat_combo.addItem(engine_label(eid), eid)
        idx = self.heat_combo.findData(self.s.heat_engine)
        self.heat_combo.setCurrentIndex(max(0, idx))
        self.heat_combo.blockSignals(False)

    def _on_import(self) -> None:
        eid = self.imp_engine.currentData()
        engines = {eid} if eid else self.s.ghost_engines
        self.import_requested.emit(engines, self.imp_loop.isChecked(), self.imp_snap.isChecked())

    def _on_engines(self, _v=None) -> None:
        on = {eid for eid, cb in self.eng_checks.items() if cb.isChecked()}
        self.s.ghost_engines = on
        self._emit()

    def _on_heat_engine(self, _i) -> None:
        self.s.heat_engine = self.heat_combo.currentData()
        self._emit()

    def _on_source(self, _i) -> None:
        self.s.source = self.source.currentData() or "auto"
        self._emit()

    def _on_thr(self, v) -> None:
        self.s.ghost_min_conf = v / 100.0
        self.thr_lbl.setText(f"{v}%")
        self._emit()

    def _on_labels(self, v) -> None:
        self.s.ghost_labels = bool(v)
        self._emit()

    def _emit(self) -> None:
        self.changed.emit()
