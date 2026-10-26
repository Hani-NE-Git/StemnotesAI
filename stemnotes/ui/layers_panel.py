"""Background layer controls for the piano roll."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .piano_roll import Layer, LayerSettings
from .track_panel import SOURCE_LABELS


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

    def __init__(self, settings: LayerSettings, parent=None):
        super().__init__(parent)
        self.s = settings
        lay = QVBoxLayout(self)
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
