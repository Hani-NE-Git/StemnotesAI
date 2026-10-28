"""Dialogs: instrument picker (GM + VST3), analysis runner, audio settings, soundfont download."""
from __future__ import annotations

import os
import subprocess
import threading
import urllib.request
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)

from .. import paths
from ..gm import GM_PROGRAMS, QUICK_PICKS
from ..project import Project, Track


def default_vst3_dirs() -> list[Path]:
    if paths.is_windows():
        dirs = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Common Files" / "VST3",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Common" / "VST3"]
    else:
        dirs = [Path.home() / ".vst3", Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3")]
    dirs.append(paths.HOME / "vst3")
    return [d for d in dirs if d and d.exists()]


def scan_vst3() -> list[Path]:
    found: list[Path] = []
    for d in default_vst3_dirs():
        for p in sorted(d.rglob("*.vst3")):
            if p.is_dir() or p.is_file():
                found.append(p)
                if p.is_dir():
                    pass
    # .vst3 bundles are directories; rglob would also descend into them -> keep only top-level bundles
    result = []
    for p in found:
        if not any(parent in found for parent in p.parents):
            result.append(p)
    return result


class InstrumentDialog(QDialog):
    def __init__(self, track: Track, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Инструмент: {track.name}")
        self.resize(520, 520)
        self.result_instrument: dict | None = None
        tabs = QTabWidget()

        # ---- SoundFont tab
        sf = QWidget()
        v = QVBoxLayout(sf)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск (например: bass, guitar, violin)…")
        self.search.textChanged.connect(self._filter)
        v.addWidget(self.search)
        self.list = QListWidget()
        for label, prog, drums in QUICK_PICKS:
            it = QListWidgetItem(f"★ {label}")
            it.setData(Qt.ItemDataRole.UserRole, {"type": "sf2", "bank": 0, "program": prog, "drums": drums})
            self.list.addItem(it)
        for i, name in enumerate(GM_PROGRAMS):
            it = QListWidgetItem(f"{i + 1:3d}  {name}")
            it.setData(Qt.ItemDataRole.UserRole, {"type": "sf2", "bank": 0, "program": i, "drums": False})
            self.list.addItem(it)
        self.list.itemDoubleClicked.connect(lambda _: self._accept_sf2())
        v.addWidget(self.list)
        tabs.addTab(sf, "SoundFont (встроенные)")

        # ---- VST3 tab
        vst = QWidget()
        v2 = QVBoxLayout(vst)
        v2.addWidget(QLabel("Найденные VST3 (стандартные папки и папка vst3 рядом с программой):"))
        self.vst_list = QListWidget()
        for p in scan_vst3():
            it = QListWidgetItem(p.name)
            it.setData(Qt.ItemDataRole.UserRole, str(p))
            it.setToolTip(str(p))
            self.vst_list.addItem(it)
        self.vst_list.itemDoubleClicked.connect(lambda _: self._accept_vst())
        v2.addWidget(self.vst_list)
        row = QHBoxLayout()
        browse = QPushButton("Выбрать файл .vst3…")
        browse.clicked.connect(self._browse_vst)
        row.addWidget(browse)
        row.addStretch(1)
        v2.addLayout(row)
        tabs.addTab(vst, "VST3 плагин")

        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(lambda: self._accept_vst() if tabs.currentIndex() == 1 else self._accept_sf2())
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.tabs = tabs
        if track.instrument.get("type") == "vst3":
            tabs.setCurrentIndex(1)

    def _filter(self, text: str) -> None:
        t = text.lower().strip()
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(bool(t) and t not in it.text().lower())

    def _accept_sf2(self) -> None:
        it = self.list.currentItem()
        if it is None:
            return
        self.result_instrument = dict(it.data(Qt.ItemDataRole.UserRole))
        self.accept()

    def _accept_vst(self) -> None:
        it = self.vst_list.currentItem()
        if it is None:
            return
        self.result_instrument = {"type": "vst3", "path": it.data(Qt.ItemDataRole.UserRole), "plugin_name": None, "state": None}
        self.accept()

    def _browse_vst(self) -> None:
        if paths.is_windows():
            p, _ = QFileDialog.getOpenFileName(self, "VST3 плагин", str(default_vst3_dirs()[0]) if default_vst3_dirs() else "", "VST3 (*.vst3)")
        else:
            p = QFileDialog.getExistingDirectory(self, "VST3 плагин (папка .vst3)", str(Path.home()))
        if p:
            self.result_instrument = {"type": "vst3", "path": p, "plugin_name": None, "state": None}
            self.accept()


class AnalysisDialog(QDialog):
    """Runs the analysis pipeline in a background thread with progress."""
    finished_ok = Signal()

    def __init__(self, project: Project, parent=None, first_time: bool = True, settings: dict | None = None):
        super().__init__(parent)
        from ..analysis.pipeline import AnalysisOptions
        self.project = project
        self.settings = settings if settings is not None else {}
        self.job = None
        self.setWindowTitle("ИИ-анализ песни")
        self.resize(640, 560)
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.cb_tempo = QCheckBox("Определить темп (BPM) и сетку")
        self.cb_tempo.setChecked(True)
        self.cb_sep = QCheckBox("Разделить на дорожки (Demucs, 6 стемов)")
        self.cb_sep.setChecked(first_time)
        self.cb_spec = QCheckBox("Спектрограммы по нотам")
        self.cb_spec.setChecked(True)
        self.cb_notes = QCheckBox("Призрачные ноты (Basic Pitch + детектор ударных)")
        self.cb_notes.setChecked(True)
        for cb in (self.cb_tempo, self.cb_sep, self.cb_spec, self.cb_notes):
            f.addRow(cb)
        from ..analysis.engines import ENGINES
        self.engine_checks: dict[str, QCheckBox] = {}
        saved = (settings or {}).get("engines")
        for e in ENGINES:
            cb = QCheckBox(f"    {e.name} — {', '.join(e.roles)}")
            cb.setToolTip(e.description)
            if not e.available():
                cb.setEnabled(False)
                cb.setText(cb.text() + "  (не установлен)")
                cb.setChecked(False)
            else:
                cb.setChecked(e.default_on if saved is None else e.id in saved)
            self.engine_checks[e.id] = cb
            f.addRow(cb)
        self.musc_model = QComboBox()
        for sz, lbl in (("small", "small — 100M, можно на CPU"), ("medium", "medium — 300M, по умолчанию"), ("large", "large — 1.4B, лучшее качество, нужна GPU")):
            self.musc_model.addItem(lbl, sz)
        self.musc_model.setCurrentIndex(max(0, self.musc_model.findData(self.settings.get("muscriptor_model", "medium"))))
        f.addRow("MuScriptor модель:", self.musc_model)
        self.hf_token = QLineEdit(self.settings.get("hf_token", ""))
        self.hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token.setPlaceholderText("hf_… (read-токен; сначала принять лицензию на huggingface.co/MuScriptor)")
        self.hf_token.setToolTip("Нужен только для MuScriptor: модель закрытая (gated), скачивается с Hugging Face после принятия лицензии.")
        f.addRow("HF token:", self.hf_token)
        self.model = QComboBox()
        self.model.addItem("htdemucs_6s — ударные, бас, гитара, пианино, вокал, остальное", "htdemucs_6s")
        self.model.addItem("htdemucs_ft — 4 стема, лучшее качество (в 4 раза дольше)", "htdemucs_ft")
        self.model.addItem("htdemucs — 4 стема, быстро", "htdemucs")
        f.addRow("Модель:", self.model)
        lay.addLayout(f)
        self.gpu = QLabel("Устройство: проверка…")
        self.gpu.setStyleSheet("color:#8a93a5;")
        lay.addWidget(self.gpu)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.msg = QLabel("")
        lay.addWidget(self.bar)
        lay.addWidget(self.msg)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Запустить")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self._cancel)
        row.addStretch(1)
        row.addWidget(self.start_btn)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)
        self._opts_cls = AnalysisOptions
        self._progress = (0.0, "")
        self._done: Exception | None | str = "pending"
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        threading.Thread(target=self._probe_gpu, daemon=True).start()

    def _probe_gpu(self) -> None:
        try:
            from ..analysis.separate import gpu_info
            info = gpu_info()
        except Exception as e:
            info = f"ИИ-модуль не установлен: {e}"
        self._gpu_text = info

    def _poll(self) -> None:
        if hasattr(self, "_gpu_text"):
            self.gpu.setText("Устройство: " + self._gpu_text)
        frac, msg = self._progress
        self.bar.setValue(int(frac * 1000))
        self.msg.setText(msg)
        if self._done != "pending":
            self.timer.stop()
            err = self._done
            if err is None:
                self.finished_ok.emit()
                self.accept()
            else:
                QMessageBox.warning(self, "Анализ", f"Ошибка анализа:\n{err}")
                self.start_btn.setEnabled(True)
                self._done = "pending"

    def _start(self) -> None:
        from ..analysis.pipeline import AnalysisJob
        engines = [eid for eid, cb in self.engine_checks.items() if cb.isChecked()]
        self.settings["engines"] = engines
        self.settings["muscriptor_model"] = self.musc_model.currentData()
        self.settings["hf_token"] = self.hf_token.text().strip()
        try:
            from ..ui.main_window import save_settings
            save_settings(self.settings)
        except Exception:
            pass
        opts = self._opts_cls(tempo=self.cb_tempo.isChecked(), separate=self.cb_sep.isChecked(),
                              spectrograms=self.cb_spec.isChecked(), transcribe=self.cb_notes.isChecked(),
                              model=self.model.currentData(), engines=engines)
        self.start_btn.setEnabled(False)

        def progress(frac, msg):
            self._progress = (frac, msg)

        def done(err):
            self._done = err
        self.job = AnalysisJob(self.project, opts, progress, done)
        self.job.start()
        self.timer.start(100)

    def _cancel(self) -> None:
        if self.job is not None and self.job.thread.is_alive():
            self.job.cancel.set()
            self.msg.setText("Отмена после текущего шага…")
            return
        self.reject()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if not self.timer.isActive():
            self.timer.start(200)


class SettingsDialog(QDialog):
    def __init__(self, engine, settings: dict, parent=None):
        super().__init__(parent)
        from ..audio.engine import list_output_devices
        self.setWindowTitle("Настройки звука")
        self.resize(480, 160)
        self.settings = settings
        lay = QFormLayout(self)
        self.dev = QComboBox()
        self.dev.addItem("Авто (WASAPI shared / системное по умолчанию)", None)
        for d in list_output_devices():
            self.dev.addItem(f"[{d['hostapi']}] {d['name']}", d["index"])
        idx = self.dev.findData(settings.get("device"))
        self.dev.setCurrentIndex(max(0, idx))
        lay.addRow("Устройство вывода:", self.dev)
        self.buf = QComboBox()
        for b in (256, 512, 1024, 2048, 4096):
            ms = 1000.0 * b / max(1, engine.sample_rate)
            self.buf.addItem(f"{b} сэмплов (~{ms:.0f} мс)", b)
        self.buf.setCurrentIndex(max(0, self.buf.findData(settings.get("block", engine.block))))
        self.buf.setToolTip("Больше буфер — меньше хрипов и щелчков, но больше задержка при прослушивании нот")
        lay.addRow("Буфер:", self.buf)
        status = QLabel(("Работает, %d Гц" % engine.sample_rate) if engine.available else f"Нет звука: {engine.error}")
        status.setWordWrap(True)
        lay.addRow("Состояние:", status)
        lay.addRow(QLabel("Используется общий режим (без ASIO и без захвата устройства)."))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def device(self):
        return self.dev.currentData()

    def block(self) -> int:
        return int(self.buf.currentData())


class DownloadDialog(QDialog):
    """Downloads a file with progress (used for the SoundFont on first start)."""

    def __init__(self, url: str, dest: Path, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 120)
        self.url, self.dest = url, dest
        lay = QVBoxLayout(self)
        self.lbl = QLabel(f"Скачивание {dest.name}…")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.lbl)
        lay.addWidget(self.bar)
        self._frac = 0.0
        self._err: str | None = None
        self._ok = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(100)
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        tmp = self.dest.with_suffix(self.dest.suffix + ".part")
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": "StemNotes"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        self._frac = got / total
            tmp.replace(self.dest)
            self._ok = True
        except Exception as e:
            self._err = str(e)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _poll(self) -> None:
        self.bar.setValue(int(self._frac * 1000))
        if self._ok:
            self.timer.stop()
            self.accept()
        elif self._err:
            self.timer.stop()
            self.lbl.setText("Ошибка: " + self._err)
            QMessageBox.warning(self, "Скачивание", f"Не удалось скачать:\n{self._err}\n\nПоложите любой .sf2 в папку soundfonts.")
            self.reject()


class UpdateDialog(QDialog):
    """Checks GitHub for a newer version and applies the update without touching models/Python/projects."""

    def __init__(self, parent=None, force: bool = False):
        super().__init__(parent)
        from .. import updater
        self.updater = updater
        repo, branch, _ = updater.repo_config()
        self.setWindowTitle("Обновление StemNotes")
        self.resize(520, 260)
        lay = QVBoxLayout(self)
        mode = "git (приватный репозиторий, без токена)" if updater.git_available() else "zip-архив"
        self.info = QLabel(f"Источник: github.com/{repo} (ветка {branch}), способ: {mode}\nТекущая версия: {updater.local_version()}\n"
                           "Обновляется только код программы. Python, модели, SoundFont и проекты остаются.")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        lay.addWidget(self.bar)
        lay.addWidget(self.msg)
        row = QHBoxLayout()
        self.check_btn = QPushButton("Проверить и обновить")
        self.check_btn.clicked.connect(lambda: self._start(False))
        self.force_btn = QPushButton("Переустановить код")
        self.force_btn.setToolTip("Скачать и записать текущую версию из GitHub, даже если номер версии совпадает")
        self.force_btn.clicked.connect(lambda: self._start(True))
        self.restart_btn = QPushButton("Перезапустить программу")
        self.restart_btn.setEnabled(False)
        self.restart_btn.clicked.connect(self._restart)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.reject)
        row.addWidget(self.check_btn)
        row.addWidget(self.force_btn)
        row.addStretch(1)
        row.addWidget(self.restart_btn)
        row.addWidget(close_btn)
        lay.addLayout(row)
        self._progress = (0.0, "")
        self._result: dict | None = None
        self._error: str | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        if force:
            self._start(True)

    def _start(self, force: bool) -> None:
        self.check_btn.setEnabled(False)
        self.force_btn.setEnabled(False)
        self._result, self._error = None, None

        def work():
            try:
                self._result = self.updater.update(force=force, progress=lambda f, m: setattr(self, "_progress", (f, m)))
            except Exception as e:
                self._error = str(e)
        threading.Thread(target=work, daemon=True).start()
        self.timer.start(100)

    def _poll(self) -> None:
        f, m = self._progress
        self.bar.setValue(int(f * 1000))
        self.msg.setText(m)
        if self._error is not None:
            self.timer.stop()
            self.msg.setText("Ошибка: " + self._error + "\n\nДля приватного репозитория нужен git в системе (обновление через git) "
                             "или github_token в settings.json (обновление через zip).")
            self.check_btn.setEnabled(True)
            self.force_btn.setEnabled(True)
        elif self._result is not None:
            self.timer.stop()
            self.check_btn.setEnabled(True)
            self.force_btn.setEnabled(True)
            if self._result["updated"]:
                self.restart_btn.setEnabled(True)
                self.restart_btn.setStyleSheet("background:#4f8cff;color:white;")
                self.msg.setText(self._progress[1] + "\nНажмите «Перезапустить программу», чтобы применить.")

    def _restart(self) -> None:
        import os
        import sys
        win = self.parent()
        if win is not None and hasattr(win, "project") and win.project is not None and win.project.dirty:
            if QMessageBox.question(self, "Перезапуск", "Сохранить проект перед перезапуском?") == QMessageBox.StandardButton.Yes:
                win.save_project()
        try:
            if win is not None and hasattr(win, "engine"):
                win.engine.stop()
                win.engine.stop_stream()
        except Exception:
            pass
        args = [sys.executable, "-m", "stemnotes"]
        if paths.is_windows():
            subprocess.Popen(args, cwd=str(paths.HOME), close_fds=True)
            os._exit(0)
        os.execv(sys.executable, args)
