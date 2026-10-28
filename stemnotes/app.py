"""Application entry point."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMessageBox

from . import paths
from .audio.engine import Engine


def dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor("#23262d"))
    p.setColor(QPalette.ColorRole.WindowText, QColor("#e6e6e6"))
    p.setColor(QPalette.ColorRole.Base, QColor("#1b1d22"))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor("#23262d"))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor("#2d3340"))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor("#e6e6e6"))
    p.setColor(QPalette.ColorRole.Text, QColor("#e6e6e6"))
    p.setColor(QPalette.ColorRole.Button, QColor("#2b2f38"))
    p.setColor(QPalette.ColorRole.ButtonText, QColor("#e6e6e6"))
    p.setColor(QPalette.ColorRole.Highlight, QColor("#4f8cff"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor("#7c8494"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#777"))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#777"))
    app.setPalette(p)
    app.setStyleSheet("QToolButton:checked{background:#4f8cff;color:white;border-radius:3px;} QToolTip{color:#eee;background:#2d3340;border:1px solid #555;}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("StemNotes")
    from . import __version__
    from .log import setup as setup_log
    setup_log().info("StemNotes %s starting, home=%s", __version__, paths.HOME)
    dark_palette(app)

    from .ui.main_window import MainWindow, load_settings
    from .ui.dialogs import DownloadDialog

    sf = paths.find_soundfont()
    if sf is None:
        r = QMessageBox.question(None, "SoundFont", "Для встроенных инструментов нужен SoundFont (GeneralUser GS, 32 МБ).\n"
                                 f"Скачать его в папку {paths.SOUNDFONTS_DIR}?")
        if r == QMessageBox.StandardButton.Yes:
            dlg = DownloadDialog(paths.SOUNDFONT_URL, paths.SOUNDFONTS_DIR / paths.SOUNDFONT_NAME, "SoundFont")
            dlg.exec()
            sf = paths.find_soundfont()

    engine = Engine()
    settings = load_settings()
    engine.start(settings.get("device"), settings.get("block"))
    if sf is not None:
        engine.load_soundfont(sf)

    win = MainWindow(engine)
    win.show()
    if not engine.available:
        QMessageBox.warning(win, "Звук", "Не удалось открыть аудиоустройство (звука не будет).\n"
                            f"{engine.error}\n\nВыберите устройство в меню «Звук…».")
    for a in argv[1:]:
        from pathlib import Path
        p = Path(a)
        if p.is_dir() and (p / "project.json").exists():
            win._load_project(p)
        elif p.name == "project.json":
            win._load_project(p.parent)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
