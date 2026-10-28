"""Application log: logs/stemnotes.log in the app folder (rotating), plus console."""
from __future__ import annotations

import logging
import logging.handlers

from . import paths

LOG_DIR = paths.HOME / "logs"
LOG_FILE = LOG_DIR / "stemnotes.log"
_configured = False


def setup() -> logging.Logger:
    global _configured
    log = logging.getLogger("stemnotes")
    if _configured:
        return log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    log.addHandler(sh)
    log.propagate = False
    _configured = True
    return log


def get(name: str) -> logging.Logger:
    setup()
    return logging.getLogger("stemnotes." + name)
