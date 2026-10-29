"""Fetch the prebuilt native host (stemnotes-host) from the GitHub release 'host-latest' into bin/."""
from __future__ import annotations

import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from .. import paths
from ..log import get as _get_log
from .native_host import BIN_DIR, EXE_NAME, expected_host_version

log = _get_log("host-download")


def release_url(repo: str) -> str:
    name = "stemnotes-host-windows.zip" if sys.platform.startswith("win") else "stemnotes-host-linux.tar.gz"
    return f"https://github.com/{repo}/releases/download/host-latest/{name}"


def installed_host_version() -> str:
    try:
        return (BIN_DIR / "host-version.txt").read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def host_up_to_date() -> bool:
    return (BIN_DIR / EXE_NAME).exists() and installed_host_version() == expected_host_version()


def download_host(progress: Callable[[float, str], None] | None = None) -> Path:
    from ..updater import repo_config
    repo, _, _ = repo_config()
    url = release_url(repo)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.0, f"Скачивание аудиохоста: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "StemNotes"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf = io.BytesIO()
        got = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            buf.write(chunk)
            got += len(chunk)
            if progress and total:
                progress(0.9 * got / total, f"Скачивание аудиохоста… {got // 1024} КБ")
    data = buf.getvalue()
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                (BIN_DIR / Path(name).name).write_bytes(z.read(name))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            for m in t.getmembers():
                if m.isfile():
                    (BIN_DIR / Path(m.name).name).write_bytes(t.extractfile(m).read())
    exe = BIN_DIR / EXE_NAME
    if not sys.platform.startswith("win"):
        exe.chmod(0o755)
    if not (BIN_DIR / "host-version.txt").exists():
        (BIN_DIR / "host-version.txt").write_text(expected_host_version(), encoding="utf-8")
    log.info("native host installed: %s (version %s)", exe, installed_host_version())
    if progress:
        progress(1.0, "Аудиохост установлен")
    return exe
