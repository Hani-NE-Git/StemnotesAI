"""Self-update from GitHub: replaces only the program code, never python/, models/, soundfonts/, projects/, cache/.

Two modes:
  * git mode (preferred, works with private repos without storing any token): the app folder is a git clone
    and `git` is installed -> fetch + fast-forward pull. Authentication is handled by Git Credential Manager.
  * zip mode: download the branch archive (public repo, or private repo with github_token in settings.json).

Usage from the app (toolbar "Обновить") or standalone:  python -m stemnotes.updater [--force] [--check]
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from . import paths

DEFAULT_REPO = "Hani-NE-Git/stemnotes"
DEFAULT_BRANCH = "main"
# Top-level entries that are user data / downloaded content and must survive an update.
PROTECTED = {"python", "venv", "models", "cache", "soundfonts", "projects", "vst3", "settings.json", ".git"}
# Files that trigger a dependency re-install when they change.
REQ_FILES = ["requirements-base.txt", "requirements-ai.txt"]

Progress = Callable[[float, str], None]


def _settings() -> dict:
    try:
        return json.loads(paths.SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def repo_config() -> tuple[str, str, str | None]:
    """(owner/repo, branch, token) — overridable in settings.json: github_repo, github_branch, github_token."""
    s = _settings()
    return (s.get("github_repo") or DEFAULT_REPO, s.get("github_branch") or DEFAULT_BRANCH, s.get("github_token") or None)


def local_version() -> str:
    try:
        return (paths.HOME / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"


def _open(url: str, token: str | None):
    headers = {"User-Agent": "StemNotes-updater"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=60)


def remote_version(repo: str, branch: str, token: str | None) -> str:
    if token:
        url = f"https://api.github.com/repos/{repo}/contents/VERSION?ref={branch}"
        headers = {"User-Agent": "StemNotes-updater", "Authorization": f"Bearer {token}", "Accept": "application/vnd.github.raw"}
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
            return r.read().decode("utf-8").strip()
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/VERSION"
    with _open(url, token) as r:
        return r.read().decode("utf-8").strip()


def _vtuple(v: str) -> tuple:
    out = []
    for part in v.strip().split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def is_newer(remote: str, local: str) -> bool:
    return _vtuple(remote) > _vtuple(local)


def download_zip(repo: str, branch: str, token: str | None, progress: Progress) -> bytes:
    if token:
        url = f"https://api.github.com/repos/{repo}/zipball/{branch}"
    else:
        url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
    progress(0.05, "Скачивание кода с GitHub…")
    with _open(url, token) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf = io.BytesIO()
        got = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            buf.write(chunk)
            got += len(chunk)
            if total:
                progress(0.05 + 0.45 * got / total, f"Скачивание кода… {got // 1024} КБ")
    return buf.getvalue()


def _file_hash(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return ""


def apply_zip(data: bytes, progress: Progress) -> dict:
    """Extracts the archive over the app folder (except PROTECTED). Returns {'changed': [...], 'requirements_changed': bool}."""
    home = paths.HOME
    before = {f: _file_hash(home / f) for f in REQ_FILES}
    progress(0.55, "Распаковка…")
    changed: list[str] = []
    with tempfile.TemporaryDirectory(dir=str(paths.CACHE_DIR)) as tmp:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(tmp)
        roots = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("Неожиданная структура архива")
        src = roots[0]
        # 1) copy new/changed files
        all_new: set[str] = set()
        files = [p for p in src.rglob("*") if p.is_file()]
        for i, p in enumerate(files):
            rel = p.relative_to(src)
            top = rel.parts[0]
            if top in PROTECTED or "__pycache__" in rel.parts:
                continue
            all_new.add(rel.as_posix())
            dest = home / rel
            if dest.exists() and _file_hash(dest) == _file_hash(p):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            changed.append(rel.as_posix())
            if i % 10 == 0:
                progress(0.55 + 0.4 * i / max(1, len(files)), f"Обновление файлов… {rel}")
        # 2) remove stale files inside the code package (never outside it)
        pkg = home / "stemnotes"
        if pkg.exists():
            for p in list(pkg.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    rel = p.relative_to(home).as_posix()
                    if rel not in all_new:
                        p.unlink()
                        changed.append("− " + rel)
            for p in pkg.rglob("__pycache__"):
                shutil.rmtree(p, ignore_errors=True)
    after = {f: _file_hash(home / f) for f in REQ_FILES}
    return {"changed": changed, "requirements_changed": before != after}


def install_requirements(progress: Progress, ai: bool | None = None) -> None:
    """pip install for changed requirement files (fast no-op when everything is already satisfied)."""
    home = paths.HOME
    # compiled numba caches from the previous library set can crash the new one
    for d in paths.CACHE_DIR.glob("numba*"):
        shutil.rmtree(d, ignore_errors=True)
    py = sys.executable
    cmds = [[py, "-m", "pip", "install", "--no-warn-script-location", "-q", "-r", str(home / "requirements-base.txt")],
            [py, "-m", "pip", "install", "--no-warn-script-location", "-q", "--no-deps", "tinysoundfont"]]
    if ai is None:
        try:
            import importlib.util
            ai = importlib.util.find_spec("torch") is not None
        except Exception:
            ai = False
    if ai:
        cmds.append([py, "-m", "pip", "install", "--no-warn-script-location", "-q", "torchaudio"])
        cmds.append([py, "-m", "pip", "install", "--no-warn-script-location", "-q", "-r", str(home / "requirements-ai.txt")])
        cmds.append([py, "-m", "pip", "install", "--no-warn-script-location", "-q", "--no-deps", "basic-pitch"])
    for i, cmd in enumerate(cmds):
        progress(0.95 + 0.05 * i / len(cmds), "Обновление библиотек… (может занять минуту)")
        subprocess.run(cmd, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _git() -> str | None:
    return shutil.which("git")


def git_available() -> bool:
    return (paths.HOME / ".git").exists() and _git() is not None


def _run_git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([_git(), "-C", str(paths.HOME), *args], capture_output=True, text=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def update_via_git(force: bool, progress: Progress) -> dict:
    repo, branch, _ = repo_config()
    home = paths.HOME
    local = local_version()
    before = {f: _file_hash(home / f) for f in REQ_FILES}
    progress(0.05, f"git fetch origin {branch}…")
    # make sure origin points at the configured repo
    r = _run_git("remote", "get-url", "origin")
    if r.returncode != 0:
        _run_git("remote", "add", "origin", f"https://github.com/{repo}.git")
    r = _run_git("fetch", "origin", branch)
    if r.returncode != 0:
        raise RuntimeError("git fetch не удался:\n" + (r.stderr or r.stdout).strip())
    head = _run_git("rev-parse", "HEAD").stdout.strip()
    remote = _run_git("rev-parse", "FETCH_HEAD").stdout.strip()
    if head == remote and not force:
        progress(1.0, f"Установлена актуальная версия {local} (git {head[:7]})")
        return {"updated": False, "local": local, "remote": local, "changed": []}
    progress(0.4, "Применение изменений (git)…")
    changed = [l for l in _run_git("diff", "--name-status", "HEAD", "FETCH_HEAD").stdout.splitlines() if l.strip()]
    if force:
        r = _run_git("reset", "--hard", "FETCH_HEAD")
    else:
        r = _run_git("merge", "--ff-only", "FETCH_HEAD")
        if r.returncode != 0:
            # local edits in the way: keep them aside and take the remote version
            _run_git("stash", "push", "-u", "-m", "stemnotes-updater")
            r = _run_git("reset", "--hard", "FETCH_HEAD")
    if r.returncode != 0:
        raise RuntimeError("git не смог применить обновление:\n" + (r.stderr or r.stdout).strip())
    for p in (home / "stemnotes").rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    after = {f: _file_hash(home / f) for f in REQ_FILES}
    if before != after:
        install_requirements(progress)
    new_local = local_version()
    progress(1.0, f"Обновлено: {local} → {new_local} (git {remote[:7]}), файлов изменено: {len(changed)}")
    return {"updated": True, "local": local, "remote": new_local, "changed": changed}


def update(force: bool = False, progress: Progress = lambda f, m: None) -> dict:
    """Full update. Returns {'updated': bool, 'local': str, 'remote': str, 'changed': [...]}"""
    if git_available():
        return update_via_git(force, progress)
    repo, branch, token = repo_config()
    local = local_version()
    progress(0.0, f"Проверка версии ({repo}, ветка {branch})…")
    remote = remote_version(repo, branch, token)
    if not force and not is_newer(remote, local):
        progress(1.0, f"Установлена актуальная версия {local}")
        return {"updated": False, "local": local, "remote": remote, "changed": []}
    data = download_zip(repo, branch, token, progress)
    res = apply_zip(data, progress)
    if res["requirements_changed"]:
        install_requirements(progress)
    progress(1.0, f"Обновлено: {local} → {remote}, файлов изменено: {len(res['changed'])}")
    return {"updated": True, "local": local, "remote": remote, "changed": res["changed"]}


def main(argv: list[str]) -> int:
    force = "--force" in argv
    if "--check" in argv:
        repo, branch, token = repo_config()
        print("mode:", "git" if git_available() else "zip")
        print("local:", local_version(), "remote:", remote_version(repo, branch, token) if not git_available() else "(git)")
        return 0
    try:
        res = update(force=force, progress=lambda f, m: print(f"[{int(f * 100):3d}%] {m}"))
    except Exception as e:
        print("Ошибка обновления:", e)
        return 1
    for c in res["changed"]:
        print("  ", c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
