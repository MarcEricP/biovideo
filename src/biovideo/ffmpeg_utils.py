from __future__ import annotations
import os
import stat
import shutil
import tarfile
import zipfile
import platform
import tempfile
import urllib.request
from pathlib import Path

import matplotlib as mpl
from appdirs import user_cache_dir

try:
    import imageio_ffmpeg  # optional helper
except Exception:
    imageio_ffmpeg = None  # type: ignore

PKG_NAME = "dendro_morph"


def _is_exec(p: Path) -> bool:
    return p.exists() and os.access(str(p), os.X_OK)


def _ensure_exec(p: Path) -> None:
    try:
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def get_ffmpeg_path(force_download: bool = False) -> str:
    """Return a path to an ffmpeg binary; download into app cache if needed."""
    # 1) Matplotlib rcParams
    rc_path = mpl.rcParams.get("animation.ffmpeg_path")
    if rc_path and not force_download and _is_exec(Path(rc_path)):
        return rc_path

    # 2) Environment variable
    env_path = os.getenv("FFMPEG_BINARY")
    if env_path and not force_download and _is_exec(Path(env_path)):
        return env_path

    # 3) On PATH
    if not force_download:
        found = shutil.which("ffmpeg")
        if found:
            return found

    # 4) imageio-ffmpeg helper
    if imageio_ffmpeg is not None and not force_download:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()  # downloads into cache if needed
        except Exception:
            pass

    # 5) Download static build ourselves
    cache_dir = Path(user_cache_dir(PKG_NAME)) / "ffmpeg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    exe = _download_static_ffmpeg(cache_dir)
    _ensure_exec(exe)
    return str(exe)


def _download(url: str, dest: Path) -> None:
    tmp = Path(tempfile.mkstemp(prefix="ffmpeg_")[1])
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        shutil.move(str(tmp), str(dest))
    finally:
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def _download_static_ffmpeg(cache_dir: Path) -> Path:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows":
        url = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-6.1.1-full_build.zip"
        archive = cache_dir / "ffmpeg.zip"
        _download(url, archive)
        with zipfile.ZipFile(archive) as zf:
            exe_name = "ffmpeg.exe"
            cand = [m for m in zf.namelist() if m.endswith(exe_name)]
            if not cand:
                raise RuntimeError("ffmpeg.exe not found in archive")
            member = cand[0]
            zf.extract(member, cache_dir)
            return (cache_dir / member).resolve()

    # macOS and Linux (tar.xz variants)
    if system == "Darwin":
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n6.1.1-macos64-lgpl-shared-6.1.tar.xz"
    else:  # Linux defaults to amd64; adapt for arm64 if needed
        arch = "amd64" if ("x86_64" in machine or "amd64" in machine) else ("arm64" if ("aarch64" in machine or "arm64" in machine) else "amd64")
        url = f"https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-{arch}-static.tar.xz"

    archive = cache_dir / "ffmpeg.tar.xz"
    _download(url, archive)
    with tarfile.open(archive, mode="r:xz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith("/ffmpeg") or m.name.endswith("ffmpeg")]
        if not members:
            raise RuntimeError("ffmpeg binary not found in archive")
        member = members[0]
        tf.extract(member, cache_dir)
        return (cache_dir / member.name).resolve()