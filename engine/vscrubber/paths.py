import os
import sys
from pathlib import Path

APP = "Video Scrubber"


def data_dir() -> Path:
    env = os.environ.get("VS_DATA_DIR")
    if env:
        p = Path(env)
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / APP
    elif os.name == "nt":
        p = Path(os.environ.get("APPDATA", Path.home())) / APP
    else:
        p = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "video-scrubber"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_save_dir() -> Path:
    env = os.environ.get("VS_SAVE_DIR")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Movies" / APP
    return Path.home() / "Videos" / APP


def library_dir() -> Path:
    p = data_dir() / "library"
    p.mkdir(parents=True, exist_ok=True)
    return p
