# SPDX-License-Identifier: GPL-3.0-or-later
import json
import threading

from . import catalog
from .paths import data_dir, default_save_dir

_lock = threading.Lock()
_cache: dict | None = None


def _file():
    return data_dir() / "settings.json"


def load() -> dict:
    global _cache
    with _lock:
        if _cache is None:
            base = {**catalog.defaults(), "speechEngine": "auto", "language": "auto", "parallel": 2,
                    "saveDir": str(default_save_dir())}
            try:
                base.update(json.loads(_file().read_text()))
            except (OSError, ValueError):
                pass
            rec = None
            for k in catalog.KINDS:
                try:
                    catalog.get(k, base[k])
                except KeyError:
                    rec = rec or catalog.recommend(base["language"])
                    base[k] = rec[k]
            _cache = base
        return dict(_cache)


def update(patch: dict) -> dict:
    global _cache
    cur = load()
    for k in ("speech", "vision", "caption", "text"):
        if k in patch:
            catalog.get(k, patch[k])  # validates
    cur.update({k: v for k, v in patch.items() if k in cur})
    cur["parallel"] = max(1, min(6, int(cur["parallel"])))
    with _lock:
        _cache = cur
        _file().write_text(json.dumps(cur, indent=2))
    return dict(cur)
