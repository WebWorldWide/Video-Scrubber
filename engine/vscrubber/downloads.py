# SPDX-License-Identifier: GPL-3.0-or-later
"""Link downloads through yt-dlp (direct files, most video sites, and playlists)."""

import json
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from . import media, settings
from .events import bus
from .paths import data_dir

VIDEO_EXT = {"mp4", "webm", "mov", "m4v", "mkv", "ogv", "avi"}
QUALITIES = {"Best": None, "1080p": 1080, "720p": 720, "Audio only": 0}


def _recent_file() -> Path:
    return data_dir() / "downloads.json"


def recent() -> list[dict]:
    try:
        items = json.loads(_recent_file().read_text())
    except (OSError, ValueError):
        return []
    return [d for d in items if Path(d["path"]).exists()][:8]


def _remember(item: dict):
    items = [item] + [d for d in recent() if d["path"] != item["path"]]
    _recent_file().write_text(json.dumps(items[:8]))


def _estimate(formats: list[dict], dur: float, height: int | None) -> int:
    def size(f):
        return f.get("filesize") or f.get("filesize_approx") or ((f.get("tbr") or 0) * 125 * dur)

    audio = [f for f in formats if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")]
    best_a = max((size(f) for f in audio), default=0)
    if height == 0:
        return int(best_a)
    video = [f for f in formats if f.get("vcodec") not in (None, "none") and f.get("height")
             and (height is None or f["height"] <= height)]
    if not video:
        return 0
    top = max(video, key=lambda f: (f["height"], f.get("tbr") or 0))
    return int(size(top) + (best_a if top.get("acodec") in (None, "none") else 0))


def info(url: str) -> dict:
    import yt_dlp

    if not urlparse(url).scheme:
        url = "https://" + url
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": "in_playlist",
            "noplaylist": False}
    with yt_dlp.YoutubeDL(opts) as y:
        i = y.extract_info(url, download=False)
    host = urlparse(url).hostname or ""
    host = host[4:] if host.startswith("www.") else host
    if i.get("_type") == "playlist":
        entries = [e for e in (i.get("entries") or []) if e]
        return {"url": url, "host": host, "title": i.get("title") or "Playlist", "playlist": True, "count": len(entries),
                "duration": sum((e.get("duration") or 0) for e in entries) or None, "direct": False,
                "thumbnail": (i.get("thumbnails") or [{}])[-1].get("url"), "sizes": {}}
    dur = float(i.get("duration") or 0)
    direct = i.get("extractor") in ("generic",) and (i.get("ext") in VIDEO_EXT) and not i.get("formats", [None])[1:]
    fmts = i.get("formats") or []
    return {"url": url, "host": host, "title": i.get("title") or host, "playlist": False, "count": 1,
            "duration": dur or None, "direct": bool(direct), "thumbnail": i.get("thumbnail"),
            "ext": i.get("ext"), "sizes": {q: _estimate(fmts, dur, h) for q, h in QUALITIES.items()} if dur else {}}


_jobs: dict[str, dict] = {}


def _format(quality: str, fmt: str) -> dict:
    h = QUALITIES.get(quality)
    if h == 0:
        codec = "mp3" if fmt == "MP3" else "m4a"
        return {"format": "ba/b", "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": codec}]}
    cap = f"[height<={h}]" if h else ""
    # Skip HEVC so the result plays everywhere; VP9/AV1/H.264 all decode in the player.
    if fmt == "WebM":
        return {"format": f"bv*{cap}[ext=webm]+ba[ext=webm]/bv*{cap}+ba/b{cap}/b", "merge_output_format": "webm"}
    return {"format": f"bv*{cap}[vcodec!*=hev][vcodec!*=hvc]+ba/bv*{cap}+ba/b{cap}/b", "merge_output_format": "mp4"}


def start(url: str, quality: str, fmt: str, title: str, host: str, playlist: bool) -> str:
    jid = uuid.uuid4().hex[:10]
    _jobs[jid] = {"cancel": False}
    threading.Thread(target=_run, args=(jid, url, quality, fmt, title, host, playlist), daemon=True).start()
    return jid


def cancel(jid: str):
    if jid in _jobs:
        _jobs[jid]["cancel"] = True


def _run(jid, url, quality, fmt, title, host, playlist):
    import yt_dlp

    save = Path(settings.load()["saveDir"])
    save.mkdir(parents=True, exist_ok=True)
    seen: dict[str, tuple[float, float]] = {}
    t0 = time.monotonic()
    state = {"item": 1, "count": 1}

    def hook(d):
        if _jobs[jid]["cancel"]:
            raise yt_dlp.utils.DownloadCancelled()
        info = d.get("info_dict") or {}
        state["item"] = info.get("playlist_index") or state["item"]
        state["count"] = info.get("n_entries") or state["count"]
        if d["status"] in ("downloading", "finished"):
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            got = d.get("downloaded_bytes") or (total if d["status"] == "finished" else 0)
            seen[d.get("filename") or "?"] = (got, total)
            g = sum(a for a, _ in seen.values())
            t = sum(b for _, b in seen.values())
            bus.publish("download", throttle_key="dl:" + jid, id=jid, state="downloading", got=g, total=t,
                        speed=d.get("speed") or g / max(0.1, time.monotonic() - t0),
                        item=state["item"], count=state["count"])

    opts = {"outtmpl": str(save / "%(title).150B [%(id)s].%(ext)s"), "noplaylist": not playlist, "quiet": True,
            "no_warnings": True, "progress_hooks": [hook], "ffmpeg_location": media.ffmpeg(),
            "windowsfilenames": True, "noprogress": True, **_format(quality, fmt)}
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            i = y.extract_info(url, download=True)
        entries = [e for e in (i.get("entries") or [i]) if e] if i.get("_type") == "playlist" else [i]
        files = []
        for e in entries:
            for r in e.get("requested_downloads") or []:
                p = r.get("filepath") or r.get("_filename")
                if p and Path(p).exists():
                    files.append(p)
        if not files:
            raise RuntimeError("The download finished but no file was saved")
        items = []
        for p in files:
            item = {"path": p, "name": Path(p).name, "size": Path(p).stat().st_size, "host": host, "at": time.time()}
            _remember(item)
            items.append(item)
        bus.publish("download", id=jid, state="done", items=items)
    except yt_dlp.utils.DownloadCancelled:
        bus.publish("download", id=jid, state="cancelled")
    except Exception as e:
        msg = str(e).replace("ERROR: ", "")
        bus.publish("download", id=jid, state="error", error=msg[:400])
    finally:
        _jobs.pop(jid, None)
