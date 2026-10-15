# SPDX-License-Identifier: GPL-3.0-or-later
"""The set of open videos, their on-disk indexes, and the queue that builds them.

Per video (library/<id>/):
  transcript.json  [{t, e, x}]           speech segments
  scenes.json      [{t, e, f, d}]        key frame start/end, image file, description
  frames/NNNN.jpg                        key frames
  img.npy / seg.npy / cap.npy            picture, speech and description embeddings
  index.json                             which models produced each part
"""

import hashlib
import json
import shutil
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import media, models, settings
from .events import bus
from .paths import data_dir, library_dir

STEPS = ["Transcribing speech", "Pulling key frames", "Indexing picture + speech"]


def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _write(p: Path, data):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(p)


class Library:
    def __init__(self):
        self.file = data_dir() / "library.json"
        self.videos: dict[str, dict] = {v["id"]: v for v in _read(self.file, [])}
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.running = 0
        self.cancelled: set[str] = set()
        self.cache: dict[str, dict] = {}
        for v in self.videos.values():
            if v["status"] in ("queued", "indexing"):
                v["status"] = "queued"
        for v in list(self.videos.values()):
            if v["status"] == "queued":
                self._spawn(v["id"])

    # ------------------------------------------------------------ records

    def dir(self, vid: str) -> Path:
        return library_dir() / vid

    def save(self):
        with self.lock:
            _write(self.file, list(self.videos.values()))

    def public(self, v: dict) -> dict:
        return {k: v[k] for k in ("id", "path", "name", "size", "added", "probe", "status", "step", "pct", "note",
                                  "error", "warnings", "proxy", "counts") if k in v} | {"exists": Path(v["path"]).exists()}

    def list(self) -> list[dict]:
        with self.lock:
            return [self.public(v) for v in sorted(self.videos.values(), key=lambda v: v["added"])]

    def add(self, path: str) -> dict:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{p} isn't a file")
        vid = hashlib.sha1(str(p).encode()).hexdigest()[:16]
        st = p.stat()
        with self.lock:
            v = self.videos.get(vid)
            if v and v["size"] == st.st_size and v.get("mtime") == st.st_mtime and v["status"] != "error":
                return self.public(v)
            if v:
                shutil.rmtree(self.dir(vid), ignore_errors=True)
            pr = media.probe(str(p))
            if not pr["hasVideo"] and not pr["hasAudio"]:
                raise ValueError(f"{p.name} has no video or audio")
            v = {"id": vid, "path": str(p), "name": p.name, "size": st.st_size, "mtime": st.st_mtime,
                 "added": time.time(), "probe": pr, "status": "queued", "step": 0, "pct": 0, "note": "",
                 "error": None, "warnings": [], "proxy": None, "counts": {}}
            self.videos[vid] = v
            self.dir(vid).mkdir(parents=True, exist_ok=True)
            self.save()
        self._spawn(vid)
        self._emit(vid)
        return self.public(v)

    def remove(self, vid: str):
        with self.lock:
            self.cancelled.add(vid)
            self.videos.pop(vid, None)
            self.cache.pop(vid, None)
            self.save()
        shutil.rmtree(self.dir(vid), ignore_errors=True)
        bus.publish("removed", id=vid)

    def reindex(self, vid: str):
        with self.lock:
            v = self.videos[vid]
            if v["status"] in ("queued", "indexing"):
                return
            v.update(status="queued", step=0, pct=0, error=None, note="", warnings=[])
            self.save()
        self._spawn(vid)
        self._emit(vid)

    def detail(self, vid: str) -> dict:
        d = self.dir(vid)
        with self.lock:
            v = self.public(self.videos[vid])
        return {**v, "transcript": _read(d / "transcript.json", []), "scenes": _read(d / "scenes.json", []),
                "models": _read(d / "index.json", {})}

    def _emit(self, vid: str, **extra):
        with self.lock:
            v = self.videos.get(vid)
            if v:
                bus.publish("video", throttle_key=None if extra.get("force") else "v:" + vid, video=self.public(v))

    def _set(self, vid: str, force=False, **kw):
        with self.lock:
            v = self.videos.get(vid)
            if not v:
                return
            v.update(kw)
        self._emit(vid, force=force)

    # ------------------------------------------------------------ queue

    def _spawn(self, vid: str):
        self.cancelled.discard(vid)
        threading.Thread(target=self._worker, args=(vid,), daemon=True).start()

    def _worker(self, vid: str):
        with self.cond:
            while self.running >= settings.load()["parallel"]:
                self.cond.wait(1.0)
                if vid in self.cancelled or vid not in self.videos:
                    return
            self.running += 1
        try:
            self._index(vid)
        except media.Cancelled:
            pass
        except Exception as e:
            traceback.print_exc()
            self._set(vid, force=True, status="error", error=str(e) or e.__class__.__name__, note="")
        finally:
            with self.cond:
                self.running -= 1
                self.cond.notify_all()
            with self.lock:
                self.cache.pop(vid, None)
                self.save()

    def _index(self, vid: str):
        v = self.videos[vid]
        path, pr, dur = v["path"], v["probe"], max(0.1, v["probe"]["duration"])
        d = self.dir(vid)
        s = settings.load()
        meta = _read(d / "index.json", {})
        cancel = lambda: vid in self.cancelled
        warnings = []

        def prog(step, lo, hi, note=""):
            return lambda p: self._set(vid, step=step, pct=lo + (hi - lo) * p, note=note)

        def dl(label):
            return lambda p: self._set(vid, note=f"{label} {int(p * 100)}%")

        def check():
            if cancel():
                raise media.Cancelled()

        self._set(vid, force=True, status="indexing", step=0, pct=0, error=None)

        # 1. Speech
        transcript = _read(d / "transcript.json", None)
        if transcript is None or meta.get("speech") != s["speech"] or meta.get("language") != s["language"]:
            transcript = []
            if pr["hasAudio"]:
                wav = d / "audio.wav"
                media.extract_audio(path, wav, dur, prog(0, 0, 0.08), cancel)
                check()
                models.ensure("speech", s["speech"], dl("Downloading speech model"))
                with models.lock("speech"):
                    check()
                    self._set(vid, note="")
                    sp = models.get("speech", s["speech"], s["speechEngine"])
                    transcript = sp.transcribe(str(wav), dur, s["language"], prog(0, 0.08, 1.0))
                    meta["speechEngine"] = sp.engine
                wav.unlink(missing_ok=True)
            _write(d / "transcript.json", transcript)
            meta.update(speech=s["speech"], language=s["language"])
            meta.pop("text", None)
            _write(d / "index.json", meta)
        self._set(vid, force=True, step=1, pct=0, counts={"segments": len(transcript)})
        bus.publish("transcript", id=vid)
        check()

        # 2. Key frames
        scenes = _read(d / "scenes.json", None)
        if scenes is None or not (d / "frames").exists():
            if pr["hasVideo"]:
                cuts = media.scan_scenes(path, dur, prog(1, 0, 0.6), cancel)
                ends = cuts[1:] + [dur]
                times = [round(a + min(1.0, 0.3 * (b - a)), 3) for a, b in zip(cuts, ends)]
                files = media.grab_frames(path, times, d / "frames", cb=prog(1, 0.6, 1.0), cancel=cancel, duration=dur)
                scenes = [{"t": a, "e": round(b, 3), "f": f, "d": ""} for a, b, f in zip(cuts, ends, files) if f]
            else:
                scenes = []
            _write(d / "scenes.json", scenes)
            for k in ("vision", "caption", "text"):
                meta.pop(k, None)
            _write(d / "index.json", meta)
        self._set(vid, force=True, step=2, pct=0, counts={"segments": len(transcript), "scenes": len(scenes)})
        bus.publish("scenes", id=vid)
        check()

        # 3. Picture + speech index
        frames = [str(d / "frames" / sc["f"]) for sc in scenes]
        if scenes and (meta.get("vision") != s["vision"] or not (d / "img.npy").exists()):
            try:
                models.ensure("vision", s["vision"], dl("Downloading picture model"))
                with models.lock("vision"):
                    check()
                    self._set(vid, note="")
                    emb = models.get("vision", s["vision"]).embed_images(frames, prog(2, 0, 0.25))
                np.save(d / "img.npy", emb)
                meta["vision"] = s["vision"]
            except media.Cancelled:
                raise
            except Exception as e:
                traceback.print_exc()
                warnings.append(f"Picture search unavailable: {e}")
        check()

        if scenes and s["caption"] != "none" and meta.get("caption") != s["caption"]:
            try:
                models.ensure("caption", s["caption"], dl("Downloading describer"))
                with models.lock("caption"):
                    self._set(vid, note="")
                    cap = models.get("caption", s["caption"])
                    for i, sc in enumerate(scenes):
                        check()
                        sc["d"] = cap.describe(frames[i])
                        self._set(vid, step=2, pct=0.25 + 0.65 * (i + 1) / len(scenes), note=f"Describing frame {i + 1} of {len(scenes)}")
                        if i % 4 == 3 or i == len(scenes) - 1:
                            _write(d / "scenes.json", scenes)
                            bus.publish("scenes", throttle_key="sc:" + vid, id=vid)
                meta["caption"] = s["caption"]
                meta.pop("text", None)
            except media.Cancelled:
                raise
            except Exception as e:
                traceback.print_exc()
                warnings.append(f"Frame descriptions unavailable: {e}")
            _write(d / "scenes.json", scenes)
            bus.publish("scenes", id=vid)
        check()

        if meta.get("text") != s["text"]:
            try:
                models.ensure("text", s["text"], dl("Downloading text model"))
                with models.lock("text"):
                    self._set(vid, note="")
                    te = models.get("text", s["text"])
                    np.save(d / "seg.npy", te.docs([g["x"] for g in transcript]))
                    np.save(d / "cap.npy", te.docs([sc["d"] for sc in scenes]))
                meta["text"] = s["text"]
            except media.Cancelled:
                raise
            except Exception as e:
                traceback.print_exc()
                warnings.append(f"Meaning search unavailable, using keywords: {e}")
        _write(d / "index.json", meta)
        self._set(vid, force=True, status="ready", step=3, pct=1, note="", warnings=warnings)

    # ------------------------------------------------------------ search data

    def index_data(self, vid: str) -> dict | None:
        with self.lock:
            if vid in self.cache:
                return self.cache[vid]
            v = self.videos.get(vid)
        if not v:
            return None
        d = self.dir(vid)

        def npy(name):
            try:
                return np.load(d / name)
            except (OSError, ValueError):
                return None

        data = {"video": v, "transcript": _read(d / "transcript.json", []), "scenes": _read(d / "scenes.json", []),
                "models": _read(d / "index.json", {}), "img": npy("img.npy"), "seg": npy("seg.npy"), "cap": npy("cap.npy")}
        if v["status"] == "ready":
            with self.lock:
                self.cache[vid] = data
        return data

    # ------------------------------------------------------------ proxy

    def ensure_proxy(self, vid: str):
        v = self.videos[vid]
        if v.get("proxy") and Path(v["proxy"]).exists():
            return
        out = self.dir(vid) / "proxy.mp4"

        def work():
            try:
                self._set(vid, force=True, note="Converting for playback")
                media.make_proxy(v["path"], out, v["probe"]["duration"],
                                 lambda p: bus.publish("proxy", throttle_key="px:" + vid, id=vid, pct=p))
                self._set(vid, force=True, proxy=str(out), note="")
                bus.publish("proxy", id=vid, pct=1, done=True)
                self.save()
            except Exception as e:
                bus.publish("proxy", id=vid, error=str(e))

        threading.Thread(target=work, daemon=True).start()
