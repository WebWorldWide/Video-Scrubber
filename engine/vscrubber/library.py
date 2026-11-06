# SPDX-License-Identifier: GPL-3.0-or-later
"""The set of open videos, their on-disk indexes, and the queue that builds them.

Per video (library/<id>/):
  transcript.json  [{t, e, x}]           speech segments
  scenes.json      [{t, e, f, d}]        key frame start/end, image file, description
  frames/NNNN.jpg                        key frames
  words.json       [[[t, word], ...]]    word start times, one list per speech segment
  dense/dNNNNN.jpg, dense.json [{t, f}]  picture-search samples every ~2 s (repeats dropped)
  img.npy / dense.npy                    picture embeddings: key frames, samples
  seg.npy / win.npy / cap.npy            text embeddings: lines, lines with neighbours, descriptions
  index.json                             which models produced each part
"""

from __future__ import annotations

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


def window_spans(transcript: list[dict], gap: float = 4.0) -> list[tuple[int, int]]:
    """(first, last) line index of each line's window: the line and its neighbours, unless a pause separates them."""
    n = len(transcript)
    return [(i - 1 if i > 0 and g["t"] - transcript[i - 1]["e"] < gap else i,
             i + 1 if i + 1 < n and transcript[i + 1]["t"] - g["e"] < gap else i) for i, g in enumerate(transcript)]


def windows(transcript: list[dict]) -> list[str]:
    """Each line with its neighbours, so meaning search sees context a short line like
    "Yeah, that one." lacks, and phrases split across two lines still match."""
    return [" ".join(g["x"] for g in transcript[a:b + 1]) for a, b in window_spans(transcript)]


SAME_EMB, SAME_PIX = 0.985, 0.008  # near-identical only: talking heads differ by gestures people search for


def _alike(d: Path, frames: list[str]):
    """alike(i, j): key frames i and j show the same thing. Needs both a near-identical picture
    embedding and a near-identical greyscale thumbnail, so slides that differ only in text stay apart."""
    try:
        emb = np.load(d / "img.npy")
    except (OSError, ValueError):
        emb = None
    if emb is None or len(emb) != len(frames):
        return lambda i, j: False
    from PIL import Image

    tiny: dict[int, np.ndarray] = {}

    def thumb(i):
        if i not in tiny:
            with Image.open(frames[i]) as im:
                tiny[i] = np.asarray(im.convert("L").resize((96, 54), Image.BILINEAR), np.float32) / 255.0
        return tiny[i]

    return lambda i, j: float(emb[i] @ emb[j]) >= SAME_EMB and float(np.abs(thumb(i) - thumb(j)).mean()) < SAME_PIX


class _Job(threading.Thread):
    """Run fn in the background; result() joins and re-raises its exception."""

    def __init__(self, fn):
        super().__init__(daemon=True)
        self.fn, self.res, self.err = fn, None, None
        self.start()

    def run(self):
        try:
            self.res = self.fn()
        except BaseException as e:
            self.err = e

    def result(self):
        self.join()
        if self.err:
            raise self.err
        return self.res


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

    def _keyframes(self, path: str, dur: float, d: Path, scenes: list[dict] | None, cb, cancel):
        """Scene cuts, one key frame per scene, and dense picture-search samples. Given existing
        `scenes` (an index from before dense sampling), only the samples are added."""
        regen = scenes is None
        cuts, dense = media.scan_video(path, dur, (lambda p: cb(0.6 * p)) if regen else cb, cancel, d / "dense")
        if regen:
            ends = cuts[1:] + [dur]
            times = [round(a + min(1.0, 0.3 * (b - a)), 3) for a, b in zip(cuts, ends)]
            files = media.grab_frames(path, times, d / "frames", cb=lambda p: cb(0.6 + 0.4 * p), cancel=cancel, duration=dur)
            scenes = [{"t": a, "e": round(b, 3), "f": f, "d": ""} for a, b, f in zip(cuts, ends, files) if f]
        return scenes, dense, regen

    def _index(self, vid: str):
        v = self.videos[vid]
        path, pr, dur = v["path"], v["probe"], max(0.1, v["probe"]["duration"])
        d = self.dir(vid)
        s = settings.load()
        meta = _read(d / "index.json", {})
        stop = [False]
        cancel = lambda: vid in self.cancelled or stop[0]
        warnings = []
        shown = [0]  # the step the progress card shows
        frames_pct = [0.0]  # key frames run alongside speech, so remember where they are

        def check():
            if cancel():
                raise media.Cancelled()

        def prog(step, lo, hi, note="", stoppable=False):
            def f(p):
                if stoppable:
                    check()
                if shown[0] == step:
                    self._set(vid, step=step, pct=lo + (hi - lo) * p, note=note)
            return f

        def frames_prog(p):
            frames_pct[0] = p
            if shown[0] == 1:
                self._set(vid, step=1, pct=p)

        def dl(label):
            return lambda p: self._set(vid, note=f"{label} {int(p * 100)}%")

        self._set(vid, force=True, status="indexing", step=0, pct=0, error=None)

        # Key frames decode on the media engine and CPU while speech runs on the GPU, so start them first.
        scenes = _read(d / "scenes.json", None)
        dense = _read(d / "dense.json", None)
        have_frames = scenes is not None and (d / "frames").exists()
        job = None
        if pr["hasVideo"] and (not have_frames or dense is None):
            job = _Job(lambda: self._keyframes(path, dur, d, scenes if have_frames else None, frames_prog, cancel))
        elif not pr["hasVideo"]:
            scenes, dense = [], []

        try:
            # 1. Speech
            transcript = _read(d / "transcript.json", None)
            if transcript is None or meta.get("speech") != s["speech"] or meta.get("language") != s["language"]:
                transcript, words = [], []
                if pr["hasAudio"]:
                    wav = d / "audio.wav"
                    media.extract_audio(path, wav, dur, prog(0, 0, 0.08), cancel)
                    check()
                    models.ensure("speech", s["speech"], dl("Downloading speech model"))
                    with models.lock("speech"):
                        check()
                        self._set(vid, note="")
                        sp = models.get("speech", s["speech"], s["speechEngine"])
                        transcript = sp.transcribe(str(wav), dur, s["language"], prog(0, 0.08, 1.0, stoppable=True))
                        meta["speechEngine"] = sp.engine
                    wav.unlink(missing_ok=True)
                    words = [g.pop("w", []) for g in transcript]
                _write(d / "transcript.json", transcript)
                _write(d / "words.json", words)
                meta.update(speech=s["speech"], language=s["language"])
                meta.pop("text", None)
                _write(d / "index.json", meta)
            shown[0] = 1
            self._set(vid, force=True, step=1, pct=frames_pct[0], counts={"segments": len(transcript)})
            bus.publish("transcript", id=vid)
            check()

            # 2. Key frames
            if job:
                scenes, dense, regen = job.result()
                _write(d / "scenes.json", scenes)
                _write(d / "dense.json", dense)
                for k in ("vision", "caption", "text") if regen else ("vision",):
                    meta.pop(k, None)
                _write(d / "index.json", meta)
        except BaseException:
            stop[0] = True  # don't leave ffmpeg running for a job that's over
            raise
        self._set(vid, force=True, step=2, pct=0, counts={"segments": len(transcript), "scenes": len(scenes)})
        shown[0] = 2
        bus.publish("scenes", id=vid)
        check()

        # 3. Picture + speech index
        frames = [str(d / "frames" / sc["f"]) for sc in scenes]
        samples = [str(d / "dense" / x["f"]) for x in dense or []]
        if scenes and (meta.get("vision") != s["vision"] or not (d / "img.npy").exists()
                       or not (d / "dense.npy").exists()):
            try:
                models.ensure("vision", s["vision"], dl("Downloading picture model"))
                with models.lock("vision"):
                    check()
                    self._set(vid, note="")
                    vm = models.get("vision", s["vision"])
                emb = vm.embed_images(frames + samples, prog(2, 0, 0.25, stoppable=True), lock=models.lock("vision"))
                np.save(d / "img.npy", emb[:len(frames)])
                np.save(d / "dense.npy", emb[len(frames):])
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
                alike, last = _alike(d, frames), None
                with models.lock("caption"):
                    self._set(vid, note="")
                    cap = models.get("caption", s["caption"])
                    for i, sc in enumerate(scenes):
                        check()
                        # A key frame that looks just like the last described one gets the same description.
                        if last is not None and scenes[last]["d"] and alike(i, last):
                            sc["d"] = scenes[last]["d"]
                        else:
                            sc["d"], last = cap.describe(frames[i]), i
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

        if meta.get("text") != s["text"] or not (d / "win.npy").exists():
            try:
                models.ensure("text", s["text"], dl("Downloading text model"))
                lock = models.lock("text")
                with lock:
                    self._set(vid, note="")
                    te = models.get("text", s["text"])
                np.save(d / "seg.npy", te.docs([g["x"] for g in transcript], lock))
                np.save(d / "win.npy", te.docs(windows(transcript), lock))
                np.save(d / "cap.npy", te.docs([sc["d"] for sc in scenes], lock))
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
                "words": _read(d / "words.json", []), "dense": _read(d / "dense.json", []),
                "models": _read(d / "index.json", {}), "img": npy("img.npy"), "seg": npy("seg.npy"), "cap": npy("cap.npy"),
                "win": npy("win.npy"), "dimg": npy("dense.npy")}
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
                                 lambda p: bus.publish("proxy", throttle_key="px:" + vid, id=vid, pct=p), probe=v["probe"])
                self._set(vid, force=True, proxy=str(out), note="")
                bus.publish("proxy", id=vid, pct=1, done=True)
                self.save()
            except Exception as e:
                bus.publish("proxy", id=vid, error=str(e))

        threading.Thread(target=work, daemon=True).start()
