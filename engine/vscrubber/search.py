# SPDX-License-Identifier: GPL-3.0-or-later
"""Describe-a-moment search across one or many indexed videos.

Each speech segment and key frame gets a 0–1 relevance from up to three signals:
  - meaning:  sentence-embedding similarity (transcript lines, frame descriptions)
  - keywords: stemmed term overlap, so exact names and numbers still win
  - picture:  SigLIP/CLIP similarity between the query and the key frame itself
Hits close together in one video merge into a single moment ("Picture + speech").
"""

import re

import numpy as np

from . import models

STOP = set(("a an the of to in on at and or is are was were be been it its this that these when where what which who "
            "whom does do did part moment someone somebody they them their there show shows showing shown find me i we "
            "you he she his her about mention mentions mentioned say says said talk talks talking scene video clip "
            "with for from up just like some any thing time point goes go get gets has have had into by how why while "
            "during").split())

MIN_SCORE = 0.38


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower().replace("’", "").replace("'", ""))


def stem(w: str) -> str:
    return w if len(w) <= 4 else w[: max(4, len(w) - 3)]


def keyword_scores(q: str, texts: list[str]) -> np.ndarray:
    words = [w for w in toks(q) if w not in STOP]
    if not words or not texts:
        return np.zeros(len(texts), np.float32)
    stems = list(dict.fromkeys(stem(w) for w in words))
    out = np.zeros(len(texts), np.float32)
    for i, t in enumerate(texts):
        tk = toks(t)
        hits = sum(1 for st in stems if any(x.startswith(st) for x in tk))
        out[i] = hits / len(stems)
    # Single-word partial matches are weak evidence on their own.
    return out * (0.85 if len(stems) > 1 else 0.7)


class QueryCache:
    """Embeds the query once per model, shared across videos."""

    def __init__(self, q: str):
        self.q = q
        self.text: dict[str, tuple[np.ndarray, object]] = {}
        self.vision: dict[str, tuple[np.ndarray, object]] = {}

    def text_emb(self, model_id: str):
        if model_id not in self.text:
            with models.lock("text"):
                m = models.get("text", model_id)
                self.text[model_id] = (m.query(self.q), m)
        return self.text[model_id]

    def vision_emb(self, model_id: str):
        if model_id not in self.vision:
            with models.lock("vision"):
                m = models.get("vision", model_id)
                self.vision[model_id] = (m.embed_text(self.q), m)
        return self.vision[model_id]


def search_video(data: dict, qc: QueryCache) -> list[dict]:
    q = qc.q
    tr, sc, meta = data["transcript"], data["scenes"], data["models"]
    dur = max(1.0, data["video"]["probe"]["duration"])
    cands = []

    # Speech
    if tr:
        kw = keyword_scores(q, [g["x"] for g in tr])
        sem = np.zeros(len(tr), np.float32)
        if data["seg"] is not None and meta.get("text") and len(data["seg"]) == len(tr):
            try:
                qe, m = qc.text_emb(meta["text"])
                sem = m.relevance(data["seg"] @ qe)
            except Exception:
                pass
        score = np.maximum(sem, kw) + 0.2 * np.minimum(sem, kw)
        for i in np.argsort(-score)[:40]:
            if score[i] >= MIN_SCORE * 0.8:
                g = tr[i]
                cands.append({"t": g["t"], "e": g["e"], "kind": "Speech", "score": float(score[i]), "seg": int(i)})

    # Picture
    if sc:
        caps = [s.get("d", "") for s in sc]
        kw = keyword_scores(q, caps)
        sem = np.zeros(len(sc), np.float32)
        vis = np.zeros(len(sc), np.float32)
        if data["cap"] is not None and meta.get("text") and len(data["cap"]) == len(sc) and any(caps):
            try:
                qe, m = qc.text_emb(meta["text"])
                sem = m.relevance(data["cap"] @ qe) * np.array([1.0 if c else 0.0 for c in caps], np.float32)
            except Exception:
                pass
        if data["img"] is not None and meta.get("vision") and len(data["img"]) == len(sc):
            try:
                qe, m = qc.vision_emb(meta["vision"])
                sims = data["img"] @ qe
                vis = m.relevance(sims)
                # A frame that stands out from the rest of this video is stronger evidence.
                if len(sims) > 3:
                    z = (sims - sims.mean()) / (sims.std() + 1e-6)
                    vis = 0.7 * vis + 0.3 * np.clip((z - 1.0) / 2.5, 0, 1)
            except Exception:
                pass
        # Generated descriptions are less reliable than what was actually said.
        text_side = np.maximum(sem, kw) * 0.85
        score = np.maximum(vis, text_side) + 0.25 * np.minimum(vis, text_side)
        for i in np.argsort(-score)[:40]:
            if score[i] >= MIN_SCORE * 0.8:
                s = sc[i]
                cands.append({"t": s["t"], "e": s["e"], "kind": "Picture", "score": float(score[i]), "scene": int(i)})

    # Merge neighbours into moments.
    window = min(30.0, max(6.0, dur * 0.03))
    cands.sort(key=lambda c: (-c["score"], c["t"]))
    out: list[dict] = []
    for c in cands:
        near = next((o for o in out if abs(o["t"] - c["t"]) < window), None)
        if near:
            if near["kind"] != c["kind"] and not near.get("merged"):
                near["merged"] = True
                near["kind"] = "Picture + speech"
                near["score"] = min(1.0, near["score"] + 0.1 * c["score"])
                near.setdefault("seg", c.get("seg"))
                near.setdefault("scene", c.get("scene"))
            continue
        out.append(dict(c))
    out = [o | {"score": min(1.0, o["score"])} for o in out if o["score"] >= MIN_SCORE]
    return [_present(o, data) for o in out]


def _scene_at(scenes: list[dict], t: float) -> int:
    i = 0
    for k, s in enumerate(scenes):
        if s["t"] <= t + 0.001:
            i = k
    return i


def _present(o: dict, data: dict) -> dict:
    tr, sc = data["transcript"], data["scenes"]
    si = o.get("scene")
    if si is None and sc:
        si = _scene_at(sc, o["t"])
    seg = tr[o["seg"]] if o.get("seg") is not None else None
    scene = sc[si] if si is not None and sc else None
    if o["kind"] == "Speech":
        title, snip = f"“{seg['x']}”", (scene.get("d") or "Speech") if scene else "Speech"
    elif o["kind"] == "Picture":
        title = scene.get("d") or f"Key frame {si + 1:02d}"
        snip = f"Key frame {si + 1:02d}"
    else:
        title = scene.get("d") or f"Key frame {si + 1:02d}"
        snip = f"“{seg['x']}”" if seg else f"Key frame {si + 1:02d}"
    return {"videoId": data["video"]["id"], "videoName": data["video"]["name"], "t": o["t"], "e": o["e"],
            "kind": o["kind"], "title": title, "snip": snip, "scene": si, "frame": scene["f"] if scene else None,
            "score": round(o["score"], 4),
            "conf": int(max(1, min(99, round(o["score"] * 100))))}


def run(lib, q: str, ids: list[str] | None, per_video: int = 5, total: int = 12) -> dict:
    q = q.strip()
    if not q:
        return {"results": [], "searched": 0}
    ids = ids or [v["id"] for v in lib.list()]
    qc = QueryCache(q)
    results, searched = [], 0
    for vid in ids:
        data = lib.index_data(vid)
        if not data or (not data["transcript"] and not data["scenes"]):
            continue
        searched += 1
        results += search_video(data, qc)[:per_video]
    results.sort(key=lambda r: (-r["score"], r["t"]))
    limit = per_video if len(ids) == 1 else total
    return {"results": results[:limit], "searched": searched}
