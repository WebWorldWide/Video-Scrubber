# SPDX-License-Identifier: GPL-3.0-or-later
"""Describe-a-moment search across one or many indexed videos.

Each speech segment and key frame gets a 0–1 relevance from up to three signals:
  - meaning:  sentence-embedding similarity (transcript lines with and without their neighbours,
              frame descriptions)
  - keywords: stemmed, IDF-weighted term overlap, so exact names and numbers still win and rare
              words count for more than common ones
  - picture:  SigLIP/CLIP similarity between the query and the key frame, plus picture samples
              taken every couple of seconds inside each scene
Hits close together in one video merge into a single moment ("Picture + speech").
"""

import math
import re

import numpy as np

from . import models

STOP = set(("a an the of to in on at and or is are was were be been it its this that these when where what which who "
            "whom does do did part moment someone somebody they them their there show shows showing shown find me i we "
            "you he she his her about mention mentions mentioned say says said talk talks talking scene video clip "
            "with for from up just like some any thing time point goes go get gets has have had into by how why while "
            "during one").split())

MIN_SCORE = 0.38

# "the part where …", "find the moment when …", "show me …": wrapping that hurts picture matching.
LEAD = re.compile(r"^\s*(?:(?:please\s+)?(?:find|show|get|give|take|jump|go)\s+(?:me\s+)?(?:to\s+)?)?(?:(?:the|a|an|any)\s+)?"
                  r"(?:(?:one|part|parts|moment|moments|bit|section|scene|scenes|clip|point|time|spot|place|segment|shot)\s+"
                  r"(?:where|when|that|in\s+which|with|of|about|showing)\s+)?(?:(?:when|where|whenever)\s+)?", re.I)

NUMBERS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NUMBERS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
                "ninety": "90", "hundred": "100", "thousand": "1000"})
NUMBERS.pop("one")  # "the one where …" is almost never the number

SUFFIXES = ("ingly", "edly", "ings", "ing", "ies", "ied", "ers", "est", "ed", "es", "er", "ly", "s", "e")


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower().replace("’", "").replace("'", ""))


def stem(w: str) -> str:
    """Light suffix stripping, applied the same way to queries and documents:
    price/prices/pricing → pric, meeting/meets → meet, running → run, studies → study."""
    w = NUMBERS.get(w, w)
    if w.isdigit() or len(w) <= 3:
        return w
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if suf in ("ies", "ied"):
                base += "y"
            elif suf in ("ing", "ings", "ed") and len(base) >= 4 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]
            return base
    return w


def clean_query(q: str) -> str:
    c = LEAD.sub("", q, count=1).strip(" ?.!")
    return c if toks(c) else q


def terms(q: str) -> list[str]:
    return list(dict.fromkeys(stem(w) for w in toks(q) if w not in STOP))


def doc_stems(text: str) -> list[str]:
    return [stem(w) for w in toks(text) if w not in STOP]


class KeywordIndex:
    """Stems and an inverted index for one video's lines, windows and descriptions, built once per video."""

    def __init__(self, transcript: list[dict], windows: list[str], captions: list[str]):
        self.fields = {"seg": [doc_stems(g["x"]) for g in transcript], "win": [doc_stems(t) for t in windows],
                       "cap": [doc_stems(c) for c in captions]}
        self.bigrams = {k: [set(zip(d, d[1:])) for d in docs] for k, docs in self.fields.items()}
        self.post: dict[str, dict[str, set[int]]] = {}
        for k, docs in self.fields.items():
            p: dict[str, set[int]] = {}
            for i, d in enumerate(docs):
                for w in d:
                    p.setdefault(w, set()).add(i)
            self.post[k] = p
        # Document frequency over the whole video (lines and descriptions), for IDF.
        self.n = len(self.fields["seg"]) + len(self.fields["cap"])
        self.df: dict[str, int] = {}
        for k in ("seg", "cap"):
            for w, ids in self.post[k].items():
                self.df[w] = self.df.get(w, 0) + len(ids)

    def _match(self, field: str, t: str) -> set[int]:
        post = self.post[field]
        hit = set(post.get(t, ()))
        if len(t) >= 6:  # compounds and longer forms: photograph → photography
            for w, ids in post.items():
                if w != t and w.startswith(t):
                    hit |= ids
        return hit

    def idf(self, t: str) -> float:
        df = self.df.get(t, 0)
        if df == 0 and len(t) >= 6:
            df = sum(n for w, n in self.df.items() if w.startswith(t))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, field: str, qterms: list[str]) -> np.ndarray:
        docs = self.fields[field]
        out = np.zeros(len(docs), np.float32)
        if not qterms or not docs:
            return out
        w = np.array([self.idf(t) for t in qterms], np.float32)
        for t, wt in zip(qterms, w):
            for i in self._match(field, t):
                out[i] += wt
        out /= w.sum()
        if len(qterms) > 1:
            # Terms next to each other, in order, are much stronger evidence than scattered ones.
            pairs = list(zip(qterms, qterms[1:]))
            for i in np.nonzero(out)[0]:
                bg = self.bigrams[field][i]
                out[i] += 0.15 * sum(p in bg for p in pairs) / len(pairs)
        # Single-word partial matches are weak evidence on their own.
        return out * (0.85 if len(qterms) > 1 else 0.7)


def keyword_scores(q: str, texts: list[str]) -> np.ndarray:
    """Keyword relevance of `texts` to `q` (standalone helper; search uses a cached KeywordIndex)."""
    return KeywordIndex([{"x": t} for t in texts], [], []).scores("seg", terms(q))


class QueryCache:
    """Embeds the query once per model, shared across videos."""

    def __init__(self, q: str):
        self.q = q
        self.clean = clean_query(q)
        self.terms = terms(self.clean)
        self.text: dict[str, tuple[np.ndarray, object]] = {}
        self.vision: dict[str, tuple[np.ndarray, object]] = {}

    def text_emb(self, model_id: str):
        if model_id not in self.text:
            with models.lock("text"):
                m = models.get("text", model_id)
                self.text[model_id] = (m.query(self.clean), m)
        return self.text[model_id]

    def vision_emb(self, model_id: str):
        if model_id not in self.vision:
            with models.lock("vision"):
                m = models.get("vision", model_id)
                self.vision[model_id] = (m.embed_text(self.clean), m)
        return self.vision[model_id]


def _spans(data: dict) -> list[tuple[int, int]]:
    if "_spans" not in data:
        from .library import window_spans

        data["_spans"] = window_spans(data["transcript"])
    return data["_spans"]


def _kw_index(data: dict) -> KeywordIndex:
    if "_kw" not in data:
        from .library import windows

        data["_kw"] = KeywordIndex(data["transcript"], windows(data["transcript"]), [s.get("d", "") for s in data["scenes"]])
    return data["_kw"]


def _sem(data: dict, key: str, n: int, qc: QueryCache) -> np.ndarray:
    emb, meta = data.get(key), data["models"]
    if emb is None or not meta.get("text") or len(emb) != n or n == 0:
        return np.zeros(n, np.float32)
    try:
        qe, m = qc.text_emb(meta["text"])
        return m.relevance(emb @ qe)
    except Exception:
        return np.zeros(n, np.float32)


def _word_time(data: dict, i: int, qterms: list[str]) -> float | None:
    """Start of the first query word said in segment i, if it's well into the line."""
    words = data.get("words") or []
    if i >= len(words) or not qterms:
        return None
    seg = data["transcript"][i]
    for t, w in words[i]:
        st = [stem(x) for x in toks(w)]
        if any(s == q or (len(q) >= 6 and s.startswith(q)) for s in st for q in qterms):
            # Land a beat before the word, but don't bother if it's near the start of the line anyway.
            return round(max(seg["t"], t - 1.0), 2) if t - seg["t"] > 3 else None
    return None


def search_video(data: dict, qc: QueryCache) -> list[dict]:
    tr, sc, meta = data["transcript"], data["scenes"], data["models"]
    dur = max(1.0, data["video"]["probe"]["duration"])
    kwi = _kw_index(data)
    cands = []

    # Speech: each line on its own, and with its neighbours (slightly discounted).
    if tr:
        kw_line, kw_win = kwi.scores("seg", qc.terms), 0.9 * kwi.scores("win", qc.terms)
        sem_line, sem_win = _sem(data, "seg", len(tr), qc), 0.92 * _sem(data, "win", len(tr), qc)
        kw, sem = np.maximum(kw_line, kw_win), np.maximum(sem_line, sem_win)
        score = np.maximum(sem, kw) + 0.2 * np.minimum(sem, kw)
        own = np.maximum(sem_line, kw_line)
        spans = _spans(data)
        for i in np.argsort(-score)[:40]:
            if score[i] >= MIN_SCORE * 0.8:
                j = int(i)
                if kw_win[i] > kw_line[i] or sem_win[i] > sem_line[i]:
                    # The match came from the line's neighbours: point at the line in the window that matched.
                    a, b = spans[i]
                    j = a + int(np.argmax(own[a:b + 1]))
                g = tr[j]
                t = _word_time(data, j, qc.terms) if kw[i] > 0 else None
                cands.append({"t": t if t is not None else g["t"], "e": g["e"], "kind": "Speech", "score": float(score[i]),
                              "seg": j})

    # Picture
    if sc:
        caps = [s.get("d", "") for s in sc]
        kw = kwi.scores("cap", qc.terms)
        sem = _sem(data, "cap", len(sc), qc) * np.array([1.0 if c else 0.0 for c in caps], np.float32)
        vis = np.zeros(len(sc), np.float32)
        best_t = [s["t"] for s in sc]
        best_f = [None] * len(sc)
        dense, dimg = data.get("dense") or [], data.get("dimg")
        if data["img"] is not None and meta.get("vision") and len(data["img"]) == len(sc):
            try:
                qe, m = qc.vision_emb(meta["vision"])
                has_dense = dimg is not None and len(dimg) == len(dense) and len(dense) > 0
                sims = np.concatenate([data["img"] @ qe, dimg @ qe]) if has_dense else data["img"] @ qe
                rel = m.relevance(sims)
                # A frame that stands out from the rest of this video is stronger evidence.
                if len(sims) > 3:
                    z = (sims - sims.mean()) / (sims.std() + 1e-6)
                    rel = 0.7 * rel + 0.3 * np.clip((z - 1.0) / 2.5, 0, 1)
                vis = rel[:len(sc)].copy()
                if has_dense:
                    # Each sample counts toward the scene it falls in, and the best one sets where to land.
                    starts = np.array([s["t"] for s in sc])
                    owner = np.clip(np.searchsorted(starts, [x["t"] for x in dense], side="right") - 1, 0, len(sc) - 1)
                    for j, r in enumerate(rel[len(sc):]):
                        k = owner[j]
                        if r > vis[k]:
                            vis[k], best_t[k], best_f[k] = r, dense[j]["t"], dense[j]["f"]
            except Exception:
                pass
        # Generated descriptions are less reliable than what was actually said.
        text_side = np.maximum(sem, kw) * 0.85
        score = np.maximum(vis, text_side) + 0.25 * np.minimum(vis, text_side)
        for i in np.argsort(-score)[:40]:
            if score[i] >= MIN_SCORE * 0.8:
                s = sc[i]
                # Land on the best-matching sample only when the picture is what matched.
                t, f = (best_t[i], best_f[i]) if vis[i] >= text_side[i] else (s["t"], None)
                cands.append({"t": t, "e": s["e"], "kind": "Picture", "score": float(score[i]), "scene": int(i), "frame": f})

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
                near["score"] = near["score"] + 0.1 * c["score"]
                near.setdefault("seg", c.get("seg"))
                near.setdefault("scene", c.get("scene"))
            continue
        out.append(dict(c))
    # Scores aren't capped, so strong matches keep their order; only the shown confidence is.
    out = [o for o in out if o["score"] >= MIN_SCORE]
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
            "kind": o["kind"], "title": title, "snip": snip, "scene": si,
            "frame": o.get("frame") or (scene["f"] if scene else None),
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
