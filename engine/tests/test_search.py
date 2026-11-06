# SPDX-License-Identifier: GPL-3.0-or-later
"""Search: query cleanup, keyword scoring, ranking, and where results land."""

import numpy as np
import pytest

from vscrubber import search as S


class Fake:
    """Stands in for a text or picture model: relevance is the similarity itself, uncapped."""

    def relevance(self, sims):
        return np.asarray(sims, np.float32)


def unit(*xs):
    v = np.array(xs, np.float32)
    return v / np.linalg.norm(v)


def data(transcript=(), scenes=(), words=None, **arrays):
    tr = [dict(g) for g in transcript]
    d = {"video": {"id": "v1", "name": "v.mp4", "probe": {"duration": 600.0}}, "transcript": tr,
         "scenes": [dict(s) for s in scenes], "words": words or [], "dense": arrays.pop("dense", []),
         "models": arrays.pop("models", {}), "img": None, "seg": None, "cap": None, "win": None, "dimg": None}
    d.update(arrays)
    return d


def query(q, text=None, vision=None):
    qc = S.QueryCache(q)
    if text is not None:
        qc.text["t"] = (text, Fake())
    if vision is not None:
        qc.vision["p"] = (vision, Fake())
    return qc


# ---------------------------------------------------------------- text handling


@pytest.mark.parametrize("word,stem", [("price", "pric"), ("prices", "pric"), ("pricing", "pric"), ("priced", "pric"),
                                       ("meeting", "meet"), ("running", "run"), ("studies", "study"),
                                       ("whiteboards", "whiteboard"), ("cats", "cat"), ("catalog", "catalog"),
                                       ("two", "2"), ("one", "one")])
def test_stem(word, stem):
    assert S.stem(word) == stem


@pytest.mark.parametrize("q,clean", [
    ("the part where someone shows the pricing whiteboard", "someone shows the pricing whiteboard"),
    ("find the moment when the dog jumps", "the dog jumps"),
    ("show me the red car", "red car"),
    ("the one where they talk about 2 cats", "they talk about 2 cats"),
    ("cat on a couch", "cat on a couch"),
    ("the", "the"),  # nothing left after stripping: keep the original
])
def test_clean_query(q, clean):
    assert S.clean_query(q) == clean


def test_keywords_match_word_forms_not_prefixes():
    s = S.keyword_scores("cat", ["We looked at the catalog", "My cat sat down", "two cats"])
    assert s[0] == 0 and s[1] > 0 and s[2] > 0


def test_keywords_prefer_adjacent_terms_and_full_matches():
    s = S.keyword_scores("pricing whiteboard", ["the price went up", "Pricing whiteboard is set", "the whiteboard was empty"])
    assert s[1] > s[0] and s[1] > s[2]


def test_rare_terms_outweigh_common_ones():
    texts = ["the team met"] * 8 + ["the team met the giraffe", "a giraffe"]
    s = S.keyword_scores("team giraffe", texts)
    assert s[9] > s[0]  # "giraffe" alone beats "team" alone


def test_numbers_match_digits_and_words():
    s = S.keyword_scores("2 cats", ["two cats", "one cat"])
    assert s[0] > s[1]


# ---------------------------------------------------------------- ranking and landing


class Lib:
    def __init__(self, items):
        self.items = items

    def list(self):
        return [{"id": k} for k in self.items]

    def index_data(self, vid):
        return self.items[vid]


def test_stronger_meaning_match_ranks_first_even_past_calibration(monkeypatch):
    # Both lines are above the top of the calibration range; the closer one must still win,
    # through the final sort in run() too (capped scores used to tie and fall back to time order).
    tr = [{"t": 10, "e": 14, "x": "alpha"}, {"t": 100, "e": 104, "x": "beta"}]
    q = unit(1, 0)
    seg = np.stack([unit(1.0, 0.3), unit(1, 0)]) * 3  # relevance > 1 for both
    d = data(tr, seg=seg, win=seg, models={"text": "t"})
    qc = query("something", text=q)
    monkeypatch.setattr(S, "QueryCache", lambda q: qc)
    res = S.run(Lib({"v1": d}), "something", ["v1"])["results"]
    assert [r["t"] for r in res][:2] == [100, 10]
    assert all(r["conf"] <= 99 for r in res)


def test_text_relevance_keeps_order_past_calibration():
    from vscrubber.models import TextEmbedder

    te = object.__new__(TextEmbedder)
    te.meta = {"calib": (0.4, 0.6)}
    rel = te.relevance(np.array([0.1, 0.2, 0.3, 0.7, 0.9], np.float32))
    assert rel[4] > rel[3] > 0 and rel[0] == 0


def test_picture_relevance_keeps_order_past_calibration():
    from vscrubber.models import Vision

    v = object.__new__(Vision)
    v.sigmoid, v.meta = False, {"calib": (0.2, 0.3)}
    rel = v.relevance(np.array([0.1, 0.35, 0.5], np.float32))
    assert rel[2] > rel[1] > 0 and rel[0] == 0
    v.sigmoid, v.scale, v.bias = True, 100.0, -10.0
    rel = v.relevance(np.array([0.0, 0.1, 0.2], np.float32))
    assert rel[2] > rel[1] > rel[0]


def test_keyword_hit_lands_near_the_word():
    tr = [{"t": 20.0, "e": 32.0, "x": "we talked for a while and then the giraffe walked in"}]
    w = [[(20.0 + i, x) for i, x in enumerate(tr[0]["x"].split())]]
    res = S.search_video(data(tr, words=w), query("giraffe"))
    assert res[0]["t"] == pytest.approx(20.0 + 8 - 1.0)  # a beat before "giraffe" at 28 s


def test_keyword_near_line_start_keeps_line_start():
    tr = [{"t": 20.0, "e": 26.0, "x": "giraffe at the start"}]
    w = [[(20.5, "giraffe"), (21.0, "at")]]
    assert S.search_video(data(tr, words=w), query("giraffe"))[0]["t"] == 20.0


def test_window_match_points_at_the_line_that_matched():
    # Line 0's window (lines 0+1) matches best, but it's line 1 that says it.
    tr = [{"t": 0, "e": 3, "x": "so anyway"}, {"t": 3, "e": 6, "x": "x"}, {"t": 300, "e": 303, "x": "y"}]
    q = unit(1, 0, 0)
    seg = np.stack([unit(0, 1, 0), unit(1, 0.1, 0), unit(0, 0, 1)])
    win = np.stack([unit(1, 0, 0) * 2, unit(0, 1, 0), unit(0, 0, 1)])
    res = S.search_video(data(tr, seg=seg, win=win, models={"text": "t"}), query("q", text=q))
    assert res[0]["t"] == 3


def test_phrase_split_across_lines_points_at_the_matching_line():
    tr = [{"t": 0, "e": 3, "x": "it is all about"}, {"t": 3, "e": 6, "x": "the purple elephant"},
          {"t": 6, "e": 9, "x": "and nothing else"}, {"t": 200, "e": 203, "x": "unrelated words"}]
    res = S.search_video(data(tr), query("about the purple elephant"))
    assert res[0]["t"] == 3


def test_picture_result_lands_on_best_dense_sample():
    scenes = [{"t": 0.0, "e": 100.0, "f": "0001.jpg", "d": ""}, {"t": 100.0, "e": 200.0, "f": "0002.jpg", "d": ""}]
    dense = [{"t": 20.0, "f": "d00010.jpg"}, {"t": 140.0, "f": "d00070.jpg"}, {"t": 160.0, "f": "d00080.jpg"}]
    q = unit(1, 0, 0)
    img = np.stack([unit(0, 1, 0), unit(0.2, 1, 0)])
    dimg = np.stack([unit(0, 0, 1), unit(1, 0.4, 0), unit(1, 0.05, 0)])
    d = data(scenes=scenes, dense=dense, img=img, dimg=dimg, models={"vision": "p"})
    res = S.search_video(d, query("a red ball", vision=q))
    assert res[0]["t"] == 160.0
    assert res[0]["frame"] == "d00080.jpg"
    assert res[0]["scene"] == 1


def test_search_without_dense_samples_uses_key_frames():
    scenes = [{"t": 0.0, "e": 50.0, "f": "0001.jpg", "d": ""}, {"t": 50.0, "e": 100.0, "f": "0002.jpg", "d": ""}]
    img = np.stack([unit(0, 1), unit(1, 0)])
    res = S.search_video(data(scenes=scenes, img=img, models={"vision": "p"}), query("x", vision=unit(1, 0)))
    assert (res[0]["t"], res[0]["frame"]) == (50.0, "0002.jpg")


def test_description_match_without_picture_model():
    scenes = [{"t": 0.0, "e": 50.0, "f": "0001.jpg", "d": "A man at a desk"},
              {"t": 50.0, "e": 100.0, "f": "0002.jpg", "d": "A whiteboard covered in pricing tables"}]
    res = S.search_video(data(scenes=scenes), query("pricing whiteboard"))
    assert res[0]["t"] == 50.0 and res[0]["kind"] == "Picture"


def test_nearby_speech_and_picture_merge():
    tr = [{"t": 51.0, "e": 55.0, "x": "here is the pricing whiteboard"}]
    scenes = [{"t": 50.0, "e": 100.0, "f": "0001.jpg", "d": "A pricing whiteboard"}]
    res = S.search_video(data(tr, scenes), query("pricing whiteboard"))
    assert len(res) == 1 and res[0]["kind"] == "Picture + speech"


def test_no_results_for_unrelated_query():
    tr = [{"t": 0, "e": 3, "x": "hello world"}]
    assert S.search_video(data(tr), query("giraffe")) == []


def test_run_limits_and_skips_empty_videos():
    many = [{"t": i * 60.0, "e": i * 60.0 + 3, "x": f"giraffe number {i}"} for i in range(10)]
    a = data(many)
    b = data(many)
    b["video"] = {"id": "v2", "name": "w.mp4", "probe": {"duration": 600.0}}
    lib = Lib({"v1": a, "v2": b, "v3": data()})
    one = S.run(lib, "giraffe", ["v1"])
    assert len(one["results"]) == 5 and one["searched"] == 1
    every = S.run(lib, "giraffe", None)
    assert every["searched"] == 2 and len(every["results"]) == 10
    assert S.run(lib, "  ", None) == {"results": [], "searched": 0}
