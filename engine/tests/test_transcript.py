# SPDX-License-Identifier: GPL-3.0-or-later
"""Transcript cleanup: hallucination filtering, repeat collapsing, word timing, and line splitting."""

from vscrubber.models import _clean, _split


def seg(t, e, x, nsp=0.0, lp=-0.2, w=None):
    return {"t": t, "e": e, "x": x, "nsp": nsp, "lp": lp, "w": w or []}


def test_drops_unsure_silence_hallucinations():
    out = _clean([seg(0, 2, "Thank you.", nsp=0.5, lp=-0.5), seg(2, 32, "you", nsp=0.1),
                  seg(40, 42, "Subtitles by the Amara.org community", nsp=0.3)])
    assert out == []


def test_keeps_confident_short_thanks():
    out = _clean([seg(10, 12, "Thank you.", nsp=0.05, lp=-0.2)])
    assert [g["x"] for g in out] == ["Thank you."]


def test_drops_very_unsure_lines_of_any_text():
    assert _clean([seg(0, 3, "Real words here", nsp=0.7, lp=-1.2)]) == []


def test_collapses_looping_text_and_merges_repeats():
    out = _clean([seg(0, 4, "I'm going home. I'm going home. I'm going home. I'm going home."),
                  seg(4, 6, "I'm going home.")])
    assert len(out) == 1
    assert out[0]["x"] == "I'm going home."
    assert out[0]["e"] == 6


def test_line_starts_at_first_word():
    out = _clean([seg(5.0, 9.0, "Hello there", w=[(6.3, " Hello"), (6.8, " there")])])
    assert out[0]["t"] == 6.3
    assert out[0]["w"] == [(6.3, "Hello"), (6.8, "there")]


def test_ignores_word_times_far_from_the_segment():
    out = _clean([seg(5.0, 9.0, "Hello", w=[(30.0, " Hello")])])
    assert out[0]["t"] == 5.0


def test_minimum_line_length():
    out = _clean([seg(3.0, 3.0, "Hi")])
    assert out[0]["e"] >= out[0]["t"] + 0.3


def words(text: str, start: float, step: float):
    return [(start + i * step, " " + w) for i, w in enumerate(text.split())]


def test_split_leaves_short_segments_alone():
    s = seg(0, 10, "Short one.", w=words("Short one.", 0, 1))
    assert _split(s) == [s]


def test_split_breaks_long_segments_at_sentences():
    text = "First sentence goes here. Second sentence is this one. Third and final sentence ends it."
    s = seg(0, 30, text, w=words(text, 0.0, 30 / 15))
    parts = _split(s)
    assert [p["x"].strip() for p in parts] == ["First sentence goes here.", "Second sentence is this one.",
                                               "Third and final sentence ends it."]
    assert parts[0]["t"] == 0.0 and parts[-1]["e"] == 30
    for a, b in zip(parts, parts[1:]):
        assert a["e"] == b["t"]  # contiguous


def test_split_caps_unpunctuated_lines():
    text = " ".join(["word"] * 40)
    parts = _split(seg(0, 40, text, w=words(text, 0.0, 1.0)))
    assert len(parts) >= 3
    assert all(p["e"] - p["t"] <= 14.0 for p in parts)
    assert sum(len(p["w"]) for p in parts) == 40
