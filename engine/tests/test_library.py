# SPDX-License-Identifier: GPL-3.0-or-later
"""The indexing pipeline end to end on a real clip, with fake models in place of the real ones."""

import json
import threading
import time

import numpy as np
import pytest
from PIL import Image

from vscrubber import library as L
from vscrubber import media, models


class FakeSpeech:
    engine = "fake"

    def __init__(self, gate=None, fail=None):
        self.gate, self.fail = gate, fail

    def transcribe(self, wav, duration, language, cb):
        if self.fail:
            raise self.fail
        while self.gate is not None and not self.gate.is_set():
            cb(0.5)  # progress callbacks are where cancellation is noticed
            time.sleep(0.01)
        cb(1.0)
        return [{"t": 0.5, "e": 3.0, "x": "Hello there.", "w": [(0.5, "Hello"), (1.0, "there.")]},
                {"t": 3.2, "e": 6.0, "x": "This is a test.", "w": [(3.2, "This"), (3.5, "is")]}]


class FakeVision:
    def __init__(self):
        self.calls = []

    def embed_images(self, paths, cb=None, lock=None):
        self.calls.append(list(paths))
        rng = np.random.default_rng(len(paths))
        e = rng.normal(size=(len(paths), 8)).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class FakeCaptioner:
    def __init__(self):
        self.calls = 0

    def describe(self, path):
        self.calls += 1
        return f"Frame {self.calls}"


class FakeText:
    def docs(self, texts, lock=None):
        if not texts:
            return np.zeros((0, 1), np.float32)
        return np.ones((len(texts), 4), np.float32) / 2


@pytest.fixture
def fakes(data_dir, monkeypatch):
    f = {"speech": FakeSpeech(), "vision": FakeVision(), "caption": FakeCaptioner(), "text": FakeText()}
    monkeypatch.setattr(models, "ensure", lambda *a, **k: None)
    monkeypatch.setattr(models, "get", lambda kind, *a, **k: f[kind])
    return f


def wait(lib, vid, timeout=60):
    end = time.monotonic() + timeout
    while lib.videos[vid]["status"] not in ("ready", "error"):
        assert time.monotonic() < end, "indexing timed out"
        time.sleep(0.05)
    return lib.videos[vid]


def wait_idle(lib, timeout=30):
    end = time.monotonic() + timeout
    while lib.running:
        assert time.monotonic() < end, "worker didn't stop"
        time.sleep(0.05)


def test_index_builds_every_part(fakes, clip):
    lib = L.Library()
    vid = lib.add(str(clip))["id"]
    v = wait(lib, vid)
    assert v["status"] == "ready", v.get("error")
    assert v["warnings"] == []
    d = lib.dir(vid)
    tr = json.loads((d / "transcript.json").read_text())
    assert [g["x"] for g in tr] == ["Hello there.", "This is a test."]
    assert all("w" not in g for g in tr)  # word timings live in words.json
    assert json.loads((d / "words.json").read_text())[0][0] == [0.5, "Hello"]
    scenes = json.loads((d / "scenes.json").read_text())
    dense = json.loads((d / "dense.json").read_text())
    assert len(scenes) == 3 and len(dense) == 3
    assert [s["d"] for s in scenes] == ["Frame 1", "Frame 2", "Frame 3"]
    assert np.load(d / "img.npy").shape[0] == 3 and np.load(d / "dense.npy").shape[0] == 3
    assert np.load(d / "seg.npy").shape[0] == 2 and np.load(d / "win.npy").shape[0] == 2
    assert np.load(d / "cap.npy").shape[0] == 3
    assert not (d / "audio.wav").exists()
    meta = json.loads((d / "index.json").read_text())
    assert meta["vision"] and meta["caption"] and meta["text"] and meta["speechEngine"] == "fake"
    data = lib.index_data(vid)
    assert data["dimg"] is not None and data["win"] is not None and len(data["words"]) == 2


def test_upgrading_an_old_index_keeps_transcript_and_descriptions(fakes, clip):
    lib = L.Library()
    vid = lib.add(str(clip))["id"]
    wait(lib, vid)
    d = lib.dir(vid)
    for name in ("dense.json", "dense.npy", "win.npy"):  # an index from before dense sampling
        (d / name).unlink()
    import shutil

    shutil.rmtree(d / "dense")
    before = json.loads((d / "scenes.json").read_text())
    fakes["caption"].calls = 0
    fakes["speech"].fail = AssertionError("speech shouldn't run again")
    lib.reindex(vid)
    time.sleep(0.2)
    v = wait(lib, vid)
    assert v["status"] == "ready", v.get("error")
    assert fakes["caption"].calls == 0
    assert json.loads((d / "scenes.json").read_text()) == before
    assert len(json.loads((d / "dense.json").read_text())) == 3
    assert (d / "dense.npy").exists() and (d / "win.npy").exists()


def test_audio_only_file(fakes, clip, tmp_path):
    wav = tmp_path / "talk.m4a"
    media.clip_audio(str(clip), 0, 6, wav)
    lib = L.Library()
    vid = lib.add(str(wav))["id"]
    v = wait(lib, vid)
    assert v["status"] == "ready", v.get("error")
    assert v["counts"] == {"segments": 2, "scenes": 0}
    assert fakes["vision"].calls == []


def test_speech_failure_marks_the_video(fakes, clip):
    fakes["speech"].fail = RuntimeError("model exploded")
    lib = L.Library()
    vid = lib.add(str(clip))["id"]
    v = wait(lib, vid)
    assert v["status"] == "error" and "model exploded" in v["error"]
    wait_idle(lib)


def test_removing_a_video_stops_transcription(fakes, clip):
    gate = threading.Event()
    fakes["speech"].gate = gate
    lib = L.Library()
    vid = lib.add(str(clip))["id"]
    end = time.monotonic() + 30
    while lib.videos[vid]["status"] != "indexing":
        assert time.monotonic() < end
        time.sleep(0.02)
    time.sleep(0.2)
    lib.remove(vid)
    wait_idle(lib, timeout=10)  # the gate never opens: only cancellation can end the worker
    assert not lib.dir(vid).exists()
    gate.set()


def test_windows():
    tr = [{"t": 0, "e": 2, "x": "a"}, {"t": 2.5, "e": 4, "x": "b"}, {"t": 20, "e": 22, "x": "c"}]
    assert L.window_spans(tr) == [(0, 1), (0, 1), (2, 2)]
    assert L.windows(tr) == ["a b", "a b", "c"]
    assert L.windows([]) == []


def test_alike_needs_matching_picture_and_pixels(tmp_path):
    frames = []
    for i, shade in enumerate((100, 100, 140)):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (192, 108), (shade, shade, shade)).save(p)
        frames.append(str(p))
    e = np.array([[1, 0], [1, 0], [1, 0]], np.float32)  # identical embeddings for all three
    np.save(tmp_path / "img.npy", e)
    alike = L._alike(tmp_path, frames)
    assert alike(1, 0)
    assert not alike(2, 0)  # same embedding, visibly different picture
    np.save(tmp_path / "img.npy", np.array([[1, 0], [0, 1], [1, 0]], np.float32))
    assert not L._alike(tmp_path, frames)(1, 0)  # same pixels, different embedding


def test_alike_without_embeddings(tmp_path):
    assert not L._alike(tmp_path, ["a", "b"])(1, 0)


def test_job_reraises():
    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        L._Job(boom).result()
    assert L._Job(lambda: 5).result() == 5
