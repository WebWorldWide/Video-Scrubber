# SPDX-License-Identifier: GPL-3.0-or-later
"""ffmpeg side: scene cuts, dense picture samples, and proxies."""

import numpy as np
import pytest

from vscrubber import media


def reference_scores(frames):
    """The original per-frame implementation, kept to pin the vectorised one to the same output."""
    a = frames.astype(np.float32) / 255.0
    pix = np.abs(a[1:] - a[:-1]).mean(axis=(1, 2, 3))
    hists = np.stack([np.concatenate([np.histogram(f[..., c], bins=16, range=(0, 1))[0] for c in range(3)])
                      for f in a]).astype(np.float32)
    hists /= hists.sum(axis=1, keepdims=True)
    hd = np.abs(hists[1:] - hists[:-1]).sum(axis=1) / 2
    return 0.5 * np.minimum(1, pix * 3) + 0.5 * hd


def reference_cuts(frames, fps, duration, min_len):
    score = reference_scores(frames)
    cuts, n = [0.0], len(score)
    for i, s in enumerate(score):
        lo, hi = max(0, i - 6), min(n, i + 7)
        neigh = np.delete(score[lo:hi], i - lo)
        base = float(np.median(neigh)) if len(neigh) else 0.0
        t = (i + 1) / fps
        if s > max(0.2, base * 2.5 + 0.08) and t - cuts[-1] >= min_len and duration - t >= 1.0:
            cuts.append(t)
    return cuts


def synthetic_frames(n, seed=0):
    rng = np.random.default_rng(seed)
    frames = np.empty((n, 36, 64, 3), np.uint8)
    i = 0
    while i < n:
        length = int(rng.integers(3, 50))
        base = rng.integers(0, 256, size=(36, 64, 3))
        for j in range(i, min(n, i + length)):
            frames[j] = np.clip(base + rng.normal(0, 6, size=base.shape), 0, 255)
        i += length
    return frames


@pytest.mark.parametrize("n,seed", [(2, 0), (5, 1), (300, 2), (2500, 3)])  # 2500 crosses the 1024-frame chunks
def test_change_scores_match_reference(n, seed):
    frames = synthetic_frames(n, seed)
    np.testing.assert_allclose(media._change_scores(frames), reference_scores(frames), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n,seed", [(2, 0), (5, 1), (300, 2), (2500, 3)])
def test_cuts_match_reference(n, seed):
    frames = synthetic_frames(n, seed)
    assert media._cuts(frames, 2.0, n / 2, 2.0) == pytest.approx(reference_cuts(frames, 2.0, n / 2, 2.0))


def test_fill_adds_key_frames_in_long_static_stretches():
    out = media._fill([0.0, 100.0], 400.0, 30.0, 2.0)
    assert out[0] == 0.0 and 100.0 in out
    gaps = np.diff(out + [400.0])
    assert gaps.max() <= 30.0


def test_fill_caps_key_frames():
    assert len(media._fill([float(t) for t in range(0, 5000, 3)], 5000.0, 30.0, 2.0)) <= 400


def test_dense_fps_caps_samples():
    assert media.dense_fps(60) == 0.5
    assert 7200 * media.dense_fps(7200) <= media.DENSE_MAX + 1


def test_keep_dense_drops_repeats(tmp_path):
    for k in range(6):
        (tmp_path / f"d{k:05d}.jpg").write_bytes(b"x")
    tiny = np.zeros((24, 36, 64, 3), np.uint8)  # 2 fps tiny frames for 12 s
    tiny[8:16] = 200  # a different shot from 4 s to 8 s
    kept = media._keep_dense(tmp_path, 0.5, tiny, 2.0)  # samples at 0, 2, 4, 6, 8, 10 s
    assert [k["t"] for k in kept] == [0.0, 4.0, 8.0]
    assert sorted(p.name for p in tmp_path.iterdir()) == [k["f"] for k in kept]


def test_scan_video_finds_cuts_and_samples(clip, tmp_path):
    dur = media.probe(str(clip))["duration"]
    cuts, dense = media.scan_video(str(clip), dur, dense_dir=tmp_path / "dense")
    assert len(cuts) == 3
    assert cuts[0] == 0.0 and cuts[1] == pytest.approx(4.0, abs=0.5) and cuts[2] == pytest.approx(8.0, abs=0.5)
    assert len(dense) == 3  # one sample per solid-colour shot; the repeats are dropped
    for x in dense:
        assert (tmp_path / "dense" / x["f"]).stat().st_size > 0
    assert len(list((tmp_path / "dense").iterdir())) == 3


def test_scan_video_without_samples(clip):
    cuts, dense = media.scan_video(str(clip), 12.0)
    assert len(cuts) == 3 and dense == []


def test_scan_video_cancels(clip, tmp_path):
    with pytest.raises(media.Cancelled):
        media.scan_video(str(clip), 12.0, cancel=lambda: True, dense_dir=tmp_path / "dense")


def test_probe(clip):
    p = media.probe(str(clip))
    assert p["hasVideo"] and p["hasAudio"] and p["vcodec"] == "h264"
    assert p["duration"] == pytest.approx(12.0, abs=0.3)


def test_proxy_remuxes_playable_streams(clip, tmp_path, monkeypatch):
    calls = []
    real = media._run_progress
    monkeypatch.setattr(media, "_run_progress", lambda args, *a, **k: (calls.append(args), real(args, *a, **k)))
    out = tmp_path / "proxy.mp4"
    media.make_proxy(str(clip), out, 12.0, probe=media.probe(str(clip)))
    assert len(calls) == 1 and "copy" in calls[0]  # one pass, stream copy
    p = media.probe(str(out))
    assert p["vcodec"] == "h264" and p["hasAudio"]
    assert p["duration"] == pytest.approx(12.0, abs=0.3)
    assert not out.with_suffix(".part.mp4").exists()


def test_grab_frames(clip, tmp_path):
    files = media.grab_frames(str(clip), [1.0, 5.0, 9.0], tmp_path / "frames", duration=12.0)
    assert files == ["0001.jpg", "0002.jpg", "0003.jpg"]
