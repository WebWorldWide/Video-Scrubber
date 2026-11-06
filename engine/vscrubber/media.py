# SPDX-License-Identifier: GPL-3.0-or-later
"""ffmpeg wrappers: probing, audio extraction, scene detection, key frames, clips and proxies."""

import functools
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from .hardware import NO_WINDOW

Progress = Callable[[float], None]


def _which(name: str, env: str) -> str | None:
    p = os.environ.get(env)
    if p and Path(p).exists():
        return p
    return shutil.which(name)


@functools.lru_cache(None)
def ffmpeg() -> str:
    p = _which("ffmpeg", "VS_FFMPEG")
    if p:
        return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg isn't installed. Install it, or set VS_FFMPEG to its path.")


@functools.lru_cache(None)
def ffprobe() -> str | None:
    return _which("ffprobe", "VS_FFPROBE")


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, creationflags=NO_WINDOW, **kw)


def probe(path: str) -> dict:
    fp = ffprobe()
    if fp:
        r = run([fp, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
        if r.returncode == 0:
            d = json.loads(r.stdout)
            v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
            a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
            fps = 0.0
            if v and v.get("avg_frame_rate", "0/0") != "0/0":
                n, dd = v["avg_frame_rate"].split("/")
                fps = float(n) / float(dd) if float(dd) else 0.0
            return {
                "duration": float(d.get("format", {}).get("duration") or (v or {}).get("duration") or 0),
                "width": int((v or {}).get("width") or 0), "height": int((v or {}).get("height") or 0),
                "fps": round(fps, 3), "vcodec": (v or {}).get("codec_name"), "acodec": (a or {}).get("codec_name"),
                "hasVideo": v is not None, "hasAudio": a is not None,
            }
    # Fallback: parse `ffmpeg -i` banner.
    err = run([ffmpeg(), "-hide_banner", "-i", path]).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    dur = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0
    vm = re.search(r"Video: (\w+).*?, (\d{2,5})x(\d{2,5})", err)
    fm = re.search(r"([\d.]+) fps", err)
    am = re.search(r"Audio: (\w+)", err)
    return {"duration": dur, "width": int(vm[2]) if vm else 0, "height": int(vm[3]) if vm else 0,
            "fps": float(fm[1]) if fm else 0, "vcodec": vm[1] if vm else None, "acodec": am[1] if am else None,
            "hasVideo": bool(vm), "hasAudio": bool(am)}


def _run_progress(args: list[str], duration: float, cb: Progress | None, cancel=None):
    """Run ffmpeg with -progress on stdout, reporting 0–1."""
    p = subprocess.Popen([args[0], "-progress", "pipe:1", "-nostats", *args[1:]], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, creationflags=NO_WINDOW)
    for line in p.stdout:
        if cancel and cancel():
            p.kill()
            raise Cancelled()
        if line.startswith("out_time_us=") and cb and duration > 0:
            try:
                cb(min(1.0, int(line.split("=")[1]) / 1e6 / duration))
            except ValueError:
                pass
    err = p.stderr.read()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed: " + err.strip().splitlines()[-1] if err.strip() else "ffmpeg failed")


class Cancelled(Exception):
    pass


def extract_audio(path: str, out: Path, duration: float, cb: Progress | None = None, cancel=None):
    _run_progress([ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
                   "-ar", "16000", "-c:a", "pcm_s16le", str(out)], duration, cb, cancel)


DENSE_MAX = 1800  # most picture-search samples per video
DENSE_W = 384  # SigLIP so400m's input size; smaller models downscale


def dense_fps(duration: float) -> float:
    """Picture-search sampling rate: every 2 s, sparser for long videos."""
    return min(0.5, DENSE_MAX / max(duration, 1.0))


def scan_video(path: str, duration: float, cb: Progress | None = None, cancel=None, dense_dir: Path | None = None,
               fps: float = 2.0, min_len: float = 2.0, max_gap: float | None = None) -> tuple[list[float], list[dict]]:
    """One decode pass that finds scene cuts and, with `dense_dir`,
    saves small frames every couple of seconds for picture search.

    Returns (scene start times, [{t, f}] dense samples). Dense samples that look the same as the
    previous kept one are dropped, so static stretches cost one picture embedding instead of dozens."""
    W, H = 64, 36
    max_gap = max_gap or max(30.0, duration / 150)
    dfps = dense_fps(duration)
    frames = None
    # Multi-threaded software decode first: ffmpeg's hwaccel decoders work a frame at a time, which
    # measured 2-11x slower for this full-pass scan (1080p H.264 and 4K HEVC on Apple Silicon).
    for hw in ([], ["-hwaccel", "auto"]):
        if dense_dir:
            shutil.rmtree(dense_dir, ignore_errors=True)
            dense_dir.mkdir(parents=True)
        try:
            frames = _read_frames(path, _scan_outputs(W, H, fps, dfps, dense_dir), W, H, duration * fps, cb, cancel, hw)
            break
        except Cancelled:
            raise
        except RuntimeError:
            continue
    if frames is None or len(frames) < 2:
        return [0.0], _keep_dense(dense_dir, dfps, frames, fps)
    cuts = _cuts(frames, fps, duration, min_len)
    dense = _keep_dense(dense_dir, dfps, frames, fps)
    return _fill(cuts, duration, max_gap, min_len), dense


def _scan_outputs(W: int, H: int, fps: float, dfps: float, dense_dir: Path | None) -> list[str]:
    tiny = f"fps={fps},scale={W}:{H}:flags=area,format=rgb24"
    if not dense_dir:
        return ["-vf", tiny, "-an", "-f", "rawvideo", "-"]
    big = (f"fps={dfps},scale={DENSE_W}:{DENSE_W}:force_original_aspect_ratio=decrease:"
           "force_divisible_by=2:flags=bilinear")
    return ["-filter_complex", f"[0:V:0]split=2[a][b];[a]{tiny}[s];[b]{big}[d]",
            "-map", "[s]", "-an", "-f", "rawvideo", "-",
            "-map", "[d]", "-an", "-q:v", "4", "-start_number", "0", str(dense_dir / "d%05d.jpg")]


def _change_scores(frames: np.ndarray) -> np.ndarray:
    """How much each tiny frame changed from the previous one, 0–1 (pixels and colour histograms).
    Works in uint8/int16 chunks, so a two-hour video stays around 100 MB instead of 1 GB+."""
    n = len(frames)
    pix = np.empty(n - 1, np.float32)
    hd = np.empty(n - 1, np.float32)
    off = np.arange(3, dtype=np.int64) * 16
    prev_h = None
    CH = 1024
    for s in range(0, n, CH):
        f = frames[s:s + CH]
        # Pixel change against the previous frame (including the last frame of the previous chunk).
        left = frames[max(0, s - 1):s + len(f) - 1] if s else f[:-1]
        right = f if s else f[1:]
        pix[max(0, s - 1):s + len(f) - 1] = np.abs(right.astype(np.int16) - left).mean(axis=(1, 2, 3)) / 255.0
        # 16-bin colour histograms per channel, built with one bincount for the whole chunk.
        k = len(f)
        idx = ((f >> 4).reshape(k, -1, 3).astype(np.int64) + off) + (np.arange(k, dtype=np.int64) * 48)[:, None, None]
        h = np.bincount(idx.ravel(), minlength=k * 48).reshape(k, 48).astype(np.float32)
        h /= h[:, :16].sum(axis=1, keepdims=True)  # every channel has the same pixel count
        both = h if prev_h is None else np.concatenate([prev_h, h])
        d = np.abs(both[1:] - both[:-1]).sum(axis=1) / 6  # /2 per channel-normalised histogram, /3 channels
        hd[max(0, s - 1):s + k - 1] = d
        prev_h = h[-1:]
    return 0.5 * np.minimum(1, pix * 3) + 0.5 * hd


def _cuts(frames: np.ndarray, fps: float, duration: float, min_len: float) -> list[float]:
    """Cut where the change from the previous frame spikes."""
    score = _change_scores(frames)
    # A cut has to spike above its neighbourhood, so constant motion (pans, zooms) doesn't read as cuts
    # while a hard cut out of a still shot does. Neighbourhood: 6 either side, excluding itself.
    m = len(score)
    pad = np.concatenate([np.full(6, np.nan, np.float32), score, np.full(6, np.nan, np.float32)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 13).copy()
    win[:, 6] = np.nan
    with np.errstate(all="ignore"):
        base = np.nan_to_num(np.nanmedian(win, axis=1)) if m > 1 else np.zeros(m, np.float32)
    thresh = np.maximum(0.2, base * 2.5 + 0.08)
    cuts = [0.0]
    for i in np.nonzero(score > thresh)[0]:
        t = (int(i) + 1) / fps
        if t - cuts[-1] >= min_len and duration - t >= 1.0:
            cuts.append(t)
    return cuts


def _keep_dense(dense_dir: Path | None, dfps: float, frames: np.ndarray | None, fps: float,
                same: float = 0.02) -> list[dict]:
    """Keep dense samples that differ from the last kept one; delete the rest."""
    if not dense_dir or not dense_dir.exists():
        return []
    files = sorted(dense_dir.glob("d*.jpg"))
    out, last = [], None
    for p in files:
        k = int(p.stem[1:])
        t = k / dfps
        tiny = None
        if frames is not None and len(frames):
            tiny = frames[min(len(frames) - 1, int(round(t * fps)))].astype(np.int16)
        if last is not None and tiny is not None and np.abs(tiny - last).mean() / 255.0 < same:
            p.unlink(missing_ok=True)
            continue
        last = tiny
        out.append({"t": round(t, 3), "f": p.name})
    return out


def _fill(cuts: list[float], duration: float, max_gap: float, min_len: float) -> list[float]:
    # Long static stretches still get a key frame every `max_gap` seconds.
    filled = []
    bounds = cuts + [duration]
    for s, e in zip(bounds[:-1], bounds[1:]):
        filled.append(s)
        n = int((e - s) // max_gap)
        for k in range(1, n + 1):
            t = s + (e - s) * k / (n + 1)
            if e - t >= min_len:
                filled.append(t)
    if len(filled) > 400:
        idx = np.linspace(0, len(filled) - 1, 400).round().astype(int)
        filled = [filled[i] for i in sorted(set(idx))]
    return [round(t, 3) for t in filled]


def _read_frames(path, outputs, W, H, expected, cb, cancel, hw) -> np.ndarray:
    args = [ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", *hw, "-i", path, *outputs]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    size = W * H * 3
    out = []
    while True:
        if cancel and cancel():
            p.kill()
            raise Cancelled()
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        out.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
        if cb and expected and len(out) % 8 == 0:
            cb(min(1.0, len(out) / expected))
    err = p.stderr.read().decode(errors="replace")
    if p.wait() != 0 and not out:
        raise RuntimeError(err.strip() or "decode failed")
    return np.stack(out) if out else np.zeros((0, H, W, 3), np.uint8)


def grab_frames(path: str, times: list[float], out_dir: Path, width: int = 640, cb: Progress | None = None,
                cancel=None, duration: float = 0) -> list[str | None]:
    """Save one JPEG per time. Seeks are input-side so each grab is fast."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = [0]

    def one(i_t):
        i, t = i_t
        if cancel and cancel():
            return None
        dst = out_dir / f"{i + 1:04d}.jpg"
        for tt in (t, max(0, t - 0.5), max(0, min(t, duration - 1.5))):
            r = run([ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{tt:.3f}", "-i", path,
                     "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(dst)])
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                break
        done[0] += 1
        if cb:
            cb(done[0] / len(times))
        return dst.name if dst.exists() else None

    with ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        res = list(ex.map(one, enumerate(times)))
    if cancel and cancel():
        raise Cancelled()
    return res


def _video_codec_args() -> list[str]:
    from .hardware import ffmpeg_hw, vaapi_device

    hw = ffmpeg_hw()
    if hw["encoder"] == "h264_vaapi":
        return ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", *hw["encoder_args"]]
    return ["-c:v", hw["encoder"], *hw["encoder_args"], "-pix_fmt", "yuv420p"]


def _pre_args() -> list[str]:
    from .hardware import ffmpeg_hw, vaapi_device

    return ["-vaapi_device", vaapi_device()] if ffmpeg_hw()["encoder"] == "h264_vaapi" else []


def clip_video(path: str, a: float, b: float, out: Path, cb: Progress | None = None):
    """Frame-accurate cut, re-encoded on the hardware encoder."""
    base = [ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", *_pre_args(), "-ss", f"{a:.3f}", "-i", path,
            "-t", f"{b - a:.3f}", "-map", "0:v:0", "-map", "0:a:0?"]
    tail = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    try:
        _run_progress([*base, *_video_codec_args(), *tail], b - a, cb)
    except RuntimeError:
        _run_progress([*base, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", *tail], b - a, cb)


def clip_audio(path: str, a: float, b: float, out: Path, cb: Progress | None = None):
    codec = {".m4a": ["-c:a", "aac", "-b:a", "192k"], ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
             ".wav": ["-c:a", "pcm_s16le"]}[out.suffix]
    _run_progress([ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{a:.3f}", "-i", path,
                   "-t", f"{b - a:.3f}", "-vn", *codec, str(out)], b - a, cb)


def save_frame(path: str, t: float, out: Path):
    r = run([ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", path,
             "-frames:v", "1", str(out)])
    if r.returncode != 0 or not out.exists():
        raise RuntimeError("Couldn’t read this frame")


def make_proxy(path: str, out: Path, duration: float, cb: Progress | None = None, cancel=None, probe: dict | None = None):
    """H.264/AAC copy for formats the player can't decode (HEVC, ProRes, AVI, …); AAC alone for audio files.
    When the video stream is already playable and only the container isn't (MKV, AVI, TS), the
    picture is copied as-is: seconds instead of a full re-encode, and no quality loss."""
    from .hardware import IS_MAC

    tmp = out.with_suffix(".part.mp4")
    vc, ac = (probe or {}).get("vcodec"), (probe or {}).get("acodec")
    if vc == "h264" or (vc == "hevc" and IS_MAC):
        audio = ["-c:a", "copy"] if ac in ("aac", "mp3", None) else ["-c:a", "aac", "-b:a", "160k"]
        tag = ["-tag:v", "hvc1"] if vc == "hevc" else []
        try:
            _run_progress([ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:V:0",
                           "-map", "0:a:0?", "-c:v", "copy", *tag, *audio, "-movflags", "+faststart", str(tmp)],
                          duration, cb, cancel)
            tmp.replace(out)
            return
        except RuntimeError:
            pass
    base = [ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", *_pre_args(), "-i", path,
            "-map", "0:v:0?", "-map", "0:a:0?"]
    tail = ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)]
    try:
        _run_progress([*base, *_video_codec_args(), *tail], duration, cb, cancel)
    except RuntimeError:
        _run_progress([ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:v:0?",
                       "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                       *tail], duration, cb, cancel)
    tmp.replace(out)
