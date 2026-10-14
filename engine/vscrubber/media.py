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


def scan_scenes(path: str, duration: float, cb: Progress | None = None, cancel=None,
                fps: float = 2.0, min_len: float = 2.0, max_gap: float | None = None) -> list[float]:
    """Return scene start times. Decodes a tiny 64×36 stream (hardware-decoded when possible),
    scores each frame against the previous one, and cuts where the change spikes."""
    W, H = 64, 36
    max_gap = max_gap or max(30.0, duration / 150)
    vf = f"fps={fps},scale={W}:{H}:flags=area,format=rgb24"
    frames = None
    for hw in (["-hwaccel", "auto"], []):
        try:
            frames = _read_frames(path, vf, W, H, duration * fps, cb, cancel, hw)
            break
        except Cancelled:
            raise
        except RuntimeError:
            continue
    if frames is None or len(frames) < 2:
        return [0.0]
    a = frames.astype(np.float32) / 255.0
    pix = np.abs(a[1:] - a[:-1]).mean(axis=(1, 2, 3))
    hists = np.stack([np.concatenate([np.histogram(f[..., c], bins=16, range=(0, 1))[0] for c in range(3)]) for f in a]).astype(np.float32)
    hists /= hists.sum(axis=1, keepdims=True)
    hd = np.abs(hists[1:] - hists[:-1]).sum(axis=1) / 2
    score = 0.5 * np.minimum(1, pix * 3) + 0.5 * hd
    # A cut has to spike above its neighbourhood, so constant motion (pans, zooms) doesn't read as cuts
    # while a hard cut out of a still shot does.
    cuts = [0.0]
    n = len(score)
    for i, s in enumerate(score):
        lo, hi = max(0, i - 6), min(n, i + 7)
        neigh = np.delete(score[lo:hi], i - lo)
        base = float(np.median(neigh)) if len(neigh) else 0.0
        t = (i + 1) / fps
        if s > max(0.2, base * 2.5 + 0.08) and t - cuts[-1] >= min_len and duration - t >= 1.0:
            cuts.append(t)
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


def _read_frames(path, vf, W, H, expected, cb, cancel, hw) -> np.ndarray:
    args = [ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", *hw, "-i", path, "-vf", vf, "-an", "-f", "rawvideo", "-"]
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


def make_proxy(path: str, out: Path, duration: float, cb: Progress | None = None, cancel=None):
    """H.264/AAC copy for formats the player can't decode (HEVC, ProRes, AVI, …)."""
    tmp = out.with_suffix(".part.mp4")
    base = [ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-hwaccel", "auto", *_pre_args(), "-i", path,
            "-map", "0:v:0", "-map", "0:a:0?"]
    tail = ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)]
    try:
        _run_progress([*base, *_video_codec_args(), *tail], duration, cb, cancel)
    except RuntimeError:
        _run_progress([ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:v:0",
                       "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                       *tail], duration, cb, cancel)
    tmp.replace(out)
