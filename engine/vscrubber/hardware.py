# SPDX-License-Identifier: GPL-3.0-or-later
"""Detects what this machine can accelerate: torch device, MLX, CTranslate2 CUDA,
whisper.cpp, Ollama, and ffmpeg hardware decoders/encoders."""

import functools
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request

IS_MAC = sys.platform == "darwin"
IS_WIN = os.name == "nt"
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WIN else 0


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@functools.lru_cache(None)
def cpu_name() -> str:
    try:
        if IS_MAC:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if IS_WIN:
            return platform.processor() or "CPU"
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "CPU"


@functools.lru_cache(None)
def torch_accel() -> dict:
    """Best torch device: CUDA/ROCm > Intel XPU > Apple MPS > DirectML > CPU."""
    if not has_module("torch"):
        return {"device": "cpu", "backend": "CPU", "name": cpu_name(), "torch": False}
    import torch

    if torch.cuda.is_available():
        hip = getattr(torch.version, "hip", None)
        return {"device": "cuda", "backend": "ROCm" if hip else "CUDA", "name": torch.cuda.get_device_name(0), "torch": True}
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return {"device": "xpu", "backend": "Intel XPU", "name": torch.xpu.get_device_name(0), "torch": True}
    if torch.backends.mps.is_available():
        return {"device": "mps", "backend": "Metal", "name": cpu_name(), "torch": True}
    if has_module("torch_directml"):
        import torch_directml

        if torch_directml.is_available():
            return {"device": "directml", "backend": "DirectML", "name": torch_directml.device_name(0), "torch": True}
    return {"device": "cpu", "backend": "CPU", "name": cpu_name(), "torch": True}


def torch_device():
    acc = torch_accel()
    if acc["device"] == "directml":
        import torch_directml

        return torch_directml.device()
    return acc["device"]


def torch_dtype():
    import torch

    return torch.float16 if torch_accel()["device"] in ("cuda", "mps", "xpu") else torch.float32


@functools.lru_cache(None)
def ct2_cuda() -> bool:
    if not has_module("ctranslate2"):
        return False
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


@functools.lru_cache(None)
def mlx_ok() -> bool:
    return IS_APPLE_SILICON and has_module("mlx_whisper")


@functools.lru_cache(None)
def mlx_vlm_ok() -> bool:
    return IS_APPLE_SILICON and has_module("mlx_vlm")


GB = 1024**3


@functools.lru_cache(None)
def memory() -> dict:
    """RAM, GPU memory, and how much of it the app's models may use at once.

    Apple Silicon shares memory between CPU and GPU; leave room for macOS, the player and the
    video being decoded. Discrete GPUs get most of their VRAM. Without a GPU, models run from RAM."""
    import psutil

    ram = psutil.virtual_memory().total
    acc = torch_accel()
    gpu_mem, unified = None, False
    if IS_APPLE_SILICON:
        unified, gpu_mem = True, ram
        budget = ram * 0.6 - 2 * GB
    elif acc["device"] in ("cuda", "xpu"):
        try:
            import torch

            mod = torch.cuda if acc["device"] == "cuda" else torch.xpu
            gpu_mem = mod.get_device_properties(0).total_memory
        except Exception:
            gpu_mem = None
        budget = (gpu_mem * 0.85 - 0.5 * GB) if gpu_mem else ram * 0.3
    elif acc["device"] == "directml":
        # DirectML doesn't report VRAM; assume a mid-range card or shared memory.
        budget = min(ram * 0.3, 6 * GB)
    else:
        budget = ram * 0.4 - 1 * GB
    return {"ram": ram, "gpuMem": gpu_mem, "unified": unified, "budget": int(max(1.5 * GB, budget)),
            "device": "gpu" if acc["device"] != "cpu" or IS_APPLE_SILICON else "cpu"}


def disk_free() -> int:
    from huggingface_hub.constants import HF_HUB_CACHE

    p = HF_HUB_CACHE
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p or os.path.expanduser("~")).free


@functools.lru_cache(None)
def whisper_cpp_bin() -> str | None:
    env = os.environ.get("VS_WHISPER_CPP")
    if env and os.path.exists(env):
        return env
    for n in ("whisper-cli", "whisper-cpp", "whisper"):
        p = shutil.which(n)
        # `whisper` may be openai-whisper's Python CLI; only accept it if it speaks whisper.cpp flags.
        if p and (n != "whisper" or _is_whisper_cpp(p)):
            return p
    return None


def _is_whisper_cpp(p: str) -> bool:
    try:
        out = subprocess.run([p, "--help"], capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW)
        return "--output-json" in (out.stdout + out.stderr)
    except Exception:
        return False


def ollama_host() -> str:
    h = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    return h if h.startswith("http") else "http://" + h


def ollama_models() -> list[str] | None:
    """Installed Ollama model tags, or None when Ollama isn't running."""
    try:
        with urllib.request.urlopen(ollama_host() + "/api/tags", timeout=1.5) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return None


# ffmpeg hardware encoders, in preference order, with quality-oriented args.
ENCODERS = [
    ("h264_videotoolbox", ["-b:v", "12M", "-allow_sw", "1"]),
    ("h264_nvenc", ["-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]),
    ("h264_qsv", ["-global_quality", "20"]),
    ("h264_amf", ["-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20"]),
    ("h264_vaapi", ["-qp", "20"]),
    ("h264_mf", ["-b:v", "12M", "-hw_encoding", "1"]),
]
SOFTWARE_ENCODER = ("libx264", ["-preset", "veryfast", "-crf", "18"])


@functools.lru_cache(None)
def ffmpeg_hw() -> dict:
    from .media import ffmpeg

    ff = ffmpeg()
    try:
        out = subprocess.run([ff, "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW).stdout
        hwaccels = [l.strip() for l in out.splitlines()[1:] if l.strip()]
    except Exception:
        hwaccels = []
    encoder = None
    for name, args in ENCODERS:
        if _encoder_works(ff, name):
            encoder = (name, args)
            break
    if not encoder:
        encoder = SOFTWARE_ENCODER if _encoder_works(ff, "libx264") else ("mpeg4", ["-q:v", "3"])
    return {"hwaccels": hwaccels, "encoder": encoder[0], "encoder_args": encoder[1]}


def vaapi_device() -> str | None:
    p = "/dev/dri/renderD128"
    return p if os.path.exists(p) else None


def _encoder_works(ff: str, name: str) -> bool:
    pre, vf = [], []
    if name == "h264_vaapi":
        dev = vaapi_device()
        if not dev:
            return False
        pre, vf = ["-vaapi_device", dev], ["-vf", "format=nv12,hwupload"]
    cmd = [ff, "-hide_banner", "-loglevel", "error", *pre, "-f", "lavfi", "-i", "color=black:s=256x144:d=0.2", *vf, "-c:v", name, "-frames:v", "3", "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=15, creationflags=NO_WINDOW).returncode == 0
    except Exception:
        return False


def summary() -> dict:
    acc = torch_accel()
    ff = ffmpeg_hw()
    oll = ollama_models()
    if mlx_ok():
        speech_accel = "Metal (MLX)"
    elif ct2_cuda():
        speech_accel = "CUDA"
    elif acc["backend"] != "CPU":
        speech_accel = acc["backend"]
    elif whisper_cpp_bin():
        speech_accel = "whisper.cpp"
    else:
        speech_accel = "CPU"
    return {
        "os": {"darwin": "macOS", "win32": "Windows"}.get(sys.platform, "Linux"),
        "arch": platform.machine(),
        "cpu": cpu_name(),
        "cores": os.cpu_count(),
        "gpu": acc["name"] if acc["backend"] != "CPU" else None,
        "torch": acc,
        "speechAccel": speech_accel,
        "mlx": mlx_ok(),
        "ct2Cuda": ct2_cuda(),
        "whisperCpp": whisper_cpp_bin(),
        "ollama": oll is not None,
        "ollamaModels": oll or [],
        "mlxVlm": mlx_vlm_ok(),
        "memory": memory(),
        "diskFree": disk_free(),
        "ffmpegHwaccels": ff["hwaccels"],
        "videoEncoder": ff["encoder"],
        "label": f"{acc['name'] if acc['backend'] != 'CPU' else cpu_name()} · {acc['backend'] if acc['backend'] != 'CPU' else speech_accel}",
    }
