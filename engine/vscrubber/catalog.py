# SPDX-License-Identifier: GPL-3.0-or-later
"""Every local model the app can use, the Hugging Face repo each runtime loads, and a
recommender that picks the best set this machine can hold in memory.

Kinds:
  speech   – speech-to-text (Whisper family, Distil-Whisper, Moonshine)
  vision   – image/text embedding for picture search (SigLIP, CLIP)
  caption  – vision-language model that describes each key frame (Qwen-VL, InternVL, SmolVLM)
  text     – sentence embeddings for meaning-based transcript search

Per model:
  params   parameter count, used to estimate memory for the runtime that will load it
  quality  relative accuracy within its kind, 0–100 (curated ranking, not a benchmark score)
  speed    relative throughput within its kind, 1 (slow) – 5 (fast)
  license  license of the weights, as published on Hugging Face
"""

import itertools
import json
import threading

# Speech runtimes: mlx (Apple Metal), faster-whisper (CTranslate2: CUDA or CPU int8),
# whisper.cpp (Metal / CUDA / Vulkan / CPU via the whisper-cli binary), transformers (torch device).
SPEECH = [
    {"id": "whisper-large-v3-turbo", "label": "Whisper large-v3 turbo", "params": 809e6, "quality": 92, "speed": 4,
     "license": "MIT", "note": "Best speed/accuracy balance",
     "mlx": "mlx-community/whisper-large-v3-turbo", "faster-whisper": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-large-v3-turbo.bin"), "transformers": "openai/whisper-large-v3-turbo"},
    {"id": "whisper-large-v3", "label": "Whisper large-v3", "params": 1550e6, "quality": 95, "speed": 1,
     "license": "MIT", "note": "Most accurate, slowest",
     "mlx": "mlx-community/whisper-large-v3-mlx", "faster-whisper": "Systran/faster-whisper-large-v3",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-large-v3.bin"), "transformers": "openai/whisper-large-v3"},
    {"id": "distil-large-v3.5", "label": "Distil-Whisper large-v3.5", "params": 756e6, "quality": 91, "speed": 5,
     "license": "MIT", "english": True, "note": "Near large-v3 accuracy in English, very fast",
     "faster-whisper": "distil-whisper/distil-large-v3.5-ct2", "transformers": "distil-whisper/distil-large-v3.5"},
    {"id": "distil-large-v3", "label": "Distil-Whisper large-v3", "params": 756e6, "quality": 88, "speed": 5,
     "license": "MIT", "english": True, "note": "Fast English transcription",
     "mlx": "mlx-community/distil-whisper-large-v3", "faster-whisper": "Systran/faster-distil-whisper-large-v3",
     "whisper.cpp": ("distil-whisper/distil-large-v3-ggml", "ggml-distil-large-v3.bin"), "transformers": "distil-whisper/distil-large-v3"},
    {"id": "whisper-medium", "label": "Whisper medium", "params": 769e6, "quality": 82, "speed": 2,
     "license": "MIT", "note": "Good multilingual accuracy",
     "mlx": "mlx-community/whisper-medium-mlx", "faster-whisper": "Systran/faster-whisper-medium",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-medium.bin"), "transformers": "openai/whisper-medium"},
    {"id": "whisper-medium.en", "label": "Whisper medium.en", "params": 769e6, "quality": 84, "speed": 2,
     "license": "MIT", "english": True, "note": "English-tuned medium",
     "mlx": "mlx-community/whisper-medium.en-mlx", "faster-whisper": "Systran/faster-whisper-medium.en",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-medium.en.bin"), "transformers": "openai/whisper-medium.en"},
    {"id": "whisper-small", "label": "Whisper small", "params": 244e6, "quality": 72, "speed": 4,
     "license": "MIT", "note": "Fast on CPU, decent accuracy",
     "mlx": "mlx-community/whisper-small-mlx", "faster-whisper": "Systran/faster-whisper-small",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-small.bin"), "transformers": "openai/whisper-small"},
    {"id": "distil-small.en", "label": "Distil-Whisper small.en", "params": 166e6, "quality": 70, "speed": 5,
     "license": "MIT", "english": True, "note": "Very fast English on CPU",
     "faster-whisper": "Systran/faster-distil-whisper-small.en", "transformers": "distil-whisper/distil-small.en"},
    {"id": "whisper-base", "label": "Whisper base", "params": 74e6, "quality": 58, "speed": 5,
     "license": "MIT", "note": "Quick drafts",
     "mlx": "mlx-community/whisper-base-mlx", "faster-whisper": "Systran/faster-whisper-base",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-base.bin"), "transformers": "openai/whisper-base"},
    {"id": "whisper-tiny", "label": "Whisper tiny", "params": 39e6, "quality": 45, "speed": 5,
     "license": "MIT", "note": "Fastest, roughest",
     "mlx": "mlx-community/whisper-tiny-mlx", "faster-whisper": "Systran/faster-whisper-tiny",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-tiny.bin"), "transformers": "openai/whisper-tiny"},
    {"id": "moonshine-base", "label": "Moonshine base", "params": 61e6, "quality": 62, "speed": 5,
     "license": "MIT", "english": True, "note": "Built for edge devices",
     "transformers": "UsefulSensors/moonshine-base"},
    {"id": "moonshine-tiny", "label": "Moonshine tiny", "params": 27e6, "quality": 50, "speed": 5,
     "license": "MIT", "english": True, "note": "Smallest English model",
     "transformers": "UsefulSensors/moonshine-tiny"},
]

# Picture search: contrastive image/text models. `calib` maps raw similarity to 0–1 relevance.
VISION = [
    {"id": "siglip-so400m", "label": "SigLIP so400m 384", "params": 878e6, "quality": 92, "speed": 2, "license": "Apache-2.0",
     "transformers": "google/siglip-so400m-patch14-384", "note": "Best picture search", "sigmoid": True},
    {"id": "siglip2-base", "label": "SigLIP 2 base 224", "params": 375e6, "quality": 84, "speed": 4, "license": "Apache-2.0",
     "transformers": "google/siglip2-base-patch16-224", "note": "Newer SigLIP, multilingual queries", "sigmoid": True},
    {"id": "siglip-base", "label": "SigLIP base 224", "params": 203e6, "quality": 80, "speed": 4, "license": "Apache-2.0",
     "transformers": "google/siglip-base-patch16-224", "note": "Accurate and light", "sigmoid": True},
    {"id": "clip-vit-l14", "label": "CLIP ViT-L/14", "params": 428e6, "quality": 76, "speed": 3, "license": "MIT",
     "transformers": "openai/clip-vit-large-patch14", "note": "Classic high-quality CLIP", "calib": (0.17, 0.29)},
    {"id": "clip-vit-b32", "label": "CLIP ViT-B/32", "params": 151e6, "quality": 62, "speed": 5, "license": "MIT",
     "transformers": "openai/clip-vit-base-patch32", "note": "Fastest, good for long videos on CPU", "calib": (0.19, 0.31)},
]

# Frame descriptions. Hugging Face models run on MLX (4-bit, Apple Silicon) or transformers on the
# torch device. Ollama models run in Ollama, which accelerates on Metal, CUDA, ROCm and Vulkan.
CAPTION = [
    {"id": "none", "label": "Off", "params": 0, "quality": 0, "speed": 5, "note": "Picture search uses SigLIP/CLIP only"},
    {"id": "qwen3-vl-8b", "label": "Qwen3-VL 8B", "params": 8.8e9, "quality": 94, "speed": 2, "license": "Apache-2.0",
     "note": "Best descriptions and on-screen text reading",
     "mlx": "mlx-community/Qwen3-VL-8B-Instruct-4bit", "transformers": "Qwen/Qwen3-VL-8B-Instruct"},
    {"id": "qwen2.5-vl-7b", "label": "Qwen2.5-VL 7B", "params": 8.3e9, "quality": 88, "speed": 2, "license": "Apache-2.0",
     "note": "Strong descriptions, widely tested",
     "mlx": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit", "transformers": "Qwen/Qwen2.5-VL-7B-Instruct"},
    {"id": "internvl3-8b", "label": "InternVL3 8B", "params": 7.9e9, "quality": 87, "speed": 2, "license": "Qwen (other)",
     "note": "Detailed scenes, charts and documents", "transformers": "OpenGVLab/InternVL3-8B-hf"},
    {"id": "qwen3-vl-4b", "label": "Qwen3-VL 4B", "params": 4.4e9, "quality": 87, "speed": 3, "license": "Apache-2.0",
     "note": "Near-8B quality at half the memory",
     "mlx": "mlx-community/Qwen3-VL-4B-Instruct-4bit", "transformers": "Qwen/Qwen3-VL-4B-Instruct"},
    {"id": "qwen3-vl-2b", "label": "Qwen3-VL 2B", "params": 2.1e9, "quality": 78, "speed": 4, "license": "Apache-2.0",
     "note": "Accurate and light, reads on-screen text",
     "mlx": "mlx-community/Qwen3-VL-2B-Instruct-4bit", "transformers": "Qwen/Qwen3-VL-2B-Instruct"},
    {"id": "internvl3-2b", "label": "InternVL3 2B", "params": 2.1e9, "quality": 72, "speed": 3, "license": "Qwen (other)",
     "note": "Compact, good with slides", "transformers": "OpenGVLab/InternVL3-2B-hf"},
    {"id": "smolvlm2-2.2b", "label": "SmolVLM2 2.2B", "params": 2.2e9, "quality": 66, "speed": 3, "license": "Apache-2.0",
     "note": "Fully open data and weights",
     "mlx": "mlx-community/SmolVLM2-2.2B-Instruct-mlx", "transformers": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"},
    {"id": "internvl3-1b", "label": "InternVL3 1B", "params": 0.94e9, "quality": 60, "speed": 4, "license": "Qwen (other)",
     "note": "Small, reads text well", "transformers": "OpenGVLab/InternVL3-1B-hf"},
    {"id": "smolvlm-500m", "label": "SmolVLM 500M", "params": 0.51e9, "quality": 50, "speed": 5, "license": "Apache-2.0",
     "note": "Runs anywhere", "transformers": "HuggingFaceTB/SmolVLM-500M-Instruct"},
    {"id": "smolvlm-256m", "label": "SmolVLM 256M", "params": 0.256e9, "quality": 40, "speed": 5, "license": "Apache-2.0",
     "note": "Tiny, for older machines", "transformers": "HuggingFaceTB/SmolVLM-256M-Instruct"},
    {"id": "ollama:qwen2.5vl:7b", "label": "Qwen2.5-VL 7B", "ollama": "qwen2.5vl:7b", "note": "Via Ollama"},
    {"id": "ollama:gemma3:4b", "label": "Gemma 3 4B", "ollama": "gemma3:4b", "note": "Via Ollama"},
    {"id": "ollama:gemma3:12b", "label": "Gemma 3 12B", "ollama": "gemma3:12b", "note": "Via Ollama"},
    {"id": "ollama:minicpm-v:8b", "label": "MiniCPM-V 8B", "ollama": "minicpm-v:8b", "note": "Via Ollama"},
    {"id": "ollama:llama3.2-vision:11b", "label": "Llama 3.2 Vision 11B", "ollama": "llama3.2-vision:11b", "note": "Via Ollama"},
    {"id": "ollama:llava:7b", "label": "LLaVA 1.6 7B", "ollama": "llava:7b", "note": "Via Ollama"},
    {"id": "ollama:moondream:1.8b", "label": "Moondream 2", "ollama": "moondream:1.8b", "note": "Via Ollama"},
]

# Sentence embeddings for transcript and caption search. `calib` maps cosine to 0–1 relevance;
# `q`/`d` are the query/document prefixes some models expect.
TEXT = [
    {"id": "bge-m3", "label": "BGE-M3", "params": 568e6, "quality": 90, "speed": 2, "license": "MIT", "multilingual": True,
     "transformers": "BAAI/bge-m3", "note": "Best multilingual", "calib": (0.35, 0.6)},
    {"id": "mxbai-embed-large-v1", "label": "mxbai embed large v1", "params": 335e6, "quality": 88, "speed": 3, "license": "Apache-2.0",
     "transformers": "mixedbread-ai/mxbai-embed-large-v1", "note": "Top-tier English retrieval", "calib": (0.4, 0.65),
     "q": "Represent this sentence for searching relevant passages: "},
    {"id": "bge-base-en-v1.5", "label": "BGE base en v1.5", "params": 109e6, "quality": 84, "speed": 4, "license": "MIT",
     "transformers": "BAAI/bge-base-en-v1.5", "note": "Higher-quality English", "calib": (0.45, 0.65),
     "q": "Represent this sentence for searching relevant passages: "},
    {"id": "nomic-embed-text-v1.5", "label": "Nomic Embed v1.5", "params": 137e6, "quality": 82, "speed": 4, "license": "Apache-2.0",
     "transformers": "nomic-ai/nomic-embed-text-v1.5", "note": "Long context English", "calib": (0.4, 0.65),
     "q": "search_query: ", "d": "search_document: ", "remote_code": True},
    {"id": "bge-small-en-v1.5", "label": "BGE small en v1.5", "params": 33e6, "quality": 78, "speed": 5, "license": "MIT",
     "transformers": "BAAI/bge-small-en-v1.5", "note": "Fast and accurate English", "calib": (0.45, 0.65),
     "q": "Represent this sentence for searching relevant passages: "},
    {"id": "multilingual-e5-small", "label": "Multilingual E5 small", "params": 118e6, "quality": 72, "speed": 5, "license": "MIT",
     "multilingual": True, "transformers": "intfloat/multilingual-e5-small", "note": "100 languages, small",
     "calib": (0.78, 0.88), "q": "query: ", "d": "passage: "},
    {"id": "all-minilm-l6-v2", "label": "MiniLM L6 v2", "params": 22e6, "quality": 62, "speed": 5, "license": "Apache-2.0",
     "transformers": "sentence-transformers/all-MiniLM-L6-v2", "note": "Fastest", "calib": (0.2, 0.5)},
]

KINDS = {"speech": SPEECH, "vision": VISION, "caption": CAPTION, "text": TEXT}
SPEECH_ENGINES = ["mlx", "faster-whisper", "whisper.cpp", "transformers"]


# Models added by repo id from the Models dialog, kept in custom-models.json.
_custom: dict[str, list[dict]] | None = None
_custom_lock = threading.Lock()


def _custom_file():
    from .paths import data_dir

    return data_dir() / "custom-models.json"


def _customs() -> dict[str, list[dict]]:
    global _custom
    if _custom is None:
        try:
            _custom = json.loads(_custom_file().read_text())
        except (OSError, ValueError):
            _custom = {}
    return _custom


def models(kind: str) -> list[dict]:
    """Built-in models of `kind`, then the ones the user added."""
    return KINDS[kind] + _customs().get(kind, [])


def add_custom(kind: str, model: dict) -> dict:
    with _custom_lock:
        rows = [m for m in _customs().setdefault(kind, []) if m["id"] != model["id"]]
        _customs()[kind] = rows + [{**model, "custom": True}]
        _custom_file().write_text(json.dumps(_customs(), indent=2))
    return model


def remove_custom(kind: str, model_id: str):
    with _custom_lock:
        _customs()[kind] = [m for m in _customs().get(kind, []) if m["id"] != model_id]
        _custom_file().write_text(json.dumps(_customs(), indent=2))


def get(kind: str, model_id: str) -> dict:
    for m in models(kind):
        if m["id"] == model_id:
            return m
    raise KeyError(f"Unknown {kind} model: {model_id}")


# ---------------------------------------------------------------- runtimes


def speech_engine_for(model: dict, preferred: str = "auto") -> str:
    """Pick the fastest engine this machine has that can run `model`."""
    from . import hardware as hw

    if preferred != "auto" and preferred in model and _engine_available(preferred):
        return preferred
    order = []
    if hw.mlx_ok():
        order.append("mlx")
    if hw.ct2_cuda():
        order.append("faster-whisper")
    acc = hw.torch_accel()["backend"]
    if acc in ("ROCm", "Intel XPU", "DirectML", "CUDA"):
        order.append("transformers")
    if hw.whisper_cpp_bin():
        order.append("whisper.cpp")
    order += ["faster-whisper", "transformers"]
    for e in order:
        if e in model and _engine_available(e):
            return e
    raise RuntimeError(f"No installed engine can run {model['id']}")


def engine_for(kind: str, model: dict, preferred: str = "auto") -> str | None:
    """Runtime that will load this model here: mlx, faster-whisper, whisper.cpp, transformers or ollama."""
    from . import hardware as hw

    if kind == "speech":
        try:
            return speech_engine_for(model, preferred)
        except RuntimeError:
            return None
    if model.get("ollama"):
        return "ollama"
    if model["id"] == "none":
        return "none"
    if kind == "caption" and "mlx" in model and hw.mlx_vlm_ok():
        return "mlx"
    return "transformers" if _engine_available("transformers") else None


def repo_for(kind: str, model: dict, engine: str):
    target = model.get(engine) if engine in ("mlx", "faster-whisper", "whisper.cpp", "transformers") else None
    return tuple(target) if isinstance(target, list) else target  # (repo, file) pairs come back from JSON as lists


def _engine_available(engine: str) -> bool:
    from . import hardware as hw

    return {
        "mlx": hw.mlx_ok,
        "faster-whisper": lambda: hw.has_module("faster_whisper"),
        "whisper.cpp": lambda: bool(hw.whisper_cpp_bin()),
        "transformers": lambda: hw.has_module("transformers") and hw.has_module("torch"),
    }[engine]()


# ---------------------------------------------------------------- memory


GB = 1024**3
OVERHEAD = {"speech": 0.4 * GB, "vision": 0.3 * GB, "caption": 1.0 * GB, "text": 0.2 * GB}


def bytes_per_param(kind: str, engine: str) -> float:
    """How many bytes each weight takes once loaded by `engine` on this machine."""
    from . import hardware as hw

    gpu = hw.torch_accel()["device"] in ("cuda", "mps", "xpu")
    if engine == "mlx":
        return 0.75 if kind == "caption" else 2.0  # 4-bit VLMs, fp16 Whisper
    if engine == "faster-whisper":
        return 2.0 if hw.ct2_cuda() else 1.0  # fp16 on CUDA, int8 on CPU
    if engine == "whisper.cpp":
        return 2.0
    if kind == "text":
        return 4.0  # sentence-transformers keeps fp32
    return 2.0 if gpu else 4.0


def memory_needed(kind: str, model: dict, engine: str | None) -> int:
    if not engine or engine in ("none", "ollama"):
        return 0
    if model.get("bytes"):  # added models: size the weights on disk instead of guessing the precision
        return int(model["bytes"] * 1.2 + OVERHEAD[kind])
    if not model.get("params"):
        return 0
    return int(model["params"] * bytes_per_param(kind, engine) * 1.15 + OVERHEAD[kind])


# Largest model per kind worth running without a GPU: bigger ones fit in RAM but index too slowly.
CPU_PARAM_CAP = {"speech": 300e6, "vision": 250e6, "caption": 600e6, "text": 150e6}
WEIGHTS = {"speech": 0.35, "caption": 0.35, "vision": 0.15, "text": 0.15}


def on_gpu(engine: str) -> bool:
    from . import hardware as hw

    if engine == "mlx":
        return True
    if engine == "faster-whisper":
        return hw.ct2_cuda()
    if engine == "whisper.cpp":
        return hw.IS_MAC  # Homebrew and release builds use Metal; elsewhere assume CPU
    return hw.torch_accel()["device"] != "cpu"


def effective_speed(model: dict, engine: str) -> int:
    """A model's speed on this machine: CPU-bound runtimes lose ground when a GPU sits idle."""
    from . import hardware as hw

    if hw.memory()["device"] == "gpu" and not on_gpu(engine):
        return max(1, model["speed"] - 3)
    return model["speed"]


def candidates(kind: str, language: str) -> list[tuple[dict, str, int]]:
    """Models of `kind` this machine can run for `language`, as (model, engine, bytes)."""
    from . import hardware as hw

    cpu = hw.memory()["device"] == "cpu"
    out = []
    for m in KINDS[kind]:
        if m.get("ollama") or m["id"] == "none":
            continue
        if m.get("english") and language not in ("en",):
            continue
        if kind == "text" and language not in ("auto", "en") and not m.get("multilingual"):
            continue
        if cpu and m["params"] > CPU_PARAM_CAP[kind]:
            continue
        e = engine_for(kind, m)
        if e:
            out.append((m, e, memory_needed(kind, m, e)))
    return out


def recommend(language: str = "auto", budget: int | None = None) -> dict:
    """Best model per kind whose combined memory fits the budget.

    Each kind's model stays loaded while videos index, so the set has to fit at once. Scores weigh
    quality first and speed second; the search is exhaustive (a few thousand combinations)."""
    from . import hardware as hw

    budget = budget if budget is not None else hw.memory()["budget"]
    kinds = ["speech", "caption", "vision", "text"]
    pools = {k: candidates(k, language) for k in kinds}
    best, best_score = None, -1.0
    for combo in itertools.product(*(pools[k] for k in kinds)):
        mem = sum(c[2] for c in combo)
        if mem > budget:
            continue
        score = sum(WEIGHTS[k] * (c[0]["quality"] + 2 * effective_speed(c[0], c[1])) for k, c in zip(kinds, combo)) - mem / GB * 0.01
        if score > best_score:
            best, best_score = combo, score
    if best is None:
        # Nothing fits: take the smallest of each kind.
        best = tuple(min(pools[k], key=lambda c: c[2]) if pools[k] else None for k in kinds)
    out = {k: (c[0]["id"] if c else fallback) for k, c, fallback in zip(kinds, best, ["whisper-base", "none", "clip-vit-b32", "all-minilm-l6-v2"])}
    return out


def defaults() -> dict:
    return recommend("auto")
