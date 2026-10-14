"""Every local model the app can use, grouped by job, with the repo each engine loads.

Kinds:
  speech   – speech-to-text (Whisper family, Distil-Whisper, Moonshine)
  vision   – image/text embedding for picture search (SigLIP, CLIP)
  caption  – vision-language model that describes each key frame (SmolVLM, Ollama VLMs)
  text     – sentence embeddings for meaning-based transcript search
"""

# Speech engines: mlx (Apple Metal), faster-whisper (CTranslate2: CUDA or CPU int8),
# whisper.cpp (Metal / CUDA / Vulkan / CPU via the whisper-cli binary), transformers (torch device).
SPEECH = [
    {"id": "whisper-large-v3-turbo", "label": "Whisper large-v3 turbo", "size": 1.6e9, "langs": "99 languages",
     "note": "Best speed/accuracy balance on a GPU",
     "mlx": "mlx-community/whisper-large-v3-turbo", "faster-whisper": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-large-v3-turbo.bin"), "transformers": "openai/whisper-large-v3-turbo"},
    {"id": "whisper-large-v3", "label": "Whisper large-v3", "size": 3.1e9, "langs": "99 languages",
     "note": "Most accurate, slowest",
     "mlx": "mlx-community/whisper-large-v3-mlx", "faster-whisper": "Systran/faster-whisper-large-v3",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-large-v3.bin"), "transformers": "openai/whisper-large-v3"},
    {"id": "distil-large-v3.5", "label": "Distil-Whisper large-v3.5", "size": 1.5e9, "langs": "English",
     "note": "Near large-v3 accuracy, ~1.5× faster than turbo",
     "faster-whisper": "distil-whisper/distil-large-v3.5-ct2", "transformers": "distil-whisper/distil-large-v3.5"},
    {"id": "distil-large-v3", "label": "Distil-Whisper large-v3", "size": 1.5e9, "langs": "English",
     "note": "Fast English transcription",
     "mlx": "mlx-community/distil-whisper-large-v3", "faster-whisper": "Systran/faster-distil-whisper-large-v3",
     "whisper.cpp": ("distil-whisper/distil-large-v3-ggml", "ggml-distil-large-v3.bin"), "transformers": "distil-whisper/distil-large-v3"},
    {"id": "whisper-medium", "label": "Whisper medium", "size": 1.5e9, "langs": "99 languages",
     "note": "Good multilingual accuracy",
     "mlx": "mlx-community/whisper-medium-mlx", "faster-whisper": "Systran/faster-whisper-medium",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-medium.bin"), "transformers": "openai/whisper-medium"},
    {"id": "whisper-medium.en", "label": "Whisper medium.en", "size": 1.5e9, "langs": "English",
     "note": "English-tuned medium",
     "mlx": "mlx-community/whisper-medium.en-mlx", "faster-whisper": "Systran/faster-whisper-medium.en",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-medium.en.bin"), "transformers": "openai/whisper-medium.en"},
    {"id": "whisper-small", "label": "Whisper small", "size": 4.8e8, "langs": "99 languages",
     "note": "Fast on CPU, decent accuracy",
     "mlx": "mlx-community/whisper-small-mlx", "faster-whisper": "Systran/faster-whisper-small",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-small.bin"), "transformers": "openai/whisper-small"},
    {"id": "distil-small.en", "label": "Distil-Whisper small.en", "size": 3.3e8, "langs": "English",
     "note": "Very fast English on CPU",
     "faster-whisper": "Systran/faster-distil-whisper-small.en", "transformers": "distil-whisper/distil-small.en"},
    {"id": "whisper-base", "label": "Whisper base", "size": 1.5e8, "langs": "99 languages",
     "note": "Quick drafts",
     "mlx": "mlx-community/whisper-base-mlx", "faster-whisper": "Systran/faster-whisper-base",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-base.bin"), "transformers": "openai/whisper-base"},
    {"id": "whisper-tiny", "label": "Whisper tiny", "size": 7.5e7, "langs": "99 languages",
     "note": "Fastest, roughest",
     "mlx": "mlx-community/whisper-tiny-mlx", "faster-whisper": "Systran/faster-whisper-tiny",
     "whisper.cpp": ("ggerganov/whisper.cpp", "ggml-tiny.bin"), "transformers": "openai/whisper-tiny"},
    {"id": "moonshine-base", "label": "Moonshine base", "size": 2.4e8, "langs": "English",
     "note": "Built for edge devices, low latency",
     "transformers": "UsefulSensors/moonshine-base"},
    {"id": "moonshine-tiny", "label": "Moonshine tiny", "size": 1.1e8, "langs": "English",
     "note": "Smallest English model",
     "transformers": "UsefulSensors/moonshine-tiny"},
]

# Picture search: contrastive image/text models. `calib` maps raw similarity to 0–1 relevance.
VISION = [
    {"id": "siglip-base", "label": "SigLIP base 224", "size": 8.1e8, "repo": "google/siglip-base-patch16-224",
     "note": "Accurate picture search, light enough for any machine", "sigmoid": True},
    {"id": "siglip2-base", "label": "SigLIP 2 base 224", "size": 1.5e9, "repo": "google/siglip2-base-patch16-224",
     "note": "Newer SigLIP, multilingual queries", "sigmoid": True},
    {"id": "siglip-so400m", "label": "SigLIP so400m 384", "size": 3.5e9, "repo": "google/siglip-so400m-patch14-384",
     "note": "Best picture search, wants a GPU", "sigmoid": True},
    {"id": "clip-vit-b32", "label": "CLIP ViT-B/32", "size": 6.1e8, "repo": "openai/clip-vit-base-patch32",
     "note": "Fastest, good for long videos on CPU", "calib": (0.19, 0.31)},
    {"id": "clip-vit-l14", "label": "CLIP ViT-L/14", "size": 1.7e9, "repo": "openai/clip-vit-large-patch14",
     "note": "Classic high-quality CLIP", "calib": (0.17, 0.29)},
]

# Frame descriptions. Built-in models run through transformers on the torch device;
# Ollama models run in Ollama, which accelerates on Metal, CUDA, ROCm and Vulkan.
CAPTION = [
    {"id": "none", "label": "Off", "size": 0, "note": "Picture search uses SigLIP/CLIP only"},
    {"id": "smolvlm-256m", "label": "SmolVLM 256M", "size": 5.1e8, "repo": "HuggingFaceTB/SmolVLM-256M-Instruct",
     "note": "Built in. Tiny, runs anywhere"},
    {"id": "smolvlm-500m", "label": "SmolVLM 500M", "size": 1.0e9, "repo": "HuggingFaceTB/SmolVLM-500M-Instruct",
     "note": "Built in. Better detail and on-screen text"},
    {"id": "smolvlm2-2.2b", "label": "SmolVLM2 2.2B", "size": 4.5e9, "repo": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
     "note": "Built in. Strongest built-in describer, wants a GPU"},
    {"id": "ollama:qwen2.5vl:7b", "label": "Qwen2.5-VL 7B", "size": 6.0e9, "ollama": "qwen2.5vl:7b",
     "note": "Via Ollama. Excellent descriptions and text reading"},
    {"id": "ollama:qwen2.5vl:3b", "label": "Qwen2.5-VL 3B", "size": 3.2e9, "ollama": "qwen2.5vl:3b",
     "note": "Via Ollama. Lighter Qwen vision"},
    {"id": "ollama:qwen3-vl:8b", "label": "Qwen3-VL 8B", "size": 6.1e9, "ollama": "qwen3-vl:8b",
     "note": "Via Ollama. Newest Qwen vision"},
    {"id": "ollama:qwen3-vl:4b", "label": "Qwen3-VL 4B", "size": 3.3e9, "ollama": "qwen3-vl:4b",
     "note": "Via Ollama. Fast Qwen3 vision"},
    {"id": "ollama:gemma3:4b", "label": "Gemma 3 4B", "size": 3.3e9, "ollama": "gemma3:4b",
     "note": "Via Ollama. Balanced, multilingual"},
    {"id": "ollama:minicpm-v:8b", "label": "MiniCPM-V 8B", "size": 5.5e9, "ollama": "minicpm-v:8b",
     "note": "Via Ollama. Strong OCR"},
    {"id": "ollama:llama3.2-vision:11b", "label": "Llama 3.2 Vision 11B", "size": 7.8e9, "ollama": "llama3.2-vision:11b",
     "note": "Via Ollama. Large, detailed"},
    {"id": "ollama:llava:7b", "label": "LLaVA 1.6 7B", "size": 4.7e9, "ollama": "llava:7b",
     "note": "Via Ollama. Widely compatible"},
    {"id": "ollama:granite3.2-vision:2b", "label": "Granite 3.2 Vision 2B", "size": 2.4e9, "ollama": "granite3.2-vision:2b",
     "note": "Via Ollama. Documents, charts and slides"},
    {"id": "ollama:moondream:1.8b", "label": "Moondream 2", "size": 1.7e9, "ollama": "moondream:1.8b",
     "note": "Via Ollama. Tiny and quick"},
]

# Sentence embeddings for transcript and caption search. `calib` maps cosine to 0–1 relevance;
# `q`/`d` are the query/document prefixes some models expect.
TEXT = [
    {"id": "bge-small-en-v1.5", "label": "BGE small en v1.5", "size": 1.3e8, "repo": "BAAI/bge-small-en-v1.5",
     "note": "Fast and accurate English", "calib": (0.45, 0.65),
     "q": "Represent this sentence for searching relevant passages: "},
    {"id": "all-minilm-l6-v2", "label": "MiniLM L6 v2", "size": 9.1e7, "repo": "sentence-transformers/all-MiniLM-L6-v2",
     "note": "Fastest", "calib": (0.2, 0.5)},
    {"id": "bge-base-en-v1.5", "label": "BGE base en v1.5", "size": 4.4e8, "repo": "BAAI/bge-base-en-v1.5",
     "note": "Higher-quality English", "calib": (0.45, 0.65),
     "q": "Represent this sentence for searching relevant passages: "},
    {"id": "multilingual-e5-small", "label": "Multilingual E5 small", "size": 4.7e8, "repo": "intfloat/multilingual-e5-small",
     "note": "100 languages, small", "calib": (0.78, 0.88), "q": "query: ", "d": "passage: "},
    {"id": "bge-m3", "label": "BGE-M3", "size": 2.3e9, "repo": "BAAI/bge-m3",
     "note": "Best multilingual, wants a GPU", "calib": (0.35, 0.6)},
    {"id": "nomic-embed-text-v1.5", "label": "Nomic Embed v1.5", "size": 5.5e8, "repo": "nomic-ai/nomic-embed-text-v1.5",
     "note": "Long context English", "calib": (0.4, 0.65), "q": "search_query: ", "d": "search_document: ", "remote_code": True},
    {"id": "mxbai-embed-large-v1", "label": "mxbai embed large v1", "size": 6.7e8, "repo": "mixedbread-ai/mxbai-embed-large-v1",
     "note": "Top-tier English retrieval", "calib": (0.4, 0.65),
     "q": "Represent this sentence for searching relevant passages: "},
]

KINDS = {"speech": SPEECH, "vision": VISION, "caption": CAPTION, "text": TEXT}
SPEECH_ENGINES = ["mlx", "faster-whisper", "whisper.cpp", "transformers"]


def get(kind: str, model_id: str) -> dict:
    for m in KINDS[kind]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"Unknown {kind} model: {model_id}")


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


def _engine_available(engine: str) -> bool:
    from . import hardware as hw

    return {
        "mlx": hw.mlx_ok,
        "faster-whisper": lambda: hw.has_module("faster_whisper"),
        "whisper.cpp": lambda: bool(hw.whisper_cpp_bin()),
        "transformers": lambda: hw.has_module("transformers") and hw.has_module("torch"),
    }[engine]()


def defaults() -> dict:
    """Sensible picks for this hardware."""
    from . import hardware as hw

    gpu = hw.mlx_ok() or hw.ct2_cuda() or hw.torch_accel()["backend"] != "CPU"
    caption = "smolvlm-500m" if gpu else "smolvlm-256m"
    installed = hw.ollama_models() or []
    for m in CAPTION:
        if m.get("ollama") and m["ollama"] in installed:
            caption = m["id"]
            break
    return {
        "speech": "whisper-large-v3-turbo" if gpu else "whisper-small",
        "vision": "siglip-base" if gpu else "clip-vit-b32",
        "caption": caption,
        "text": "bge-small-en-v1.5",
    }
