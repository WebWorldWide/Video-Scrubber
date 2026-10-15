# SPDX-License-Identifier: GPL-3.0-or-later
"""Loads and runs local models. One instance per kind stays warm; switching models unloads the old one."""

import base64
import gc
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import types
import urllib.request
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from . import catalog
from . import hardware as hw
from .events import bus

Progress = Callable[[float], None]

# ---------------------------------------------------------------- downloads


def _hf_cache() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def _repo_dir(repo: str) -> Path:
    return _hf_cache() / ("models--" + repo.replace("/", "--"))


IGNORE = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "openvino/*", "*.tflite", "coreml/*", "*fp32*",
          "flax_model*", "tf_model*", "rust_model*", "*.mlmodel*", "*.gguf", "*.ckpt"]
_files_cache: dict[str, list[tuple[str, int]]] = {}


def repo_files(repo: str, network: bool = True) -> list[tuple[str, int]]:
    """(path, bytes) for every file in a Hugging Face repo, cached in memory and on disk.
    With network=False, raises LookupError instead of asking Hugging Face."""
    if repo in _files_cache:
        return _files_cache[repo]
    from huggingface_hub import HfApi

    from .paths import data_dir

    cache_file = data_dir() / "hf-files.json"
    try:
        disk = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        disk = {}
    if repo in disk:
        _files_cache[repo] = [tuple(x) for x in disk[repo]]
        return _files_cache[repo]
    if not network:
        raise LookupError(repo)
    info = HfApi().model_info(repo, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    _files_cache[repo] = files
    disk[repo] = files
    try:
        cache_file.write_text(json.dumps(disk))
    except OSError:
        pass
    return files


def _weight_patterns(repo: str, engine: str | None, network: bool = True) -> list[str]:
    """Fetch one copy of the weights: skip TF/Flax/ONNX/OpenVINO and duplicate .bin when safetensors exist."""
    if engine in ("mlx", "faster-whisper"):
        return []
    ignore = list(IGNORE)
    try:
        if any(f.endswith(".safetensors") for f, _ in repo_files(repo, network)):
            ignore += ["*.bin", "*.pt", "*.pth"]
    except Exception:
        pass
    return ignore


WEIGHT_EXT = (".safetensors", ".bin", ".npz", ".gguf", ".pt", ".pth")


def add_custom(kind: str, ref: str) -> dict:
    """Make any Hugging Face repo selectable (or an Ollama tag, for descriptions), working out which
    runtime here can load it. `ref` is `owner/name`, a huggingface.co link, or a link to one ggml file."""
    ref = ref.strip()
    if kind == "caption" and ref.startswith("ollama:"):
        tag = ref.removeprefix("ollama:").strip()
        return catalog.add_custom(kind, {"id": "ollama:" + tag, "label": tag, "ollama": tag, "note": "Ollama"})
    parts = re.sub(r"^(https?://)?(www\.)?(huggingface\.co|hf\.co)/", "", ref).split("?")[0].strip("/").split("/")
    if len(parts) > 1 and parts[1] in ("blob", "resolve", "tree"):  # legacy repos without an owner
        parts = [""] + parts
    repo = "/".join(parts[:2]).strip("/")
    file = parts[-1] if len(parts) > 4 and parts[2] in ("blob", "resolve") else None
    if not re.fullmatch(r"[\w.-]+(/[\w.-]+)?", repo):
        raise ValueError("Use owner/model or a huggingface.co link")

    from huggingface_hub import HfApi, get_token
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = HfApi().model_info(repo)
        files = repo_files(repo)
    except RepositoryNotFoundError:
        raise ValueError(f"{repo} not found on Hugging Face")
    if info.gated and not get_token():
        raise ValueError(f"{repo} is gated. Accept its terms on huggingface.co and run `hf auth login`, then add it again.")
    names = {f for f, _ in files}
    tags = set(info.tags or [])
    cfg = info.config or {}
    mtype = str(cfg.get("model_type", ""))
    mlx = "mlx" in tags or info.library_name == "mlx"
    m = {"id": "hf:" + repo, "label": repo.split("/")[-1], "note": repo.split("/")[0] if "/" in repo else "",
         "license": (info.card_data or {}).get("license") if info.card_data else None}

    if kind == "speech":
        ggml = sorted(f for f in names if f.endswith(".bin") and "ggml" in f)
        if file or (ggml and not mlx and "model.bin" not in names):
            if not file and len(ggml) > 1:
                raise ValueError("This repo has several whisper.cpp files. Paste the link to the one you want.")
            file = file or ggml[0]
            m.update({"id": f"hf:{repo}/{file}", "label": file.removesuffix(".bin"), "whisper.cpp": [repo, file]})
        elif "model.bin" in names and any(f.startswith("vocabulary") for f in names):
            m["faster-whisper"] = repo
        elif mlx:
            m["mlx"] = repo
        elif info.pipeline_tag == "automatic-speech-recognition":
            m["transformers"] = repo
        else:
            raise ValueError(f"{repo} isn't a speech-to-text model")
    elif kind == "caption":
        if mlx:
            if not hw.mlx_vlm_ok():
                raise ValueError("MLX models run on Apple Silicon only")
            m["mlx"] = repo
        elif info.pipeline_tag in ("image-text-to-text", "visual-question-answering", "image-to-text"):
            m["transformers"] = repo
        else:
            raise ValueError(f"{repo} isn't a vision-language model")
    elif kind == "vision":
        if "siglip" in mtype:
            m.update(transformers=repo, sigmoid=True)
        elif "clip" in mtype:
            m.update(transformers=repo, calib=(0.19, 0.31))
        else:
            raise ValueError(f"{repo} isn't a CLIP or SigLIP model")
    else:
        if "sentence-transformers" not in tags and info.pipeline_tag not in ("sentence-similarity", "feature-extraction"):
            raise ValueError(f"{repo} isn't a sentence-embedding model")
        m.update(transformers=repo, calib=(0.3, 0.6), multilingual=True, remote_code="auto_map" in cfg)

    m["bytes"] = next((n for f, n in files if f == file), 0) if file else _weights_bytes(files)
    if info.safetensors and not mlx:
        m["params"] = info.safetensors.total
    return catalog.add_custom(kind, m)


def _weights_bytes(files: list[tuple[str, int]]) -> int:
    weights = [(f, n) for f, n in files if f.endswith(WEIGHT_EXT) and not f.startswith(("onnx/", "openvino/"))]
    if any(f.endswith(".safetensors") for f, _ in weights):
        weights = [(f, n) for f, n in weights if f.endswith(".safetensors")]
    return sum(n for _, n in weights)


def download_size(kind: str, model_id: str, network: bool = True) -> int | None:
    """Bytes this model will download for the runtime it uses here, from the Hugging Face file listing."""
    import fnmatch

    try:
        r = resolve(kind, model_id)
        if r["type"] == "hf_file":
            return next((n for f, n in repo_files(r["repo"], network) if f == r["file"]), None)
        if r["type"] != "hf":
            return 0 if r["type"] == "none" else None
        files = repo_files(r["repo"], network)
        ignore = _weight_patterns(r["repo"], r["engine"], network)
        return sum(n for f, n in files if not any(fnmatch.fnmatch(f, p) for p in ignore))
    except Exception:
        return None


def resolve(kind: str, model_id: str, engine: str | None = None) -> dict:
    """What has to be on disk for a model: an HF repo, one file in a repo, or an Ollama tag."""
    from . import settings

    m = catalog.get(kind, model_id)
    engine = engine or catalog.engine_for(kind, m, settings.load()["speechEngine"] if kind == "speech" else "auto")
    if engine == "ollama":
        return {"type": "ollama", "tag": m["ollama"]}
    if engine in (None, "none"):
        if model_id == "none":
            return {"type": "none"}
        raise RuntimeError(f"No installed runtime can run {m['label']}")
    target = catalog.repo_for(kind, m, engine)
    if isinstance(target, tuple):
        return {"type": "hf_file", "repo": target[0], "file": target[1], "engine": engine}
    return {"type": "hf", "repo": target, "engine": engine}


def is_installed(kind: str, model_id: str, ollama_tags: list[str] | None = None) -> bool:
    try:
        r = resolve(kind, model_id)
    except Exception:
        return False
    if r["type"] == "none":
        return True
    if r["type"] == "ollama":
        tags = ollama_tags if ollama_tags is not None else (hw.ollama_models() or [])
        return r["tag"] in tags or (r["tag"] + ":latest") in tags
    d = _repo_dir(r["repo"]) / "snapshots"
    if not d.exists():
        return False
    if r["type"] == "hf_file":
        return any((s / r["file"]).exists() for s in d.iterdir())
    marker = _repo_dir(r["repo"]) / ".vs-complete"
    return marker.exists()


def _dir_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


_pulls: dict[str, dict] = {}


def pull(kind: str, model_id: str) -> None:
    """Download a model in the background, publishing `model` progress events."""
    key = f"{kind}:{model_id}"
    if key in _pulls:
        return
    _pulls[key] = {"pct": 0}
    threading.Thread(target=_pull, args=(kind, model_id, key), daemon=True).start()


def pull_state() -> dict:
    return {k: v for k, v in _pulls.items()}


_ensure_locks: dict[str, threading.Lock] = {}


def ensure(kind: str, model_id: str, cb: Progress | None = None):
    """Download synchronously if missing (used by indexing jobs). Parallel jobs wait on one download."""
    key = f"{kind}:{model_id}"
    with _ensure_locks.setdefault(key, threading.Lock()):
        if not is_installed(kind, model_id):
            _pull(kind, model_id, key, cb)


def _pull(kind, model_id, key, cb: Progress | None = None):
    def report(p):
        _pulls[key] = {"pct": p}
        bus.publish("model", throttle_key=key, kind=kind, id=model_id, state="downloading", pct=p)
        if cb:
            cb(p)

    try:
        r = resolve(kind, model_id)
        _check_disk(kind, model_id, r)
        if r["type"] == "ollama":
            _ollama_pull(r["tag"], report)
        elif r["type"] == "hf_file":
            from huggingface_hub import hf_hub_download

            _with_size_poll(r["repo"], [r["file"]], report, lambda: hf_hub_download(r["repo"], r["file"]))
        elif r["type"] == "hf":
            from huggingface_hub import snapshot_download

            ignore = _weight_patterns(r["repo"], r["engine"])
            _with_size_poll(r["repo"], None, report,
                            lambda: snapshot_download(r["repo"], ignore_patterns=ignore or None))
            (_repo_dir(r["repo"]) / ".vs-complete").touch()
        bus.publish("model", kind=kind, id=model_id, state="installed", pct=1)
    except Exception as e:
        bus.publish("model", kind=kind, id=model_id, state="error", error=str(e))
        raise
    finally:
        _pulls.pop(key, None)


def _check_disk(kind, model_id, r):
    if r["type"] not in ("hf", "hf_file"):
        return
    need = download_size(kind, model_id) or 0
    have = _dir_size(_repo_dir(r["repo"])) if _repo_dir(r["repo"]).exists() else 0
    free = hw.disk_free()
    if need - have > free - 2 * catalog.GB:
        raise RuntimeError(f"Not enough disk space: {catalog.get(kind, model_id)['label']} needs "
                           f"{(need - have) / 1e9:.1f} GB and {free / 1e9:.1f} GB is free")


def _with_size_poll(repo, files, report, fn):
    """Report progress by watching the cache directory grow against the repo's listed sizes."""
    import fnmatch

    total = 0
    try:
        ignore = [] if files else _weight_patterns(repo, None)
        total = sum(n for f, n in repo_files(repo)
                    if (f in files if files else not any(fnmatch.fnmatch(f, p) for p in ignore)))
    except Exception:
        pass
    d = _repo_dir(repo)
    start = _dir_size(d) if d.exists() else 0
    done = threading.Event()

    def poll():
        while not done.wait(0.5):
            if total:
                report(min(0.99, max(0, (_dir_size(d) - start)) / total))

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        done.set()
        report(1.0)


def _ollama_pull(tag, report):
    req = urllib.request.Request(hw.ollama_host() + "/api/pull", data=json.dumps({"model": tag, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            ev = json.loads(line)
            if ev.get("error"):
                raise RuntimeError(ev["error"])
            if ev.get("total"):
                report(min(0.99, ev.get("completed", 0) / ev["total"]))


# ---------------------------------------------------------------- registry

_loaded: dict[str, tuple[str, object]] = {}
_locks = {k: threading.Lock() for k in ("speech", "vision", "caption", "text")}


def lock(kind: str) -> threading.Lock:
    return _locks[kind]


def get(kind: str, model_id: str, engine: str = "auto"):
    """Warm instance for (kind, model). Call while holding lock(kind)."""
    key = f"{model_id}@{engine}"
    cur = _loaded.get(kind)
    if cur and cur[0] == key:
        return cur[1]
    if cur:
        _loaded.pop(kind)
        del cur
        _free()
    inst = {"speech": Speech, "vision": Vision, "caption": Captioner, "text": TextEmbedder}[kind](model_id, engine)
    _loaded[kind] = (key, inst)
    return inst


def _free():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------- speech


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


class Speech:
    def __init__(self, model_id: str, engine: str = "auto"):
        self.meta = catalog.get("speech", model_id)
        self.engine = catalog.speech_engine_for(self.meta, engine)
        self.target = self.meta[self.engine]
        self.model = None

    def transcribe(self, wav: str, duration: float, language: str | None, cb: Progress) -> list[dict]:
        lang = None if language in (None, "", "auto") else language
        segs = getattr(self, "_" + self.engine.replace(".", "_").replace("-", "_"))(wav, duration, lang, cb)
        return _clean(segs)

    def _mlx(self, wav, duration, lang, cb):
        import mlx_whisper

        T = sys.modules["mlx_whisper.transcribe"]  # the package re-exports a function under the same name

        class Bar:
            def __init__(self, *a, total=None, **k):
                self.total, self.n = total or 1, 0

            def update(self, n):
                self.n += n
                cb(min(1.0, self.n / self.total))

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        orig = T.tqdm
        T.tqdm = types.SimpleNamespace(tqdm=Bar)
        try:
            res = mlx_whisper.transcribe(wav, path_or_hf_repo=self.target, verbose=False, language=lang,
                                         condition_on_previous_text=False)
        finally:
            T.tqdm = orig
        return [{"t": s["start"], "e": s["end"], "x": s["text"], "nsp": s.get("no_speech_prob", 0),
                 "lp": s.get("avg_logprob", 0)} for s in res["segments"]]

    def _faster_whisper(self, wav, duration, lang, cb):
        from faster_whisper import WhisperModel

        if self.model is None:
            cuda = hw.ct2_cuda()
            self.model = WhisperModel(self.target, device="cuda" if cuda else "cpu",
                                      compute_type="float16" if cuda else "int8", cpu_threads=os.cpu_count() or 4)
        # Hand over decoded samples: faster-whisper's own PyAV loader breaks on PyAV 15+ (metadata_errors).
        segs, _ = self.model.transcribe(load_wav(wav), language=lang, vad_filter=True, beam_size=5 if hw.ct2_cuda() else 1,
                                        condition_on_previous_text=False)
        out = []
        for s in segs:
            out.append({"t": s.start, "e": s.end, "x": s.text, "nsp": s.no_speech_prob, "lp": s.avg_logprob})
            cb(min(1.0, s.end / max(duration, 1)))
        return out

    def _whisper_cpp(self, wav, duration, lang, cb):
        from huggingface_hub import hf_hub_download

        model = hf_hub_download(*self.target)
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "out")
            cmd = [hw.whisper_cpp_bin(), "-m", model, "-f", wav, "-oj", "-of", base, "-pp", "-l", lang or "auto",
                   "-t", str(max(1, (os.cpu_count() or 4) - 1))]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=hw.NO_WINDOW)
            tail = []
            for line in p.stderr:
                m = re.search(r"progress\s*=\s*(\d+)%", line)
                if m:
                    cb(int(m[1]) / 100)
                tail = (tail + [line])[-5:]
            p.stdout.read()
            if p.wait() != 0:
                raise RuntimeError("whisper.cpp failed: " + "".join(tail).strip())
            data = json.loads(Path(base + ".json").read_text(errors="replace"))
        return [{"t": s["offsets"]["from"] / 1000, "e": s["offsets"]["to"] / 1000, "x": s["text"]}
                for s in data.get("transcription", [])]

    def _transformers(self, wav, duration, lang, cb):
        import torch
        from transformers import pipeline

        if self.model is None:
            self.model = pipeline("automatic-speech-recognition", model=self.target, dtype=hw.torch_dtype(),
                                  device=hw.torch_device())
        whisper = self.model.model.config.model_type == "whisper"
        audio = load_wav(wav)
        sr, win = 16000, 30 if whisper else 15
        out = []
        kw = {"return_timestamps": True, "generate_kwargs": {"task": "transcribe", **({"language": lang} if lang else {})}} if whisper else {}
        for start in range(0, len(audio), win * sr):
            chunk = audio[start:start + win * sr]
            off = start / sr
            if np.abs(chunk).max(initial=0) < 0.003:
                continue
            with torch.inference_mode():
                r = self.model({"raw": chunk, "sampling_rate": sr}, **kw)
            if whisper and r.get("chunks"):
                for c in r["chunks"]:
                    a, b = c["timestamp"]
                    out.append({"t": off + (a or 0), "e": off + (b if b is not None else len(chunk) / sr), "x": c["text"]})
            elif r.get("text", "").strip():
                out.append({"t": off, "e": off + len(chunk) / sr, "x": r["text"]})
            cb(min(1.0, (start + win * sr) / len(audio)))
        return out


def _clean(segs: list[dict]) -> list[dict]:
    """Drop silence hallucinations and immediate repeats."""
    out = []
    for s in segs:
        x = re.sub(r"\s+", " ", s["x"]).strip()
        if not x or (s.get("nsp", 0) > 0.6 and s.get("lp", 0) < -1.0):
            continue
        if out and out[-1]["x"] == x:
            out[-1]["e"] = s["e"]
            continue
        out.append({"t": round(float(s["t"]), 2), "e": round(float(max(s["e"], s["t"] + 0.3)), 2), "x": x})
    return out


# ---------------------------------------------------------------- picture search


class Vision:
    def __init__(self, model_id: str, engine: str = "auto"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.meta = catalog.get("vision", model_id)
        self.device = hw.torch_device()
        self.dtype = hw.torch_dtype()
        self.model = AutoModel.from_pretrained(self.meta["transformers"], dtype=self.dtype).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(self.meta["transformers"])
        self.sigmoid = self.meta.get("sigmoid", False)
        if self.sigmoid:
            self.scale = float(self.model.logit_scale.exp().item())
            self.bias = float(self.model.logit_bias.item())
        self.torch = torch

    def _norm(self, x):
        # transformers 5 returns an output object whose pooler_output is the projected embedding; 4.x a tensor.
        x = getattr(x, "pooler_output", x).float()
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy()

    def embed_images(self, paths: list[str], cb: Progress | None = None, batch: int = 16) -> np.ndarray:
        from PIL import Image

        out = []
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inputs = self.proc(images=imgs, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            with self.torch.inference_mode():
                out.append(self._norm(self.model.get_image_features(**inputs)))
            if cb:
                cb(min(1.0, (i + batch) / len(paths)))
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        kw = {"padding": "max_length", "max_length": 64} if self.sigmoid else {"padding": True}
        inputs = self.proc(text=[text], return_tensors="pt", truncation=True, **kw).to(self.device)
        with self.torch.inference_mode():
            return self._norm(self.model.get_text_features(**inputs))[0]

    def relevance(self, sims: np.ndarray) -> np.ndarray:
        """Map cosine similarity to a 0–1 relevance the search can compare with text scores."""
        if self.sigmoid:
            # Matching frames land around -3…+6, unrelated ones below -8.
            logits = sims * self.scale + self.bias
            return np.clip((logits + 10.0) / 10.0, 0, 1)
        lo, hi = self.meta["calib"]
        return np.clip((sims - lo) / (hi - lo), 0, 1)


# ---------------------------------------------------------------- frame descriptions

CAPTION_PROMPT = ("Describe this video frame in one plain sentence of at most 30 words: who or what is visible, "
                  "what is happening, the setting, and any clearly readable on-screen text. No preamble.")


class Captioner:
    def __init__(self, model_id: str, engine: str = "auto"):
        self.meta = catalog.get("caption", model_id)
        self.engine = catalog.engine_for("caption", self.meta)
        self.tag = self.meta.get("ollama")
        repo = catalog.repo_for("caption", self.meta, self.engine)
        if self.engine == "mlx":
            from mlx_vlm import load
            from mlx_vlm.prompt_utils import apply_chat_template

            self.model, self.proc = load(repo)
            self.prompt = apply_chat_template(self.proc, self.model.config, CAPTION_PROMPT, num_images=1)
        elif self.engine == "transformers":
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.torch = torch
            self.device = hw.torch_device()
            self.proc = AutoProcessor.from_pretrained(repo)
            self.model = AutoModelForImageTextToText.from_pretrained(repo, dtype=hw.torch_dtype()).to(self.device).eval()

    def describe(self, path: str) -> str:
        if self.engine == "ollama":
            return self._ollama(path)
        if self.engine == "mlx":
            from mlx_vlm import generate

            r = generate(self.model, self.proc, self.prompt, [path], max_tokens=70, temperature=0.0, verbose=False)
            return _tidy(getattr(r, "text", r))
        return self._local(path)

    def _ollama(self, path: str) -> str:
        img = base64.b64encode(Path(path).read_bytes()).decode()
        body = {"model": self.tag, "prompt": CAPTION_PROMPT, "images": [img], "stream": False,
                "options": {"temperature": 0, "num_predict": 80}}
        req = urllib.request.Request(hw.ollama_host() + "/api/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return _tidy(json.load(r).get("response", ""))

    def _local(self, path: str) -> str:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(text=prompt, images=[img], return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=70, do_sample=False)
        text = self.proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return _tidy(text)


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().strip('"')
    s = re.sub(r"^(Assistant:|The image shows|This image shows|In this (image|frame),?)\s*", "", s, flags=re.I).strip()
    return (s[:1].upper() + s[1:]) if s else s


# ---------------------------------------------------------------- text embeddings


class TextEmbedder:
    def __init__(self, model_id: str, engine: str = "auto"):
        from sentence_transformers import SentenceTransformer

        self.meta = catalog.get("text", model_id)
        dev = hw.torch_accel()["device"]
        self.model = SentenceTransformer(self.meta["transformers"], device=dev if dev != "directml" else "cpu",
                                         trust_remote_code=self.meta.get("remote_code", False))

    def docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), np.float32)
        pre = self.meta.get("d", "")
        return self.model.encode([pre + t for t in texts], normalize_embeddings=True, batch_size=32,
                                 convert_to_numpy=True).astype(np.float32)

    def query(self, text: str) -> np.ndarray:
        return self.model.encode([self.meta.get("q", "") + text], normalize_embeddings=True, convert_to_numpy=True)[0]

    def relevance(self, sims: np.ndarray) -> np.ndarray:
        """Absolute similarity, blended with how far a line stands out from the rest of this video."""
        lo, hi = self.meta["calib"]
        rel = np.clip((sims - lo) / (hi - lo), 0, 1)
        if len(sims) >= 4:
            lift = np.clip((sims - np.median(sims)) / ((hi - lo) * 0.75), 0, 1)
            rel = np.clip(rel * (0.8 + 0.4 * lift), 0, 1)
        return rel
