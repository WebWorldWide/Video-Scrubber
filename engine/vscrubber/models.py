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


def _weight_patterns(repo: str, engine: str | None) -> tuple[list[str] | None, list[str]]:
    """Fetch one copy of the weights: skip TF/Flax/ONNX/OpenVINO and duplicate .bin when safetensors exist."""
    if engine in ("mlx", "faster-whisper"):
        return None, []
    ignore = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "openvino/*", "*.tflite", "coreml/*", "*fp32*",
              "flax_model*", "tf_model*", "rust_model*", "*.mlmodel*"]
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo)
        if any(f.endswith(".safetensors") for f in files):
            ignore += ["*.bin", "*.pt", "*.pth"]
    except Exception:
        pass
    return None, ignore


def resolve(kind: str, model_id: str, engine: str | None = None) -> dict:
    """What has to be on disk for a model: an HF repo, one file in a repo, or an Ollama tag."""
    m = catalog.get(kind, model_id)
    if kind == "speech":
        engine = engine or catalog.speech_engine_for(m)
        target = m[engine]
        if isinstance(target, tuple):
            return {"type": "hf_file", "repo": target[0], "file": target[1], "engine": engine}
        return {"type": "hf", "repo": target, "engine": engine}
    if m.get("ollama"):
        return {"type": "ollama", "tag": m["ollama"]}
    if m.get("repo"):
        return {"type": "hf", "repo": m["repo"], "engine": None}
    return {"type": "none"}


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
        if r["type"] == "ollama":
            _ollama_pull(r["tag"], report)
        elif r["type"] == "hf_file":
            from huggingface_hub import hf_hub_download

            _with_size_poll(r["repo"], [r["file"]], report, lambda: hf_hub_download(r["repo"], r["file"]))
        elif r["type"] == "hf":
            from huggingface_hub import snapshot_download

            allow, ignore = _weight_patterns(r["repo"], r["engine"])
            _with_size_poll(r["repo"], None, report,
                            lambda: snapshot_download(r["repo"], allow_patterns=allow, ignore_patterns=ignore or None))
            (_repo_dir(r["repo"]) / ".vs-complete").touch()
        bus.publish("model", kind=kind, id=model_id, state="installed", pct=1)
    except Exception as e:
        bus.publish("model", kind=kind, id=model_id, state="error", error=str(e))
        raise
    finally:
        _pulls.pop(key, None)


def _with_size_poll(repo, files, report, fn):
    """Report progress by watching the cache directory grow against the repo's listed sizes."""
    total = 0
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        total = sum((s.size or 0) for s in info.siblings if files is None or s.rfilename in files)
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
        segs, _ = self.model.transcribe(wav, language=lang, vad_filter=True, beam_size=5 if hw.ct2_cuda() else 1,
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
        whisper = "whisper" in self.target
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
        self.model = AutoModel.from_pretrained(self.meta["repo"], dtype=self.dtype).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(self.meta["repo"])
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
            logits = sims * self.scale + self.bias
            return np.clip((logits + 9.0) / 9.0, 0, 1)
        lo, hi = self.meta["calib"]
        return np.clip((sims - lo) / (hi - lo), 0, 1)


# ---------------------------------------------------------------- frame descriptions

CAPTION_PROMPT = ("Describe this video frame in one plain sentence of at most 30 words: who or what is visible, "
                  "what is happening, the setting, and any clearly readable on-screen text. No preamble.")


class Captioner:
    def __init__(self, model_id: str, engine: str = "auto"):
        self.meta = catalog.get("caption", model_id)
        self.tag = self.meta.get("ollama")
        if not self.tag and self.meta.get("repo"):
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.torch = torch
            self.device = hw.torch_device()
            self.proc = AutoProcessor.from_pretrained(self.meta["repo"])
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.meta["repo"], dtype=hw.torch_dtype()).to(self.device).eval()

    def describe(self, path: str) -> str:
        if self.tag:
            return self._ollama(path)
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
        self.model = SentenceTransformer(self.meta["repo"], device=dev if dev != "directml" else "cpu",
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
