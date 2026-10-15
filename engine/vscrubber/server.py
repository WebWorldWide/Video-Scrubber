# SPDX-License-Identifier: GPL-3.0-or-later
import asyncio
import json
import mimetypes
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from . import __version__, catalog, downloads, hardware, media, models, search, settings
from .events import bus
from .library import Library

TOKEN = os.environ.get("VS_TOKEN", "")
lib: Library | None = None


@asynccontextmanager
async def lifespan(app):
    global lib
    bus.attach(asyncio.get_running_loop())
    lib = Library()
    # Probe hardware and warm the search models off the request path, so the first search is quick.
    asyncio.get_running_loop().run_in_executor(None, hardware.summary)
    asyncio.get_running_loop().run_in_executor(None, _warm)
    asyncio.get_running_loop().run_in_executor(None, _prefetch_sizes)
    # The desktop shell waits for this line to learn the port.
    print(f"VSCRUBBER_READY port={os.environ.get('VS_PORT', '')}", flush=True)
    yield


def _warm():
    if not any(v["status"] == "ready" for v in lib.videos.values()):
        return
    s = settings.load()
    for kind in ("text", "vision"):
        if models.is_installed(kind, s[kind]):
            try:
                with models.lock(kind):
                    models.get(kind, s[kind])
            except Exception:
                pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def auth(request: Request, call_next):
    if TOKEN and request.method != "OPTIONS":
        got = request.headers.get("x-vs-token") or request.query_params.get("token")
        if got != TOKEN:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


def _video(vid: str) -> dict:
    v = lib.videos.get(vid)
    if not v:
        raise HTTPException(404, "No such video")
    return v


# ---------------------------------------------------------------- system


@app.get("/health")
def health():
    return {"ok": True, "version": __version__}


@app.get("/hardware")
def get_hardware():
    return hardware.summary()


@app.get("/settings")
def get_settings():
    return settings.load()


@app.patch("/settings")
def patch_settings(patch: dict):
    try:
        return settings.update(patch)
    except KeyError as e:
        raise HTTPException(400, str(e))


def _model_row(kind: str, m: dict, s: dict, tags: list[str] | None, budget: int, rec: dict) -> dict:
    engine = catalog.engine_for(kind, m, s["speechEngine"] if kind == "speech" else "auto")
    mem = catalog.memory_needed(kind, m, engine)
    repo = catalog.repo_for(kind, m, engine) if engine else None
    return {
        "id": m["id"], "label": m["label"], "note": m.get("note", ""), "params": m.get("params"),
        "quality": m.get("quality"), "speed": catalog.effective_speed(m, engine) if engine and m.get("speed") else m.get("speed"),
        "license": m.get("license"), "englishOnly": bool(m.get("english")),
        "via": "ollama" if m.get("ollama") else ("none" if m["id"] == "none" else "huggingface"),
        "repo": repo[0] if isinstance(repo, tuple) else repo,
        "engine": engine, "engines": [e for e in catalog.SPEECH_ENGINES if e in m] if kind == "speech" else None,
        "onGpu": catalog.on_gpu(engine) if engine and engine not in ("none", "ollama") else None,
        "memory": mem, "download": models.download_size(kind, m["id"], network=False) if engine not in (None, "ollama") else None,
        "fits": mem <= budget, "recommended": rec.get(kind) == m["id"], "custom": bool(m.get("custom")),
        "available": engine is not None and not (engine == "ollama" and tags is None),
        "installed": models.is_installed(kind, m["id"], tags or []) if engine else False,
    }


@app.get("/models")
def get_models():
    tags = hardware.ollama_models()
    s = settings.load()
    mem = hardware.memory()
    rec = catalog.recommend(s["language"])
    out = {kind: [_model_row(kind, m, s, tags, mem["budget"], rec) for m in catalog.models(kind)] for kind in catalog.KINDS}
    used = sum(catalog.memory_needed(k, catalog.get(k, s[k]), catalog.engine_for(k, catalog.get(k, s[k]))) for k in catalog.KINDS)
    return {"models": out, "selected": {k: s[k] for k in catalog.KINDS}, "recommended": rec, "ollama": tags is not None,
            "pulls": models.pull_state(), "memory": {**mem, "used": used}, "diskFree": hardware.disk_free()}


@app.post("/models/recommended")
def use_recommended():
    """Switch to the recommended set for this machine and download whatever is missing."""
    s = settings.load()
    rec = catalog.recommend(s["language"])
    s = settings.update(rec)
    for kind, mid in rec.items():
        if not models.is_installed(kind, mid):
            models.pull(kind, mid)
    return s


def _prefetch_sizes():
    """Fill the Hugging Face file-size cache so the models dialog shows real download sizes."""
    for kind, items in catalog.KINDS.items():
        for m in items:
            models.download_size(kind, m["id"])


class PullReq(BaseModel):
    kind: str
    id: str


@app.post("/models/pull")
def pull_model(r: PullReq):
    catalog.get(r.kind, r.id)
    models.pull(r.kind, r.id)
    return {"ok": True}


class CustomReq(BaseModel):
    kind: str
    ref: str


@app.post("/models/custom")
def add_custom_model(r: CustomReq):
    """Add any Hugging Face repo (or Ollama tag) as a model choice."""
    if r.kind not in catalog.KINDS:
        raise HTTPException(400, f"Unknown kind: {r.kind}")
    try:
        m = models.add_custom(r.kind, r.ref)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": m["id"]}


@app.post("/models/custom/remove")
def remove_custom_model(r: PullReq):
    catalog.remove_custom(r.kind, r.id)
    s = settings.load()
    if s.get(r.kind) == r.id:
        settings.update({r.kind: catalog.recommend(s["language"])[r.kind]})
    return {"ok": True}


# ---------------------------------------------------------------- library


class AddReq(BaseModel):
    paths: list[str]


@app.get("/library")
def get_library():
    return lib.list()


@app.post("/library")
async def add_videos(r: AddReq):
    added, errors = [], []
    for p in r.paths:
        try:
            added.append(await run_in_threadpool(lib.add, p))
        except Exception as e:
            errors.append({"path": p, "error": str(e)})
    return {"added": added, "errors": errors}


@app.delete("/library/{vid}")
def remove_video(vid: str):
    _video(vid)
    lib.remove(vid)
    return {"ok": True}


@app.post("/library/{vid}/reindex")
def reindex(vid: str):
    _video(vid)
    lib.reindex(vid)
    return {"ok": True}


@app.post("/library/reindex")
def reindex_all():
    for vid in list(lib.videos):
        lib.reindex(vid)
    return {"ok": True}


@app.get("/videos/{vid}")
def video_detail(vid: str):
    _video(vid)
    return lib.detail(vid)


MIME = {".mov": "video/mp4", ".m4v": "video/mp4", ".mkv": "video/webm", ".ogv": "video/ogg", ".m4a": "audio/mp4",
        ".m4b": "audio/mp4", ".aac": "audio/aac", ".flac": "audio/flac", ".oga": "audio/ogg", ".opus": "audio/ogg",
        ".weba": "audio/webm", ".mka": "audio/webm", ".caf": "audio/x-caf", ".aif": "audio/aiff", ".aiff": "audio/aiff",
        ".wav": "audio/wav"}


@app.get("/videos/{vid}/media")
def video_media(vid: str, request: Request, proxy: int = 0):
    v = _video(vid)
    path = v["proxy"] if proxy and v.get("proxy") else v["path"]
    if not os.path.exists(path):
        raise HTTPException(404, "The file has moved or been deleted")
    size = os.path.getsize(path)
    ext = Path(path).suffix.lower()
    ctype = MIME.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"
    rng = request.headers.get("range")
    m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
    if not m:
        return FileResponse(path, media_type=ctype, headers={"Accept-Ranges": "bytes"})
    start = int(m[1]) if m[1] else max(0, size - int(m[2] or 0))
    end = min(size - 1, int(m[2])) if m[1] and m[2] else size - 1
    if start >= size:
        return PlainTextResponse("", status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def body():
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    return StreamingResponse(body(), status_code=206, media_type=ctype, headers={
        "Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)})


@app.get("/videos/{vid}/frames/{name}")
def frame(vid: str, name: str):
    _video(vid)
    if not re.fullmatch(r"\d{4}\.jpg", name):
        raise HTTPException(400, "bad frame name")
    p = lib.dir(vid) / "frames" / name
    if not p.exists():
        raise HTTPException(404, "no frame")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.post("/videos/{vid}/proxy")
def proxy(vid: str):
    _video(vid)
    lib.ensure_proxy(vid)
    return {"ok": True}


def _ts(t: float, srt: bool) -> str:
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    if srt:
        return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"
    return (f"{h}:{m:02d}" if h else f"{m:02d}") + f":{int(s):02d}"


@app.get("/videos/{vid}/export")
def export(vid: str, fmt: str = "txt"):
    d = lib.detail(vid)
    tr, sc = d["transcript"], d["scenes"]
    if fmt == "srt":
        body = "\n".join(f"{i + 1}\n{_ts(g['t'], True)} --> {_ts(g['e'], True)}\n{g['x']}\n" for i, g in enumerate(tr))
    elif fmt == "show":
        parts = [f"{d['name']}\n"]
        for i, s in enumerate(sc):
            parts.append(f"\n[Key frame {i + 1:02d} · {_ts(s['t'], False)}]" + (f" {s['d']}" if s.get("d") else ""))
            parts += [f"  [{_ts(g['t'], False)}] {g['x']}" for g in tr if s["t"] <= g["t"] < s["e"] or (i == 0 and g["t"] < s["t"])]
        body = "\n".join(parts)
    else:
        body = "\n".join(f"[{_ts(g['t'], False)}] {g['x']}" for g in tr)
    return PlainTextResponse(body)


# ---------------------------------------------------------------- search and clips


class SearchReq(BaseModel):
    q: str
    ids: list[str] | None = None


@app.post("/search")
async def do_search(r: SearchReq):
    t0 = time.monotonic()
    res = await run_in_threadpool(search.run, lib, r.q, r.ids)
    return {**res, "ms": int((time.monotonic() - t0) * 1000)}


class ClipReq(BaseModel):
    id: str
    kind: str  # video | audio | frame
    a: float
    b: float | None = None
    format: str | None = None


def _stamp(t: float) -> str:
    return _ts(t, False).replace(":", "-")


@app.post("/clips")
async def make_clip(r: ClipReq):
    v = _video(r.id)
    out_dir = Path(settings.load()["saveDir"]) / "Clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(v["name"]).stem[:80]
    try:
        if r.kind == "frame":
            out = out_dir / f"{stem} — frame {_stamp(r.a)}.png"
            await run_in_threadpool(media.save_frame, v["path"], r.a, out)
        else:
            if r.b is None or r.b - r.a < 0.2:
                raise HTTPException(400, "Select a range first")
            if r.kind == "audio":
                ext = {"MP3": ".mp3", "WAV": ".wav"}.get(r.format or "", ".m4a")
                out = out_dir / f"{stem} — audio {_stamp(r.a)} to {_stamp(r.b)}{ext}"
                fn = media.clip_audio
            else:
                out = out_dir / f"{stem} — video {_stamp(r.a)} to {_stamp(r.b)}.mp4"
                fn = media.clip_video
            key = f"clip:{r.id}:{r.kind}"
            await run_in_threadpool(fn, v["path"], r.a, r.b, out,
                                    lambda p: bus.publish("clip", throttle_key=key, id=r.id, kind=r.kind, pct=p))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return {"path": str(out), "name": out.name, "size": out.stat().st_size}


# ---------------------------------------------------------------- downloads


class InfoReq(BaseModel):
    url: str


class DlReq(BaseModel):
    url: str
    quality: str = "Best"
    format: str = "MP4"
    title: str = ""
    host: str = ""
    playlist: bool = False


@app.post("/downloads/info")
async def dl_info(r: InfoReq):
    try:
        return await run_in_threadpool(downloads.info, r.url.strip())
    except Exception as e:
        msg = str(e).replace("ERROR: ", "")
        raise HTTPException(400, msg[:400] or "Couldn’t read that link")


@app.post("/downloads")
def dl_start(r: DlReq):
    return {"id": downloads.start(r.url, r.quality, r.format, r.title, r.host, r.playlist),
            "saveDir": settings.load()["saveDir"]}


@app.delete("/downloads/{jid}")
def dl_cancel(jid: str):
    downloads.cancel(jid)
    return {"ok": True}


@app.get("/downloads/recent")
def dl_recent():
    return downloads.recent()


# ---------------------------------------------------------------- events


@app.get("/events")
async def events(request: Request):
    q = bus.subscribe()

    async def stream():
        try:
            yield "retry: 1000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
