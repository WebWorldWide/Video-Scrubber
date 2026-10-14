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
    # Probe hardware off the request path so the first /hardware call is instant.
    asyncio.get_running_loop().run_in_executor(None, hardware.summary)
    # The desktop shell waits for this line to learn the port.
    print(f"VSCRUBBER_READY port={os.environ.get('VS_PORT', '')}", flush=True)
    yield


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


@app.get("/models")
def get_models():
    tags = hardware.ollama_models()
    s = settings.load()
    out = {}
    for kind, items in catalog.KINDS.items():
        rows = []
        for m in items:
            row = {k: m[k] for k in ("id", "label", "size", "note") if k in m}
            row["langs"] = m.get("langs")
            row["via"] = "ollama" if m.get("ollama") else ("builtin" if m.get("repo") or kind == "speech" else None)
            if kind == "speech":
                row["engines"] = [e for e in catalog.SPEECH_ENGINES if e in m]
                try:
                    row["engine"] = catalog.speech_engine_for(m, s["speechEngine"])
                except RuntimeError:
                    row["engine"] = None
            row["available"] = not (row["via"] == "ollama" and tags is None) and (kind != "speech" or row["engine"] is not None)
            row["installed"] = models.is_installed(kind, m["id"], tags or [])
            rows.append(row)
        out[kind] = rows
    return {"models": out, "selected": {k: s[k] for k in catalog.KINDS}, "ollama": tags is not None,
            "pulls": models.pull_state()}


class PullReq(BaseModel):
    kind: str
    id: str


@app.post("/models/pull")
def pull_model(r: PullReq):
    catalog.get(r.kind, r.id)
    models.pull(r.kind, r.id)
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


MIME = {".mov": "video/mp4", ".m4v": "video/mp4", ".mkv": "video/webm", ".ogv": "video/ogg", ".m4a": "audio/mp4"}


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
