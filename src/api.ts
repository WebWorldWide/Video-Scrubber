// SPDX-License-Identifier: GPL-3.0-or-later
// Client for the local engine (engine/vscrubber/server.py).

export type EngineState = {
  state: 'installing' | 'starting' | 'ready' | 'error';
  url: string | null;
  token: string;
  step: string;
  pct: number;
  log: string[];
  error: string | null;
};

export type Probe = { duration: number; width: number; height: number; fps: number; vcodec: string | null; acodec: string | null; hasVideo: boolean; hasAudio: boolean };

export type Video = {
  id: string;
  path: string;
  name: string;
  size: number;
  added: number;
  probe: Probe;
  status: 'queued' | 'indexing' | 'ready' | 'error';
  step: number;
  pct: number;
  note: string;
  error: string | null;
  warnings: string[];
  proxy: string | null;
  counts: { segments?: number; scenes?: number };
  exists: boolean;
};

export type Segment = { t: number; e: number; x: string };
export type Scene = { t: number; e: number; f: string; d: string };
export type Detail = Video & { transcript: Segment[]; scenes: Scene[]; models: Record<string, string> };

export type Result = { videoId: string; videoName: string; t: number; e: number; kind: string; title: string; snip: string; scene: number | null; frame: string | null; score: number; conf: number };

export type ModelRow = {
  id: string;
  label: string;
  note: string;
  params: number | null;
  quality: number | null;
  speed: number | null;
  license: string | null;
  englishOnly: boolean;
  via: 'huggingface' | 'ollama' | 'none';
  repo: string | null;
  engine: string | null;
  engines: string[] | null;
  onGpu: boolean | null;
  memory: number;
  download: number | null;
  fits: boolean;
  recommended: boolean;
  available: boolean;
  installed: boolean;
  custom: boolean;
};
export type ModelKind = 'speech' | 'vision' | 'caption' | 'text';
export type Memory = { ram: number; gpuMem: number | null; unified: boolean; budget: number; device: 'gpu' | 'cpu'; used: number };
export type Models = { models: Record<ModelKind, ModelRow[]>; selected: Record<ModelKind, string>; recommended: Record<ModelKind, string>; ollama: boolean; pulls: Record<string, { pct: number }>; memory: Memory; diskFree: number };

export type Settings = Record<ModelKind, string> & { speechEngine: string; language: string; parallel: number; saveDir: string };

export type Hardware = { os: string; cpu: string; gpu: string | null; label: string; speechAccel: string; videoEncoder: string; ollama: boolean; whisperCpp: string | null; torch: { backend: string } };

export type DlInfo = { url: string; host: string; title: string; playlist: boolean; count: number; duration: number | null; direct: boolean; thumbnail: string | null; sizes: Record<string, number> };
export type DlItem = { path: string; name: string; size: number; host: string; at: number };

export type EngineEvent =
  | { type: 'video'; video: Video }
  | { type: 'removed'; id: string }
  | { type: 'transcript' | 'scenes'; id: string }
  | { type: 'model'; kind: ModelKind; id: string; state: 'downloading' | 'installed' | 'error'; pct?: number; error?: string }
  | { type: 'proxy'; id: string; pct?: number; done?: boolean; error?: string }
  | { type: 'clip'; id: string; kind: string; pct: number }
  | { type: 'download'; id: string; state: 'downloading' | 'done' | 'cancelled' | 'error'; got?: number; total?: number; speed?: number; item?: number; count?: number; items?: DlItem[]; error?: string };


let base = '';
let token = '';

export function configure(url: string, tok: string) {
  base = url;
  token = tok;
}

export const connected = () => !!base;

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(base + path, {
    method,
    headers: { 'x-vs-token': token, ...(body !== undefined ? { 'content-type': 'application/json' } : {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let msg = r.statusText;
    try {
      msg = (await r.json()).detail || msg;
    } catch {}
    throw new Error(msg);
  }
  const ct = r.headers.get('content-type') || '';
  return (ct.includes('json') ? r.json() : r.text()) as Promise<T>;
}

export const api = {
  hardware: () => req<Hardware>('GET', '/hardware'),
  settings: () => req<Settings>('GET', '/settings'),
  saveSettings: (p: Partial<Settings>) => req<Settings>('PATCH', '/settings', p),
  models: () => req<Models>('GET', '/models'),
  pull: (kind: ModelKind, id: string) => req('POST', '/models/pull', { kind, id }),
  useRecommended: () => req<Settings>('POST', '/models/recommended'),
  addModel: (kind: ModelKind, ref: string) => req<{ id: string }>('POST', '/models/custom', { kind, ref }),
  removeModel: (kind: ModelKind, id: string) => req('POST', '/models/custom/remove', { kind, id }),
  library: () => req<Video[]>('GET', '/library'),
  add: (paths: string[]) => req<{ added: Video[]; errors: { path: string; error: string }[] }>('POST', '/library', { paths }),
  remove: (id: string) => req('DELETE', `/library/${id}`),
  reindex: (id: string) => req('POST', `/library/${id}/reindex`),
  reindexAll: () => req('POST', '/library/reindex'),
  detail: (id: string) => req<Detail>('GET', `/videos/${id}`),
  proxy: (id: string) => req('POST', `/videos/${id}/proxy`),
  exportText: (id: string, fmt: 'txt' | 'srt' | 'show') => req<string>('GET', `/videos/${id}/export?fmt=${fmt}`),
  search: (q: string, ids: string[] | null) => req<{ results: Result[]; searched: number; ms: number }>('POST', '/search', { q, ids }),
  clip: (id: string, kind: 'video' | 'audio' | 'frame', a: number, b?: number) => req<{ path: string; name: string; size: number }>('POST', '/clips', { id, kind, a, b }),
  dlInfo: (url: string) => req<DlInfo>('POST', '/downloads/info', { url }),
  dlStart: (b: { url: string; quality: string; format: string; title: string; host: string; playlist: boolean }) => req<{ id: string; saveDir: string }>('POST', '/downloads', b),
  dlCancel: (id: string) => req('DELETE', `/downloads/${id}`),
  dlRecent: () => req<DlItem[]>('GET', '/downloads/recent'),
};

// Media and key frames go through "localhost" while API calls use 127.0.0.1. Chromium allows six
// connections per host, so a page full of frames or a buffering video can't starve searches.
const assetBase = () => base.replace('127.0.0.1', 'localhost');
export const mediaUrl = (id: string, proxy: boolean) => `${assetBase()}/videos/${id}/media?token=${token}${proxy ? '&proxy=1' : ''}`;
export const frameUrl = (id: string, f: string) => `${assetBase()}/videos/${id}/frames/${f}?token=${token}`;

export function subscribe(cb: (e: EngineEvent) => void): () => void {
  const es = new EventSource(`${base}/events?token=${token}`);
  es.onmessage = m => {
    try {
      cb(JSON.parse(m.data));
    } catch {}
  };
  return () => es.close();
}
