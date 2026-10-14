import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { api, configure, frameUrl, mediaUrl, subscribe, type Detail, type DlItem, type EngineEvent, type EngineState, type Hardware, type Result, type Settings, type Video } from './api';
import { DownloadDialog, initialDl, type DlState } from './components/DownloadDialog';
import { EngineGate } from './components/EngineGate';
import { LibraryStrip } from './components/LibraryStrip';
import { ModelsDialog, type Pulls } from './components/ModelsDialog';
import { ClipsPane, FindPane, ProcCard, ShowPane, TranscriptPane, type Clip } from './components/Panes';
import { Stage } from './components/Stage';
import { Timeline } from './components/Timeline';
import { TitleBar, detectOS } from './components/TitleBar';
import { fmt, fmtD, fmtSize, stem } from './format';
import { native } from './native';
import { IconBack10, IconDownload, IconFilm, IconFolder, IconFwd10, IconImage, IconNext, IconPause, IconPlay, IconPrev, IconWave } from './icons';

type Tab = 'find' | 'transcript' | 'show' | 'clips';
const TABS: [Tab, string][] = [['find', 'Find'], ['transcript', 'Transcript'], ['show', 'Transcribe + Show'], ['clips', 'Clips']];

/** Index of the last item starting at or before t (items sorted by t). */
function indexAt(items: { t: number }[], t: number): number {
  let lo = 0, hi = items.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].t <= t + 0.001) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

const upsert = (vs: Video[], v: Video) => (vs.some(x => x.id === v.id) ? vs.map(x => (x.id === v.id ? v : x)) : [...vs, v]);
const isBusy = (v: Video) => v.status === 'indexing' || v.status === 'queued';

function saveText(body: string, name: string) {
  const url = URL.createObjectURL(new Blob([body], { type: 'text/plain' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function App() {
  const os = useMemo(detectOS, []);
  const [engine, setEngine] = useState<EngineState | null>(null);
  const [ready, setReady] = useState(false);
  const [hw, setHw] = useState<Hardware | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, Detail>>({});
  const [tab, setTab] = useState<Tab>('find');
  const [q, setQ] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<Result[] | null>(null);
  const [lastQ, setLastQ] = useState('');
  const [scope, setScope] = useState<'this' | 'all'>('this');
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [inT, setInT] = useState<number | null>(null);
  const [outT, setOutT] = useState<number | null>(null);
  const [clips, setClips] = useState<Clip[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('vs.clips') || '[]');
    } catch {
      return [];
    }
  });
  const [busy, setBusy] = useState<{ kind: 'Video' | 'Audio'; pct: number } | null>(null);
  const [toast, setToastText] = useState<string | null>(null);
  const [dl, setDlState] = useState<DlState>(initialDl);
  const [modelsOpen, setModelsOpen] = useState(false);
  const [pulls, setPulls] = useState<Pulls>({});
  const [proxied, setProxied] = useState<Record<string, boolean>>({});

  const videoRef = useRef<HTMLVideoElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const qRef = useRef<HTMLInputElement>(null);
  const raf = useRef(0);
  const toastTimer = useRef(0);
  const searchTok = useRef(0);
  const pendingSeek = useRef<{ t: number; inT?: number; outT?: number } | null>(null);
  const detailTimers = useRef<Record<string, number>>({});

  const active = videos.find(v => v.id === activeId) ?? null;
  const detail = activeId ? details[activeId] ?? null : null;
  const scenes = detail?.scenes ?? [];
  const segs = detail?.transcript ?? [];
  const D = dur || active?.probe.duration || 0;
  const curScene = Math.max(0, indexAt(scenes, t));
  const curSeg = indexAt(segs, t);
  const useProxy = !!active && (!!proxied[active.id] || !!active.proxy);
  const src = active && active.exists ? mediaUrl(active.id, useProxy) : null;

  const setDl = useCallback((p: Partial<DlState>) => setDlState(d => ({ ...d, ...p })), []);
  const say = useCallback((msg: string) => {
    clearTimeout(toastTimer.current);
    setToastText(msg);
    toastTimer.current = window.setTimeout(() => setToastText(null), 3600);
  }, []);

  // ---------------------------------------------------------------- engine

  useEffect(() => {
    native.engine().then(setEngine);
    return native.onEngine(setEngine);
  }, []);

  const refreshDetail = useCallback((id: string) => {
    clearTimeout(detailTimers.current[id]);
    detailTimers.current[id] = window.setTimeout(() => {
      api.detail(id).then(d => setDetails(x => ({ ...x, [id]: d }))).catch(() => {});
    }, 250);
  }, []);

  const live = useRef({ activeId, dl, details });
  live.current = { activeId, dl, details };

  const onEvent = useRef<(e: EngineEvent) => void>(() => {});
  onEvent.current = e => {
    switch (e.type) {
      case 'video':
        setVideos(vs => upsert(vs, e.video));
        if (!isBusy(e.video)) refreshDetail(e.video.id);
        break;
      case 'removed':
        setVideos(vs => vs.filter(v => v.id !== e.id));
        break;
      case 'transcript':
      case 'scenes':
        if (e.id === live.current.activeId || live.current.details[e.id]) refreshDetail(e.id);
        break;
      case 'model':
        setPulls(p => ({ ...p, [`${e.kind}:${e.id}`]: { pct: e.pct ?? 0, state: e.state, error: e.error } }));
        break;
      case 'proxy':
        if (e.done) {
          setProxied(p => ({ ...p, [e.id]: true }));
          say('Ready to play');
        } else if (e.error) say('Couldn’t convert this video: ' + e.error);
        break;
      case 'clip':
        setBusy(b => (b ? { ...b, pct: e.pct } : b));
        break;
      case 'download': {
        const cur = live.current.dl;
        if (cur.jobId && e.id !== cur.jobId) break;
        if (e.state === 'downloading') setDl({ got: e.got ?? 0, total: e.total ?? 0, speed: e.speed ?? 0, item: e.item ?? 1, count: e.count ?? 1 });
        else if (e.state === 'done') {
          const items = e.items ?? [];
          setDl({ phase: 'done', last: items, jobId: null });
          api.dlRecent().then(recent => setDl({ recent })).catch(() => {});
          if (cur.openAfter) openDownloaded(items);
          else say('Download finished');
        } else if (e.state === 'cancelled') setDl({ phase: 'info', jobId: null });
        else if (e.state === 'error') setDl({ phase: 'info', jobId: null, err: e.error || 'Download failed' });
        break;
      }
    }
  };

  useEffect(() => {
    if (engine?.state !== 'ready' || !engine.url) {
      setReady(false);
      return;
    }
    configure(engine.url, engine.token);
    let alive = true;
    Promise.all([api.settings(), api.library(), api.dlRecent()])
      .then(([s, lib, recent]) => {
        if (!alive) return;
        setSettings(s);
        setVideos(lib);
        setDl({ recent });
        setActiveId(a => a ?? lib[lib.length - 1]?.id ?? null);
        setReady(true);
      })
      .catch(err => say('Couldn’t reach the local engine: ' + err.message));
    api.hardware().then(h => alive && setHw(h)).catch(() => {});
    const unsub = subscribe(e => onEvent.current(e));
    return () => {
      alive = false;
      unsub();
    };
  }, [engine?.state, engine?.url, engine?.token, say, setDl]);

  // ---------------------------------------------------------------- library

  const addPaths = useCallback(async (paths: string[]) => {
    if (!paths.length) return;
    try {
      const r = await api.add(paths);
      setVideos(vs => r.added.reduce(upsert, vs));
      if (r.added.length) setActiveId(r.added[0].id);
      if (r.errors.length) say(r.errors.length === 1 ? r.errors[0].error : `${r.errors.length} files couldn’t be opened`);
      else if (r.added.length > 1) say(`Indexing ${r.added.length} videos on this device…`);
    } catch (err) {
      say((err as Error).message);
    }
  }, [say]);

  useEffect(() => {
    if (!ready) return;
    const a = native.onOpenPaths(addPaths), b = native.onDropPaths(addPaths);
    return () => {
      a();
      b();
    };
  }, [ready, addPaths]);

  const openFile = async () => {
    if (native.available) addPaths(await native.openFiles());
    else say('Opening files needs the desktop app');
  };

  const removeVideo = async (v: Video) => {
    await api.remove(v.id).catch(() => {});
    setVideos(vs => {
      const rest = vs.filter(x => x.id !== v.id);
      if (activeId === v.id) setActiveId(rest[rest.length - 1]?.id ?? null);
      return rest;
    });
    setResults(r => r && r.filter(x => x.videoId !== v.id));
    say(`Removed ${v.name} from the library. The file wasn’t touched.`);
  };

  // Switching videos resets playback; a pending seek (from a search result) is applied on load.
  useEffect(() => {
    cancelAnimationFrame(raf.current);
    setPlaying(false);
    setT(0);
    setDur(0);
    setInT(null);
    setOutT(null);
    if (scope === 'this') setResults(null);
    if (activeId) refreshDetail(activeId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  useEffect(() => localStorage.setItem('vs.clips', JSON.stringify(clips.slice(0, 200))), [clips]);

  // ---------------------------------------------------------------- playback

  const seek = useCallback((to: number) => {
    const v = videoRef.current;
    const max = v && isFinite(v.duration) ? v.duration : D;
    to = Math.max(0, Math.min(max || 0, to));
    if (v) v.currentTime = to;
    setT(to);
  }, [D]);

  const loop = () => {
    cancelAnimationFrame(raf.current);
    const f = () => {
      const v = videoRef.current;
      if (!v) return;
      setT(v.currentTime);
      if (!v.paused && !v.ended) raf.current = requestAnimationFrame(f);
      else setPlaying(false);
    };
    raf.current = requestAnimationFrame(f);
  };

  const play = () => {
    const v = videoRef.current;
    if (!v || busy) return;
    if (!v.paused) {
      v.pause();
      setPlaying(false);
      return;
    }
    if (v.ended || v.currentTime >= v.duration - 0.05) v.currentTime = 0;
    v.play().then(() => { setPlaying(true); loop(); }).catch(() => setPlaying(false));
  };

  const onMeta = () => {
    const v = videoRef.current;
    if (!v) return;
    setDur(isFinite(v.duration) ? v.duration : active?.probe.duration || 0);
    const p = pendingSeek.current;
    pendingSeek.current = null;
    if (p) {
      v.currentTime = p.t;
      setT(p.t);
      if (p.inT != null) {
        setInT(p.inT);
        setOutT(p.outT ?? null);
      }
    }
  };

  const onVideoError = () => {
    if (!active) return;
    if (useProxy) {
      say('This video can’t be played, even after converting it.');
      return;
    }
    api.proxy(active.id).catch(() => {});
    say('This format can’t play here directly. Converting a copy for playback…');
  };

  const setIn = () => {
    setInT(t);
    setOutT(o => (o != null && o > t ? o : Math.min(D, t + 10)));
  };
  const setOut = () => {
    setOutT(t);
    setInT(i => (i != null && i < t ? i : Math.max(0, t - 10)));
  };

  const onTrackDown = (e: PointerEvent<HTMLDivElement>) => {
    const el = trackRef.current;
    if (!el || busy || !D) return;
    const r = el.getBoundingClientRect();
    const toT = (x: number) => Math.max(0, Math.min(1, (x - r.left) / r.width)) * D;
    const x0 = e.clientX, t0 = toT(x0);
    let drag = false;
    seek(t0);
    const mv = (ev: globalThis.PointerEvent) => {
      if (!drag && Math.abs(ev.clientX - x0) > 4) drag = true;
      if (drag) {
        const tt = toT(ev.clientX);
        setInT(Math.min(t0, tt));
        setOutT(Math.max(t0, tt));
        seek(tt);
      }
    };
    const up = () => {
      window.removeEventListener('pointermove', mv);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
  };

  // ---------------------------------------------------------------- search

  const readyCount = videos.filter(v => v.status === 'ready').length;
  const runSearch = async (qq?: string) => {
    const query = (qq ?? q).trim();
    if (!query) {
      qRef.current?.focus();
      return;
    }
    if (!videos.length) {
      say('Open a video first');
      return;
    }
    const all = scope === 'all' && videos.length > 1;
    if (!all && active && isBusy(active)) {
      say('Still indexing — search will be ready in a moment');
      return;
    }
    setQ(query);
    setSearching(true);
    setResults(null);
    setLastQ(query);
    setTab('find');
    const tok = ++searchTok.current;
    try {
      const r = await api.search(query, all ? null : activeId ? [activeId] : null);
      if (tok === searchTok.current) setResults(r.results);
    } catch (err) {
      if (tok === searchTok.current) {
        setResults([]);
        say((err as Error).message);
      }
    } finally {
      if (tok === searchTok.current) setSearching(false);
    }
  };

  const pickResult = (r: Result) => {
    const a = r.t, b = Math.min(r.t + 30, Math.max(r.e, r.t + 5));
    if (r.videoId !== activeId) {
      pendingSeek.current = { t: a, inT: a, outT: b };
      setActiveId(r.videoId);
    } else {
      seek(a);
      setInT(a);
      setOutT(Math.min(D, b));
    }
  };

  // ---------------------------------------------------------------- clips and exports

  const addClip = (c: Omit<Clip, 'id'>) => setClips(cs => [{ id: Date.now() + '-' + Math.random().toString(36).slice(2), ...c }, ...cs]);

  const exportRange = async (kind: 'Video' | 'Audio') => {
    if (busy || !active) return;
    if (inT == null || outT == null || outT - inT < 0.2) {
      say('Select a range first: drag across the timeline, or press I and O.');
      return;
    }
    videoRef.current?.pause();
    setBusy({ kind, pct: 0 });
    try {
      const r = await api.clip(active.id, kind === 'Video' ? 'video' : 'audio', inT, outT);
      addClip({ kind, videoId: active.id, videoName: active.name, a: inT, b: outT, path: r.path, name: r.name, size: r.size });
      say(`${kind} clip saved to ${settings?.saveDir ?? 'your videos folder'}/Clips`);
    } catch (err) {
      say((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const saveFrame = async () => {
    if (!active) return;
    try {
      const r = await api.clip(active.id, 'frame', t);
      addClip({ kind: 'Frame', videoId: active.id, videoName: active.name, a: t, b: null, path: r.path, name: r.name, size: r.size });
      say('Frame saved as PNG');
    } catch (err) {
      say((err as Error).message);
    }
  };

  const exportText = async (f: 'txt' | 'srt' | 'show') => {
    if (!active) return;
    const body = await api.exportText(active.id, f);
    saveText(body, stem(active.name) + (f === 'srt' ? '.srt' : f === 'show' ? ' — key frames.txt' : '.txt'));
  };

  const jumpToClip = (c: Clip) => {
    if (c.videoId !== activeId) {
      if (!videos.some(v => v.id === c.videoId)) return say('That video is no longer in the library');
      pendingSeek.current = { t: c.a, inT: c.b != null ? c.a : undefined, outT: c.b ?? undefined };
      setActiveId(c.videoId);
    } else {
      seek(c.a);
      if (c.b != null) {
        setInT(c.a);
        setOutT(c.b);
      }
    }
  };

  // ---------------------------------------------------------------- downloads

  const fetchInfo = async () => {
    const raw = dl.url.trim();
    if (!raw) return setDl({ err: 'Paste a link first.' });
    try {
      const u = new URL(/^https?:\/\//i.test(raw) ? raw : 'https://' + raw);
      if (!u.hostname.includes('.')) throw new Error();
    } catch {
      return setDl({ phase: 'idle', info: null, err: 'That doesn’t look like a link. Paste a full URL.' });
    }
    setDl({ phase: 'fetching', err: '', info: null });
    try {
      const info = await api.dlInfo(raw);
      setDl({ phase: 'info', info, q: 'Best', f: 'MP4' });
    } catch (err) {
      setDl({ phase: 'idle', err: (err as Error).message });
    }
  };

  const startDl = async () => {
    const info = dl.info;
    if (!info) return;
    setDl({ phase: 'downloading', got: 0, total: 0, speed: 0, item: 1, count: info.count, err: '', jobId: null });
    try {
      const r = await api.dlStart({ url: info.url, quality: dl.q, format: dl.f, title: info.title, host: info.host, playlist: info.playlist });
      setDl({ jobId: r.id });
    } catch (err) {
      setDl({ phase: 'info', err: (err as Error).message });
    }
  };

  const cancelDl = () => {
    if (dl.jobId) api.dlCancel(dl.jobId).catch(() => {});
    setDl({ phase: 'info', jobId: null });
  };

  const openDownloaded = (items: DlItem[]) => {
    addPaths(items.map(i => i.path));
    setDl({ open: false, phase: 'idle', info: null, url: '' });
    say(items.length > 1 ? `Downloaded ${items.length} videos. Indexing on this device…` : 'Downloaded. Indexing on this device…');
  };

  // ---------------------------------------------------------------- keyboard

  const keyHandler = useRef<(e: KeyboardEvent) => void>(() => {});
  keyHandler.current = e => {
    if (dl.open || modelsOpen) {
      if (e.key === 'Escape') {
        setDl({ open: false });
        setModelsOpen(false);
      }
      return;
    }
    const tag = ((e.target as HTMLElement).tagName || '').toLowerCase(), mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      focusFind();
      return;
    }
    if (mod && e.key.toLowerCase() === 'o') {
      e.preventDefault();
      openFile();
      return;
    }
    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      if (e.key === 'Escape') (e.target as HTMLElement).blur();
      return;
    }
    if (mod || !active) return;
    const fps = active.probe.fps || 30;
    const step = e.shiftKey ? 5 : 1 / fps;
    switch (e.key) {
      case ' ': e.preventDefault(); play(); break;
      case 'i': case 'I': setIn(); break;
      case 'o': case 'O': setOut(); break;
      case 'ArrowLeft': e.preventDefault(); seek(t - step); break;
      case 'ArrowRight': e.preventDefault(); seek(t + step); break;
      case 'j': case 'J': seek(t - 10); break;
      case 'l': case 'L': seek(t + 10); break;
      case '/': e.preventDefault(); focusFind(); break;
      case 'Escape': setInT(null); setOutT(null); break;
    }
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => keyHandler.current(e);
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, []);
  const focusFind = () => {
    setTab('find');
    setTimeout(() => qRef.current?.focus(), 0);
  };

  // ---------------------------------------------------------------- view

  const indexingCount = videos.filter(isBusy).length;
  const engineUp = engine?.state === 'ready' && ready;
  const statusLabel = !engine || engine.state === 'starting' ? 'Starting' : engine.state === 'installing' ? 'Setting up' : engine.state === 'error' ? 'Engine stopped' : indexingCount ? (indexingCount > 1 ? `Indexing ${indexingCount}` : 'Indexing') : 'On-device';
  const pulsing = !engineUp || indexingCount > 0;
  const modelLine = settings ? [settings.speech, settings.caption !== 'none' ? settings.caption.replace(/^ollama:/, '') : settings.vision].join(' · ') : 'local models';
  const hasRange = inT != null && outT != null;
  const others = videos.filter(v => v.id !== activeId && isBusy(v)).length;
  const firstFrame = (v: Video) => {
    const s = details[v.id]?.scenes[0];
    return s ? frameUrl(v.id, s.f) : null;
  };
  const fileMeta = active
    ? [active.probe.width ? `${active.probe.width}×${active.probe.height}` : 'audio', active.probe.fps ? `${Math.round(active.probe.fps)} fps` : null, fmt(active.probe.duration), fmtSize(active.size)].filter(Boolean).join(' · ')
    : '';
  const markers = (results || []).map((r, i) => ({ t: r.t, n: i + 1, v: r.videoId })).filter(r => r.v === activeId);

  return (
    <div className="app">
      <TitleBar os={os} />

      <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 14, padding: '12px 24px', borderBottom: '1px solid #E4E9F0', background: '#FFFFFF', flexWrap: 'wrap' }}>
        <button className="btn" onClick={openFile} style={{ gap: 8 }}><IconFolder />Open video</button>
        <button className="btn-tint" onClick={() => setDl({ open: true })}>
          <IconDownload />
          {dl.phase === 'downloading' ? 'Downloading ' + (dl.total ? Math.floor((dl.got / dl.total) * 100) + '%' : '…') : 'Download'}
        </button>
        <div style={{ minWidth: 0, flex: '1 1 220px', display: 'flex', alignItems: 'baseline', gap: 10 }}>
          <span style={{ fontSize: 13.5, fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', minWidth: 0 }}>{active ? active.name : 'No video open'}</span>
          <span className="eyebrow tnum" style={{ whiteSpace: 'nowrap' }}>{fileMeta}</span>
        </div>
        <button className="status-pill" onClick={() => engineUp && setModelsOpen(true)} title="Local models and hardware">
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: engine?.state === 'error' ? '#B42318' : pulsing ? '#2563EB' : '#1F9D57', animation: pulsing && engine?.state !== 'error' ? 'vsPulse 1s ease-in-out infinite' : 'none', flex: 'none' }} />
          <span style={{ fontSize: 12, fontWeight: 500 }}>{statusLabel}</span>
          <span className="eyebrow" style={{ whiteSpace: 'nowrap' }}>{modelLine}</span>
        </button>
      </div>

      {videos.length > 0 && (
        <LibraryStrip videos={videos} activeId={activeId} firstFrame={firstFrame} onPick={setActiveId} onRemove={removeVideo} onAdd={openFile} />
      )}

      <div style={{ flex: '1 1 auto', minHeight: 0, display: 'flex', flexWrap: 'wrap', overflow: 'auto' }}>
        <div style={{ flex: '3 1 560px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 16, padding: 24 }}>
          <Stage video={active} detail={detail} src={src} t={t} curScene={curScene} videoRef={videoRef} onMeta={onMeta} onEnded={() => setPlaying(false)} onError={onVideoError} onPlay={play} onOpen={openFile} />

          <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            <button className="icon-btn" onClick={() => seek(t - 10)} title="Back 10 s (J)"><IconBack10 /></button>
            <button className="icon-btn" onClick={() => seek(t - 1 / (active?.probe.fps || 30))} title="Previous frame (←)"><IconPrev /></button>
            <button className="play-btn" onClick={play} title="Play / pause (Space)">{playing ? <IconPause /> : <IconPlay />}</button>
            <button className="icon-btn" onClick={() => seek(t + 1 / (active?.probe.fps || 30))} title="Next frame (→)"><IconNext /></button>
            <button className="icon-btn" onClick={() => seek(t + 10)} title="Forward 10 s (L)"><IconFwd10 /></button>
            <div className="tnum" style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginLeft: 8, whiteSpace: 'nowrap' }}>
              <span style={{ fontSize: 16, fontWeight: 500 }}>{fmtD(t)}</span>
              <span style={{ fontSize: 12.5, color: '#6B7584' }}>/ {fmt(D)}</span>
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8, color: '#6B7584', fontSize: 11.5 }}>
              <span>Find</span><span className="kbd">{os === 'macOS' ? '⌘K' : 'Ctrl K'}</span>
              <span style={{ marginLeft: 6 }}>Play</span><span className="kbd">Space</span>
            </div>
          </div>

          <Timeline videoId={activeId} dur={Math.max(D, 0.001)} t={t} scenes={scenes} segs={segs} results={markers} inT={inT} outT={outT} showPictures={!active || active.probe.hasVideo} trackRef={trackRef} onDown={onTrackDown} />

          <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', minHeight: 36 }}>
            {busy ? (
              <div style={{ flex: '1 1 240px', display: 'flex', alignItems: 'center', gap: 12 }}>
                <span className="tnum" style={{ fontSize: 12.5, fontWeight: 500, whiteSpace: 'nowrap' }}>{busy.kind === 'Video' ? 'Exporting video clip' : 'Extracting audio'} · {Math.round(busy.pct * 100)}%</span>
                <div style={{ flex: 1, height: 3, borderRadius: 2, background: '#E4E9F0', overflow: 'hidden' }}><div style={{ height: '100%', width: Math.round(busy.pct * 100) + '%', background: '#2563EB' }} /></div>
              </div>
            ) : (
              <div style={{ flex: '1 1 240px', minWidth: 0, display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                {hasRange ? (
                  <>
                    <span className="eyebrow" style={{ color: '#2563EB' }}>Range</span>
                    <span className="tnum" style={{ fontSize: 13, fontWeight: 500 }}>{fmtD(inT!)} → {fmtD(outT!)}</span>
                    <span className="tnum" style={{ fontSize: 12, color: '#6B7584' }}>{(outT! - inT!).toFixed(1)} s</span>
                    <button className="link-btn" onClick={() => { setInT(null); setOutT(null); }}>Clear</button>
                  </>
                ) : (
                  <>
                    <span style={{ fontSize: 12, color: '#6B7584' }}>Drag across the timeline to select, or set in and out with</span>
                    <span className="kbd">I</span><span className="kbd">O</span>
                  </>
                )}
              </div>
            )}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <button className="btn" onClick={() => exportRange('Video')} disabled={!active || !active.probe.hasVideo}><IconFilm />Clip video</button>
              <button className="btn" onClick={() => exportRange('Audio')} disabled={!active || !active.probe.hasAudio}><IconWave />Extract audio</button>
              <button className="btn" onClick={saveFrame} disabled={!active || !active.probe.hasVideo}><IconImage />Save frame</button>
            </div>
          </div>
        </div>

        <div style={{ flex: '1 1 360px', minWidth: 0, display: 'flex', flexDirection: 'column', borderLeft: '1px solid #E4E9F0', background: '#FAFBFD' }}>
          <div style={{ flex: 'none', padding: '20px 20px 0' }}>
            <div style={{ display: 'flex', gap: 2, padding: 3, borderRadius: 10, background: '#EEF2F7', border: '1px solid #E4E9F0' }}>
              {TABS.map(([id, label]) => (
                <button key={id} onClick={() => setTab(id)} style={{ flex: '1 1 auto', height: 28, padding: '0 10px', border: 'none', borderRadius: 6, background: id === tab ? '#FFFFFF' : 'transparent', color: id === tab ? '#14181F' : '#4A5464', boxShadow: id === tab ? '0 1px 2px rgba(20,24,31,.08), 0 0 0 1px #E4E9F0' : 'none', cursor: 'pointer', fontSize: 12.5, fontWeight: 500, whiteSpace: 'nowrap' }}>
                  {label}{id === 'clips' && clips.length ? ' · ' + clips.length : ''}
                </button>
              ))}
            </div>
          </div>

          {active && isBusy(active) && <ProcCard video={active} others={others} />}
          {active && active.status === 'error' && (
            <div style={{ flex: 'none', margin: '14px 20px 0', padding: 14, borderRadius: 12, background: '#FFFFFF', border: '1px solid #F3C8C3', display: 'flex', flexDirection: 'column', gap: 8 }}>
              <span className="eyebrow" style={{ color: '#B42318' }}>Indexing stopped</span>
              <span style={{ fontSize: 12.5, color: '#2A3240', lineHeight: 1.5, overflowWrap: 'anywhere' }}>{active.error}</span>
              <button className="plain-btn" style={{ alignSelf: 'flex-start' }} onClick={() => api.reindex(active.id)}>Try again</button>
            </div>
          )}
          {active && active.status === 'ready' && active.warnings.length > 0 && (
            <div style={{ flex: 'none', margin: '14px 20px 0', padding: '10px 14px', borderRadius: 12, background: '#FFF8EB', border: '1px solid #F5DDB0', fontSize: 12, color: '#6B4A0F', lineHeight: 1.5, overflowWrap: 'anywhere' }}>
              {active.warnings.map(w => <div key={w}>{w.split('\n')[0]}</div>)}
            </div>
          )}

          <div style={{ flex: '1 1 auto', minHeight: 440, position: 'relative' }}>
            <div style={{ position: 'absolute', inset: 0, overflow: 'auto', padding: '20px 20px 32px' }}>
              {tab === 'find' && (
                <FindPane q={q} qRef={qRef} onQ={setQ} onSearch={runSearch} searching={searching} results={results} lastQ={lastQ}
                  durLabel={fmt(D)} scope={scope} onScope={s => { setScope(s); setResults(null); }} libraryCount={videos.length} readyCount={readyCount}
                  hasVideo={videos.length > 0} onPick={pickResult} />
              )}
              {tab === 'transcript' && <TranscriptPane detail={detail} curSeg={curSeg} playing={playing} onSeek={seek} onExport={exportText} />}
              {tab === 'show' && <ShowPane detail={detail} curScene={curScene} curSeg={curSeg} onSeek={seek} onExport={() => exportText('show')} />}
              {tab === 'clips' && (
                <ClipsPane clips={clips} onReveal={c => native.reveal(c.path)} onSeek={jumpToClip} onRemove={c => setClips(cs => cs.filter(x => x.id !== c.id))} />
              )}
            </div>
          </div>
        </div>
      </div>

      {dl.open && engineUp && (
        <DownloadDialog s={dl} saveDir={settings?.saveDir ?? ''} set={setDl} onClose={() => setDl({ open: false })} onFetch={fetchInfo} onStart={startDl} onCancel={cancelDl} onOpen={openDownloaded} />
      )}
      {modelsOpen && settings && (
        <ModelsDialog hw={hw} settings={settings} pulls={pulls} videoCount={videos.length} onSettings={setSettings}
          onReindexAll={() => { api.reindexAll(); say('Re-indexing every video with the new models'); }} onClose={() => setModelsOpen(false)} />
      )}
      {engine && <EngineGate engine={engine} />}

      {toast && (
        <div style={{ position: 'fixed', left: '50%', bottom: 22, transform: 'translateX(-50%)', maxWidth: 'min(520px, calc(100vw - 32px))', padding: '10px 16px', borderRadius: 8, background: '#14181F', border: '1px solid #14181F', boxShadow: '0 12px 32px rgba(20,24,31,.18)', fontSize: 12.5, color: '#FFFFFF', zIndex: 30, textWrap: 'pretty', overflowWrap: 'anywhere' } as React.CSSProperties}>{toast}</div>
      )}
    </div>
  );
}
