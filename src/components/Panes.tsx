// SPDX-License-Identifier: GPL-3.0-or-later
import { memo, useEffect, useMemo, useRef, type KeyboardEvent, type RefObject } from 'react';
import { frameUrl, type Detail, type Result, type Segment, type Video } from '../api';
import { fmt, fmtSize, grad, hueOf, pad } from '../format';
import { IconFilm, IconImage, IconReveal, IconSearch, IconWave, IconX } from '../icons';

const STEPS = ['Transcribing speech', 'Pulling key frames', 'Indexing picture + speech'];

export const ScanBar = () => (
  <div style={{ position: 'relative', height: 2, borderRadius: 2, background: '#E4E9F0', overflow: 'hidden' }}>
    <div style={{ position: 'absolute', top: 0, bottom: 0, width: '40%', background: '#2563EB', animation: 'vsScan 1.1s cubic-bezier(.4,0,.2,1) infinite' }} />
  </div>
);

/** "Indexing on this device" card for the active video, plus a count of others in the queue. */
export function ProcCard({ video, others }: { video: Video; others: number }) {
  const queued = video.status === 'queued';
  return (
    <div style={{ flex: 'none', margin: '14px 20px 0', padding: 14, borderRadius: 12, background: '#FFFFFF', border: '1px solid #E4E9F0', boxShadow: '0 1px 2px rgba(20,24,31,.04)', display: 'flex', flexDirection: 'column', gap: 9 }}>
      <span className="eyebrow">{queued ? 'Waiting to index' : 'Indexing on this device'}</span>
      {STEPS.map((label, i) => {
        const done = !queued && i < video.step, act = !queued && i === video.step;
        const dot = done ? '#1F9D57' : act ? '#2563EB' : '#D0D7E2';
        const w = done ? '100%' : act ? Math.round(video.pct * 100) + '%' : '0%';
        return (
          <div key={label} style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5 }}>
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: dot, flex: 'none' }} />
              <span style={{ color: act || done ? '#14181F' : '#6B7584', fontWeight: 500 }}>{label}</span>
              <span className="tnum" style={{ marginLeft: 'auto', fontSize: 11.5, color: '#6B7584', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {done ? 'Done' : act ? video.note || Math.round(video.pct * 100) + '%' : ''}
              </span>
            </div>
            <div style={{ height: 2, borderRadius: 2, background: '#E4E9F0', overflow: 'hidden', marginLeft: 15 }}><div style={{ height: '100%', width: w, background: dot }} /></div>
          </div>
        );
      })}
      {others > 0 && <span style={{ fontSize: 11.5, color: '#6B7584' }}>{others} more {others === 1 ? 'video' : 'videos'} in the queue</span>}
    </div>
  );
}

// ------------------------------------------------------------------ Find

const SUGGESTIONS = ['someone mentions the price', 'a chart or a slide', 'a person talking to the camera', 'the opening title', 'people laughing', 'text on screen'];

type FindProps = {
  q: string;
  qRef: RefObject<HTMLInputElement>;
  onQ: (q: string) => void;
  onSearch: (q?: string) => void;
  searching: boolean;
  results: Result[] | null;
  lastQ: string;
  durLabel: string;
  scope: 'this' | 'all';
  onScope: (s: 'this' | 'all') => void;
  libraryCount: number;
  readyCount: number;
  hasVideo: boolean;
  onPick: (r: Result) => void;
};

export function FindPane(p: FindProps) {
  const many = p.libraryCount > 1;
  const multiVideo = p.scope === 'all' && many;
  const vids = new Set((p.results || []).map(r => r.videoId)).size;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div className="find-line">
        <input
          ref={p.qRef}
          value={p.q}
          onChange={e => p.onQ(e.target.value)}
          onKeyDown={(e: KeyboardEvent) => e.key === 'Enter' && p.onSearch()}
          placeholder="Describe a moment…"
          spellCheck={false}
          className="serif"
          style={{ flex: 1, minWidth: 0, fontSize: 32, lineHeight: 1.1, padding: '4px 0 10px', background: 'transparent', border: 'none', outline: 'none', color: '#14181F' }}
        />
        <button className="search-btn" onClick={() => p.onSearch()} title="Search (Enter)"><IconSearch /></button>
      </div>
      <div style={{ fontSize: 12, color: '#4A5464', lineHeight: 1.5, textWrap: 'pretty' } as React.CSSProperties}>Searches what's said and what's on screen. Nothing leaves this machine.</div>
      {many && (
        <div className="seg-ctl">
          <button className={p.scope === 'this' ? 'on' : ''} onClick={() => p.onScope('this')}>This video</button>
          <button className={p.scope === 'all' ? 'on' : ''} onClick={() => p.onScope('all')}>All {p.libraryCount} videos</button>
        </div>
      )}

      {!p.searching && !p.results && p.hasVideo && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4 }}>
          <span className="eyebrow">Try</span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {SUGGESTIONS.map(s => <button key={s} className="chip" onClick={() => { p.onQ(s); p.onSearch(s); }}>{s}</button>)}
          </div>
        </div>
      )}
      {!p.hasVideo && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 12 }}>
          <span className="serif" style={{ fontSize: 26, lineHeight: 1.05 }}>Open a few videos, <i>then ask</i></span>
          <span style={{ fontSize: 12.5, color: '#4A5464', lineHeight: 1.5, maxWidth: '38ch', textWrap: 'pretty' } as React.CSSProperties}>
            Each one is transcribed and its key frames pulled on this machine. Search one video, or all of them at once.
          </span>
        </div>
      )}

      {p.searching && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 6 }}>
          <ScanBar />
          <span style={{ fontSize: 12.5, color: '#4A5464' }}>
            {multiVideo ? `Looking through ${p.readyCount} videos of speech and picture…` : `Looking through ${p.durLabel} of speech and picture…`}
          </span>
        </div>
      )}

      {p.results && p.results.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <span className="eyebrow">
            {p.results.length} {p.results.length === 1 ? 'moment' : 'moments'} found{multiVideo ? ` in ${vids} ${vids === 1 ? 'video' : 'videos'}` : ''}
          </span>
          {p.results.map((r, i) => (
            <div key={r.videoId + r.t} className="result" onClick={() => p.onPick(r)}>
              <div style={{ position: 'relative', aspectRatio: '16/9', borderRadius: 4, overflow: 'hidden', background: r.frame ? `center / cover no-repeat url("${frameUrl(r.videoId, r.frame)}")` : grad(hueOf(r.videoName)), alignSelf: 'start' }}>
                <span style={{ position: 'absolute', left: 4, top: 4, width: 16, height: 16, borderRadius: '50%', background: '#14181F', color: '#FFFFFF', fontSize: 10, fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{i + 1}</span>
              </div>
              <div style={{ minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                  <span className="tnum" style={{ fontSize: 13, fontWeight: 600 }}>{fmt(r.t)}</span>
                  <span style={{ fontSize: 10, fontWeight: 600, fontStretch: '80%', textTransform: 'uppercase', letterSpacing: '0.07em', color: '#4A5464', whiteSpace: 'nowrap' }}>{r.kind}</span>
                  <span className="tnum" style={{ marginLeft: 'auto', fontSize: 11.5, color: '#6B7584', whiteSpace: 'nowrap' }}>{r.conf}% match</span>
                </div>
                {multiVideo && <span className="eyebrow" style={{ color: '#2563EB', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.videoName}</span>}
                <div style={{ fontSize: 13.5, lineHeight: 1.4, color: '#14181F', textWrap: 'pretty' } as React.CSSProperties}>{r.title}</div>
                <div style={{ fontSize: 12, lineHeight: 1.4, color: '#4A5464', textWrap: 'pretty' } as React.CSSProperties}>{r.snip}</div>
              </div>
            </div>
          ))}
          <span style={{ fontSize: 11.5, color: '#6B7584', marginTop: 2 }}>Selecting a result sets a range there, ready to clip.</span>
        </div>
      )}

      {p.results && p.results.length === 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 6 }}>
          <span className="serif" style={{ fontSize: 26, lineHeight: 1.05 }}>Nothing matched <i>{p.lastQ}</i></span>
          <span style={{ fontSize: 12.5, color: '#4A5464', lineHeight: 1.5, textWrap: 'pretty' } as React.CSSProperties}>
            Try describing what's on screen — a person, an object, a slide — or a phrase someone says.
          </span>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ Transcript

function SegRow({ g, active, onSeek, compact, rowRef }: { g: Segment; active: boolean; onSeek: (t: number) => void; compact?: boolean; rowRef?: RefObject<HTMLDivElement> }) {
  return (
    <div ref={rowRef} className="seg" onClick={() => onSeek(g.t)} style={{ background: active ? '#EAF1FF' : 'transparent', padding: compact ? '6px 8px' : 8 }}>
      <span className="tnum" style={{ fontSize: 11.5, fontWeight: 500, color: active ? '#2563EB' : '#6B7584', paddingTop: 1 }}>{fmt(g.t)}</span>
      <span style={{ fontSize: compact ? 13 : 13.5, lineHeight: 1.5, color: '#14181F', textWrap: 'pretty' } as React.CSSProperties}>{g.x}</span>
    </div>
  );
}

function useFollow(cur: number, playing: boolean) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (playing && ref.current) ref.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [cur, playing]);
  return ref;
}

export const TranscriptPane = memo(function TranscriptPane({ detail, curSeg, playing, onSeek, onExport }: { detail: Detail | null; curSeg: number; playing: boolean; onSeek: (t: number) => void; onExport: (f: 'txt' | 'srt') => void }) {
  const follow = useFollow(curSeg, playing);
  const segs = detail?.transcript ?? [];
  const model = detail?.models.speech;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className="eyebrow">{segs.length} segments{model ? ` · ${model}` : ''}</span>
        <button className="ghost-sm" style={{ marginLeft: 'auto' }} onClick={() => onExport('txt')} disabled={!segs.length}>.txt</button>
        <button className="ghost-sm" onClick={() => onExport('srt')} disabled={!segs.length}>.srt</button>
      </div>
      {!segs.length && <Empty title={<>No speech <i>yet</i></>} body={detail ? (detail.probe.hasAudio ? 'The transcript appears here once speech has been transcribed.' : 'This video has no audio track.') : 'Open a video to transcribe it.'} />}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, margin: '0 -8px' }}>
        {segs.map((g, i) => <SegRow key={i} g={g} active={i === curSeg} onSeek={onSeek} rowRef={i === curSeg ? follow : undefined} />)}
      </div>
    </div>
  );
});

// ------------------------------------------------------------------ Transcribe + Show

export const ShowPane = memo(function ShowPane({ detail, curScene, curSeg, onSeek, onExport }: { detail: Detail | null; curScene: number; curSeg: number; onSeek: (t: number) => void; onExport: () => void }) {
  const scenes = detail?.scenes ?? [];
  const segs = detail?.transcript ?? [];
  // Lines belong to the key frame on screen when they start.
  const byScene = useMemo(() => {
    const out = scenes.map(() => [] as { g: Segment; k: number }[]);
    let si = 0;
    segs.forEach((g, k) => {
      while (si + 1 < scenes.length && scenes[si + 1].t <= g.t) si++;
      out[si]?.push({ g, k });
    });
    return out;
  }, [scenes, segs]);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 12, color: '#4A5464', lineHeight: 1.5, textWrap: 'pretty', flex: 1 } as React.CSSProperties}>Transcript with a key frame pulled at every scene change, so each line has its picture.</span>
        <button className="ghost-sm" onClick={onExport} disabled={!scenes.length}>.txt</button>
      </div>
      {!scenes.length && <Empty title={<>No key frames <i>yet</i></>} body={detail ? 'Key frames appear here as soon as they are pulled.' : 'Open a video to pull its key frames.'} />}
      {scenes.map((s, i) => (
        <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div onClick={() => onSeek(s.t + 0.01)} style={{ position: 'relative', aspectRatio: '16/9', borderRadius: 6, overflow: 'hidden', background: `center / cover no-repeat url("${frameUrl(detail!.id, s.f)}"), ${grad(210)}`, cursor: 'pointer', outline: `2px solid ${i === curScene ? '#2563EB' : 'transparent'}`, outlineOffset: 2 }}>
            <div className="tnum" style={{ position: 'absolute', left: 10, top: 9, right: 10, display: 'flex', justifyContent: 'space-between', fontSize: 10, fontWeight: 600, fontStretch: '80%', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#2A3240' }}>
              <span style={{ padding: '3px 7px', borderRadius: 999, background: 'rgba(255,255,255,.88)' }}>Key frame {pad(i + 1)}</span>
              <span style={{ padding: '3px 7px', borderRadius: 999, background: 'rgba(255,255,255,.88)' }}>{fmt(s.t)}</span>
            </div>
          </div>
          {s.d && <div className="serif" style={{ fontSize: 22, lineHeight: 1.1, color: '#14181F', textWrap: 'balance' } as React.CSSProperties}>{s.d}</div>}
          {byScene[i].length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2, margin: '0 -8px' }}>
              {byScene[i].map(({ g, k }) => <SegRow key={k} g={g} active={k === curSeg} onSeek={onSeek} compact />)}
            </div>
          )}
        </div>
      ))}
    </div>
  );
});

// ------------------------------------------------------------------ Clips

export type Clip = { id: string; kind: 'Video' | 'Audio' | 'Frame'; videoId: string; videoName: string; a: number; b: number | null; path: string; name: string; size: number };

export function ClipsPane({ clips, onReveal, onSeek, onRemove }: { clips: Clip[]; onReveal: (c: Clip) => void; onSeek: (c: Clip) => void; onRemove: (c: Clip) => void }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!clips.length && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 24 }}>
          <span className="serif" style={{ fontSize: 30, lineHeight: 1 }}>No clips <i>yet</i></span>
          <span style={{ fontSize: 12.5, color: '#4A5464', lineHeight: 1.5, maxWidth: '34ch', textWrap: 'pretty' } as React.CSSProperties}>
            Select a range on the timeline, then clip it as video, pull the audio, or save the frame under the playhead.
          </span>
        </div>
      )}
      {clips.map(c => (
        <div key={c.id} className="card" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 34, height: 34, flex: 'none', borderRadius: 6, background: '#EEF2F7', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#2563EB' }}>
            {c.kind === 'Video' ? <IconFilm size={16} /> : c.kind === 'Audio' ? <IconWave size={16} /> : <IconImage size={16} />}
          </div>
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <span className="tnum" style={{ fontSize: 13, fontWeight: 500 }}>{c.kind} · {c.b != null ? `${fmt(c.a)} – ${fmt(c.b)}` : fmt(c.a)}</span>
            <span className="tnum" style={{ fontSize: 11.5, color: '#6B7584', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {[c.b != null ? (c.b - c.a).toFixed(1) + ' s' : null, fmtSize(c.size), c.videoName].filter(Boolean).join(' · ')}
            </span>
          </div>
          <button className="sq-btn" title="Show in folder" onClick={() => onReveal(c)}><IconReveal /></button>
          <button className="text-btn" title="Jump to clip" onClick={() => onSeek(c)}>Jump</button>
          <button className="x-btn" title="Remove from list" onClick={() => onRemove(c)}><IconX /></button>
        </div>
      ))}
    </div>
  );
}

function Empty({ title, body }: { title: React.ReactNode; body: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 16 }}>
      <span className="serif" style={{ fontSize: 26, lineHeight: 1.05 }}>{title}</span>
      <span style={{ fontSize: 12.5, color: '#4A5464', lineHeight: 1.5, maxWidth: '36ch', textWrap: 'pretty' } as React.CSSProperties}>{body}</span>
    </div>
  );
}
