import { useEffect, useRef } from 'react';
import type { DlInfo, DlItem } from '../api';
import { fmt, fmtSize, grad, hueOf } from '../format';
import { IconCheck, IconDownload } from '../icons';
import { Dialog } from './Dialog';
import { ScanBar } from './Panes';

export type DlState = {
  open: boolean;
  url: string;
  phase: 'idle' | 'fetching' | 'info' | 'downloading' | 'done';
  info: DlInfo | null;
  q: string;
  f: string;
  openAfter: boolean;
  got: number;
  total: number;
  speed: number;
  item: number;
  count: number;
  err: string;
  jobId: string | null;
  recent: DlItem[];
  last: DlItem[];
};

export const initialDl: DlState = { open: false, url: '', phase: 'idle', info: null, q: 'Best', f: 'MP4', openAfter: true, got: 0, total: 0, speed: 0, item: 1, count: 1, err: '', jobId: null, recent: [], last: [] };

type Props = {
  s: DlState;
  saveDir: string;
  set: (p: Partial<DlState>) => void;
  onClose: () => void;
  onFetch: () => void;
  onStart: () => void;
  onCancel: () => void;
  onOpen: (items: DlItem[]) => void;
};

const QUALITIES = ['Best', '1080p', '720p', 'Audio only'];

export function DownloadDialog({ s, saveDir, set, onClose, onFetch, onStart, onCancel, onOpen }: Props) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    setTimeout(() => ref.current?.focus(), 40);
  }, []);
  const info = s.info;
  const pctStr = s.total ? Math.floor((s.got / s.total) * 100) + '%' : fmtSize(s.got || 1);
  const detail =
    (s.count > 1 ? `Video ${s.item} of ${s.count} · ` : '') +
    (s.total ? fmtSize(s.got || 1) + ' of ' + fmtSize(s.total) : 'received') +
    (s.speed ? ' · ' + fmtSize(s.speed) + '/s' : '') +
    (s.total && s.speed ? ' · ' + Math.max(0, Math.ceil((s.total - s.got) / s.speed)) + ' s left' : '');
  const formats = s.q === 'Audio only' ? ['M4A', 'MP3'] : ['MP4', 'WebM'];
  const meta = info ? [info.host, info.playlist ? `Playlist · ${info.count} videos` : info.direct ? 'direct file' : null, info.duration ? fmt(info.duration) : null].filter(Boolean).join(' · ') : '';

  return (
    <Dialog title={<>Download a <i>video</i></>} sub="Paste a link. The file downloads on this machine, then opens here to be transcribed and indexed." onClose={onClose}>
      <div style={{ display: 'flex', gap: 8 }}>
        <input ref={ref} className="url-input" value={s.url} onChange={e => set({ url: e.target.value, err: '' })} onKeyDown={e => e.key === 'Enter' && onFetch()} placeholder="https://…" spellCheck={false} />
        <button className="dark-btn" onClick={onFetch}>Get video</button>
      </div>
      {s.phase === 'idle' && !s.err && (
        <span style={{ fontSize: 12.5, color: '#6B7584', lineHeight: 1.5, marginTop: -10 }}>Works with direct files (.mp4, .webm, .mov), links from most video sites, and whole playlists.</span>
      )}
      {s.err && <span style={{ fontSize: 12.5, color: '#B42318', lineHeight: 1.5, marginTop: -10, overflowWrap: 'anywhere' }}>{s.err}</span>}
      {s.phase === 'fetching' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}><ScanBar /><span style={{ fontSize: 12.5, color: '#4A5464' }}>Reading the link…</span></div>
      )}
      {info && ['info', 'downloading', 'done'].includes(s.phase) && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '136px minmax(0,1fr)', gap: 16, alignItems: 'center' }}>
            <div style={{ position: 'relative', aspectRatio: '16/9', borderRadius: 10, background: info.thumbnail ? `center / cover no-repeat url("${info.thumbnail}"), ${grad(hueOf(info.url))}` : grad(hueOf(info.url)), overflow: 'hidden' }}>
              <span className="tnum" style={{ position: 'absolute', right: 6, bottom: 6, padding: '2px 6px', borderRadius: 999, background: 'rgba(255,255,255,.9)', fontSize: 10.5, fontWeight: 600, color: '#2A3240' }}>
                {info.playlist ? `${info.count} videos` : info.duration ? fmt(info.duration) : 'File'}
              </span>
            </div>
            <div style={{ minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 15, fontWeight: 500, color: '#14181F', lineHeight: 1.35, overflowWrap: 'anywhere' }}>{info.title}</span>
              <span className="eyebrow">{meta}</span>
            </div>
          </div>

          {s.phase === 'info' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
              {!info.direct && (
                <>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <span className="eyebrow">Quality</span>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: 8 }}>
                      {QUALITIES.map(q => (
                        <button key={q} className={'opt' + (q === s.q ? ' on' : '')}
                          onClick={() => set({ q, f: q === 'Audio only' ? (['M4A', 'MP3'].includes(s.f) ? s.f : 'M4A') : ['MP4', 'WebM'].includes(s.f) ? s.f : 'MP4' })}>
                          <span style={{ fontSize: 13, fontWeight: 500 }}>{q}</span>
                          <span className="tnum" style={{ fontSize: 11.5, opacity: 0.8 }}>{info.sizes[q] ? '~' + fmtSize(info.sizes[q]) : ' '}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <span className="eyebrow">Format</span>
                    <div className="seg-ctl">
                      {formats.map(f => <button key={f} className={f === s.f ? 'on' : ''} onClick={() => set({ f })}>{f}</button>)}
                    </div>
                  </div>
                </>
              )}
              {info.direct && <span style={{ fontSize: 12.5, color: '#4A5464', lineHeight: 1.5 }}>Direct file link. It downloads as-is, in its original quality and format.</span>}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 14, borderRadius: 12, background: '#F5F7FA' }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, minWidth: 0 }}>
                  <span className="eyebrow" style={{ flex: 'none' }}>Save to</span>
                  <span style={{ fontSize: 12.5, color: '#2A3240', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{saveDir}</span>
                </div>
                <button onClick={() => set({ openAfter: !s.openAfter })} style={{ display: 'flex', alignItems: 'center', gap: 10, border: 'none', background: 'transparent', padding: 0, cursor: 'pointer', textAlign: 'left', fontSize: 13, color: '#14181F' }}>
                  <span style={{ width: 18, height: 18, flex: 'none', borderRadius: 5, border: `1.5px solid ${s.openAfter ? '#2563EB' : '#C3CCD8'}`, background: s.openAfter ? '#2563EB' : '#FFFFFF', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#FFFFFF' }}>
                    {s.openAfter && <IconCheck />}
                  </span>
                  Open, transcribe and index when done
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                <span style={{ flex: '1 1 200px', fontSize: 11.5, color: '#6B7584', lineHeight: 1.5 }}>Only download videos you own or have permission to use.</span>
                <button className="primary" onClick={onStart}><IconDownload w={2} />{info.playlist ? `Download ${info.count}` : 'Download'}</button>
              </div>
            </div>
          )}

          {s.phase === 'downloading' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
                <span className="tnum" style={{ fontSize: 30, fontWeight: 500, letterSpacing: '-0.01em', color: '#14181F' }}>{pctStr}</span>
                <span className="tnum" style={{ fontSize: 12.5, color: '#4A5464' }}>{detail}</span>
                <button className="plain-btn" style={{ marginLeft: 'auto' }} onClick={onCancel}>Cancel</button>
              </div>
              <div style={{ height: 6, borderRadius: 999, background: '#EEF2F7', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: s.total ? ((s.got / s.total) * 100).toFixed(1) + '%' : '30%', borderRadius: 999, background: '#2563EB' }} />
              </div>
              <span style={{ fontSize: 11.5, color: '#6B7584' }}>You can close this window. The download keeps going.</span>
            </div>
          )}

          {s.phase === 'done' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13.5, fontWeight: 500, color: '#1F7A47', minWidth: 0 }}>
                <IconCheck size={16} w={2.2} />Saved to {saveDir}
              </span>
              <button className="primary" style={{ marginLeft: 'auto', height: 40, boxShadow: 'none' }} onClick={() => onOpen(s.last)}>Open in Scrubber</button>
            </div>
          )}
        </div>
      )}
      {s.recent.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, paddingTop: 18, borderTop: '1px solid #EEF2F7' }}>
          <span className="eyebrow">Recent downloads</span>
          {s.recent.map(d => (
            <div key={d.path} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 0' }}>
              <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontSize: 13, fontWeight: 500, color: '#14181F', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{d.name}</span>
                <span className="tnum" style={{ fontSize: 11.5, color: '#6B7584' }}>{fmtSize(d.size)} · {d.host}</span>
              </div>
              <button className="plain-btn" onClick={() => onOpen([d])}>Open</button>
            </div>
          ))}
        </div>
      )}
    </Dialog>
  );
}
