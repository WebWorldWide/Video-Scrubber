// SPDX-License-Identifier: GPL-3.0-or-later
import type { Video } from '../api';
import { grad, hueOf } from '../format';
import { IconPlus, IconX } from '../icons';

/** Every open video, its indexing state, and a way to add more. */
export function LibraryStrip({ videos, activeId, firstFrame, onPick, onRemove, onAdd }: {
  videos: Video[];
  activeId: string | null;
  firstFrame: (v: Video) => string | null;
  onPick: (id: string) => void;
  onRemove: (v: Video) => void;
  onAdd: () => void;
}) {
  return (
    <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 8, padding: '8px 24px', borderBottom: '1px solid #E4E9F0', background: '#FAFBFD', overflowX: 'auto' }}>
      <span className="eyebrow tnum" style={{ flex: 'none', marginRight: 4 }}>Library · {videos.length}</span>
      {videos.map(v => {
        const f = firstFrame(v);
        const busy = v.status === 'indexing' || v.status === 'queued';
        const overall = Math.round(((Math.min(3, v.step) + (v.status === 'ready' ? 0 : v.pct)) / 3) * 100);
        return (
          <div key={v.id} className={'vid-tab' + (v.id === activeId ? ' on' : '')} onClick={() => onPick(v.id)} title={v.path}>
            <div style={{ width: 40, height: 24, borderRadius: 0, flex: 'none', background: f ? `center / cover no-repeat url("${f}")` : grad(hueOf(v.name)), position: 'relative', overflow: 'hidden' }}>
              {busy && <div style={{ position: 'absolute', left: 0, bottom: 0, height: 2, width: overall + '%', background: '#2563EB' }} />}
            </div>
            <span style={{ fontSize: 12.5, fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', minWidth: 0 }}>{v.name}</span>
            {busy && (
              <span className="tnum" style={{ fontSize: 11, color: '#2563EB', flex: 'none', animation: v.status === 'indexing' ? 'vsPulse 1.4s ease-in-out infinite' : 'none' }}>
                {v.status === 'queued' ? 'Queued' : overall + '%'}
              </span>
            )}
            {v.status === 'error' && <span title={v.error ?? ''} style={{ width: 7, height: 7, borderRadius: '50%', background: '#B42318', flex: 'none' }} />}
            {!v.exists && <span style={{ fontSize: 11, color: '#B42318', flex: 'none' }}>Missing</span>}
            <button className="x" title="Remove from library" onClick={e => { e.stopPropagation(); onRemove(v); }}><IconX size={12} /></button>
          </div>
        );
      })}
      <button className="btn" style={{ height: 34, flex: 'none' }} onClick={onAdd}><IconPlus />Add</button>
    </div>
  );
}
