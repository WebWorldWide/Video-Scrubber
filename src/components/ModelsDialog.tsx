// SPDX-License-Identifier: GPL-3.0-or-later
import { useCallback, useEffect, useState } from 'react';
import { api, type Hardware, type ModelKind, type ModelRow, type Models, type Settings } from '../api';
import { fmtSize } from '../format';
import { native } from '../native';
import { Dialog } from './Dialog';

const SECTIONS: { kind: ModelKind; title: string; hint: string }[] = [
  { kind: 'speech', title: 'Speech to text', hint: 'openai/whisper-small' },
  { kind: 'caption', title: 'Frame descriptions', hint: 'Qwen/Qwen3-VL-2B-Instruct or ollama:gemma3:4b' },
  { kind: 'vision', title: 'Picture search', hint: 'google/siglip-base-patch16-224' },
  { kind: 'text', title: 'Text search', hint: 'BAAI/bge-small-en-v1.5' },
];

const ENGINE_LABEL: Record<string, string> = { auto: 'Auto', mlx: 'MLX', 'faster-whisper': 'faster-whisper', 'whisper.cpp': 'whisper.cpp', transformers: 'PyTorch', ollama: 'Ollama' };
const LANGS: [string, string][] = [['auto', 'Detect'], ['en', 'English'], ['es', 'Spanish'], ['fr', 'French'], ['de', 'German'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['ja', 'Japanese'], ['zh', 'Chinese'], ['ko', 'Korean'], ['ru', 'Russian'], ['hi', 'Hindi'], ['ar', 'Arabic']];
const GB = 1024 ** 3;
const gb = (b: number) => (b / GB < 10 ? (b / GB).toFixed(1) : Math.round(b / GB)) + ' GB';

export type Pulls = Record<string, { pct: number; state: string; error?: string }>;

type Props = { hw: Hardware | null; settings: Settings; pulls: Pulls; videoCount: number; onSettings: (s: Settings) => void; onReindexAll: () => void; onClose: () => void };

/** Five-step meter for quality and speed. */
function Meter({ value, max, on }: { value: number; max: number; on: boolean }) {
  const filled = Math.max(1, Math.round((value / max) * 5));
  return (
    <span style={{ display: 'inline-flex', gap: 2 }}>
      {[0, 1, 2, 3, 4].map(i => <span key={i} style={{ width: 5, height: 3, borderRadius: 0, background: i < filled ? (on ? '#2563EB' : '#4A5464') : '#DCE3EC' }} />)}
    </span>
  );
}

export function ModelsDialog({ hw, settings, pulls, videoCount, onSettings, onReindexAll, onClose }: Props) {
  const [models, setModels] = useState<Models | null>(null);
  const [changed, setChanged] = useState(false);
  const load = useCallback(() => api.models().then(setModels).catch(() => {}), []);
  useEffect(() => {
    load();
  }, [load]);
  // Refresh install badges as downloads finish.
  const finished = Object.values(pulls).filter(p => p.state === 'installed').length;
  useEffect(() => {
    if (finished) load();
  }, [finished, load]);

  const save = async (patch: Partial<Settings>) => {
    const s = await api.saveSettings(patch);
    onSettings(s);
    setChanged(true);
    load();
  };
  const pick = async (kind: ModelKind, m: ModelRow) => {
    if (!m.available) return;
    await save({ [kind]: m.id } as Partial<Settings>);
    if (!m.installed) api.pull(kind, m.id).catch(() => {});
  };
  const add = async (kind: ModelKind, ref: string) => {
    const { id } = await api.addModel(kind, ref);
    await save({ [kind]: id } as Partial<Settings>);
    api.pull(kind, id).catch(() => {});
  };
  const remove = async (kind: ModelKind, id: string) => {
    await api.removeModel(kind, id);
    onSettings(await api.settings());
    load();
  };
  const useRecommended = async () => {
    const s = await api.useRecommended();
    onSettings(s);
    setChanged(true);
    load();
  };

  const mem = models?.memory;
  const isRecommended = !!models && (Object.keys(models.recommended) as ModelKind[]).every(k => models.selected[k] === models.recommended[k]);
  const engines = ['auto', ...new Set((models?.models.speech ?? []).flatMap(m => m.engines ?? []))];
  const where = hw?.gpu ? hw.gpu : hw?.cpu ?? 'this machine';
  const pool = mem ? (mem.unified ? `${gb(mem.ram)} unified memory` : mem.gpuMem ? `${gb(mem.gpuMem)} GPU memory` : `${gb(mem.ram)} RAM`) : '';
  const over = !!mem && mem.used > mem.budget;

  return (
    <Dialog width={820} title={<>Local <i>models</i></>} onClose={onClose}>
      {mem && models && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 14, borderRadius: 0, background: '#F5F7FA', marginTop: -6 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 13, fontWeight: 500 }}>{where}</span>
            <span className="eyebrow tnum">{[pool, hw ? `speech on ${hw.speechAccel}` : null, `${gb(models.diskFree)} free disk`].filter(Boolean).join(' · ')}</span>
          </div>
          <div style={{ height: 6, borderRadius: 0, background: '#E4E9F0', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: Math.min(100, (mem.used / mem.budget) * 100).toFixed(1) + '%', borderRadius: 0, background: over ? '#B42318' : '#2563EB' }} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span className="tnum" style={{ fontSize: 12, color: over ? '#B42318' : '#4A5464', flex: 1, minWidth: 200 }}>
              {gb(mem.used)} of {gb(mem.budget)}{over ? ' · Over budget' : ''}
            </span>
            {isRecommended ? (
              <span className="eyebrow" style={{ color: '#1F7A47' }}>Recommended</span>
            ) : (
              <button className="plain-btn" onClick={useRecommended}>Use recommended</button>
            )}
          </div>
        </div>
      )}
      {changed && videoCount > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 14px', borderRadius: 0, background: '#EAF1FF', border: '1px solid #C7D8FB' }}>
          <span style={{ flex: 1, fontSize: 12.5, color: '#1D4FD8', lineHeight: 1.45 }}>Re-index {videoCount} {videoCount === 1 ? 'file' : 'files'} with these models?</span>
          <button className="plain-btn" onClick={() => { onReindexAll(); setChanged(false); }}>Re-index all</button>
        </div>
      )}
      {!models && <span style={{ fontSize: 12.5, color: '#6B7584' }}>Loading…</span>}
      {models && SECTIONS.map(sec => {
        const rows = models.models[sec.kind].filter(m => m.via !== 'ollama' || models.ollama);
        return (
          <div key={sec.kind} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span className="eyebrow">{sec.title}</span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 8 }}>
              {rows.map(m => <ModelCard key={m.id} m={m} kind={sec.kind} on={settings[sec.kind] === m.id} pull={pulls[`${sec.kind}:${m.id}`]} onPick={() => pick(sec.kind, m)} onRemove={() => remove(sec.kind, m.id)} />)}
            </div>
            <AddModel hint={sec.hint} onAdd={ref => add(sec.kind, ref)} />
          </div>
        );
      })}
      {models && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: 14, borderRadius: 0, background: '#F5F7FA' }}>
          <Row label="Speech engine">
            <div className="seg-ctl" style={{ flexWrap: 'wrap' }}>
              {engines.map(e => <button key={e} className={settings.speechEngine === e ? 'on' : ''} onClick={() => save({ speechEngine: e })}>{ENGINE_LABEL[e] ?? e}</button>)}
            </div>
          </Row>
          <Row label="Spoken language">
            <select value={settings.language} onChange={e => save({ language: e.target.value })} style={{ height: 32, borderRadius: 0, border: '1px solid #D9E0EA', background: '#FFFFFF', padding: '0 10px', font: 'inherit', fontSize: 12.5 }}>
              {LANGS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </Row>
          <Row label="Index at once">
            <div className="seg-ctl">
              {[1, 2, 3, 4].map(n => <button key={n} className={settings.parallel === n ? 'on' : ''} onClick={() => save({ parallel: n })}>{n} {n === 1 ? 'video' : 'videos'}</button>)}
            </div>
          </Row>
          <Row label="Save to">
            <span style={{ fontSize: 12.5, color: '#2A3240', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', minWidth: 0, flex: 1 }}>{settings.saveDir}</span>
            {native.available && <button className="plain-btn" onClick={async () => { const d = await native.pickFolder(settings.saveDir); if (d) save({ saveDir: d }); }}>Change</button>}
          </Row>
        </div>
      )}
      <span style={{ fontSize: 11.5, color: '#6B7584', lineHeight: 1.5 }}>
        <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noreferrer">GNU GPL v3</a>, no warranty ·{' '}
        <a href="https://github.com/WebWorldWide/Video-Scrubber" target="_blank" rel="noreferrer">Source</a>
      </span>
    </Dialog>
  );
}

/** Any Hugging Face repo id or link, or `ollama:tag` for frame descriptions. */
function AddModel({ hint, onAdd }: { hint: string; onAdd: (ref: string) => Promise<void> }) {
  const [ref, setRef] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ref.trim() || busy) return;
    setBusy(true);
    setErr('');
    try {
      await onAdd(ref.trim());
      setRef('');
    } catch (x) {
      setErr((x as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', gap: 8 }}>
        <input className="url-input" value={ref} onChange={e => setRef(e.target.value)} placeholder={hint} spellCheck={false} style={{ height: 32, fontSize: 12.5 }} />
        <button className="plain-btn" type="submit" disabled={busy || !ref.trim()}>{busy ? 'Adding…' : 'Add'}</button>
      </div>
      {err && <span style={{ fontSize: 11.5, color: '#B42318' }}>{err}</span>}
    </form>
  );
}

function ModelCard({ m, kind, on, pull, onPick, onRemove }: { m: ModelRow; kind: ModelKind; on: boolean; pull?: { pct: number; state: string; error?: string }; onPick: () => void; onRemove: () => void }) {
  const hf = m.via === 'huggingface';
  const status = !m.available
    ? m.via === 'ollama' ? 'Needs Ollama running' : 'No runtime installed'
    : pull?.state === 'downloading' ? `Downloading ${Math.round((pull.pct || 0) * 100)}%`
    : pull?.state === 'error' ? 'Download failed'
    : m.installed ? (hf ? 'Downloaded' : '') : hf && m.download ? `Download ${fmtSize(m.download)}` : '';
  const facts = [
    hf && m.memory ? `${gb(m.memory)} memory` : null,
    m.engine && m.engine !== 'none' ? ENGINE_LABEL[m.engine] + (m.onGpu === false ? ' (CPU)' : '') : null,
    m.englishOnly ? 'English' : null,
  ].filter(Boolean).join(' · ');
  return (
    <button className={'opt' + (on ? ' on' : '')} disabled={!m.available} onClick={onPick} title={[pull?.error, m.repo ? `huggingface.co/${m.repo}` : null].filter(Boolean).join('\n')} style={{ gap: 4, position: 'relative' }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%' }}>
        <span style={{ fontSize: 13, fontWeight: 500 }}>{m.label}</span>
        {m.custom && <span role="button" title="Remove" onClick={e => { e.stopPropagation(); onRemove(); }} style={{ marginLeft: 'auto', flex: 'none', padding: '0 2px', color: '#6B7584', cursor: 'pointer' }}>×</span>}
        {m.recommended && <span style={{ marginLeft: 'auto', fontSize: 9.5, fontWeight: 700, fontStretch: '80%', textTransform: 'uppercase', letterSpacing: '0.06em', padding: '2px 6px', borderRadius: 0, background: '#1F9D57', color: '#FFFFFF', flex: 'none' }}>Best fit</span>}
      </span>
      {m.custom && m.note && <span style={{ fontSize: 11.5, opacity: 0.8, lineHeight: 1.35 }}>{m.note}</span>}
      {m.quality != null && kind && m.id !== 'none' && (
        <span style={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap', fontSize: 10.5, color: '#6B7584', marginTop: 2 }}>
          Quality <Meter value={m.quality} max={95} on={on} /> Speed <Meter value={m.speed ?? 1} max={5} on={on} />
        </span>
      )}
      {facts && <span className="tnum" style={{ fontSize: 11, color: !m.fits && hf ? '#B42318' : '#4A5464' }}>{facts}{!m.fits && hf ? ' · may not fit' : ''}</span>}
      <span className="tnum" style={{ fontSize: 10.5, fontWeight: 600, fontStretch: '80%', textTransform: 'uppercase', letterSpacing: '0.06em', color: on ? '#1D4FD8' : m.installed ? '#1F7A47' : '#6B7584' }}>
        {[status, m.license].filter(Boolean).join(' · ')}
      </span>
    </button>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0, flexWrap: 'wrap' }}>
      <span className="eyebrow" style={{ width: 110, flex: 'none' }}>{label}</span>
      {children}
    </div>
  );
}
