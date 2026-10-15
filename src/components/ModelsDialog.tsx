// SPDX-License-Identifier: GPL-3.0-or-later
import { useCallback, useEffect, useState } from 'react';
import { api, type Hardware, type ModelKind, type ModelRow, type Models, type Settings } from '../api';
import { fmtSize } from '../format';
import { native } from '../native';
import { Dialog } from './Dialog';

const SECTIONS: { kind: ModelKind; title: string; sub: string }[] = [
  { kind: 'speech', title: 'Speech to text', sub: 'Transcribes every word, with timestamps.' },
  { kind: 'caption', title: 'Frame descriptions', sub: 'A vision-language model writes a sentence for every key frame. Searchable, and shown in Transcribe + Show.' },
  { kind: 'vision', title: 'Picture search', sub: 'Matches your description against each key frame directly.' },
  { kind: 'text', title: 'Meaning search', sub: 'Finds lines by meaning, not just matching words.' },
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
      {[0, 1, 2, 3, 4].map(i => <span key={i} style={{ width: 5, height: 3, borderRadius: 2, background: i < filled ? (on ? '#2563EB' : '#4A5464') : '#DCE3EC' }} />)}
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
    <Dialog width={820} title={<>Local <i>models</i></>} sub="Downloaded from Hugging Face and run on this machine. Recommendations pick the best quality that fits in memory together." onClose={onClose}>
      {mem && models && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 14, borderRadius: 12, background: '#F5F7FA', marginTop: -6 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 13, fontWeight: 500 }}>{where}</span>
            <span className="eyebrow tnum">{[pool, hw ? `speech on ${hw.speechAccel}` : null, `${gb(models.diskFree)} free disk`].filter(Boolean).join(' · ')}</span>
          </div>
          <div style={{ height: 6, borderRadius: 999, background: '#E4E9F0', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: Math.min(100, (mem.used / mem.budget) * 100).toFixed(1) + '%', borderRadius: 999, background: over ? '#B42318' : '#2563EB' }} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span className="tnum" style={{ fontSize: 12, color: over ? '#B42318' : '#4A5464', flex: 1, minWidth: 200 }}>
              Selected models use about {gb(mem.used)} of the {gb(mem.budget)} set aside for them{over ? '. That may not fit; indexing could slow down or fail.' : '.'}
            </span>
            {isRecommended ? (
              <span className="eyebrow" style={{ color: '#1F7A47' }}>Using the recommended set</span>
            ) : (
              <button className="plain-btn" onClick={useRecommended}>Use recommended</button>
            )}
          </div>
        </div>
      )}
      {changed && videoCount > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 14px', borderRadius: 12, background: '#EAF1FF', border: '1px solid #C7D8FB' }}>
          <span style={{ flex: 1, fontSize: 12.5, color: '#1D4FD8', lineHeight: 1.45 }}>New videos use these models. Re-index the {videoCount} open {videoCount === 1 ? 'video' : 'videos'} to apply them there too.</span>
          <button className="plain-btn" onClick={() => { onReindexAll(); setChanged(false); }}>Re-index all</button>
        </div>
      )}
      {!models && <span style={{ fontSize: 12.5, color: '#6B7584' }}>Checking this machine and Hugging Face…</span>}
      {models && SECTIONS.map(sec => {
        const rows = models.models[sec.kind].filter(m => m.via !== 'ollama' || models.ollama);
        return (
          <div key={sec.kind} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
              <span className="eyebrow">{sec.title}</span>
              <span style={{ fontSize: 12, color: '#6B7584' }}>{sec.sub}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 8 }}>
              {rows.map(m => <ModelCard key={m.id} m={m} kind={sec.kind} on={settings[sec.kind] === m.id} pull={pulls[`${sec.kind}:${m.id}`]} onPick={() => pick(sec.kind, m)} />)}
            </div>
            {sec.kind === 'caption' && !models.ollama && (
              <span style={{ fontSize: 11.5, color: '#6B7584' }}>Running <a href="https://ollama.com/download" target="_blank" rel="noreferrer">Ollama</a> adds Gemma 3, MiniCPM-V, Llama 3.2 Vision and LLaVA here.</span>
            )}
          </div>
        );
      })}
      {models && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: 14, borderRadius: 12, background: '#F5F7FA' }}>
          <Row label="Speech engine">
            <div className="seg-ctl" style={{ flexWrap: 'wrap' }}>
              {engines.map(e => <button key={e} className={settings.speechEngine === e ? 'on' : ''} onClick={() => save({ speechEngine: e })}>{ENGINE_LABEL[e] ?? e}</button>)}
            </div>
          </Row>
          <Row label="Spoken language">
            <select value={settings.language} onChange={e => save({ language: e.target.value })} style={{ height: 32, borderRadius: 8, border: '1px solid #D9E0EA', background: '#FFFFFF', padding: '0 10px', font: 'inherit', fontSize: 12.5 }}>
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
        Video Scrubber is free software under the <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noreferrer">GNU GPL v3</a>, with no warranty.{' '}
        <a href="https://github.com/WebWorldWide/Video-Scrubber" target="_blank" rel="noreferrer">Source code</a>. Each model has its own license, shown on its card.
      </span>
    </Dialog>
  );
}

function ModelCard({ m, kind, on, pull, onPick }: { m: ModelRow; kind: ModelKind; on: boolean; pull?: { pct: number; state: string; error?: string }; onPick: () => void }) {
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
        {m.recommended && <span style={{ marginLeft: 'auto', fontSize: 9.5, fontWeight: 700, fontStretch: '80%', textTransform: 'uppercase', letterSpacing: '0.06em', padding: '2px 6px', borderRadius: 999, background: '#1F9D57', color: '#FFFFFF', flex: 'none' }}>Best fit</span>}
      </span>
      <span style={{ fontSize: 11.5, opacity: 0.8, lineHeight: 1.35 }}>{m.note}</span>
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
