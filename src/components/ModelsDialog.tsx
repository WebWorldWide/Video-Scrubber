import { useCallback, useEffect, useState } from 'react';
import { api, type Hardware, type ModelKind, type ModelRow, type Models, type Settings } from '../api';
import { fmtSize } from '../format';
import { native } from '../native';
import { Dialog } from './Dialog';

const SECTIONS: { kind: ModelKind; title: string; sub: string }[] = [
  { kind: 'speech', title: 'Speech to text', sub: 'Transcribes every word, with timestamps.' },
  { kind: 'vision', title: 'Picture search', sub: 'Matches your description against each key frame directly.' },
  { kind: 'caption', title: 'Frame descriptions', sub: 'Writes a sentence for every key frame, shown in Transcribe + Show and searchable.' },
  { kind: 'text', title: 'Meaning search', sub: 'Finds lines by meaning, not just matching words.' },
];

const ENGINE_LABEL: Record<string, string> = { auto: 'Auto', mlx: 'MLX', 'faster-whisper': 'faster-whisper', 'whisper.cpp': 'whisper.cpp', transformers: 'Transformers' };
const LANGS: [string, string][] = [['auto', 'Detect'], ['en', 'English'], ['es', 'Spanish'], ['fr', 'French'], ['de', 'German'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['ja', 'Japanese'], ['zh', 'Chinese'], ['ko', 'Korean'], ['ru', 'Russian'], ['hi', 'Hindi'], ['ar', 'Arabic']];

export type Pulls = Record<string, { pct: number; state: string; error?: string }>;

type Props = { hw: Hardware | null; settings: Settings; pulls: Pulls; videoCount: number; onSettings: (s: Settings) => void; onReindexAll: () => void; onClose: () => void };

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

  const engines = ['auto', ...new Set((models?.models.speech ?? []).flatMap(m => m.engines ?? []))];
  const hwLine = hw ? [hw.label, `speech on ${hw.speechAccel}`, `video encode ${hw.videoEncoder.replace('h264_', '')}`].join(' · ') : 'Detecting hardware…';

  return (
    <Dialog width={780} title={<>Local <i>models</i></>} sub={<>Everything runs on this machine. <span className="tnum" style={{ color: '#14181F' }}>{hwLine}</span></>} onClose={onClose}>
      {changed && videoCount > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 14px', borderRadius: 12, background: '#EAF1FF', border: '1px solid #C7D8FB', marginTop: -6 }}>
          <span style={{ flex: 1, fontSize: 12.5, color: '#1D4FD8', lineHeight: 1.45 }}>New videos use these models. Re-index the {videoCount} open {videoCount === 1 ? 'video' : 'videos'} to apply them there too.</span>
          <button className="plain-btn" onClick={() => { onReindexAll(); setChanged(false); }}>Re-index all</button>
        </div>
      )}
      {!models && <span style={{ fontSize: 12.5, color: '#6B7584' }}>Loading models…</span>}
      {models && SECTIONS.map(sec => (
        <div key={sec.kind} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
            <span className="eyebrow">{sec.title}</span>
            <span style={{ fontSize: 12, color: '#6B7584' }}>{sec.sub}</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(168px, 1fr))', gap: 8 }}>
            {models.models[sec.kind].map(m => {
              const on = settings[sec.kind] === m.id;
              const pull = pulls[`${sec.kind}:${m.id}`];
              const status = !m.available
                ? m.via === 'ollama' ? 'Needs Ollama running' : 'No engine installed'
                : pull?.state === 'downloading' ? `Downloading ${Math.round((pull.pct || 0) * 100)}%`
                : pull?.state === 'error' ? 'Download failed'
                : m.installed ? (m.size ? 'On this machine' : '') : `Download ${fmtSize(m.size)}`;
              return (
                <button key={m.id} className={'opt' + (on ? ' on' : '')} disabled={!m.available} onClick={() => pick(sec.kind, m)} title={pull?.error || m.note}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>{m.label}</span>
                  <span style={{ fontSize: 11.5, opacity: 0.8, lineHeight: 1.35 }}>{m.note}</span>
                  <span className="tnum" style={{ fontSize: 10.5, fontWeight: 600, fontStretch: '80%', textTransform: 'uppercase', letterSpacing: '0.06em', marginTop: 4, color: on ? '#1D4FD8' : m.installed ? '#1F7A47' : '#6B7584' }}>
                    {[status, m.engine && sec.kind === 'speech' ? ENGINE_LABEL[m.engine] : null, m.langs].filter(Boolean).join(' · ')}
                  </span>
                </button>
              );
            })}
          </div>
          {sec.kind === 'caption' && !models.ollama && (
            <span style={{ fontSize: 11.5, color: '#6B7584' }}>Install and start <a href="https://ollama.com/download" target="_blank" rel="noreferrer">Ollama</a> to use Qwen-VL, Gemma, MiniCPM-V, LLaVA and other vision models with GPU acceleration.</span>
          )}
        </div>
      ))}
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
    </Dialog>
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
