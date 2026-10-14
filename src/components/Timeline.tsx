import { memo, type PointerEvent, type RefObject } from 'react';
import { frameUrl, type Scene, type Segment } from '../api';
import { fmt, grad, pad } from '../format';

const SPEECH = '#2F7BC4';

const pct = (t: number, dur: number) => (Math.max(0, Math.min(1, t / dur)) * 100).toFixed(3) + '%';

const Ruler = memo(function Ruler({ dur }: { dur: number }) {
  const step = dur > 7200 ? 1800 : dur > 3600 ? 900 : dur > 1500 ? 300 : dur > 500 ? 120 : dur > 150 ? 30 : dur > 40 ? 10 : 5;
  const marks: number[] = [];
  for (let t = 0; t < dur - step * 0.4; t += step) marks.push(t);
  return (
    <div style={{ height: 16, position: 'relative' }}>
      {marks.map(t => (
        <div key={t} className="tnum" style={{ position: 'absolute', left: pct(t, dur), top: 0, bottom: 0, borderLeft: '1px solid #D9E0EA', paddingLeft: 4, fontSize: 10, color: '#6B7584', lineHeight: '12px' }}>{fmt(t)}</div>
      ))}
    </div>
  );
});

const Pictures = memo(function Pictures({ scenes, dur, videoId }: { scenes: Scene[]; dur: number; videoId: string }) {
  return (
    <div style={{ height: 42, position: 'relative', borderRadius: 4, overflow: 'hidden', background: '#EAEEF4' }}>
      {scenes.map((s, i) => (
        <div key={i} style={{ position: 'absolute', top: 0, bottom: 0, left: pct(s.t, dur), width: pct(s.e - s.t, dur), background: `center / cover no-repeat url("${frameUrl(videoId, s.f)}"), ${grad(210)}`, borderRight: '1px solid #EAEEF4', overflow: 'hidden' }}>
          <span className="tnum" style={{ position: 'absolute', left: 4, bottom: 3, fontSize: 9.5, fontWeight: 600, fontStretch: '80%', color: '#2A3240', textShadow: '0 0 3px rgba(255,255,255,.9)' }}>{pad(i + 1)}</span>
        </div>
      ))}
    </div>
  );
});

const Speech = memo(function Speech({ segs, dur }: { segs: Segment[]; dur: number }) {
  return (
    <div style={{ height: 14, position: 'relative' }}>
      {segs.map((g, i) => (
        <div key={i} style={{ position: 'absolute', top: 3, height: 8, borderRadius: 2, left: pct(g.t, dur), width: pct(Math.max(0.2, g.e - g.t), dur), background: SPEECH, opacity: 0.8 }} />
      ))}
    </div>
  );
});

type Props = {
  videoId: string | null;
  dur: number;
  t: number;
  scenes: Scene[];
  segs: Segment[];
  results: { t: number; n: number }[];
  inT: number | null;
  outT: number | null;
  showPictures: boolean;
  trackRef: RefObject<HTMLDivElement>;
  onDown: (e: PointerEvent<HTMLDivElement>) => void;
};

export function Timeline({ videoId, dur, t, scenes, segs, results, inT, outT, showPictures, trackRef, onDown }: Props) {
  const hasRange = inT != null && outT != null;
  return (
    <div style={{ flex: 'none', display: 'flex', gap: 10, padding: '14px 18px 16px 14px', borderRadius: 14, border: '1px solid #E4E9F0', background: '#FFFFFF', boxShadow: '0 1px 2px rgba(20,24,31,.04),0 8px 24px rgba(20,24,31,.04)', userSelect: 'none' }}>
      <div style={{ width: 52, flex: 'none', display: 'flex', flexDirection: 'column', gap: 6, fontSize: 10, fontWeight: 600, fontStretch: '80%', textTransform: 'uppercase', letterSpacing: '0.07em', color: '#6B7584' }}>
        <div style={{ height: 16 }} />
        {showPictures && <div style={{ height: 42, display: 'flex', alignItems: 'center' }}>Picture</div>}
        <div style={{ height: 14, display: 'flex', alignItems: 'center' }}>Speech</div>
      </div>
      <div ref={trackRef} onPointerDown={onDown} style={{ flex: 1, minWidth: 0, position: 'relative', display: 'flex', flexDirection: 'column', gap: 6, cursor: 'pointer', touchAction: 'none' }}>
        <Ruler dur={dur} />
        {showPictures && (videoId ? <Pictures scenes={scenes} dur={dur} videoId={videoId} /> : <div style={{ height: 42, borderRadius: 4, background: '#EAEEF4' }} />)}
        <Speech segs={segs} dur={dur} />
        {hasRange && (
          <div style={{ position: 'absolute', top: 16, bottom: -4, left: pct(inT!, dur), width: pct(outT! - inT!, dur), background: 'rgba(37,99,235,0.10)', borderLeft: '2px solid #2563EB', borderRight: '2px solid #2563EB', pointerEvents: 'none' }} />
        )}
        {results.map((r, i) => (
          <div key={i} style={{ position: 'absolute', top: -1, left: pct(r.t, dur), transform: 'translateX(-50%)', width: 15, height: 15, borderRadius: '50%', background: '#14181F', color: '#FFFFFF', fontSize: 9.5, fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none', boxShadow: '0 0 0 2px #FFFFFF' }}>{r.n}</div>
        ))}
        <div style={{ position: 'absolute', top: 0, bottom: -4, left: pct(t, dur), width: 2, marginLeft: -1, background: '#2563EB', pointerEvents: 'none', boxShadow: '0 0 0 1px rgba(255,255,255,.9)' }} />
      </div>
    </div>
  );
}
