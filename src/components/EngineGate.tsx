import type { EngineState } from '../api';
import { native } from '../native';
import { Dialog } from './Dialog';
import { ScanBar } from './Panes';

/** Shown while the local engine installs on first launch, or if it fails to start. */
export function EngineGate({ engine }: { engine: EngineState }) {
  if (engine.state === 'installing') {
    return (
      <Dialog title={<>Setting up the <i>local engine</i></>} sub="A one-time install of the speech, vision and search runtimes for this machine's hardware. Models download the first time they're used.">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
            <span className="tnum" style={{ fontSize: 30, fontWeight: 500, letterSpacing: '-0.01em' }}>{Math.round(engine.pct * 100)}%</span>
            <span style={{ fontSize: 12.5, color: '#4A5464' }}>{engine.step || 'Preparing'}</span>
          </div>
          <ScanBar />
          <div className="tnum" style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace', fontSize: 11, color: '#6B7584', lineHeight: 1.5, maxHeight: 96, overflow: 'hidden', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
            {engine.log.slice(-5).join('\n')}
          </div>
        </div>
      </Dialog>
    );
  }
  if (engine.state === 'error') {
    return (
      <Dialog title={<>The engine <i>stopped</i></>} sub="Video Scrubber couldn't start its local engine.">
        <span style={{ fontSize: 12.5, color: '#B42318', lineHeight: 1.5, overflowWrap: 'anywhere' }}>{engine.error}</span>
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button className="primary" onClick={() => native.restartEngine()}>Try again</button>
        </div>
      </Dialog>
    );
  }
  return null;
}
