// Per-OS title bars from the design. macOS keeps native traffic lights (inset over this bar);
// Windows and Linux draw the design's caption buttons. `data-tauri-drag-region` moves the window
// and double-click maximizes; it applies to the marked element itself, so labels ignore the pointer.
import { native } from '../native';

export type OS = 'macOS' | 'Windows' | 'Linux';

export function detectOS(): OS {
  const u = navigator.userAgent;
  return /Mac/.test(u) ? 'macOS' : /Win/.test(u) ? 'Windows' : 'Linux';
}

const Wordmark = ({ size }: { size: number }) => (
  <span className="serif" style={{ fontSize: size, lineHeight: 1, letterSpacing: 0, pointerEvents: 'none' }}>Video <i>Scrubber</i></span>
);

const bar = { flex: 'none', display: 'flex', alignItems: 'center', background: '#FFFFFF', borderBottom: '1px solid #E4E9F0', userSelect: 'none', position: 'relative' } as const;
const centered = { position: 'absolute', left: '50%', transform: 'translateX(-50%)', pointerEvents: 'none' } as const;

export function TitleBar({ os }: { os: OS }) {
  if (os === 'macOS') {
    return (
      <div data-tauri-drag-region style={{ ...bar, height: 38, padding: '0 14px' }}>
        <div style={centered}><Wordmark size={19} /></div>
      </div>
    );
  }
  if (os === 'Windows') {
    const svg = { stroke: '#14181F', strokeWidth: 1, fill: 'none' };
    return (
      <div data-tauri-drag-region style={{ ...bar, height: 34 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '0 14px', flex: 1, pointerEvents: 'none' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" style={{ fill: 'none', stroke: '#2563EB', strokeWidth: 2, strokeLinecap: 'round' }}><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M12 3v18" /></svg>
          <Wordmark size={17} />
        </div>
        <div style={{ display: 'flex', height: '100%' }}>
          <button className="win-ctl" title="Minimize" onClick={() => native.win('min')}><svg width="10" height="10" viewBox="0 0 10 10" style={svg}><path d="M0 5h10" /></svg></button>
          <button className="win-ctl" title="Maximize" onClick={() => native.win('max')}><svg width="10" height="10" viewBox="0 0 10 10" style={svg}><rect x="0.5" y="0.5" width="9" height="9" /></svg></button>
          <button className="win-ctl close" title="Close" onClick={() => native.win('close')}><svg width="10" height="10" viewBox="0 0 10 10" style={svg}><path d="M0 0l10 10M10 0L0 10" /></svg></button>
        </div>
      </div>
    );
  }
  const ctl = (action: 'min' | 'max' | 'close', path: JSX.Element, title: string) => (
    <button className="lx-ctl" title={title} onClick={() => native.win(action)}>
      <svg width="8" height="8" viewBox="0 0 10 10" style={{ fill: 'none', stroke: '#14181F', strokeWidth: 1.6 }}>{path}</svg>
    </button>
  );
  return (
    <div data-tauri-drag-region style={{ ...bar, height: 44, justifyContent: 'flex-end', padding: '0 10px' }}>
      <div style={centered}><Wordmark size={19} /></div>
      <div style={{ display: 'flex', gap: 10 }}>
        {ctl('min', <path d="M0 8h10" />, 'Minimize')}
        {ctl('max', <rect x="1" y="1" width="8" height="8" />, 'Maximize')}
        {ctl('close', <path d="M1 1l8 8M9 1L1 9" />, 'Close')}
      </div>
    </div>
  );
}
