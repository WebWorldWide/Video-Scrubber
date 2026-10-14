import type { ReactNode } from 'react';
import { IconX } from '../icons';

/** Modal shell from the design's download dialog: blurred scrim, 20px-radius card, serif title. */
export function Dialog({ title, sub, width = 580, onClose, children }: { title: ReactNode; sub?: ReactNode; width?: number; onClose?: () => void; children: ReactNode }) {
  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, zIndex: 20, background: 'rgba(20,24,31,0.22)', backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', padding: '9vh 16px 16px', overflow: 'auto' }}>
      <div onClick={e => e.stopPropagation()} style={{ width: `min(${width}px, 100%)`, background: '#FFFFFF', borderRadius: 20, boxShadow: '0 1px 2px rgba(20,24,31,.06),0 28px 72px rgba(20,24,31,.18)', padding: 32, display: 'flex', flexDirection: 'column', gap: 22 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span className="serif" style={{ fontSize: 38, lineHeight: 0.95, letterSpacing: '-0.01em', color: '#14181F' }}>{title}</span>
            {sub && <span style={{ fontSize: 13, lineHeight: 1.5, color: '#4A5464', textWrap: 'pretty' } as React.CSSProperties}>{sub}</span>}
          </div>
          {onClose && <button className="close-btn" onClick={onClose} title="Close (Esc)"><IconX size={14} w={2} /></button>}
        </div>
        {children}
      </div>
    </div>
  );
}
