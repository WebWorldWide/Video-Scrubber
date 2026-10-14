// Desktop shell bridge (Tauri). Outside Tauri — plain `vite` in a browser for UI work — it falls
// back to an engine URL passed as ?engine=http://127.0.0.1:PORT&token=…
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { open } from '@tauri-apps/plugin-dialog';
import { revealItemInDir } from '@tauri-apps/plugin-opener';
import type { EngineState } from './api';

export const inTauri = '__TAURI_INTERNALS__' in window;

const MEDIA_EXT = ['mp4', 'mov', 'm4v', 'mkv', 'webm', 'avi', 'wmv', 'flv', 'mts', 'm2ts', 'ts', 'mpg', 'mpeg', 'ogv', '3gp', 'mp3', 'm4a', 'wav', 'flac', 'ogg', 'opus', 'aac'];

type Unlisten = () => void;

/** Subscribe to a Tauri event; returns a synchronous unsubscribe for React effects. */
function on<T>(name: string, cb: (p: T) => void): Unlisten {
  let un: Unlisten | null = null, dead = false;
  listen<T>(name, e => cb(e.payload)).then(u => (dead ? u() : (un = u)));
  return () => {
    dead = true;
    un?.();
  };
}

function browserEngine(): EngineState {
  const p = new URLSearchParams(location.search);
  const url = p.get('engine');
  return { state: url ? 'ready' : 'error', url, token: p.get('token') || '', step: '', pct: 1, log: [], error: url ? null : 'Open Video Scrubber from the desktop app.' };
}

export const native = {
  available: inTauri,

  engine: (): Promise<EngineState> => (inTauri ? invoke('engine_get') : Promise.resolve(browserEngine())),
  restartEngine: () => (inTauri ? invoke<void>('engine_restart') : Promise.resolve()),
  onEngine: (cb: (s: EngineState) => void): Unlisten => (inTauri ? on('engine', cb) : () => {}),

  async openFiles(): Promise<string[]> {
    if (!inTauri) return [];
    const r = await open({ multiple: true, title: 'Open videos', filters: [{ name: 'Video and audio', extensions: MEDIA_EXT }] });
    return r ? (Array.isArray(r) ? r : [r]) : [];
  },
  async pickFolder(current: string): Promise<string | null> {
    if (!inTauri) return null;
    const r = await open({ directory: true, defaultPath: current, title: 'Save downloads and clips to' });
    return typeof r === 'string' ? r : null;
  },
  reveal: (p: string) => (inTauri ? revealItemInDir(p) : Promise.resolve()),

  /** Files opened from the OS: CLI args, "Open with", a second launch, or the Dock. */
  onOpenPaths(cb: (p: string[]) => void): Unlisten {
    if (!inTauri) return () => {};
    const un = on<string[]>('open-paths', () => invoke<string[]>('take_pending_paths').then(p => p.length && cb(p)));
    invoke<string[]>('take_pending_paths').then(p => p.length && cb(p));
    return un;
  },
  /** Files dropped on the window. The webview handles the drop natively and hands over real paths. */
  onDropPaths(cb: (p: string[]) => void): Unlisten {
    if (!inTauri) return () => {};
    let un: Unlisten | null = null, dead = false;
    getCurrentWebview()
      .onDragDropEvent(e => {
        if (e.payload.type === 'drop') cb(e.payload.paths);
      })
      .then(u => (dead ? u() : (un = u)));
    return () => {
      dead = true;
      un?.();
    };
  },
  win(action: 'min' | 'max' | 'close') {
    if (!inTauri) return;
    const w = getCurrentWindow();
    if (action === 'min') w.minimize();
    else if (action === 'max') w.toggleMaximize();
    else w.close();
  },
};
