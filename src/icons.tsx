// Icons from the design, stroke-drawn at 24×24.
import type { CSSProperties } from 'react';

const stroke = (w = 1.75): CSSProperties => ({ fill: 'none', stroke: 'currentColor', strokeWidth: w, strokeLinecap: 'round', strokeLinejoin: 'round' });

type P = { size?: number; style?: CSSProperties };

export const IconFolder = ({ size = 15 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={stroke()}><path d="m6 14 1.45-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1-1.94 1.5H4a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2h3.93a2 2 0 0 1 1.66.9l.82 1.2a2 2 0 0 0 1.66.9H18a2 2 0 0 1 2 2v2" /></svg>
);
export const IconDownload = ({ size = 15, w = 1.75 }: P & { w?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={stroke(w)}><path d="M12 15V3M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5" /></svg>
);
export const IconBack10 = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" style={stroke()}><path d="m11 17-5-5 5-5M18 17l-5-5 5-5" /></svg>
);
export const IconFwd10 = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" style={stroke()}><path d="m6 17 5-5-5-5M13 17l5-5-5-5" /></svg>
);
export const IconPrev = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" style={stroke()}><path d="m15 18-6-6 6-6" /></svg>
);
export const IconNext = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" style={stroke()}><path d="m9 18 6-6-6-6" /></svg>
);
export const IconPause = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" style={{ fill: 'currentColor' }}><rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" /></svg>
);
export const IconPlay = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" style={{ fill: 'currentColor', marginLeft: 2 }}><path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5z" /></svg>
);
export const IconFilm = ({ size = 15 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={stroke()}><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M7 3v18M17 3v18M3 12h18M3 7.5h4M3 16.5h4M17 7.5h4M17 16.5h4" /></svg>
);
export const IconWave = ({ size = 15 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={stroke()}><path d="M2 10v3M6 6v11M10 3v18M14 8v7M18 5v13M22 10v3" /></svg>
);
export const IconImage = ({ size = 15 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={stroke()}><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="2" /><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21" /></svg>
);
export const IconSearch = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" style={stroke(2)}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
);
export const IconX = ({ size = 14, w = 1.75 }: P & { w?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={stroke(w)}><path d="M18 6 6 18M6 6l12 12" /></svg>
);
export const IconCheck = ({ size = 12, w = 3 }: P & { w?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={stroke(w)}><path d="M20 6 9 17l-5-5" /></svg>
);
export const IconPlus = ({ size = 14 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={stroke(2)}><path d="M12 5v14M5 12h14" /></svg>
);
export const IconReveal = ({ size = 15 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" style={stroke()}><path d="M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /></svg>
);
