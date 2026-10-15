// SPDX-License-Identifier: GPL-3.0-or-later
export const pad = (n: number) => String(n).padStart(2, '0');

export const fmt = (t: number) => {
  t = Math.max(0, t || 0);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  return (h ? h + ':' + pad(m) : pad(m)) + ':' + pad(s);
};

export const fmtD = (t: number) => fmt(t) + '.' + Math.floor(((t || 0) % 1) * 10);

export const fmtSize = (b: number) =>
  b >= 1e9 ? (b / 1e9).toFixed(1) + ' GB' : b > 1e6 ? (b / 1e6).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1e3)) + ' KB';

export const grad = (h: number) => `linear-gradient(155deg, hsl(${h} 62% 95%), hsl(${h} 40% 84%))`;

/** Stable hue per string, for placeholder gradients. */
export const hueOf = (s: string) => [...s].reduce((a, c) => (a * 31 + c.charCodeAt(0)) % 9973, 7) % 360;

export const stem = (name: string) => name.replace(/\.[^.]+$/, '');
