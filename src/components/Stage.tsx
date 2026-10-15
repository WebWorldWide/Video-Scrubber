// SPDX-License-Identifier: GPL-3.0-or-later
import type { ReactNode, RefObject } from 'react';
import { frameUrl, type Detail, type Video } from '../api';
import { fmtD, grad, hueOf } from '../format';

const corner = (pos: React.CSSProperties) => (
  <div style={{ position: 'absolute', width: 14, height: 14, borderColor: 'rgba(20,24,31,.3)', borderStyle: 'solid', borderWidth: 0, ...pos }} />
);

/** The design's framed placeholder: corner marks, eyebrow row, serif headline. */
function Placeholder({ left, right, headline, foot, bg, onClick }: { left: string; right: string; headline: ReactNode; foot: string; bg: string; onClick: () => void }) {
  return (
    <div onClick={onClick} style={{ position: 'relative', width: 'min(100cqw, calc(100cqh * 16 / 9))', aspectRatio: '16/9', background: bg, cursor: 'pointer', overflow: 'hidden' }}>
      {corner({ left: 16, top: 16, borderLeftWidth: 1.5, borderTopWidth: 1.5 })}
      {corner({ right: 16, top: 16, borderRightWidth: 1.5, borderTopWidth: 1.5 })}
      {corner({ left: 16, bottom: 16, borderLeftWidth: 1.5, borderBottomWidth: 1.5 })}
      {corner({ right: 16, bottom: 16, borderRightWidth: 1.5, borderBottomWidth: 1.5 })}
      <div className="tnum" style={{ position: 'absolute', left: 40, top: 34, right: 40, display: 'flex', justifyContent: 'space-between', fontSize: 10.5, fontWeight: 600, fontStretch: '80%', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#2A3240' }}>
        <span>{left}</span><span>{right}</span>
      </div>
      <div style={{ position: 'absolute', left: 40, right: 40, bottom: 32, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="serif" style={{ maxWidth: '88%', fontSize: 'clamp(20px, 3.6cqw, 44px)', lineHeight: 1, letterSpacing: '-0.01em', color: '#14181F', textWrap: 'balance' } as React.CSSProperties}>{headline}</div>
        <div style={{ fontSize: 10.5, fontWeight: 600, fontStretch: '80%', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#4A5464', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{foot}</div>
      </div>
    </div>
  );
}

type Props = {
  video: Video | null;
  detail: Detail | null;
  src: string | null;
  t: number;
  curScene: number;
  videoRef: RefObject<HTMLVideoElement>;
  onMeta: () => void;
  onEnded: () => void;
  onError: () => void;
  onPlay: () => void;
  onOpen: () => void;
};

export function Stage({ video, detail, src, t, curScene, videoRef, onMeta, onEnded, onError, onPlay, onOpen }: Props) {
  const audioOnly = video && !video.probe.hasVideo;
  const scene = detail?.scenes[curScene];
  return (
    <div style={{ flex: '1 1 auto', minHeight: 260, position: 'relative', borderRadius: 16, border: '1px solid #E4E9F0', background: '#EAEEF4', overflow: 'hidden', containerType: 'size', display: 'flex', alignItems: 'center', justifyContent: 'center' } as React.CSSProperties}>
      {video && src && (
        <video
          key={src}
          ref={videoRef}
          src={src}
          onLoadedMetadata={onMeta}
          onEnded={onEnded}
          onError={onError}
          onClick={onPlay}
          playsInline
          preload="auto"
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', cursor: 'pointer', opacity: audioOnly ? 0 : 1 }}
        />
      )}
      {!video && (
        <Placeholder
          left="No video" right="00:00.0" bg={grad(214)} onClick={onOpen}
          headline={<>Drop videos here, or open a few to <i>comb through at once</i></>}
          foot="MP4 · MOV · MKV · WebM · AVI · audio files too"
        />
      )}
      {audioOnly && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
          <div style={{ pointerEvents: 'auto', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Placeholder left="Audio" right={fmtD(t)} bg={grad(hueOf(video.name))} onClick={onPlay} headline={video.name} foot={`${detail?.transcript.length ?? 0} segments transcribed`} />
          </div>
        </div>
      )}
      {video && !audioOnly && !video.exists && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Placeholder left="Missing" right="" bg={scene ? `center / cover no-repeat url("${frameUrl(video.id, scene.f)}")` : grad(hueOf(video.name))} onClick={onOpen}
            headline={<>This file has <i>moved</i></>} foot={video.path} />
        </div>
      )}
    </div>
  );
}
