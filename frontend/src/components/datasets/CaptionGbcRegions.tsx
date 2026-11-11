import { useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import type { GbcRegion } from '../../utils/captionStructure';

export default function CaptionGbcRegions({ regions }: { regions: GbcRegion[] }) {
  const overlay = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const frame = overlay.current?.parentElement;
    const image = frame?.querySelector('img');
    if (!frame || !image) return;
    const measure = () => {
      const scale = image.naturalWidth && image.naturalHeight
        ? Math.min(frame.clientWidth / image.naturalWidth, frame.clientHeight / image.naturalHeight)
        : 0;
      setSize({ width: image.naturalWidth * scale, height: image.naturalHeight * scale });
    };
    const resize = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    resize?.observe(frame);
    image.addEventListener('load', measure);
    image.addEventListener('error', measure);
    measure();
    return () => {
      resize?.disconnect();
      image.removeEventListener('load', measure);
      image.removeEventListener('error', measure);
    };
  }, []);

  return <div ref={overlay} className="caption-gbc-overlay" aria-hidden="true" style={{ ...size, visibility: size.width && size.height ? undefined : 'hidden' }}>
    {regions.map(({ label, box: [left, top, right, bottom] }, index) => <div key={index} className="caption-gbc-region" style={{
      left: `${left * 100}%`, top: `${top * 100}%`, width: `${(right - left) * 100}%`, height: `${(bottom - top) * 100}%`,
      '--caption-region-color': `hsl(${(275 + index * 137.508) % 360} 72% 65%)`,
    } as CSSProperties}>{label && <span>{label}</span>}</div>)}
  </div>;
}
