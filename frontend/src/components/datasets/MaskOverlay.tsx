import { useEffect, useRef, useState } from 'react';

const LONGEST_SIDE = 1024;

/** Tints the parts of an image that training leaves out, read from its grayscale loss mask. */
export default function MaskOverlay({ src, label }: { src: string; label: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setFailed(false);
    const mask = new Image();
    mask.onload = () => {
      const target = canvas.current;
      const context = target?.getContext('2d', { willReadFrequently: true });
      if (cancelled || !target || !context) return;
      // The preview is small; a reduced copy keeps large masks cheap to repaint.
      const scale = Math.min(1, LONGEST_SIDE / Math.max(mask.naturalWidth, mask.naturalHeight));
      target.width = Math.max(1, Math.round(mask.naturalWidth * scale));
      target.height = Math.max(1, Math.round(mask.naturalHeight * scale));
      context.drawImage(mask, 0, 0, target.width, target.height);
      const pixels = context.getImageData(0, 0, target.width, target.height);
      const data = pixels.data;
      for (let index = 0; index < data.length; index += 4) {
        const left = 255 - data[index];
        data[index] = 239; data[index + 1] = 68; data[index + 2] = 68;
        data[index + 3] = Math.round(left * 0.62);
      }
      context.putImageData(pixels, 0, 0);
    };
    mask.onerror = () => { if (!cancelled) setFailed(true); };
    mask.src = src;
    return () => { cancelled = true; mask.onload = null; mask.onerror = null; };
  }, [src]);
  if (failed) return null;
  return <canvas ref={canvas} className="caption-viewer-mask" role="img" aria-label={label}/>;
}
