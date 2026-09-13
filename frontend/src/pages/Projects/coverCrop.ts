import type { CSSProperties } from 'react';

export const COVER_MAX_BYTES = 8 * 1024 * 1024;
export const COVER_ASPECT = 16 / 10;
export type CoverCrop = { x: number; y: number; width: number; height: number };
export type CropView = { zoom: number; cx: number; cy: number };
export const INITIAL_CROP_VIEW: CropView = { zoom: 1, cx: .5, cy: .5 };
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

/** Normalized coordinates refer to the image after EXIF orientation correction. */
export function cropForView(imageWidth: number, imageHeight: number, view: CropView): CoverCrop {
  const ratio = imageWidth / imageHeight;
  const zoom = clamp(view.zoom, 1, 4);
  const width = Math.min(1, COVER_ASPECT / ratio) / zoom;
  const height = Math.min(1, ratio / COVER_ASPECT) / zoom;
  return {
    x: clamp(view.cx - width / 2, 0, 1 - width),
    y: clamp(view.cy - height / 2, 0, 1 - height), width, height,
  };
}

export function viewForCrop(imageWidth: number, imageHeight: number, crop: CoverCrop): CropView {
  const base = cropForView(imageWidth, imageHeight, INITIAL_CROP_VIEW);
  return { zoom: base.width / crop.width, cx: crop.x + crop.width / 2, cy: crop.y + crop.height / 2 };
}

/** Both the editor and the card preview display the exact server-side crop. */
export function cropImageStyle(crop: CoverCrop): CSSProperties {
  return { position: 'absolute', maxWidth: 'none', width: `${100 / crop.width}%`, height: `${100 / crop.height}%`, left: `${-100 * crop.x / crop.width}%`, top: `${-100 * crop.y / crop.height}%` };
}
