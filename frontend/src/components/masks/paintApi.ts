import { apiClient, apiUrl } from '../../api/client';
import { ApiError } from '../../api/types';
import { exportMask, maskEndpoint } from './maskApi';

export interface PaintInfo {
  width: number; height: number; revision: string; image_id: string; rel_path: string;
  has_mask: boolean; mask_source: 'sidecar' | 'alpha' | 'full'; can_restore: boolean;
  last_operation_id?: string | null; operation_id?: string;
}
export function paintEndpoint(datasetId: string, imageId: string, relPath: string, suffix = '') {
  return `/datasets/${encodeURIComponent(datasetId)}/images/${encodeURIComponent(imageId)}/paint${suffix}?rel_path=${encodeURIComponent(relPath)}`;
}
async function decode(url: string, signal: AbortSignal, width: number, height: number) {
  const response = await fetch(apiUrl(url), { signal, cache: 'no-store' });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(response.status, payload?.error || { code: 'paint.load', message: `HTTP ${response.status}` });
  }
  const objectUrl = URL.createObjectURL(await response.blob());
  try {
    const image = new Image();
    await new Promise<void>((resolve, reject) => { image.onload = () => resolve(); image.onerror = () => reject(new Error('Cannot decode image PNG')); image.src = objectUrl; });
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    if (image.naturalWidth !== width || image.naturalHeight !== height) throw new Error('Image changed during loading; reload the editor');
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    try {
      const context = canvas.getContext('2d'); if (!context) throw new Error('Canvas is unavailable');
      context.drawImage(image, 0, 0); return context.getImageData(0, 0, width, height).data;
    } finally { canvas.width = 1; canvas.height = 1; }
  } finally { URL.revokeObjectURL(objectUrl); }
}
export async function loadPaint(datasetId: string, imageId: string, relPath: string, signal: AbortSignal) {
  const infoUrl = paintEndpoint(datasetId, imageId, relPath, '/info');
  const info = await apiClient.get<PaintInfo>(infoUrl, {signal, silent:true});
  const [pixels, maskRgba] = await Promise.all([
    decode(paintEndpoint(datasetId, imageId, relPath, '/source'), signal, info.width, info.height),
    decode(maskEndpoint(datasetId, imageId, relPath), signal, info.width, info.height),
  ]);
  // All three requests must describe the same snapshot, even when another client is editing.
  const checked = await apiClient.get<PaintInfo>(infoUrl, {signal, silent:true});
  if (checked.revision !== info.revision) throw new Error('Image or mask changed during loading; reload the editor');
  const mask = new Uint8Array(info.width * info.height);
  for (let i = 0; i < mask.length; i++) mask[i] = maskRgba[i * 4];
  return {info, pixels, mask};
}
export async function exportPaint(width: number, height: number, pixels: Uint8ClampedArray): Promise<Blob> {
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const context = canvas.getContext('2d'); if (!context) throw new Error('Canvas is unavailable');
  const image = context.createImageData(width, height); image.data.set(pixels); context.putImageData(image, 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob(blob => {
    canvas.width = 1; canvas.height = 1;
    if (blob) resolve(blob); else reject(new Error('Could not encode image PNG'));
  }, 'image/png'));
}
export async function savePaint(datasetId: string, info: PaintInfo, pixels?: Uint8ClampedArray, mask?: Uint8Array) {
  const form = new FormData(); form.append('revision', info.revision);
  if (pixels) form.append('file', await exportPaint(info.width, info.height, pixels), 'paint.png');
  if (mask) form.append('mask', await exportMask(info.width, info.height, mask), 'mask.png');
  return apiClient.put<PaintInfo>(paintEndpoint(datasetId, info.image_id, info.rel_path), form, {silent:true});
}
export function restorePaint(datasetId: string, info: PaintInfo) {
  return apiClient.post<PaintInfo>(paintEndpoint(datasetId, info.image_id, info.rel_path, '/restore'), {revision:info.revision}, {silent:true});
}
