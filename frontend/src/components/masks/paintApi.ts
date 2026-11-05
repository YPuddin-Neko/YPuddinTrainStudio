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
type LoadedPaint = { info: PaintInfo; pixels: Uint8ClampedArray; mask: Uint8Array };
const decodedImages = new Map<string, { datasetId: string; imageId: string; relPath: string; value: LoadedPaint }>();
type PendingPaint = { datasetId: string; relPath: string; controller: AbortController; readers: number; promise: Promise<LoadedPaint> };
const pendingImages = new Map<string, PendingPaint>();
export function invalidatePaintCache(datasetId: string, relPath?: string) {
  for (const [key, item] of decodedImages) if (item.datasetId === datasetId && (!relPath || item.relPath === relPath)) decodedImages.delete(key);
  for (const [key, item] of pendingImages) if (item.datasetId === datasetId && (!relPath || item.relPath === relPath)) {
    pendingImages.delete(key); item.controller.abort();
  }
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
async function readPaint(datasetId: string, imageId: string, relPath: string, signal: AbortSignal) {
  const infoUrl = paintEndpoint(datasetId, imageId, relPath, '/info');
  const info = await apiClient.get<PaintInfo>(infoUrl, {signal, silent:true});
  const key = JSON.stringify([datasetId, imageId, relPath, info.revision]);
  const cached = decodedImages.get(key)?.value;
  const [pixels, maskRgba] = cached ? [cached.pixels, null] : await Promise.all([
    decode(paintEndpoint(datasetId, imageId, relPath, '/source'), signal, info.width, info.height),
    decode(maskEndpoint(datasetId, imageId, relPath), signal, info.width, info.height),
  ]);
  // Pixel reads need a second snapshot check; cached pixels already match the fresh revision.
  const checked = cached ? info : await apiClient.get<PaintInfo>(infoUrl, {signal, silent:true});
  if (checked.revision !== info.revision) throw new Error('Image or mask changed during loading; reload the editor');
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  const mask = cached?.mask ?? new Uint8Array(info.width * info.height);
  if (maskRgba) for (let i = 0; i < mask.length; i++) mask[i] = maskRgba[i * 4];
  const value = {info: checked, pixels, mask};
  decodedImages.delete(key); decodedImages.set(key, {datasetId, imageId, relPath, value});
  while (decodedImages.size > 3) decodedImages.delete(decodedImages.keys().next().value!);
  return value;
}
export function loadPaint(datasetId: string, imageId: string, relPath: string, signal: AbortSignal): Promise<LoadedPaint> {
  if (signal.aborted) return Promise.reject(new DOMException('Aborted', 'AbortError'));
  const key = JSON.stringify([datasetId, imageId, relPath]);
  let pending = pendingImages.get(key);
  if (!pending) {
    const controller = new AbortController();
    pending = {datasetId, relPath, controller, readers: 0, promise: readPaint(datasetId, imageId, relPath, controller.signal)};
    const entry = pending;
    entry.promise = entry.promise.finally(() => { if (pendingImages.get(key) === entry) pendingImages.delete(key); });
    pendingImages.set(key, entry);
  }
  const entry = pending;
  entry.readers++;
  return new Promise((resolve, reject) => {
    let settled = false;
    const release = () => {
      settled = true; signal.removeEventListener('abort', abort); entry.readers--;
      if (!entry.readers && pendingImages.get(key) === entry) { pendingImages.delete(key); entry.controller.abort(); }
    };
    const abort = () => { if (!settled) { release(); reject(new DOMException('Aborted', 'AbortError')); } };
    signal.addEventListener('abort', abort, {once:true});
    entry.promise.then(value => { if (!settled) { release(); resolve(value); } }, error => { if (!settled) { release(); reject(error); } });
  });
}
export async function preloadPaint(datasetId: string, imageId: string, relPath: string, signal: AbortSignal) {
  if ([...decodedImages.values()].some(item => item.datasetId === datasetId && item.imageId === imageId && item.relPath === relPath)) return;
  await loadPaint(datasetId, imageId, relPath, signal);
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
  const saved = await apiClient.put<PaintInfo>(paintEndpoint(datasetId, info.image_id, info.rel_path), form, {silent:true});
  invalidatePaintCache(datasetId, info.rel_path); return saved;
}
export async function restorePaint(datasetId: string, info: PaintInfo) {
  const restored = await apiClient.post<PaintInfo>(paintEndpoint(datasetId, info.image_id, info.rel_path, '/restore'), {revision:info.revision}, {silent:true});
  invalidatePaintCache(datasetId, info.rel_path); return restored;
}
