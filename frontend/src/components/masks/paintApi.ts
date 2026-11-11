import { apiClient } from '../../api/client';
import { encodeEditorPng, readEditorPng } from './editorPng';
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
const MAX_DECODED_BYTES = 80 * 1024 * 1024;
const decodedImages = new Map<string, { datasetId: string; imageId: string; relPath: string; value: LoadedPaint }>();
type PendingPaint = { datasetId: string; relPath: string; controller: AbortController; readers: number; promise: Promise<LoadedPaint> };
const pendingImages = new Map<string, PendingPaint>();
export function canPreloadPaint(width?: number | null, height?: number | null) {
  // Each decoded pixel occupies four image bytes and one mask byte.
  return width != null && height != null && width > 0 && height > 0 && width * height * 5 <= MAX_DECODED_BYTES;
}
const decodedBytes = (value: LoadedPaint) => value.pixels.byteLength + value.mask.byteLength;
export function invalidatePaintCache(datasetId: string, relPath?: string) {
  for (const [key, item] of decodedImages) if (item.datasetId === datasetId && (!relPath || item.relPath === relPath)) decodedImages.delete(key);
  for (const [key, item] of pendingImages) if (item.datasetId === datasetId && (!relPath || item.relPath === relPath)) {
    pendingImages.delete(key); item.controller.abort();
  }
}
async function readPaint(datasetId: string, imageId: string, relPath: string, signal: AbortSignal) {
  const infoUrl = paintEndpoint(datasetId, imageId, relPath, '/info');
  const info = await apiClient.get<PaintInfo>(infoUrl, {signal, silent:true});
  const key = JSON.stringify([datasetId, imageId, relPath, info.revision]);
  const cached = decodedImages.get(key)?.value;
  const cacheable = canPreloadPaint(info.width, info.height);
  const readImage = () => readEditorPng(paintEndpoint(datasetId, imageId, relPath, '/source'), info, signal, 'image');
  const readMask = () => readEditorPng(maskEndpoint(datasetId, imageId, relPath), info, signal, 'mask');
  let frame: Pick<LoadedPaint, 'pixels' | 'mask'>;
  if (cached) {
    frame = cached;
  } else if (info.mask_source === 'full') {
    const pixels = await readImage();
    frame = { pixels, mask: new Uint8Array(info.width * info.height).fill(255) };
  } else if (cacheable) {
    const [pixels, mask] = await Promise.all([readImage(), readMask()]);
    frame = { pixels, mask };
  } else {
    // Keep the source and mask decoders from holding large bitmaps at the same time.
    const pixels = await readImage();
    frame = { pixels, mask: await readMask() };
  }
  // Pixel reads need a second snapshot check; cached pixels already match the fresh revision.
  const checked = cached ? info : await apiClient.get<PaintInfo>(infoUrl, {signal, silent:true});
  if (checked.revision !== info.revision) throw new Error('Image or mask changed during loading; reload the editor');
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  const value = { info: checked, pixels: frame.pixels, mask: frame.mask };
  if (cacheable) {
    decodedImages.delete(key);
    decodedImages.set(key, { datasetId, imageId, relPath, value });
    let bytes = [...decodedImages.values()].reduce((sum, entry) => sum + decodedBytes(entry.value), 0);
    while (decodedImages.size > 3 || bytes > MAX_DECODED_BYTES) {
      const [oldest, entry] = decodedImages.entries().next().value!;
      decodedImages.delete(oldest);
      bytes -= decodedBytes(entry.value);
    }
  }
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
export function exportPaint(width: number, height: number, pixels: Uint8ClampedArray): Promise<Blob> {
  return encodeEditorPng(width, height, pixels, 'image');
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
