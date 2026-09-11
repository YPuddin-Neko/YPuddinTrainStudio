import { apiClient, apiUrl } from '../../api/client';
import { ApiError } from '../../api/types';

export interface MaskInfo {
  source: 'sidecar' | 'alpha' | 'full'; width: number; height: number;
  coverage: number; has_mask: boolean; filename: string; revision: string; resized: boolean;
}
export function maskEndpoint(datasetId: string, imageId: string, relPath: string, suffix = '') {
  return `/datasets/${encodeURIComponent(datasetId)}/images/${encodeURIComponent(imageId)}/mask${suffix}?rel_path=${encodeURIComponent(relPath)}`;
}
export async function loadMask(datasetId: string, imageId: string, relPath: string, signal: AbortSignal) {
  const endpoint = maskEndpoint(datasetId, imageId, relPath);
  const info = await apiClient.get<MaskInfo>(maskEndpoint(datasetId, imageId, relPath, '/info'), { signal, silent: true });
  const response = await fetch(apiUrl(endpoint), { signal, cache: 'no-store' });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(response.status, payload?.error || { code: 'mask.load', message: `HTTP ${response.status}` });
  }
  const objectUrl = URL.createObjectURL(await response.blob());
  try {
    const image = new Image();
    await new Promise<void>((resolve, reject) => { image.onload = () => resolve(); image.onerror = () => reject(new Error('Cannot decode mask PNG')); image.src = objectUrl; });
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    if (image.naturalWidth !== info.width || image.naturalHeight !== info.height) throw new Error('Image changed during loading; reload the editor');
    const canvas = document.createElement('canvas'); canvas.width = info.width; canvas.height = info.height;
    const context = canvas.getContext('2d'); if (!context) throw new Error('Canvas is unavailable');
    context.drawImage(image, 0, 0);
    const rgba = context.getImageData(0, 0, info.width, info.height).data;
    const pixels = new Uint8Array(info.width * info.height);
    for (let i = 0; i < pixels.length; i++) pixels[i] = rgba[i * 4];
    canvas.width = 1; canvas.height = 1;
    return { info, pixels };
  } finally { URL.revokeObjectURL(objectUrl); }
}
export async function exportMask(width: number, height: number, pixels: Uint8Array): Promise<Blob> {
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const context = canvas.getContext('2d'); if (!context) throw new Error('Canvas is unavailable');
  const output = context.createImageData(width, height);
  for (let i = 0; i < pixels.length; i++) { const offset = i * 4; output.data[offset] = output.data[offset + 1] = output.data[offset + 2] = pixels[i]; output.data[offset + 3] = 255; }
  context.putImageData(output, 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob((blob) => { canvas.width = 1; canvas.height = 1; if (blob) resolve(blob); else reject(new Error('Could not encode mask PNG')); }, 'image/png'));
}
export async function saveMask(datasetId: string, imageId: string, relPath: string, info: MaskInfo, pixels: Uint8Array) {
  const blob = await exportMask(info.width, info.height, pixels);
  const form = new FormData(); form.append('file', blob, info.filename); form.append('revision', info.revision);
  return apiClient.put<MaskInfo>(maskEndpoint(datasetId, imageId, relPath), form, { silent: true });
}
