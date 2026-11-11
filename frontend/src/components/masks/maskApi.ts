import { apiClient } from '../../api/client';
import { encodeEditorPng, readEditorPng } from './editorPng';

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
  return { info, pixels: await readEditorPng(endpoint, info, signal, 'mask') };
}
export function exportMask(width: number, height: number, pixels: Uint8Array): Promise<Blob> {
  return encodeEditorPng(width, height, pixels, 'mask');
}
export async function saveMask(datasetId: string, imageId: string, relPath: string, info: MaskInfo, pixels: Uint8Array) {
  const blob = await exportMask(info.width, info.height, pixels);
  const form = new FormData(); form.append('file', blob, info.filename); form.append('revision', info.revision);
  return apiClient.put<MaskInfo>(maskEndpoint(datasetId, imageId, relPath), form, { silent: true });
}
