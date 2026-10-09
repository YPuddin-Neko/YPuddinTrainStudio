import type { ModelAsset, ModelDownload } from '../api/types';

export function modelOrigin(model: ModelAsset | undefined, downloads: ModelDownload[] | undefined): 'downloaded' | 'imported' | null {
  if (!model || downloads === undefined) return null;
  const downloaded = downloads.some(task => task.status === 'completed' && task.kind === model.kind && task.target_path === model.path
    && (!task.model_id || task.model_id === model.id || model.kind === 'vae' && task.family !== model.family));
  return downloaded ? 'downloaded' : 'imported';
}
