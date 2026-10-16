import type { ModelAsset } from '../api/types';

export function modelDisplayName(model: Pick<ModelAsset, 'path' | 'display_name'>): string {
  return typeof model.display_name === 'string' && model.display_name.trim()
    ? model.display_name.trim()
    : model.path.split(/[\\/]/).pop() || model.path;
}
