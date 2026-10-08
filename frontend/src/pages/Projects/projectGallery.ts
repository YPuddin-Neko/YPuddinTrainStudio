import { apiUrl } from '../../api/client';
import type { Project } from '../../api/types';

// `deletion`: files are being removed in the background, or the last deletion stopped and can be retried.
export type ProjectDeletion = { state: 'deleting' | 'failed'; error?: string | null; task_id?: string | null };
export type GalleryProject = Project & { category?: string | null; cover_url?: string | null; active_family?: string | null; deletion?: ProjectDeletion | null };
export const PROJECT_CATEGORIES = ['大模型微调', '人物 LoRA', '画风 LoRA', '语音'];
export function categoryLabel(value: string, english = false) {
  return english ? ({ '大模型微调': 'Base model finetuning', '人物 LoRA': 'Character LoRA', '画风 LoRA': 'Style LoRA', '语音': 'Speech' } as Record<string, string>)[value] || value : value;
}
export function coverSource(url: string) { return new URL(url, apiUrl('/')).toString(); }
