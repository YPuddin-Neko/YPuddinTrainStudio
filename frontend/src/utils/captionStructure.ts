import type { components } from '../api/generated';

export type CaptionFieldValue = string | string[];
export interface CaptionField { path: string[]; role: components['schemas']['CaptionField']['role']; value: CaptionFieldValue; present: boolean }
export interface CaptionStructure {
  format: 'full' | 'nested' | 'simple' | 'flat' | 'legacy_override' | 'unknown';
  document: Record<string, unknown>; fields: CaptionField[]; revision: string;
  editable: boolean; legacy_override: boolean; reason?: string | null;
}
export interface CaptionFieldChange { path: string[]; value: CaptionFieldValue }
export type CaptionFieldDraft = Record<string, CaptionFieldValue>;
export const captionFieldKey = (path: string[]) => JSON.stringify(path);
export function getCaptionStructure(image: object | undefined): CaptionStructure | null {
  return (image as { caption_structure?: CaptionStructure | null } | undefined)?.caption_structure ?? null;
}
export function captionFieldChanges(structure: CaptionStructure | null, draft: CaptionFieldDraft): CaptionFieldChange[] {
  return (structure?.fields || []).flatMap(field => {
    const value = draft[captionFieldKey(field.path)];
    return value !== undefined && JSON.stringify(value) !== JSON.stringify(field.value) ? [{ path: field.path, value }] : [];
  });
}
/** Work only on a copy; never serialize this view back as an edited document. */
export function captionMetadata(structure: CaptionStructure): Record<string, unknown> {
  const copy = JSON.parse(JSON.stringify(structure.document)) as Record<string, unknown>;
  const remove = (node: Record<string, unknown>, path: string[]) => {
    const [head, ...tail] = path;
    if (!head) return;
    if (!tail.length) { delete node[head]; return; }
    const child = node[head];
    if (child && typeof child === 'object' && !Array.isArray(child)) {
      remove(child as Record<string, unknown>, tail);
      if (!Object.keys(child).length) delete node[head];
    }
  };
  for (const field of structure.fields) if (field.present) remove(copy, field.path);
  return copy;
}
