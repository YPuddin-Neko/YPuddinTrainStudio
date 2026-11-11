import type { components } from '../api/generated';

export type CaptionFieldValue = string | string[];
export interface CaptionField { path: string[]; role: components['schemas']['CaptionField']['role']; value: CaptionFieldValue; present: boolean }
export interface CaptionStructure {
  format: 'full' | 'nested' | 'simple' | 'flat' | 'legacy_override' | 'unknown' | 'gbc' | 'moss_layout';
  document: Record<string, unknown>; fields: CaptionField[]; revision: string;
  editable: boolean; legacy_override: boolean; reason?: string | null;
}
export interface CaptionFieldChange { path: string[]; value: CaptionFieldValue }
export type CaptionFieldDraft = Record<string, CaptionFieldValue>;
export type GbcRegion = { label: string; box: [number, number, number, number] };
export const captionFieldKey = (path: string[]) => JSON.stringify(path);
export function getCaptionStructure(image: object | undefined): CaptionStructure | null {
  return (image as { caption_structure?: CaptionStructure | null } | undefined)?.caption_structure ?? null;
}
export function isSceneCaption(structure: CaptionStructure | null): boolean {
  return structure?.format === 'gbc' || structure?.format === 'moss_layout';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function gbcRegions(structure: CaptionStructure | null): GbcRegion[] {
  if (structure?.format !== 'gbc') return [];
  const graph = isRecord(structure.document.graph) ? structure.document.graph : structure.document;
  if (!Array.isArray(graph.vertices)) return [];
  const vertices = graph.vertices.filter(isRecord);
  const names = new Map<string, string>();
  for (const vertex of vertices) {
    if (!Array.isArray(vertex.out_edges)) continue;
    for (const edge of vertex.out_edges.filter(isRecord)) {
      if (typeof edge.target === 'string' && typeof edge.text === 'string' && edge.text.trim() && !names.has(edge.target)) names.set(edge.target, edge.text.trim());
    }
  }
  return vertices.flatMap(vertex => {
    if ((vertex.label !== 'entity' && vertex.label !== 'composition') || !isRecord(vertex.bbox)) return [];
    const coords = [vertex.bbox.left, vertex.bbox.top, vertex.bbox.right, vertex.bbox.bottom];
    if (!coords.every((value): value is number => typeof value === 'number' && Number.isFinite(value))) return [];
    const [left, top, right, bottom] = coords.map(value => Math.max(0, Math.min(1, value)));
    if (left >= right || top >= bottom) return [];
    const descriptions = Array.isArray(vertex.descs) ? vertex.descs.filter(isRecord) : [];
    const description = descriptions.find(item => item.label === 'short' && typeof item.text === 'string')?.text;
    const id = typeof vertex.vertex_id === 'string' ? vertex.vertex_id : '';
    const label = names.get(id) || (typeof description === 'string' ? description : id);
    return [{ label, box: [left, top, right, bottom] as GbcRegion['box'] }];
  });
}

export function captionStructureLabel(structure: CaptionStructure | null, english = false): string {
  if (structure?.format === 'gbc') return 'GBC JSON';
  if (structure?.format === 'moss_layout') return 'MOSS Layout JSON';
  if (!structure || structure.format === 'unknown') return english ? 'JSON captions' : 'JSON 标签';
  return 'Anima JSON';
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
