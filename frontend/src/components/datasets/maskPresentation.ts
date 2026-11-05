import type { VisionModel } from '../../api/types';

export const MASK_KINDS = ['head', 'torso', 'legs', 'foot'] as const;
export type MaskKind = typeof MASK_KINDS[number];
export const MASK_MODEL_IDS: Record<MaskKind, string> = { head: 'anime-head-detector-v2', torso: 'anime-body-detector-yolo11m', legs: 'anime-body-detector-yolo11m', foot: 'anime-foot-detector-v3' };
const labels: Record<MaskKind, [string, string]> = { head: ['头部', 'Heads'], torso: ['躯干', 'Torso'], legs: ['腿部', 'Legs'], foot: ['脚部', 'Feet'] };
export function maskKindLabel(kind: string, english: boolean) {
  return labels[kind as MaskKind]?.[english ? 1 : 0] || kind;
}
export function modelMaskKinds(model: VisionModel): MaskKind[] {
  return model.mask_kinds?.length ? model.mask_kinds.filter((kind): kind is MaskKind => MASK_KINDS.includes(kind as MaskKind)) : model.role === 'head_detector' ? ['head'] : [];
}
