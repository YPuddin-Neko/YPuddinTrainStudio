import { FAMILY_NAMES } from './trainingFamilies';

/** Export identity is determined from the saved artifact and its training snapshot by the server. */
export function artifactTypeLabel(artifact: { kind: string; export_type?: string | null }, english = false): string {
  switch (artifact.export_type) {
    case 'lora': return 'LoRA';
    case 'checkpoint': return 'Checkpoint';
    case 'diffusion_model': return 'Diffusion Model';
    default: return artifact.kind === 'model' ? (english ? 'Full-model components' : '全量模型组件') : artifact.kind === 'weights' ? (english ? 'Weights' : '权重') : artifact.kind;
  }
}
/** Use saved output identity; a source job's frozen family only fills missing metadata. */
export function artifactBaseModelLabel(artifact: { family?: string | null; metadata?: Record<string, unknown> }, sourceFamily?: string | null, english = false): string {
  const nonempty = (value: unknown) => typeof value === 'string' && value.trim() ? value.trim() : undefined;
  const baseVersion = nonempty(artifact.metadata?.['ss_base_model_version']);
  const savedFamily = nonempty(artifact.family) || nonempty(artifact.metadata?.['ypuddin.family']) || baseVersion;
  const aliases: Record<string, string> = { 'sdxl_base_v1-0': 'sdxl', flux1: 'flux', flux_2_klein_4b: 'flux2', flux_2_klein_9b: 'flux2' };
  const recordedFamily = savedFamily || nonempty(sourceFamily);
  const family = recordedFamily && (aliases[recordedFamily] || recordedFamily);
  if (!family) return english ? 'Unknown' : '未知';
  if (family === 'flux2') {
    const variant = baseVersion || savedFamily;
    return variant === 'flux_2_klein_4b' ? 'FLUX.2 Klein 4B' : variant === 'flux_2_klein_9b' ? 'FLUX.2 Klein 9B' : 'FLUX.2';
  }
  if (family === 'krea2') return 'Krea 2';
  return FAMILY_NAMES[family] || family;
}
