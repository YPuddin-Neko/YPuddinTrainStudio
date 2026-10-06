/** Export identity is determined from the saved artifact and its training snapshot by the server. */
export function artifactTypeLabel(artifact: { kind: string; export_type?: string | null }, english = false): string {
  switch (artifact.export_type) {
    case 'lora': return 'LoRA';
    case 'checkpoint': return 'Checkpoint';
    case 'diffusion_model': return 'Diffusion Model';
    default: return artifact.kind === 'model' ? (english ? 'Full-model components' : '全量模型组件') : artifact.kind === 'weights' ? (english ? 'Weights' : '权重') : artifact.kind;
  }
}
