export function modelComponentLabel(family: string, kind: string, text: (zh: string, en: string) => string): string {
  if (kind === 'dit') return family === 'sdxl' ? 'Checkpoint / UNet'
    : ['anima', 'krea2', 'flux2', 'flux3'].includes(family) ? 'DiT' : text('主模型', 'Base model');
  if (kind === 'vae') return 'VAE';
  if (kind === 'text_encoder') return family === 'sdxl' ? 'CLIP-L' : text('文本编码器', 'Text encoder');
  if (kind === 'text_encoder_2') return family === 'sdxl' ? 'CLIP-G' : text('第二文本编码器', 'Second text encoder');
  if (kind === 'tokenizer') return family === 'flux3' ? text('文本处理器', 'Text processor') : text('分词器目录', 'Tokenizer directory');
  return kind;
}
