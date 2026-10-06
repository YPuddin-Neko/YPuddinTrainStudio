export function optionalValueLabel(name: string, english: boolean, modelDefault?: string) {
  const labels: Record<string, [string, string]> = {
    'dataset.bucket_step': ['自动对齐', 'Automatic alignment'],
    'optimizer.beta3': ['自动（√β2）', 'Automatic (√β2)'],
    'optimizer.growth_rate': ['不限制', 'Unlimited'],
    'loop.max_steps': ['不限制步数', 'No step limit'],
    'loop.epochs': ['由最大步数决定', 'Use the step limit'],
    'checkpoint.keep_last_n': ['全部保留', 'Keep all'],
    'scheduler.decay_steps': ['自动（总步数的 10%）', 'Automatic (10% of steps)'],
    'validation.max_images': ['使用全部验证图', 'Use all validation images'],
  };
  if (/^training\..*_lr$/.test(name)) return english ? 'Inherit learning rate' : '跟随学习率';
  if (labels[name]) return labels[name][english ? 1 : 0];
  if (/^(sampling|validation|checkpoint)\.(every_|save_every_|save_state_every_)/.test(name)) return english ? 'Disabled' : '关闭';
  if (/^sampling\.(steps|cfg|shift|guidance)$/.test(name)) {
    if (modelDefault && !Number.isFinite(Number(modelDefault))) return modelDefault;
    const suffix = modelDefault && Number.isFinite(Number(modelDefault)) ? ` (${modelDefault})` : '';
    return (english ? 'Model default' : '沿用模型默认值') + suffix;
  }
  return english ? 'Automatic' : '自动';
}
