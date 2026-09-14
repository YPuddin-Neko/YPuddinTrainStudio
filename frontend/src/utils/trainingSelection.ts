/** Apply only the dependent values required by an explicit component selection. */
export function selectTrainingComponents(next: Record<string, any>, previous: Record<string, any>) {
  if (JSON.stringify(next.training) === JSON.stringify(previous.training)) return next;
  const result = structuredClone(next);
  if (result.training?.mode === 'full') {
    result.memory = {...result.memory, base_precision:'fp32', blocks_to_swap:0};
    result.adapter = {...result.adapter, resume_weights:null};
    if (result.training.train_text_encoder) {
      result.memory.offload_text_encoder = false;
      result.dataset = {...result.dataset, text_encoding:'online'};
    }
  } else {
    result.training = {...result.training, train_backbone:true, train_text_encoder:false, resume_weights:null};
  }
  return result;
}

export function trainingManagedReason(config: Record<string, any>, path: string, en: boolean) {
  const full = config.training?.mode === 'full';
  if (!full && ['training.train_backbone','training.train_text_encoder'].includes(path)) return en ? 'Adapter training updates the main model through LoRA/LoKr. Choose full fine-tuning to train the text encoder.' : '适配器模式通过 LoRA/LoKr 训练主模型；训练文本编码器请选择全量微调。';
  if (full && path === 'memory.base_precision' && config.memory?.base_precision === 'fp32') return en ? 'Full fine-tuning retains FP32 trainable weights. Mixed precision controls forward computation.' : '全量微调保留 FP32 可训练权重；前向计算精度由混合精度控制。';
  if (full && path === 'memory.blocks_to_swap' && !config.memory?.blocks_to_swap) return en ? 'Block swapping currently supports frozen adapter bases only.' : '当前层换出仅支持适配器训练中的冻结底模。';
  if (full && config.training.train_text_encoder && ((path === 'dataset.text_encoding' && config.dataset?.text_encoding === 'online') || (path === 'memory.offload_text_encoder' && !config.memory?.offload_text_encoder))) return en ? 'The text encoder participates in every backward pass; online encoding is required and unloading is disabled.' : '文本编码器参与每次反向传播，需要在线编码并保留在计算设备上。';
  return undefined;
}
