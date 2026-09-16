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
  if (!full && path === 'training.train_backbone' && config.training?.train_backbone !== false) return en ? 'LoRA trains adapters on the main model. Choose LoRA, LoKr or another algorithm in the adapter settings.' : 'LoRA 模式训练主模型上的适配器；具体使用 LoRA、LoKr 等算法，在适配器设置中选择。';
  if (!full && path === 'training.train_text_encoder' && !config.training?.train_text_encoder) return en ? 'Text encoder LoRA is not implemented yet; the text encoder stays frozen. Full fine-tuning can update its original weights.' : '目前尚未实现文本编码器 LoRA，此处保持冻结。全量微调模式可更新文本编码器原始权重。';
  if (full && path === 'memory.base_precision' && config.memory?.base_precision === 'fp32') return en ? 'Full fine-tuning retains FP32 trainable weights. Mixed precision controls forward computation.' : '全量微调保留 FP32 可训练权重；前向计算精度由混合精度控制。';
  if (full && path === 'memory.blocks_to_swap' && !config.memory?.blocks_to_swap) return en ? 'Block swapping currently supports frozen adapter bases only.' : '当前层换出仅支持适配器训练中的冻结底模。';
  if (full && config.training.train_text_encoder && ((path === 'dataset.text_encoding' && config.dataset?.text_encoding === 'online') || (path === 'memory.offload_text_encoder' && !config.memory?.offload_text_encoder))) return en ? 'The text encoder participates in every backward pass; online encoding is required and unloading is disabled.' : '文本编码器参与每次反向传播，需要在线编码并保留在计算设备上。';
  return undefined;
}
