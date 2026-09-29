import type { FamilyInfo } from '../api/types';
import { configOptionLabel } from './configPresentation';
import { familyHasConvolutions, presetHasConvolutions, selectedPreset } from './fieldContext';

/**
 * The layer types a run cannot choose: full fine-tuning of the main model trains every parameter,
 * convolutions included, and a scope without convolutions trains linear layers only.
 */
export function adapterLayerTypesLock(config: Record<string, any>, family: FamilyInfo | undefined, en: boolean) {
  if (!familyHasConvolutions(family)) return null;
  const lock = (value: 'linear' | 'linear_conv', reason: string) => ({
    value, reason, label: configOptionLabel('adapter.layer_types', value, en), tag: en ? 'Locked' : '已锁定',
  });
  if (config.training?.mode === 'full') {
    return config.training?.train_backbone === false ? null : lock('linear_conv', en ? 'Full fine-tuning trains every parameter, convolutions included.' : '全量微调训练全部参数，包括卷积层。');
  }
  if (!presetHasConvolutions(selectedPreset(family, config))) return lock('linear', en ? 'The selected scope has no convolution layers.' : '当前训练层范围不含卷积层。');
  return null;
}

/** Apply only the dependent values required by an explicit component selection. */
export function selectTrainingComponents(next: Record<string, any>, previous: Record<string, any>) {
  if (JSON.stringify(next.training) === JSON.stringify(previous.training)) return next;
  const result = structuredClone(next);
  if (result.training?.mode === 'full') {
    result.memory = {...result.memory, base_precision:'fp32', blocks_to_swap:0};
    result.adapter = {...result.adapter, resume_weights:null};
  } else {
    result.training = {...result.training, resume_weights:null};
  }
  if (result.training.train_text_encoder) {
    result.memory = {...result.memory, offload_text_encoder:false};
    result.dataset = {...result.dataset, text_encoding:'online'};
  }
  return result;
}

export function trainingManagedReason(config: Record<string, any>, path: string, en: boolean) {
  const full = config.training?.mode === 'full';
  if (config.loop?.distributed_strategy === 'fsdp' && path === 'training.train_text_encoder' && !config.training?.train_text_encoder) return en ? 'Memory sharding currently trains the main model. Choose data parallelism to train text encoders.' : '显存分片当前训练主模型；需要训练文本编码器时请选择数据并行。';
  if (full && path === 'memory.base_precision' && config.memory?.base_precision === 'fp32') return en ? 'Full fine-tuning retains FP32 trainable weights. Mixed precision controls forward computation.' : '全量微调保留 FP32 可训练权重；前向计算精度由混合精度控制。';
  if (full && path === 'memory.blocks_to_swap' && !config.memory?.blocks_to_swap) return en ? 'Block swapping currently supports frozen adapter bases only.' : '当前层换出仅支持适配器训练中的冻结底模。';
  if (config.training?.train_text_encoder && ((path === 'dataset.text_encoding' && config.dataset?.text_encoding === 'online') || (path === 'memory.offload_text_encoder' && !config.memory?.offload_text_encoder))) return en ? 'Text encoder training requires online encoding and keeps it on the training device.' : '训练文本编码器时，需要在线编码且不能卸载。';
  return undefined;
}
