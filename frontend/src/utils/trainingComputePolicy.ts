import {confirmedExtraAdapterPolicy, extraAdapterPolicyIds, extraAdapterPrecisionLabel, extraAdapterPolicyHint, type ExtraAdapterComputePolicy} from './dtkAdapterComputePolicy';

/** A server-confirmed effective policy; never infer the host runtime in the editor. */
interface CommonTrainingComputePolicy {
  allow_tf32: false;
  attention: 'sdpa';
  sdpa_backend: 'math';
}

interface BF16TrainingComputePolicy extends CommonTrainingComputePolicy {
  mixed_precision: 'bf16';
  linear_forward: 'native-bf16';
  linear_backward: 'fp32-contractions-grad-original-dtype';
  linear_backward_implementation: 'linear-bf16-forward-fp32-backward-v1';
}

interface SDXLTrainingComputePolicy extends BF16TrainingComputePolicy {
  conv_forward: 'fp32-output-bf16';
  conv_implementation: 'conv2d-fp32-output-bf16-v1';
}

interface AnimaTrainingComputePolicy extends CommonTrainingComputePolicy {
  mixed_precision: 'bf16';
  linear_forward: 'bf16-rounded-operands-fp32-contraction-bf16-output';
  linear_backward: 'fp32-contractions-grad-original-dtype';
  linear_backward_implementation: 'linear-bf16-operands-fp32-compute-v1';
}

const animaPolicyIds = new Set([
  'dtk-anima-bf16-linear-fp32-compute-v1',
  'dtk-anima-ddp-bf16-linear-fp32-compute-v1',
  'dtk-anima-fsdp-bf16-linear-fp32-compute-v1',
]);

const textLoraPolicyIds = new Set([
  'dtk-anima-text-lora-bf16-fp32-contractions-v1',
  'dtk-sdxl-text-lora-bf16-fp32-contractions-v1',
  'dtk-sdxl-text-lora-single-bf16-compute-preview-v2',
  'dtk-sdxl-text-lora-ddp-bf16-compute-preview-v2',
  'dtk-krea2-text-lora-bf16-fp32-contractions-v1',
]);

const textPreviewPolicyIds = new Set(['dtk-sdxl-text-lora-single-bf16-compute-preview-v2', 'dtk-sdxl-text-lora-ddp-bf16-compute-preview-v2']);
const previewPolicyIds = new Set([
  ...['lora', 'lokr'].flatMap(algo => ['single', 'ddp', 'fsdp'].map(strategy =>
    `dtk-sdxl-backbone-${algo}-${strategy}-bf16-compute-preview-v2`)),
  'dtk-anima-backbone-lora-fsdp-bf16-compute-preview-v2',
  'dtk-anima-backbone-lokr-fsdp-bf16-compute-preview-v2',
]);
const previewPolicyFields = {
  preview_operator_components: ['backbone'],
  preview_linear_forward: 'bf16-rounded-operands-fp32-contraction-bf16-output',
  preview_linear_implementation: 'linear-native-dispatch-bf16-operands-fp32-preview-v1',
} as const;

const backboneAdapterPolicyIds = new Set(
  [...['anima', 'sdxl'].flatMap(family => [
    ...['single', 'ddp', 'fsdp'].map(strategy => `dtk-${family}-backbone-lora-${strategy}-bf16-compute-v1`),
    `dtk-${family}-backbone-lokr-fsdp-bf16-compute-v1`,
  ]), ...previewPolicyIds],
);

interface BackboneAdapterComputePolicy extends CommonTrainingComputePolicy {
  id: `dtk-${'anima' | 'sdxl'}-backbone-${`lora-${'single' | 'ddp' | 'fsdp'}` | 'lokr-fsdp'}-bf16-compute-v1`
    | `dtk-sdxl-backbone-${'lora' | 'lokr'}-${'single' | 'ddp' | 'fsdp'}-bf16-compute-preview-v2`
    | `dtk-anima-backbone-${'lora' | 'lokr'}-fsdp-bf16-compute-preview-v2`;
  mixed_precision: 'bf16';
  linear_forward: 'native-bf16' | 'bf16-rounded-operands-fp32-contraction-bf16-output';
  linear_backward: 'fp32-contractions-grad-original-dtype';
  linear_backward_implementation: 'linear-bf16-forward-fp32-backward-v1' | 'linear-bf16-operands-fp32-compute-v1';
  adapter_algorithm: 'lora' | 'lokr';
  adapter_implementation?: 'lora-bf16-operands-fp32-contractions-v1';
  adapter_forward?: 'bf16-rounded-operands-fp32-contractions-bf16-intermediates';
  adapter_backward?: 'fp32-contractions-grad-original-dtype';
  trainable_components: ['backbone'];
  operator_components: ['backbone'];
  distributed_strategy: 'single' | 'ddp' | 'fsdp';
  fsdp_param_dtype?: 'bfloat16';
  fsdp_reduce_dtype?: 'float32';
  conv_forward?: 'fp32-output-bf16';
  conv_implementation?: 'conv2d-fp32-output-bf16-v1';
  sdxl_max_token_length?: 75 | 150 | 225;
  preview_operator_components?: readonly ['backbone'];
  preview_linear_forward?: typeof previewPolicyFields.preview_linear_forward;
  preview_linear_implementation?: typeof previewPolicyFields.preview_linear_implementation;
}

interface TextLoraComputePolicy extends AnimaTrainingComputePolicy {
  id: 'dtk-anima-text-lora-bf16-fp32-contractions-v1' | 'dtk-sdxl-text-lora-bf16-fp32-contractions-v1' | 'dtk-krea2-text-lora-bf16-fp32-contractions-v1' | `dtk-sdxl-text-lora-${'single' | 'ddp'}-bf16-compute-preview-v2`;
  mixed_precision: 'bf16';
  text_linear_forward: 'bf16-rounded-operands-fp32-contraction-bf16-output';
  text_linear_backward_implementation: 'linear-bf16-operands-fp32-compute-v1';
  adapter_implementation: 'lora-bf16-operands-fp32-contractions-v1';
  adapter_forward: 'bf16-rounded-operands-fp32-contractions-bf16-intermediates';
  adapter_backward: 'fp32-contractions-grad-original-dtype';
  trainable_components: string[];
  operator_components: string[];
  distributed_strategy: 'single' | 'ddp';
  conv_forward?: 'fp32-output-bf16';
  conv_implementation?: 'conv2d-fp32-output-bf16-v1';
  sdxl_max_token_length?: 75 | 150 | 225;
}

export type TrainingComputePolicy = ExtraAdapterComputePolicy | (CommonTrainingComputePolicy & {
  id: 'dtk-full-fp32-math-v1';
  mixed_precision: 'no';
}) | (AnimaTrainingComputePolicy & {
  id: 'dtk-anima-bf16-linear-fp32-compute-v1' | 'dtk-anima-ddp-bf16-linear-fp32-compute-v1';
}) | (AnimaTrainingComputePolicy & {
  id: 'dtk-anima-fsdp-bf16-linear-fp32-compute-v1';
  fsdp_param_dtype: 'bfloat16';
  fsdp_reduce_dtype: 'float32';
}) | (BF16TrainingComputePolicy & {
  id: 'dtk-krea2-fsdp-bf16-linear-fp32-backward-v1';
  fsdp_param_dtype: 'bfloat16';
  fsdp_reduce_dtype: 'float32';
}) | (SDXLTrainingComputePolicy & {
  id: 'dtk-sdxl-bf16-conv-fp32-linear-backward-v1';
}) | (SDXLTrainingComputePolicy & {
  id: 'dtk-sdxl-fsdp-bf16-conv-fp32-linear-backward-v1';
  fsdp_param_dtype: 'bfloat16';
  fsdp_reduce_dtype: 'float32';
}) | TextLoraComputePolicy | BackboneAdapterComputePolicy | (AnimaTrainingComputePolicy & {
  id: 'dtk-sdxl-long-text-bf16-conv-fp32-linear-compute-v1';
  conv_forward: 'fp32-output-bf16';
  conv_implementation: 'conv2d-fp32-output-bf16-v1';
  sdxl_max_token_length: 150 | 225;
});

function confirmedBackboneAdapterPolicy(policy: Record<string, unknown>, config: Record<string, any>): BackboneAdapterComputePolicy | null {
  const family = config.model?.family;
  const algo = config.adapter?.algo;
  const gpuCount = config.loop?.gpu_count;
  const strategy = gpuCount > 1 ? config.loop?.distributed_strategy : 'single';
  const rules = config.adapter?.rules ?? [];
  const tokens = config.model?.sdxl_max_token_length ?? 75;
  const stablePreview = (family === 'sdxl' && [150, 225].includes(tokens))
    || (family === 'anima' && gpuCount >= 2 && strategy === 'fsdp');
  if (!['anima', 'sdxl'].includes(family) || !['lora', 'lokr'].includes(algo)
    || !Number.isInteger(gpuCount) || gpuCount < 1
    || !['ddp', 'fsdp'].includes(config.loop?.distributed_strategy)
    || (algo === 'lokr' && strategy !== 'fsdp' && !stablePreview)
    || config.training?.mode !== 'adapter' || config.training?.train_backbone !== true
    || config.training?.train_text_encoder === true || config.loop?.mixed_precision !== 'bf16'
    || !['auto', 'bypass'].includes(config.adapter?.mode ?? 'auto')
    || config.adapter?.dora === true || (config.adapter?.param_dtype ?? 'fp32') !== 'fp32'
    || !Array.isArray(rules) || !rules.every(rule => rule && typeof rule === 'object' && [null, algo, 'none'].includes(rule.algo ?? null))
    || String(config.memory?.base_precision ?? 'auto').startsWith('fp8')
    || config.memory?.compile === true || (config.memory?.blocks_to_swap ?? 0) !== 0
    || !['none', 'block'].includes(config.memory?.activation_checkpointing ?? 'none')) return null;
  if (family === 'sdxl' && ![75, 150, 225].includes(tokens)) return null;
  const fp32Forward = algo === 'lora' || family === 'anima' || tokens > 75;
  const expected: Record<string, unknown> = {
    id: `dtk-${family}-backbone-${algo}-${strategy}-bf16-compute-${stablePreview ? 'preview-v2' : 'v1'}`,
    mixed_precision: 'bf16', allow_tf32: false, attention: 'sdpa', sdpa_backend: 'math',
    linear_forward: fp32Forward ? 'bf16-rounded-operands-fp32-contraction-bf16-output' : 'native-bf16',
    linear_backward: 'fp32-contractions-grad-original-dtype',
    linear_backward_implementation: fp32Forward ? 'linear-bf16-operands-fp32-compute-v1' : 'linear-bf16-forward-fp32-backward-v1',
    adapter_algorithm: algo, trainable_components: ['backbone'], operator_components: ['backbone'], distributed_strategy: strategy,
    ...(algo === 'lora' ? {
      adapter_implementation: 'lora-bf16-operands-fp32-contractions-v1',
      adapter_forward: 'bf16-rounded-operands-fp32-contractions-bf16-intermediates',
      adapter_backward: 'fp32-contractions-grad-original-dtype',
    } : {}),
    ...(strategy === 'fsdp' ? {fsdp_param_dtype: 'bfloat16', fsdp_reduce_dtype: 'float32'} : {}),
    ...(family === 'sdxl' ? {conv_forward: 'fp32-output-bf16', conv_implementation: 'conv2d-fp32-output-bf16-v1', sdxl_max_token_length: tokens} : {}),
    ...(stablePreview ? previewPolicyFields : {}),
  };
  // Match the whole versioned contract, including the absence of text/LoKr overrides.
  if (Object.keys(policy).length !== Object.keys(expected).length
    || !Object.entries(expected).every(([key, value]) => Array.isArray(value)
      ? Array.isArray(policy[key]) && policy[key].length === value.length && policy[key].every((item, index) => item === value[index])
      : policy[key] === value)) return null;
  return policy as unknown as BackboneAdapterComputePolicy;
}

function confirmedTextLoraPolicy(policy: Record<string, unknown>, config: Record<string, any>): TrainingComputePolicy | null {
  const family = config.model?.family;
  const rules = config.adapter?.rules ?? [];
  const gpuCount = config.loop?.gpu_count;
  const stablePreview = family === 'sdxl' && config.model?.sdxl_max_token_length === 150
    && config.training?.train_backbone === true && [1,2].includes(gpuCount);
  const expectedId = stablePreview ? `dtk-sdxl-text-lora-${gpuCount === 1 ? 'single' : 'ddp'}-bf16-compute-preview-v2`
    : `dtk-${family}-text-lora-bf16-fp32-contractions-v1`;
  if (stablePreview && !Object.entries(previewPolicyFields).every(([key, value]) => Array.isArray(value)
    ? Array.isArray(policy[key]) && policy[key].length === value.length && policy[key].every((item, index) => item === value[index])
    : policy[key] === value)) return null;
  if (Object.keys(policy).some(key => key.startsWith('preview_') && !(stablePreview && key in previewPolicyFields))) return null;
  if (policy.id !== expectedId
    || config.training?.mode !== 'adapter' || config.training?.train_text_encoder !== true
    || config.loop?.mixed_precision !== 'bf16' || policy.mixed_precision !== 'bf16'
    || !Number.isInteger(gpuCount) || gpuCount < 1 || config.loop?.distributed_strategy !== 'ddp'
    || config.adapter?.algo !== 'lora' || !['auto', 'bypass'].includes(config.adapter?.mode ?? 'auto')
    || config.adapter?.dora === true || (config.adapter?.param_dtype ?? 'fp32') !== 'fp32'
    || !Array.isArray(rules) || !rules.every(rule => rule && typeof rule === 'object' && [null, 'lora', 'none'].includes(rule.algo ?? null))
    || String(config.memory?.base_precision ?? 'auto').startsWith('fp8')
    || !['auto', 'online'].includes(config.dataset?.text_encoding ?? 'auto')
    || config.memory?.offload_text_encoder !== false || (config.memory?.blocks_to_swap ?? 0) !== 0
    || config.memory?.compile === true
    || !['none', 'block'].includes(config.memory?.activation_checkpointing ?? 'none')) return null;
  const textComponents = family === 'sdxl' ? ['text_encoder', 'text_encoder_2'] : ['text_encoder'];
  const sameComponents = (actual: unknown, expected: string[]) => Array.isArray(actual)
    && actual.length === expected.length && actual.every((value, index) => value === expected[index]);
  if (!sameComponents(policy.operator_components, ['backbone', ...textComponents])
    || !sameComponents(policy.trainable_components, config.training.train_backbone === false ? textComponents : ['backbone', ...textComponents])
    || policy.distributed_strategy !== (gpuCount === 1 ? 'single' : 'ddp')
    || policy.linear_forward !== 'bf16-rounded-operands-fp32-contraction-bf16-output'
    || policy.linear_backward !== 'fp32-contractions-grad-original-dtype'
    || policy.linear_backward_implementation !== 'linear-bf16-operands-fp32-compute-v1'
    || policy.text_linear_forward !== 'bf16-rounded-operands-fp32-contraction-bf16-output'
    || policy.text_linear_backward_implementation !== 'linear-bf16-operands-fp32-compute-v1'
    || policy.adapter_implementation !== 'lora-bf16-operands-fp32-contractions-v1'
    || policy.adapter_forward !== 'bf16-rounded-operands-fp32-contractions-bf16-intermediates'
    || policy.adapter_backward !== 'fp32-contractions-grad-original-dtype'
    || ['fsdp_param_dtype', 'fsdp_reduce_dtype'].some(key => key in policy)) return null;
  if (family === 'sdxl') {
    const tokens = config.model?.sdxl_max_token_length ?? 75;
    if (![75, 150, 225].includes(tokens) || policy.sdxl_max_token_length !== tokens
      || policy.conv_forward !== 'fp32-output-bf16' || policy.conv_implementation !== 'conv2d-fp32-output-bf16-v1') return null;
  } else if (['conv_forward', 'conv_implementation', 'sdxl_max_token_length'].some(key => key in policy)) return null;
  return policy as unknown as TrainingComputePolicy;
}

export function confirmedTrainingComputePolicy(candidate: unknown, config: Record<string, any>): TrainingComputePolicy | null {
  if (!candidate || typeof candidate !== 'object') return null;
  const policy = candidate as Record<string, unknown>;
  if (extraAdapterPolicyIds.has(policy.id as string)) return confirmedExtraAdapterPolicy(policy, config);
  if (policy.allow_tf32 !== false || policy.attention !== 'sdpa' || policy.sdpa_backend !== 'math') return null;
  if (config.loop?.deterministic !== true || !['anima', 'sdxl', 'krea2'].includes(config.model?.family)) return null;
  if (!previewPolicyIds.has(policy.id as string) && !textPreviewPolicyIds.has(policy.id as string) && Object.keys(previewPolicyFields).some(key => key in policy)) return null;
  if (backboneAdapterPolicyIds.has(policy.id as string)) return confirmedBackboneAdapterPolicy(policy, config);
  if (textLoraPolicyIds.has(policy.id as string)) return confirmedTextLoraPolicy(policy, config);
  if (config.training?.train_backbone === false) return null;
  const full = config.training?.mode === 'full';
  const rules = config.adapter?.rules ?? [];
  const bypassLokr = config.training?.mode === 'adapter'
    && config.adapter?.algo === 'lokr' && ['auto', 'bypass'].includes(config.adapter?.mode ?? 'auto')
    && config.adapter?.dora !== true && (config.adapter?.param_dtype ?? 'fp32') === 'fp32'
    && Array.isArray(rules) && rules.every(rule => rule && typeof rule === 'object' && [null, 'lokr', 'none'].includes(rule.algo ?? null))
    && !String(config.memory?.base_precision ?? 'auto').startsWith('fp8');
  const sdxlLokr = bypassLokr && config.model.family === 'sdxl';
  const animaLokr = bypassLokr && config.model.family === 'anima';
  if (!full && !sdxlLokr && !animaLokr) return null;
  if (full && policy.id === 'dtk-full-fp32-math-v1' && policy.mixed_precision === 'no') return policy as unknown as TrainingComputePolicy;
  if (policy.id === 'dtk-sdxl-long-text-bf16-conv-fp32-linear-compute-v1') return null;
  if (animaPolicyIds.has(policy.id as string)) {
    if (config.model.family !== 'anima'
      || config.training.train_backbone !== true || config.training.train_text_encoder === true
      || config.loop.mixed_precision !== 'bf16'
      || !['none', 'block'].includes(config.memory?.activation_checkpointing ?? 'none')
      || policy.mixed_precision !== 'bf16'
      || policy.linear_forward !== 'bf16-rounded-operands-fp32-contraction-bf16-output'
      || policy.linear_backward !== 'fp32-contractions-grad-original-dtype'
      || policy.linear_backward_implementation !== 'linear-bf16-operands-fp32-compute-v1'
      || ['conv_forward', 'conv_implementation'].some(key => key in policy)) return null;
    const multipleGpus = Number.isInteger(config.loop.gpu_count) && config.loop.gpu_count >= 2;
    if (policy.id === 'dtk-anima-fsdp-bf16-linear-fp32-compute-v1') {
      if (!full || !multipleGpus || config.loop.distributed_strategy !== 'fsdp'
        || policy.fsdp_param_dtype !== 'bfloat16' || policy.fsdp_reduce_dtype !== 'float32') return null;
    } else {
      if (['fsdp_param_dtype', 'fsdp_reduce_dtype'].some(key => key in policy)) return null;
      if (policy.id === 'dtk-anima-ddp-bf16-linear-fp32-compute-v1') {
        if (!animaLokr || !multipleGpus || config.loop.distributed_strategy !== 'ddp') return null;
      } else if (config.loop.gpu_count !== 1) return null;
    }
    return policy as unknown as TrainingComputePolicy;
  }
  if (policy.mixed_precision !== 'bf16' || policy.linear_forward !== 'native-bf16'
    || policy.linear_backward !== 'fp32-contractions-grad-original-dtype'
    || policy.linear_backward_implementation !== 'linear-bf16-forward-fp32-backward-v1'
    || config.training.train_text_encoder === true || config.loop.mixed_precision !== 'bf16') return null;
  if (sdxlLokr && (config.model?.sdxl_max_token_length ?? 75) !== 75) return null;
  if (policy.id === 'dtk-krea2-fsdp-bf16-linear-fp32-backward-v1'
    && policy.fsdp_param_dtype === 'bfloat16' && policy.fsdp_reduce_dtype === 'float32'
    && full && config.model.family === 'krea2' && config.loop.distributed_strategy === 'fsdp'
    && Number.isInteger(config.loop.gpu_count) && config.loop.gpu_count >= 2) return policy as unknown as TrainingComputePolicy;
  if (policy.id === 'dtk-sdxl-bf16-conv-fp32-linear-backward-v1'
    && policy.conv_forward === 'fp32-output-bf16' && policy.conv_implementation === 'conv2d-fp32-output-bf16-v1'
    && config.model.family === 'sdxl' && (config.loop.gpu_count === 1
      || (sdxlLokr && config.loop.distributed_strategy === 'ddp'
        && Number.isInteger(config.loop.gpu_count) && config.loop.gpu_count >= 2))) return policy as unknown as TrainingComputePolicy;
  if (policy.id === 'dtk-sdxl-fsdp-bf16-conv-fp32-linear-backward-v1'
    && policy.conv_forward === 'fp32-output-bf16' && policy.conv_implementation === 'conv2d-fp32-output-bf16-v1'
    && policy.fsdp_param_dtype === 'bfloat16' && policy.fsdp_reduce_dtype === 'float32'
    && full && config.model.family === 'sdxl' && config.loop.distributed_strategy === 'fsdp'
    && Number.isInteger(config.loop.gpu_count) && config.loop.gpu_count >= 2) return policy as unknown as TrainingComputePolicy;
  return null;
}

export function currentTrainingComputePolicy(candidate: unknown, config: Record<string, any>, validatedConfig: string, validating: boolean) {
  if (validating || validatedConfig !== JSON.stringify(config)) return null;
  return confirmedTrainingComputePolicy(candidate, config);
}

export function trainingComputeManagedField(policy: TrainingComputePolicy | null, path: string, english: boolean) {
  if (!policy) return null;
  const reason = english
    ? 'Managed by reproducible training on DTK. Your original choice is retained and used again when the switch is off.'
    : '由 DTK 可复现训练管理。保留原选择，关闭开关后恢复使用。';
  if (path === 'loop.mixed_precision') return {
    value: policy.mixed_precision,
    label: extraAdapterPrecisionLabel(policy.id, english) ?? (backboneAdapterPolicyIds.has(policy.id) && 'adapter_algorithm' in policy && policy.adapter_algorithm === 'lora'
      ? (english ? 'BF16 (FP32 linear and LoRA operations)' : 'BF16（FP32 线性层与 LoRA 运算）')
      : backboneAdapterPolicyIds.has(policy.id) && 'linear_forward' in policy && policy.linear_forward !== 'native-bf16'
      ? ('conv_forward' in policy
        ? (english ? 'BF16 (FP32 convolution and linear operations)' : 'BF16（FP32 卷积、线性层运算）')
        : (english ? 'BF16 (FP32 linear operations)' : 'BF16（线性层 FP32 运算）'))
      : textLoraPolicyIds.has(policy.id)
      ? (english ? 'BF16 (FP32 operations for reproducibility)' : 'BF16（使用 FP32 运算提高可复现性）')
      : policy.id === 'dtk-sdxl-long-text-bf16-conv-fp32-linear-compute-v1'
      ? (english ? 'BF16 (FP32 convolution and linear operations)' : 'BF16（FP32 卷积、线性层运算）')
      : animaPolicyIds.has(policy.id)
      ? (english ? 'BF16 (FP32 linear operations)' : 'BF16（线性层 FP32 运算）')
      : 'conv_forward' in policy
      ? (english ? 'BF16 (FP32 convolution and linear backward)' : 'BF16（FP32 卷积、线性层反向）')
      : policy.mixed_precision === 'bf16'
      ? (english ? 'BF16 forward (FP32 linear backward)' : 'BF16 前向（线性层反向 FP32）')
      : (english ? 'FP32 computation (mixed precision off)' : 'FP32 计算（关闭混合精度）')),
    reason,
  };
  if (path === 'memory.allow_tf32') return { value: policy.allow_tf32, label: english ? 'Disabled' : '关闭', reason };
  if (path === 'model.attention') return { value: policy.attention, label: policy.attention === 'flash_attn' ? 'FlashAttention 2 (DTK)' : english ? 'PyTorch SDPA (math)' : 'PyTorch SDPA（数学实现）', reason };
  return null;
}

export function trainingComputePolicyHint(policy: TrainingComputePolicy | null, english: boolean) {
  if (!policy) return undefined;
  if (extraAdapterPolicyIds.has(policy.id)) return extraAdapterPolicyHint(policy.id, english);
  return english
    ? 'Precision and attention settings are managed for reproducibility. Memory use and training time may increase. Resume with the same settings and environment.'
    : '已自动设置精度和注意力，可能增加显存与训练耗时。续训须保持相同设置和环境。';
}
