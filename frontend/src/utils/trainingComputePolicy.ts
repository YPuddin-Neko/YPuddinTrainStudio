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
  'dtk-krea2-text-lora-bf16-fp32-contractions-v1',
]);

const previewPolicyIds = new Set([
  ...['lora', 'lokr'].flatMap(algo => ['ddp', 'fsdp'].map(strategy =>
    `dtk-sdxl-backbone-${algo}-${strategy}-bf16-compute-preview-v2`)),
  'dtk-anima-backbone-lora-fsdp-bf16-compute-preview-v2',
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
    | `dtk-sdxl-backbone-${'lora' | 'lokr'}-${'ddp' | 'fsdp'}-bf16-compute-preview-v2`
    | 'dtk-anima-backbone-lora-fsdp-bf16-compute-preview-v2';
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
  id: 'dtk-anima-text-lora-bf16-fp32-contractions-v1' | 'dtk-sdxl-text-lora-bf16-fp32-contractions-v1' | 'dtk-krea2-text-lora-bf16-fp32-contractions-v1';
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

export type TrainingComputePolicy = (CommonTrainingComputePolicy & {
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
  const stablePreview = gpuCount >= 2 && (
    (family === 'sdxl' && [150, 225].includes(tokens))
    || (family === 'anima' && algo === 'lora' && strategy === 'fsdp')
  );
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
  if (policy.id !== `dtk-${family}-text-lora-bf16-fp32-contractions-v1`
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
  if (policy.allow_tf32 !== false || policy.attention !== 'sdpa' || policy.sdpa_backend !== 'math') return null;
  if (config.loop?.deterministic !== true || !['anima', 'sdxl', 'krea2'].includes(config.model?.family)) return null;
  if (!previewPolicyIds.has(policy.id as string) && Object.keys(previewPolicyFields).some(key => key in policy)) return null;
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
  if (policy.id === 'dtk-sdxl-long-text-bf16-conv-fp32-linear-compute-v1') {
    if (!sdxlLokr || config.training?.train_text_encoder === true
      || ![150, 225].includes(config.model?.sdxl_max_token_length)
      || policy.sdxl_max_token_length !== config.model.sdxl_max_token_length
      || config.loop?.mixed_precision !== 'bf16' || policy.mixed_precision !== 'bf16'
      || config.loop.gpu_count !== 1
      || config.loop.distributed_strategy !== 'ddp' || config.memory?.compile === true
      || (config.memory?.blocks_to_swap ?? 0) !== 0
      || !['none', 'block'].includes(config.memory?.activation_checkpointing ?? 'none')
      || policy.linear_forward !== 'bf16-rounded-operands-fp32-contraction-bf16-output'
      || policy.linear_backward !== 'fp32-contractions-grad-original-dtype'
      || policy.linear_backward_implementation !== 'linear-bf16-operands-fp32-compute-v1'
      || policy.conv_forward !== 'fp32-output-bf16' || policy.conv_implementation !== 'conv2d-fp32-output-bf16-v1'
      || ['fsdp_param_dtype', 'fsdp_reduce_dtype', 'adapter_implementation', 'text_linear_forward'].some(key => key in policy)) return null;
    return policy as unknown as TrainingComputePolicy;
  }
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
    label: backboneAdapterPolicyIds.has(policy.id) && 'adapter_algorithm' in policy && policy.adapter_algorithm === 'lora'
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
      : (english ? 'FP32 computation (mixed precision off)' : 'FP32 计算（关闭混合精度）'),
    reason,
  };
  if (path === 'memory.allow_tf32') return { value: policy.allow_tf32, label: english ? 'Disabled' : '关闭', reason };
  if (path === 'model.attention') return { value: policy.attention, label: english ? 'PyTorch SDPA (math)' : 'PyTorch SDPA（数学实现）', reason };
  return null;
}

export function trainingComputePolicyHint(policy: TrainingComputePolicy | null, english: boolean) {
  if (!policy) return undefined;
  if (backboneAdapterPolicyIds.has(policy.id)) {
    const adapterPolicy = policy as BackboneAdapterComputePolicy;
    const parallel = adapterPolicy.distributed_strategy === 'fsdp'
      ? (english ? 'FSDP shards model parameters, adapter gradients and optimizer states across GPUs, gathering parameters in BF16 and reducing gradients in FP32. ' : 'FSDP 将模型参数、适配器梯度和优化器状态分摊到多卡，按 BF16 汇集参数，以 FP32 汇总梯度。')
      : adapterPolicy.distributed_strategy === 'ddp'
      ? (english ? 'DDP keeps the complete model on each GPU and synchronizes adapter gradients. ' : 'DDP 每卡保留完整模型并汇总适配器梯度。')
      : (english ? 'This run uses one GPU. ' : '本次使用单卡。');
    const linear = adapterPolicy.linear_forward === 'native-bf16'
      ? (english ? 'Backbone linear forward uses native BF16 and backward matrix operations use FP32. ' : '主模型线性层使用原生 BF16 前向，反向矩阵运算使用 FP32。')
      : (english ? 'Backbone linear operations retain BF16 rounding and outputs, with FP32 matrix operations. ' : '主模型线性层保留 BF16 舍入和输出，矩阵运算使用 FP32。');
    const adapter = adapterPolicy.adapter_algorithm === 'lora'
      ? (english ? 'LoRA also uses BF16-rounded operands and intermediates with FP32 matrix operations. ' : 'LoRA 也保留 BF16 操作数舍入和中间结果，矩阵运算使用 FP32。')
      : (english ? 'LoKr operations remain native. ' : 'LoKr 运算保持原生实现。');
    const conv = adapterPolicy.conv_forward
      ? (english ? 'Convolution uses FP32 and returns BF16 outputs. ' : '卷积使用 FP32，输出转回 BF16。') : '';
    return parallel + (english ? 'Text encoders stay frozen. ' : '文本编码器保持冻结。') + linear + adapter + conv
      + (previewPolicyIds.has(policy.id)
        ? (english ? 'Preview backbone linear layers also use FP32 operations with BF16 rounding and outputs for consistent repeated previews. ' : '预览的主模型线性层也保留 BF16 舍入与输出，并使用 FP32 运算以提高重复预览的一致性。') : '')
      + (english
        ? 'TF32 is off and attention uses SDPA math. Memory use and runtime may increase. Resume requires the same compute policy and environment'
        : '关闭 TF32，使用 SDPA 数学实现，可能增加显存和耗时。续训须保持相同计算策略和运行环境')
      + (adapterPolicy.sdxl_max_token_length ? (english ? ' and token limit' : '及标签长度') : '')
      + (english ? '; states from previous policies cannot be mixed.' : '，不可混用旧策略的训练状态。');
  }
  if (policy.id === 'dtk-sdxl-long-text-bf16-conv-fp32-linear-compute-v1') return english
    ? 'This SDXL long-caption LoKr run retains BF16 rounding and outputs. Training linear matrix operations and convolution use FP32; TF32 is off and attention uses SDPA math. Memory use and runtime may increase. Resume requires the same token limit, compute policy and environment.'
    : '本次 SDXL 长标签 LoKr 训练保留 BF16 舍入和输出，训练中的线性层矩阵运算及卷积使用 FP32。关闭 TF32，使用 SDPA 数学实现，可能增加显存和耗时。续训须保持相同标签长度、计算策略和运行环境。';
  if (textLoraPolicyIds.has(policy.id)) {
    const textPolicy = policy as TextLoraComputePolicy;
    const parallel = textPolicy.distributed_strategy === 'ddp'
      ? (english ? 'Each GPU keeps the complete model and synchronizes adapter gradients. ' : '每卡保留完整模型并汇总适配器梯度。')
      : (english ? 'This run uses one GPU. ' : '本次使用单卡。');
    const backbone = textPolicy.id.startsWith('dtk-sdxl-')
      ? (english ? 'Backbone linear operations and convolution also use FP32. ' : '主模型线性层运算及卷积也使用 FP32。')
      : (english ? 'Backbone linear operations also use FP32. ' : '主模型线性层运算也使用 FP32。');
    return parallel + (english
      ? 'Text-encoder LoRA training keeps BF16 rounding and outputs, with FP32 matrix operations in text linear layers and LoRA projections. '
      : '文本编码器 LoRA 训练保留 BF16 舍入和输出，文本线性层与 LoRA 矩阵运算使用 FP32。')
      + backbone + (english
        ? 'TF32 is off and attention uses SDPA math. Memory use and runtime may increase. Resume requires this same policy and environment; states from the previous native BF16 path cannot be mixed.'
        : '关闭 TF32，使用 SDPA 数学实现，可能增加显存和耗时。续训须保持相同策略和环境，不能混用旧版原生 BF16 路径的训练状态。');
  }
  if (policy.id === 'dtk-anima-ddp-bf16-linear-fp32-compute-v1') return english
    ? 'This Anima LoKr run uses data parallelism: each GPU keeps the complete backbone, processes its own data and synchronizes adapter gradients. Linear inputs and parameters are rounded to BF16, then matrix operations use FP32 and return BF16 outputs. Linear backward matrix operations also use FP32. TF32 is disabled and attention uses SDPA math. Multi-GPU speed depends on communication. Resume requires the same compute policy and environment.'
    : '本次 Anima LoKr 使用多卡数据并行：每卡保留完整主模型，分别处理数据并汇总适配器梯度。线性层先按 BF16 舍入输入与参数，再以 FP32 进行矩阵运算并返回 BF16；反向矩阵运算也使用 FP32。关闭 TF32，使用 SDPA 数学实现。多卡速度取决于跨卡通信；续训需保持相同计算策略和运行环境。';
  if (policy.id === 'dtk-anima-fsdp-bf16-linear-fp32-compute-v1') return english
    ? 'This Anima full-model run shards parameters, gradients and optimizer states across GPUs. Parameters are gathered in BF16 and gradients are reduced in FP32. Linear inputs and parameters are rounded to BF16, then matrix operations use FP32 and return BF16 outputs. Linear backward matrix operations also use FP32. TF32 is disabled and attention uses SDPA math. Multi-GPU speed depends on communication. Resume requires the same compute policy and environment.'
    : '本次 Anima 全量微调使用多卡显存分片，分担参数、梯度和优化器状态。按 BF16 汇集参数，以 FP32 汇总梯度。线性层先按 BF16 舍入输入与参数，再以 FP32 进行矩阵运算并返回 BF16；反向矩阵运算也使用 FP32。关闭 TF32，使用 SDPA 数学实现。多卡速度取决于跨卡通信；续训需保持相同计算策略和运行环境。';
  if (policy.id === 'dtk-anima-bf16-linear-fp32-compute-v1') return english
    ? 'This Anima run uses one GPU. Linear inputs and parameters are rounded to BF16, then matrix operations use FP32 and return BF16 outputs. Linear backward matrix operations also use FP32. TF32 is disabled and attention uses SDPA math. Resume requires the same compute policy and environment.'
    : '本次 Anima 使用单卡。线性层先按 BF16 舍入输入与参数，再以 FP32 进行矩阵运算并返回 BF16；反向矩阵运算也使用 FP32。关闭 TF32，使用 SDPA 数学实现。续训需保持相同计算策略和运行环境。';
  if (policy.id === 'dtk-sdxl-fsdp-bf16-conv-fp32-linear-backward-v1') return english
    ? 'This SDXL run shards the model across GPUs and uses BF16 for gathered parameters and main computation. Convolution, linear backward matrix operations and gradient reduction use FP32. Convolution outputs return to BF16. TF32 is disabled and attention uses native SDPA math. Resume requires the same compute policy and environment.'
    : '本次 SDXL 使用多卡显存分片，按 BF16 汇集参数并进行主体计算。卷积、线性层反向矩阵运算和梯度汇总使用 FP32，卷积输出转回 BF16。关闭 TF32，并使用原生 SDPA 数学实现。续训需保持相同计算策略和运行环境。';
  if (policy.id === 'dtk-krea2-fsdp-bf16-linear-fp32-backward-v1') return english
    ? 'This Krea2 run shards the model across GPUs and retains BF16 forward computation. Linear backward matrix operations and gradient reduction use FP32, with TF32 disabled and native SDPA math. Resume requires the same compute policy and environment.'
    : '本次 Krea2 使用多卡显存分片，保留 BF16 前向计算。线性层反向矩阵运算和梯度汇总使用 FP32，关闭 TF32，并使用原生 SDPA 数学实现。续训需保持相同计算策略和运行环境。';
  if (policy.id === 'dtk-sdxl-bf16-conv-fp32-linear-backward-v1') return english
    ? 'This SDXL run retains BF16 computation, with FP32 convolution and linear backward matrix operations. Convolution outputs return to BF16. TF32 is disabled and attention uses native SDPA math. Resume requires the same compute policy and environment.'
    : '本次 SDXL 保留 BF16 主体计算，卷积和线性层反向矩阵运算使用 FP32，卷积输出转回 BF16。关闭 TF32，并使用原生 SDPA 数学实现。续训需保持相同计算策略和运行环境。';
  return english
    ? 'This DTK full-backbone run uses FP32 computation, disables TF32 and uses native SDPA math. The three settings below are managed by this switch. Activation memory and training time may increase; bitwise equality is not guaranteed across environments.'
    : '本次 DTK 主模型全量微调使用 FP32 计算、关闭 TF32，并使用原生 SDPA 数学实现。这三项设置由此开关管理，可能增加激活显存和训练耗时，不保证不同环境逐位一致。';
}
