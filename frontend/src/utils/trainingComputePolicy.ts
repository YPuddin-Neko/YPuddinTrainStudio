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

export type TrainingComputePolicy = (CommonTrainingComputePolicy & {
  id: 'dtk-full-fp32-math-v1';
  mixed_precision: 'no';
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
});

export function confirmedTrainingComputePolicy(candidate: unknown, config: Record<string, any>): TrainingComputePolicy | null {
  if (!candidate || typeof candidate !== 'object') return null;
  const policy = candidate as Record<string, unknown>;
  if (policy.allow_tf32 !== false || policy.attention !== 'sdpa' || policy.sdpa_backend !== 'math') return null;
  if (config.loop?.deterministic !== true
    || config.training?.train_backbone === false || !['anima', 'sdxl', 'krea2'].includes(config.model?.family)) return null;
  const full = config.training?.mode === 'full';
  const rules = config.adapter?.rules ?? [];
  const sdxlLokr = config.training?.mode === 'adapter' && config.model.family === 'sdxl'
    && config.adapter?.algo === 'lokr' && ['auto', 'bypass'].includes(config.adapter?.mode ?? 'auto')
    && config.adapter?.dora !== true && (config.adapter?.param_dtype ?? 'fp32') === 'fp32'
    && Array.isArray(rules) && rules.every(rule => rule && typeof rule === 'object' && [null, 'lokr', 'none'].includes(rule.algo ?? null))
    && !String(config.memory?.base_precision ?? 'auto').startsWith('fp8');
  if (!full && !sdxlLokr) return null;
  if (full && policy.id === 'dtk-full-fp32-math-v1' && policy.mixed_precision === 'no') return policy as unknown as TrainingComputePolicy;
  if (policy.mixed_precision !== 'bf16' || policy.linear_forward !== 'native-bf16'
    || policy.linear_backward !== 'fp32-contractions-grad-original-dtype'
    || policy.linear_backward_implementation !== 'linear-bf16-forward-fp32-backward-v1'
    || config.training.train_text_encoder === true || config.loop.mixed_precision !== 'bf16') return null;
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
    label: 'conv_forward' in policy
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
