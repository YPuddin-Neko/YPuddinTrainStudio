/** A server-confirmed effective policy; never infer the host runtime in the editor. */
export interface TrainingComputePolicy {
  id: 'dtk-full-fp32-math-v1';
  mixed_precision: 'no';
  allow_tf32: false;
  attention: 'sdpa';
  sdpa_backend: 'math';
}

export function confirmedTrainingComputePolicy(candidate: unknown, config: Record<string, any>): TrainingComputePolicy | null {
  if (!candidate || typeof candidate !== 'object') return null;
  const policy = candidate as Record<string, unknown>;
  if (policy.id !== 'dtk-full-fp32-math-v1' || policy.mixed_precision !== 'no'
    || policy.allow_tf32 !== false || policy.attention !== 'sdpa' || policy.sdpa_backend !== 'math') return null;
  if (config.loop?.deterministic !== true || config.training?.mode !== 'full'
    || config.training?.train_backbone === false || !['anima', 'sdxl', 'krea2'].includes(config.model?.family)) return null;
  return policy as unknown as TrainingComputePolicy;
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
  if (path === 'loop.mixed_precision') return { value: policy.mixed_precision, label: english ? 'FP32 computation (mixed precision off)' : 'FP32 计算（关闭混合精度）', reason };
  if (path === 'memory.allow_tf32') return { value: policy.allow_tf32, label: english ? 'Disabled' : '关闭', reason };
  if (path === 'model.attention') return { value: policy.attention, label: english ? 'PyTorch SDPA (math)' : 'PyTorch SDPA（数学实现）', reason };
  return null;
}

export function trainingComputePolicyHint(policy: TrainingComputePolicy | null, english: boolean) {
  if (!policy) return undefined;
  return english
    ? 'This DTK full-backbone run uses FP32 computation, disables TF32 and uses native SDPA math. The three settings below are managed by this switch. Activation memory and training time may increase; bitwise equality is not guaranteed across environments.'
    : '本次 DTK 主模型全量微调使用 FP32 计算、关闭 TF32，并使用原生 SDPA 数学实现。这三项设置由此开关管理，可能增加激活显存和训练耗时，不保证不同环境逐位一致。';
}
