/** Exact contracts for the measured FP16 and Klein 4B/9B paths. Host confirmation is required. */
export interface ExtraAdapterComputePolicy {
  id: 'dtk-anima-backbone-lora-single-fp16-compute-v1'
    | 'dtk-sdxl-backbone-lokr-single-fp16-preview-v1'
    | `dtk-klein${'4b' | '9b'}-backbone-${'lora' | 'lokr'}-${'ddp' | 'fsdp'}-bf16-compute-preview-v1`
    | 'dtk-klein4b-backbone-lokr-fsdp-bf16-compute-preview-v2'
    | `dtk-klein9b-backbone-${'lora' | 'lokr'}-${'ddp' | 'fsdp'}-bf16-compute-preview-v4`;
  mixed_precision: 'bf16' | 'fp16';
  allow_tf32: false;
  attention: 'sdpa';
  sdpa_backend: 'math';
}
const animaFp16 = 'dtk-anima-backbone-lora-single-fp16-compute-v1';
const sdxlFp16 = 'dtk-sdxl-backbone-lokr-single-fp16-preview-v1';
const kleinVersion = (size: string, algo: string, strategy: string) =>
  size === '9b' ? 4 : algo === 'lokr' && strategy === 'fsdp' ? 2 : 1;
const kleinIds = ['4b','9b'].flatMap(size => ['lora','lokr'].flatMap(algo => ['ddp','fsdp'].map(strategy => `dtk-klein${size}-backbone-${algo}-${strategy}-bf16-compute-preview-v${kleinVersion(size,algo,strategy)}`)));
export const extraAdapterPolicyIds = new Set<string>([animaFp16, sdxlFp16, ...kleinIds]);

export function sameComputeContract(policy: Record<string, unknown>, expected: Record<string, unknown>) {
  return Object.keys(policy).length === Object.keys(expected).length
    && Object.entries(expected).every(([key, value]) => Array.isArray(value)
      ? Array.isArray(policy[key]) && policy[key].length === value.length && policy[key].every((item, index) => item === value[index])
      : policy[key] === value);
}

export function confirmedExtraAdapterPolicy(policy: Record<string, unknown>, config: Record<string, any>): ExtraAdapterComputePolicy | null {
  const family = config.model?.family;
  const algo = config.adapter?.algo;
  const rules = config.adapter?.rules ?? [];
  const gpus = config.loop?.gpu_count;
  const strategy = config.loop?.distributed_strategy;
  const klein = family === 'flux2' && ['klein-base-4b','klein-base-9b'].includes(config.model?.flux2_variant);
  const kleinSize = config.model?.flux2_variant === 'klein-base-9b' ? '9b' : '4b';
  const anima = family === 'anima' && algo === 'lora';
  const sdxl = family === 'sdxl' && algo === 'lokr' && config.model?.sdxl_max_token_length === 150;
  if ((!klein && !anima && !sdxl) || config.loop?.deterministic !== true
    || config.model?.dtype !== 'bf16' || !['auto','bf16'].includes(config.memory?.base_precision ?? 'auto')
    || config.training?.mode !== 'adapter' || config.training?.train_backbone !== true
    || config.training?.train_text_encoder === true || !['lora','lokr'].includes(algo)
    || !['auto','bypass'].includes(config.adapter?.mode ?? 'auto') || config.adapter?.dora === true
    || (config.adapter?.param_dtype ?? 'fp32') !== 'fp32' || !Array.isArray(rules)
    || !rules.every(rule => rule && typeof rule === 'object' && [null,algo,'none'].includes(rule.algo ?? null))
    || !['none','block'].includes(config.memory?.activation_checkpointing ?? 'none')
    || config.memory?.compile === true || (config.memory?.blocks_to_swap ?? 0) !== 0) return null;
  if (klein ? (gpus !== 2 || !['ddp','fsdp'].includes(strategy) || config.loop?.mixed_precision !== 'bf16')
    : (gpus !== 1 || strategy !== 'ddp' || config.loop?.mixed_precision !== 'fp16')) return null;
  if (sdxl && config.memory?.base_precision !== 'bf16') return null;
  const dtype = klein ? 'bf16' : 'fp16';
  const expected: Record<string, unknown> = {
    id: klein ? `dtk-klein${kleinSize}-backbone-${algo}-${strategy}-bf16-compute-preview-v${kleinVersion(kleinSize,algo,strategy)}` : anima ? animaFp16 : sdxlFp16,
    mixed_precision: dtype, allow_tf32:false, attention:'sdpa', sdpa_backend:'math',
  };
  if (klein || sdxl) Object.assign(expected, {
    preview_operator_components:['backbone'],
    preview_linear_forward:`${dtype}-rounded-operands-fp32-contraction-${dtype}-output`,
    preview_linear_implementation:`linear-native-dispatch-${dtype}-operands-fp32-preview-v1`,
  });
  if (klein || anima) Object.assign(expected, {adapter_algorithm:algo, distributed_strategy:klein ? strategy : 'single'});
  if (algo === 'lora' || klein && (strategy === 'fsdp' || kleinSize === '9b')) Object.assign(expected, {
    linear_forward:`${dtype}-rounded-operands-fp32-contraction-${dtype}-output`,
    linear_backward:'fp32-contractions-grad-original-dtype',
    linear_backward_implementation:anima ? 'linear-lora-fp16-operands-fp32-compute-v1' : 'linear-bf16-operands-fp32-compute-v1',
    ...(algo === 'lora' ? {
    adapter_forward:`${dtype}-rounded-operands-fp32-contractions-${dtype}-intermediates`,
    adapter_backward:'fp32-contractions-grad-original-dtype',
    adapter_implementation:anima ? 'linear-lora-fp16-operands-fp32-compute-v1' : `${algo}-bf16-operands-fp32-contractions-v1`,
    } : {}),
    operator_components:['backbone'], trainable_components:['backbone'],
    ...(klein && strategy === 'fsdp' ? {fsdp_param_dtype:'bfloat16',fsdp_reduce_dtype:'float32'} : {}),
  });
  if (klein && kleinSize === '9b') expected.frozen_text_implementation = 'qwen3-bf16-linear-fp32-v1';
  return sameComputeContract(policy, expected) ? policy as unknown as ExtraAdapterComputePolicy : null;
}

export function extraAdapterPrecisionLabel(id: string, english: boolean) {
  if (id === animaFp16) return english ? 'FP16 (FP32 linear and LoRA operations)' : 'FP16（FP32 线性层与 LoRA 运算）';
  if (id === sdxlFp16) return english ? 'FP16 (FP32 preview linear operations)' : 'FP16（预览线性层 FP32 运算）';
  if (kleinIds.includes(id)) return id.includes('-lora-')
    ? (english ? 'BF16 (FP32 linear and LoRA operations)' : 'BF16（FP32 线性层与 LoRA 运算）')
    : id.includes('-fsdp-') || id.startsWith('dtk-klein9b-') ? (english ? 'BF16 (FP32 linear operations)' : 'BF16（线性层 FP32 运算）')
    : (english ? 'BF16 (FP32 preview linear operations)' : 'BF16（预览线性层 FP32 运算）');
  return null;
}

export function extraAdapterPolicyHint(id: string, english: boolean) {
  if (!extraAdapterPolicyIds.has(id)) return undefined;
  const ending = english
    ? ' Parameter storage is unchanged. TF32 is off and attention uses SDPA math. Memory use and runtime may increase; resume requires the same compute policy and environment.'
    : '参数存储精度不变。关闭 TF32，使用 SDPA 数学实现，可能增加显存和耗时；续训须使用相同计算策略和运行环境。';
  if (id === animaFp16) return (english
    ? 'Anima single-GPU LoRA keeps FP16 rounding, intermediates and outputs, with FP32 matrix operations. AMP and GradScaler remain active; previews use native operations.'
    : 'Anima 单卡 LoRA 保留 FP16 舍入、中间结果与输出，矩阵运算使用 FP32。自动混合精度和梯度缩放保持启用，预览使用原生计算。') + ending;
  if (id === sdxlFp16) return (english
    ? 'SDXL single-GPU LoKr retains native FP16 training and GradScaler. Only preview backbone linear contractions use FP32, retaining FP16 rounding and outputs.'
    : 'SDXL 单卡 LoKr 保留原生 FP16 训练和梯度缩放。仅预览主模型线性层使用 FP32 矩阵运算，仍保留 FP16 舍入与输出。') + ending;
  const size = id.startsWith('dtk-klein9b-') ? '9B' : '4B';
  const strategy = id.includes('-fsdp-')
    ? (english ? `Klein ${size} uses two-GPU FSDP sharding. ` : `Klein ${size} 使用双卡 FSDP 显存分片。`)
    : (english ? `Klein ${size} uses two-GPU DDP data parallelism. ` : `Klein ${size} 使用双卡 DDP 数据并行。`);
  if (size === '9B') return strategy + (english
    ? `Backbone and frozen Qwen3 linear layers${id.includes('-lora-') ? ' and LoRA' : ''} use FP32 matrix operations, preserving BF16 rounding and outputs. Memory and runtime may increase; resume requires the same settings and environment.`
    : `主模型与冻结 Qwen3 的线性层${id.includes('-lora-') ? ' 和 LoRA' : ''} 使用 FP32 矩阵运算，保留 BF16 舍入与输出。可能增加显存和耗时；续训须保持相同设置与环境。`);
  return strategy + (id.includes('-lora-')
    ? (english ? 'Backbone linear layers and LoRA retain BF16 rounding and outputs with FP32 matrix operations. ' : '主模型线性层和 LoRA 保留 BF16 舍入与输出，矩阵运算使用 FP32。')
    : id.includes('-fsdp-')
    ? (english ? 'Backbone linear layers use FP32 contractions with BF16 rounding and outputs; LoKr operations remain native. ' : '主模型线性层使用 FP32 矩阵运算并保留 BF16 舍入与输出，LoKr 运算保持原生实现。')
    : (english ? 'Training retains native BF16 operations. ' : '训练保持原生 BF16 计算。'))
    + (english ? 'Preview backbone linear layers use FP32 contractions with BF16 rounding and outputs. Text encoders remain frozen.' : '预览主模型线性层使用 FP32 矩阵运算，保留 BF16 舍入与输出。文本编码器保持冻结。') + ending;
}
