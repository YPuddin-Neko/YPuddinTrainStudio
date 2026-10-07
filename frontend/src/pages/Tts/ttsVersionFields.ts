import schema from './ttsVersionSchema.json';
import type { TtsConfigResponse, TtsTrainSchema, TtsVersionConfig } from '../../api/tts';

export type VoxConfigResponse = Omit<TtsConfigResponse, 'config'> & { config: TtsVersionConfig };
export type TtsField = keyof TtsVersionConfig;
export type TtsDraft = Record<TtsField, string | boolean | string[]>;
export type FieldProblem = { field: string; message: string };
export type FieldSchema = { type?: string; minimum?: number; maximum?: number; exclusiveMinimum?: number; exclusiveMaximum?: number; anyOf?: FieldSchema[]; items?: { enum?: string[] } };
export const fieldSchemas = schema.properties as Record<TtsField, FieldSchema>;
export const fields = Object.keys(fieldSchemas) as TtsField[];
export const groups: { id: string; zh: string; en: string; fields: TtsField[] }[] = [
  { id: 'environment', zh: '运行环境与模型', en: 'Environment and model', fields: ['python_path', 'trainer_path', 'model_path'] },
  { id: 'batching', zh: '批次与数据加载', en: 'Batching and data loading', fields: ['batch_size', 'grad_accum_steps', 'num_workers', 'preprocessing_num_workers', 'max_batch_tokens'] },
  { id: 'optimization', zh: '训练与学习率', en: 'Training and learning rate', fields: ['num_iters', 'learning_rate', 'warmup_steps', 'weight_decay', 'max_grad_norm', 'max_steps', 'loss_diff_weight', 'loss_stop_weight'] },
  { id: 'intervals', zh: '保存与验证', en: 'Saving and validation', fields: ['save_interval', 'valid_interval', 'log_interval'] },
  { id: 'lora', zh: 'LoRA', en: 'LoRA', fields: ['lora_rank', 'lora_alpha', 'lora_dropout', 'lora_enable_lm', 'lora_target_modules_lm', 'lora_enable_dit', 'lora_target_modules_dit', 'lora_enable_proj', 'lora_target_proj_modules'] },
];
export const targetOwners: Partial<Record<TtsField, TtsField>> = { lora_target_modules_lm: 'lora_enable_lm', lora_target_modules_dit: 'lora_enable_dit', lora_target_proj_modules: 'lora_enable_proj' };

export function resolveTrainingSchema(value: TtsTrainSchema) {
  const properties = value.json_schema.properties as Record<TtsField, FieldSchema> | undefined;
  const grouped = value.groups.flatMap(group => group.fields);
  if (value.engine !== 'voxcpm1.5' || value.schema_version !== 1 || !properties
    || grouped.length !== fields.length || new Set(grouped).size !== fields.length
    || !fields.every(field => grouped.includes(field) && properties[field])
    || !value.groups.every(group => groups.some(known => known.id === group.id))) return null;
  return {
    properties,
    groups: value.groups.map(group => ({ ...groups.find(known => known.id === group.id)!, fields: group.fields.filter(field => field !== 'engine') as TtsField[] })),
  };
}

export function toDraft(config: TtsVersionConfig): TtsDraft {
  return Object.fromEntries(fields.map(field => {
    const value = config[field];
    return [field, typeof value === 'number' ? field === 'learning_rate' ? value.toExponential() : String(value) : value ?? ''];
  })) as TtsDraft;
}

export function parseDraft(draft: TtsDraft, invalidMessage: string, properties = fieldSchemas): { config: TtsVersionConfig; problems: FieldProblem[] } {
  const problems: FieldProblem[] = [];
  const config = Object.fromEntries(fields.map(field => {
    const property = properties[field];
    const rule = property.anyOf?.find(item => item.type !== 'null') || property;
    const raw = draft[field];
    if (rule.type !== 'integer' && rule.type !== 'number') return [field, raw];
    if (field === 'valid_interval' && raw === '') return [field, null];
    const value = Number(raw);
    if (typeof raw !== 'string' || !raw.trim() || !Number.isFinite(value) || rule.type === 'integer' && !Number.isSafeInteger(value)
      || rule.minimum != null && value < rule.minimum || rule.maximum != null && value > rule.maximum
      || rule.exclusiveMinimum != null && value <= rule.exclusiveMinimum || rule.exclusiveMaximum != null && value >= rule.exclusiveMaximum) problems.push({ field, message: invalidMessage });
    return [field, value];
  })) as TtsVersionConfig;
  return { config, problems };
}

/** Rebase only fields edited in this draft; untouched remote values remain intact. */
export function mergeDraft(base: TtsVersionConfig, local: TtsDraft, remote: TtsVersionConfig): TtsDraft {
  const before = toDraft(base), next = toDraft(remote);
  for (const field of fields) if (JSON.stringify(local[field]) !== JSON.stringify(before[field])) next[field] = local[field];
  return next;
}

export const draftKey = (projectId: string, versionId: string) => `tts-version-draft:v1:${projectId}:${versionId}`;
export function readDraft(key: string, projectId: string, versionId: string): { base: VoxConfigResponse; draft: TtsDraft } | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) || 'null');
    if (value?.base?.scope?.project_id !== projectId || value?.base?.scope?.version_id !== versionId || !Number.isInteger(value.base.revision) || value.base.config?.engine !== 'voxcpm1.5') return null;
    if (!fields.every(field => field in value.base.config && (typeof value.draft?.[field] === 'string' || typeof value.draft?.[field] === 'boolean' || Array.isArray(value.draft?.[field])))) return null;
    return value;
  } catch { return null; }
}

type Copy = [string, string, string, string, string?, string?];
const copy: Record<Exclude<TtsField, 'engine'>, Copy> = {
  python_path: ['Python 解释器', 'Python executable', 'VoxCPM 独立环境中的 Python。', 'Python in the dedicated VoxCPM environment.', '填写服务端可执行文件的绝对路径，例如 /opt/voxcpm/.venv/bin/python 或 D:\\VoxCPM\\.venv\\Scripts\\python.exe。', 'Absolute server path, e.g. /opt/voxcpm/.venv/bin/python or D:\\VoxCPM\\.venv\\Scripts\\python.exe.'],
  trainer_path: ['VoxCPM 源码目录', 'VoxCPM source directory', '包含固定版本训练器的本地目录。', 'Local directory of the pinned trainer.', '选择含 scripts/train_voxcpm_finetune.py 的仓库根目录。', 'Select the repository root containing scripts/train_voxcpm_finetune.py.'],
  model_path: ['VoxCPM 1.5 模型目录', 'VoxCPM 1.5 model directory', '完整预训练模型的本地目录。', 'Local directory of the complete pretrained model.', '包含配置、分词器及全部底模权重；不能使用 LoRA 目录或其他模型版本。', 'Include model configuration, tokenizer and all base weights; not a LoRA directory or another model version.'],
  batch_size: ['微批大小', 'Microbatch size', '每次前向与反向处理的音频条数。', 'Audio clips per forward and backward pass.'],
  grad_accum_steps: ['梯度累积', 'Gradient accumulation', '每次参数更新累计的微批数量。', 'Microbatches accumulated per optimizer update.', '单卡有效批量等于微批大小乘以梯度累积。', 'Effective single-GPU batch size is microbatch size × gradient accumulation.'],
  num_workers: ['数据加载进程', 'Data loader workers', '填 0 在训练主进程中加载。', 'Set 0 to load data in the training process.'],
  preprocessing_num_workers: ['数据预处理进程', 'Preprocessing workers', '准备训练数据时使用的进程数量。', 'Processes used to prepare training data.', '至少 1 个进程；此参数不支持填 0 关闭。', 'At least one process; 0 is not supported.'],
  max_batch_tokens: ['样本长度过滤', 'Sample length filter', '填 0 关闭长度过滤。', 'Set 0 to disable length filtering.', '限制估计的单条样本 token 长度：阈值为本值除以微批大小后向下取整。正值不能小于微批大小；不是动态批量或音频秒数上限。', 'Limits estimated sample tokens to floor(value / microbatch size). A positive value must be at least the microbatch size. This is neither dynamic batching nor an audio duration limit.'],
  num_iters: ['训练更新次数', 'Optimizer updates', '本次训练执行的参数更新总数。', 'Total optimizer updates for the run.'],
  learning_rate: ['学习率', 'Learning rate', '控制每次参数更新的幅度。', 'Controls the size of parameter updates.', '支持科学计数法，例如 1e-4；训练时由预热与余弦调度调整。', 'Accepts scientific notation, e.g. 1e-4. Warmup and cosine scheduling adjust the rate during training.'],
  warmup_steps: ['学习率预热', 'Warmup steps', '填 0 关闭预热。', 'Set 0 to disable warmup.', '不能超过训练更新次数或解析后的调度步数。', 'Cannot exceed total optimizer updates or the resolved scheduler steps.'],
  weight_decay: ['权重衰减', 'Weight decay', '填 0 关闭权重衰减。', 'Set 0 to disable weight decay.'],
  max_grad_norm: ['梯度裁剪阈值', 'Gradient clipping norm', '填 0 关闭梯度裁剪。', 'Set 0 to disable gradient clipping.'],
  max_steps: ['学习率调度步数', 'Scheduler steps', '填 0 跟随训练更新次数。', 'Set 0 to follow total optimizer updates.', '控制余弦学习率调度计划，不决定训练何时停止。小于训练更新次数时，后续学习率仍按该调度器计算。', 'Controls the cosine schedule, not when training stops. If shorter than training, the scheduler continues computing subsequent rates.'],
  loss_diff_weight: ['扩散损失权重', 'Diffusion loss weight', '填 0 不计入该损失；两项权重不能同时为 0。', 'Set 0 to exclude this loss; both weights cannot be 0.'],
  loss_stop_weight: ['停止预测损失权重', 'Stop loss weight', '填 0 不计入该损失；两项权重不能同时为 0。', 'Set 0 to exclude this loss; both weights cannot be 0.'],
  save_interval: ['权重保存间隔', 'Save interval', '按更新步数安排 LoRA 权重保存。', 'Schedules LoRA saves by update steps.', '沿训练器零基步数判断，首步与末步也可能保存。', 'The trainer uses a zero-based step condition; the first and final steps can also save.'],
  valid_interval: ['验证间隔', 'Validation interval', '留空跟随保存间隔；无验证数据时不运行验证。', 'Leave empty to follow the save interval; no validation data disables validation.', '只接受正整数或留空，0 不表示关闭验证。', 'Accepts a positive integer or empty; 0 does not disable validation.'],
  log_interval: ['指标记录间隔', 'Metrics interval', '按更新步数安排指标与进度记录。', 'Schedules metrics and progress records by update steps.'],
  lora_rank: ['LoRA Rank', 'LoRA Rank', 'LoRA 低秩矩阵的维度。', 'Dimension of the LoRA low-rank matrices.'],
  lora_alpha: ['LoRA Alpha', 'LoRA Alpha', 'LoRA 更新按 Alpha / Rank 缩放。', 'Scales LoRA updates by Alpha / Rank.'],
  lora_dropout: ['LoRA Dropout', 'LoRA Dropout', '填 0 关闭随机丢弃。', 'Set 0 to disable dropout.'],
  lora_enable_lm: ['语言模型', 'Language model', '', '', '是否训练语言模型的 LoRA；至少启用一个组件。', 'Train language-model LoRA; enable at least one component.'],
  lora_enable_dit: ['扩散模型', 'Diffusion model', '', '', '是否训练扩散模型的 LoRA；至少启用一个组件。', 'Train diffusion-model LoRA; enable at least one component.'],
  lora_enable_proj: ['投影层', 'Projection layers', '', '', '是否训练投影层的 LoRA；至少启用一个组件。', 'Train projection-layer LoRA; enable at least one component.'],
  lora_target_modules_lm: ['语言模型目标层', 'Language model targets', '选择参与 LoRA 训练的线性层。', 'Select linear layers for LoRA training.'],
  lora_target_modules_dit: ['扩散模型目标层', 'Diffusion model targets', '选择参与 LoRA 训练的线性层。', 'Select linear layers for LoRA training.'],
  lora_target_proj_modules: ['投影层目标', 'Projection targets', '选择参与 LoRA 训练的投影层。', 'Select projection layers for LoRA training.'],
};
export function fieldCopy(field: TtsField, english: boolean) {
  if (field === 'engine') return { label: 'VoxCPM 1.5', hint: '', help: '' };
  const c = copy[field];
  return { label: c[english ? 1 : 0], hint: c[english ? 3 : 2], help: c[english ? 5 : 4] || c[english ? 3 : 2] };
}
