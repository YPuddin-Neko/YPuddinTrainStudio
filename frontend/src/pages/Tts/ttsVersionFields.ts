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
  batch_size: ['微批大小', 'Microbatch size', '每次前向与反向处理的音频条数。', 'Audio clips per forward and backward pass.', '增大会同时处理更多音频，通常需要更多显存。不足一批的尾部样本不参与该轮训练。', 'Larger batches process more clips at once and usually need more GPU memory. An incomplete final training batch is dropped.'],
  grad_accum_steps: ['梯度累积', 'Gradient accumulation', '每次参数更新累计的微批数量。', 'Microbatches accumulated per optimizer update.', '单卡有效批量 = 微批大小 × 梯度累积。例如微批为 2、累积为 4 时，每次更新累计 8 条音频，无需一次将 8 条音频放入显存。', 'Effective single-GPU batch size = microbatch size × accumulation. A microbatch of 2 accumulated 4 times uses 8 clips per update without loading all 8 into GPU memory at once.'],
  num_workers: ['数据加载进程', 'Data loader workers', '并行加载训练数据；填 0 在主进程中加载。', 'Load training data in parallel; 0 uses the training process.', '大于 0 时使用独立子进程读取数据，可减少训练等待数据的时间，也会增加内存占用。训练前的数据准备由“数据预处理进程”控制。', 'Positive values use separate processes to read data, which can reduce waits for data while using more memory. Preprocessing before training is controlled separately.'],
  preprocessing_num_workers: ['数据预处理进程', 'Preprocessing workers', '准备训练数据时使用的进程数量。', 'Processes used to prepare training data.', '增大可让多个进程同时准备数据，也会增加 CPU 和内存占用。填 1 为单进程，不能用 0 跳过预处理。', 'Larger values prepare data in parallel and use more CPU and memory. Use 1 for a single process; 0 cannot skip preprocessing.'],
  max_batch_tokens: ['样本长度过滤', 'Sample length filter', '过滤过长的训练样本；填 0 关闭。', 'Filter long training samples; 0 disables filtering.', '按估计的 token 长度过滤整条样本，不裁剪音频。单条上限为本值 ÷ 微批大小后向下取整；例如填 400、微批为 2，上限为 200 token。正值不能小于微批大小；它不是动态批量或秒数上限。', 'Filters whole samples by estimated token length without trimming audio. The per-sample limit is floor(value / microbatch size): 400 with a microbatch of 2 allows 200 tokens. A positive value must be at least the microbatch size. This is not dynamic batching or a duration limit.'],
  num_iters: ['训练更新次数', 'Optimizer updates', '本次训练执行的参数更新总数。', 'Total optimizer updates for the run.', '梯度累积的多个微批合计为一次参数更新。例如填 1000，完成 1000 次更新后结束训练。', 'Multiple accumulated microbatches count as one optimizer update. For example, 1000 ends training after 1000 updates.'],
  learning_rate: ['学习率', 'Learning rate', '控制每次参数更新的幅度。', 'Controls the size of parameter updates.', '例如 0.0001（1e-4）。训练时由预热与余弦调度调整实际学习率；降低此值会减小参数更新幅度。', 'For example, 0.0001 (1e-4). Warmup and cosine scheduling adjust the actual rate during training. Lower values reduce the size of parameter updates.'],
  warmup_steps: ['学习率预热', 'Warmup steps', '训练初期逐步提高学习率；填 0 关闭。', 'Gradually raise the learning rate at the start; 0 disables warmup.', '按参数更新次数计数，不能超过训练更新次数或学习率调度步数；调度步数填 0 时按训练更新次数计算。', 'Counted in optimizer updates. It cannot exceed total updates or scheduler steps. Scheduler steps set to 0 resolve to total updates.'],
  weight_decay: ['权重衰减', 'Weight decay', '抑制权重持续增大；填 0 关闭。', 'Limit weight growth; 0 disables weight decay.', 'AdamW 在参数更新时按此系数衰减权重，用于正则化。例如 0.01；增大会加强衰减。', 'AdamW decays weights by this coefficient during updates for regularization. For example, 0.01; larger values increase decay.'],
  max_grad_norm: ['梯度裁剪阈值', 'Gradient clipping norm', '限制过大的梯度；填 0 关闭。', 'Limit large gradients; 0 disables clipping.', '梯度范数超过阈值时按比例缩小梯度，未超过时保持原值。例如填 1，将范数上限设为 1。', 'Gradients whose norm exceeds the threshold are scaled down; smaller gradients are unchanged. For example, 1 sets a maximum norm of 1.'],
  max_steps: ['学习率调度步数', 'Scheduler steps', '安排学习率变化；填 0 跟随训练更新次数。', 'Set the learning-rate schedule length; 0 follows total updates.', '只控制余弦调度的计划长度，不决定训练何时结束。设得比训练更新次数短时，后续学习率仍按余弦公式变化，不会停在最低值。', 'Controls the cosine schedule length, not when training stops. If shorter than total training updates, later rates still follow the cosine formula rather than staying at a minimum.'],
  loss_diff_weight: ['扩散损失权重', 'Diffusion loss weight', '调整扩散损失的占比；填 0 不计入。', 'Scale the diffusion loss contribution; 0 excludes it.', '扩散损失乘以此权重后与停止预测损失相加。例如填 0.5，扩散损失的贡献减半；两项损失权重不能同时为 0。', 'Diffusion loss is multiplied by this weight before being added to the stop-prediction loss. For example, 0.5 halves its contribution. The two loss weights cannot both be 0.'],
  loss_stop_weight: ['停止预测损失权重', 'Stop loss weight', '调整停止预测损失的占比；填 0 不计入。', 'Scale the stop-prediction loss contribution; 0 excludes it.', '这项损失用于学习何时结束语音生成，乘以此权重后计入总损失。例如填 0.5，其贡献减半；与扩散损失权重不能同时为 0。', 'Weights the loss for learning when speech generation should stop. For example, 0.5 halves its contribution. This and the diffusion loss weight cannot both be 0.'],
  save_interval: ['权重保存间隔', 'Save interval', '按更新步数保存 LoRA 权重。', 'Save LoRA weights at this update-step interval.', '按从 0 开始的训练步数判断保存间隔，首步和末步也可能保存，因此不只在完成 N、2N 次更新时生成权重。', 'The save condition uses steps starting at 0, and the first and last steps may also save, rather than only after N, 2N completed updates.'],
  valid_interval: ['验证间隔', 'Validation interval', '按更新步数验证；留空跟随保存间隔。', 'Validate at this update-step interval; empty follows the save interval.', '没有登记验证数据时不执行验证；有验证数据时每次最多计算 10 批。0 不能关闭验证。', 'No registered validation data means no validation. Each validation run processes at most 10 batches. A value of 0 does not disable validation.'],
  log_interval: ['指标记录间隔', 'Metrics interval', '按更新步数记录指标与进度。', 'Record metrics and progress at this update-step interval.', '例如填 100，按 100 步的间隔记录损失、学习率和进度。间隔越大，记录次数越少，页面更新也越稀疏。', 'For example, 100 records loss, learning rate and progress at 100-step intervals. Larger intervals produce fewer records and less frequent updates in the interface.'],
  lora_rank: ['LoRA Rank', 'LoRA Rank', 'LoRA 低秩矩阵的维度。', 'Dimension of the LoRA low-rank matrices.', 'Rank 越大，参与训练的 LoRA 参数越多，权重文件和显存开销也会增加。', 'A larger rank adds trainable LoRA parameters and increases weight-file size and GPU memory use.'],
  lora_alpha: ['LoRA Alpha', 'LoRA Alpha', '按 Alpha / Rank 缩放 LoRA 更新。', 'Scale LoRA updates by Alpha / Rank.', '例如 Rank 为 32、Alpha 为 16 时，缩放系数为 0.5；改变 Rank 时也会改变此比例。', 'Rank 32 with Alpha 16 gives a scale of 0.5. Changing Rank also changes this ratio.'],
  lora_dropout: ['LoRA Dropout', 'LoRA Dropout', '随机丢弃部分 LoRA 输入；填 0 关闭。', 'Randomly drop some LoRA inputs; 0 disables dropout.', '以小数表示丢弃概率，例如 0.1 表示 10%，不是填 10。仅在训练时随机丢弃输入，用于正则化。', 'Use a decimal probability: 0.1 means 10%, not 10. Inputs are randomly dropped only during training for regularization.'],
  lora_enable_lm: ['语言模型', 'Language model', '', '', '开启：训练语言模型中所选层的 LoRA。\n关闭：保留目标层选择，不训练该组件。\n语言模型、扩散模型和投影层至少开启一项。', 'Enabled: trains LoRA on the selected language-model layers.\nDisabled: keeps the selection without training this component.\nEnable at least one of the language model, diffusion model or projection layers.'],
  lora_enable_dit: ['扩散模型', 'Diffusion model', '', '', '开启：训练扩散模型中所选层的 LoRA。\n关闭：保留目标层选择，不训练该组件。\n语言模型、扩散模型和投影层至少开启一项。', 'Enabled: trains LoRA on the selected diffusion-model layers.\nDisabled: keeps the selection without training this component.\nEnable at least one of the language model, diffusion model or projection layers.'],
  lora_enable_proj: ['投影层', 'Projection layers', '', '', '开启：训练所选投影层的 LoRA。\n关闭：保留目标层选择，不训练该组件。\n语言模型、扩散模型和投影层至少开启一项。', 'Enabled: trains LoRA on the selected projection layers.\nDisabled: keeps the selection without training this component.\nEnable at least one of the language model, diffusion model or projection layers.'],
  lora_target_modules_lm: ['语言模型目标层', 'Language model targets', '选择语言模型中参与 LoRA 训练的层。', 'Select language-model layers for LoRA training.', 'q_proj、k_proj、v_proj、o_proj 分别对应注意力的查询、键、值和输出投影。开启语言模型时至少选一项。', 'q_proj, k_proj, v_proj and o_proj are the attention query, key, value and output projections. Select at least one when the language model is enabled.'],
  lora_target_modules_dit: ['扩散模型目标层', 'Diffusion model targets', '选择扩散模型中参与 LoRA 训练的层。', 'Select diffusion-model layers for LoRA training.', 'q_proj、k_proj、v_proj、o_proj 分别对应注意力的查询、键、值和输出投影。开启扩散模型时至少选一项。', 'q_proj, k_proj, v_proj and o_proj are the attention query, key, value and output projections. Select at least one when the diffusion model is enabled.'],
  lora_target_proj_modules: ['投影层目标', 'Projection targets', '选择参与 LoRA 训练的投影层。', 'Select projection layers for LoRA training.', '这些投影层连接语音编码、语言模型与扩散模型。开启投影层时至少选一项。', 'These projections connect speech encoding, the language model and the diffusion model. Select at least one when projection layers are enabled.'],
};
export function fieldCopy(field: TtsField, english: boolean) {
  if (field === 'engine') return { label: 'VoxCPM 1.5', hint: '', help: '' };
  const c = copy[field];
  return { label: c[english ? 1 : 0], hint: c[english ? 3 : 2], help: c[english ? 5 : 4] || c[english ? 3 : 2] };
}
