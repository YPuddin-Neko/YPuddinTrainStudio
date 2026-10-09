import schema from './gptSovitsVersionSchema.json';
import type { GptSovitsVersionConfig, TtsConfigResponse, TtsTrainSchema } from '../../api/tts';

export type GptSovitsConfigResponse = Omit<TtsConfigResponse, 'config'> & { config: GptSovitsVersionConfig };
type GptSettings = NonNullable<GptSovitsVersionConfig['gpt']>;
type SovitsSettings = NonNullable<GptSovitsVersionConfig['sovits']>;
export type GptSovitsField = Exclude<keyof GptSovitsVersionConfig, 'gpt' | 'sovits'> | `gpt.${keyof GptSettings}` | `sovits.${keyof SovitsSettings}`;
export type GptSovitsDraft = Record<GptSovitsField, string | boolean>;
export type GptSovitsFieldProblem = { field: string; message: string };
export type GptSovitsFieldSchema = {
  type?: string; default?: string | number | boolean; const?: string; enum?: (string | number)[];
  minimum?: number; maximum?: number; exclusiveMinimum?: number; exclusiveMaximum?: number;
  $ref?: string; properties?: Record<string, GptSovitsFieldSchema>; $defs?: Record<string, GptSovitsFieldSchema>;
};
export const gptSovitsGroups: { id: string; zh: string; en: string; fields: GptSovitsField[] }[] = [
  { id: 'environment', zh: '运行环境与模型', en: 'Environment and model', fields: ['engine', 'variant', 'stage', 'python_path', 'trainer_path', 'model_path', 'pretrained_gpt', 'pretrained_sovits'] },
  { id: 'gpt', zh: 'GPT 训练', en: 'GPT training', fields: ['gpt.epochs', 'gpt.batch_size', 'gpt.precision', 'gpt.seed', 'gpt.save_every_epoch', 'gpt.save_latest', 'gpt.learning_rate', 'gpt.initial_learning_rate', 'gpt.final_learning_rate', 'gpt.warmup_steps', 'gpt.decay_steps', 'gpt.dpo', 'gpt.max_seconds', 'gpt.num_workers'] },
  { id: 'sovits', zh: 'SoVITS 训练', en: 'SoVITS training', fields: ['sovits.epochs', 'sovits.batch_size', 'sovits.precision', 'sovits.seed', 'sovits.save_every_epoch', 'sovits.save_latest', 'sovits.learning_rate', 'sovits.adam_beta1', 'sovits.adam_beta2', 'sovits.adam_epsilon', 'sovits.lr_decay', 'sovits.log_interval', 'sovits.lora_rank', 'sovits.gradient_checkpointing'] },
];
export const gptSovitsFields = gptSovitsGroups.flatMap(group => group.fields);

function flattenSchema(root: GptSovitsFieldSchema) {
  const properties: Partial<Record<GptSovitsField, GptSovitsFieldSchema>> = {};
  for (const field of gptSovitsFields) {
    let node: GptSovitsFieldSchema | undefined = root;
    for (const segment of field.split('.')) {
      if (node?.$ref) node = node.$ref.startsWith('#/$defs/') ? root.$defs?.[node.$ref.slice(8)] : undefined;
      node = node?.properties?.[segment];
    }
    if (!node || !['string', 'integer', 'number', 'boolean'].includes(node.type || '')) return null;
    properties[field] = node;
  }
  return properties as Record<GptSovitsField, GptSovitsFieldSchema>;
}
export const gptSovitsFieldSchemas = flattenSchema(schema as GptSovitsFieldSchema)!;
export function resolveGptSovitsSchema(value: TtsTrainSchema) {
  if (value.engine !== 'gpt-sovits-v5' || value.schema_version !== 1 || value.groups.length !== gptSovitsGroups.length) return null;
  const properties = flattenSchema(value.json_schema as GptSovitsFieldSchema);
  if (!properties || properties.engine.const !== 'gpt-sovits-v5') return null;
  if (!value.groups.every((group, index) => group.id === gptSovitsGroups[index].id
    && group.fields.length === gptSovitsGroups[index].fields.length
    && new Set(group.fields).size === group.fields.length
    && group.fields.every(field => gptSovitsGroups[index].fields.includes(field as GptSovitsField)))) return null;
  return { properties, groups: value.groups.map((group, index) => ({ ...gptSovitsGroups[index], fields: group.fields as GptSovitsField[] })) };
}
export const isGptSovitsConfigResponse = (value: TtsConfigResponse): value is GptSovitsConfigResponse => value.config.engine === 'gpt-sovits-v5';
export const isScientificField = (field: string) => field.endsWith('learning_rate') || field === 'sovits.adam_epsilon';
export function gptSovitsValue(config: GptSovitsVersionConfig, field: GptSovitsField): unknown {
  const [group, name] = field.split('.');
  if (group === 'gpt') return config.gpt?.[name as keyof GptSettings];
  if (group === 'sovits') return config.sovits?.[name as keyof SovitsSettings];
  return config[group as Exclude<keyof GptSovitsVersionConfig, 'gpt' | 'sovits'>];
}
export function toGptSovitsDraft(config: GptSovitsVersionConfig): GptSovitsDraft {
  return Object.fromEntries(gptSovitsFields.map(field => {
    const value = gptSovitsValue(config, field) ?? gptSovitsFieldSchemas[field].default ?? '';
    return [field, typeof value === 'number' ? isScientificField(field) ? value.toExponential() : String(value) : value];
  })) as GptSovitsDraft;
}
export function parseGptSovitsDraft(draft: GptSovitsDraft, english: boolean, properties = gptSovitsFieldSchemas) {
  const problems: GptSovitsFieldProblem[] = [];
  const config: Record<string, unknown> = { engine: 'gpt-sovits-v5', gpt: {}, sovits: {} };
  for (const field of gptSovitsFields) {
    const rule = properties[field], raw = draft[field];
    const numeric = rule.type === 'integer' || rule.type === 'number';
    const value = numeric ? Number(raw) : raw;
    const invalid = numeric ? typeof raw !== 'string' || !raw.trim() || !Number.isFinite(value)
      || rule.type === 'integer' && !Number.isSafeInteger(value)
      || rule.minimum != null && Number(value) < rule.minimum || rule.maximum != null && Number(value) > rule.maximum
      || rule.exclusiveMinimum != null && Number(value) <= rule.exclusiveMinimum || rule.exclusiveMaximum != null && Number(value) >= rule.exclusiveMaximum
      : rule.type === 'boolean' ? typeof raw !== 'boolean' : typeof raw !== 'string';
    if (invalid || rule.enum && !rule.enum.includes(value as string | number) || rule.const != null && value !== rule.const) {
      problems.push({ field, message: english ? 'Enter a valid value within the allowed range.' : '请填写符合范围的有效数值或选项。' });
    }
    const [group, name] = field.split('.');
    if (name) (config[group] as Record<string, unknown>)[name] = value;
    else config[group] = value;
  }
  if (!problems.length) {
    for (const group of ['gpt', 'sovits'] as const) {
      const epochs = Number(draft[`${group}.epochs`]), interval = Number(draft[`${group}.save_every_epoch`]);
      if (interval > epochs || epochs % interval !== 0) problems.push({ field: `${group}.save_every_epoch`, message: english ? 'The save interval must divide the total epochs without a remainder.' : '保存间隔不能超过总轮数，且必须能整除总轮数。' });
    }
    if (Number(draft['gpt.warmup_steps']) >= Number(draft['gpt.decay_steps'])) problems.push({ field: 'gpt.warmup_steps', message: english ? 'Warmup steps must be less than decay steps.' : '预热步数必须小于衰减步数。' });
  }
  return { config: config as GptSovitsVersionConfig, problems };
}
export function mergeGptSovitsDraft(base: GptSovitsVersionConfig, local: GptSovitsDraft, remote: GptSovitsVersionConfig) {
  const before = toGptSovitsDraft(base), next = toGptSovitsDraft(remote);
  for (const field of gptSovitsFields) if (local[field] !== before[field]) next[field] = local[field];
  return next;
}
export const gptSovitsDraftKey = (pid: string, vid: string) => `tts-version-draft:gpt-sovits-v5:v1:${pid}:${vid}`;
export function readGptSovitsDraft(key: string, pid: string, vid: string): { base: GptSovitsConfigResponse; draft: GptSovitsDraft } | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) || 'null');
    if (value?.base?.scope?.project_id !== pid || value?.base?.scope?.version_id !== vid || !Number.isInteger(value.base.revision) || value.base.config?.engine !== 'gpt-sovits-v5') return null;
    if (!gptSovitsFields.every(field => typeof value.draft?.[field] === (gptSovitsFieldSchemas[field].type === 'boolean' ? 'boolean' : 'string'))) return null;
    return value;
  } catch { return null; }
}

type Copy = [string, string, string, string, string?, string?];
const copy: Partial<Record<GptSovitsField, Copy>> = {
  engine: ['GPT-SoVITS v5', 'GPT-SoVITS v5', '', ''],
  variant: ['模型变体', 'Model variant', '选择 v5dev 或 v5turbo 配套模型。', 'Choose the matching v5dev or v5turbo models.', 'SoVITS 基础权重须与变体匹配；两个变体共用默认 GPT 基础权重。', 'SoVITS base weights must match the variant. Both variants share the default GPT base weights.'],
  stage: ['训练阶段', 'Training stages', '可依次训练两阶段，或只训练其中一阶段。', 'Train both stages in sequence or a single stage.', '两阶段：先训练 SoVITS，再训练 GPT。\n仅 GPT / 仅 SoVITS：另一阶段设置保留但不执行，最终结果配对其基础权重。', 'Both: trains SoVITS first, then GPT.\nGPT only / SoVITS only: keeps the other stage’s settings without running it and pairs its base weights in the final result.'],
  python_path: ['Python 解释器', 'Python executable', 'GPT-SoVITS 独立环境中的 Python。', 'Python in the dedicated GPT-SoVITS environment.', '填写训练服务器上的可执行文件路径，例如 /opt/gpt-sovits/.venv/bin/python；每个任务使用一张 NVIDIA CUDA 显卡。', 'Enter the executable path on the training server, e.g. /opt/gpt-sovits/.venv/bin/python. Each job uses one NVIDIA CUDA GPU.'],
  trainer_path: ['GPT-SoVITS 源码目录', 'GPT-SoVITS source directory', '包含固定版本上游训练器的本地目录。', 'Local directory of the pinned upstream trainer.', '使用 cuda_graph_accel_v5 分支提交 f652b1da5af29a6955f9c3911aa71b7daa6618bc 的干净检出。', 'Use a clean checkout of cuda_graph_accel_v5 at f652b1da5af29a6955f9c3911aa71b7daa6618bc.'],
  model_path: ['预训练模型目录', 'Pretrained model directory', '用于查找基础权重及配套模型资源。', 'Locate base weights and supporting model resources.', '用于查找 GPT 基础权重、所选变体的 SoVITS 基础权重、声码器，以及文本和语音预处理资源。GPT 与 SoVITS 基础权重可在下方单独指定；使用训练器下载的模型包时，选择该模型包目录。', 'Locates GPT base weights, the selected variant’s SoVITS base weights, the vocoder, and text and speech preprocessing resources. GPT and SoVITS base weights can be specified separately below. For a package downloaded in the app, select its package directory.'],
  pretrained_gpt: ['GPT 基础权重', 'GPT base weights', '留空使用模型目录中的默认 GPT 基础权重。', 'Leave empty to use the default GPT base weights in the model directory.', '填写训练服务器上的 GPT 基础权重文件路径，不能使用 LoRA 导出。v5dev 和 v5turbo 共用默认 GPT 基础权重。', 'Enter the server path to GPT base weights, not a LoRA export. v5dev and v5turbo share the default GPT base weights.'],
  pretrained_sovits: ['SoVITS 基础权重', 'SoVITS base weights', '留空使用所选变体的官方基础权重。', 'Leave empty to use the variant’s official base weights.', '可填写训练服务器上的 SoVITS 基础权重文件路径，不能使用 LoRA 导出。留空时从预训练模型目录选择当前变体的官方权重。', 'Optional server path to SoVITS base weights, not a LoRA export. Empty selects the variant’s official weights from the pretrained model directory.'],
  'gpt.learning_rate': ['GPT 学习率', 'GPT learning rate', '预热结束时的目标学习率。', 'Target learning rate at the end of warmup.'],
  'gpt.initial_learning_rate': ['GPT 初始学习率', 'GPT initial learning rate', '学习率预热的起始值。', 'Starting learning rate for warmup.'],
  'gpt.final_learning_rate': ['GPT 最终学习率', 'GPT final learning rate', '学习率衰减的目标值。', 'Target learning rate after decay.'],
  'gpt.warmup_steps': ['GPT 预热步数', 'GPT warmup steps', '预热步数必须小于衰减步数。', 'Warmup steps must be less than decay steps.', '不能用 0 关闭预热。', 'A value of 0 does not disable warmup.'],
  'gpt.decay_steps': ['GPT 衰减步数', 'GPT decay steps', '学习率衰减计划的步数。', 'Number of steps in the learning-rate decay schedule.'],
  'gpt.batch_size': ['GPT 批大小', 'GPT batch size', '设置 GPT 每批处理的音频条数。', 'Set the number of audio clips per GPT batch.', '启用 DPO 时先将此值减半并向下取整，再按有效样本数调整实际批大小；每批至少保留 1 条音频。', 'DPO first halves this value and rounds down, then the trainer adjusts it for the number of valid samples. Each batch contains at least one clip.'],
  'gpt.dpo': ['GPT DPO', 'GPT DPO', '', '', '在常规损失之外加入偏好损失，以训练样本构造对比序列。开启后会将配置的批大小减半，再按有效样本数调整；实际每批至少 1 条。', 'Adds a preference loss using comparison sequences constructed from the training samples. Enabling it halves the configured batch size before adjustment for available samples, with at least one clip per batch.'],
  'gpt.max_seconds': ['GPT 音频时长上限', 'GPT maximum audio duration', '过滤超过指定时长的 GPT 训练样本。', 'Filter GPT training samples longer than this duration.', '单位为秒，例如 20 表示排除超过约 20 秒的样本。实际按预处理后的语义 token 长度筛选，也会排除过长的音素序列，不会裁剪或修改原始录音。', 'Measured in seconds: 20 excludes samples longer than approximately 20 seconds. Filtering uses preprocessed semantic-token lengths and also excludes overly long phoneme sequences. It does not trim or modify original recordings.'],
  'gpt.num_workers': ['GPT 数据加载进程', 'GPT data loader workers', '并行加载 GPT 训练样本。', 'Load GPT training samples in parallel.', '增加进程可并行读取数据，也会增加内存占用。GPT 使用常驻加载进程，不能填 0。', 'More workers load data in parallel and use more memory. GPT uses persistent workers, so 0 is not supported.'],
  'sovits.learning_rate': ['SoVITS 学习率', 'SoVITS learning rate', '控制 SoVITS 参数更新幅度。', 'Controls the size of SoVITS parameter updates.', '例如 0.0001（1e-4）。每轮结束后再乘以“学习率衰减”系数。', 'For example, 0.0001 (1e-4). The learning-rate decay factor is applied after each epoch.'],
  'sovits.adam_beta1': ['SoVITS Adam Beta1', 'SoVITS Adam Beta1', '控制梯度移动平均对历史梯度的保留程度。', 'Control how much gradient history is retained in the moving average.', 'AdamW 用此系数计算梯度的一阶矩。越接近 1，历史梯度的影响保留越久。', 'AdamW uses this coefficient for the first gradient moment. Values closer to 1 retain older gradients longer.'],
  'sovits.adam_beta2': ['SoVITS Adam Beta2', 'SoVITS Adam Beta2', '控制梯度平方移动平均的平滑程度。', 'Control smoothing of the squared-gradient moving average.', 'AdamW 用此系数计算梯度的二阶矩。越接近 1，对近期梯度变化的反应越缓慢。', 'AdamW uses this coefficient for the second gradient moment. Values closer to 1 respond more slowly to recent gradient changes.'],
  'sovits.adam_epsilon': ['SoVITS Adam Epsilon', 'SoVITS Adam epsilon', '防止优化器更新时除数过小。', 'Prevent very small denominators in optimizer updates.', '加在 AdamW 更新公式分母上的稳定项，例如 0.000000001（1e-9）；不能填 0 关闭。', 'A stability term added to the denominator in AdamW updates, for example 0.000000001 (1e-9). It cannot be disabled with 0.'],
  'sovits.lr_decay': ['SoVITS 学习率衰减', 'SoVITS learning rate decay', '每轮乘到学习率上的系数；填 1 不衰减。', 'Factor applied to the learning rate each epoch; 1 keeps it unchanged.', '例如 0.99 表示每轮结束后将学习率降为前一轮的 99%；不能用 0 关闭衰减。', 'For example, 0.99 keeps 99% of the previous epoch’s learning rate after each epoch. A value of 0 does not disable decay.'],
  'sovits.log_interval': ['SoVITS 指标记录间隔', 'SoVITS metrics interval', '按训练步数记录指标。', 'Record metrics at this training-step interval.', '例如 100 表示按 100 步的间隔记录损失和学习率；增大后页面指标更新会更稀疏。', 'For example, 100 records loss and learning rate at 100-step intervals. Larger values update displayed metrics less often.'],
  'sovits.lora_rank': ['SoVITS LoRA Rank', 'SoVITS LoRA Rank', '选择 SoVITS LoRA 的低秩维度。', 'Choose the rank of the SoVITS LoRA matrices.', 'Rank 越大，LoRA 参数量与权重文件越大；Alpha 与 Rank 相同，目标层由模型固定。', 'Larger ranks add LoRA parameters and increase weight-file size. Alpha equals Rank, and target layers are fixed by the model.'],
  'sovits.gradient_checkpointing': ['SoVITS 梯度检查点', 'SoVITS gradient checkpointing', '', '', '在反向传播时重新计算部分中间结果，以减少显存占用并增加计算量。', 'Recomputes intermediate results during backpropagation to reduce GPU memory use at the cost of extra computation.'],
};
const stageCopy: Record<string, Copy> = {
  epochs: ['训练轮数', 'epochs', '本阶段遍历训练数据的轮数。', 'Passes over training data in this stage.', '完成指定轮数后结束本阶段。每轮使用过滤和预处理后的训练数据，实际样本数可能与原始清单不同。', 'This stage ends after the specified number of epochs over the filtered, preprocessed training data. The sample count may differ from the original list.'],
  batch_size: ['批大小', 'batch size', '本阶段每批处理的音频条数。', 'Audio clips processed per batch in this stage.', '增大后会同时处理更多音频，通常需要更多显存；每轮批次数也会随之变化。', 'Larger batches process more clips at once and usually need more GPU memory. They also change the number of batches per epoch.'],
  precision: ['训练精度', 'precision', '本阶段训练使用的数值精度。', 'Numerical precision used by this training stage.', 'FP16：使用半精度训练以减少显存占用。\nFP32：使用单精度训练，显存占用通常更高。', 'FP16: uses half-precision training to reduce GPU memory use.\nFP32: uses single-precision training, usually with higher GPU memory use.'],
  seed: ['随机种子', 'seed', '固定本阶段的随机数起点；0 也是有效种子。', 'Starting seed for this stage; 0 is a valid seed.', '对比参数时保持种子一致，便于比较训练结果。不同数据、硬件或训练环境仍可能产生不同结果。', 'Keep the seed unchanged when comparing settings. Different data, hardware or training environments can still produce different results.'],
  save_every_epoch: ['保存间隔', 'save interval', '按训练轮数保存权重。', 'Save weights at this epoch interval.', '间隔必须整除总轮数，且不能超过总轮数。例如训练 15 轮可每 5 轮保存，以包含最后一轮权重。', 'The interval must divide total epochs without a remainder and cannot exceed them. For example, a 15-epoch run can save every 5 epochs so the final epoch is included.'],
  save_latest: ['只保留最新训练状态', 'keep latest training state', '', '', '开启：新的训练状态覆盖上一份。\n关闭：按保存间隔保留各份训练状态。\n两种方式都保留间隔导出的模型权重；此开关不提供断点续训。', 'Enabled: new training state replaces the previous state.\nDisabled: keeps training states at each save interval.\nBoth retain interval weight exports. This option does not enable resuming training.'],
};
export function gptSovitsFieldCopy(field: string, english: boolean) {
  const [stage, name] = field.split('.');
  const shared = stageCopy[name];
  const c = copy[field as GptSovitsField] || (shared && (stage === 'gpt' || stage === 'sovits') ? shared.map((value, index) => index < 2 ? `${stage === 'gpt' ? 'GPT' : 'SoVITS'} ${value}` : value) as Copy : undefined);
  if (!c) return { label: field, hint: '', help: '' };
  return { label: c[english ? 1 : 0], hint: c[english ? 3 : 2], help: c[english ? 5 : 4] || c[english ? 3 : 2] };
}
