/**
 * Chinese readings of the trainer's fixed log lines. The log file stays in English;
 * a line without a rule is shown as written.
 */

type Rule = [RegExp, (match: RegExpMatchArray) => string];

const FAMILIES: Record<string, string> = { anima: 'Anima', krea2: 'Krea 2', flux: 'FLUX.1', flux2: 'FLUX.2', sdxl: 'SDXL', toy: '测试模型' };
const SAVED: Record<string, string> = { weights: '权重', 'training state': '训练状态', model: '模型', file: '文件' };
const OUTCOMES: Record<string, string> = { finished: '完成', paused: '已暂停', stopped: '已停止', failed: '失败' };

/** "1904x2656 (133), 2832x2304 (19)" → "1904×2656（133 个样本）、2832×2304（19 个样本）". */
function layoutSizes(value: string): string {
  return value.split(', ').map(size => size.replace(/^(\d+)x(\d+) \((\d+)\)$/, '$1×$2（$3 个样本）')).join('、');
}

function seconds(value: string): string {
  return `${value} 秒`;
}

/** Durations logged as 70h28m, 5m31s or 12s. */
function duration(value: string): string {
  return value.replace(/(\d+)h/, '$1 小时 ').replace(/(\d+)m(?!s)/, '$1 分 ').replace(/(\d+)s/, '$1 秒').trim();
}

function bytes(value: string): string {
  const size = Number(value);
  if (!Number.isFinite(size)) return value;
  return size >= 1024 ** 3 ? `${(size / 1024 ** 3).toFixed(1)} GB` : `${(size / 1024 ** 2).toFixed(0)} MB`;
}

function adapterKinds(value: string): string {
  return value.split(', ').map(part => part.replace(/^(\w+) (\d+)$/, (_, kind: string, count: string) => `${kind.toUpperCase()} ${count} 层`)).join('，');
}

function vaePrecision(value: string): string {
  const [dtype] = value.split(', ');
  return dtype === 'unknown' ? '未知' : dtype.toUpperCase();
}

function vaeTiling(value: string): string {
  if (value === 'off') return '分块关闭';
  if (value === 'on' || /(?:^|, )\d+ px tiles(?:,|$)/.test(value)) return '分块开启';
  if (value === 'large-images' || /^\d+ px tiles above \d+x\d+$/.test(value)) return '大图分块开启';
  return value;
}

const RULES: Rule[] = [
  // PyTorch launcher and communication diagnostics; keep rank and all option values.
  [/^Setting OMP_NUM_THREADS environment variable for each process to be (\d+) in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed\.\s*$/, m => `为避免系统负载过高，默认将每个进程的 OMP_NUM_THREADS 设为 ${m[1]}；可按应用需要调整线程数以提高性能。`],
  [/^Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass\. This flag results in an extra traversal of the autograd graph every iteration,\s+which can adversely affect performance\. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off\. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters\.(?: \(function operator\(\)\))?$/, () => 'DDP 已设置 find_unused_parameters=True，但本次前向计算未发现未使用的参数。此选项每次迭代额外遍历一次自动求导图，可能影响性能；仅在确认每次迭代所有参数都参与计算时才可关闭。若模型有分支控制，后续迭代仍可能出现未使用参数，此警告可能是误报。'],
  [/^(?:WARNING: )?Logging before InitGoogleLogging\(\) is written to STDERR$/, () => 'InitGoogleLogging() 初始化前的日志会写入标准错误输出（STDERR）'],
  [/^(\[PG ID \d+ PG GUID \S+ Rank \d+\]) ProcessGroupNCCL initialization options: ([\s\S]*)$/, m => `${m[1]} ProcessGroupNCCL 初始化选项：${m[2]}`],
  [/^(\[PG ID \d+ PG GUID \S+ Rank \d+\]) ProcessGroupNCCL environments: ([\s\S]*)$/, m => `${m[1]} ProcessGroupNCCL 环境配置：${m[2]}`],
  [/^((?:.*:\d+: UserWarning: )?)(?:1)?Torch was not compiled with memory efficient attention\.(?: \(Triggered internally at (.+)\))?$/, m => `${m[1]}当前 PyTorch 未编译内存高效注意力后端。${m[2] ? `（触发位置：${m[2]}）` : ''}`],
  // Optional Triton: neither line means attention cannot run.
  [/^triton not found; flop counting will not work for triton kernels$/, () => '未找到 Triton：PyTorch 的 FLOPs 统计不包含 Triton 内核，只影响这项统计。'],
  [/^A matching Triton is not available, some optimizations will not be enabled$/, () => 'xFormers 没有可用的匹配 Triton：部分可选的 Triton 内核不可用，xFormers 的其他算子照常使用。'],

  // Model loading
  [/^loading (\w+) model components$/, m => `正在加载 ${FAMILIES[m[1]] || m[1]} 模型组件`],
  [/^model components loaded in ([\d.]+)s$/, m => `模型组件加载完成，用时 ${seconds(m[1])}`],
  [/^loaded Anima DiT: width=(\S+) blocks=(\S+) heads=(\S+)$/, m => `已加载 Anima DiT：宽度 ${m[1]}，${m[2]} 个模块，${m[3]} 个注意力头`],
  [/^prepared Krea 2 DiT metadata: features=(\S+) layers=(\S+) heads=(\S+)\/(\S+); weights deferred until after caching$/, m => `已读取 Krea 2 DiT 结构：宽度 ${m[1]}，${m[2]} 层，注意力头 ${m[3]}/${m[4]}；权重在缓存完成后加载`],
  [/^Krea 2: keeping CPU materialization \(weights (\d+) \+ margin (\d+) > free CUDA (\d+) bytes\)$/, m => `Krea 2：可用显存 ${bytes(m[3])} 放不下权重 ${bytes(m[1])} 加余量 ${bytes(m[2])}，改在内存中展开权重`],
  [/^Krea 2: materializing on (\S+) before block swap to release the source mmap before pinned allocation \(weights (\d+), free (\d+), margin (\d+) bytes\)$/, m => `Krea 2：先在 ${m[1]} 上展开权重（${bytes(m[2])}，可用 ${bytes(m[3])}），再启用块交换`],
  [/^Krea 2: (\d+) fp8_scaled linear layers kept in fp8 with their checkpoint scales$/, m => `Krea 2：${m[1]} 个 fp8_scaled 线性层保持 fp8，沿用权重文件中的缩放系数`],
  [/^Initializing image-only QwenImage 2D VAE$/, () => '正在初始化 QwenImage 2D VAE（仅图像）'],
  [/^Initializing VAE$/, () => '正在初始化 VAE'],
  [/^Loading VAE from (.+)$/, m => `正在加载 VAE：${m[1]}`],
  [/^Converted ComfyUI AutoencoderKL state dict keys to official format$/, () => '已将 ComfyUI 格式的 VAE 权重名转换为官方格式'],
  [/^Skipped (\d+) temporal-only VAE weights for QwenImage 2D VAE$/, m => `已跳过 ${m[1]} 个仅用于视频的 VAE 权重`],
  [/^Loaded (image-only QwenImage 2D )?VAE: <All keys matched successfully>$/, m => `${m[1] ? 'QwenImage 2D VAE' : 'VAE'} 加载完成，全部权重已匹配`],

  // Dataset and caches
  [/^dataset: (\d+) images \((\d+) captioned\), (\d+) training items in (\d+) buckets(?:, (\d+) validation images)?$/, m => `数据集：${m[1]} 张图片（${m[2]} 张有标注），${m[3]} 个训练样本，分为 ${m[4]} 个分桶${m[5] ? `；另有 ${m[5]} 张验证图片` : ''}`],
  [/^dataset: (\d+) images \((\d+) captioned\), (\d+) training items(?:, (\d+) validation images)?$/, m => `数据集：${m[1]} 张图片（${m[2]} 张有标注），${m[3]} 个训练样本${m[4] ? `；另有 ${m[4]} 张验证图片` : ''}`],
  [/^training layout: native resolution, (\d+) image layouts: (.+)$/, m => `训练方法：原生分辨率，${m[1]} 个图片布局：${layoutSizes(m[2])}`],
  [/^training layout: buckets, (\d+) buckets: (.+)$/, m => `训练方法：分桶，${m[1]} 个分桶：${layoutSizes(m[2])}`],
  [/^VAE encoding: (\d+)\/(\d+)$/, m => `VAE 编码中：${m[1]}/${m[2]}`],
  [/^text encoding: (\d+)\/(\d+)$/, m => `文本编码中：${m[1]}/${m[2]}`],
  [/^checking VAE cache: (\d+)\/(\d+)$/, m => `正在检查 VAE 缓存：${m[1]}/${m[2]}`],
  [/^VAE cache preparation: (\d+)\/(\d+)$/, m => `VAE 缓存处理中：${m[1]}/${m[2]}`],
  [/^checking text cache: (\d+)\/(\d+)$/, m => `正在检查文本缓存：${m[1]}/${m[2]}`],
  [/^text cache preparation: (\d+)\/(\d+)$/, m => `文本缓存处理中：${m[1]}/${m[2]}`],
  [/^checking shared (VAE|text) cache: (\d+)\/(\d+)$/, m => `检查共享${m[1] === 'VAE' ? ' VAE ' : '文本'}缓存：${m[2]}/${m[3]}`],
  [/^shared (VAE|text) cache preparation: (\d+)\/(\d+)$/, m => `共享${m[1] === 'VAE' ? ' VAE ' : '文本'}缓存准备中：${m[2]}/${m[3]}`],
  [/^shared (VAE|text) cache ready: (\d+) entries added in ([\d.]+)s$/, m => `共享${m[1] === 'VAE' ? ' VAE ' : '文本'}缓存已生成：新增 ${m[2]} 项，用时 ${seconds(m[3])}`],
  [/^reusing shared (VAE|text) cache: (\d+) items checked; no re-encoding needed$/, m => `复用共享${m[1] === 'VAE' ? ' VAE ' : '文本'}缓存：已检查 ${m[2]} 项，无需重新编码`],
  [/^cached (\d+) latents in ([\d.]+)s$/, m => `VAE 编码完成：新缓存 ${m[1]} 张图片的潜空间，用时 ${seconds(m[2])}`],
  [/^cached (\d+) latents$/, m => `VAE 编码完成：新缓存 ${m[1]} 张图片的潜空间`],
  [/^all latents were already cached$/, () => '复用 VAE 缓存：所有图片都已有缓存，无需重新编码'],
  [/^(VAE cache encode|online VAE encode): batch (\d+), precision (\S+), tiling (off|on|large-images)$/, m => `${m[1] === 'VAE cache encode' ? 'VAE 缓存编码' : 'VAE 在线编码'}：批次 ${m[2]}，精度 ${vaePrecision(m[3])}，${vaeTiling(m[4])}`],
  [/^(online )?VAE encode settings: VAE encode batch (\d+) \(images per VAE call\), training batch (\d+), VAE precision (.+?), spatial tiling (.+)$/, m => `${m[1] ? 'VAE 在线编码' : 'VAE 缓存编码'}：批次 ${m[2]}，精度 ${vaePrecision(m[4])}，${vaeTiling(m[5])}`],
  [/^VAE encode finished: images (\d+), VAE calls (\d+), at most (\d+) per call$/, m => `VAE 编码调用：${m[1]} 张图片，共 ${m[2]} 次调用，每次最多 ${m[3]} 张`],
  [/^online VAE encode, first training batch: images (\d+), VAE calls (\d+), at most (\d+) per call$/, m => `在线 VAE 编码（第一个训练批次）：${m[1]} 张图片，共 ${m[2]} 次调用，每次最多 ${m[3]} 张`],
  [/^VAE encode memory on (\S+): peak allocated ([\d.]+) GiB, peak reserved ([\d.]+) GiB \(this process, (this phase|run so far)\); device in use up to ([\d.]+) of ([\d.]+) GiB \(all processes\)$/, m => `VAE 编码显存（${m[1]}）：本进程已分配峰值 ${m[2]} GiB，本进程保留峰值 ${m[3]} GiB（${m[4] === 'this phase' ? '本阶段' : '训练开始至今'}）；整卡占用最高 ${m[5]} GiB，共 ${m[6]} GiB（含其他程序）`],
  [/^all text encodings were already cached$/, () => '复用文本缓存：所有文本都已有缓存，无需重新编码'],
  [/^cached (\d+) text encodings \((\d+) distinct captions\) in ([\d.]+)s$/, m => `文本编码完成：新缓存 ${m[1]} 条（${m[2]} 条不同的标注），用时 ${seconds(m[3])}`],
  [/^cached (\d+) text encodings \((\d+) captions of images and prompts\) in ([\d.]+)s$/, m => `文本编码完成：新缓存 ${m[1]} 条（共 ${m[2]} 条图片标注和提示词），用时 ${seconds(m[3])}`],
  [/^skipping unreadable image (.+) \((.+)\)$/, m => `跳过无法读取的图片 ${m[1]}（${m[2]}）`],

  // Training setup
  [/^injected adapters into (\d+) layers \(([^)]*)\): ([\d,]+) trainable parameters$/, m => `已注入适配器：${m[1]} 层（${adapterKinds(m[2])}），可训练参数 ${m[3]} 个`],
  [/^injected adapters: \{'layers': (\d+), 'trainable_params': (\d+)/, m => `已注入适配器：${m[1]} 层，可训练参数 ${Number(m[2]).toLocaleString('en-US')} 个`],
  [/^optimizer (\S+): (\d+) parameter groups, learning rates (.+), weight decay (\S+)$/, m => `优化器 ${m[1]}：${m[2]} 个参数组，学习率 ${m[3]}，权重衰减 ${m[4]}`],
  [/^scheduler managed by optimizer: warmup (\S+), (\d+) total steps$/, m => `学习率由优化器自行调度：预热 ${m[1]}，共 ${m[2]} 步`],
  [/^scheduler (\S+): warmup (\S+), (\d+) total steps$/, m => `学习率调度 ${m[1]}：预热 ${m[2]}，共 ${m[3]} 步`],
  [/^training (\d+) steps: (\d+) per epoch, effective batch (\d+) \((\d+) x (\d+) accumulation x (\d+) GPU\)$/, m => `开始训练：共 ${m[1]} 步，每轮 ${m[2]} 步，等效批次 ${m[3]}（批次 ${m[4]} × 梯度累积 ${m[5]} × ${m[6]} 张 GPU）`],
  [/^resuming at step (\d+)\/(\d+)$/, m => `从第 ${m[1]} 步继续训练（共 ${m[2]} 步）`],
  [/^resume requested: restoring the run from (.+)$/, m => `收到继续训练信号：正在从恢复点恢复（${m[1]}）`],
  [/^resumed training (\S+ \S+) \| step (\d+)\/(\d+) \| epoch (\S+) \| loss (\S+) \| from (.+)$/, m => `开始继续训练：${m[1]} | 第 ${m[2]}/${m[3]} 步 | 第 ${m[4]} 轮 | Loss ${m[5]} | 恢复点 ${m[6]}`],
  [/^preview seed (\d+) \(kept from the resume point\)$/, m => `预览图种子 ${m[1]}（沿用恢复点中的种子）`],
  [/^preview seed (\d+) \(random for this run\)$/, m => `预览图种子 ${m[1]}（本次训练随机生成，所有预览图共用）`],
  [/^preview seed (\d+)$/, m => `预览图种子 ${m[1]}`],

  // Progress
  [/^step (\d+)\/(\d+) \| epoch (\S+) \| loss (\S+) \| avg loss (\S+) \| lr (.+?) \| grad norm (\S+) \| (\S+) it\/s \| eta (\S+)$/, m => `步数 ${m[1]}/${m[2]} | 轮次 ${m[3]} | Loss ${m[4]} | 平均 Loss ${m[5]} | 学习率 ${m[6]} | 梯度范数 ${m[7]} | ${m[8]} it/s | 剩余 ${m[9] === '-' ? '—' : duration(m[9])}`],
  [/^epoch (\d+) finished at step (\d+) \| loss (\S+) \| (\S+)( since resume)?$/, m => `第 ${m[1]} 轮完成（第 ${m[2]} 步）| 本轮平均 Loss ${m[3]} | 用时 ${duration(m[4])}${m[5] ? '（自恢复起）' : ''}`],
  [/^epoch (\d+) finished at step (\d+)$/, m => `第 ${m[1]} 轮完成（第 ${m[2]} 步）`],
  [/^sampling (\d+) previews at step (\d+) \(seed (\d+)\)$/, m => `第 ${m[2]} 步：开始生成 ${m[1]} 张预览图（种子 ${m[3]}）`],
  [/^preview (\d+)\/(\d+) saved: (\S+) \((\d+)x(\d+), seed (\d+), ([\d.]+)s\)$/, m => `预览图 ${m[1]}/${m[2]} 已保存：${m[3]}（${m[4]}×${m[5]}，种子 ${m[6]}，用时 ${seconds(m[7])}）`],
  [/^previews finished in ([\d.]+)s$/, m => `预览图生成完成，用时 ${seconds(m[1])}`],
  [/^saved (weights|model|file)( \(EMA\))?: (.+?) \| step (\d+)\/(\d+) \| epoch (\S+) \| loss (\S+)$/, m => `已保存${m[2] ? ' EMA ' : ''}${SAVED[m[1]]}：${m[3]} | 第 ${m[4]}/${m[5]} 步 | 第 ${m[6]} 轮 | Loss ${m[7]}`],
  [/^saved resume point (\S+ \S+) \| step (\d+)\/(\d+) \| epoch (\S+) \| loss (\S+) \| (.+)$/, m => `已保存恢复点：${m[1]} | 第 ${m[2]}/${m[3]} 步 | 第 ${m[4]} 轮 | Loss ${m[5]} | ${m[6]}`],
  [/^saved (weights|training state|model|file)( \(EMA\))?: (.+)$/, m => `已保存${m[2] ? ' EMA ' : ''}${SAVED[m[1]]}：${m[3]}`],
  [/^(pause|stop) requested; (?:pausing|stopping) after step (\d+)\/(\d+) and saving a resume point(?: \(about (\S+)\))?$/, m => `收到${m[1] === 'pause' ? '暂停' : '停止'}请求：第 ${m[2]}/${m[3]} 步完成后保存恢复点并${m[1] === 'pause' ? '暂停' : '停止'}${m[1] === 'stop' && m[4] ? `（约 ${duration(m[4])}）` : ''}`],
  [/^save requested; saving a resume point after step (\d+)\/(\d+)(?: \(about (\S+)\))?$/, m => `收到保存请求：第 ${m[1]}/${m[2]} 步完成后保存恢复点${m[3] ? `（约 ${duration(m[3])}）` : ''}`],
  [/^(pause|stop) requested; stopping after the current preparation item$/, m => `收到${m[1] === 'pause' ? '暂停' : '停止'}请求：当前准备项完成后停止`],
  [/^(pausing|stopping) at step (\d+)\/(\d+) \(epoch (\S+)\); saving a resume point$/, m => `开始${m[1] === 'pausing' ? '暂停' : '停止'}：正在保存第 ${m[2]}/${m[3]} 步（第 ${m[4]} 轮）的恢复点，保存后退出`],
  [/^(pause|stop) requested at step (\d+)\/(\d+) \(epoch (\S+)\); saving a resume point$/, m => `收到${m[1] === 'pause' ? '暂停' : '停止'}信号：正在保存第 ${m[2]}/${m[3]} 步（第 ${m[4]} 轮）的恢复点，保存后退出`],
  [/^training (paused|stopped); resume point saved (\S+ \S+) \| step (\d+)\/(\d+) \| epoch (\S+) \| loss (\S+) \| (.+?)(?: \| duration (\S+))?$/, m => `${m[1] === 'paused' ? '已暂停训练' : '已停止训练'}，恢复点已保存：${m[2]} | 第 ${m[3]}/${m[4]} 步 | 第 ${m[5]} 轮 | Loss ${m[6]} | ${m[7]}${m[8] ? ` | 耗时：${m[8].replace(/s$/, '')} 秒` : ''}`],

  // Outcome
  [/^training (finished|paused|stopped|failed) at step (\d+)\/(\d+) after (\S+)$/, m => `训练${OUTCOMES[m[1]]}：第 ${m[2]}/${m[3]} 步，总用时 ${duration(m[4])}`],
  [/^training (finished|paused|stopped|failed)$/, m => `训练进程结束：${OUTCOMES[m[1]]}`],
  [/^caching (finished|paused|stopped|failed)$/, m => `缓存任务结束：${OUTCOMES[m[1]]}`],
  [/^worker stopped by an unhandled exception$/, () => '训练进程因未处理的错误而退出'],
  [/^thread (\S+) stopped by an unhandled exception$/, m => `线程 ${m[1]} 因未处理的错误而退出`],

  // Warnings
  [/^config changed since the checkpoint was written; resuming anyway$/, () => '配置与保存训练状态时不同，仍然继续训练'],
  [/^resuming a legacy checkpoint; exact RNG compatibility is not guaranteed$/, () => '正在从旧版训练状态继续，无法保证随机数序列完全一致'],
  [/^MPS training uses fp32 without autocast for numerical compatibility$/, () => 'Apple 芯片（MPS）训练使用 fp32 且不启用混合精度，以保证数值稳定'],
  [/^memory\.compile is only applied on CUDA; running eagerly$/, () => '模型编译仅在 CUDA 上生效，本次按普通模式运行'],
  [/^training state is not available during preparation; cached items are preserved$/, () => '准备阶段还没有训练状态可保存；已缓存的内容会保留'],
  [/^krea2: unsloth offloaded checkpointing is not available for this family, using block checkpointing$/, () => 'Krea 2 不支持“开启并卸载到内存”的梯度检查点，已改为普通梯度检查点'],
  [/^factor (\d+) does not divide (\d+); using the largest divisor <= \d+$/, m => `LoKr 分解因子 ${m[1]} 不能整除 ${m[2]}，改用不超过它的最大因数`],
  [/^dimension (\d+) has no useful factorization \(prime\?\); LoKr degenerates to a scaled LoRA$/, m => `维度 ${m[1]} 无法有效分解（可能是质数），该层 LoKr 退化为带缩放的 LoRA`],
  [/^ignoring (\d+) unexpected tensors in (Anima|Krea 2) checkpoint, e\.g\. (.+)$/, m => `忽略 ${m[2]} 权重文件中 ${m[1]} 个多余的张量，例如 ${m[3]}`],

  // Debug trace
  [/^phase (\w+) -> (\w+) after ([\d.]+)s$/, m => `阶段 ${m[1]} → ${m[2]}，用时 ${seconds(m[3])}`],
];

const TONES: Array<[RegExp, 'pause' | 'resume' | 'success']> = [
  [/^(pause|stop) requested\b/, 'pause'],
  [/^(pausing|stopping) at step /, 'pause'],
  [/^training (paused|stopped)\b/, 'pause'],
  [/^resume requested: /, 'resume'],
  [/^resumed training /, 'resume'],
  [/^training finished\b/, 'success'],
];

/** Pause reads in yellow, resume and successful completion in green. */
export function logTone(message: string): 'pause' | 'resume' | 'success' | null {
  const text = message.replace(/^\[rank\d+\]:\s*/, '');
  return TONES.find(([pattern]) => pattern.test(text))?.[1] ?? null;
}

// The log view reads every loaded line again whenever output arrives; each distinct line is matched once.
const translations = new Map<string, string | null>();
const TRANSLATION_CACHE_LIMIT = 20_000;

/** The Chinese reading of a fixed trainer message, or null when the line has none. */
export function translateLogMessage(message: string): string | null {
  const cached = translations.get(message);
  if (cached !== undefined) return cached;
  const translated = readLogMessage(message);
  // Lines carrying steps and timings rarely repeat, so the oldest readings make room once the cache is full.
  if (translations.size >= TRANSLATION_CACHE_LIMIT) {
    const oldest = translations.keys().next().value;
    if (oldest !== undefined) translations.delete(oldest);
  }
  translations.set(message, translated);
  return translated;
}

function readLogMessage(message: string): string | null {
  const rank = message.match(/^(\[rank\d+\]:\s*)([\s\S]*)$/);
  if (rank) {
    const translated = translateLogMessage(rank[2]);
    return translated ? rank[1] + translated : null;
  }
  for (const [pattern, render] of RULES) {
    const match = message.match(pattern);
    if (match) return render(match);
  }
  return null;
}
