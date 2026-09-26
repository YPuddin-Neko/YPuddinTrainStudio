/**
 * Chinese readings of the trainer's fixed log lines. The log file stays in English;
 * a line without a rule is shown as written.
 */

type Rule = [RegExp, (match: RegExpMatchArray) => string];

const FAMILIES: Record<string, string> = { anima: 'Anima', krea2: 'Krea 2', flux: 'FLUX.1', flux2: 'FLUX.2', sdxl: 'SDXL', toy: '测试模型' };
const SAVED: Record<string, string> = { weights: '权重', 'training state': '训练状态', model: '模型', file: '文件' };
const OUTCOMES: Record<string, string> = { finished: '完成', paused: '已暂停', stopped: '已停止', failed: '失败' };

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

const RULES: Rule[] = [
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
  [/^VAE encoding: (\d+)\/(\d+)$/, m => `VAE 编码中：${m[1]}/${m[2]}`],
  [/^text encoding: (\d+)\/(\d+)$/, m => `文本编码中：${m[1]}/${m[2]}`],
  [/^cached (\d+) latents in ([\d.]+)s$/, m => `VAE 编码完成：新缓存 ${m[1]} 张图片的潜空间，用时 ${seconds(m[2])}`],
  [/^cached (\d+) latents$/, m => `VAE 编码完成：新缓存 ${m[1]} 张图片的潜空间`],
  [/^all latents were already cached$/, () => 'VAE 编码：所有图片都已有缓存，无需重新编码'],
  [/^all text encodings were already cached$/, () => '文本编码：所有标注都已有缓存，无需重新编码'],
  [/^cached (\d+) text encodings \((\d+) distinct captions\) in ([\d.]+)s$/, m => `文本编码完成：新缓存 ${m[1]} 条（${m[2]} 条不同的标注），用时 ${seconds(m[3])}`],
  [/^skipping unreadable image (.+) \((.+)\)$/, m => `跳过无法读取的图片 ${m[1]}（${m[2]}）`],

  // Training setup
  [/^injected adapters into (\d+) layers \(([^)]*)\): ([\d,]+) trainable parameters$/, m => `已注入适配器：${m[1]} 层（${adapterKinds(m[2])}），可训练参数 ${m[3]} 个`],
  [/^injected adapters: \{'layers': (\d+), 'trainable_params': (\d+)/, m => `已注入适配器：${m[1]} 层，可训练参数 ${Number(m[2]).toLocaleString('en-US')} 个`],
  [/^optimizer (\S+): (\d+) parameter groups, learning rates (.+), weight decay (\S+)$/, m => `优化器 ${m[1]}：${m[2]} 个参数组，学习率 ${m[3]}，权重衰减 ${m[4]}`],
  [/^scheduler managed by optimizer: warmup (\S+), (\d+) total steps$/, m => `学习率由优化器自行调度：预热 ${m[1]}，共 ${m[2]} 步`],
  [/^scheduler (\S+): warmup (\S+), (\d+) total steps$/, m => `学习率调度 ${m[1]}：预热 ${m[2]}，共 ${m[3]} 步`],
  [/^training (\d+) steps: (\d+) per epoch, effective batch (\d+) \((\d+) x (\d+) accumulation x (\d+) GPU\)$/, m => `开始训练：共 ${m[1]} 步，每轮 ${m[2]} 步，等效批次 ${m[3]}（批次 ${m[4]} × 梯度累积 ${m[5]} × ${m[6]} 张 GPU）`],
  [/^resuming at step (\d+)\/(\d+)$/, m => `从第 ${m[1]} 步继续训练（共 ${m[2]} 步）`],
  [/^preview seed (\d+) \(random for this run\)$/, m => `预览图种子 ${m[1]}（本次训练随机生成，所有预览图共用）`],
  [/^preview seed (\d+)$/, m => `预览图种子 ${m[1]}`],

  // Progress
  [/^step (\d+)\/(\d+) \| epoch (\S+) \| loss (\S+) \| avg loss (\S+) \| lr (.+?) \| grad norm (\S+) \| (\S+) it\/s \| eta (\S+)$/, m => `步数 ${m[1]}/${m[2]} | 轮次 ${m[3]} | Loss ${m[4]} | 平均 Loss ${m[5]} | 学习率 ${m[6]} | 梯度范数 ${m[7]} | ${m[8]} it/s | 剩余 ${m[9] === '-' ? '—' : duration(m[9])}`],
  [/^epoch (\d+) finished at step (\d+)$/, m => `第 ${m[1]} 轮完成（第 ${m[2]} 步）`],
  [/^sampling (\d+) previews at step (\d+) \(seed (\d+)\)$/, m => `第 ${m[2]} 步：开始生成 ${m[1]} 张预览图（种子 ${m[3]}）`],
  [/^preview (\d+)\/(\d+) saved: (\S+) \((\d+)x(\d+), seed (\d+), ([\d.]+)s\)$/, m => `预览图 ${m[1]}/${m[2]} 已保存：${m[3]}（${m[4]}×${m[5]}，种子 ${m[6]}，用时 ${seconds(m[7])}）`],
  [/^previews finished in ([\d.]+)s$/, m => `预览图生成完成，用时 ${seconds(m[1])}`],
  [/^saved (weights|training state|model|file)( \(EMA\))?: (.+)$/, m => `已保存${m[2] ? ' EMA ' : ''}${SAVED[m[1]]}：${m[3]}`],

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

/** The Chinese reading of a fixed trainer message, or null when the line has none. */
export function translateLogMessage(message: string): string | null {
  for (const [pattern, render] of RULES) {
    const match = message.match(pattern);
    if (match) return render(match);
  }
  return null;
}
