import type { FamilyInfo } from '../api/types';
import { configFieldLabel, configOptionLabel } from './configPresentation';

/**
 * Settings that depend on the selected model and on the machine the service runs on. A setting neither uses is
 * hidden, options they cannot run are left out, and help describes only what applies here.
 */
export interface FieldContext {
  family?: FamilyInfo;
  config: Record<string, any>;
  english: boolean;
}

type Runtime = 'cuda' | 'hip' | 'mps' | 'cpu';
const runtimeOf = (family?: FamilyInfo) => (family?.runtime_backend ?? undefined) as Runtime | undefined;
const onGpu = (runtime?: Runtime) => runtime === 'cuda' || runtime === 'hip';
const has = (family: FamilyInfo | undefined, capability: string) => !!family?.capabilities?.includes(capability);
const modelName = (family?: FamilyInfo) => family?.name === 'sdxl' ? 'SDXL' : family?.name === 'flux2' ? 'Klein' : family?.name === 'krea2' ? 'Krea 2' : family?.name === 'anima' ? 'Anima' : family?.label || '';

/** The text encoding training really uses; the service resolves `auto` the same way. */
export function effectiveTextEncoding(family: FamilyInfo | undefined, config: Record<string, any>): 'online' | 'cached' {
  const mode = config.dataset?.text_encoding ?? 'auto';
  return config.training?.train_text_encoder || mode === 'online' || (mode === 'auto' && has(family, 'online_text')) ? 'online' : 'cached';
}

/** The noise-level sampling a new configuration of this model starts with. */
export function defaultTimestepSampling(family: FamilyInfo): string {
  return family.objective === 'ddpm' ? 'uniform' : family.sampling?.shift == null ? 'resolution_shift' : 'shift';
}

/**
 * Why the selected model or this machine does not use a setting, for the note shown while it is still set;
 * undefined when the setting applies.
 */
export function unusedSettingReason(path: string, { family, config, english }: FieldContext): string | undefined {
  if (!family) return undefined;
  const runtime = runtimeOf(family);
  const text = (zh: string, en: string) => english ? en : zh;
  const name = modelName(family);
  switch (path) {
    case 'memory.allow_tf32':
      return runtime === 'mps' || runtime === 'cpu' ? text('当前设备不使用 TF32。', 'This device does not use TF32.') : undefined;
    case 'memory.compile':
      if (!has(family, 'compile')) return text(`${name} 不支持编译模型，请关闭。`, `${name} does not support compiling the model. Turn it off.`);
      return runtime === 'mps' || runtime === 'cpu' ? text('只有 NVIDIA / 海光显卡会编译模型，当前设备直接运行。', 'Only NVIDIA / DTK GPUs compile the model; this device runs it directly.') : undefined;
    case 'memory.blocks_to_swap':
      return has(family, 'block_swap') ? undefined : text(`${name} 不支持块换出，请设为 0。`, `${name} does not support block swapping. Set it to 0.`);
    case 'memory.offload_text_encoder':
      if (config.training?.train_text_encoder) return text('训练文本编码器时不能卸载它，请关闭。', 'The text encoder cannot be offloaded while it is trained. Turn this off.');
      if (runtime === 'cpu') return text('CPU 训练时文本编码器本来就在内存里。', 'On a CPU the text encoder is already in system memory.');
      return effectiveTextEncoding(family, config) === 'cached' ? text('标签在训练前已编码，训练时不加载文本编码器。', 'Captions are encoded before training, so the text encoder is not loaded while training.') : undefined;
    case 'loop.mixed_precision':
      return runtime === 'mps' ? text('Apple 芯片训练统一使用 FP32，不使用混合精度。', 'Apple chips train in FP32 without mixed precision.') : undefined;
    default:
      return undefined;
  }
}

/** An unused setting is hidden unless it is still switched on or set (or would stop training), so it can be reset. */
export function hideUnusedSetting(path: string, value: unknown, { config }: FieldContext): boolean {
  if (path === 'memory.offload_text_encoder') return !(value && config.training?.train_text_encoder);
  if (path === 'memory.allow_tf32' || path === 'loop.mixed_precision') return true;
  return !value;
}

/** Options this model and machine can run, keeping the current value visible; undefined keeps the list as it is. */
export function contextOptions(path: string, { family }: FieldContext, options: string[] | undefined, current: unknown): string[] | undefined {
  if (!family || !options) return undefined;
  const runtime = runtimeOf(family);
  let usable: string[] | undefined;
  if (path === 'memory.base_precision' && !(has(family, 'fp8_base') && onGpu(runtime))) usable = options.filter(option => !option.startsWith('fp8'));
  // bitsandbytes 8-bit optimizers run on NVIDIA / DTK GPUs only.
  if (path === 'optimizer.type' && !onGpu(runtime)) usable = options.filter(option => !option.endsWith('8bit'));
  if (path === 'loop.distributed_strategy' && family.runtime_platform === 'windows') usable = options.filter(option => option !== 'fsdp');
  if (!usable) return undefined;
  return typeof current === 'string' && options.includes(current) && !usable.includes(current) ? [...usable, current] : usable;
}

type Lines = Array<string | false | undefined>;
const join = (lines: Lines) => lines.filter(Boolean).join('\n');

/** Help written for the selected model and this machine; undefined leaves the general text. */
export function contextHelp(path: string, context: FieldContext, options?: string[]): string | undefined {
  const { family, english } = context;
  if (!family) return undefined;
  const runtime = runtimeOf(family);
  const text = (zh: string, en: string) => english ? en : zh;
  const label = (option: string) => configOptionLabel(path, option, english);
  const listed = (option: string) => !options || options.includes(option);
  // One line per option the dropdown offers, named as the dropdown names it.
  const optionLines = (describe: Record<string, [string, string]>) => Object.entries(describe)
    .filter(([option]) => listed(option)).map(([option, [zh, en]]) => `${label(option)}${english ? `: ${en}` : `：${zh}`}`);
  const name = modelName(family);
  const ddpm = family.objective === 'ddpm';
  switch (path) {
    case 'model.attention': {
      const sdpa: [string, string] = runtime === 'cuda'
        ? ['PyTorch 内置的注意力；在 NVIDIA 显卡上会自动选用内置的 FlashAttention 或省显存内核，一般直接用它。', "PyTorch's built-in attention; on NVIDIA GPUs it picks PyTorch's own FlashAttention or memory-efficient kernel, usually the right choice."]
        : runtime === 'cpu' ? ['PyTorch 内置的注意力；CPU 训练只能用它。', "PyTorch's built-in attention; the only option for CPU training."]
          : ['PyTorch 内置的注意力，一般直接用它。', "PyTorch's built-in attention, usually the right choice."];
      return join([
        text('主模型计算注意力的方式，影响训练速度和显存占用。', 'How the main model computes attention; it affects training speed and memory use.'),
        ...optionLines({
          sdpa,
          xformers: ['Meta 的注意力加速库，省显存，速度与 SDPA 接近。', "Meta's attention library; saves memory at about the speed of SDPA."],
          flash_attn: runtime === 'hip'
            ? ['海光 DTK 版 FlashAttention，速度快、省显存。', 'The DTK build of FlashAttention; fast and memory-saving.']
            : ['独立的 FlashAttention 2，速度快、省显存，需要 RTX 30 系列（Ampere）及更新的显卡。', 'The standalone FlashAttention 2; fast and memory-saving, needs an RTX 30-series (Ampere) or newer GPU.'],
          sage: ['把注意力量化后计算，速度更快，精度略低。', 'Computes attention in quantized form; faster, slightly less precise.'],
          metal_flash: ['Apple 芯片上的 FlashAttention，只加速 FP32 主模型的注意力，其余调用自动改用 SDPA。', "FlashAttention for Apple chips; it speeds up the main model's FP32 attention and uses SDPA for anything else."],
        }),
        text('只影响主模型，文本编码器和 VAE 不受影响。', 'Affects the main model only; the text encoder and VAE are unchanged.'),
      ]);
    }
    case 'memory.allow_tf32':
      return runtime === 'hip'
        ? text('允许显卡用 TF32 计算 FP32 矩阵乘法（需显卡支持）：更快，精度略低。可复现训练会自动关闭。', 'Lets the GPU run FP32 matrix multiplications in TF32 where supported: faster, slightly less precise. Reproducible training turns it off.')
        : text('允许 NVIDIA RTX 30 系列（Ampere）及更新的显卡用 TF32 计算 FP32 矩阵乘法：更快，精度略低，对训练效果基本没有影响，建议保持开启。需要与 FP32 完全一致的结果时再关闭；更早的显卡不受影响。', 'Lets NVIDIA RTX 30-series (Ampere) and newer GPUs run FP32 matrix multiplications in TF32: faster and slightly less precise, with practically no effect on training results. Keep it on unless results must match FP32 exactly; older GPUs are unaffected.');
    case 'memory.offload_text_encoder':
      return text('当前设置会在训练时实时编码标签，文本编码器一直留在显存里。开启后每次编码完就把它移到内存：省下文本编码器占用的显存，但每步都要来回搬运，训练会变慢。显存够用时保持关闭。', 'These settings encode captions during training, so the text encoder stays in GPU memory. Turning this on moves it to system memory after each encoding: it frees that memory, but every step moves it back and forth, so training is slower. Leave it off when memory is sufficient.');
    case 'memory.compile':
      return join([
        text('用 torch.compile 编译模型：第一次启动要多等几分钟编译，之后每步更快。', 'Compiles the model with torch.compile: the first start takes a few extra minutes, later steps are faster.'),
        has(family, 'block_swap') && text(`不能与「${configFieldLabel('memory.blocks_to_swap', '')}」同时使用。`, 'Cannot be combined with block swapping.'),
      ]);
    case 'memory.blocks_to_swap':
      return join([
        text('把多少个模型块的冻结权重先放在内存里，用到时再搬回显存，默认 0 不换出。数值越大越省显存，但搬运越多、训练越慢；超过实际块数时按全部块处理。', 'How many model blocks keep their frozen weights in system memory and move to the GPU only when used. 0 (default) swaps none. Larger values save more GPU memory but move more data and train slower; values above the block count swap every block.'),
        has(family, 'compile') && onGpu(runtime) && text(`不能与「${configFieldLabel('memory.compile', '')}」同时使用。`, 'Cannot be combined with compiling the model.'),
      ]);
    case 'memory.base_precision':
      return join([
        text('降低冻结底模的存储精度来省显存：只转换加了 LoRA 的线性层，原模型文件和 LoRA 本身的精度不变。精度越低越省显存，但可能影响效果。', 'Stores the frozen base model at a lower precision to save memory. Only the linear layers that carry LoRA are converted; the model file and the LoRA weights keep their precision. Lower precision saves more memory but can affect results.'),
        listed('fp8_e4m3') && text('FP8 最省显存，在启动时量化。', 'FP8 saves the most and is quantized at start-up.'),
        text('全量微调只能选沿用加载精度或 FP32。', 'Full fine-tuning accepts only the loaded precision or FP32.'),
      ]);
    case 'memory.activation_checkpointing':
      return join([
        text('开启后每个模块只保留输入，反向传播时逐块重新计算：显存大幅减少，训练会慢一些（多一次前向计算）。可与梯度累积同时使用。', 'On keeps only each block\'s input and recomputes blocks during backward: much less memory, somewhat slower (one extra forward pass). Works with gradient accumulation.'),
        listed('unsloth') && text('开启并卸载到内存：再把这些输入暂存到内存，显存再少一些，但占用内存并增加传输；开启后显存仍不够时再用。', 'On + offload also parks those inputs in system memory for a little more saving, at the cost of RAM and transfers; use it when On is not enough.'),
      ]);
    case 'loop.mixed_precision':
      return runtime === 'cpu'
        ? text('CPU 上选 BF16 会用 BF16 计算，其他选项都按 FP32 计算。不改变权重本身的精度。', 'On a CPU, BF16 computes in BF16 and the other choices compute in FP32. Weight precision is unchanged.')
        : text('训练运算的自动混合精度，默认 BF16；FP16 需要显卡和模型支持。关闭只停用自动混合精度，不改变权重本身的精度；底模和导出文件的精度分别设置。可复现训练以参数检查显示的设置为准。', 'Automatic mixed precision for training, BF16 by default; FP16 needs GPU and model support. Turning it off disables only mixed precision, not weight precision; base-model and export precision are set separately. Reproducible training uses the values the parameter check shows.');
    case 'loop.distributed_strategy':
      return join([
        text('数据并行（DDP）：每张卡保留完整模型，最通用。', 'Data parallel (DDP): each GPU holds the whole model; works everywhere.'),
        listed('fsdp') && text('显存分片（FSDP）：把参数、梯度和优化器状态分到多张卡，适合单卡装不下的主模型；需要至少两张显卡，支持冻结文本编码器的主模型全量微调、LoRA 和 LoKr，以及 AdamW、Adafactor 或 SGD，暂不支持 FP8 底模。', 'Sharded (FSDP): splits parameters, gradients and optimizer state across GPUs, for main models too large for one card. Needs at least two GPUs and supports full fine-tuning of the main model with a frozen text encoder, LoRA and LoKr, with AdamW, Adafactor or SGD; FP8 base weights are not supported yet.'),
        family.runtime_platform === 'windows' && text('Windows 只支持数据并行。', 'Windows supports data parallel only.'),
      ]);
    case 'adapter.algo':
      return join([
        text('附加权重的结构，决定参数量和表达能力：', 'The structure of the added weights; it sets their parameter count and capacity:'),
        ...optionLines({
          lora: ['两个低秩矩阵相乘，最常用，兼容性最好。', 'two low-rank matrices multiplied; the most common and most widely supported.'],
          lokr: ['用 Kronecker 积组合两个小矩阵，参数通常最少，文件最小。', 'two small matrices combined by a Kronecker product; usually the fewest parameters and the smallest file.'],
          loha: ['两组低秩矩阵逐元素相乘，同样的秩下表达能力更强，参数约为 LoRA 的两倍。', 'two low-rank pairs multiplied element by element; more capacity at the same rank, with about twice the parameters of LoRA.'],
          ortho: ['在底模权重最主要的几个方向上做正交旋转和缩放，可训练参数很少，训练稳定；导出为普通 LoRA。', "rotates and rescales the base weight's main directions; very few trained parameters and steady training, exported as a plain LoRA."],
          tlora: ['LoRA 的变体：噪声越大可用的秩越少，减轻小数据集的过拟合；导出为普通 LoRA。', 'a LoRA whose usable rank shrinks as the noise grows, which curbs overfitting on small datasets; exported as a plain LoRA.'],
        }),
        text('Rank 里的 full 只对 LoKr 生效：保留完整的 Kronecker 因子，仍然是 LoKr。', 'full in Rank applies to LoKr only: it keeps the whole Kronecker factors and is still a LoKr adapter.'),
      ]);
    case 'scheduler.type':
      return join([
        text('学习率随训练步数变化的曲线：', 'How the learning rate changes over the run:'),
        ...optionLines({
          constant: ['始终保持设定的学习率。', 'keeps the set learning rate.'],
          linear: ['从设定值匀速降到最低学习率比例。', 'falls at a steady rate to the minimum ratio.'],
          cosine: ['沿余弦曲线降低，开头和结尾变化慢；默认。', 'falls along a cosine curve, slowly at the start and the end; the default.'],
          cosine_restarts: ['按周期数重复余弦下降，每个周期开始时回到设定值。', 'repeats the cosine fall for the set number of cycles, returning to the set rate at each start.'],
          polynomial: ['按多项式的幂降低，幂为 1 时与线性相同。', 'falls along a polynomial curve; a power of 1 equals linear.'],
          warmup_stable_decay: ['预热后保持设定值，最后一段再降低。', 'holds the set rate after warm-up and falls only in the final stretch.'],
          rex: ['前期降得慢，临近结束时降得快。', 'falls slowly at first and quickly near the end.'],
        }),
        text('免调度优化器固定使用恒定。', 'Schedule-free optimizers always use Constant.'),
      ]);
    case 'sampling.scheduler':
      return join([
        text(`预览时各步噪声强度的取法，只影响预览图；都会按「${configFieldLabel('sampling.shift', '')}」调整：`, 'How the noise levels of the preview steps are chosen; previews only, and every choice applies the sampling shift:'),
        ...optionLines({
          uniform: ['在整个噪声范围内均匀取点；默认。', 'evenly spaced over the whole noise range; the default.'],
          simple: ['从 1000 级离散噪声表中等间隔取点。', 'evenly spaced picks from the 1000-level discrete noise table.'],
          sgm_uniform: ['从最高噪声到最低一级噪声均匀取点，不含最低一级。', 'evenly spaced from the highest noise towards the lowest discrete level, leaving that level out.'],
          normal: ['与 SGM 相同，但包含最低一级。', 'as SGM Uniform, but including the lowest discrete level.'],
        }),
      ]);
    case 'optimizer.type':
      return join([
        text('优化器决定每一步怎样用梯度更新参数，不同优化器对学习率的要求和显存占用不同。', 'The optimizer decides how each step turns gradients into weight updates; optimizers differ in the learning rate they need and the memory they use.'),
        ...optionLines({
          adamw: ['默认选择，稳定通用。', 'The default; stable and general-purpose.'],
          adam: ['经典 Adam，权重衰减的算法与 AdamW 不同，一般选 AdamW。', 'Classic Adam; it applies weight decay differently from AdamW, which is usually preferred.'],
          sgd: ['最省显存，但对学习率很敏感，训练 LoRA 很少用。', 'Uses the least memory but is very sensitive to the learning rate; rarely used for LoRA.'],
          adamw8bit: ['AdamW 的 8 位版本，优化器状态的显存约为原来的四分之一，效果接近。', 'AdamW with 8-bit state: about a quarter of the optimizer-state memory, with similar results.'],
          lion: ['只按梯度方向更新，状态比 AdamW 少一半；学习率通常设为 AdamW 的 1/10 到 1/3。', 'Updates by the sign of the gradient and keeps half the state of AdamW; use about 1/10 to 1/3 of the AdamW learning rate.'],
          lion8bit: ['Lion 的 8 位版本，更省显存。', 'Lion with 8-bit state; uses less memory.'],
          prodigy: ['自动估计学习率，学习率保持 1 即可。', 'Estimates the learning rate itself; keep the learning rate at 1.'],
          prodigy_plus_sf: ['Prodigy 的改进版，自动估计学习率，也不需要学习率调度器。', 'An improved Prodigy that estimates the learning rate and needs no learning-rate schedule.'],
          adafactor: ['只保存分解后的二阶统计量，显存占用低。', 'Keeps factored second-moment statistics only; low memory use.'],
          came: ['在 Adafactor 的基础上加入置信度修正，省显存且更稳定。', 'Adafactor with a confidence correction; low memory and more stable.'],
          adamw_sf: ['不需要学习率调度器的 AdamW。', 'AdamW that needs no learning-rate schedule.'],
          automagic: ['为每个参数自动调节学习率，显存占用低。', 'Adjusts the learning rate of every parameter automatically; low memory use.'],
        }),
      ]);
    case 'dataset.num_workers':
      return family.runtime_platform === 'windows'
        ? text('读取图片的后台进程数。Windows 上建议保持 0，多进程容易卡住或占用大量内存。', 'Background processes that load images. Keep 0 on Windows, where extra processes tend to hang or use a lot of memory.')
        : text('读取图片的后台进程数，0 表示在训练进程里读取。数据读取跟不上训练时再调高。', 'Background processes that load images; 0 loads them in the training process. Raise it when loading cannot keep up with training.');
    case 'dataset.text_encoding':
      if (!has(family, 'online_text')) return text(`${name} 在训练前把所有标签编码并缓存，训练时不加载文本编码器。修改标签后会重新编码。`, `${name} encodes and caches every caption before training, so the text encoder is not loaded while training. Changed captions are encoded again.`);
      return join([
        text(`自动：${name} 默认在训练时编码，打乱标签顺序、随机丢弃标签等每步都会变化的标签需要这样。`, `Auto: ${name} encodes during training by default, which captions that change every step (shuffled order, dropped tags) need.`),
        text('训练前缓存：训练前一次性编码并保存，训练时不占文本编码器的显存；标签的随机变化只能使用预先生成的几个版本。', 'Cached: encodes once before training and saves the result, so training needs no text-encoder memory; caption randomness is limited to a few pre-generated variants.'),
        text('每步编码：每一步都重新编码，文本编码器一直占用显存（可开启「卸载文本编码器」）。', 'Online: encodes at every step; the text encoder stays in GPU memory (see Offload text encoder).'),
      ]);
    case 'dataset.caption.cache_variants':
      return text('标签在训练前缓存时，每张图预先生成几个随机版本（打乱顺序、丢弃标签、通配符），训练时轮流使用。只有开启这些随机变化时才有意义。', 'When captions are cached before training, each image gets this many pre-generated random versions (shuffled order, dropped tags, wildcards) that training rotates through. Useful only when those variations are on.');
    case 'objective.timestep_sampling': {
      const describe: Record<string, [string, string]> = ddpm ? {
        uniform: ['0–999 的每个时间步出现机会相同。', 'every timestep from 0 to 999 is equally likely.'],
        logit_normal: ['多抽中间的时间步，两端较少。', 'favours the middle timesteps, fewer at both ends.'],
      } : {
        uniform: ['各种噪声强度出现的机会相同。', 'every noise level is equally likely.'],
        logit_normal: ['多抽中等噪声，很低和很高的噪声较少。', 'favours medium noise, fewer very low or very high levels.'],
        shift: ['在逻辑正态的基础上偏向高噪声，偏移量见下方「时间步偏移」。', 'logit-normal pushed towards high noise by the shift set below.'],
        resolution_shift: ['偏移量随图像大小自动调整，图越大越偏向高噪声。', 'the shift follows the image size: larger images lean towards higher noise.'],
        mode: ['同样偏向中等噪声，但两端比逻辑正态保留更多，形状由下方系数控制。', 'also favours medium noise but keeps more at both ends than logit-normal; the coefficient below sets the shape.'],
        cosmap: ['中间多、两端少，比逻辑正态平缓。', 'more in the middle, fewer at the ends, flatter than logit-normal.'],
      };
      return join([
        text('训练时每一步随机抽一个噪声强度（t 越大噪声越多），这里决定怎么抽：', 'Each training step draws a noise level (higher t means more noise). This sets how it is drawn:'),
        ...optionLines(describe),
        text(`${name} 默认使用 ${label(defaultTimestepSampling(family))}，一般不用修改。预览图使用下方单独的采样设置。`, `${name} defaults to ${label(defaultTimestepSampling(family))}; there is usually no need to change it. Previews use their own sampling settings below.`),
      ]);
    }
    case 'objective.weighting': {
      const describe: Record<string, [string, string]> = {
        none: ['所有噪声强度一样重要。', 'every noise level counts the same.'],
        sigma_sqrt: ['低噪声（细节）的步权重更大。', 'low-noise (detail) steps weigh more.'],
        cosmap: ['中等噪声的步权重更大。', 'medium-noise steps weigh more.'],
        snr_like: ['降低高噪声步的权重，程度由下方 Gamma 控制。', 'lowers the weight of high-noise steps; Gamma below sets how much.'],
        cosmos: ['很低和很高噪声的步权重略大。', 'very low and very high noise weigh slightly more.'],
        min_snr: ['限制低噪声步的权重，让各时间步更均衡，常用来加快收敛；强度由下方 Gamma 控制。', 'caps the weight of low-noise steps so timesteps are more balanced, often for faster convergence; Gamma below sets the strength.'],
      };
      return join([
        text('给不同噪声强度的损失乘上权重，改变训练重点，不改变抽样：', 'Multiplies the loss by a weight per noise level. It changes what training emphasises, not which levels are drawn:'),
        ...optionLines(describe),
        text('不确定时保持不加权。', 'When unsure, keep None.'),
      ]);
    }
    case 'objective.snr_gamma':
      return ddpm
        ? text('Min-SNR 的截断值，默认 5：越小越压低低噪声步的权重。按实际噪声调度和预测方式计算，只在选择最小信噪比加权时生效。', 'The Min-SNR cap, 5 by default: smaller values lower low-noise steps further. Computed from the real noise schedule and prediction type; used only with Min-SNR weighting.')
        : text('类信噪比加权的截断值，默认 5：越小，高噪声步的权重降得越多。只在选择类信噪比加权时生效，不改变时间步抽样。', 'The SNR-like cap, 5 by default: smaller values lower high-noise steps further. Used only with SNR-like weighting; sampling is unchanged.');
    case 'objective.res_shift_tokens':
      return text(`按图像 token 数插值偏移量的两个参考点，${name} 默认 256 和 ${family.name === 'krea2' ? 6400 : 4096}。它们不是图像尺寸上限，范围外会外推；与下方 mu 成对使用。`, `The two image token counts the shift is interpolated between; ${name} uses 256 and ${family.name === 'krea2' ? 6400 : 4096}. They are not size limits (values outside extrapolate) and pair with the mu values below.`);
    case 'sampling.sampler': {
      const describe: Record<string, [string, string]> = {
        euler: ['每步计算一次，最快，适合日常预览。', 'one evaluation per step, the fastest; good for routine previews.'],
        heun: ['先预测再校正，每步多算一次，更精细但更慢。', 'predicts then corrects, one extra evaluation per step: finer but slower.'],
        er_sde: ['结合前几步结果和随机噪声，阶数和噪声强度见下方。', 'combines earlier steps with random noise; order and noise strength are below.'],
      };
      return join([
        text('只影响预览图，不影响训练：', 'Affects previews only, not training:'),
        ...optionLines(describe),
      ]);
    }
    case 'sampling.steps':
    case 'sampling.cfg':
    case 'sampling.shift': {
      const sampling = family.sampling || {};
      const key = path.split('.')[1] as 'steps' | 'cfg' | 'shift';
      const fallback = sampling[key] == null
        ? key === 'shift' && family.objective !== 'ddpm' ? text('按图像大小计算', 'computed from the image size') : text('读取所选模型的配置', 'read from the selected model')
        : String(sampling[key]);
      const general = key === 'steps' ? text('生成预览图的步数，越多越慢；单条提示词里的设置优先。', 'Steps per preview; more is slower. A prompt\'s own value takes priority.')
        : key === 'cfg' ? text('提示词引导强度：1 只用正向提示词，提高会放大正负提示词的差异；单条提示词里的设置优先。', 'Prompt guidance: 1 uses the positive prompt only, higher values widen the gap between positive and negative prompts. A prompt\'s own value takes priority.')
          : text('预览时噪声的偏移：1 不偏移，越大越偏向高噪声，与训练里的偏移分开设置。', 'Noise shift for previews: 1 applies none, larger values lean towards high noise. Separate from the training shift.');
      return join([general, text(`留空使用 ${name} 的默认值：${fallback}。`, `Leave blank for the ${name} default: ${fallback}.`)]);
    }
    case 'training.train_backbone':
      return family.name === 'sdxl'
        ? text('训练生成图像的主模型，SDXL 为 UNet。全量微调时包括卷积、归一化、嵌入与偏置，不局限于线性层。', 'Trains the model that generates images, the UNet for SDXL. Full fine-tuning includes convolutions, norms, embeddings and biases, not only linear layers.')
        : text(`训练生成图像的主模型，${name} 为 DiT。全量微调时包括归一化、嵌入与偏置，不局限于线性层。`, `Trains the model that generates images, the DiT for ${name}. Full fine-tuning includes norms, embeddings and biases, not only linear layers.`);
    case 'training.train_text_encoder':
      return join([
        family.name === 'sdxl'
          ? text('同时训练 SDXL 的两个文本编码器 CLIP-L 和 CLIP-G。', 'Also trains both SDXL text encoders, CLIP-L and CLIP-G.')
          : text(`同时训练 ${name} 的文本编码器。`, `Also trains the ${name} text encoder.`),
        text('LoRA 只更新其线性层的附加权重，全量微调更新其原始参数。每步都要重新编码标签，不使用文本缓存，显存需求增加。', 'LoRA updates only the added weights of its linear layers; full fine-tuning updates its original parameters. Captions are encoded at every step without the text cache, which needs more memory.'),
      ]);
    case 'model.tokenizer_path':
      return family.name === 'anima'
        ? text('可选：指定旧版 T5 spiece 分词器目录。留空使用模型目录或内置分词器。', 'Optional: a legacy T5 spiece tokenizer directory. Leave blank to use the model directory or the built-in tokenizer.')
        : family.name === 'sdxl' ? undefined
          : text('可选：自定义分词器目录。留空使用模型目录或内置分词器。', 'Optional: a custom tokenizer directory. Leave blank to use the model directory or the built-in tokenizer.');
    default:
      return undefined;
  }
}
