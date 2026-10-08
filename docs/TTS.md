# 语音训练

语音训练支持 **VoxCPM 1.5 LoRA** 和 **GPT-SoVITS v5dev / v5turbo**，按语音项目和版本组织配置与数据。创建项目时选择语音类型及训练引擎；每个版本保存自己的环境、模型、参数和数据来源。训练与试听进入任务队列，共用 GPU 调度、日志和训练曲线。

| 引擎 | 训练方式 | 训练精度 | 录音格式 |
| --- | --- | --- | --- |
| VoxCPM 1.5 | LM、DiT、投影层的 LoRA | BF16 | 单声道 44100 Hz PCM WAV |
| GPT-SoVITS v5 | GPT 全参微调、SoVITS LoRA，可单独或顺序执行 | GPT：FP16 混合精度或 FP32；SoVITS：FP16 或 FP32 | 单声道 32000、44100 或 48000 Hz PCM WAV |

两种引擎都使用单张 NVIDIA CUDA GPU、独立 Python 环境、本地官方源码与本地模型。模型可在训练器中下载，也可选择已有本地目录；语音依赖安装到独立环境中。当前不提供多卡、暂停恢复或手动保存；「重试」创建新任务，从该任务记录的初始权重重新执行。两种引擎的参数和检查点格式各自独立，切换版本引擎后须重新检查数据。

## 下载模型

在模型管理中选择 VoxCPM 1.5、GPT-SoVITS v5dev 或 v5turbo，下载完整模型包。下载使用设置中的模型目录、Hugging Face 凭据和代理；进度与取消也可在后台任务中心查看。网络文件全部接收后还会解压和校验，完成后才能选择到语音版本的配置草稿中，再保存配置。

GPT-SoVITS 包包含所选变体的权重、GPT 权重、声码器、BERT 模型及词表、HuBERT、G2PW 和语言识别模型。不同变体共有的已下载文件可经校验后复制使用，减少再次联网下载。模型包使用固定官方版本及文件摘要；已有目标目录不会被覆盖。失败或取消后可重试，重试沿用原下载目录。修改模型目录设置不移动已下载的模型。

模型包与运行环境分别准备。模型下载不会安装独立 Python、官方训练源码、CUDA 版 PyTorch 或 NLTK/OpenJTalk 环境资源；对应要求见下文。已有本地模型可以继续使用，无需再次下载。

## 模型类型与参数预设

版本设置中可以重新选择模型类型。切换 VoxCPM 与 GPT-SoVITS 时使用新引擎的默认参数，并清空原 Python、源码和模型路径；数据登记保留，需要按新引擎重新检查。GPT-SoVITS 同一引擎切换 dev/turbo 时保留训练参数、Python 和源码路径，清除旧模型绑定，重新选择对应变体的权重。已有训练和试听任务始终使用创建时保存的配置。

TTS 参数预设按训练引擎管理，可以把当前参数另存为预设、载入到草稿、重命名、删除或导入导出 JSON。载入后检查并保存版本配置。预设只携带可复用训练参数，不包含本机环境和模型路径、数据来源或任务输出；GPT-SoVITS 的执行阶段属于预设，dev/turbo 变体由当前版本保留。不同引擎的预设不能互相载入。

## VoxCPM 1.5

### 准备运行环境

1. 创建独立 Python 3.10 或 3.11 环境，安装与 NVIDIA 驱动匹配的 CUDA 版 PyTorch 2.5 或更高版本及对应 torchaudio。环境中还需要 VoxCPM 的依赖、可加载的 TorchCodec 音频解码后端和 `tensorboardX`。安装方法参见 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 与 [VoxCPM 官方项目](https://github.com/OpenBMB/VoxCPM)。
2. 把 VoxCPM 放在独立源码目录，切换到本接入使用的提交：

   ```bash
   git clone https://github.com/OpenBMB/VoxCPM.git /path/to/VoxCPM
   git -C /path/to/VoxCPM checkout f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629
   /path/to/tts-venv/bin/python -m pip install -e /path/to/VoxCPM
   /path/to/tts-venv/bin/python -m pip install tensorboardX
   ```

   Windows 中将 Python 路径替换为独立环境的 `Scripts/python.exe`。不要在该源码目录里修改训练文件；版本或源码不符时，配置检查会说明原因。
3. 在训练器中下载并选择 VoxCPM 1.5 模型包。若使用已有的 [openbmb/VoxCPM1.5 模型](https://huggingface.co/openbmb/VoxCPM1.5)，模型目录需包含 `config.json`、词表相关文件、主模型权重和 AudioVAE 权重，例如 `model.safetensors`、`audiovae.safetensors`、`tokenizer.json`、`tokenizer_config.json`。也可使用该模型原有的 `pytorch_model.bin`、`audiovae.pth` 权重。
4. 在语音版本的训练参数中填写独立环境的 Python、VoxCPM 源码目录和模型目录，使用服务器上的绝对路径。配置检查会通过这个 Python 检查依赖、所选显卡的 CUDA 与 BF16 能力、分词器和实际 LoRA 目标；任务取得 GPU 后会再次检查实际分配的显卡。更换显卡后重新检查。

VoxCPM 1.0、VoxCPM 2、ONNX、GGUF 或其他量化推理导出不能作为这里的底模。VoxCPM 1.5 的 AudioVAE 采样率为 44100 Hz，模型配置和训练入口必须匹配。

### 准备录音与转写

本接入接受 **单声道、44100 Hz、PCM WAV**。请先把素材切成独立语句，校对逐句转写，去掉背景音乐、重叠说话和明显噪声。清单中的文本应与录音实际说出的内容一致。

使用 UTF-8 编码的 JSONL 清单，每行一个对象：

```jsonl
{"audio":"wavs/0001.wav","text":"今天的天气很好。"}
{"audio":"wavs/0002.wav","text":"我们下次再见。"}
```

`audio` 可以是绝对路径，也可以相对于 JSONL 所在目录。可选 `ref_audio` 指向同一说话人的参考录音；可选 `dataset_id` 为非负整数。程序会读取完整音频，核对通道、采样率和帧数据；已有 `duration` 值会按音频重新计算。

验证集使用相同格式，可留空。验证集中的句子用于观察模型对未参与训练内容的表现。短于 1 秒或长于 30 秒的录音会给出提示；较长录音会增加显存占用。批大小不能大于训练样本数。

登记清单后执行检查，可以按页查看每一行的录音、转写和问题。完整扫描即使全部行无效，也保留可浏览结果；文件无法读完时显示检查错误。短时长等警告不减少有效行数。移除登记只移除该版本的引用，保留原清单和音频。

任务入队时会保存规范化清单，音频路径转换为绝对路径；原清单和原音频不被覆盖。任务保存音频、模型配置、分词器和权重的内容身份，启动时发现变化会拒绝执行。清单仍引用原始录音和本地底模，因此排队和训练期间应保留这些文件及其内容。

### 训练参数

| 参数 | 含义 |
| --- | --- |
| 批大小 | 每个微批的录音数量。录音时长也会影响显存占用。 |
| 梯度累积 | 每次优化器更新累计多少个微批；单卡每次更新使用约「批大小 × 梯度累积」条录音。 |
| 数据加载进程 | 并行读取数据的进程数；0 表示在训练进程内加载。 |
| 预处理进程 | 数据清单预处理使用的进程数，至少为 1。 |
| 每批 token 上限 | 0 关闭长度过滤。正值按「上限 ÷ 批大小」向下取整，使用固定训练入口的分词和序列长度算法过滤训练样本；不用于过滤验证集。 |
| 迭代次数 | 优化器更新次数，不是遍历数据集的轮数。 |
| 保存间隔 | 两次定期保存之间的迭代数。上游还会在首次更新和训练结束时保存。 |
| 学习率 | 参数更新步长，支持 `1e-4` 等科学计数法。 |
| 预热步数 | 逐步提高学习率的更新次数，0 表示不预热；不能超过迭代次数或解析后的调度步数。 |
| 权重衰减 | AdamW 的权重衰减系数，0 关闭。 |
| 梯度范数上限 | 梯度裁剪阈值，0 关闭。 |
| 调度步数 | 余弦学习率调度使用的计划步数；0 使用迭代次数。它不改变训练停止的迭代次数。 |
| 验证间隔 | 两次验证之间的更新次数；留空跟随保存间隔。没有验证清单时不执行验证。 |
| 日志间隔 | 两次训练指标日志之间的更新次数；不改变保存间隔或停止次数。 |
| 扩散损失权重 / 停止损失权重 | 分别设置 `loss/diff` 与 `loss/stop` 在总损失中的系数。 |
| LoRA rank | 适配器矩阵的秩，会影响参数量与显存占用。 |
| LoRA alpha | LoRA 的缩放参数，与 rank 一起决定适配器缩放。 |
| LoRA dropout | LoRA 分支的 dropout 概率，0 关闭。 |
| LM / DiT / 投影层 LoRA | 分别启用三个组件；至少启用一个。每个组件有独立的目标模块列表，检查时核对实际可训练参数。 |

默认启用 LM 和 DiT 的 LoRA，投影层关闭。训练采用固定入口的 AdamW 和带预热的余弦调度，优化器类型、调度器类型、训练随机种子和检查点保留数量不开放配置。所有 29 个配置字段及默认值、范围和有序分组由 `GET /api/tts/schema/train?engine=voxcpm1.5` 返回；固定项和未接入能力也可通过 `GET /api/tts/capabilities` 查看。

曲线中的 Loss 是最后一个微批各损失项按配置权重求和的结果；启用梯度累积时，它不是所有微批损失的平均值。验证曲线沿用上游最多 10 个验证批次的总损失均值。训练会丢弃不足一个完整批次的尾批，验证保留尾批；配置检查分别返回过滤前后数量、完整批次、尾批丢弃数和实际验证上限。Epoch 沿用上游按已消费微批数估计的数据遍历量。日志中的学习率始终使用科学计数法，结构化记录保留原始数值。

保存配置与检查数据是两次独立操作：配置使用 `revision`，来源登记与内容身份使用 `data_revision`。校验和启动只使用已保存的版本；旧修订会返回冲突。启动请求须携带 UUID 格式的 `Idempotency-Key`；同一请求重放返回原任务，使用新 UUID 才表示另建任务。

### 检查点与试听

每个训练任务使用独立输出目录，检查点保存在其中的 `voxcpm/step_*/`。目录包含 `lora_config.json` 和 `lora_weights.safetensors` 或 `lora_weights.ckpt`；上游还会保存优化器与调度器文件。这些训练状态不能通过本接入恢复。检查点显示的步数取实际保存事件，不从目录名推算；缺少旧事件时步数留空。不同目录即使对应同一次更新，也保留各自身份。

训练期间和训练结束后，可以从任务的完整检查点创建试听任务，输入要合成的文本。失败或取消的训练只要留下完整检查点，也可用于试听。可同时提供 `reference_audio` 和 `reference_text`，参考音频同样需要单声道 44100 Hz PCM WAV。试听的 seed、CFG 和推理步数分别控制随机生成、引导强度和推理迭代次数。

试听使用来源训练任务记录的本地基础模型和选定 LoRA 检查点，不会按检查点里的地址下载模型。基础模型路径须与检查点记录一致。生成的 WAV 与来源任务、检查点、文本、请求 seed、实际 seed、CFG、推理步数、时长和采样率一起记录。上游可能在重试生成时递增 seed，因此实际 seed 可能与请求不同；旧记录缺少实际值时保持为空。排队和失败的试听也保留任务与请求信息，没有音频时不会生成下载地址。已登记音频丢失后保留记录，并显示文件不可用。

通过 API 创建试听时，提交检查点的 `checkpoint_id` 与 `checkpoint_revision`，并携带 UUID `Idempotency-Key`。检查点变化、来源归档、项目或版本不可写时拒绝创建。检查点文件下载只接受已登记的文件 ID；试听不会读取客户端指定的任意检查点目录。

取消任务会停止当前训练或试听；已经保存的检查点保留。取消时上游可能额外保存当前状态，但任务仍显示为「已取消」。若训练进程提前结束，即使退出码是 0，也不会显示为训练完成。

## GPT-SoVITS v5

### 准备运行环境与模型

使用独立的 Python 环境，安装与显卡驱动匹配的 CUDA 版 PyTorch、对应 torchaudio 及固定官方源码的依赖，并准备 FFmpeg。通用安装可使用 Python 3.10 或 3.11；Windows 的 Python 3.12 / CUDA 12.8 环境按 [固定上游的 Windows 安装清单](https://github.com/RVC-Boss/GPT-SoVITS/blob/f652b1da5af29a6955f9c3911aa71b7daa6618bc/requirements-py312-win-flash_attention/requirements_py312_win_cu128_overseas.txt) 使用 `torch==2.7.1+cu128` 与 `torchaudio==2.7.1+cu128`。环境安装参照 [固定版本的官方说明](https://github.com/RVC-Boss/GPT-SoVITS/blob/f652b1da5af29a6955f9c3911aa71b7daa6618bc/README.md) 和 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)。此接入要求 CUDA；不会自动改用 CPU、MPS 或 DTK。

Windows 的 GPT 阶段即使单卡也需要可用的 Gloo。PyTorch 能识别 CUDA 显卡，并不表示该构建能初始化 Gloo；出现 `unsupported gloo device` 时，请核对所选独立环境中的 PyTorch 版本与上述安装清单。

使用官方 `cuda_graph_accel_v5` 分支的固定提交：

```bash
git clone --branch cuda_graph_accel_v5 https://github.com/RVC-Boss/GPT-SoVITS.git /path/to/GPT-SoVITS
git -C /path/to/GPT-SoVITS checkout f652b1da5af29a6955f9c3911aa71b7daa6618bc
/path/to/gpt-sovits-venv/bin/python -m pip install -r /path/to/GPT-SoVITS/requirements.txt
/path/to/gpt-sovits-venv/bin/python -m pip install 'resampy>=0.4.2,<0.5'
```

试听重采样需要 `resampy`；固定上游的 Windows 安装清单已包含它，使用通用 `requirements.txt` 时须按上述命令补装。Windows 使用对应的安装清单替代通用依赖安装命令，并保留清单中的 PyTorch 版本。

保留这个提交的已跟踪源码，不修改训练脚本或配置模板。填写 `trainer_path` 时指向 Git 仓库根目录；`python_path` 指向独立环境的 Python，Windows 对应 `Scripts/python.exe`。任务会在自己的目录中使用源码副本与预处理产物。

在训练器中下载所选 v5 变体的模型包，并把包目录选为 `model_path`。若使用原有官方目录，`model_path` 指向 **`GPT_SoVITS/pretrained_models` 根目录**，不指向其中的 `gsv-v5-pretrained`。模型根目录需要以下文件；主权重来自 [官方模型仓库](https://huggingface.co/lj1995/GPT-SoVITS)，v5 权重见 [gsv-v5-pretrained 目录](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v5-pretrained)。

| 相对 `model_path` 的位置 | 用途 |
| --- | --- |
| `s1v3.ckpt` | 两个 v5 变体共用的 GPT 初始权重。 |
| `gsv-v5-pretrained/s2Gv5dev.pth` 或 `s2Gv5turbo.pth` | 所选变体的完整 SoVITS 底模。使用哪种变体就准备相应文件。 |
| `gsv-v5-pretrained/vocoder.pth` | 48 kHz 试听的声码器。 |
| `chinese-roberta-wwm-ext-large/` | BERT 模型目录，至少包含 `config.json`、`vocab.txt` 和 `pytorch_model.bin` 或 `model.safetensors`，并保留官方分词器文件。 |
| `chinese-hubert-base/` | HuBERT 模型目录，包含 `config.json` 和 `pytorch_model.bin` 或 `model.safetensors`。 |

训练器下载包中的 `G2PWModel/` 已包含中文需要的 `g2pW.onnx`、`MONOPHONIC_CHARS.txt`、`POLYPHONIC_CHARS.txt`、`bopomofo_to_pinyin_wo_tune_dict.json`、`char_bopomofo_dict.json` 及官方配置。训练器下载的完整模型包只使用包内配套资源；`G2PWModel/` 或 `fast_langdetect/` 整个目录缺失时也会报错，不从源码目录补用另一套资源。旧本地布局仍可把完整 [G2PW 模型包](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip) 解压到源码目录的 `GPT_SoVITS/text/G2PWModel/`；若模型根目录已有 `G2PWModel/`，则使用该目录，缺件时不混用其他目录的文件。

非纯英文试听还需要 `fast_langdetect/lid.176.bin` 与 `lid.176.ftz`，训练器下载包已包含。旧本地布局可以放在源码目录的 `GPT_SoVITS/pretrained_models/fast_langdetect/`。任务在独立工作目录中读取这些资源，缺件时会说明缺少的文件，不在试听进程中临时联网下载。

英文与日文还依赖语言资源：按固定官方安装脚本，把 `nltk_data` 放在所选 Python 环境的 `sys.prefix` 下，把 OpenJTalk 字典放到该环境的 `pyopenjtalk` 包目录。仅安装 `requirements.txt` 不会完成这些字典的部署；启动任务前应一并准备好。

`pretrained_gpt` 和 `pretrained_sovits` 留空时使用表中权重；也可填入兼容的完整初始权重绝对路径。GPT 权重须兼容 `s1v3` 架构；SoVITS 权重须与所选 `v5dev` 或 `v5turbo` 一致。SoVITS 的 LoRA 导出与训练恢复文件不能填作完整底模。即使只训练一个阶段，也需要另一阶段的权重来提取特征并组成最终试听检查点；声码器、BERT、HuBERT 同样需要准备。

### 数据格式与检查

录音须为单声道 PCM WAV，允许 **32000、44100、48000 Hz**。准备逐句切分和准确转写；训练中的官方预处理会在任务目录内生成所需采样率的音频与特征，原录音不被覆盖。本入口不执行自动转写、降噪或人声分离，也不使用独立验证清单。

JSONL 每行填写录音、转写和语言：

```jsonl
{"audio":"wavs/0001.wav","text":"今天的天气很好。","language":"zh","speaker":"voice_a"}
{"audio":"wavs/0002.wav","text":"See you tomorrow.","language":"en","speaker":"voice_a"}
```

也可直接登记 UTF-8 编码的官方 `.list` 文件：

```text
wavs/0001.wav|voice_a|zh|今天的天气很好。
wavs/0002.wav|voice_a|en|See you tomorrow.
```

`audio` 可使用绝对路径或清单所在目录的相对路径。`language` 为 `zh`、`en`、`ja`、`ko`、`yue`，也接受大写；分别对应普通话、英语、日语、韩语和粤语。JSONL 省略 `speaker` 时使用 `speaker`，显式填写时不能为空；`.list` 的四列均须填写。`speaker` 记录来源说话人，生成声线还取决于训练权重与试听参考录音。

路径、转写和说话人不能包含 `|`、换行、制表符或空字符。GPT-SoVITS 不接受 VoxCPM 的 `ref_audio`、`dataset_id` 字段；试听参考录音在创建试听时单独提供。

登记后执行数据检查，可按行查看语言、说话人、录音与问题。任务会把每条录音复制到自己的预处理目录并分配唯一名称，避免不同目录下的同名录音相互覆盖。文本、HuBERT 和语义提取全部完成后才进入训练；缺失任何一条的必要产物会使任务失败。

录音检查通过后的样本还会经过官方音素、语义长度与分桶规则过滤，实际样本数与批次数不能直接等同于清单行数。预检中尚未计算的统计保持为空；样本过少或过滤后没有可训练批次时，应检查录音时长、转写与批大小。

### 训练阶段与参数

`variant` 选择 `v5dev` 或 `v5turbo`；两者使用各自的 SoVITS 底模。`stage` 有三个选项：

- `both`：先训练 SoVITS，再训练 GPT，最终配对两阶段的导出权重。
- `sovits`：只训练 SoVITS，最终配对配置中的 GPT 初始权重。
- `gpt`：只训练 GPT，最终配对配置中的 SoVITS 初始权重。

GPT 使用官方全参微调。SoVITS 使用官方 LoRA 训练入口：CFM 注意力的 Q/K/V/out 投影使用 LoRA，alpha 固定等于 rank；其他仍可训练的模块沿用上游设置。声码器不参与这两个训练阶段。SoVITS 的 rank 可选 16、32、64、128，默认 32；LoRA 目标模块与 alpha 不单独开放。

GPT 与 SoVITS 参数分别保存，未选择执行的阶段仍保留其设置。

| GPT 参数 | 含义与约束 |
| --- | --- |
| `epochs`、`batch_size` | 训练轮数与批大小，默认 15 轮、8 条。实际批量还受上游数据和 DPO 规则影响。 |
| `precision`、`seed` | `16-mixed` 为 FP16 混合精度，`32-true` 为 FP32；默认 `16-mixed`，seed 默认 1234。 |
| `save_every_epoch`、`save_latest` | 导出间隔及是否仅保留最新完整训练状态；默认每 5 轮导出。 |
| `learning_rate`、`initial_learning_rate`、`final_learning_rate` | 调度的目标、起始和结束学习率，默认分别为 `1e-2`、`1e-5`、`1e-4`。 |
| `warmup_steps`、`decay_steps` | 预热与调度总步数，默认 2000、40000；预热须小于调度总步数。 |
| `dpo` | 使用上游 GPT DPO 训练选项，默认关闭。 |
| `max_seconds`、`num_workers` | GPT 样本时长上限和数据加载进程数；默认 54 秒、4 个进程，进程数至少为 1。 |

| SoVITS 参数 | 含义与约束 |
| --- | --- |
| `epochs`、`batch_size` | 训练轮数与批大小，默认 2 轮、1 条。 |
| `precision`、`seed` | `fp16` 或 `fp32`；默认 `fp16`，seed 默认 1234。预处理与试听也使用这个精度选择。 |
| `save_every_epoch`、`save_latest` | 导出间隔及是否仅保留最新完整训练状态；默认每 1 轮导出。 |
| `learning_rate` | AdamW 学习率，默认 `1e-4`。 |
| `adam_beta1`、`adam_beta2`、`adam_epsilon` | AdamW 的动量与数值稳定项，默认 0.8、0.99、`1e-9`。 |
| `lr_decay` | 每轮学习率衰减倍率，默认 0.999875。 |
| `log_interval` | 训练指标记录间隔，默认 100 步。 |
| `lora_rank`、`gradient_checkpointing` | LoRA rank 与梯度检查点；后者通过重算部分中间结果减少训练显存占用。 |

两个阶段的保存间隔都必须整除各自训练轮数，才能保留最后一轮导出权重。GPT 的梯度累积、SoVITS 的数据加载进程数及两阶段的模型结构沿用固定官方入口，不提供额外通用开关。完整字段、默认值和范围由 `GET /api/tts/schema/train?engine=gpt-sovits-v5` 返回。

### 检查点与试听

任务输出目录中的 `gpt_sovits/` 保存预处理文件、各阶段训练状态和定期导出；任务完成后在 `gpt_sovits/final/` 发布配对检查点：

```text
gpt_sovits/final/
├── checkpoint.json
├── gpt.ckpt
└── sovits.pth
```

三个文件须一起保留。`checkpoint.json` 记录模型变体、执行阶段、每阶段的 epoch / global_step 以及两个权重文件的内容摘要；缺少文件、内容变化或与来源任务不匹配时不能新建试听。GPT 和 SoVITS 的计数分别记录，不合成一个检查点步数，也不从文件名猜测；未训练的阶段计数为空。阶段中途的导出保留在任务目录，当前结果入口只发布最终配对检查点。

试听还需要训练任务中记录的完整基础模型、声码器和文本／音频模型，因此保留 `final/` 三文件后也不能移走这些依赖。参考录音与对应文本均必填；参考录音为 **3–10 秒**的单声道 PCM WAV，采样率同样允许 32/44.1/48 kHz。输出为 48 kHz PCM WAV。

| 试听参数 | 行为与范围 |
| --- | --- |
| 文本语言、参考语言 | `zh/en/ja/ko/yue/auto`；默认均为 `zh`，`auto` 使用上游语言识别。 |
| seed | 随机种子，范围 0–4294967295，默认 42。 |
| `top_k`、`top_p` | GPT 采样候选数与累计概率阈值；默认 15、1.0，范围分别为 1–100、大于 0 且不超过 1。 |
| `temperature` | GPT 采样温度，默认 1.0，范围大于 0 且不超过 2。 |
| `speed` | 语速倍率，默认 1.0，范围 0.5–2。 |
| `repetition_penalty` | 重复惩罚，默认 1.35，范围 1–2。 |
| `fragment_interval` | 分段之间的间隔秒数，默认 0.3，范围 0–2。 |
| `sample_steps`、`cfg_scale` | 留空时，v5dev 使用 32 步 / CFG 1.3，v5turbo 使用 4 步 / CFG 0；步数可选 4、8、16、32，CFG 范围 0–2。 |

通过既有 `POST /api/tts/jobs/{job_id}/samples` 创建试听，提供 `checkpoint_id`、`checkpoint_revision`、`text`、`reference_audio`、`reference_text` 和 UUID `Idempotency-Key`。`seed` 使用顶层字段，其余 GSV 选项放在 `gpt_sovits` 对象中；不要同时提交 Vox 的 `cfg_value` 或 `inference_timesteps`。例如：

```json
{
  "checkpoint_id": "所选检查点ID",
  "checkpoint_revision": "所选检查点修订",
  "text": "这是新的试听内容。",
  "reference_audio": "/path/to/reference.wav",
  "reference_text": "参考录音实际说出的内容。",
  "seed": 42,
  "gpt_sovits": {
    "text_language": "zh",
    "reference_language": "zh",
    "sample_steps": 32,
    "cfg_scale": 1.3
  }
}
```

试听任务保存解析后的参数与实际生成信息；还在排队、失败或尚未产生 WAV 时，任务记录保留，音频为空。此入口按完整音频生成，不提供流式返回；CUDA Graph 与 Flash Attention 推理加速未启用。

## 版本复制与任务管理

复制语音版本时，可以复制配置与数据，或创建空数据版本。GPT-SoVITS 复制保留引擎与变体；空版本即使重置训练参数，也保留所选变体。复制数据前，每份已登记清单都须检查通过且内容未变化；复制过程会真正复制录音、参考录音和规范化清单，并把清单路径改为目标版本自己的文件。配置中的运行环境和基础模型路径仍引用原目录。复制不包含任务、检查点、试听音频或历史校验结果；复制失败保留失败状态并释放来源版本的占用。

语音任务支持取消、强制开始和终态重试。排队任务可调整优先级或选择显卡；优先级越大越靠前，但不会抢占正在运行的任务。`gpu_devices=[]` 表示自动选择一张符合任务要求的可见 CUDA 卡：VoxCPM 训练要求支持 BF16，GPT-SoVITS 使用所选 FP16/FP32 配置；试听要求 CUDA。自动模式逐卡检查，有合格候选即可入队；显式选择只使用所选卡，检查失败不会改选其他卡。显卡忙时继续排队。`cuda:N` 是服务当前可见设备的逻辑编号，受 `CUDA_VISIBLE_DEVICES` 和 `CUDA_DEVICE_ORDER` 影响。重试继承原任务的显卡选择并重新检查，需要新的 UUID 请求标识；返回的新任务从头训练或重新生成试听，网络重放同一请求标识只返回原任务。

项目或版本归档后仍可读取结果和播放已有音频，创建训练、试听与重试须先恢复。项目内还有排队任务、运行进程、数据检查或版本复制时不能归档。任务先归档后才能删除；来源训练仍有活动试听时，删除记录和删除文件都会被拒绝。子试听结束后删除来源训练记录，不会连带删除其已有 WAV；来源权重仍在时可按冻结快照重试，权重已删除时只能读取保留的音频。

删除任务文件前还会检查所有已登记语音来源及正在复制的数据。任务目录中的清单、训练音频或参考音频仍被引用时，删除会被拒绝；无法读完来源清单时也须先解决读取问题。仅删除任务记录会保留文件。

旧全局配置保留为独立草稿，不自动应用到新项目。旧无项目任务仍保留历史归属，可读取和按规则重试；新的首次训练须从语音项目版本创建，旧 `POST /api/tts/jobs` 返回 `tts.scope_required`。

## 常见问题

- **环境不是 CUDA 构建或 GPU 不符合要求**：核对 TTS Python 指向的独立环境、PyTorch 构建和 NVIDIA GPU。VoxCPM 训练还要求 BF16；GPT-SoVITS 按所选 FP16/FP32 精度执行。此入口不会自动回退到 CPU、MPS 或 DTK。
- **源码版本不匹配**：在对应引擎的独立源码目录切换到本页列出的固定提交，保留已跟踪文件的原始内容。VoxCPM 和 GPT-SoVITS 使用不同仓库及提交。
- **音频不完整或采样率不符**：重新导出有效的单声道 PCM WAV。VoxCPM 要求 44.1 kHz；GPT-SoVITS 接受 32/44.1/48 kHz。程序不会改写原始音频。
- **CUDA 显存不足**：先降低对应阶段的批大小或缩短录音片段。VoxCPM 可调整梯度累积；GPT-SoVITS 的 SoVITS 阶段可启用梯度检查点。单条长录音仍会增加显存占用。
- **检查点不能试听**：保留检查点目录里的配置和权重，以及训练时使用的本地基础模型。GPT-SoVITS 必须保留同一配对的三个文件，变体与阶段须和来源任务一致；不能混用 dev 与 turbo。
- **GPT-SoVITS 缺少模型或语言资源**：检查 `model_path` 是否指向 `pretrained_models` 根目录，逐项补齐 GPT、所选 SoVITS、声码器、BERT 与 HuBERT；中文另需完整 G2PWModel。
- **GPT-SoVITS 无有效训练样本**：核对逐句转写、语言、音频时长和批大小。音频可播放不等于通过上游音素与语义长度过滤；根据任务中对应阶段的错误修正数据后重新创建任务。

VoxCPM 的代码与模型许可见其 [官方许可说明](https://github.com/OpenBMB/VoxCPM#-license)；GPT-SoVITS 及相关权重见 [官方项目](https://github.com/RVC-Boss/GPT-SoVITS/tree/cuda_graph_accel_v5) 和各模型页面的许可。训练资料和声线授权由所用素材的许可决定。
