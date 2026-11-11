# JSON 标签格式

标签文件与图片同名，例如 `portrait.png` 对应 `portrait.json`。Anima JSON 是已有分类标签格式：训练时解析标签字段并组成文本，JSON 键名和其他元数据不直接作为提示词。MOSS JSON 则是下文的官方 Layout 布局格式，两者分别读取。

## 基本格式

```json
{
  "tags": ["1girl", "blue hair", "smile"],
  "nl": "A woman smiles in a garden.",
  "meta": {"trigger": "my_character"}
}
```

该示例生成文本：`my_character, 1girl, blue hair, smile. A woman smiles in a garden.`

`tags` 接受字符串数组或逗号分隔的字符串。`nl` 为自然语言描述，必须是字符串。`meta.trigger` 放在标签前，其余元数据保留在文件中。

## 分类格式

```json
{
  "tags": {
    "quality": ["high quality"],
    "count": "1girl",
    "character": {"name": "character_name", "variant": "summer outfit"},
    "series": "original",
    "artist": "@artist_name",
    "appearance": ["blue hair"],
    "tags": ["smile"],
    "environment": ["garden"],
    "nl": "A woman smiles in a garden."
  }
}
```

分类字段可位于顶层，也可放在 `tags` 对象中。使用 `tags` 对象时，以该对象的分类字段为准。

| 字段 | 含义 |
| --- | --- |
| `quality`、`count`、`character`、`series`、`artist` | 按此顺序构成固定标签组 |
| `appearance`、`tags`、`environment` | 外观、通用标签和环境标签组 |
| `nl` | 自然语言描述 |
| `meta.trigger` | 文件内触发词 |

`character` 可用字符串、数组或对象。对象优先读取非空 `full`；否则组合 `name` 与 `variant`。标签去除空项，按不区分大小写去重，并保留首次出现的写法。

## 兼容结构

存在 `fixed`、`ai_output` 或 `from_path` 时，按以下结构读取：

| 字段路径 | 对应内容 |
| --- | --- |
| `fixed.quality`、`fixed.series`、`fixed.artist` | 固定标签 |
| 顶层 `character`、`ai_output.count` | 角色与数量 |
| `ai_output.appearance` 与 `from_path.appearance/extra_appearance` | 外观组 |
| `ai_output.tags` 与 `from_path.tags/extra_tags` | 通用标签组 |
| `ai_output.environment`、`ai_output.nl` | 环境与自然语言描述 |

该结构中的顶层 `tags` 数组表示已编辑的标签列表，会覆盖原分类标签。保留该覆盖关系可防止删除过的标签重新进入训练文本。

图片打标的“输出格式”选“JSON（完整格式）”时按此结构写入：`fixed` 保存质量、作品和画师，顶层 `character` 保存角色，数量、外观、通用和环境标签以及 `nl` 写入 `ai_output`。选“JSON（简化格式）”时，这些分类字段直接写在顶层。

当前自然语言字段为 `nl`。`caption`、`description`、`detail`、`short`、`long` 不自动映射为训练描述。无法识别结构或字段类型错误的文件会被拒绝读取。

## 编辑与恢复

分类编辑按字段路径保存，保留字符串或数组类型以及未知元数据。文件版本变化时拒绝覆盖。保存可能改变 JSON 的缩进和字节布局。

打标写入的 JSON 结构与文件原有结构不同时，文件按所选输出格式转换；新结构中没有位置的字段保存在 `meta.previous_caption_fields`。

清空标签不清空 `nl`，也不取消训练配置中的触发词、前缀和后缀。逐图保存没有流水线撤销记录；需要恢复文件时应保留自己的备份。

## 训练处理

分类 JSON 的固定组和自然语言描述不参与逐标签丢弃。三个可变组分别打乱和丢弃；`keep_tokens` 不作用于这些分类组。整条描述丢弃仍可清空完整文字条件。

文本缓存基于最终文本和编码器身份生成，随机标签使用有限的预编码变体。只修改未知元数据、且最终文本不变时，可复用文本缓存；完整状态的数据指纹仍包含标签文件内容，修改文件可能阻止精确续训。

## MOSS-Image 标签复用

MOSS-Image 与 Anima 共用本页的标签读取和训练处理规则。训练器生成的完整 JSON、简化 JSON，以及基本 `tags`／`nl` 格式都可直接复用，不需要仅因更换模型重新打标。逗号分隔的 Danbooru 标签也可保存为同名 TXT，按普通文本读取。

JSON 标签先按所选打乱、丢弃、触发词等设置拼成文本，`nl` 接在标签之后；MOSS-Image 再将最终文本放入配套 Qwen3.5 的聊天模板并编码。程序不会把 Tag 自动扩写成自然语言。标注文件可以共用，编码后的文本缓存仍按各模型的编码器分别生成。

MOSS 官方的 JSON Layout 是包含 `summary`、`items` 和位置框的布局提示词。已有分类标签不会自动转换成 Layout；可从外部导入原始 MOSS Layout，也可生成或导入 GBC 图后按下节转换。

## GBC 图和 MOSS Layout

`caption_ext` 选择标签文件后缀，可使用 `auto`、`.txt`、`.json` 或已有自定义后缀。结构标签的最终扩展名须为 `.json`，例如 `.gbc.json`；自动模式不会猜测复合后缀。`dataset.caption.content_type` 的正常选项为 `auto` 自动识别、`text` TXT、`anima_json` Anima JSON、`moss_layout` MOSS JSON。数据来源的 `caption` 设置可覆盖全局设置。

MOSS JSON 与结构用法共同决定输入：`moss_layout+native` 读取原始 MOSS Layout；`moss_layout+gbc_to_moss` 读取 GBC 图并转换成 MOSS Layout，不能把已有 Layout 再作为 GBC 输入。GBC 不单列为新的训练内容类型。旧 `gbc` 配置值仍可加载、保存，并在 MOSS 下沿原 `gbc_to_moss` 路径执行和恢复，不自动改写为 `moss_layout`；原生 GBC 仍拒绝。旧 `text` 的 TXT 与分类 JSON 读取兼容语义保持不变。

`dataset.caption.structured_mode` 决定结构内容如何进入模型：

| 内容 | `native` 原生格式 | `gbc_to_moss` GBC 转 MOSS Layout |
| --- | --- | --- |
| GBC | 当前模型没有原生 GBC 图入口 | 仅 MOSS-Image：按 KohakuGBC 规则转换后编译为官方 Layout |
| MOSS Layout | 仅 MOSS-Image：接收编译后的区域提示词 | 已是 Layout，请使用原生格式 |

自动识别默认使用 `native`，不会自动转换 GBC。用户需主动选择 `gbc_to_moss`，内容类型为 `moss_layout` 或 `auto`，模型为 MOSS-Image；其他底模不开放该转换。旧 `natural_language` 值可打开、保存纠正，执行时会明确拒绝，不静默改成另一训练用途。已有分类 JSON 的 `nl` 字段继续使用，与这个停用的配置值无关。模型能力接口的 `caption_content_types` 提供正常内容选项，MOSS 不再单列 `gbc`；`native_caption_types` 表示原生支持，文件能力仍由 `caption_formats` 表示。

`auto` 下的结构用法只作用于实际结构标签：普通 TXT、Anima 分类 JSON 和缺少侧标注时的类别提示词仍沿已有文本路径；GBC 才执行转换，已是 MOSS Layout 的来源须使用 `native`。来源可单独覆盖用法。

转换固定采用 KohakuGBC `from_gbc` 默认规则：描述按 `detail`、`composition`、`short`、`hardcode` 优先，每类取首条；标签阈值为 0.35，按置信度排序并将下划线转为空格。随后按 `to_moss(relations=True)` 将有框节点铺平为 `object` 项，关系描述并入 `summary.caption`，画布为 `single`。图层级不会成为 MOSS 图注意力输入，也不会推断人物类型或实际文字；原始图与元数据保留在文件中。需要明确的 `character` 或 `text` 项时，应直接提供相应 MOSS Layout。

MOSS 的原生布局指模型已学习的 Layout 提示词格式，编译后的 JSON 仍经配套 Qwen3.5 文本编码器进入模型，不新增独立区域张量或图注意力输入。超长内容仍受该编码器的 4096 token 上限约束。

GBC 接受单个已解析图（`vertices`）或带 `graph` 的单图记录，节点使用 `vertex_id`、`label`、`descs`、归一化 `bbox` 和 `out_edges`。图必须有一个 `image` 根节点，节点 ID 唯一，边引用存在且没有环，所有节点可从根到达；提供 `in_edges` 时须与出边一致。`img_size` 按 `[width, height]` 解释。原始查询记录、整包 JSONL、生成模型的中间回答不直接作为单图标签。

以下是配合 64×64 图片的最小 Layout 草稿：

```json
{
  "width": 64,
  "height": 64,
  "bbox_coord_space": "abs",
  "bbox_format": "xyxy",
  "summary": {"caption": "A blue cup on a white table."},
  "items": [
    {"label": "object", "caption": "A blue ceramic cup.", "bbox": [8, 8, 40, 56]}
  ]
}
```

草稿尺寸为 16 的正整数倍，框为绝对像素 `xyxy`。`items` 可包含 `object`、`character` 或 `text`；文字项的 `text` 保存实际可见字符，`caption` 描述字体与外观。未知 Layout 字段、非有限坐标和无效框会报错。

训练时先读取 EXIF 定向后的原图尺寸。草稿画布可与原图等比不同，比例不一致会拒绝；归一化 GBC 框按原图尺寸映射。随后复用图像实际的 resize、crop 或 pad 几何，最后同步水平翻转。MOSS 编译结果使用 `abs_centralized`、`yxyx`，按官方规则对左/上边向下取整、右/下边向上取整，再减画布中心。

也可读取官方编译后的 canonical Layout。它没有 `width`、`height`，因此以配对原图的 EXIF 定向尺寸为其原始画布；不能把按另一尺寸编译的框直接配给图片。编译前草稿更适合跨分辨率管理。裁剪会去除不可见元素及对应 GBC 边；Layout 全部元素被裁出时会报错。文字框被部分裁断，或含可见文字的 Layout 开启水平翻转，也会在训练预检查中拒绝，请调整裁剪或使用补边、关闭翻转。

原生和转换后的 Layout 都将触发词、前缀和后缀加入 `summary.caption`。标签打乱和逐标签丢弃不拆散布局描述；通配符与提示词权重需关闭，整条 caption dropout 仍可清空文字条件。文本缓存包含最终几何和翻转对应的提示词，按实际采样方向读取；改变坐标、内容或用途后不会复用另一提示词的缓存。已有普通 TXT 与分类 JSON 的 `auto/native` 默认配置和恢复指纹保持原有语义；改变结构用途会改变训练身份，不能按旧用途精确续训。

数据集编辑器显示完整原始 JSON，但 GBC 与 Layout 保持只读。普通标签编辑、批量标签操作和分类格式转换不能覆盖它们。需要修改结构时，可在外部编辑后重新导入或更新对应标签文件。也可在[辅助打标](DATASETS.md#辅助打标)中开启 GBC 细化，并选择保存同名 `.gbc.json`。

图片预处理不会单独裁剪或缩放带有 GBC/Layout 侧标注的原图，避免标注坐标失效；需在外部同时修改图片和标注后导入。训练时的画布变换在内存中同步处理坐标，不会改写这些原文件。

训练预览的 `sampling.prompts_file` 可读取单份 GBC/Layout `.json`，复用所选用途与同一编译器；GBC 使用 `img_size`，缺省时使用采样尺寸，Layout 草稿使用其声明尺寸，canonical 使用采样尺寸。坐标映射到模型实际对齐后的预览画布。TXT 每行提示词和 TOML 逐条参数继续适用。结构 JSON 应作为完整文件读取，不能按多行文本拆成多条提示词。

格式依据：[KohakuGBC `d90f360`](https://github.com/KohakuBlueleaf/KohakuGBC/tree/d90f360dbf561f63270522e44fa0e74ebed59c12)、[MOSS Layout `1961ca3`](https://github.com/OpenMOSS/MOSS-Image/blob/1961ca3d111b828e65278656cde0c8f94484f4dd/skills/moss-image-prompt-enhancer/references/layout.md) 和[官方编译器](https://github.com/OpenMOSS/MOSS-Image/blob/1961ca3d111b828e65278656cde0c8f94484f4dd/skills/moss-image-prompt-enhancer/scripts/compile_prompt.py)。

## Anima 格式检查

Anima 的数据检查会对已识别的分类字段提供大小写、空格和画师前缀建议。建议不自动写入文件，触发词、自然语言和未知元数据保持原样。

纯 TXT 不推断画师字段。标签与自然语言混写时，按分隔符打乱或丢弃可能拆散句子；需要保留整段描述时可使用 `nl`。
