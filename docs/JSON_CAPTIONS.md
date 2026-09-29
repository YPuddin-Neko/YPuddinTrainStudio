# JSON 标签格式

标签文件与图片同名，例如 `portrait.png` 对应 `portrait.json`。训练时解析标签字段并组成文本，JSON 键名和其他元数据不直接作为提示词。

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

## Anima 格式检查

Anima 的数据检查会对已识别的分类字段提供大小写、空格和画师前缀建议。建议不自动写入文件，触发词、自然语言和未知元数据保持原样。

纯 TXT 不推断画师字段。标签与自然语言混写时，按分隔符打乱或丢弃可能拆散句子；需要保留整段描述时可使用 `nl`。
