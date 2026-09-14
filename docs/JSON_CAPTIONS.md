# JSON 标签使用说明

图片与标签放在同一目录、使用同名文件，例如 `portrait.png` 对应 `portrait.json`。程序先解析 JSON，再将标签和自然语言描述组成训练文本；不会把 JSON 的键名、括号或任意元数据直接交给文本编码器。

## 选择标签文件

- 新导入来源默认使用 `caption_ext="auto"`：同名 JSON 优先，其次 TXT。两者都存在时，只使用 JSON；扩展名大小写不影响识别。
- 已保存的显式 `.txt` 配置保持原样；显式 `.json` 或自定义后缀也只选择指定格式，不向其他格式回退。自定义非 JSON 后缀按普通文本读取。
- 若所选 JSON 损坏、字段类型错误或含 `NaN/Infinity`，会报告文件名和原因，**不会改读 TXT 或类别提示词**。应修复文件，或明确修改来源的标签格式。上传、ZIP 和本机导入会先检查所选 JSON；失败时不登记为可用数据集。
- 使用 `auto` 且没有现成标签时，在逐图编辑器保存会创建 TXT。修改来源的标签格式后，已注册数据集会重新建立对应标签索引。

## 当前支持的结构

最简单的写法是顶层 `tags` 数组，也支持逗号分隔的字符串：

```json
{
  "tags": ["1girl", "blue_hair", "smile"],
  "nl": "A woman smiles in a garden.",
  "meta": {"trigger": "my_character", "source": "manual"}
}
```

默认读取结果为 `my_character, 1girl, blue_hair, smile. A woman smiles in a garden.`。`meta.source` 等额外信息保留在文件中，不进入训练文本。这里演示的是通用文件解析；程序不会自动替换标签大小写、下划线或画师前缀。Anima 的格式建议见下文，不会全局套用到 SDXL、Krea2 或其他模型。

也可以按类别组织，类别既可放在顶层，也可统一放在 `tags` 对象中：

```json
{
  "meta": {"trigger": "my_character"},
  "tags": {
    "quality": ["high_quality"],
    "count": "1girl",
    "character": {"name": "character_name", "variant": "summer_outfit"},
    "series": "original",
    "artist": "artist_name",
    "appearance": ["blue_hair"],
    "tags": ["smile"],
    "environment": ["garden"],
    "nl": "A woman smiles in a garden."
  }
}
```

| 字段 | 实际含义 |
| --- | --- |
| `quality/count/character/series/artist` | 按此顺序组成固定标签组 |
| `appearance/tags/environment` | 依次组成外观、一般标签和环境组 |
| `nl` | 自然语言描述，接在标签后面；必须是字符串 |
| `meta.trigger` | 文件内触发词，放在固定组之前 |

标签字段接受字符串或字符串数组；字符串按逗号拆分。空项会移除，重复标签按不区分大小写去重并保留首次写法。`character` 还支持对象：优先使用非空 `full`，否则组合 `name` 和 `variant`。类别在 `tags` 对象中时，以该对象的字段为准，不与同名顶层类别混合。

已有的完整结构也可读取：

| 完整结构位置 | 读取到的内容 |
| --- | --- |
| `fixed.quality/series/artist` | 对应固定标签 |
| 顶层 `character`、`ai_output.count` | 角色与数量 |
| `ai_output.appearance` + `from_path.appearance/extra_appearance` | 合并为外观组 |
| `ai_output.tags` + `from_path.tags/extra_tags` | 合并为一般标签组 |
| `ai_output.environment/nl` | 环境与自然语言描述 |

出现 `fixed/ai_output/from_path` 任一字段即按完整结构解析。完整结构中的顶层 `tags` **数组**表示已编辑的权威标签列表，会取代旧分类标签，避免删掉的标签重新出现。

当前自然语言字段是 **`nl`**：普通结构为顶层或 `tags.nl`，完整结构为 `ai_output.nl`。`detail/short/long`、`caption/description` 以及其他未列出的字段不被自动识别为训练描述；它们可作为额外信息保留。如需训练这些描述，请先明确整理到对应的 `nl`，不要只改文件扩展名。无法识别结构的 JSON 可查看原文，但不能按分类编辑，也会在训练数据预检和读取时拒绝；不会静默作为空描述、改读 TXT 或丢掉原字段。

## 查看、编辑与恢复

数据工作区的“标签查看”显示解析后的完整文本。逐图编辑器将可编辑标签与 JSON 自然语言描述分开；描述保持原文，不被逗号拆成标签。

分类编辑按原字段路径保存，只提交改动的字段，保留字符串或数组类型及所属分类；修改自然语言描述不会改标签，未知元数据保持原值。程序不再创建顶层覆盖列表或内部编辑标记。保存前会核对读取时的文件版本，文件已被其他操作修改时拒绝覆盖。保留的是字段内容，保存后的 JSON 缩进和字节布局可能改变。

历史上已经保存的覆盖列表仍代表当前实际训练文本，不会擅自恢复原分类、让已删除标签重新出现。编辑器明确说明这种状态，只编辑当前生效的标签和描述；原分类保留为历史内容。清空标签仍会保留 `nl`，不等于清空全部训练文本；训练配置另外设置的触发词、前后缀仍按配置应用。

“操作记录与恢复”中带恢复按钮的流水线操作可恢复此前文件，JSON 会恢复到操作前的原始字节。如果文件在操作后被另行修改，或备份已损坏，恢复会拒绝覆盖。**逐图保存和直接批量标签接口不会自动生成流水线撤销记录**；批量接口遇到解析或写入错误会回滚本次已写文件，成功后的手动恢复仍需自己的备份。

## 训练变换、缓存与续训

正常分类 JSON（包括按分类编辑后）保留组结构：固定标签和自然语言描述不参与逐标签丢弃，三个可变组分别打乱、丢弃；`keep_tokens` 不用于结构化分组。整条 caption dropout 仍可置空全部条件。历史上已经扁平化的文件使用其当前标签列表，不再具有原分类的固定保护；新分类编辑不会再次扁平化文件。

开启文本缓存后，JSON 同样使用 `cache_variants` 生成有限且可重复的标签变体，训练从已编码变体中选择；整条 caption dropout 仍在取样时执行。缓存键使用**最终文本和文本编码器指纹**，不是 JSON 原文；图片 latent 缓存不因只改标签而重编码。只改元数据且最终文本不变时，可继续复用相同文本缓存。

数据身份校验仍包含标签文件内容。因此修改标签、描述甚至 JSON 元数据，可能使完整 state 精确续训拒绝恢复；缓存可复用不等于旧训练状态可恢复。本次结构化支持没有改动原有 TXT 的身份算法；JSON 增加了解析语义标记，旧版把 JSON 当原始文本的训练状态不能静默续用新语义。已有 `.txt` 来源也不会因升级自动切成 `auto`。

## Anima 标签格式检查

在 Anima 版本的「数据 → 检查与筛选」执行「检查数据」，可展开「Anima 标签格式建议」查看已知 JSON 分类字段的原值和建议值。检查只读文件，不自动保存、不批量替换；确认某项建议适合当前数据后，再进入现有标签编辑器修改。切换模型或修改标签变换设置后，旧检查报告会失效，需要重新检查。

普通标签建议使用小写与空格，画师名使用 `@` 前缀；`score_*` 分数标签是保留下划线的例外。自然语言保持正常大小写，可以与标签混排，不强制放到最后，也不把推荐的标签顺序当作训练错误。这些规则来自 [Anima 模型作者的 Prompting 说明](https://huggingface.co/circlestone-labs/Anima#prompting)。用户提供的 [AnimaLoraStudio 打标指南](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/master/docs/user-guide/tagging-guide.md) 另给出了按分类组织标签、将描述放在末尾的工作流建议；本工具不会把这类建议升级为必须重写数据的规则。

- JSON 修正预览只针对已识别的标签分类，保留原字段归属及字符串/数组类型。`score_*`、文件或训练配置中的触发词、自然语言与其他元数据不改；未知 JSON、历史覆盖格式和动态提示词不生成推断修正。
- 纯 TXT 不猜测哪个词是画师，也不自动更改大小写或下划线。开启标签打乱/丢弃时，若可变片段中检测到疑似完整句子，会提示用户核对；这是有限的启发式风险提示，不是可靠的语言分类，也不会阻止合法的标签洗牌。
- TXT 的通用处理按配置分隔符拆分，混合自然语言时可能打乱句子或丢掉部分语句。分类 JSON 的 `nl` 字段不会按逗号洗牌；可考虑使用这个独立字段保留描述，但程序不会自动迁移原文。

这些检查仅在选择 Anima 模型族时启用。JSON 支持表示结构可以正确读取和编辑，不代表其中的标签适合所有模型，更不等于训练效果已经验证。

实现与回归：[解析器](../ypuddin/data/caption_json.py)、[标签读取与变换](../ypuddin/data/captions.py)、[文件选择与数据身份](../ypuddin/data/index.py)、[Anima 只读检查](../ypuddin/data/anima_caption_inspection.py)、[JSON 单元测试](../tests/unit/test_caption_json.py)、[导入、编辑、恢复与版本隔离测试](../tests/unit/test_json_caption_workflow.py)。其他训练参数见 [训练参数说明](TRAINING_PARAMETERS.md)。
