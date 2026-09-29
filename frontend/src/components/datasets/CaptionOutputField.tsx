import { useWorkspaceText } from '../../utils/workspaceText';
import ConfigHelp from '../ConfigHelp';
import StudioSelect from '../StudioSelect';

export type CaptionOutputFormat = 'txt' | 'json' | 'json_simplified';

export default function CaptionOutputField({value, disabled, onChange}: {
  value: CaptionOutputFormat; disabled: boolean; onChange: (value: CaptionOutputFormat) => void;
}) {
  const text = useWorkspaceText();
  const hint = value === 'txt' ? text('保存为同名 .txt 文件。', 'Saves a matching .txt file.')
    : value === 'json' ? text('保存为 .json，使用 ai_output 等分组字段。', 'Saves .json with groups such as ai_output.')
      : text('保存为 .json，标签字段直接放在最外层。', 'Saves .json with caption fields at the top level.');
  return <div className="vision-field"><span className="vision-field-label vision-format-label">{text('输出格式', 'Output format')}
    <ConfigHelp label={text('输出格式说明', 'Output format help')}>
      {text('TXT 保存纯文本；JSON 完整格式将标签放在 fixed、character、ai_output 等字段中，简化格式将标签字段直接放在最外层。两种 JSON 都保留独立的自然语言描述字段。\n切换格式不会删除另一种格式的旧文件。训练读取哪份标签由数据集的标签格式决定；Auto 在同名 TXT 和 JSON 都存在时优先读取 JSON。', 'TXT saves plain text. Full JSON groups captions under fixed, character and ai_output; simplified JSON uses top-level caption fields. Both JSON layouts keep a separate description field.\nChoosing a different format does not delete the other file. Training follows the dataset caption format; Auto prefers JSON when both files exist.')}
    </ConfigHelp></span>
    <StudioSelect aria-label={text('输出格式', 'Output format')} value={value} disabled={disabled}
      options={[{value:'txt',label:'TXT'}, {value:'json',label:text('JSON（完整格式）','JSON (full)')}, {value:'json_simplified',label:text('JSON（简化格式）','JSON (simplified)')}]}
      onValueChange={next=>onChange(next as CaptionOutputFormat)}/>
    <span className="vision-field-hint">{hint}</span>
  </div>;
}
