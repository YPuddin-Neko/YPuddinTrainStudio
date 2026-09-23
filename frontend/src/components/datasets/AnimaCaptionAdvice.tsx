import { useState } from 'react';
import { useWorkspaceText } from '../../utils/workspaceText';
import './anima-caption-advice.css';

export type CaptionFormatAdvice = {
  profile: 'anima';
  suggestions: { path: string[]; role: string; before: string | string[]; after: string | string[] }[];
};
type AdviceImage = { rel_path: string; path: string; caption_format?: CaptionFormatAdvice };
const LABELS: Record<string, [string, string]> = {
  quality: ['画面质量', 'Quality'], count: ['人物数量', 'Subject count'], character: ['角色', 'Character'],
  character_name: ['角色名称', 'Character name'], character_variant: ['角色形态', 'Character variant'],
  character_full: ['角色完整名称', 'Full character name'], series: ['作品 / 系列', 'Series'], artist: ['画师', 'Artist'],
  appearance: ['外观与服装', 'Appearance and clothing'], tags: ['画面内容', 'Image content'],
  environment: ['环境与背景', 'Environment and background'],
};
const valueText = (value: string | string[]) => Array.isArray(value) ? value.join('\n') : value;

export default function AnimaCaptionAdvice({ images, onEdit }: { images: AdviceImage[]; onEdit: () => void }) {
  const text = useWorkspaceText();
  const [limit, setLimit] = useState(20);
  const proposed = images.filter(image => !!image.caption_format?.suggestions.length);
  return <details className="anima-caption-advice">
    <summary>{text('Anima 标签格式建议', 'Anima caption format advice')}{proposed.length > 0 && ` · ${text(`${proposed.length} 张可预览`, `${proposed.length} image previews`)}`}</summary>
    <p>{text('建议普通标签用小写和空格，score_* 保留下划线，画师名加 @。', 'Use lowercase and spaces for ordinary tags, keep underscores in score_* tags, and prefix artists with @.')}</p>
    <p>{text('自然语言保留正常大小写。TXT 含完整句子时，请核对标签打乱和丢弃设置；JSON 自然语言字段不打乱。', 'Keep normal capitalization in prose. For TXT captions containing sentences, check shuffle and dropout settings. JSON prose fields are not shuffled.')}</p>
    <p>{text('以下为 JSON 标签格式预览，不会自动修改文件。', 'Preview of JSON tag formatting; files are not changed automatically.')}</p>
    <a href="https://huggingface.co/circlestone-labs/Anima#prompting" target="_blank" rel="noreferrer">{text('查看模型作者的标签指南', 'Read the model author’s prompting guide')}</a>
    {proposed.length > 0 && <>
      <div className="anima-caption-previews">
        {proposed.slice(0, limit).map(image => <details className="anima-caption-file" key={image.path}>
          <summary title={image.path}>{image.rel_path}</summary>
          {image.caption_format!.suggestions.map(field => <section key={JSON.stringify(field.path)}>
            <h4>{text(...(LABELS[field.role] || [field.role, field.role]))}{field.path[0] === 'from_path' ? ` · ${text('来自目录', 'From folder')}` : ''}</h4>
            <dl><div><dt>{text('原值', 'Original')}</dt><dd><pre>{valueText(field.before)}</pre></dd></div><div><dt>{text('建议值', 'Proposed')}</dt><dd><pre>{valueText(field.after)}</pre></dd></div></dl>
          </section>)}
        </details>)}
      </div>
      {proposed.length > limit && <button type="button" onClick={() => setLimit(limit + 20)}>{text('显示更多格式预览', 'Show more format previews')}</button>}
      <button type="button" onClick={onEdit}>{text('打开标签编辑', 'Open caption editor')}</button>
    </>}
  </details>;
}
