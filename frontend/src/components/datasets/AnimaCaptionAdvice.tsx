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
    <p>{text('一般标签建议使用小写与空格；score_* 分数标签保留下划线，画师名加 @。这是格式建议，不是训练限制。', 'Use lowercase and spaces in ordinary tags, retain underscores in score_* tags, and prefix artist names with @. This is advice, not a training restriction.')}</p>
    <p>{text('自然语言可以与标签混排，保留正常大小写。TXT 混入完整句子时，请核对标签打乱和丢弃设置；JSON 的自然语言字段不会按逗号打乱。', 'Natural language can appear before or after tags and keeps normal capitalization. For TXT containing sentences, review tag shuffle and dropout settings. The JSON prose field is not shuffled by commas.')}</p>
    <p>{text('以下仅预览已知 JSON 分类字段；原文件未修改，触发词、score_*、自然语言和其他元数据保持原样。TXT 不推断画师或自动改写。', 'Previews cover known JSON tag fields only. Original files, triggers, score_* tokens, prose and other metadata are unchanged. TXT artist roles are not inferred and TXT is not rewritten.')}</p>
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
