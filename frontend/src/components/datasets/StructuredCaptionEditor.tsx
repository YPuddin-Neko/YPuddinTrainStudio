import React from 'react';
import { Plus, X } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';
import { captionFieldKey, captionMetadata, type CaptionFieldDraft, type CaptionStructure } from '../../utils/captionStructure';
import './structured-caption-editor.css';

const LABELS: Record<string, [string, string]> = {
  quality: ['画面质量', 'Quality'], count: ['人物数量', 'Subject count'], character: ['角色', 'Character'],
  character_name: ['角色名称', 'Character name'], character_variant: ['角色形态', 'Character variant'],
  character_full: ['角色完整名称', 'Full character name'], series: ['作品 / 系列', 'Series'], artist: ['画师', 'Artist'],
  appearance: ['外观与服装', 'Appearance and clothing'], tags: ['画面内容', 'Image content'],
  environment: ['环境与背景', 'Environment and background'], nl: ['自然语言描述', 'Natural-language description'], trigger: ['触发词', 'Trigger word'],
};
export default function StructuredCaptionEditor({ structure, draft, onChange, disabled = false, readOnly = false }: {
  structure: CaptionStructure | null; draft: CaptionFieldDraft; onChange: (draft: CaptionFieldDraft) => void;
  disabled?: boolean; readOnly?: boolean;
}) {
  const text = useWorkspaceText();
  const metadata = React.useMemo(() => structure ? captionMetadata(structure) : {}, [structure]);
  if (!structure) return <p className="structured-caption-notice">{text('未取得 JSON 分类数据，请刷新后再编辑。', 'JSON fields could not be loaded. Refresh before editing.')}</p>;
  const locked = disabled || readOnly || !structure.editable;
  return <div className="structured-caption-editor" aria-label={text('JSON 分类标签', 'Structured JSON captions')}>
    {structure.legacy_override && <p className="structured-caption-notice">{text('此文件曾被合并为普通标签。这里编辑当前实际使用的标签；历史分类保留在原文中。', 'This file was previously flattened. Edit its active tags here; historical categories remain in the source document.')}</p>}
    {!structure.editable && <p className="structured-caption-notice">{text('此 JSON 格式暂不能按分类编辑，可查看原文。', 'This JSON format is available for inspection but does not support category editing.')}</p>}
    <div className="structured-caption-fields">
      {structure.fields.map(field => {
        const id = captionFieldKey(field.path), value = draft[id] ?? field.value;
        const label = text(...(LABELS[field.role] || [field.role, field.role]));
        const origin = field.path[0] === 'from_path' ? text('来自目录', 'From folder') : '';
        const name = origin ? `${label} · ${origin}` : label;
        const update = (next: typeof value) => onChange({ ...draft, [id]: next });
        return <section className={`structured-caption-field${field.role === 'nl' ? ' structured-caption-prose' : ''}`} key={id} aria-label={name}>
          <header><h4 title={field.path.join('.')}>{name}</h4>{Array.isArray(value) && <span>{value.length}</span>}</header>
          {Array.isArray(value) ? <><div className="structured-caption-tag-list">
            {value.map((tag, index) => <div className="structured-caption-tag" key={index}>
              <input aria-label={text(`${name} · 标签 ${index + 1}`, `${name} · Tag ${index + 1}`)} value={tag} size={Math.min(30, Math.max(6, tag.length + 1))} disabled={locked} onChange={event => update(value.map((item, i) => i === index ? event.target.value : item))}/>
              {!readOnly && <button type="button" aria-label={text(`删除${name}标签 ${index + 1}`, `Remove ${name} tag ${index + 1}`)} disabled={locked} onClick={() => update(value.filter((_, i) => i !== index))}><X size={14}/></button>}
            </div>)}
            {!value.length && <span className="structured-caption-empty">{text('暂无标签', 'No tags')}</span>}
          </div>{!readOnly && <button className="structured-caption-add" type="button" disabled={locked} onClick={() => update([...value, ''])}><Plus size={14}/>{text(`添加${name}标签`, `Add ${name} tag`)}</button>}</>
            : field.role === 'nl'
              ? <textarea aria-label={name} value={value} rows={4} disabled={locked} onChange={event => update(event.target.value)}/>
              : <input type="text" aria-label={name} value={value} disabled={locked} onChange={event => update(event.target.value)}/>}
        </section>;
      })}
    </div>
    {!!Object.keys(metadata).length && <details className="structured-caption-document"><summary>{text('其他元数据', 'Other metadata')}</summary><pre>{JSON.stringify(metadata, null, 2)}</pre></details>}
    <details className="structured-caption-document"><summary>{text('查看完整 JSON 原文', 'View original JSON document')}</summary><pre>{JSON.stringify(structure.document, null, 2)}</pre></details>
  </div>;
}
