import type React from 'react';
import ConfigHelp from '../../components/ConfigHelp';
import Switch from '../../components/Switch';
import { PathInput } from '../../components/PathBrowser';
import { fieldCopy, targetOwners, type TtsDraft, type TtsField, type FieldProblem, type FieldSchema } from './ttsVersionFields';
import type { TtsParameterGroup } from './TtsParameterForm';

type Props = {
  draft: TtsDraft; properties: Record<TtsField, FieldSchema>; problems: FieldProblem[];
  orderedGroups: { id: string; zh: string; en: string; fields: TtsField[] }[];
  readOnly: boolean; saving: boolean; english: boolean; preset?: boolean;
  text: (zh: string, en: string) => string; change: (field: TtsField, value: TtsDraft[TtsField]) => void;
};
export function ttsParameterGroups({ draft, properties, problems, orderedGroups, readOnly, saving, english, text, change, preset = false }: Props): TtsParameterGroup[] {
  const bounds = (field: TtsField) => {
    const prop = properties[field], rule = prop.anyOf?.find(item => item.type !== 'null') || prop;
    return [rule.type === 'integer' ? text('整数', 'Integer') : rule.type === 'number' ? text('数值', 'Number') : '',
      rule.minimum != null ? `≥ ${rule.minimum}` : rule.exclusiveMinimum != null ? `> ${rule.exclusiveMinimum}` : '',
      rule.maximum != null ? `≤ ${rule.maximum}` : rule.exclusiveMaximum != null ? `< ${rule.exclusiveMaximum}` : ''].filter(Boolean).join(' · ');
  };
  const renderField = (field: TtsField): React.ReactNode => {
    const property = properties[field], copy = fieldCopy(field, english), value = draft[field];
    const id = `tts-version-${field}`, problem = problems.filter(issue => issue.field === field).map(issue => issue.message).join(' ');
    const heading = <div className="config-field-heading"><label htmlFor={property.type === 'string' ? undefined : id}>{copy.label}</label><span className="config-field-reference"><code tabIndex={0} className="config-field-key">{field}</code><ConfigHelp label={`${copy.label} · ${text('说明', 'Help')}`}>{[copy.help, bounds(field), property.type === 'array' ? text('只选列表中的层名；启用该组件时至少选择一层，不支持正则表达式。', 'Choose listed layer names, at least one for an enabled component. Regular expressions are not supported.') : ''].filter(Boolean).join('\n\n')}</ConfigHelp></span></div>;
    if (property.type === 'boolean') {
      const target = Object.keys(targetOwners).find(name => targetOwners[name as TtsField] === field) as TtsField;
      return <section key={field} data-field={field} className="tts-lora-component"><div className="tts-lora-heading"><Switch id={id} aria-label={copy.label} aria-controls={`${id}-targets`} checked={value === true} disabled={readOnly || saving} onCheckedChange={checked => change(field, checked)}/>{heading}</div>{problem && <p role="alert" className="config-field-error">{problem}</p>}<div id={`${id}-targets`}>{renderField(target)}</div></section>;
    }
    return <div key={field} data-field={field} className={`config-field${property.type === 'string' || property.type === 'array' ? ' config-field-wide' : ''}${problem ? ' config-field-invalid' : ''}`}>
      {heading}<fieldset className={`config-field-control${property.type === 'string' ? ' model-path-control' : ''}`} disabled={readOnly || saving || !!targetOwners[field] && draft[targetOwners[field]!] !== true}>
        {property.type === 'string' ? <PathInput ariaLabel={copy.label} value={String(value)} onChange={next => change(field, next)} directoryOnly={field !== 'python_path'}/>
          : property.type === 'array' ? <div id={id} role="group" aria-label={copy.label} className="tts-target-options">{property.items?.enum?.map(target => <label key={target}><input type="checkbox" checked={Array.isArray(value) && value.includes(target)} onChange={event => change(field, event.target.checked ? [...(Array.isArray(value) ? value : []), target] : (value as string[]).filter(item => item !== target))}/><span>{target}</span></label>)}</div>
            : <input id={id} type="text" inputMode="decimal" autoComplete="off" spellCheck={false} value={String(value)} aria-invalid={!!problem} aria-describedby={`${id}-hint${problem ? ` ${id}-error` : ''}`} placeholder={field === 'valid_interval' ? text('跟随保存间隔', 'Follow save interval') : undefined} onChange={event => change(field, event.target.value)} onBlur={() => { if (field === 'learning_rate' && String(value).trim() && Number.isFinite(Number(value))) change(field, Number(value).toExponential()); }}/>}</fieldset>
      <div className="config-field-footer"><p id={`${id}-hint`} className="config-field-hint">{copy.hint}</p>{problem && <p id={`${id}-error`} role="alert" className="config-field-error">{problem}</p>}</div>
    </div>;
  };
  return orderedGroups.map(group => ({ ...group, fields: group.fields.filter(field => !preset || !['engine', 'variant', 'python_path', 'trainer_path', 'model_path', 'pretrained_gpt', 'pretrained_sovits'].includes(field)) })).filter(group => group.fields.length).map(group => {
    const buckets = group.id === 'environment' ? [
      { label: text('训练模型', 'Training model'), fields: ['model_path'] },
      { label: text('运行环境', 'Runtime environment'), fields: ['python_path', 'trainer_path'] },
    ] : [{ label: undefined, fields: group.fields }];
    return { id: group.id, label: english ? group.en : group.zh, sections: buckets.map(bucket => ({ label: bucket.label,
      fields: group.fields.filter(field => bucket.fields.includes(field) && !targetOwners[field]).map(field => {
        const copy = fieldCopy(field, english);
        const targets = Object.keys(targetOwners).filter(name => targetOwners[name as TtsField] === field);
        const targetSearch = targets.map(name => { const target = fieldCopy(name as TtsField, english); return `${name} ${target.label} ${target.hint}`; }).join(' ');
        return { id: field, search: `${field} ${copy.label} ${copy.hint} ${copy.help} ${targetSearch}`, node: renderField(field) };
      }),
    })).filter(section => section.fields.length) };
  });
}
