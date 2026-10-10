import type React from 'react';
import ConfigHelp from '../../components/ConfigHelp';
import CheckboxSelect from '../../components/CheckboxSelect';
import Switch from '../../components/Switch';
import TtsEnvironmentPanel from '../../components/TtsEnvironmentPanel';
import { PathInput } from '../../components/PathBrowser';
import { fieldCopy, targetOwners, type TtsDraft, type TtsField, type FieldProblem, type FieldSchema } from './ttsVersionFields';
import type { TtsParameterGroup } from './TtsParameterForm';
import { TtsDecimalInput, TtsScientificBadge } from './ttsParameterControls';
import { voxAdvancedParameters } from './ttsParameterMetadata';

type Props = {
  draft: TtsDraft; properties: Record<TtsField, FieldSchema>; problems: FieldProblem[];
  orderedGroups: { id: string; zh: string; en: string; fields: TtsField[] }[];
  readOnly: boolean; saving: boolean; english: boolean; preset?: boolean;
  text: (zh: string, en: string) => string; change: (field: TtsField, value: TtsDraft[TtsField]) => void;
};
export function ttsParameterGroups({ draft, properties, problems, orderedGroups, readOnly, saving, english, text, change, preset = false }: Props): TtsParameterGroup[] {
  const renderField = (field: TtsField): React.ReactNode => {
    const property = properties[field], copy = fieldCopy(field, english), value = draft[field];
    const id = `tts-version-${field}`, problem = problems.filter(issue => issue.field === field).map(issue => issue.message).join(' ');
    const disabled = readOnly || saving || !!targetOwners[field] && draft[targetOwners[field]!] !== true;
    const describedBy = `${copy.hint ? `${id}-hint` : ''}${problem ? ` ${id}-error` : ''}`.trim() || undefined;
    const help = <ConfigHelp label={`${copy.label} · ${text('说明', 'Help')}`}>{copy.help}</ConfigHelp>;
    const heading = <div className="config-field-heading"><label htmlFor={property.type === 'string' || property.type === 'array' ? undefined : id}>{copy.label}{field === 'learning_rate' && <TtsScientificBadge value={value} english={english}/>}</label><span className="config-field-reference"><code tabIndex={0} className="config-field-key">{field}</code>{help}</span></div>;
    if (property.type === 'boolean') {
      const target = Object.keys(targetOwners).find(name => targetOwners[name as TtsField] === field) as TtsField;
      return <section key={field} data-field={field} className="parameter-toggle-layout tts-lora-component" aria-labelledby={`${id}-title`}>
        <header className="parameter-toggle-heading">
          <div className="parameter-toggle-copy"><div className="parameter-toggle-title"><h3 id={`${id}-title`}>{copy.label}</h3>{help}</div>{problem && <p id={`${id}-error`} role="alert" className="config-field-error">{problem}</p>}</div>
          <Switch id={id} aria-label={copy.label} aria-invalid={!!problem} aria-describedby={problem ? `${id}-error` : undefined} aria-controls={`${id}-targets`} checked={value === true} disabled={readOnly || saving} onCheckedChange={checked => change(field, checked)}/>
        </header>
        <div id={`${id}-targets`}>{renderField(target)}</div>
      </section>;
    }
    return <div key={field} data-field={field} className={`config-field${property.type === 'string' ? ' config-field-wide' : ''}${problem ? ' config-field-invalid' : ''}`}>
      {heading}<fieldset className={`config-field-control${property.type === 'string' ? ' model-path-control' : ''}`} disabled={disabled}>
        {property.type === 'string' ? <PathInput ariaLabel={copy.label} value={String(value)} onChange={next => change(field, next)} directoryOnly={field !== 'python_path'}/>
          : property.type === 'array' ? <CheckboxSelect aria-label={copy.label} aria-describedby={describedBy} values={Array.isArray(value) ? value : []} options={(property.items?.enum || []).map(target => ({ value: target, label: target }))} disabled={disabled} onValuesChange={next => change(field, next)}/>
            : field === 'learning_rate' ? <TtsDecimalInput id={id} value={String(value)} aria-invalid={!!problem} aria-describedby={`${id}-hint${problem ? ` ${id}-error` : ''}`} onValueChange={next => change(field, next)}/>
            : <input id={id} type="text" inputMode="decimal" autoComplete="off" spellCheck={false} value={String(value)} aria-invalid={!!problem} aria-describedby={`${id}-hint${problem ? ` ${id}-error` : ''}`} placeholder={field === 'valid_interval' ? text('跟随保存间隔', 'Follow save interval') : undefined} onChange={event => change(field, event.target.value)}/>}</fieldset>
      <div className="config-field-footer"><p id={`${id}-hint`} className="config-field-hint">{copy.hint}</p>{problem && <p id={`${id}-error`} role="alert" className="config-field-error">{problem}</p>}</div>
    </div>;
  };
  return orderedGroups.map(group => ({ ...group, fields: group.fields.filter(field => !preset || !['engine', 'variant', 'python_path', 'trainer_path', 'model_path', 'pretrained_gpt', 'pretrained_sovits'].includes(field)) })).filter(group => group.fields.length).map(group => {
    const buckets = group.id === 'environment' ? [
      { label: text('训练模型', 'Training model'), fields: ['model_path'] },
      { label: text('运行环境', 'Runtime environment'), fields: ['python_path', 'trainer_path'] },
    ] : [{ label: undefined, fields: group.fields }];
    return { id: group.id, label: english ? group.en : group.zh, sections: buckets.map(bucket => ({ label: bucket.label,
      fields: [...(!preset && group.id === 'environment' && bucket.fields.includes('python_path') ? [{
        id: 'runtime_environment',
        search: '运行环境 自动 默认 准备 检查 外部 Python 源码 runtime environment automatic default prepare check external python_path trainer_path',
        node: <div className="config-field config-field-wide"><TtsEnvironmentPanel engine="voxcpm1.5" compact disabled={readOnly || saving} pythonPath={String(draft.python_path)} trainerPath={String(draft.trainer_path)} onUseAutomatic={() => { change('python_path', ''); change('trainer_path', ''); }}/></div>,
      }] : []), ...group.fields.filter(field => bucket.fields.includes(field) && !targetOwners[field]).map(field => {
        const copy = fieldCopy(field, english);
        const targets = Object.keys(targetOwners).filter(name => targetOwners[name as TtsField] === field);
        const targetSearch = targets.map(name => { const target = fieldCopy(name as TtsField, english); return `${name} ${target.label} ${target.hint}`; }).join(' ');
        return { id: field, advanced: voxAdvancedParameters.has(field), search: `${field} ${copy.label} ${copy.hint} ${copy.help} ${targetSearch}`, node: renderField(field) };
      })],
    })).filter(section => section.fields.length) };
  });
}
