import type React from 'react';
import ConfigHelp from '../../components/ConfigHelp';
import Switch from '../../components/Switch';
import StudioSelect from '../../components/StudioSelect';
import { PathInput } from '../../components/PathBrowser';
import { gptSovitsFieldCopy as fieldCopy, isScientificField, type GptSovitsDraft as TtsDraft, type GptSovitsField as TtsField, type GptSovitsFieldProblem as FieldProblem, type GptSovitsFieldSchema } from './gptSovitsVersionFields';
import type { TtsParameterGroup } from './TtsParameterForm';

type Props = {
  draft: TtsDraft; properties: Record<TtsField, GptSovitsFieldSchema>; problems: FieldProblem[];
  orderedGroups: { id: string; zh: string; en: string; fields: TtsField[] }[];
  readOnly: boolean; saving: boolean; english: boolean; preset?: boolean; schemaUnavailable?: boolean;
  text: (zh: string, en: string) => string; change: (field: TtsField, value: TtsDraft[TtsField]) => void;
};
export function gptSovitsParameterGroups({ draft, properties, problems, orderedGroups, readOnly, saving, english, text, change, preset = false, schemaUnavailable = false }: Props): TtsParameterGroup[] {
  const bounds = (field: TtsField) => {
    const rule = properties[field];
    return [rule.type === 'integer' ? text('整数', 'Integer') : rule.type === 'number' ? text('数值', 'Number') : '',
      rule.minimum != null ? `≥ ${rule.minimum}` : rule.exclusiveMinimum != null ? `> ${rule.exclusiveMinimum}` : '',
      rule.maximum != null ? `≤ ${rule.maximum}` : rule.exclusiveMaximum != null ? `< ${rule.exclusiveMaximum}` : ''].filter(Boolean).join(' · ');
  };
  const optionLabel = (field: TtsField, option: string | number) => {
    if (field === 'stage') return ({ both: text('两阶段（SoVITS → GPT）', 'Both (SoVITS → GPT)'), gpt: text('仅 GPT', 'GPT only'), sovits: text('仅 SoVITS', 'SoVITS only') })[String(option) as 'both' | 'gpt' | 'sovits'];
    if (field === 'gpt.precision') return option === '16-mixed' ? text('FP16（混合精度）', 'FP16 (mixed precision)') : text('FP32（全精度）', 'FP32 (full precision)');
    if (field === 'sovits.precision') return option === 'fp16' ? 'FP16' : 'FP32';
    return String(option);
  };
  const renderField = (field: TtsField): React.ReactNode => {
    if (field === 'engine') return null;
    const property = properties[field], copy = fieldCopy(field, english), value = draft[field];
    const pathField = property.type === 'string' && !property.enum;
    const id = `tts-gsv-${field}`, problem = problems.filter(issue => issue.field === field).map(issue => issue.message).join(' ');
    const describedBy = `${copy.hint ? `${id}-hint` : ''}${problem ? ` ${id}-error` : ''}`.trim() || undefined;
    const help = [copy.help, bounds(field), isScientificField(field) ? text('支持科学计数法，例如 1e-4；数值必须大于 0。', 'Accepts scientific notation, e.g. 1e-4; the value must be greater than 0.') : ''].filter(Boolean).join('\n\n');
    if (property.type === 'boolean') return <div key={field} data-field={field} className={`config-field config-field-boolean${problem ? ' config-field-invalid' : ''}`}>
      <fieldset className="config-field-control">
        <Switch id={id} aria-label={copy.label} aria-invalid={!!problem} aria-describedby={describedBy} disabled={readOnly || saving || schemaUnavailable} checked={value === true} onCheckedChange={checked => change(field, checked)}/>
        <label htmlFor={id}>{copy.label}</label>{help && <ConfigHelp label={`${copy.label} · ${text('说明', 'Help')}`}>{help}</ConfigHelp>}
      </fieldset>
      {problem && <p id={`${id}-error`} role="alert" className="config-field-error">{problem}</p>}
    </div>;
    return <div key={field} data-field={field} className={`config-field${pathField ? ' config-field-wide' : ''}${property.type === 'boolean' ? ' config-field-boolean' : ''}${problem ? ' config-field-invalid' : ''}`}>
      <div className="config-field-heading"><label htmlFor={pathField ? undefined : id}>{copy.label}</label><span className="config-field-reference"><code tabIndex={0} className="config-field-key">{field}</code>{help && <ConfigHelp label={`${copy.label} · ${text('说明', 'Help')}`}>{help}</ConfigHelp>}</span></div>
      <fieldset className={`config-field-control${pathField ? ' model-path-control' : ''}`} disabled={readOnly || saving || schemaUnavailable}>
        {property.enum ? <StudioSelect id={id} aria-label={copy.label} aria-invalid={!!problem} aria-describedby={describedBy} value={String(value)} options={property.enum.map(option => ({ value: String(option), label: optionLabel(field, option) }))} onValueChange={next => change(field, next)}/>
            : pathField ? <PathInput ariaLabel={copy.label} value={String(value)} onChange={next => change(field, next)} directoryOnly={field === 'trainer_path' || field === 'model_path'}/>
              : <input id={id} type="text" inputMode="decimal" autoComplete="off" spellCheck={false} value={String(value)} aria-invalid={!!problem} aria-describedby={describedBy} onChange={event => change(field, event.target.value)} onBlur={() => { if (isScientificField(field) && String(value).trim() && Number.isFinite(Number(value))) change(field, Number(value).toExponential()); }}/>}</fieldset>
      <div className="config-field-footer">{copy.hint && <p id={`${id}-hint`} className="config-field-hint">{copy.hint}</p>}{problem && <p id={`${id}-error`} role="alert" className="config-field-error">{problem}</p>}</div>
    </div>;
  };
  return orderedGroups.map(group => ({ ...group, fields: group.fields.filter(field => !preset || !['engine', 'variant', 'python_path', 'trainer_path', 'model_path', 'pretrained_gpt', 'pretrained_sovits'].includes(field)) })).filter(group => group.fields.length).map(group => {
    const inactive = group.id !== 'environment' && draft.stage !== 'both' && draft.stage !== group.id;
    const buckets = group.id === 'environment' ? [
      { label: preset ? undefined : text('模型与训练阶段', 'Model and training stages'), fields: ['variant', 'stage', 'model_path', 'pretrained_gpt', 'pretrained_sovits'] },
      { label: text('运行环境', 'Runtime environment'), fields: ['python_path', 'trainer_path'] },
    ] : [
      { label: text('训练批次', 'Training batches'), fields: ['epochs', 'batch_size', 'precision', 'seed', 'max_seconds', 'num_workers'].map(field => `${group.id}.${field}`) },
      { label: text('优化器与学习率', 'Optimizer and learning rate'), fields: ['learning_rate', 'initial_learning_rate', 'final_learning_rate', 'warmup_steps', 'decay_steps', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'lr_decay'].map(field => `${group.id}.${field}`) },
      { label: text('训练选项', 'Training options'), fields: ['lora_rank', 'dpo', 'gradient_checkpointing'].map(field => `${group.id}.${field}`) },
      { label: text('保存与记录', 'Saving and logging'), fields: ['save_every_epoch', 'log_interval', 'save_latest'].map(field => `${group.id}.${field}`) },
    ];
    return { id: group.id, label: preset && group.id === 'environment' ? text('训练阶段', 'Training stages') : english ? group.en : group.zh, inactive,
      notice: <>{inactive && <p className="config-field-hint" role="status">{text('此阶段的参数会保留，本次任务不执行该阶段。', 'These settings are retained; this stage will not run in the current job.')}</p>}{problems.filter(issue => issue.field === group.id).map(issue => <p key={issue.message} role="alert" className="config-field-error">{issue.message}</p>)}</>,
      sections: buckets.map(bucket => ({ label: bucket.label, fields: group.fields.filter(field => bucket.fields.includes(field)).map(field => {
        const copy = fieldCopy(field, english);
        return { id: field, search: `${field} ${copy.label} ${copy.hint} ${copy.help}`, node: renderField(field), toggle: properties[field].type === 'boolean' };
      }) })).filter(section => section.fields.length),
    };
  });
}
