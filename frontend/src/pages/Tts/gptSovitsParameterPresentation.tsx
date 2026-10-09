import type React from 'react';
import ConfigHelp from '../../components/ConfigHelp';
import Switch from '../../components/Switch';
import StudioSelect from '../../components/StudioSelect';
import { PathInput } from '../../components/PathBrowser';
import { gptSovitsFieldCopy as fieldCopy, isScientificField, type GptSovitsDraft as TtsDraft, type GptSovitsField as TtsField, type GptSovitsFieldProblem as FieldProblem, type GptSovitsFieldSchema } from './gptSovitsVersionFields';
import type { TtsParameterGroup } from './TtsParameterForm';
import { TtsDecimalInput, TtsScientificBadge } from './ttsParameterControls';
import { gptSovitsAdvancedParameters } from './ttsParameterMetadata';
import { ignoredGptLearningRateFields } from './gptSovitsLearningRate';

type Props = {
  draft: TtsDraft; properties: Record<TtsField, GptSovitsFieldSchema>; problems: FieldProblem[];
  orderedGroups: { id: string; zh: string; en: string; fields: TtsField[] }[];
  readOnly: boolean; saving: boolean; english: boolean; preset?: boolean; schemaUnavailable?: boolean;
  fixedLearningRate?: { initial_optimizer_lr: number; after_first_scheduler_step_lr: number };
  onRestoreLegacyLearningRate?: () => void;
  text: (zh: string, en: string) => string; change: (field: TtsField, value: TtsDraft[TtsField]) => void;
};
export function gptSovitsParameterGroups({ draft, properties, problems, orderedGroups, readOnly, saving, english, text, change, preset = false, schemaUnavailable = false, fixedLearningRate, onRestoreLegacyLearningRate }: Props): TtsParameterGroup[] {
  const optionLabel = (field: TtsField, option: string | number) => {
    if (field === 'stage') return ({ both: text('两阶段（SoVITS → GPT）', 'Both (SoVITS → GPT)'), gpt: text('仅 GPT', 'GPT only'), sovits: text('仅 SoVITS', 'SoVITS only') })[String(option) as 'both' | 'gpt' | 'sovits'];
    if (field === 'gpt.precision') return option === '16-mixed' ? text('FP16（混合精度）', 'FP16 (mixed precision)') : text('FP32（全精度）', 'FP32 (full precision)');
    if (field === 'sovits.precision') return option === 'fp16' ? 'FP16' : 'FP32';
    return String(option);
  };
  const renderField = (field: TtsField): React.ReactNode => {
    if (field === 'engine') return null;
    if (field === 'gpt.learning_rate' && fixedLearningRate) {
      const legacyProblems = problems.filter(issue => ignoredGptLearningRateFields.includes(issue.field as typeof ignoredGptLearningRateFields[number]));
      const id = 'tts-gsv-fixed-learning-rate';
      return <div key={field} data-field={field} className={`config-field${legacyProblems.length ? ' config-field-invalid' : ''}`}>
        <div className="config-field-heading"><label htmlFor={id}>{text('GPT 学习率', 'GPT learning rate')}</label><span className="config-field-reference"><ConfigHelp label={text('GPT 学习率 · 说明', 'GPT learning rate · Help')}>{text(`优化器初始学习率为 ${fixedLearningRate.initial_optimizer_lr}，第一次调度后固定为 ${fixedLearningRate.after_first_scheduler_step_lr}。此模型的 GPT 学习率无法调整，不使用预热或余弦衰减。`, `The optimizer starts at ${fixedLearningRate.initial_optimizer_lr}; after the first scheduler call, its learning rate is fixed at ${fixedLearningRate.after_first_scheduler_step_lr}. This model does not expose GPT learning-rate tuning, warmup or cosine decay.`)}</ConfigHelp></span></div>
        <div className="config-field-control"><div className="config-managed-value"><output id={id}>{text(`初始 ${fixedLearningRate.initial_optimizer_lr}；调度后 ${fixedLearningRate.after_first_scheduler_step_lr}`, `Initial ${fixedLearningRate.initial_optimizer_lr}; after scheduling ${fixedLearningRate.after_first_scheduler_step_lr}`)}</output><span>{text('固定', 'Fixed')}</span></div></div>
        <div className="config-field-footer"><p className="config-field-hint">{text('控制 GPT 模型每次参数更新的幅度。', 'Controls the size of each GPT parameter update.')}</p>{legacyProblems.length > 0 && <><p role="alert" className="config-field-error">{text('旧版学习率参数无效。', 'Legacy learning-rate settings are invalid.')}</p><button type="button" data-restore-legacy-learning-rate className="ui-link" disabled={readOnly || saving || schemaUnavailable} onClick={onRestoreLegacyLearningRate}>{text('恢复原值', 'Restore values')}</button></>}</div>
      </div>;
    }
    const property = properties[field], copy = fieldCopy(field, english), value = draft[field];
    const pathField = property.type === 'string' && !property.enum;
    const id = `tts-gsv-${field}`, problem = problems.filter(issue => issue.field === field).map(issue => issue.message).join(' ');
    const describedBy = `${copy.hint ? `${id}-hint` : ''}${problem ? ` ${id}-error` : ''}`.trim() || undefined;
    const help = copy.help;
    if (property.type === 'boolean') return <div key={field} data-field={field} className={`config-field config-field-boolean${problem ? ' config-field-invalid' : ''}`}>
      <fieldset className="config-field-control">
        <Switch id={id} aria-label={copy.label} aria-invalid={!!problem} aria-describedby={describedBy} disabled={readOnly || saving || schemaUnavailable} checked={value === true} onCheckedChange={checked => change(field, checked)}/>
        <label htmlFor={id}>{copy.label}</label>{help && <ConfigHelp label={`${copy.label} · ${text('说明', 'Help')}`}>{help}</ConfigHelp>}
      </fieldset>
      {problem && <p id={`${id}-error`} role="alert" className="config-field-error">{problem}</p>}
    </div>;
    return <div key={field} data-field={field} className={`config-field${pathField ? ' config-field-wide' : ''}${property.type === 'boolean' ? ' config-field-boolean' : ''}${problem ? ' config-field-invalid' : ''}`}>
      <div className="config-field-heading"><label htmlFor={pathField ? undefined : id}>{copy.label}{isScientificField(field) && <TtsScientificBadge value={value} english={english}/>}</label><span className="config-field-reference"><code tabIndex={0} className="config-field-key">{field}</code>{help && <ConfigHelp label={`${copy.label} · ${text('说明', 'Help')}`}>{help}</ConfigHelp>}</span></div>
      <fieldset className={`config-field-control${pathField ? ' model-path-control' : ''}`} disabled={readOnly || saving || schemaUnavailable}>
        {property.enum ? <StudioSelect id={id} aria-label={copy.label} aria-invalid={!!problem} aria-describedby={describedBy} value={String(value)} options={property.enum.map(option => ({ value: String(option), label: optionLabel(field, option) }))} onValueChange={next => change(field, next)}/>
            : pathField ? <PathInput ariaLabel={copy.label} value={String(value)} onChange={next => change(field, next)} directoryOnly={field === 'trainer_path' || field === 'model_path'}/>
              : isScientificField(field) ? <TtsDecimalInput id={id} value={String(value)} aria-invalid={!!problem} aria-describedby={describedBy} onValueChange={next => change(field, next)}/>
              : <input id={id} type="text" inputMode="decimal" autoComplete="off" spellCheck={false} value={String(value)} aria-invalid={!!problem} aria-describedby={describedBy} onChange={event => change(field, event.target.value)}/>}</fieldset>
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
      sections: buckets.map(bucket => ({ label: bucket.label, fields: group.fields.filter(field => bucket.fields.includes(field) && (!fixedLearningRate || field === 'gpt.learning_rate' || !ignoredGptLearningRateFields.includes(field as typeof ignoredGptLearningRateFields[number]))).map(field => {
        const copy = fieldCopy(field, english);
        const legacySearch = field === 'gpt.learning_rate' && fixedLearningRate ? ignoredGptLearningRateFields.map(key => { const oldCopy = fieldCopy(key, english); return `${key} ${oldCopy.label}`; }).join(' ') : '';
        return { id: field, advanced: gptSovitsAdvancedParameters.has(field), search: `${field} ${copy.label} ${copy.hint} ${copy.help} ${legacySearch}`, node: renderField(field), toggle: properties[field].type === 'boolean' };
      }) })).filter(section => section.fields.length),
    };
  });
}
