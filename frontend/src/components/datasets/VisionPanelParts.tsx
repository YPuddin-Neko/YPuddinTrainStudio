import type { ReactNode } from 'react';
import { Undo2 } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';
import ConfigHelp from '../ConfigHelp';
import StudioSelect from '../StudioSelect';
import Switch from '../Switch';
import type { PipelineOperation } from './DatasetPipelinePanel';
import type { useScopeOptions } from './visionHooks';

export function ScopeField({ scopes, value, onChange, disabled, label, hint }: { scopes: ReturnType<typeof useScopeOptions>; value: string; onChange: (value: string) => void; disabled: boolean; label: string; hint: string }) {
  const text = useWorkspaceText();
  return <div className="vision-field"><span className="vision-field-label">{label}</span>
    <StudioSelect aria-label={label} value={value} disabled={disabled || !scopes.options.length} placeholder={scopes.loading ? text('读取数据集…', 'Loading datasets…') : text('还没有数据集', 'No datasets yet')} onValueChange={onChange} options={scopes.options}/>
    <span className="vision-field-hint">{hint}</span></div>;
}

/** A slider with an exact number box beside it. */
export function RangeField({ label, hint, value, min, max, step, disabled, onChange }: { label: string; hint: string; value: number; min: number; max: number; step: number; disabled: boolean; onChange: (value: number) => void }) {
  const text = useWorkspaceText();
  return <label className="vision-field">
    <span className="vision-field-label">{label}</span>
    <span className="vision-number"><input type="range" min={min} max={max} step={step} value={value} disabled={disabled} aria-label={label} onChange={event => onChange(Number(event.target.value))}/>
      <input type="number" min={min} max={max} step={step} value={value} disabled={disabled} aria-label={`${label} ${text('数值', 'value')}`} onChange={event => { const next = Number(event.target.value); if (event.target.value !== '' && Number.isFinite(next)) onChange(Math.min(max, Math.max(min, next))); }}/></span>
    <span className="vision-field-hint">{hint}</span>
  </label>;
}

/** GPU or CPU; shown only on machines where ONNX Runtime can use the GPU. */
export function DeviceField({ value, onChange, disabled }: { value: 'auto' | 'cpu'; onChange: (value: 'auto' | 'cpu') => void; disabled: boolean }) {
  const text = useWorkspaceText();
  return <div className="vision-field"><span className="vision-field-label">{text('运行设备', 'Device')}</span>
    <StudioSelect aria-label={text('运行设备', 'Device')} value={value} disabled={disabled} onValueChange={next => onChange(next as 'auto' | 'cpu')} options={[{ value: 'auto', label: text('显卡（空闲显存最多的一张）', 'GPU (most free memory)') }, { value: 'cpu', label: 'CPU' }]}/>
    <span className="vision-field-hint">{text('训练占满显存时可改用 CPU。', 'Use the CPU while training fills the GPU.')}</span></div>;
}

export function TagOutputOptions({ replaceUnderscore, escapeParentheses, onChange, disabled, escapeRequired = false }: {
  replaceUnderscore: boolean; escapeParentheses: boolean; escapeRequired?: boolean;
  onChange: (patch: { replaceUnderscore?: boolean; escapeParentheses?: boolean }) => void;
  disabled: boolean;
}) {
  const text = useWorkspaceText();
  return <div className="vision-inline-options" aria-label={text('标签输出格式', 'Tag output formatting')}>
    <Switch checked={replaceUnderscore} disabled={disabled} onCheckedChange={checked => onChange({ replaceUnderscore: checked })}>{text('下划线转空格', 'Replace underscores with spaces')}</Switch>
    <span className="vision-option-with-help"><Switch checked={escapeRequired || escapeParentheses} disabled={disabled || escapeRequired} onCheckedChange={checked => { if (!escapeRequired && !disabled) onChange({ escapeParentheses: checked }); }}>{text('括号转义', 'Escape parentheses')}</Switch><ConfigHelp label={text('括号转义说明', 'Bracket escaping help')}>{[
      text('在新生成的标签中，将普通圆括号写成 \\( 和 \\)。已有转义不会重复添加。', 'Writes literal parentheses as \\( and \\) in newly generated tags. Already escaped parentheses are not escaped again.'),
      ...(escapeRequired ? [text('当前启用了加权标注，普通圆括号和方括号自动转义。关闭加权标注后恢复此选项的原选择；已有标签文件不会改写。', 'Caption weights are enabled, so literal parentheses and square brackets are escaped automatically. Disabling caption weights restores your previous choice; existing caption files are not rewritten.')] : []),
    ].join('\n\n')}</ConfigHelp></span>
  </div>;
}

/** A finished run of this action, with its Undo, beside the button that started it. */
export function OperationResult({ operation, done, undoLabel, locked, onUndo, children }: { operation?: PipelineOperation; done: string; undoLabel: string; locked: boolean; onUndo: (id: string) => void; children?: ReactNode }) {
  const text = useWorkspaceText();
  if (!operation || operation.status !== 'completed') return null;
  if (operation.result.undone_by) return <p role="status" className="vision-result">{text('已撤销。', 'Undone.')}</p>;
  return <p role="status" className="vision-result"><span>{done}</span>{children}
    {operation.can_undo && <button type="button" className="ui-btn ui-btn-sm" disabled={locked} onClick={() => onUndo(operation.id)}><Undo2 size={13}/>{undoLabel}</button>}</p>;
}

/** A plain number box; an empty box means "not set" when ``optional``. */
export function NumberField({ label, hint, value, min, max, step, disabled, optional = false, placeholder, onChange }: { label: string; hint: string; value: number | null; min: number; max: number; step: number; disabled: boolean; optional?: boolean; placeholder?: string; onChange: (value: number | null) => void }) {
  return <label className="vision-field"><span className="vision-field-label">{label}</span>
    <input type="number" aria-label={label} value={value ?? ''} min={min} max={max} step={step} disabled={disabled} placeholder={placeholder} onChange={event => {
      if (event.target.value === '') { if (optional) onChange(null); return; }
      const next = Number(event.target.value);
      if (Number.isFinite(next)) onChange(Math.min(max, Math.max(min, next)));
    }}/>
    <span className="vision-field-hint">{hint}</span></label>;
}

export function SelectField({ label, hint, value, options, disabled, onChange }: { label: string; hint: string; value: string; options: { value: string; label: string }[]; disabled: boolean; onChange: (value: string) => void }) {
  return <div className="vision-field"><span className="vision-field-label">{label}</span>
    <StudioSelect aria-label={label} value={value} disabled={disabled} options={options} onValueChange={onChange}/>
    <span className="vision-field-hint">{hint}</span></div>;
}
