import React from 'react';
import { evaluateShowWhen } from '../showWhen';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronRight, Plus, Trash2, ArrowUp, ArrowDown, FolderOpen, HelpCircle } from 'lucide-react';
import { PathInput, PathPickerModal } from '../../components/PathBrowser';
import { apiClient } from '../../api/client';
import { FamilyInfo } from '../../api/types';
import { configFieldLabel, configOptionLabel } from '../../utils/configPresentation';
import NumericControl from './NumericControl';
import StudioSelect from '../../components/StudioSelect';

interface SchemaProperty {
  type?: string;
  title?: string;
  description?: string;
  default?: any;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  enum?: any[];
  anyOf?: SchemaProperty[];
  items?: SchemaProperty;
  properties?: Record<string, SchemaProperty>;
  additionalProperties?: boolean | SchemaProperty;
  const?: any;
  $ref?: string;
  'x-ui'?: {
    group?: string;
    order?: number;
    advanced?: boolean;
    control?: string;
    unit?: string;
    help?: string;
    show_when?: string;
    min?: number;
    max?: number;
    step?: number;
  };
}

export interface ValidationError {
  loc?: string;
  msg: string;
}

interface SchemaFormProps {
  schema: any;
  value: Record<string, any>;
  onChange: (newValue: Record<string, any>) => void;
  showAdvanced?: boolean;
  errors?: ValidationError[];
  /** 当前模型族信息（GET /api/families），驱动 preset 下拉 / text_modes / weights 提示 / sampling 默认值 */
  family?: FamilyInfo;
  families?: FamilyInfo[];
  compact?: boolean;
  groupFilter?: string[];
  search?: string;
}

const resolveRef = (rootSchema: any, refPath: string) => {
  if (!refPath || !refPath.startsWith('#/')) return null;
  const parts = refPath.substring(2).split('/');
  let current = rootSchema;
  for (const part of parts) {
    current = current?.[part];
    if (!current) return null;
  }
  return current;
};

const getNestedValue = (obj: any, path: string[]) => {
  let current = obj;
  for (const key of path) {
    if (current === undefined || current === null) return undefined;
    current = current[key];
  }
  return current;
};

const setNestedValue = (obj: any, path: string[], value: any): any => {
  const newObj = { ...obj };
  let current = newObj;
  for (let i = 0; i < path.length - 1; i++) {
    const key = path[i];
    current[key] = { ...(current[key] || {}) };
    current = current[key];
  }
  current[path[path.length - 1]] = value;
  return newObj;
};

// 2. Rules 专用组件 (AdapterRule 列表)
const RulesEditor: React.FC<{
  value: any[];
  onChange: (val: any[]) => void;
}> = ({ value = [], onChange }) => {
  const { t } = useTranslation();
  const addRule = () => {
    onChange([
      ...value,
      { match: '', algo: 'lokr', rank: 16, alpha: 16, factor: -1, lr: null },
    ]);
  };

  const removeRule = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  const moveRule = (index: number, direction: 'up' | 'down') => {
    const targetIndex = direction === 'up' ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= value.length) return;
    const newRules = [...value];
    const temp = newRules[index];
    newRules[index] = newRules[targetIndex];
    newRules[targetIndex] = temp;
    onChange(newRules);
  };

  const updateRule = (index: number, field: string, val: any) => {
    const newRules = [...value];
    newRules[index] = { ...newRules[index], [field]: val };
    onChange(newRules);
  };

  return (
    <div className="space-y-3" data-testid="rules-editor">
      {value.map((rule, idx) => (
        <div key={idx} className="flex flex-wrap items-center gap-2 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 text-xs">
          <input
            type="text"
            placeholder={t('train.matchPlaceholder', 'Match 正则 / glob')}
            value={rule.match || ''}
            onChange={(e) => updateRule(idx, 'match', e.target.value)}
            className="flex-1 min-w-[140px] px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <StudioSelect aria-label={`${t('train.algo', '算法')} ${idx+1}`} value={rule.algo ?? ''} onValueChange={value => updateRule(idx,'algo',value || null)}
            options={[{value:'',label:t('train.inherit')},...['none','lora','lokr','loha','full'].map(value=>({value,label:value}))]}/>

          <input
            type="text"
            placeholder="Rank / full"
            value={rule.rank ?? ''}
            onChange={(e) => {
              const val = e.target.value === '' ? null : e.target.value === 'full' ? 'full' : Number(e.target.value);
              updateRule(idx, 'rank', val);
            }}
            className="w-16 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <input
            type="number"
            placeholder="Alpha"
            value={rule.alpha ?? ''}
            onChange={(e) => updateRule(idx, 'alpha', e.target.value === '' ? null : Number(e.target.value))}
            className="w-16 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <input
            type="number"
            placeholder="Factor"
            value={rule.factor ?? ''}
            onChange={(e) => updateRule(idx, 'factor', e.target.value === '' ? null : Number(e.target.value))}
            className="w-16 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <input
            type="number"
            placeholder="LR"
            step="0.0001"
            value={rule.lr ?? ''}
            onChange={(e) => updateRule(idx, 'lr', e.target.value === '' ? null : Number(e.target.value))}
            className="w-20 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <div className="flex items-center space-x-1">
            <button type="button" onClick={() => moveRule(idx, 'up')} disabled={idx === 0} className="p-1 text-slate-400 hover:text-slate-600 disabled:opacity-30">
              <ArrowUp className="w-3.5 h-3.5" />
            </button>
            <button type="button" onClick={() => moveRule(idx, 'down')} disabled={idx === value.length - 1} className="p-1 text-slate-400 hover:text-slate-600 disabled:opacity-30">
              <ArrowDown className="w-3.5 h-3.5" />
            </button>
            <button type="button" onClick={() => removeRule(idx)} className="p-1 text-red-500 hover:text-red-700">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      ))}
      <button
        type="button"
        onClick={addRule}
        data-testid="add-rule"
        className="flex items-center space-x-1 px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded border border-slate-300 dark:border-slate-600"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>{t('train.addRule', '添加规则')}</span>
      </button>
    </div>
  );
};

// 3. Key-Value 自由对象编辑器 (additionalProperties)
const KeyValueEditor: React.FC<{
  value: Record<string, any>;
  onChange: (val: Record<string, any>) => void;
}> = ({ value = {}, onChange }) => {
  const { t, i18n } = useTranslation();
  const [keyDrafts, setKeyDrafts] = React.useState<Record<string,string>>({});
  const [keyError, setKeyError] = React.useState('');
  const entries = Object.entries(value);

  const addEntry = () => {
    onChange({ ...value, [`key_${Date.now()}`]: 1.0 });
  };

  const updateKey = (oldKey: string, rawKey: string) => {
    const newKey = rawKey.trim();
    if (!newKey || (newKey !== oldKey && Object.prototype.hasOwnProperty.call(value, newKey))) {
      setKeyError(i18n.language.startsWith('en') ? 'Keys must be non-empty and unique.' : '参数名不能为空或重复，请重新填写。');
      setKeyDrafts(drafts => ({...drafts, [oldKey]:oldKey})); return;
    }
    setKeyError('');
    if (oldKey === newKey) return;
    onChange(Object.fromEntries(Object.entries(value).map(([key, val]) => [key === oldKey ? newKey : key, val])));
  };

  const updateVal = (key: string, val: any) => {
    onChange({ ...value, [key]: val });
  };

  const removeEntry = (key: string) => {
    const newObj = { ...value };
    delete newObj[key];
    onChange(newObj);
  };

  return (
    <div className="space-y-2" data-testid="key-value-editor">
      {entries.map(([k, v]) => (
        <div key={k} className="flex items-center space-x-2 text-xs">
          <input
            type="text"
            value={keyDrafts[k] ?? k}
            aria-label={`${t('train.keyPlaceholder', '键')} ${k}`}
            onChange={event => setKeyDrafts(drafts => ({...drafts,[k]:event.target.value}))}
            onBlur={(e) => updateKey(k, e.target.value)}
            className="flex-1 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600 font-mono"
            placeholder={t('train.keyPlaceholder', '键')}
          />
          <input
            type="text"
            value={typeof v === 'object' ? JSON.stringify(v) : v ?? ''}
            aria-label={`${t('train.valuePlaceholder', '值')} ${k}`}
            onChange={(e) => {
              let next: unknown = e.target.value;
              try { next = JSON.parse(e.target.value); } catch { /* String values remain strings. */ }
              updateVal(k, next);
            }}
            className="flex-1 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600 font-mono"
            placeholder={t('train.valuePlaceholder', '值')}
          />
          <button type="button" aria-label={`${t('common.remove', '移除')} ${k}`} onClick={() => removeEntry(k)} className="text-red-500 hover:text-red-700 p-1">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
      {keyError && <p role="alert" className="text-xs text-amber-600">{keyError}</p>}
      <button
        type="button"
        onClick={addEntry}
        data-testid="add-property"
        className="flex items-center space-x-1 px-2 py-1 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded border border-slate-300 dark:border-slate-600"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>{t('train.addProperty', '添加属性')}</span>
      </button>
    </div>
  );
};

// 4. Data Sources 列表控件 (DatasetSourceConfig 列表)
const SourcesEditor: React.FC<{
  value: any[];
  onChange: (val: any[]) => void;
}> = ({ value = [], onChange }) => {
  const { t } = useTranslation();
  const [modalIndex, setModalIndex] = React.useState<number | null>(null);

  const addSource = () => {
    onChange([
      ...value,
      { path: '', repeats: 1, caption_ext: '.txt', is_reg: false, prior_weight: 1.0, class_prompt: '' },
    ]);
  };

  const removeSource = (idx: number) => {
    onChange(value.filter((_, i) => i !== idx));
  };

  const updateSource = (idx: number, field: string, val: any) => {
    const next = [...value];
    next[idx] = { ...next[idx], [field]: val };
    onChange(next);
  };

  return (
    <div className="space-y-3" data-testid="sources-editor">
      {value.map((src, idx) => (
        <div key={idx} className="p-3 bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2 text-xs">
          <div className="flex justify-between items-center">
            <span className="font-semibold text-slate-700 dark:text-slate-300">
              {t('train.sourceN', { n: idx + 1, defaultValue: '数据源 #{n}' })}
            </span>
            <button type="button" onClick={() => removeSource(idx)} className="text-red-500 hover:text-red-700">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="flex space-x-2">
            <input
              type="text"
              placeholder="/path/to/dataset"
              value={src.path || ''}
              onChange={(e) => updateSource(idx, 'path', e.target.value)}
              className="flex-1 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600 font-mono"
            />
            <button
              type="button"
              onClick={() => setModalIndex(idx)}
              className="px-2 py-1 bg-slate-200 dark:bg-slate-700 rounded hover:bg-slate-300 dark:hover:bg-slate-600 flex items-center space-x-1"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              <span>{t('common.browse')}</span>
            </button>
          </div>
          <label className="block space-y-1"><span className="text-[10px] text-slate-400">{t('projectDetail.classPrompt')}</span>
            <input type="text" value={src.class_prompt ?? ''} onChange={(e) => updateSource(idx, 'class_prompt', e.target.value || null)} className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600" />
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            <div>
              <label className="text-[10px] text-slate-400">{t('dataset.repeats')}</label>
              <input
                type="number"
                value={src.repeats ?? 1}
                onChange={(e) => updateSource(idx, 'repeats', Number(e.target.value))}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400">{t('projectDetail.captionExt')}</label>
              <input
                type="text"
                value={src.caption_ext || '.txt'}
                onChange={(e) => updateSource(idx, 'caption_ext', e.target.value)}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400">{t('projectDetail.priorWeight')}</label>
              <input
                type="number"
                step="0.1"
                value={src.prior_weight ?? 1.0}
                onChange={(e) => updateSource(idx, 'prior_weight', Number(e.target.value))}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
            <div className="flex items-center space-x-2 pt-3">
              <input
                type="checkbox"
                checked={!!src.is_reg}
                onChange={(e) => updateSource(idx, 'is_reg', e.target.checked)}
                className="rounded text-blue-600"
              />
              <span className="text-[11px]">{t('projectDetail.regularization')}</span>
            </div>
          </div>
        </div>
      ))}
      <button
        type="button"
        onClick={addSource}
        data-testid="add-source"
        className="flex items-center space-x-1 px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded border border-slate-300 dark:border-slate-600"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>{t('train.addSource', '添加数据集源')}</span>
      </button>

      {modalIndex !== null && (
        <PathPickerModal
          isOpen={true}
          initialPath={value[modalIndex]?.path || '/'}
          onSelect={(p) => updateSource(modalIndex, 'path', p)}
          onClose={() => setModalIndex(null)}
        />
      )}
    </div>
  );
};

// 5. Prompts 专用组件 (SamplePrompt 列表)
const PromptsEditor: React.FC<{
  value: any[];
  inheritedValues?: Record<string, number | null | undefined>;
  onChange: (val: any[]) => void;
}> = ({ value = [], inheritedValues = {}, onChange }) => {
  const { t } = useTranslation();
  const addPrompt = () => {
    onChange([...value, { prompt: '', negative: '', seed: null, width: null, height: null, steps: null, cfg: null }]);
  };

  const removePrompt = (idx: number) => {
    onChange(value.filter((_, i) => i !== idx));
  };

  const updatePrompt = (idx: number, field: string, val: any) => {
    const next = [...value];
    next[idx] = { ...next[idx], [field]: val };
    onChange(next);
  };

  return (
    <div className="space-y-3" data-testid="prompts-editor">
      {value.map((p, idx) => (
        <div key={idx} className="p-3 bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2 text-xs">
          <div className="flex justify-between items-center">
            <span className="font-semibold text-slate-700 dark:text-slate-300">
              {t('train.promptN', { n: idx + 1, defaultValue: '提示词 #{n}' })}
            </span>
            <button type="button" onClick={() => removePrompt(idx)} className="text-red-500 hover:text-red-700">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
          <textarea
            rows={2}
            placeholder={t('train.promptPlaceholder', '提示词文本')}
            value={p.prompt || ''}
            onChange={(e) => updatePrompt(idx, 'prompt', e.target.value)}
            className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <textarea
            rows={1}
            placeholder={t('train.negativePromptPlaceholder', '负面提示词（可选）')}
            value={p.negative || ''}
            onChange={(e) => updatePrompt(idx, 'negative', e.target.value)}
            className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {[
              { key: 'seed', label: t('job.seed'), step: 1 },
              { key: 'width', label: t('train.width'), step: 1, min: 32, max: 8192 },
              { key: 'height', label: t('train.height'), step: 1, min: 32, max: 8192 },
              { key: 'steps', label: t('train.sampleSteps'), step: 1, min: 1, max: 1000 },
              { key: 'cfg', label: 'CFG', step: 'any', min: 0 },
            ].map(({ key, label, step, min, max }) => (
              <label key={key} className="min-w-0 text-[10px] text-slate-400">
                {label}
                <input
                  aria-label={`sampling.prompts.${idx}.${key}`}
                  type="number" step={step} min={min} max={max}
                  value={p[key] ?? ''}
                  placeholder={inheritedValues[key] == null ? t('train.inherit') : `${t('train.inherit')} (${inheritedValues[key]})`}
                  onChange={(e) => updatePrompt(idx, key, e.target.value === '' ? null : Number(e.target.value))}
                  className="w-full px-2 py-1 border rounded text-xs dark:bg-slate-800 dark:border-slate-600"
                />
              </label>
            ))}
          </div>
          <p className="text-[10px] text-slate-500">{t('train.promptInheritHint')}</p>
        </div>
      ))}
      <button
        type="button"
        onClick={addPrompt}
        data-testid="add-prompt"
        className="flex items-center space-x-1 px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded border border-slate-300 dark:border-slate-600"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>{t('train.addPrompt', '添加采样提示词')}</span>
      </button>
    </div>
  );
};

// 6. Betas 控件 (长度 2 的数字数组)
const BetasEditor: React.FC<{
  value: [number, number];
  onChange: (val: [number, number]) => void;
}> = ({ value = [0.9, 0.999], onChange }) => {
  return (
    <div className="flex space-x-2">
      <input
        type="number"
        step="0.001"
        value={value[0] ?? 0.9}
        onChange={(e) => onChange([Number(e.target.value), value[1]])}
        className="w-1/2 px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
        placeholder="beta1"
      />
      <input
        type="number"
        step="0.0001"
        value={value[1] ?? 0.999}
        onChange={(e) => onChange([value[0], Number(e.target.value)])}
        className="w-1/2 px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
        placeholder="beta2"
      />
    </div>
  );
};

// 模型路径输入：PathInput + 从已注册模型权重快速选择
const ModelPathInput: React.FC<{
  value: string;
  kind: string | null;
  onChange: (val: string) => void;
  label?: string;
  familyName?: string;
}> = ({ value, kind, onChange, label, familyName }) => {
  const { t } = useTranslation();
  const [models, setModels] = React.useState<Array<{ id: string; path: string; kind: string; family: string }>>([]);

  React.useEffect(() => {
    if (!kind) return;
    apiClient
      .get<Array<{ id: string; path: string; kind: string; family: string }>>('/models')
      .then((list) => setModels(Array.isArray(list) ? list : []))
      .catch(() => setModels([]));
  }, [kind]);

  const matched = kind ? models.filter((m) => m.kind === kind && (!familyName || m.family === familyName)) : [];

  return (
    <div className="space-y-1.5">
      <PathInput ariaLabel={label} value={value} onChange={onChange} />
      {matched.length > 0 && (
        <StudioSelect aria-label={`${label || kind} · ${t('models.fromRegistry')}`} value="" onValueChange={onChange} data-testid="model-registry-select"
          options={[{value:'',label:t('models.fromRegistry'),disabled:true},...matched.map(model=>({value:model.path,label:`[${model.family}] ${model.path}`}))]}/>

      )}
    </div>
  );
};

function ResolutionInput({value, onChange, label}: {value: number[] | string; onChange: (next: number[] | string) => void; label: string}) {
  const {i18n} = useTranslation();
  const english = i18n.language.startsWith('en');
  const encoded = Array.isArray(value) ? value.join(', ') : String(value || '');
  const [draft, setDraft] = React.useState(encoded);
  React.useEffect(() => setDraft(encoded), [encoded]);
  const update = (raw: string) => {
    setDraft(raw);
    const tokens = raw.replace(/[\u005b\u005d]/g, '').split(/[,，\s]+/).filter(Boolean);
    onChange(tokens.length && tokens.every(token => /^\d+$/.test(token)) ? tokens.map(Number) : raw);
  };
  return <div className="resolution-editor"><input aria-label={label} value={draft} onChange={event => update(event.target.value)} placeholder="512, 768, 1024"/><div>{[512,768,1024].map(size => <button type="button" key={size} aria-label={`${english ? 'Use resolution' : '使用分辨率'} ${size}`} onClick={() => update(String(size))}>{size}</button>)}</div><span>{english ? 'Separate multiple resolutions with commas.' : '多个分辨率用逗号分隔'}</span></div>;
}

/** Nullable unions retain their actual scalar/object type and explicit null value. */
const SchemaValueInput: React.FC<{
  schema: any; property: SchemaProperty; value: any; name: string; placeholder?: string; compact?: boolean; onChange: (value: any) => void;
}> = ({ schema, property, value, name, placeholder, compact = false, onChange }) => {
  const { t, i18n } = useTranslation();
  const alternatives = property.anyOf || [property];
  const nullable = alternatives.some((p) => p.type === 'null');
  const constant = alternatives.find((p) => p.const !== undefined);
  const branch = alternatives.find((p) => p.type !== 'null' && p.const === undefined) || alternatives[0];
  const resolved = branch.$ref ? { ...resolveRef(schema, branch.$ref), ...branch } : branch;
  const prop: SchemaProperty = { ...property, ...resolved };
  const cls = 'w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600';
  const initialValue = () => prop.default ?? (prop.type === 'object' ? {} : prop.type === 'boolean' ? false : prop.type === 'string' ? '' : prop.type === 'array' ? [] : 0);
  let input: React.ReactNode;
  if (prop.type === 'object' && prop.properties) {
    input = value == null ? null : <div className="space-y-3 border-l pl-3 dark:border-slate-600">
      {Object.entries(prop.properties).map(([key, child]) => <label key={key} className="block space-y-1 text-xs">
        <span>{t(`fields.${key}`, child.title || key)}</span>
        <SchemaValueInput schema={schema} property={child} value={value[key] === undefined ? child.default : value[key]}
          name={`${name}.${key}`} compact={compact} onChange={(next) => onChange({ ...value, [key]: next })} />
      </label>)}
    </div>;
  } else if (prop.enum) {
    input = <StudioSelect aria-label={name} value={value == null ? '' : String(value)} onValueChange={next => onChange(next === '' && nullable ? null : prop.enum!.find(item => String(item) === next))}
      options={[...(nullable ? [{value:'',label:t('train.unset')}] : []),...prop.enum.map(item=>({value:String(item),label:String(item)}))]}/>;
  } else if (prop.type === 'boolean') {
    input = <input type="checkbox" aria-label={name} checked={!!value} onChange={(e) => onChange(e.target.checked)} />;
  } else if (prop.type === 'array') {
    input = <textarea className={cls} aria-label={name} value={typeof value === 'string' ? value : JSON.stringify(value ?? [])}
      onChange={(e) => { try { onChange(JSON.parse(e.target.value)); } catch { onChange(e.target.value); } }} />;
  } else {
    const numeric = prop.type === 'integer' || prop.type === 'number';
    input = <input className={cls} aria-label={name} type={numeric ? 'number' : 'text'}
      value={constant && value === constant.const ? '' : value ?? ''} disabled={!!constant && value === constant.const}
      min={(prop as any).minimum ?? prop['x-ui']?.min} max={(prop as any).maximum ?? prop['x-ui']?.max}
      step={prop['x-ui']?.step ?? (prop.type === 'integer' ? 1 : 'any')} placeholder={placeholder}
      onChange={(e) => onChange(e.target.value === '' && nullable ? null : numeric ? Number(e.target.value) : e.target.value)} />;
  }
  return <div className={compact ? 'config-union' : 'space-y-2'}>
    {nullable && <label className="flex items-center gap-2 text-xs text-slate-500">
      <input type="checkbox" aria-label={`${name}.unset`} checked={value == null}
        onChange={(e) => onChange(e.target.checked ? null : initialValue())} /><span title={t('train.unset')}>{compact ? (i18n.resolvedLanguage?.startsWith('en') ? 'Unset' : '不设置') : t('train.unset')}</span>
    </label>}
    {input}
    {constant && <label className="flex items-center gap-2 text-xs"><input type="checkbox" aria-label={`${name}.${constant.const}`}
      checked={value === constant.const} onChange={(e) => onChange(e.target.checked ? constant.const : property.default ?? 16)} />{String(constant.const)}</label>}
  </div>;
};

// 分组组件（header 右侧显示该组当前可见字段数）
const FieldGroup: React.FC<{
  title: string;
  count?: number;
  children: React.ReactNode;
  compact?: boolean;
  groupKey?: string;
}> = ({ title, count, children, compact = false, groupKey }) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = React.useState(true);
  return (
    <section data-group={groupKey} className={compact ? `config-group ${['model', 'dataset', 'caption', 'sampling', 'validation'].includes(groupKey || '') ? 'config-group-wide' : ''}` : 'border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800'}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        className={compact ? 'config-group-title' : 'w-full flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors'}
      >
        <span className="font-medium text-slate-700 dark:text-slate-200">{title}</span>
        <span className="flex items-center space-x-2">
          {typeof count === 'number' && (
            <span className="text-xs text-slate-400 font-mono" data-testid="group-count">
              {t('groups.fieldCount', { count, defaultValue: '{count} 项' })}
            </span>
          )}
          {isOpen ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
        </span>
      </button>
      {isOpen && <div className={compact ? 'config-fields' : 'p-4 space-y-4'}>{children}</div>}
    </section>
  );
};

export const SchemaForm: React.FC<SchemaFormProps> = ({
  schema,
  value,
  onChange,
  showAdvanced = false,
  errors = [],
  family,
  families,
  compact = false,
  groupFilter,
  search = '',
}) => {
  const { t, i18n } = useTranslation();
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  const groups: Record<string, { order: number; fields: React.ReactNode[] }> = {};
  const conditionValue = { ...value, dataset: { resolution_mode: 'bucket', ...value.dataset } };

  const renderField = (key: string, prop: SchemaProperty, parentPath: string[] = []) => {
    const path = [...parentPath, key];
    const fullPathKey = path.join('.');
    // Keep legacy cloud-log data in the draft, but do not expose controls that enable it.
    if (fullPathKey === 'logging.wandb' || fullPathKey.startsWith('logging.wandb.')) return null;
    const ui = { ...(prop['x-ui'] || {}), ...(compact && fullPathKey === 'dataset.batch_size' ? {group:'loop'} : {}) };
    if (conditionValue.dataset.resolution_mode === 'native' && ['dataset.resolutions', 'dataset.aspect_ratio_limit', 'dataset.area_tolerance', 'dataset.bucket_step', 'dataset.bucket_no_upscale'].includes(fullPathKey)) return null;
    if (compact && !showAdvanced && fullPathKey === 'adapter.rules' && !value.adapter?.rules?.length) return null;

    const nested = prop.$ref ? resolveRef(schema, prop.$ref) : prop;
    if (nested?.type === 'object' && nested.properties) {
      Object.entries(nested.properties).forEach(([childKey, child]) => renderField(childKey, child as SchemaProperty, path));
      return null;
    }
    const fieldLabel = configFieldLabel(fullPathKey, t(`fields.${key}`, prop.title || key), english);
    const fieldId = `config-${fullPathKey}`;
    const currentGroup = ui.group || parentPath[0] || 'default';
    if (groupFilter && !groupFilter.includes(currentGroup)) return null;
    if (search && !`${fieldLabel} ${fullPathKey} ${prop.description || ''}`.toLowerCase().includes(search.toLowerCase())) return null;

    if (ui.advanced && !showAdvanced) return null;
    if (ui.show_when) {
      try {
        if (!evaluateShowWhen(ui.show_when, conditionValue)) return null;
      } catch (err) {
        console.error(`[SchemaForm] Failed to evaluate show_when for ${fullPathKey}: "${ui.show_when}"`, err);
        // On evaluation error, default to showing the field
      }
    }

    const errorItem = errors.find((e) => e.loc === fullPathKey || e.loc?.startsWith(`${fullPathKey}.`));
    const fieldValue = getNestedValue(value, path) !== undefined ? getNestedValue(value, path) : prop.default;
    const groupName = ui.group || (parentPath.length > 0 ? parentPath[0] : 'default');
    // These fields are probabilities/fractions, unlike EMA decay, timesteps, and
    // warmup's mixed steps-or-ratio contract, which retain their native units.
    const percentage = ['adapter.dropout', 'adapter.rank_dropout', 'adapter.module_dropout', 'dataset.area_tolerance', 'dataset.caption.tag_dropout', 'dataset.caption.caption_dropout', 'scheduler.min_lr_ratio', 'validation.split_ratio'].includes(fullPathKey);
    const numericMin = ui.min ?? prop.minimum ?? prop.exclusiveMinimum;
    const numericMax = ui.max ?? prop.maximum ?? prop.exclusiveMaximum;

    let control = null;

    // 1. 递归对象渲染
    if (compact && fullPathKey === 'dataset.resolutions') {
      control = <ResolutionInput label={fieldLabel} value={fieldValue} onChange={next => onChange(setNestedValue(value, path, next))} />;
    } else if (prop.type === 'object' && prop.properties) {
      control = (
        <div className="pl-4 border-l-2 border-slate-200 dark:border-slate-700 space-y-4">
          {Object.entries(prop.properties).map(([subKey, subProp]) =>
            renderField(subKey, subProp, path)
          )}
        </div>
      );
    } else if (prop.$ref) {
      const resolved = resolveRef(schema, prop.$ref);
      if (resolved && resolved.type === 'object' && resolved.properties) {
        control = (
          <div className="pl-4 border-l-2 border-slate-200 dark:border-slate-700 space-y-4">
            {Object.entries(resolved.properties).map(([subKey, subProp]) =>
              renderField(subKey, subProp as SchemaProperty, path)
            )}
          </div>
        );
      }
    }
    // 2. 专用控件判断
    else if (ui.control === 'rules' || key === 'rules') {
      control = (
        <RulesEditor
          value={fieldValue || []}
          onChange={(val) => onChange(setNestedValue(value, path, val))}
        />
      );
    } else if (ui.control === 'prompts' || key === 'prompts') {
      control = (
        <PromptsEditor
          value={fieldValue || []}
          inheritedValues={{
            ...value.sampling,
            steps: value.sampling?.steps ?? family?.sampling?.steps,
            cfg: value.sampling?.cfg ?? family?.sampling?.cfg,
          }}
          onChange={(val) => onChange(setNestedValue(value, path, val))}
        />
      );
    } else if (fullPathKey.includes('sources') && prop.type === 'array') {
      control = (
        <SourcesEditor
          value={fieldValue || []}
          onChange={(val) => onChange(setNestedValue(value, path, val))}
        />
      );
    } else if (key === 'betas' || (prop.type === 'array' && (prop as any).maxItems === 2)) {
      control = (
        <BetasEditor
          value={fieldValue || [0.9, 0.999]}
          onChange={(val) => onChange(setNestedValue(value, path, val))}
        />
      );
    } else if (
      prop.type === 'object' &&
      prop.additionalProperties &&
      !prop.properties
    ) {
      control = (
        <KeyValueEditor
          value={fieldValue || {}}
          onChange={(val) => onChange(setNestedValue(value, path, val))}
        />
      );
    } else if (ui.control === 'path' || prop.type === 'string' && key.endsWith('path')) {
      // 模型权重字段（dit_path / text_encoder_path / vae_path / tokenizer_path）支持从注册表快速选择
      const modelKind = key === 'dit_path' ? 'dit'
        : key === 'text_encoder_path' ? 'text_encoder'
        : key === 'vae_path' ? 'vae'
        : key === 'tokenizer_path' ? 'tokenizer'
        : null;
      control = (
        <ModelPathInput
          value={fieldValue || ''}
          kind={modelKind}
          label={fieldLabel}
          familyName={value.model?.family}
          onChange={(val) => onChange(setNestedValue(value, path, val === '' && prop.anyOf?.some((p) => p.type === 'null') ? null : val))}
        />
      );
    } else if (prop.anyOf) {
      control = <SchemaValueInput schema={schema} property={prop} value={fieldValue} name={fullPathKey} compact={compact}
        placeholder={family && fullPathKey === 'sampling.shift' ? (family.sampling?.shift != null ? String(family.sampling.shift) : t('sampling.shiftAuto')) : undefined}
        onChange={(val) => onChange(setNestedValue(value, path, val))} />;
    } else if (fullPathKey === 'model.family' && families?.length) {
      control = <StudioSelect aria-label="model.family" value={fieldValue || families[0].name}
        onValueChange={next => onChange(setNestedValue(value,path,next))} options={families.map(item=>({value:item.name,label:item.label || item.name}))}/>;
    } else if (fullPathKey === 'adapter.preset' && family) {
      // 族内预设下拉：name — description（N 层）
      control = (
        <StudioSelect aria-label={fieldLabel} value={fieldValue || family.default_preset || ''} data-testid="adapter-preset-select"
          onValueChange={next => onChange(setNestedValue(value,path,next))} options={(family.presets || []).map(preset=>({value:preset.name,label:`${preset.name} — ${preset.description}（${t('preset.layers',{n:preset.layers})}）`}))}/>

      );
    } else if (fullPathKey === 'dataset.text_encoding' && family) {
      // 文本编码选项受族 text_modes 约束（krea2 无 online）
      control = (
        <div className="space-y-1">
          <StudioSelect aria-label={fieldLabel} value={fieldValue || 'auto'} data-testid="text-encoding-select"
            onValueChange={next => onChange(setNestedValue(value,path,next))} options={(family.text_modes || []).map(mode=>({value:mode,label:t(`textMode.${mode}`,mode)}))}/>

          {(family.text_modes || []).length === 2 && !family.text_modes.includes('online') && (
            <p className="text-[11px] text-slate-400">{t('textMode.autoOnly')}</p>
          )}
        </div>
      );
    } else if (prop.enum) {
      control = (
        <StudioSelect aria-label={fieldLabel} value={fieldValue == null ? '' : String(fieldValue)}
          onValueChange={next => onChange(setNestedValue(value,path,prop.enum!.find(option=>String(option)===next)))}
          options={prop.enum.map(option=>({value:String(option),label:configOptionLabel(fullPathKey,String(option),english)}))}/>

      );
    } else if (prop.type === 'boolean' || ui.control === 'switch') {
      control = (
        <input
          type="checkbox"
          className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
          checked={!!fieldValue}
          onChange={(e) => onChange(setNestedValue(value, path, e.target.checked))}
        />
      );
    } else if (prop.type === 'integer' || prop.type === 'number') {
      if ((percentage || ui.control === 'slider') && numericMin !== undefined && numericMax !== undefined) {
        control = <NumericControl id={fieldId} label={fieldLabel} value={fieldValue} percentage={percentage} unit={ui.unit}
          min={numericMin} max={numericMax} step={ui.step ?? (prop.type === 'integer' ? 1 : 0.01)} invalid={!!errorItem}
          sliderLabel={english ? 'slider' : '滑条'} onChange={next => onChange(setNestedValue(value, path, next))}/>;
      } else {
        // sampling.steps / cfg / shift 用族默认做 placeholder（shift=null 时显示"自动（按分辨率）"）
        let placeholder: string | undefined;
        if (family && fullPathKey === 'sampling.steps' && family.sampling?.steps != null) {
          placeholder = String(family.sampling.steps);
        } else if (family && fullPathKey === 'sampling.cfg' && family.sampling?.cfg != null) {
          placeholder = String(family.sampling.cfg);
        } else if (family && fullPathKey === 'sampling.shift') {
          placeholder =
            family.sampling?.shift != null ? String(family.sampling.shift) : t('sampling.shiftAuto');
        }
        control = (
          <input
            type="number"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
            value={fieldValue ?? ''}
            min={numericMin}
            max={numericMax}
            step={ui.step ?? (prop.type === 'integer' ? 1 : 'any')}
            placeholder={placeholder}
            onChange={(e) => {
              const val = e.target.value === '' ? undefined : Number(e.target.value);
              onChange(setNestedValue(value, path, val));
            }}
          />
        );
      }
    } else if (prop.type === 'array') {
      control = (
        <textarea
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600 font-mono"
          rows={3}
          value={typeof fieldValue === 'string' ? fieldValue : JSON.stringify(fieldValue || [])}
          onChange={(e) => {
            try {
              const val = JSON.parse(e.target.value);
              onChange(setNestedValue(value, path, val));
            } catch {
              onChange(setNestedValue(value, path, e.target.value));
            }
          }}
        />
      );
    } else {
      control = (
        <input
          type="text"
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
          value={fieldValue || ''}
          onChange={(e) => onChange(setNestedValue(value, path, e.target.value))}
        />
      );
    }

    // 模型族 weights 提示（dit_path / text_encoder_path / vae_path 等）
    const weightMeta =
      family && parentPath[0] === 'model'
        ? (family.weights || []).find((w) => w.field === key)
        : undefined;

    const wide = ['sources', 'rules', 'prompts', 'resolutions', 'args', 'group_lr'].includes(key) || ui.control === 'path' || key.endsWith('_path') || key === 'output_dir';
    if (React.isValidElement(control) && (typeof control.type === 'string' || control.type === StudioSelect)) {
      control = React.cloneElement(control as React.ReactElement<any>, {id: fieldId, 'aria-label': (control.props as any)['aria-label'] || fieldLabel, 'aria-invalid': !!errorItem});
    }
    const help = [prop.description, weightMeta?.hint].filter(Boolean).join('\n');
    const label = (
      <div key={fullPathKey} id={`field-${fullPathKey}`} data-testid={`field-${fullPathKey}`} className={compact ? `config-field ${prop.type === 'boolean' ? 'config-field-toggle' : ''} ${wide ? 'config-field-wide' : ''} ${errorItem ? 'config-field-invalid' : ''}` : `flex flex-col space-y-1 p-2 rounded ${errorItem ? 'bg-red-50 dark:bg-red-950/30 border border-red-300 dark:border-red-800' : ''}`}>
        <div className="flex justify-between items-center">
          <label htmlFor={fieldId} className="text-sm font-medium text-slate-700 dark:text-slate-300">
            {compact ? fieldLabel : weightMeta?.label || fieldLabel}
            {ui.unit && !percentage && ui.control !== 'slider' && <span className="ml-1 text-xs text-slate-500">({ui.unit})</span>}
          </label>
          {compact && help && <details className="config-help"><summary aria-label={`${fieldLabel} ${english ? 'help' : '说明'}`}><HelpCircle size={13} /></summary><p>{help}</p></details>}
        </div>
        {!compact && prop.description && <p className="text-xs text-slate-500 dark:text-slate-400">{prop.description}</p>}
        {!compact && weightMeta?.hint && (
          <p className="text-[11px] text-slate-400 dark:text-slate-500" data-testid={`weight-hint-${key}`}>
            {weightMeta.hint}
          </p>
        )}
        <div className="mt-1">{control}</div>
        {errorItem && <p className="text-xs text-red-600 dark:text-red-400">{errorItem.msg}</p>}
      </div>
    );

    if (!groups[groupName]) {
      groups[groupName] = { order: ui.order || 0, fields: [] };
    }
    groups[groupName].fields.push(label);
  };

  if (schema?.properties) {
    Object.entries(schema.properties).forEach(([key, prop]) => {
      const p = prop as SchemaProperty;
      if (p.$ref) {
        const resolved = resolveRef(schema, p.$ref);
        if (resolved && resolved.type === 'object' && resolved.properties) {
          Object.entries(resolved.properties).forEach(([subKey, subProp]) => {
            renderField(subKey, subProp as SchemaProperty, [key]);
          });
        }
      } else {
        renderField(key, p, []);
      }
    });
  }

  const groupOrder = groupFilter || schema?.['x-ui-groups'] || [];
  const sortedGroups = Object.entries(groups).sort(([keyA, groupA], [keyB, groupB]) => {
    const indexA = groupOrder.indexOf(keyA);
    const indexB = groupOrder.indexOf(keyB);
    if (indexA !== -1 && indexB !== -1) return indexA - indexB;
    if (indexA !== -1) return -1;
    if (indexB !== -1) return 1;
    return groupA.order - groupB.order;
  });

  return (
    <div className={compact ? 'compact-schema' : 'space-y-6'} data-testid="schema-form">
      {sortedGroups.map(([groupName, groupData]) => (
        <FieldGroup key={groupName} title={t(`groups.${groupName}`, groupName)} count={groupData.fields.length} compact={compact} groupKey={groupName}>
          {groupData.fields}
        </FieldGroup>
      ))}
      {sortedGroups.length === 0 && <p className="p-6 text-sm text-slate-500">{english ? 'No matching parameters.' : '没有匹配的参数。'}</p>}
    </div>
  );
};
