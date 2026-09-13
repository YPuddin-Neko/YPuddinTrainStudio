import React from 'react';
import { evaluateShowWhen } from '../showWhen';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronRight, Plus, Trash2, ArrowUp, ArrowDown, FolderOpen } from 'lucide-react';
import { PathInput, PathPickerModal } from '../../components/PathBrowser';
import { apiClient } from '../../api/client';
import { FamilyInfo } from '../../api/types';
import { configFieldLabel, configOptionLabel } from '../../utils/configPresentation';
import { MODEL_PATH_FIELDS } from '../../utils/workspaceConfig';
import { familyParameterOptions, modelFamilyWeights, trainingFamilyOptions } from '../../utils/trainingFamilies';
import NumericControl from './NumericControl';
import StudioSelect from '../../components/StudioSelect';
import ConfigHelp from '../../components/ConfigHelp';
import CaptionFormatSelect from '../../components/CaptionFormatSelect';
import './config-fields.css';

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
    options?: string[];
    allow_custom?: boolean;
  };
}

export interface ValidationError {
  loc?: string;
  msg: string;
}

export interface SourceRoleInfo {
  path: string;
  section: string;
  is_reg: boolean;
  managed: boolean;
  root: string | null;
  origin: string;
  images?: number | null;
}

export interface OutputBindingInfo {
  directory_template: string;
  name: string;
  automatic_name: boolean;
  inherits_output_dir: boolean;
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
  readOnly?: boolean;
  groupFilter?: string[];
  search?: string;
  onClearSearch?: () => void;
  sourceRoles?: SourceRoleInfo[];
  outputBinding?: OutputBindingInfo | null;
  versionSources?: boolean;
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
  sourceRoles?: SourceRoleInfo[];
  section: string;
  versionSources?: boolean;
}> = ({ value = [], onChange, sourceRoles = [], section, versionSources = false }) => {
  const { t, i18n } = useTranslation();
  const text = (zh: string, en: string) => i18n.language.startsWith("en") ? en : zh;
  const [modalIndex, setModalIndex] = React.useState<number | null>(null);

  const addSource = () => {
    onChange([
      ...value,
      { path: '', repeats: 1, caption_ext: 'auto', is_reg: false, prior_weight: 1.0, class_prompt: '' },
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
      {value.map((src, idx) => {
        const role = sourceRoles.find(item => item.path === src.path && item.section === section);
        const isReg = role?.managed ? role.is_reg : src.is_reg === undefined ? !!role?.is_reg : !!src.is_reg;
        const folder = String(src.path || '').replace(/\\/g,'/').split('/').filter(Boolean).pop() || text('未选择文件夹','No folder selected');
        return (
        <div key={idx} role="group" aria-label={t('train.sourceN', { n: idx + 1 })} className="p-3 bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2 text-xs">
          <div className="flex justify-between items-center">
            <span className="font-semibold text-slate-700 dark:text-slate-300">
              {folder}
            </span>
            <span>{isReg ? text('正则集','Regularization') : text('训练集','Training')} · {role?.images == null ? text('图片数待索引','Count pending indexing') : text(`${role.images} 张图片`,`${role.images} images`)}</span>
          </div>
          <p>{role?.managed ? text(`当前版本 / ${role.is_reg ? 'reg' : 'traindata'}`,`Current version / ${role.is_reg ? 'reg' : 'traindata'}`) : versionSources && !role ? text('目录用途待核对','Directory ownership pending') : text('外部 / 旧版来源','External / legacy source')} · {text(`每图重复 ${src.repeats ?? 1} 次`,`Repeats ${src.repeats ?? 1}`)}{isReg ? text(` · 正则权重 ${src.prior_weight ?? 1}`,` · Prior weight ${src.prior_weight ?? 1}`) : ''}</p>
          <details><summary>{text('高级来源设置','Advanced source settings')}</summary>
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
          <div className="source-settings-grid">
            <label><span>{t('dataset.repeats')}<ConfigHelp label={text('重复次数说明','Repeats help')}>{text('每轮让这组图片出现几次；默认 1。例如 20 张图重复 5 次计为 100 个样本，不复制文件。增加次数会增加训练占比和总步数，也可能过拟合。图片本身不能决定这个数值。','How often these images appear per epoch; default 1. Twenty images repeated five times count as 100 samples without copying files. More repeats increase their training share and step count, and may overfit. The image itself cannot determine this setting.')}</ConfigHelp></span><input aria-label={text(`重复次数 ${idx+1}`,`Repeats ${idx+1}`)} type="number" min="1" step="1" value={src.repeats ?? 1} onChange={event=>updateSource(idx,'repeats',Number(event.target.value))}/></label>
            <div className="source-reg-toggle"><span>{isReg ? text('正则集','Regularization') : text('训练集','Training')}</span><ConfigHelp label={text('数据用途说明','Dataset purpose help')}>{text('当前版本 traindata 中的图片自动作为训练集，reg 中的图片自动作为正则集。正则图用于类别先验保持，默认不继承训练触发词。外部与旧版来源保留已有用途。','Images inside this version’s traindata are training examples; images inside reg are regularization examples. Class priors do not inherit the training trigger by default. External and legacy sources retain their existing purpose.')}</ConfigHelp></div>
            {isReg && <label><span>{text('正则损失权重','Regularization loss weight')}<ConfigHelp label={text('正则损失权重说明','Regularization loss weight help')}>{text('只对正则图片的损失生效。默认 1；0.5 表示这些图片的损失乘以一半，0 则不贡献训练梯度。它不是生成正则图片的数量。','Applies only to regularization-image losses. Default 1; 0.5 halves their contribution, while 0 contributes no training gradient. This is not the number of images to generate.')}</ConfigHelp></span><input aria-label={text(`正则损失权重 ${idx+1}`,`Regularization loss weight ${idx+1}`)} type="number" min="0" step="0.1" value={src.prior_weight ?? 1} onChange={event=>updateSource(idx,'prior_weight',Number(event.target.value))}/></label>}
          </div>
          {role?.managed ? <p>{text('目录归属：当前版本','Directory: current version')} / <strong>{role.is_reg ? 'reg' : 'traindata'}</strong><br/><code className="break-all">{role.root}</code></p> : versionSources && !role ? <p>{text('正在核对目录归属；保留当前用途。','Checking directory ownership; retaining the current purpose.')}</p> : <details><summary>{text('外部 / 旧版来源兼容设置','External / legacy source compatibility')}</summary><p>{text('此路径不属于当前版本的 traindata 或 reg，文件保持原位置。仅为已有外部训练配置显式设置用途。','This path is outside this version’s traindata and reg. Files stay in place; adjust purpose only for existing external training configurations.')}</p><label><input type="checkbox" checked={isReg} onChange={event=>updateSource(idx,'is_reg',event.target.checked)}/>{text('外部来源用于正则训练','Use external source for regularization')}</label></details>}
          <details className="source-fallback"><summary>{text('缺少标签时的默认描述（可选）','Fallback description when captions are missing (optional)')}</summary><input aria-label={text(`默认描述 ${idx+1}`,`Fallback description ${idx+1}`)} value={src.class_prompt ?? ''} onChange={event=>updateSource(idx,'class_prompt',event.target.value || null)} placeholder={text('例如：a person；不生成或修改标签文件','For example: a person; does not create or edit caption files')}/></details>
          <button type="button" onClick={() => removeSource(idx)} className="text-red-500 hover:text-red-700">{text('从本次配置移除来源（保留文件）','Remove from this configuration (keep files)')}</button>
          </details>
        </div>
      );})}
      <details><summary>{text('高级：引用已有数据目录','Advanced: reference an existing dataset folder')}</summary><p>{text('管理、上传或导入图片请使用项目的训练数据步骤。此处仅为兼容已有外部配置。','Manage, upload and import images in the project’s training-data step. This option supports existing external configurations.')}</p><button
        type="button"
        onClick={addSource}
        data-testid="add-source"
        className="flex items-center space-x-1 px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded border border-slate-300 dark:border-slate-600"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>{t('train.addSource', '添加数据集源')}</span>
      </button>
      </details>

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
            <button type="button" aria-label={`${t('common.delete')} ${t('train.promptN', { n: idx + 1 })}`} onClick={() => removePrompt(idx)} className="text-red-500 hover:text-red-700">
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
  const [models, setModels] = React.useState<Array<{ id: string; path: string; kind: string; family: string; exists?: boolean }>>([]);

  React.useEffect(() => {
    if (!kind) return;
    let active = true;
    const refresh = () => void apiClient
      .get<Array<{ id: string; path: string; kind: string; family: string; exists?: boolean }>>('/models', { silent: true })
      .then((list) => { if (active) setModels(Array.isArray(list) ? list : []); })
      .catch(() => { if (active) setModels([]); });
    refresh(); window.addEventListener('studio-models-changed', refresh); window.addEventListener('focus', refresh);
    return () => { active = false; window.removeEventListener('studio-models-changed', refresh); window.removeEventListener('focus', refresh); };
  }, [kind]);

  const matched = kind ? models.filter((m) => m.exists !== false && m.kind === kind && (!familyName || m.family === familyName)) : [];

  return (
    <div className={`model-path-control ${matched.length > 0 ? 'has-registry' : ''}`}>
      <PathInput ariaLabel={label} value={value} onChange={onChange} />
      {matched.length > 0 && (
        <StudioSelect aria-label={`${label || kind} · ${t('models.fromRegistry')}`} value="" onValueChange={onChange} data-testid="model-registry-select"
          options={[{value:'',label:t('models.fromRegistry'),disabled:true},...matched.map(model=>({value:model.path,label:model.path.split(/[\\/]/).pop() || model.path}))]}/>

      )}
    </div>
  );
};

function ResolutionInput({value, onChange, label}: {value: number[] | string; onChange: (next: number[] | string) => void; label: string}) {
  const encoded = Array.isArray(value) ? value.join(', ') : String(value || '');
  const [draft, setDraft] = React.useState(encoded);
  React.useEffect(() => setDraft(encoded), [encoded]);
  const update = (raw: string) => {
    setDraft(raw);
    const tokens = raw.replace(/[\u005b\u005d]/g, '').split(/[,，\s]+/).filter(Boolean);
    onChange(tokens.length && tokens.every(token => /^\d+$/.test(token)) ? tokens.map(Number) : raw);
  };
  return <div className="resolution-editor"><input aria-label={label} inputMode="numeric" value={draft} onChange={event => update(event.target.value)} placeholder="1024"/></div>;
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
  const initialValue = () => {
    if (prop.default != null) return prop.default;
    if (prop.type === 'object') return {};
    if (prop.type === 'boolean') return false;
    if (prop.type === 'string') return '';
    if (prop.type === 'array') return [];
    const familyDefault = placeholder?.trim() ? Number(placeholder) : Number.NaN;
    if (Number.isFinite(familyDefault)) return familyDefault;
    const minimum = prop.minimum ?? prop['x-ui']?.min ?? 0;
    return prop.exclusiveMinimum != null && minimum <= prop.exclusiveMinimum
      ? prop.exclusiveMinimum + (prop['x-ui']?.step || 1)
      : minimum;
  };
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
    <section data-group={groupKey} className={`${groupKey === 'adapter' ? 'config-adapter-group ' : ''}${compact ? `config-group ${['model', 'dataset', 'caption', 'sampling', 'validation'].includes(groupKey || '') ? 'config-group-wide' : ''}` : 'border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800'}`}>
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
      {isOpen && <div className={groupKey === 'adapter' ? 'config-fields config-adapter-fields' : compact ? 'config-fields' : 'p-4 space-y-4'}>{children}</div>}
    </section>
  );
};

export const SchemaForm: React.FC<SchemaFormProps> = ({
  schema,
  value,
  onChange: onValueChange,
  showAdvanced = false,
  errors = [],
  family,
  families,
  compact = false,
  readOnly = false,
  groupFilter,
  search = '',
  onClearSearch,
  sourceRoles,
  outputBinding,
  versionSources = false,
}) => {
  const { t, i18n } = useTranslation();
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  const [editCaptionOverrides, setEditCaptionOverrides] = React.useState(false);
  const [editOutput, setEditOutput] = React.useState(false);
  const lastLowRank = React.useRef<number | null>(null);
  const lastEmittedConfig = React.useRef<Record<string, any> | null>(null);
  React.useEffect(() => {
    // Only retain a rank across changes emitted by this editor. Selecting another
    // preset or loading an external draft must not inherit this temporary value.
    if (value !== lastEmittedConfig.current) lastLowRank.current = null;
    lastEmittedConfig.current = null;
  }, [value]);
  const captionOverrideKeys = ['prefix', 'suffix', 'trigger_word'];
  const hasCaptionOverrides = captionOverrideKeys.some(key => !!value.dataset?.caption?.[key]);
  const onChange = (next: Record<string, any>) => {
    if (readOnly) return;
    lastEmittedConfig.current = next;
    onValueChange(next);
  };
  const lokrModeLabel = english ? 'LoKr parameter mode' : 'LoKr 参数形式';
  const groups: Record<string, { order: number; fields: React.ReactNode[] }> = {};
  const conditionValue = { ...value, dataset: { resolution_mode: 'bucket', ...value.dataset } };
  const weights = modelFamilyWeights(family);

  const renderField = (key: string, prop: SchemaProperty, parentPath: string[] = []) => {
    const path = [...parentPath, key];
    const fullPathKey = path.join('.');
    const lokrRank = fullPathKey === 'adapter.rank' && value.adapter?.algo === 'lokr';
    const weightMeta = parentPath[0] === 'model' ? weights.find(weight => weight.field === key) : undefined;
    const supportedOptions = familyParameterOptions(family, fullPathKey);
    if (supportedOptions?.length === 0) return null;
    if (family?.objective === 'ddpm' && ['sampling.shift', 'sampling.er_sde_order', 'sampling.er_sde_s_noise', 'objective.shift', 'objective.res_shift_tokens', 'objective.res_shift_mu', 'objective.mode_scale', 'objective.snr_gamma'].includes(fullPathKey)) return null;
    if (parentPath[0] === 'model' && key in MODEL_PATH_FIELDS && weights.length && !weightMeta) return null;
    if (family?.name === 'sdxl' && weightMeta?.required === false && !showAdvanced) return null;
    if (fullPathKey === 'model.zero_terminal_snr' && value.model?.prediction_type !== 'v_prediction' && !value.model?.zero_terminal_snr) return null;
    // Keep legacy cloud-log data in the draft, but do not expose controls that enable it.
    if (fullPathKey === 'logging.wandb' || fullPathKey.startsWith('logging.wandb.')) return null;
    // The service assigns a separate samples/<job_id> destination when starting a task.
    if (fullPathKey === 'sampling.output_dir') return null;
    if (versionSources && ['checkpoint.output_dir', 'checkpoint.name'].includes(fullPathKey) && !showAdvanced && !editOutput) return null;
    if (fullPathKey === 'adapter.alpha' && value.adapter?.algo === 'lokr' && value.adapter?.rank === 'full') return null;
    const ui = { ...(prop['x-ui'] || {}), ...(compact && fullPathKey === 'dataset.batch_size' ? {group:'loop'} : {}) };
    if (conditionValue.dataset.resolution_mode === 'native' && ['dataset.resolutions', 'dataset.aspect_ratio_limit', 'dataset.area_tolerance', 'dataset.bucket_step', 'dataset.bucket_no_upscale'].includes(fullPathKey)) return null;
    if (compact && !showAdvanced && fullPathKey === 'adapter.rules' && !value.adapter?.rules?.length) return null;

    const nested = prop.$ref ? resolveRef(schema, prop.$ref) : prop;
    if (nested?.type === 'object' && nested.properties) {
      Object.entries(nested.properties).forEach(([childKey, child]) => renderField(childKey, child as SchemaProperty, path));
      return null;
    }
    const fieldLabel = weightMeta?.label || configFieldLabel(fullPathKey, t(`fields.${key}`, prop.title || key), english);
    const fieldId = `config-${fullPathKey}`;
    const currentGroup = ui.group || parentPath[0] || 'default';
    if (groupFilter && !groupFilter.includes(currentGroup)) return null;
    if (search.trim() && !`${fieldLabel} ${fullPathKey} ${prop.description || ''} ${lokrRank ? lokrModeLabel : ''}`.toLowerCase().includes(search.trim().toLowerCase())) return null;

    const captionOverride = fullPathKey.startsWith('dataset.caption.') && captionOverrideKeys.includes(key);
    if ((ui.advanced || captionOverride) && !showAdvanced && !(captionOverride && editCaptionOverrides) && !(editOutput && ['checkpoint.output_dir', 'checkpoint.name'].includes(fullPathKey))) return null;
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
    const compactField = compact || groupName === 'adapter';
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
          sourceRoles={sourceRoles}
          section={parentPath[0] || 'dataset'}
          versionSources={versionSources}
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
      const modelKind = parentPath[0] === 'model' ? weightMeta?.kind || MODEL_PATH_FIELDS[key as keyof typeof MODEL_PATH_FIELDS] || null : null;
      control = (
        <ModelPathInput
          value={fieldValue || ''}
          kind={modelKind}
          label={fieldLabel}
          familyName={value.model?.family}
          onChange={(val) => onChange(setNestedValue(value, path, val === '' && prop.anyOf?.some((p) => p.type === 'null') ? null : val))}
        />
      );
    } else if (fullPathKey === 'adapter.rank') {
      control = <input type="number" aria-label={fieldLabel} min="1" step="1" value={fieldValue === 'full' ? '' : fieldValue ?? 16}
        onChange={event => {
          const rank = event.target.value === '' ? undefined : Number(event.target.value);
          if (typeof rank === 'number' && Number.isInteger(rank) && rank > 0) lastLowRank.current = rank;
          onChange(setNestedValue(value, path, rank));
        }}/>;
    } else if (prop.anyOf) {
      control = <SchemaValueInput schema={schema} property={prop} value={fieldValue} name={fullPathKey} compact={compactField}
        placeholder={family && fullPathKey.startsWith('sampling.') && ['steps', 'cfg', 'shift'].includes(key) ? (family.sampling?.[key as 'steps' | 'cfg' | 'shift'] != null ? String(family.sampling[key as 'steps' | 'cfg' | 'shift']) : key === 'shift' ? t('sampling.shiftAuto') : undefined) : undefined}
        onChange={(val) => onChange(setNestedValue(value, path, val))} />;
    } else if (fullPathKey === 'model.family' && families) {
      control = <StudioSelect aria-label="model.family" value={fieldValue || families[0]?.name || ''} disabled={!families.length}
        onValueChange={next => onChange(setNestedValue(value,path,next))} options={trainingFamilyOptions(families, english, fieldValue)}/>;
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
    } else if (supportedOptions) {
      control = <StudioSelect aria-label={fieldLabel} value={fieldValue == null ? '' : String(fieldValue)}
        onValueChange={next => onChange(setNestedValue(value, path, next))}
        options={supportedOptions.map(option => ({ value: option, label: configOptionLabel(fullPathKey, option, english) }))}/>;
    } else if (ui.options?.length) {
      control = <div className="space-y-2"><StudioSelect aria-label={fieldLabel} value={ui.options.includes(fieldValue) ? fieldValue : '__custom__'}
        onValueChange={next => onChange(setNestedValue(value,path,next === '__custom__' ? '' : next))}
        options={[...ui.options.map(option=>({value:option,label:configOptionLabel(fullPathKey,option,english)})),...(ui.allow_custom ? [{value:'__custom__',label:english?'Custom Python class…':'自定义 Python 类…'}] : [])]}/>
        {ui.allow_custom && !ui.options.includes(fieldValue) && <input aria-label={`${fieldLabel} ${english?'custom class':'自定义类'}`} value={fieldValue || ''} placeholder="package.module.OptimizerClass" onChange={event=>onChange(setNestedValue(value,path,event.target.value))}/>}
      </div>;
    } else if (prop.enum) {
      control = (
        <StudioSelect aria-label={fieldLabel} value={fieldValue == null ? '' : String(fieldValue)}
          onValueChange={next => {
            const updated = setNestedValue(value, path, prop.enum!.find(option => String(option) === next));
            // A full LoKr factor matrix is not a valid numeric rank for LoRA/LoHa.
            onChange(fullPathKey === 'model.prediction_type' && next !== 'v_prediction'
              ? setNestedValue(updated, ['model', 'zero_terminal_snr'], false)
              : fullPathKey === 'adapter.algo' && next !== 'lokr' && value.adapter?.rank === 'full'
              ? setNestedValue(updated, ['adapter', 'rank'], lastLowRank.current ?? 16)
              : updated);
          }}
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

    const wide = ['sources', 'rules', 'prompts', 'resolutions', 'args', 'group_lr'].includes(key) || ui.control === 'path' || key.endsWith('_path') || key === 'output_dir' || fullPathKey === 'adapter.lr_scale';
    if (React.isValidElement(control) && (typeof control.type === 'string' || control.type === StudioSelect)) {
      control = React.cloneElement(control as React.ReactElement<any>, {id: fieldId, 'aria-label': (control.props as any)['aria-label'] || fieldLabel, 'aria-invalid': !!errorItem});
    }
    const help = weightMeta?.hint || prop.description;
    const label = (
      <div key={fullPathKey} id={`field-${fullPathKey}`} data-testid={`field-${fullPathKey}`} data-field-path={fullPathKey} data-control-kind={prop.type === 'boolean' ? 'toggle' : undefined} className={compactField ? `config-field ${prop.type === 'boolean' ? 'config-field-toggle' : ''} ${wide ? 'config-field-wide' : ''} ${errorItem ? 'config-field-invalid' : ''}` : `flex flex-col space-y-1 p-2 rounded ${errorItem ? 'bg-red-50 dark:bg-red-950/30 border border-red-300 dark:border-red-800' : ''}`}>
        <div className="flex justify-between items-center">
          <label htmlFor={fieldId} className="text-sm font-medium text-slate-700 dark:text-slate-300">
            {fieldLabel}{weightMeta?.required === false && !['flux', 'flux2'].includes(family?.name || '') && <span className="ml-1 text-xs text-slate-500">{english ? '(optional)' : '（可选）'}</span>}
            {ui.unit && !percentage && ui.control !== 'slider' && <span className="ml-1 text-xs text-slate-500">({ui.unit})</span>}
          </label>
          {compactField && help && <ConfigHelp label={`${fieldLabel} ${english ? 'help' : '说明'}`}>{help}</ConfigHelp>}
        </div>
        {!compactField && prop.description && !weightMeta?.hint && <p className="text-xs text-slate-500 dark:text-slate-400">{prop.description}</p>}
        {!compactField && weightMeta?.hint && (
          <p className="text-[11px] text-slate-400 dark:text-slate-500" data-testid={`weight-hint-${key}`}>
            {weightMeta.hint}
          </p>
        )}
        <div className="mt-1">{readOnly ? <fieldset disabled style={{ border: 0, margin: 0, padding: 0, minWidth: 0 }}>{control}</fieldset> : control}</div>
        {errorItem && <p className="text-xs text-red-600 dark:text-red-400">{errorItem.msg}</p>}
      </div>
    );

    if (!groups[groupName]) {
      groups[groupName] = { order: ui.order || 0, fields: [] };
    }
    if (lokrRank) {
      const modeId = 'config-adapter-parameter-mode';
      groups[groupName].fields.push(<div key="adapter.parameter_mode" id={fieldValue === 'full' ? 'field-adapter.rank' : 'field-adapter.parameter_mode'} data-testid="field-adapter.parameter_mode" data-field-path="adapter.parameter_mode" className={`config-field ${fieldValue === 'full' && errorItem ? 'config-field-invalid' : ''}`}>
        <div className="flex justify-between items-center">
          <label htmlFor={modeId} className="text-sm font-medium text-slate-700 dark:text-slate-300">{lokrModeLabel}</label>
          <ConfigHelp label={`${lokrModeLabel} ${english ? 'help' : '说明'}`}>{english ? 'Full retains the complete LoKr factor matrices; it does not fine-tune the whole model and does not use Alpha. Low rank decomposes the factors using Rank and Alpha.' : 'Full 保留 LoKr 完整因子矩阵，不是全量微调，也不使用 Alpha。低秩模式通过 Rank 和 Alpha 设置因子分解与缩放。'}</ConfigHelp>
        </div>
        <div className="mt-1"><StudioSelect id={modeId} aria-label={lokrModeLabel} disabled={readOnly} value={fieldValue === 'full' ? 'full' : 'low_rank'}
          aria-invalid={fieldValue === 'full' && !!errorItem}
          onValueChange={next => {
            if (typeof fieldValue === 'number' && Number.isInteger(fieldValue) && fieldValue > 0) lastLowRank.current = fieldValue;
            const defaultRank = typeof prop.default === 'number' && Number.isInteger(prop.default) && prop.default > 0 ? prop.default : 16;
            onChange(setNestedValue(value, path, next === 'full' ? 'full' : lastLowRank.current ?? defaultRank));
          }} options={[
            {value: 'full', label: english ? 'Full · full factor matrices' : 'Full · 完整因子矩阵'},
            {value: 'low_rank', label: english ? 'Low rank · factor decomposition' : '低秩 · 分解因子矩阵'},
          ]}/></div>
        {fieldValue === 'full' && errorItem && <p>{errorItem.msg}</p>}
      </div>);
    }
    if (!lokrRank || fieldValue !== 'full') groups[groupName].fields.push(label);
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

  const captionSources = Array.isArray(value.dataset?.sources) ? value.dataset.sources : [];
  const showCaptionFormats = captionSources.length > 0 && (!groupFilter || groupFilter.includes('caption')) && (!search || /标签|格式|caption|format|json|txt/i.test(search));
  if (showCaptionFormats && !groups.caption) groups.caption = {order: 0, fields: []};
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
      {search.trim() && sortedGroups.length > 0 && <p className="config-search-results" role="status">{english ? `${sortedGroups.reduce((count, [, group]) => count + group.fields.length, 0)} matching parameters · ${sortedGroups.length} sections` : `${sortedGroups.reduce((count, [, group]) => count + group.fields.length, 0)} 个匹配参数 · ${sortedGroups.length} 个分组`}</p>}
      {sortedGroups.map(([groupName, groupData]) => (
        <FieldGroup key={`${groupName}:${search.trim()}`} title={t(`groups.${groupName}`, groupName)} count={groupData.fields.length} compact={compact} groupKey={groupName}>
          {groupName === 'checkpoint' && versionSources && <div className="output-binding-summary">
            <div className="output-binding-heading"><strong>{english ? 'Training weights' : '训练权重'}</strong><button type="button" onClick={() => setEditOutput(previous => !previous)}>{editOutput ? (english ? 'Collapse custom settings' : '收起自定义设置') : (english ? 'Customize save location or name' : '自定义保存位置或名称')}</button></div>
            {outputBinding ? <><div><span>{english ? 'File name' : '文件名'}</span><code>{outputBinding.name}-final.safetensors</code></div><div><span>{english ? 'Save location' : '保存位置'}</span><code>{outputBinding.directory_template.replace('{job_id}', english ? '<run ID>' : '<运行 ID>')}</code></div></> : <p>{english ? 'Resolving the save location…' : '正在读取保存位置…'}</p>}
          </div>}
          {groupName === 'caption' && showCaptionFormats && <div className="caption-source-formats">
            {captionSources.map((source: any, index: number) => {
              const name = String(source.path || '').split(/[\\/]/).filter(Boolean).pop() || `${english ? 'Data source' : '数据源'} ${index + 1}`;
              const label = `${english ? 'Caption format' : '标签格式'} · ${name}`;
              return <div className="caption-source-format" key={`${index}-${source.path}`}>
                <span className="caption-source-name" title={source.path}>{name}</span>
                <label><span>{english ? 'Caption format' : '标签格式'}</span><CaptionFormatSelect label={label} value={source.caption_ext || 'auto'} disabled={readOnly} onChange={caption_ext => onChange(setNestedValue(value, ['dataset', 'sources'], captionSources.map((item: any, itemIndex: number) => itemIndex === index ? {...item, caption_ext} : item)))}/></label>
              </div>;
            })}
          </div>}
          {groupName === 'caption' && hasCaptionOverrides && !showAdvanced && !editCaptionOverrides && <div className="caption-override-notice" role="status"><span>{english ? 'This configuration adds text to your existing captions.' : '当前配置会额外改写已有标签。'}</span><button type="button" onClick={() => setEditCaptionOverrides(true)}>{english ? 'Edit extra caption changes' : '编辑额外标签改写'}</button></div>}
          {compact && groupName === 'model' ? (() => {
            const order = ['model.family', 'model.dit_path', 'model.text_encoder_path', 'model.text_encoder_2_path', 'model.vae_path', 'model.tokenizer_path', 'model.dtype', 'model.attention', 'model.prediction_type', 'model.zero_terminal_snr'];
            return [...groupData.fields].sort((a, b) => {
              const rank = (node: React.ReactNode) => { const index = order.indexOf(String((node as React.ReactElement).key)); return index < 0 ? order.length : index; };
              return rank(a) - rank(b);
            });
          })() : groupName === 'adapter' ? (() => {
            const order = ['algo', 'preset', 'parameter_mode', 'factor', 'rank', 'alpha', 'init', 'mode', 'param_dtype', 'dropout', 'rank_dropout', 'module_dropout', 'lr_scale', 'rules', 'resume_weights'];
            const isToggle = (node: React.ReactNode) => (node as React.ReactElement).props['data-control-kind'] === 'toggle';
            const fields = groupData.fields.filter(node => !isToggle(node)).sort((a, b) => {
              const rank = (node: React.ReactNode) => { const index = order.indexOf(String((node as React.ReactElement).key).split('.').pop() || ''); return index < 0 ? order.length : index; };
              return rank(a) - rank(b);
            });
            const switches = groupData.fields.filter(isToggle);
            const fieldName = (node: React.ReactNode) => String((node as React.ReactElement).key).split('.').pop() || '';
            const structure = fields.filter(node => ['algo', 'preset'].includes(fieldName(node)));
            const capacity = fields.filter(node => ['parameter_mode', 'factor', 'rank', 'alpha'].includes(fieldName(node)));
            const tuning = fields.filter(node => !['algo', 'preset', 'parameter_mode', 'factor', 'rank', 'alpha'].includes(fieldName(node)));
            if (!compact) return <>{structure}{capacity}{switches.length > 0 && <div className="config-adapter-switches">{switches}</div>}{tuning}</>;
            return <>{structure.length > 0 && <div className="config-field-section config-adapter-structure"><h3>{english ? 'Training structure' : '训练结构'}</h3>{structure}</div>}{capacity.length > 0 && <div className="config-field-section config-adapter-capacity"><h3>{english ? 'Parameter size' : '参数规模'}</h3>{capacity}</div>}{(tuning.length > 0 || switches.length > 0) && <div className="config-field-section config-adapter-tuning"><h3>{english ? 'Initialization and regularization' : '初始化与正则'}</h3>{switches.length > 0 && <div className="config-adapter-switches">{switches}</div>}{tuning}</div>}</>;
          })() : compact && groupName === 'sampling' ? [...groupData.fields].sort((a, b) => {
            const order = ['enabled', 'at_start', 'every_steps', 'every_epochs', 'prompts', 'width', 'height', 'steps', 'cfg', 'shift', 'seed', 'sampler', 'scheduler', 'er_sde_order', 'er_sde_s_noise', 'prompts_file'];
            const rank = (node: React.ReactNode) => { const index = order.indexOf(String((node as React.ReactElement).key).split('.').pop() || ''); return index < 0 ? order.length : index; };
            return rank(a) - rank(b);
          }) : groupData.fields}
        </FieldGroup>
      ))}
      {sortedGroups.length === 0 && <div className="config-search-empty" role="status"><strong>{english ? 'No matching parameters.' : '没有匹配的参数。'}</strong><p>{search.trim() ? (english ? 'Try a parameter name, keyword or configuration path.' : '试试参数名称、关键词或配置字段路径。') : (english ? 'This section has no available parameters for the current configuration.' : '当前配置在此分区没有可用参数。')}</p>{search.trim() && onClearSearch && <button type="button" className="studio-secondary" onClick={onClearSearch}>{english ? 'Return to parameter sections' : '返回参数分区'}</button>}</div>}
    </div>
  );
};
