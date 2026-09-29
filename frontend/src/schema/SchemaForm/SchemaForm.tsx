import { adapterLayerTypesLock, selectTrainingComponents, trainingManagedReason } from '../../utils/trainingSelection';
import { confirmedTrainingComputePolicy, trainingComputeManagedField, trainingComputePolicyHint } from '../../utils/trainingComputePolicy';
import { contextHelp, contextOptions, familyHasConvolutions, hideUnusedSetting, presetHasConvolutions, unusedSettingReason, type FieldContext } from '../../utils/fieldContext';
import React from 'react';
import { evaluateShowWhen } from '../showWhen';
import { useTranslation } from 'react-i18next';
import { AlertCircle, ChevronDown, ChevronRight, Plus, Trash2, ArrowUp, ArrowDown, FolderOpen } from 'lucide-react';
import { PathInput, PathPickerModal } from '../../components/PathBrowser';
import { apiClient } from '../../api/client';
import { FamilyInfo, ModelAsset } from '../../api/types';
import { configFieldHelp, configFieldHint, configFieldLabel, configOptionLabel, configPresetLabel, PERCENTAGE_FIELDS } from '../../utils/configPresentation';
import { modelPathHint } from '../../utils/fieldCopy';
import { parameterGroupLabel } from '../../utils/parameterWorkflow';
import { MODEL_PATH_FIELDS } from '../../utils/workspaceConfig';
import { familyParameterOptions, modelAssetUnsupportedReason, modelFamilyWeights, trainingFamilyOptions } from '../../utils/trainingFamilies';
import { managedValueLabel, normalizeOptimizerConfig, optimizerManagedReason, restoreOptimizerSelection, selectOptimizer } from '../../utils/optimizerCapabilities';
import NumericControl from './NumericControl';
import DecimalNumberInput from './DecimalNumberInput';
import { scientificText } from '../../utils/numberText';
import StudioSelect, { type StudioSelectOption } from '../../components/StudioSelect';
import Switch from '../../components/Switch';
import ConfigHelp from '../../components/ConfigHelp';
import CaptionFormatSelect from '../../components/CaptionFormatSelect';
import './config-fields.css';
import { optionalValueLabel } from './optionalValues';
import ParameterFields from './ParameterFields';
import RecoveryInterval from './RecoveryInterval';
import ParameterToggleSection from './ParameterToggleSection';
import { LoadingNote } from '../../components/Loading';

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
    hidden?: boolean;
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

/** A value the trainer accepts but rounds; `fix` replaces it with one that is used as written. */
export interface FieldNotice {
  loc: string;
  msg: string;
  fix?: { label: string; value: unknown };
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
  notices?: FieldNotice[];
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
  /** Effective policy returned for this exact draft by the server plan. */
  computePolicy?: unknown;
  /** Preset drafts: model files are optional and keep the training configuration's files when empty. */
  preset?: boolean;
  projectId?: string;
  versionId?: string;
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
  learningRateReason?: string;
  onChange: (val: any[]) => void;
}> = ({ value = [], learningRateReason, onChange }) => {
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
            aria-label={`LR ${idx + 1}`}
            placeholder="LR"
            step="0.0001"
            value={rule.lr ?? ''}
            disabled={!!learningRateReason}
            title={learningRateReason}
            onChange={(e) => updateRule(idx, 'lr', e.target.value === '' ? null : Number(e.target.value))}
            className="w-20 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <div className="flex items-center space-x-1">
            <button type="button" onClick={() => moveRule(idx, 'up')} disabled={idx === 0} className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon">
              <ArrowUp className="w-3.5 h-3.5" />
            </button>
            <button type="button" onClick={() => moveRule(idx, 'down')} disabled={idx === value.length - 1} className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon">
              <ArrowDown className="w-3.5 h-3.5" />
            </button>
            <button type="button" onClick={() => removeRule(idx)} className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon ui-btn-danger">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      ))}
      {!!value.length && learningRateReason && <p className="config-field-hint">{learningRateReason}</p>}
      <button
        type="button"
        onClick={addRule}
        data-testid="add-rule"
        className="ui-btn ui-btn-sm"
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
  addLabel?: string;
  onChange: (val: Record<string, any>) => void;
}> = ({ value = {}, addLabel, onChange }) => {
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
          <button type="button" aria-label={`${t('common.remove', '移除')} ${k}`} onClick={() => removeEntry(k)} className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon ui-btn-danger">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
      {keyError && <p role="alert" className="text-xs text-amber-600">{keyError}</p>}
      <button
        type="button"
        onClick={addEntry}
        data-testid="add-property"
        className="ui-btn ui-btn-sm"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>{addLabel || t('train.addProperty', '添加属性')}</span>
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
        // Version-owned purposes come from the server; never guess them from folder names.
        const pending = versionSources && !role;
        const isReg = role?.managed ? role.is_reg : src.is_reg === undefined ? !!role?.is_reg : !!src.is_reg;
        const folder = String(src.path || '').replace(/\\/g,'/').split('/').filter(Boolean).pop() || text('未选择文件夹','No folder selected');
        const n = idx + 1;
        return (
        <div key={idx} role="group" aria-label={t('train.sourceN', { n })} className="p-3 bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2 text-xs">
          <div className="flex flex-wrap justify-between items-center gap-2">
            <span className="min-w-0 flex-1 truncate font-semibold text-slate-700 dark:text-slate-300" title={src.path}>
              {folder}
            </span>
            <span className="flex items-center gap-2">
              <span>{pending ? text('用途待确认','Purpose pending') : isReg ? text('正则集','Regularization') : text('训练集','Training')} · {role?.images == null ? text('图片数待索引','Count pending indexing') : text(`${role.images} 张图片`,`${role.images} images`)}</span>{isReg && <ConfigHelp label={text('数据用途说明','Dataset purpose help')}>{text('正则图默认不继承训练触发词。','Regularization images do not inherit the training trigger by default.')}</ConfigHelp>}
              <button type="button" onClick={() => removeSource(idx)} className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon ui-btn-danger"
                aria-label={text(`从本次配置移除来源 ${folder}（保留文件）`,`Remove source ${folder} from this configuration (keep files)`)}
                title={text('从本次配置移除来源（保留文件）','Remove from this configuration (keep files)')}>
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </span>
          </div>
          <div className="source-path-control flex min-w-0 gap-2">
            <input
              type="text"
              aria-label={text(`图片目录 ${n}`, `Image folder ${n}`)}
              placeholder="/path/to/dataset"
              value={src.path || ''}
              onChange={(e) => updateSource(idx, 'path', e.target.value)}
              className="min-w-0 flex-1 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600 font-mono"
            />
            <button
              type="button"
              onClick={() => setModalIndex(idx)}
              className="ui-btn ui-btn-sm"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              <span>{t('common.browse')}</span>
            </button>
          </div>
          <div className="source-settings-grid">
            <label><span>{t('dataset.repeats')}<ConfigHelp label={text('重复次数说明','Repeats help')}>{text('每轮重复使用这组图片的次数。默认 1；20 张图重复 5 次计为 100 个样本。增加次数会增加训练占比和总步数，也可能过拟合。','Uses per image per epoch, default 1. Twenty images repeated five times count as 100 samples. More repeats increase their training share and total steps, with a risk of overfitting.')}</ConfigHelp></span><input aria-label={text(`重复次数 ${n}`,`Repeats ${n}`)} type="number" min="1" step="1" value={src.repeats ?? 1} onChange={event=>updateSource(idx,'repeats',event.target.value === '' ? '' : Number(event.target.value))}/></label>
            {!role?.managed && !pending && <div className="source-purpose"><span>{text('用途','Purpose')}</span><StudioSelect aria-label={text(`用途 ${n}`,`Purpose ${n}`)} value={isReg ? 'reg' : 'train'} onValueChange={next=>updateSource(idx,'is_reg',next === 'reg')} options={[{value:'train',label:text('训练集','Training')},{value:'reg',label:text('正则集','Regularization')}]}/></div>}
            {isReg && <label><span>{text('正则损失权重','Regularization loss weight')}<ConfigHelp label={text('正则损失权重说明','Regularization loss weight help')}>{text('正则图片的损失乘数。默认 1；0.5 减半，0 不贡献训练梯度。','Multiplier for regularization-image loss. Default 1; 0.5 halves it, while 0 contributes no training gradient.')}</ConfigHelp></span><input aria-label={text(`正则损失权重 ${n}`,`Regularization loss weight ${n}`)} type="number" min="0" step="0.1" value={src.prior_weight ?? 1} onChange={event=>updateSource(idx,'prior_weight',event.target.value === '' ? '' : Number(event.target.value))}/></label>}
            <label><span>{text('无标签时的描述','Text for uncaptioned images')}<ConfigHelp label={text('无标签时的描述说明','Uncaptioned text help')}>{text('图片没有标签文件时，用这段文字作为标签。留空则不补充；不会创建或修改标签文件。','Used as the caption for images without a caption file. Leave empty to add nothing; caption files are not created or changed.')}</ConfigHelp></span><input aria-label={text(`无标签时的描述 ${n}`,`Text for uncaptioned images ${n}`)} value={src.class_prompt ?? ''} onChange={event=>updateSource(idx,'class_prompt',event.target.value || null)} placeholder={text('例如：a person','For example: a person')}/></label>
          </div>
        </div>
      );})}
      <button
        type="button"
        onClick={addSource}
        data-testid="add-source"
        className="ui-btn ui-btn-sm"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>{t('train.addSource', '添加数据源')}</span>
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
            <button type="button" aria-label={`${t('common.delete')} ${t('train.promptN', { n: idx + 1 })}`} onClick={() => removePrompt(idx)} className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon ui-btn-danger">
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
        className="ui-btn ui-btn-sm"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>{t('train.addPrompt', '添加采样提示词')}</span>
      </button>
    </div>
  );
};

// 6. Betas 控件 (长度 2 的数字数组)
const BetasEditor: React.FC<{
  value: [number | '', number | ''];
  scheduleFree?: boolean;
  onChange: (val: [number | '', number | '']) => void;
}> = ({ value = [0.9, 0.999], scheduleFree, onChange }) => {
  const { i18n } = useTranslation();
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  return (
    <div className="config-beta-controls">
      <label><span aria-hidden="true">β1</span>
      <input
        type="number"
        min="0"
        max="0.999999"
        step="0.001"
        value={value[0] ?? 0.9}
        onChange={(e) => onChange([e.target.value === '' ? '' : Number(e.target.value), value[1]])}
        className="w-1/2 px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
        placeholder="beta1"
        aria-label={scheduleFree ? (english ? 'Weight averaging β1' : '权重平均 β1') : (english ? 'Direction smoothing β1' : '方向平滑 β1')}
      />
      </label>
      <label><span aria-hidden="true">β2</span>
      <input
        type="number"
        min="0"
        max="0.999999"
        step="0.0001"
        value={value[1] ?? 0.999}
        onChange={(e) => onChange([value[0], e.target.value === '' ? '' : Number(e.target.value)])}
        className="w-1/2 px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
        placeholder="beta2"
        aria-label={english ? 'Magnitude smoothing β2' : '幅度平滑 β2'}
      />
      </label>
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
  models: ModelAsset[];
  directoryOnly?:boolean;
  allowMissingDirectory?:boolean;
  resolveDefaultPath?:()=>Promise<string>;
}> = ({ value, kind, onChange, label, familyName, models, resolveDefaultPath, directoryOnly, allowMissingDirectory }) => {
  const { t } = useTranslation();
  const [defaultPath, setDefaultPath] = React.useState('');
  React.useEffect(() => {
    let active = true;
    setDefaultPath('');
    if (kind) void apiClient.get<{path:string}>('/models/browse-root', { params: {kind}, silent: true })
      .then(result => { if (active) setDefaultPath(result.path); }).catch(() => {});
    return () => { active = false; };
  }, [kind]);
  const matched = kind ? models.filter((m) => m.exists !== false && (m as typeof m & {purpose?:string}).purpose !== 'inference' && !modelAssetUnsupportedReason(m) && m.kind === kind && (!familyName || m.family === familyName || m.compatible_families?.includes(familyName))) : [];

  return (
    <div className={`model-path-control ${matched.length > 0 ? 'has-registry' : ''}`}>
      <PathInput ariaLabel={label} value={value} defaultPath={defaultPath} resolveDefaultPath={resolveDefaultPath} directoryOnly={directoryOnly} allowMissingDirectory={allowMissingDirectory} onChange={onChange} />
      {matched.length > 0 && (
        <StudioSelect aria-label={`${label || kind} · ${t('models.fromRegistry')}`} value={matched.some(model => model.path === value) ? value : ''} onValueChange={onChange} data-testid="model-registry-select"
          placeholder={t('models.fromRegistry')} options={matched.map(model=>({value:model.path,label:model.path.split(/[\\/]/).pop() || model.path}))}/>

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
  return <div className="resolution-editor"><input aria-label={label} inputMode="numeric" value={draft} onChange={event => update(event.target.value)} placeholder="1024, 1536"/></div>;
}

/** Comma- or space-separated numbers such as validation timesteps. */
function NumberListInput({value, onChange, label, placeholder}: {value: number[] | string; onChange: (next: number[] | string) => void; label: string; placeholder?: string}) {
  const encoded = Array.isArray(value) ? value.join(', ') : String(value || '');
  const [draft, setDraft] = React.useState(encoded);
  React.useEffect(() => setDraft(encoded), [encoded]);
  const update = (raw: string) => {
    setDraft(raw);
    const tokens = raw.replace(/[\u005b\u005d]/g, '').split(/[,，\s]+/).filter(Boolean);
    onChange(tokens.length && tokens.every(token => token.trim() !== '' && Number.isFinite(Number(token))) ? tokens.map(Number) : raw);
  };
  return <input aria-label={label} inputMode="decimal" value={draft} onChange={event => update(event.target.value)} placeholder={placeholder}/>;
}

/** Two related numbers, each named inside its own box. */
function PairInput({value, onChange, names, label, step}: {value: [number | '', number | '']; onChange: (next: [number | '', number | '']) => void; names: [string, string]; label: string; step: string}) {
  const pair = Array.isArray(value) ? value : ['', ''] as [number | '', number | ''];
  const update = (index: 0 | 1, raw: string) => {
    const next: [number | '', number | ''] = [pair[0] ?? '', pair[1] ?? ''];
    next[index] = raw === '' ? '' : Number(raw);
    onChange(next);
  };
  return <div className="config-beta-controls">{names.map((name, index) => <label key={name}><span aria-hidden="true">{name}</span>
    <input type="number" step={step} value={pair[index] ?? ''} aria-label={`${label} · ${name}`} onChange={event => update(index as 0 | 1, event.target.value)}/>
  </label>)}</div>;
}

const PAIR_NAMES: Record<string, { names: [[string, string], [string, string]]; step: string }> = {
  'objective.res_shift_tokens': { names: [['小图', 'Small'], ['大图', 'Large']], step: '1' },
  'objective.res_shift_mu': { names: [['小图', 'Small'], ['大图', 'Large']], step: '0.01' },
};

function NativePixelLimit({value, label, onChange, describedBy}: {value: number | string; label: string; onChange: (next: number | string) => void; describedBy?: string}) {
  const side = typeof value === 'number' && value > 0 ? Number(Math.sqrt(value).toFixed(2)) : '';
  return <input id="config-dataset.native_max_pixels" aria-label={label} aria-describedby={describedBy} type="number" min={32} max={8192} step="any" value={side}
    onChange={event=>onChange(event.target.value === '' ? '' : Math.round(Number(event.target.value) ** 2))}/>;
}

const nativePixelsHint = (value: unknown, english: boolean) => {
  if (typeof value !== 'number' || value <= 0) return undefined;
  return english
    ? `Up to ${(value / 1e6).toLocaleString('en', {maximumFractionDigits:2})} megapixels; keeps the image aspect ratio.`
    : `最多约 ${(value / 1e4).toLocaleString('zh-CN', {maximumFractionDigits:value < 1e4 ? 2 : 0})} 万像素，保持原图比例。`;
};

/** Nullable values keep their type; an empty numeric draft becomes null on blur. */
const SchemaValueInput: React.FC<{
  schema: any; property: SchemaProperty; value: any; name: string; placeholder?: string; compact?: boolean; onChange: (value: any) => void;
}> = ({ schema, property, value, name, placeholder, compact = false, onChange }) => {
  const { t, i18n } = useTranslation();
  const alternatives = property.anyOf || [property];
  const nullable = alternatives.some(p => p.type === 'null');
  const constant = alternatives.find(p => p.const !== undefined);
  const branch = alternatives.find(p => p.type !== 'null' && p.const === undefined) || alternatives[0];
  const resolved = branch.$ref ? { ...resolveRef(schema, branch.$ref), ...branch } : branch;
  const prop: SchemaProperty = { ...property, ...resolved };
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  const emptyLabel = optionalValueLabel(name, english, placeholder);
  const algorithmChoice = nullable && name === 'optimizer.eps';
  const lastExplicit = React.useRef<any>();
  if (value != null && value !== '' && value !== constant?.const) lastExplicit.current = value;
  const initialValue = () => {
    if (lastExplicit.current !== undefined) return lastExplicit.current;
    if (prop.default != null && prop.default !== constant?.const) return prop.default;
    if (prop.type === 'object') return {};
    if (prop.type === 'boolean') return false;
    if (prop.type === 'string') return '';
    if (prop.type === 'array') return [];
    const familyDefault = placeholder?.trim() ? Number(placeholder) : Number.NaN;
    if (Number.isFinite(familyDefault)) return familyDefault;
    const minimum = prop.minimum ?? prop['x-ui']?.min ?? 0;
    return prop.exclusiveMinimum != null && minimum <= prop.exclusiveMinimum
      ? prop.exclusiveMinimum + (prop['x-ui']?.step || 1) : minimum;
  };
  const cls = 'w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600';
  if (prop.type === 'object' && prop.properties) return <div className="config-optional-object">
    {nullable && <StudioSelect aria-label={`${name}.mode`} value={value == null ? 'none' : 'custom'}
      onValueChange={next => onChange(next === 'none' ? null : initialValue())}
      options={[{value:'none',label:english ? 'Disabled' : 'Disabled (不使用)'}, {value:'custom',label:english ? 'Configure' : 'Configure (填写参数)'}]}/>}
    {value != null && Object.entries(prop.properties).map(([key, child]) => <div key={key} className="space-y-1 text-xs">
      <span>{t(`fields.${key}`, child.title || key)}</span>
      <SchemaValueInput schema={schema} property={child} value={value[key] === undefined ? child.default : value[key]}
        name={`${name}.${key}`} compact={compact} onChange={next => onChange({ ...value, [key]: next })}/>
    </div>)}
  </div>;
  if (prop.enum) return <StudioSelect aria-label={name} value={value == null ? '' : String(value)}
    onValueChange={next => onChange(next === '' && nullable ? null : prop.enum!.find(item => String(item) === next))}
    options={[...(nullable ? [{value:'',label:emptyLabel}] : []), ...prop.enum.map(item => ({value:String(item),label:String(item)}))]}/>;
  if (prop.type === 'boolean') return nullable
    ? <StudioSelect aria-label={name} value={value == null ? '' : String(value)} onValueChange={next => onChange(next === '' ? null : next === 'true')}
      options={[{value:'',label:emptyLabel},{value:'true',label:english ? 'Enabled' : 'Enabled (开启)'},{value:'false',label:english ? 'Disabled' : 'Disabled (关闭)'}]}/>
    : <Switch aria-label={name} checked={!!value} onCheckedChange={onChange}/>;
  if (prop.type === 'array') return <textarea className={cls} aria-label={name} value={typeof value === 'string' ? value : JSON.stringify(value ?? [])}
    onChange={event => { try { onChange(JSON.parse(event.target.value)); } catch { onChange(event.target.value); } }}/>
  const numeric = prop.type === 'integer' || prop.type === 'number';
  const input = <input id={`config-${name}`} className={cls} aria-label={name} type={numeric ? 'number' : 'text'}
    value={value ?? ''} min={prop.minimum ?? prop['x-ui']?.min} max={prop.maximum ?? prop['x-ui']?.max}
    step={prop['x-ui']?.step ?? (prop.type === 'integer' ? 1 : 'any')}
    placeholder={nullable && numeric && !algorithmChoice ? emptyLabel : placeholder}
    title={nullable && numeric && !algorithmChoice ? `${english ? 'Leave blank: ' : '留空：'}${emptyLabel}` : undefined}
    onBlur={event => { if (nullable && !algorithmChoice && value === '' && event.currentTarget.value === '' && !event.currentTarget.validity.badInput) onChange(null); }}
    onChange={event => onChange(event.target.value === '' ? (nullable && !numeric ? null : '') : numeric ? Number(event.target.value) : event.target.value)}/>;
  if (algorithmChoice || constant) return <div className="config-value-mode">
    <StudioSelect aria-label={`${name}.mode`} value={algorithmChoice ? value == null ? 'automatic' : 'custom' : value === constant?.const ? 'automatic' : 'custom'}
      onValueChange={next => onChange(next === 'automatic' ? algorithmChoice ? null : constant?.const : initialValue())}
      options={[{value:'custom',label:algorithmChoice ? 'EPS' : english ? 'Custom value' : 'Custom (指定数值)'}, {value:'automatic',label:algorithmChoice ? 'Adam-atan2' : String(constant?.const)}]}/>
    {(algorithmChoice ? value != null : value !== constant?.const) && input}
  </div>;
  return input;
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
  const bodyId = React.useId();
  return (
    <section data-group={groupKey} className={compact ? 'config-group' : 'border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800'}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-controls={bodyId}
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
      <div id={bodyId} hidden={!isOpen} className={compact ? 'config-fields' : 'p-4 space-y-4'}>{children}</div>
    </section>
  );
};

export const SchemaForm: React.FC<SchemaFormProps> = ({
  schema,
  value: sourceValue,
  onChange: onValueChange,
  showAdvanced = false,
  errors = [],
  notices = [],
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
  computePolicy,
  preset = false,
  projectId, versionId,
}) => {
  const { t, i18n } = useTranslation();
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  const value = React.useMemo(() => normalizeOptimizerConfig(schema, sourceValue), [schema, sourceValue]);
  const activeComputePolicy = confirmedTrainingComputePolicy(computePolicy, value);
  const [modelAssets, setModelAssets] = React.useState<ModelAsset[]>([]);
  React.useEffect(() => {
    let active = true;
    const refresh = () => void apiClient.get<ModelAsset[]>('/models', { silent: true }).then(items => { if (active) setModelAssets(Array.isArray(items) ? items : []); }).catch(() => {});
    refresh(); window.addEventListener('studio-models-changed', refresh); window.addEventListener('focus', refresh);
    return () => { active = false; window.removeEventListener('studio-models-changed', refresh); window.removeEventListener('focus', refresh); };
  }, []);
  const selectedModel = modelAssets.find(asset => asset.family === value.model?.family && asset.kind === 'dit' && asset.path === value.model?.dit_path);
  React.useEffect(() => {
    if (!readOnly && value.model?.family === 'krea2' && selectedModel?.variant && value.model.krea2_variant !== selectedModel.variant) {
      onValueChange({...sourceValue, model:{...sourceValue.model, krea2_variant:selectedModel.variant}});
    }
  }, [readOnly, selectedModel?.variant, sourceValue, value.model, onValueChange]);

  const scheduleFree = value.optimizer?.type === 'adamw_sf' || (value.optimizer?.type === 'prodigy_plus_sf' && value.optimizer?.use_schedulefree !== false);
  const [editCaptionOverrides, setEditCaptionOverrides] = React.useState(false);
  const [editOutput, setEditOutput] = React.useState(false);
  const lastLowRank = React.useRef<number | null>(null);
  const lastEmittedConfig = React.useRef<Record<string, any> | null>(null);
  const optimizerEdits = React.useRef(new Map<string, Record<string, any>>());
  React.useEffect(() => {
    // Only retain a rank across changes emitted by this editor. Selecting another
    // preset or loading an external draft must not inherit this temporary value.
    if (sourceValue !== lastEmittedConfig.current) {
      lastLowRank.current = null;
      optimizerEdits.current.clear();
    }
    lastEmittedConfig.current = null;
  }, [sourceValue]);
  const captionOverrideKeys = ['prefix', 'suffix', 'trigger_word'];
  const hasCaptionOverrides = captionOverrideKeys.some(key => !!value.dataset?.caption?.[key]);
  const onChange = (next: Record<string, any>) => {
    if (readOnly) return;
    next = selectTrainingComponents(next, value);
    const type = next.optimizer?.type;
    if (type !== undefined && type !== value.optimizer?.type) {
      optimizerEdits.current.set(value.optimizer?.type ?? 'adamw', value);
      const previous = optimizerEdits.current.get(type);
      next = previous ? restoreOptimizerSelection(schema, next, previous) : selectOptimizer(schema, next, type);
    }
    const normalized = normalizeOptimizerConfig(schema, next);
    lastEmittedConfig.current = normalized;
    onValueChange(normalized);
  };
  const lokrModeLabel = english ? 'LoKr parameter mode' : 'LoKr 参数形式';
  const groups: Record<string, { order: number; fields: React.ReactNode[] }> = {};
  const conditionValue = { ...value, training: {mode:'adapter', ...value.training}, model: { prediction_type: 'epsilon', ...value.model }, dataset: { resolution_mode: 'bucket', image_fit: 'crop', ...value.dataset } };
  const weights = modelFamilyWeights(family);
  const fieldContext: FieldContext = { family, config: value, english };

  const renderField = (key: string, prop: SchemaProperty, parentPath: string[] = []) => {
    const path = [...parentPath, key];
    const fullPathKey = path.join('.');
    if (fullPathKey === 'checkpoint.save_state_every_epochs') return null;
    if (['checkpoint.output_dir', 'checkpoint.state_dir', 'sampling.output_dir', 'logging.output_dir', 'logging.events_path', 'dataset.cache_dir'].includes(fullPathKey)) return null;
    const lokrRank = fullPathKey === 'adapter.rank' && value.adapter?.algo === 'lokr';
    const weightMeta = parentPath[0] === 'model' ? weights.find(weight => weight.field === key) : undefined;
    const familyOptions = familyParameterOptions(family, fullPathKey);
    // Full fine-tuning hides the adapter settings but its locked layer types.
    const layerLock = fullPathKey === 'adapter.layer_types' ? adapterLayerTypesLock(value, family, english) : null;
    if (parentPath[0] === 'adapter' && value.training?.mode === 'full' && !layerLock) return null;
    if (['model.training_guidance', 'sampling.guidance'].includes(fullPathKey)) return null;
    const ddpmModifier = ['objective.scale_v_pred_loss_like_noise_pred', 'objective.v_pred_like_loss', 'objective.debiased_estimation_loss'].includes(fullPathKey);
    const incompatibleFamilyLoss = ddpmModifier && family?.objective !== 'ddpm' && !!getNestedValue(value, path);
    if (ddpmModifier && family?.objective !== 'ddpm' && !incompatibleFamilyLoss) return null;
    if (fullPathKey === 'model.krea2_variant' && selectedModel?.variant && selectedModel.variant === value.model?.krea2_variant) return null;
    if (fullPathKey === 'model.text_encoder_2_path' && value.model?.family !== 'sdxl') return null;
    if (familyOptions?.length === 0) return null;
    if (family?.objective === 'ddpm' && ['sampling.shift', 'sampling.er_sde_order', 'sampling.er_sde_s_noise', 'objective.shift', 'objective.res_shift_tokens', 'objective.res_shift_mu', 'objective.mode_scale'].includes(fullPathKey)) return null;
    if (parentPath[0] === 'model' && key in MODEL_PATH_FIELDS && weights.length && !weightMeta && !(key === 'tokenizer_path' && family?.name === 'sdxl')) return null;
    if (family?.name === 'sdxl' && weightMeta?.required === false && !showAdvanced) return null;
    if (fullPathKey === 'model.zero_terminal_snr' && value.model?.prediction_type !== 'v_prediction' && !value.model?.zero_terminal_snr) return null;
    // Keep legacy cloud-log data in the draft, but do not expose controls that enable it.
    if (fullPathKey === 'logging.wandb' || fullPathKey.startsWith('logging.wandb.')) return null;

    if (versionSources && fullPathKey === 'checkpoint.name' && !showAdvanced && !editOutput) return null;
    // Settings the selected adapter form ignores: full LoKr factors fix the scale, and
    // full target-layer weights take no rank, scale, initialization or dropout.
    if (value.adapter?.algo === 'lokr' && value.adapter?.rank === 'full' && ['adapter.alpha', 'adapter.decompose_both', 'adapter.rs_lora'].includes(fullPathKey)) return null;
    if (value.adapter?.algo === 'full' && ['adapter.rank', 'adapter.alpha', 'adapter.rs_lora', 'adapter.init', 'adapter.dropout', 'adapter.rank_dropout', 'adapter.conv_rank', 'adapter.conv_alpha'].includes(fullPathKey)) return null;
    // Convolutions that follow a full LoKr rank take its fixed scale too.
    if (fullPathKey === 'adapter.conv_alpha' && value.adapter?.algo === 'lokr' && value.adapter?.rank === 'full' && value.adapter?.conv_rank == null) return null;
    // T-LoRA masks ranks per sample and LyCORIS Full trains whole weights; neither takes DoRA. A saved
    // DoRA stays visible so it can be turned off.
    if (fullPathKey === 'adapter.dora' && ['tlora', 'full'].includes(value.adapter?.algo) && !value.adapter?.dora) return null;
    const ui = { ...(prop['x-ui'] || {}), ...(compact && parentPath[0] === 'training' ? {group:'model'} : {}), ...(compact && fullPathKey === 'dataset.batch_size' ? {group:'loop'} : {}), ...(fullPathKey === 'model.attention' ? {group:'memory',advanced:false} : {}), ...(fullPathKey === 'loop.gpu_count' ? {group:'loop',advanced:false} : {}), ...(fullPathKey === 'adapter.layer_types' && value.training?.mode === 'full' ? {group: compact ? 'model' : 'training'} : {}) };
    if (ui.hidden) return null;
    if (conditionValue.dataset.resolution_mode === 'native' && ['dataset.resolutions', 'dataset.aspect_ratio_limit', 'dataset.area_tolerance', 'dataset.bucket_step', 'dataset.bucket_no_upscale'].includes(fullPathKey)) return null;
    if (compact && !showAdvanced && fullPathKey === 'adapter.rules' && !value.adapter?.rules?.length) return null;

    const nested = prop.$ref ? resolveRef(schema, prop.$ref) : prop;
    if (nested?.type === 'object' && nested.properties) {
      Object.entries(nested.properties).forEach(([childKey, child]) => renderField(childKey, child as SchemaProperty, path));
      return null;
    }
    const backboneLabel = fullPathKey === 'training.train_backbone' && family
      ? (family.name === 'sdxl' ? (english ? 'Train main model (UNet)' : '训练主模型（UNet）') : (english ? 'Train main model (DiT)' : '训练主模型（DiT）'))
      : undefined;
    const fieldLabel = weightMeta?.label || backboneLabel || (fullPathKey === 'objective.snr_gamma' && value.objective?.weighting === 'min_snr' ? 'Min-SNR Gamma' : configFieldLabel(fullPathKey, t(`fields.${key}`, prop.title || key), english));
    const fieldId = `config-${fullPathKey}`;
    const currentGroup = ui.group || parentPath[0] || 'default';
    if (groupFilter && !groupFilter.includes(currentGroup) && !(compact && parentPath[0] === 'training' && groupFilter.includes('training'))) return null;
    if (search.trim() && !`${fieldLabel} ${fullPathKey} ${fullPathKey === 'checkpoint.save_state_every_steps' ? 'checkpoint.save_state_every_epochs epoch 轮' : ''} ${prop.description || ''} ${lokrRank ? lokrModeLabel : ''}`.toLowerCase().includes(search.trim().toLowerCase())) return null;

    const captionOverride = fullPathKey.startsWith('dataset.caption.') && captionOverrideKeys.includes(key);
    const optionalAdvanced = ui.advanced && !(currentGroup === 'optimizer' && key !== 'args');
    if ((optionalAdvanced || captionOverride) && !showAdvanced && !(captionOverride && editCaptionOverrides) && !(editOutput && fullPathKey === 'checkpoint.name')) return null;
    const incompatiblePredictionLoss = (fullPathKey === 'objective.scale_v_pred_loss_like_noise_pred' && value.objective?.scale_v_pred_loss_like_noise_pred && conditionValue.model.prediction_type !== 'v_prediction') || (fullPathKey === 'objective.v_pred_like_loss' && value.objective?.v_pred_like_loss > 0 && conditionValue.model.prediction_type !== 'epsilon');
    if (ui.show_when && !incompatiblePredictionLoss && !incompatibleFamilyLoss) {
      try {
        const previewConditions = fullPathKey === 'loop.ema_decay'
          ? {...conditionValue,loop:{...value.loop,ema:true}}
          : fullPathKey === 'dataset.caption.keep_tokens'
            ? {...conditionValue,dataset:{...conditionValue.dataset,caption:{...value.dataset?.caption,shuffle:true}}}
            : conditionValue;
        if (!evaluateShowWhen(ui.show_when, previewConditions)) return null;
      } catch (err) {
        console.error(`[SchemaForm] Failed to evaluate show_when for ${fullPathKey}: "${ui.show_when}"`, err);
        // On evaluation error, default to showing the field
      }
    }

    const errorItem = errors.find((e) => e.loc === fullPathKey || e.loc?.startsWith(`${fullPathKey}.`) || fullPathKey === 'checkpoint.save_state_every_steps' && e.loc === 'checkpoint.save_state_every_epochs');
    const computeManaged: {value: unknown; label?: string; reason: string; tag?: string} | null = layerLock ?? trainingComputeManagedField(activeComputePolicy, fullPathKey, english);
    const fieldValue = computeManaged ? computeManaged.value : getNestedValue(value, path) !== undefined ? getNestedValue(value, path) : prop.default;
    const groupName = ui.group || (parentPath.length > 0 ? parentPath[0] : 'default');
    const compactField = compact || groupName === 'adapter';
    const percentage = PERCENTAGE_FIELDS.includes(fullPathKey);
    const numericMin = ui.min ?? prop.minimum ?? prop.exclusiveMinimum;
    const numericStep = ui.step ?? (prop.type === 'integer' ? 1 : 0.01);
    const numericMax = ui.max ?? prop.maximum ?? prop.exclusiveMaximum;
    const managedReason = computeManaged?.reason || trainingManagedReason(value, fullPathKey, english) || optimizerManagedReason(schema, value, fullPathKey, english);
    if (optimizerManagedReason(schema, value, fullPathKey, english) && ['optimizer.kahan', 'optimizer.group_lr', 'adapter.lr_scale', 'scheduler.warmup_steps'].includes(fullPathKey)) return null;
    // Only what this model and machine use: an unused setting shows only while it is still set, with the reason.
    const unusedReason = computeManaged ? undefined : unusedSettingReason(fullPathKey, fieldContext);
    if (unusedReason && hideUnusedSetting(fullPathKey, fieldValue, fieldContext)) return null;
    const schemaOptions = (ui.options ?? prop.enum)?.map(String);
    const supportedOptions = contextOptions(fullPathKey, fieldContext, familyOptions ?? schemaOptions, fieldValue) ?? familyOptions;
    const offeredOptions = fullPathKey === 'model.attention' ? supportedOptions?.filter(option => option !== 'auto') : supportedOptions;
    // A choice whose package this environment lacks stays listed, greyed out, so the reason is visible where it is chosen.
    const unavailableOptions = family?.unavailable_options?.[fullPathKey];
    const choice = (option: string): StudioSelectOption => {
      const label = configOptionLabel(fullPathKey, option, english);
      const reason = unavailableOptions?.[option];
      if (!reason) return { value: option, label };
      const note = reason === 'wrong_version' ? (english ? 'wrong version' : '版本不符') : (english ? 'not installed' : '未安装');
      return { value: option, label: `${label} (${note})`, disabled: true };
    };
    // A model that offers a single choice (SDXL's preview scheduler) has nothing to pick.
    if (familyOptions && offeredOptions?.length === 1 && !managedReason && [offeredOptions[0], 'auto', undefined, null, ''].includes(fieldValue)) return null;

    let control = null;

    // 1. 递归对象渲染
    if (managedReason && ['training.train_backbone', 'training.train_text_encoder'].includes(fullPathKey)) {
      control = <div className="config-toggle-control" data-state={fieldValue ? 'on' : 'off'}>
        <Switch id={fieldId} aria-label={fieldLabel} aria-describedby={`${fieldId}-managed-reason`} checked={!!fieldValue} disabled></Switch>
      </div>;
    } else if (managedReason) {
      const display = computeManaged?.label ?? (prop.enum ? configOptionLabel(fullPathKey, String(fieldValue), english) : managedValueLabel(fieldValue, english));
      control = <div className="config-managed-value"><output id={fieldId} aria-label={fieldLabel} aria-describedby={`${fieldId}-managed-reason`}>{display}</output><span>{computeManaged?.tag ?? (english ? 'Automatic' : '自动管理')}</span></div>;
    } else if (fullPathKey === 'checkpoint.save_state_every_steps') {
      control = <RecoveryInterval id={fieldId} label={fieldLabel} english={english} invalid={!!errorItem}
        value={{save_state_every_steps:fieldValue ?? null,save_state_every_epochs:value.checkpoint?.save_state_every_epochs ?? null}}
        onChange={interval=>onChange(setNestedValue(value,['checkpoint'],{...value.checkpoint,...interval}))}/>;
    } else if (fullPathKey === 'dataset.crop_anchor') {
      control = <StudioSelect id={fieldId} aria-label={fieldLabel} aria-describedby={`${fieldId}-hint`} aria-invalid={!!errorItem}
        value={fieldValue ?? 'center'} optionColumns={3}
        options={(prop.enum || []).map((option:string)=>({value:option,label:configOptionLabel(fullPathKey,option,english)}))}
        onValueChange={next=>onChange(setNestedValue(value,path,next))}/>;
    } else if (fullPathKey === 'dataset.native_max_pixels') {
      control = <NativePixelLimit value={fieldValue} label={fieldLabel} describedBy={nativePixelsHint(fieldValue, english) ? `${fieldId}-hint` : undefined} onChange={next => onChange(setNestedValue(value, path, next))}/>;
    } else if (compact && fullPathKey === 'dataset.resolutions') {
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
          learningRateReason={optimizerManagedReason(schema, value, 'adapter.rules.lr', english)}
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
    } else if (PAIR_NAMES[fullPathKey]) {
      const pair = PAIR_NAMES[fullPathKey];
      control = <PairInput value={fieldValue} label={fieldLabel} step={pair.step} names={[pair.names[0][english ? 1 : 0], pair.names[1][english ? 1 : 0]]}
        onChange={val => onChange(setNestedValue(value, path, val))}/>;
    } else if (key === 'betas') {
      control = (
        <BetasEditor
          value={fieldValue || [0.9, 0.999]}
          scheduleFree={scheduleFree}
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
          addLabel={({'optimizer.group_lr': english ? 'Add parameter group' : '添加参数组', 'optimizer.args': english ? 'Add argument' : '添加参数', 'adapter.lr_scale': english ? 'Add multiplier' : '添加倍率'} as Record<string, string>)[fullPathKey]}
          onChange={(val) => onChange(setNestedValue(value, path, val))}
        />
      );
    } else if (ui.control === 'path' || prop.type === 'string' && key.endsWith('path')) {
      const modelKind = parentPath[0] === 'model' ? weightMeta?.kind || MODEL_PATH_FIELDS[key as keyof typeof MODEL_PATH_FIELDS] || null : null;
      control = (
        <ModelPathInput
          value={fieldValue || ''}
          directoryOnly={fullPathKey === 'checkpoint.resume'}
          kind={modelKind}
          label={fieldLabel}
          familyName={value.model?.family}
          models={modelAssets}
          resolveDefaultPath={['checkpoint.resume','adapter.resume_weights','training.resume_weights'].includes(fullPathKey)
            ? async()=> (await apiClient.get<{path:string}>('/fs/browse-root',{params:{field:fullPathKey,project_id:projectId,version_id:versionId,output_dir:value.checkpoint?.output_dir,custom_dir:fullPathKey==='checkpoint.resume'?value.checkpoint?.state_dir:undefined},silent:true})).path : undefined}
          onChange={(val) => {
            let next = setNestedValue(value, path, val === '' && prop.anyOf?.some((p) => p.type === 'null') ? null : val);
            if (fullPathKey === 'model.dit_path' && value.model?.family === 'krea2') {
              const asset = modelAssets.find(item => item.path === val && item.family === 'krea2' && item.kind === 'dit');
              next = setNestedValue(next, ['model', 'krea2_variant'], asset?.variant || 'auto');
            }
            onChange(next);
          }}
        />
      );
    } else if (fullPathKey === 'adapter.rank') {
      control = <input type="number" aria-label={fieldLabel} min="1" step="1" value={fieldValue === 'full' ? '' : fieldValue ?? 16}
        onChange={event => {
          const rank = event.target.value === '' ? '' : Number(event.target.value);
          if (typeof rank === 'number' && Number.isInteger(rank) && rank > 0) lastLowRank.current = rank;
          onChange(setNestedValue(value, path, rank));
        }}/>;
    } else if (prop.anyOf) {
      const inputProperty = fullPathKey === 'optimizer.eps' && value.optimizer?.type !== 'prodigy_plus_sf'
        ? {...prop, anyOf: prop.anyOf.filter(branch => branch.type !== 'null')} : prop;
      control = <SchemaValueInput schema={schema} property={inputProperty} value={fieldValue} name={fullPathKey} compact={compactField}
        placeholder={family && fullPathKey.startsWith('sampling.') && ['steps', 'cfg', 'shift'].includes(key) ? (family.sampling?.[key as 'steps' | 'cfg' | 'shift'] != null ? String(family.sampling[key as 'steps' | 'cfg' | 'shift']) : key === 'shift' ? t('sampling.shiftAuto') : undefined) : undefined}
        onChange={(val) => onChange(setNestedValue(value, path, val))} />;
    } else if (fullPathKey === 'model.flux2_variant') {
      const options = ['auto', 'klein-base-4b', 'klein-base-9b'];
      control = <StudioSelect aria-label={fieldLabel} value={fieldValue || 'auto'}
        onValueChange={next => onChange(setNestedValue(value, path, next))}
        options={[...(fieldValue === 'dev' ? [{ value: 'dev', label: configOptionLabel(fullPathKey, 'dev', english), disabled: true }] : []),
          ...options.map(option => ({ value: option, label: configOptionLabel(fullPathKey, option, english) }))]}/>;
    } else if (fullPathKey === 'model.family' && families) {
      control = <StudioSelect aria-label="model.family" value={fieldValue || families[0]?.name || ''} disabled={!families.length}
        onValueChange={next => onChange(setNestedValue(value,path,next))} options={trainingFamilyOptions(families, english, fieldValue)}/>;
    } else if (fullPathKey === 'adapter.preset' && family) {
      // Keep the stable preset ID in the value; show the family-owned explanation.
      control = (
        <StudioSelect aria-label={fieldLabel} value={fieldValue || family.default_preset || ''} data-testid="adapter-preset-select"
          onValueChange={next => onChange(setNestedValue(value,path,next))} options={(family.presets || []).map(preset=>({value:preset.name,label:configPresetLabel(preset.name, preset.description, family.default_preset, english)}))}/>
      );
    } else if (fullPathKey === 'dataset.text_encoding' && family) {
      // 文本编码选项受族 text_modes 约束（krea2 无 online）
      control = <StudioSelect aria-label={fieldLabel} value={fieldValue || 'auto'} data-testid="text-encoding-select"
        onValueChange={next => onChange(setNestedValue(value,path,next))} options={(family.text_modes || []).map(mode=>({value:mode,label:configOptionLabel(fullPathKey,mode,english)}))}/>;
    } else if (fullPathKey === 'model.krea2_variant') {
      control = <StudioSelect aria-label={fieldLabel} value={fieldValue === 'auto' ? '' : fieldValue || ''}
        placeholder={english ? 'Confirm local model type' : '确认本地模型类型'}
        onValueChange={next => onChange(setNestedValue(value, path, next))}
        options={['raw', 'turbo'].map(option => ({value:option, label:configOptionLabel(fullPathKey, option, english)}))}/>;
    } else if (fullPathKey === 'model.dtype') {
      control = <StudioSelect aria-label={fieldLabel} value={fieldValue || 'auto'}
        onValueChange={next => onChange(setNestedValue(value, path, next))}
        options={['auto', 'bf16', 'fp16', 'fp32'].map(option => ({value:option, label:configOptionLabel(fullPathKey, option, english)}))}/>;
    } else if (fullPathKey === 'model.attention') {
      const attentionOptions = offeredOptions?.length ? offeredOptions : ['sdpa'];
      control = <StudioSelect aria-label={fieldLabel} value={!fieldValue || fieldValue === 'auto' ? 'sdpa' : String(fieldValue)} fallbackLabel={configOptionLabel(fullPathKey, String(fieldValue), english)}
        onValueChange={next => onChange(setNestedValue(value, path, next))}
        options={attentionOptions.map(choice)}/>;
    } else if (supportedOptions) {
      control = <StudioSelect aria-label={fieldLabel} value={fieldValue == null ? '' : String(fieldValue)}
        onValueChange={next => onChange(setNestedValue(value, path, next))}
        options={supportedOptions.map(choice)}/>;
    } else if (ui.options?.length) {
      control = <div className="space-y-2"><StudioSelect aria-label={fieldLabel} value={ui.options.includes(fieldValue) || !ui.allow_custom ? fieldValue || '' : '__custom__'} fallbackLabel={String(fieldValue || '')}
        onValueChange={next => onChange(setNestedValue(value,path,next === '__custom__' ? '' : next))}
        options={[...ui.options.map(choice),...(ui.allow_custom ? [{value:'__custom__',label:english?'Custom Python class…':'自定义 Python 类…'}] : [])]}/>
        {ui.allow_custom && !ui.options.includes(fieldValue) && <input aria-label={`${fieldLabel} ${english?'custom class':'自定义类'}`} value={fieldValue || ''} placeholder="package.module.OptimizerClass" onChange={event=>onChange(setNestedValue(value,path,event.target.value))}/>}
      </div>;
    } else if (prop.enum) {
      control = (
        <StudioSelect aria-label={fieldLabel} value={fieldValue == null ? '' : String(fieldValue)}
          onValueChange={next => {
            const updated = setNestedValue(value, path, prop.enum!.find(option => String(option) === next));
            // A full LoKr factor matrix is not a valid numeric rank for LoRA/LoHa.
            onChange(fullPathKey === 'model.prediction_type'
              ? next === 'v_prediction'
                ? setNestedValue(updated, ['objective', 'v_pred_like_loss'], 0)
                : setNestedValue(setNestedValue(updated, ['model', 'zero_terminal_snr'], false), ['objective', 'scale_v_pred_loss_like_noise_pred'], false)
              : fullPathKey === 'adapter.algo' && next !== 'lokr' && value.adapter?.rank === 'full'
              ? setNestedValue(updated, ['adapter', 'rank'], lastLowRank.current ?? 16)
              : updated);
          }}
          // Full-weight training of the target layers overlaps full fine-tuning; only configurations that already use it keep it.
          options={prop.enum.map(String).map(choice)}/>

      );
    } else if (prop.type === 'boolean' || ui.control === 'switch') {
      control = (
        <div className="config-toggle-control" data-state={fieldValue ? 'on' : 'off'}>
          <Switch id={fieldId} aria-label={fieldLabel} aria-invalid={!!errorItem} checked={!!fieldValue}
            onCheckedChange={checked => onChange(setNestedValue(value, path, checked))}>
          </Switch>
        </div>
      );
    } else if (prop.type === 'integer' || prop.type === 'number') {
      if ((percentage || ui.control === 'slider') && numericMin !== undefined && numericMax !== undefined) {
        // A slider stops one step short of an exclusive bound: a dropout stays below 100%.
        control = <NumericControl id={fieldId} label={fieldLabel} value={fieldValue} percentage={percentage} unit={ui.unit}
          min={numericMin} max={ui.max ?? prop.maximum ?? numericMax - numericStep} step={numericStep} invalid={!!errorItem}
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
          <DecimalNumberInput
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
            value={fieldValue ?? ''}
            min={numericMin}
            max={numericMax}
            step={ui.step ?? (prop.type === 'integer' ? 1 : 'any')}
            placeholder={placeholder}
            onChange={(e) => {
              // Empty is an unfinished edit, not a missing setting: undefined
              // immediately restores the schema default and prefixes the next input.
              const val = e.target.value === '' ? '' : Number(e.target.value);
              onChange(setNestedValue(value, path, val));
            }}
          />
        );
      }
    } else if (prop.type === 'array' && ['number', 'integer'].includes(prop.items?.type || '')) {
      control = <NumberListInput label={fieldLabel} value={fieldValue} placeholder={Array.isArray(prop.default) ? prop.default.join(', ') : undefined}
        onChange={next => onChange(setNestedValue(value, path, next))}/>;
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

    // Model files share a row; other paths, lists and editors take the full row.
    const modelFile = (parentPath[0] === 'model' && key in MODEL_PATH_FIELDS) || ['training.resume_weights', 'adapter.resume_weights'].includes(fullPathKey);
    const wide = !modelFile && (['sources', 'rules', 'prompts', 'args', 'group_lr'].includes(key) || ui.control === 'path' || key.endsWith('_path') || key === 'output_dir' || fullPathKey === 'adapter.lr_scale');
    const booleanField = prop.type === 'boolean' || ui.control === 'switch';
    if (!booleanField && !managedReason && React.isValidElement(control) && (typeof control.type === 'string' || control.type === StudioSelect || control.type === DecimalNumberInput)) {
      control = React.cloneElement(control as React.ReactElement<any>, {id: fieldId, 'aria-label': (control.props as any)['aria-label'] || fieldLabel, 'aria-invalid': !!errorItem});
    }
    // Small values also read in scientific form beside the label, e.g. 0.0001 = 1e-4.
    const scientific = !managedReason && (prop.type === 'number' || prop.anyOf?.some((variant: SchemaProperty) => variant.type === 'number')) && ui.control !== 'slider' && !percentage ? scientificText(fieldValue) : '';
    const scopeHelp = fullPathKey === 'adapter.preset' ? [english
      ? 'Selects which layers receive adapters; a wider scope trains more layers. LoKr Full controls how each adapter is parameterized. The two choices are independent and neither unfreezes the base model.'
      : '选择哪些层添加适配器，范围越大训练的层越多；LoKr 的 Full 决定每个适配器使用完整因子矩阵，两者可同时选择，都不会解冻底模。',
    familyHasConvolutions(family) && (english ? 'Whether convolution layers train too is chosen under Layer types.' : '是否同时训练卷积层，在“训练层类型”中选择。')].filter(Boolean).join('') || null : null;
    const selectedPreset = fullPathKey === 'adapter.preset' ? family?.presets?.find(preset => preset.name === (fieldValue || family.default_preset)) : undefined;
    const modelPrecisionHint = family?.runtime_backend === 'mps'
      ? (english ? 'The current Apple GPU uses FP32 for model loading and computation.' : '当前 Apple GPU 使用 FP32 加载和计算。')
      : family?.runtime_backend === 'cpu'
        ? (english ? 'The current CPU runtime uses FP32 for model loading and computation.' : '当前 CPU 环境使用 FP32 加载和计算。')
        : family?.runtime_backend === 'cuda' || family?.runtime_backend === 'hip'
          ? (english
            ? `Follow model reads the checkpoint precision.${value.model?.family === 'krea2' ? ' Supported FP8 weights use BF16 compute.' : ''}`
            : `跟随模型读取权重精度。${value.model?.family === 'krea2' ? '受支持的 FP8 权重使用 BF16 计算。' : ''}`)
          : (english ? 'Loads using a precision compatible with the model and runtime.' : '按模型与平台兼容的精度加载。');
    const dtkReproducibility = fullPathKey === 'loop.deterministic' && family?.runtime_backend === 'hip'
      ? (english ? 'DTK manages compute precision for the model and training mode. Effective settings appear after configuration validation and may increase memory use and runtime.' : 'DTK 会按模型与训练方式管理计算精度。实际设置在参数检查后显示，可能增加显存和耗时。')
      : undefined;
    const help = scopeHelp ? [
      scopeHelp,
      selectedPreset?.description,
      showAdvanced && selectedPreset && `${t('preset.layers', {n: selectedPreset.layers})} · ${selectedPreset.name}`,
      showAdvanced && presetHasConvolutions(selectedPreset) && (english
        ? `With convolutions: ${selectedPreset.layers_with_conv} layers, ${selectedPreset.conv_layers} of them convolutions`
        : `同时训练卷积层时：共 ${selectedPreset.layers_with_conv} 层，其中卷积层 ${selectedPreset.conv_layers} 个`),
      showAdvanced && selectedPreset?.include?.length && `${english ? 'Included layers' : '包含层'}：${selectedPreset.include.join(', ')}`,
      showAdvanced && selectedPreset?.exclude?.length && `${english ? 'Excluded layers' : '排除层'}：${selectedPreset.exclude.join(', ')}`,
    ].filter(Boolean).join('\n\n') : fullPathKey === 'model.tokenizer_path' && family?.name === 'sdxl' ? (english ? 'Optional root containing tokenizer/ and tokenizer_2/. Leave blank to use the model directory’s tokenizers, or the built-in CLIP-L / CLIP-G tokenizers when absent.' : '可选根目录，需同时包含 tokenizer/ 和 tokenizer_2/。留空自动读取模型目录；没有时使用内置 CLIP-L / CLIP-G 双分词器。')
      : [weightMeta?.hint ? [configFieldHelp(fullPathKey, undefined, english), weightMeta.hint].filter(Boolean).join('\n') : contextHelp(fullPathKey, fieldContext, offeredOptions) || configFieldHelp(fullPathKey, prop.description, english, value.optimizer?.type, scheduleFree), dtkReproducibility].filter(Boolean).join('\n\n');
    // Switches carry no standing description; a status or warning still shows beneath them.
    const statusHint = (fullPathKey === 'model.dit_path' && selectedModel ? [selectedModel.variant?.toUpperCase(), selectedModel.dtype?.toUpperCase()].filter(Boolean).join(' · ') : undefined)
      || (incompatibleFamilyLoss ? (english ? 'This loss option only supports SDXL. Turn it off or set it to zero before using this model.' : '此损失参数仅适用于 SDXL，请关闭或设为 0 后再使用当前模型。') : undefined)
      || (incompatiblePredictionLoss ? (english ? 'This option is incompatible with the selected prediction type. Turn it off or choose the matching prediction type.' : '此参数与当前预测方式不兼容，请关闭此项或选择对应的预测方式。') : undefined)
      || managedReason
      || unusedReason
      || (fullPathKey === 'loop.deterministic' ? trainingComputePolicyHint(activeComputePolicy, english) : undefined);
    const recoveryField = fullPathKey === 'checkpoint.save_state_every_steps';
    const describedHint = booleanField || recoveryField ? undefined
      : fullPathKey === 'training.mode' ? value.training?.mode === 'full'
        ? (english ? 'Updates the selected model weights directly.' : '直接训练所选模型本身的权重。')
        : (english ? 'Trains LoRA weights while keeping the base model frozen.' : '只训练 LoRA 权重，底模保持不变。')
      : fullPathKey === 'model.dtype' ? modelPrecisionHint
      : fullPathKey === 'dataset.native_max_pixels' ? nativePixelsHint(fieldValue, english) || configFieldHint(fullPathKey, english)
      : fullPathKey === 'dataset.text_encoding' && family && !(family.text_modes || []).includes('online') ? t('textMode.autoOnly')
      : parentPath[0] === 'model' && key in MODEL_PATH_FIELDS ? modelPathHint(family?.name, key, english, preset)
      : configFieldHint(fullPathKey, english, value.optimizer?.type, scheduleFree, value.dataset);
    const hint = statusHint || describedHint;
    const duplicateHelp = !!help && !!hint && help.replace(/\s+/g, ' ').trim() === hint.replace(/\s+/g, ' ').trim();
    const helpButton = help && !duplicateHelp ? <ConfigHelp label={`${fieldLabel} ${english ? 'help' : '说明'}`}>{help}</ConfigHelp> : null;
    if (recoveryField && React.isValidElement(control)) control = React.cloneElement(control as React.ReactElement<any>, {help:helpButton,error:errorItem?.msg});
    const body = readOnly ? <fieldset disabled className="config-readonly-control">{control}</fieldset> : control;
    const footer = <div className="config-field-footer">
      {hint && <p id={managedReason ? `${fieldId}-managed-reason` : `${fieldId}-hint`} className="config-field-hint">{hint}</p>}
      {/* A reason Studio cannot phrase for this field stays in the preflight panel; the border still marks it. */}
      {errorItem?.msg && <p className="config-field-error">{errorItem.msg}</p>}
      {!errorItem && notices.filter(notice => notice.loc === fullPathKey).map(notice => <p key={notice.msg} className="config-field-notice">
        <span>{notice.msg}</span>
        {notice.fix && !readOnly && <button type="button" className="ui-link" onClick={() => onChange(setNestedValue(value, path, notice.fix!.value))}>{notice.fix.label}</button>}
      </p>)}
    </div>;
    const fieldProps = {id: `field-${fullPathKey}`, 'data-testid': `field-${fullPathKey}`, 'data-field-path': fullPathKey};
    const label = recoveryField ? (
      <div key={fullPathKey} {...fieldProps} className={`config-field config-field-recovery${errorItem ? ' config-field-invalid' : ''}`}>
        {body}
      </div>
    ) : booleanField ? (
      <div key={fullPathKey} {...fieldProps} data-control-kind="toggle" className={`config-field config-field-boolean${errorItem ? ' config-field-invalid' : ''}`}>
        <div className="config-field-control">
          {body}
          <label htmlFor={fieldId}>{fieldLabel}</label>
          {helpButton}
        </div>
        {footer}
      </div>
    ) : (
      <div key={fullPathKey} {...fieldProps} data-field-span={wide ? 'wide' : undefined} className={`config-field${wide ? ' config-field-wide' : ''}${errorItem ? ' config-field-invalid' : ''}`}>
        <div className="config-field-heading">
          <label htmlFor={fieldId}>
            {fieldLabel}{weightMeta?.required === false && family?.name !== 'flux2' && <span className="config-field-label-note">{english ? ' (optional)' : '（可选）'}</span>}
            {ui.unit && !percentage && ui.control !== 'slider' && <span className="config-field-label-note"> ({ui.unit})</span>}
            {scientific && <span className="config-field-badge" title={english ? 'The same value in scientific notation' : '同一数值的科学计数法'}><span className="sr-only">{english ? ', scientific notation ' : '，科学计数法 '}</span>{scientific}</span>}
          </label>
          <span className="config-field-reference">
            <code className="config-field-key" tabIndex={0} title={fullPathKey === 'checkpoint.save_state_every_steps' && value.checkpoint?.save_state_every_epochs != null ? 'checkpoint.save_state_every_epochs' : fullPathKey}>{fullPathKey === 'checkpoint.save_state_every_steps' && value.checkpoint?.save_state_every_epochs != null ? 'checkpoint.save_state_every_epochs' : fullPathKey}</code>
            {helpButton}
          </span>
        </div>
        <div className="config-field-control">{body}</div>
        {footer}
      </div>
    );

    if (!groups[groupName]) {
      groups[groupName] = { order: ui.order || 0, fields: [] };
    }
    if (lokrRank) {
      const modeId = 'config-adapter-parameter-mode';
      const modeHint = configFieldHint('adapter.parameter_mode', english);
      groups[groupName].fields.push(<div key="adapter.parameter_mode" id={fieldValue === 'full' ? 'field-adapter.rank' : 'field-adapter.parameter_mode'} data-testid="field-adapter.parameter_mode" data-field-path="adapter.parameter_mode" className={`config-field${fieldValue === 'full' && errorItem ? ' config-field-invalid' : ''}`}>
        <div className="config-field-heading">
          <label htmlFor={modeId}>{lokrModeLabel}</label>
          <span className="config-field-reference"><code className="config-field-key" tabIndex={0} title={fullPathKey}>adapter.rank</code><ConfigHelp label={`${lokrModeLabel} ${english ? 'help' : '说明'}`}>{english ? 'Full retains the complete LoKr factor matrices; it does not fine-tune the whole model and does not use Alpha. Low rank decomposes the factors using Rank and Alpha.' : 'Full 保留 LoKr 完整因子矩阵，不是全量微调，也不使用 Alpha。低秩模式通过 Rank 和 Alpha 设置因子分解与缩放。'}</ConfigHelp></span>
        </div>
        <div className="config-field-control"><StudioSelect id={modeId} aria-label={lokrModeLabel} disabled={readOnly} value={fieldValue === 'full' ? 'full' : 'low_rank'}
          aria-invalid={fieldValue === 'full' && !!errorItem}
          onValueChange={next => {
            if (typeof fieldValue === 'number' && Number.isInteger(fieldValue) && fieldValue > 0) lastLowRank.current = fieldValue;
            const defaultRank = typeof prop.default === 'number' && Number.isInteger(prop.default) && prop.default > 0 ? prop.default : 16;
            onChange(setNestedValue(value, path, next === 'full' ? 'full' : lastLowRank.current ?? defaultRank));
          }} options={[
            {value: 'full', label: english ? 'Full' : 'Full (完整因子矩阵)'},
            {value: 'low_rank', label: english ? 'Low rank' : 'Low rank (分解因子矩阵)'},
          ]}/></div>
        <div className="config-field-footer">
          {modeHint && <p className="config-field-hint">{modeHint}</p>}
          {fieldValue === 'full' && errorItem?.msg && <p className="config-field-error">{errorItem.msg}</p>}
        </div>
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
  const renderToggleSection = (path: string, children: React.ReactNode) => {
    const features: Record<string, {title:string;description:string;switchLabel:string}> = {
      'loop.ema': {title:english?'Weight averaging (EMA)':'权重平均 (EMA)',description:english?'Keep a separate copy of averaged adapter weights for comparison.':'平滑适配器权重变化，额外保存平均权重用于对比。',switchLabel:english?'Enable EMA':'启用 EMA'},
      'dataset.caption.shuffle': {title:english?'Shuffle caption tags':'打乱标签顺序',description:english?'Randomize variable tags while keeping fixed tags in place.':'随机调整可变标签的顺序，固定标签保持不变。',switchLabel:english?'Shuffle caption tags':'打乱标签顺序'},
    };
    const fieldPath=path.split('.');
    let fieldSchema=schema;
    for(const key of fieldPath)fieldSchema=(fieldSchema?.$ref ? resolveRef(schema,fieldSchema.$ref) : fieldSchema)?.properties?.[key];
    const help=configFieldHelp(path,fieldSchema?.description,english);
    const feature=features[path];
    return <ParameterToggleSection key={`${path}:${search.trim()}`} id={`feature-${path}`} {...feature} fieldPath={path}
      parametersEnabled={path==='dataset.caption.shuffle' && value.dataset?.caption?.tag_dropout>0 ? true : undefined}
      checked={!!(getNestedValue(value,fieldPath) ?? fieldSchema?.default)} disabled={readOnly} onCheckedChange={checked=>onChange(setNestedValue(value,fieldPath,checked))}
      help={help ? <ConfigHelp label={`${feature.title} ${english?'help':'说明'}`}>{help}</ConfigHelp> : undefined}
      error={errors.find(error=>error.loc===path)?.msg}>{children}</ParameterToggleSection>;
  };

  return (
    <div className={compact ? 'compact-schema' : 'space-y-6'} data-testid="schema-form">
      {search.trim() && sortedGroups.length > 0 && <p className="config-search-results" role="status">{english ? `${sortedGroups.reduce((count, [, group]) => count + group.fields.length, 0)} matching parameters · ${sortedGroups.length} sections` : `${sortedGroups.reduce((count, [, group]) => count + group.fields.length, 0)} 个匹配参数 · ${sortedGroups.length} 个分组`}</p>}
      {sortedGroups.map(([groupName, groupData]) => (
        <FieldGroup key={`${groupName}:${search.trim()}`} title={parameterGroupLabel(groupName, english) || (groupName === 'training' ? (english ? 'Training mode' : '训练方式') : t(`groups.${groupName}`, groupName))} count={groupData.fields.length} compact={compact} groupKey={groupName}>
          {/* A problem with the whole group, such as a memory estimate too large for the GPU, has no field to sit under. */}
          {errors.filter(error => error.loc === groupName && error.msg).map(error => <p key={error.msg} className="config-group-alert"><AlertCircle size={14} aria-hidden="true"/><span>{error.msg}</span></p>)}
          {groupName === 'checkpoint' && versionSources && <div className="output-binding-summary">
            <div className="output-binding-heading"><strong>{english ? 'Training weights' : '训练权重'}</strong>{!showAdvanced && <button type="button" className="ui-btn ui-btn-sm" aria-expanded={editOutput} onClick={() => setEditOutput(previous => !previous)}>{editOutput ? (english ? 'Collapse file name' : '收起文件名设置') : (english ? 'Edit file name' : '修改文件名')}</button>}</div>
            {outputBinding ? <><div><span>{english ? 'File name' : '文件名'}</span><code>{outputBinding.name}-final{value.training?.mode === 'full' ? '.model/' : '.safetensors'}</code></div><div><span>{english ? 'Save location' : '保存位置'}</span><code>{outputBinding.directory_template.replace('{job_id}', english ? '<run ID>' : '<运行 ID>')}</code></div></> : <p><LoadingNote label={english ? 'Resolving the save location…' : '正在读取保存位置…'}/></p>}
          </div>}
          {groupName === 'caption' && showCaptionFormats && <div className="config-field-section config-caption-formats"><h3>{english ? 'Caption format' : '标签格式'}</h3><div className="caption-source-formats">
            {captionSources.map((source: any, index: number) => {
              const name = String(source.path || '').split(/[\\/]/).filter(Boolean).pop() || `${english ? 'Data source' : '数据源'} ${index + 1}`;
              const label = `${english ? 'Caption format' : '标签格式'} · ${name}`;
              return <div className="caption-source-format" key={`${index}-${source.path}`}>
                <span className="caption-source-name" title={source.path}>{name}</span>
                {/* The card names the dataset; the select keeps "Caption format" as its accessible name. */}
                <CaptionFormatSelect label={label} formats={family?.caption_formats} value={source.caption_ext || 'auto'} disabled={readOnly} onChange={caption_ext => onChange(setNestedValue(value, ['dataset', 'sources'], captionSources.map((item: any, itemIndex: number) => itemIndex === index ? {...item, caption_ext} : item)))}/>
              </div>;
            })}
          </div></div>}
          {groupName === 'caption' && hasCaptionOverrides && !showAdvanced && !editCaptionOverrides && <div className="caption-override-notice" role="status"><span>{english ? 'This configuration adds text to your existing captions.' : '当前配置会额外改写已有标签。'}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => setEditCaptionOverrides(true)}>{english ? 'Edit extra caption changes' : '编辑额外标签改写'}</button></div>}
          <ParameterFields group={groupName} fields={groupData.fields} english={english} renderToggleSection={renderToggleSection}/>
        </FieldGroup>
      ))}
      {sortedGroups.length === 0 && <div className="config-search-empty" role="status"><strong>{english ? 'No matching parameters.' : '没有匹配的参数。'}</strong><p>{search.trim() ? (english ? 'Try a parameter name, keyword or configuration path.' : '试试参数名称、关键词或配置字段路径。') : (english ? 'This section has no available parameters for the current configuration.' : '当前配置在此分区没有可用参数。')}</p>{search.trim() && onClearSearch && <button type="button" className="ui-btn" onClick={onClearSearch}>{english ? 'Return to parameter sections' : '返回参数分区'}</button>}</div>}
    </div>
  );
};
