import React from 'react';
import { evaluateShowWhen } from '../showWhen';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronRight, Plus, Trash2, ArrowUp, ArrowDown, FolderOpen } from 'lucide-react';
import { PathInput, PathPickerModal } from '../../components/PathBrowser';
import { apiClient } from '../../api/client';
import { FamilyInfo } from '../../api/types';

interface SchemaProperty {
  type?: string;
  title?: string;
  description?: string;
  default?: any;
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
          <select
            value={rule.algo || 'none'}
            onChange={(e) => updateRule(idx, 'algo', e.target.value)}
            className="px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          >
            <option value="none">none</option>
            <option value="lora">lora</option>
            <option value="lokr">lokr</option>
            <option value="loha">loha</option>
            <option value="full">full</option>
          </select>
          <input
            type="text"
            placeholder="Rank / full"
            value={rule.rank ?? ''}
            onChange={(e) => {
              const val = e.target.value === 'full' ? 'full' : Number(e.target.value) || 0;
              updateRule(idx, 'rank', val);
            }}
            className="w-16 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <input
            type="number"
            placeholder="Alpha"
            value={rule.alpha ?? ''}
            onChange={(e) => updateRule(idx, 'alpha', Number(e.target.value))}
            className="w-16 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <input
            type="number"
            placeholder="Factor"
            value={rule.factor ?? ''}
            onChange={(e) => updateRule(idx, 'factor', Number(e.target.value))}
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
  const { t } = useTranslation();
  const entries = Object.entries(value);

  const addEntry = () => {
    onChange({ ...value, [`key_${Date.now()}`]: 1.0 });
  };

  const updateKey = (oldKey: string, newKey: string) => {
    if (!newKey || oldKey === newKey) return;
    const newObj: Record<string, any> = {};
    for (const [k, v] of Object.entries(value)) {
      if (k === oldKey) {
        newObj[newKey] = v;
      } else {
        newObj[k] = v;
      }
    }
    onChange(newObj);
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
            value={k}
            onBlur={(e) => updateKey(k, e.target.value)}
            className="flex-1 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600 font-mono"
            placeholder={t('train.keyPlaceholder', '键')}
          />
          <input
            type="text"
            value={typeof v === 'object' ? JSON.stringify(v) : v ?? ''}
            onChange={(e) => {
              const num = Number(e.target.value);
              updateVal(k, isNaN(num) || e.target.value === '' ? e.target.value : num);
            }}
            className="flex-1 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600 font-mono"
            placeholder={t('train.valuePlaceholder', '值')}
          />
          <button type="button" onClick={() => removeEntry(k)} className="text-red-500 hover:text-red-700 p-1">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
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
              {t('train.sourceN', { n: idx + 1, defaultValue: '数据源 #{{n}}' })}
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
  onChange: (val: any[]) => void;
}> = ({ value = [], onChange }) => {
  const { t } = useTranslation();
  const addPrompt = () => {
    onChange([...value, { prompt: '', negative: '', seed: 42, width: 1024, height: 1024 }]);
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
              {t('train.promptN', { n: idx + 1, defaultValue: '提示词 #{{n}}' })}
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
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="text-[10px] text-slate-400">{t('job.seed')}</label>
              <input
                type="number"
                value={p.seed ?? 42}
                onChange={(e) => updatePrompt(idx, 'seed', Number(e.target.value))}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400">{t('train.width', '宽度')}</label>
              <input
                type="number"
                value={p.width ?? 1024}
                onChange={(e) => updatePrompt(idx, 'width', Number(e.target.value))}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400">{t('train.height', '高度')}</label>
              <input
                type="number"
                value={p.height ?? 1024}
                onChange={(e) => updatePrompt(idx, 'height', Number(e.target.value))}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
          </div>
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
}> = ({ value, kind, onChange }) => {
  const { t } = useTranslation();
  const [models, setModels] = React.useState<Array<{ id: string; path: string; kind: string; family: string }>>([]);

  React.useEffect(() => {
    if (!kind) return;
    apiClient
      .get<Array<{ id: string; path: string; kind: string; family: string }>>('/models')
      .then((list) => setModels(Array.isArray(list) ? list : []))
      .catch(() => setModels([]));
  }, [kind]);

  const matched = kind ? models.filter((m) => m.kind === kind) : [];

  return (
    <div className="space-y-1.5">
      <PathInput value={value} onChange={onChange} />
      {matched.length > 0 && (
        <select
          className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-xs dark:bg-slate-900 dark:border-slate-600 text-slate-500"
          value=""
          onChange={(e) => {
            if (e.target.value) onChange(e.target.value);
          }}
          data-testid="model-registry-select"
        >
          <option value="">{t('models.fromRegistry')}</option>
          {matched.map((m) => (
            <option key={m.id} value={m.path}>
              [{m.family}] {m.path}
            </option>
          ))}
        </select>
      )}
    </div>
  );
};

// 分组组件（header 右侧显示该组当前可见字段数）
const FieldGroup: React.FC<{
  title: string;
  count?: number;
  children: React.ReactNode;
}> = ({ title, count, children }) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = React.useState(true);
  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden bg-white dark:bg-slate-800">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
      >
        <span className="font-medium text-slate-700 dark:text-slate-200">{title}</span>
        <span className="flex items-center space-x-2">
          {typeof count === 'number' && (
            <span className="text-xs text-slate-400 font-mono" data-testid="group-count">
              {t('groups.fieldCount', { count, defaultValue: '{{count}} 项' })}
            </span>
          )}
          {isOpen ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
        </span>
      </button>
      {isOpen && <div className="p-4 space-y-4">{children}</div>}
    </div>
  );
};

export const SchemaForm: React.FC<SchemaFormProps> = ({
  schema,
  value,
  onChange,
  showAdvanced = false,
  errors = [],
  family,
}) => {
  const { t } = useTranslation();
  const groups: Record<string, { order: number; fields: React.ReactNode[] }> = {};

  const renderField = (key: string, prop: SchemaProperty, parentPath: string[] = []) => {
    const path = [...parentPath, key];
    const fullPathKey = path.join('.');
    const ui = prop['x-ui'] || {};

    if (ui.advanced && !showAdvanced) return null;
    if (ui.show_when) {
      try {
        if (!evaluateShowWhen(ui.show_when, value)) return null;
      } catch (err) {
        console.error(`[SchemaForm] Failed to evaluate show_when for ${fullPathKey}: "${ui.show_when}"`, err);
        // On evaluation error, default to showing the field
      }
    }

    const errorItem = errors.find((e) => e.loc === fullPathKey || e.loc?.endsWith(`.${key}`));
    const fieldValue = getNestedValue(value, path) !== undefined ? getNestedValue(value, path) : prop.default;
    const groupName = ui.group || (parentPath.length > 0 ? parentPath[0] : 'default');

    let control = null;

    // 1. 递归对象渲染
    if (prop.type === 'object' && prop.properties) {
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
          onChange={(val) => onChange(setNestedValue(value, path, val))}
        />
      );
    } else if (prop.anyOf) {
      const stringConst = prop.anyOf.find((p) => p.const !== undefined);
      const isConstSelected = fieldValue === stringConst?.const;

      // sampling.shift 的 anyOf[number, null] 也要带族默认占位
      let anyOfPlaceholder: string | undefined;
      if (family && fullPathKey === 'sampling.shift') {
        anyOfPlaceholder =
          family.sampling?.shift != null ? String(family.sampling.shift) : t('sampling.shiftAuto');
      }

      control = (
        <div className="flex space-x-2 items-center">
          <input
            type="number"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600 disabled:opacity-50"
            value={isConstSelected ? '' : fieldValue ?? ''}
            disabled={isConstSelected}
            min={ui.min}
            max={ui.max}
            step={ui.step}
            placeholder={anyOfPlaceholder}
            onChange={(e) => {
              const val = e.target.value === '' ? undefined : Number(e.target.value);
              onChange(setNestedValue(value, path, val));
            }}
          />
          {stringConst && (
            <label className="flex items-center space-x-1 text-sm whitespace-nowrap">
              <input
                type="checkbox"
                checked={isConstSelected}
                onChange={(e) => {
                  onChange(setNestedValue(value, path, e.target.checked ? stringConst.const : 16));
                }}
              />
              <span>{String(stringConst.const)}</span>
            </label>
          )}
        </div>
      );
    } else if (fullPathKey === 'adapter.preset' && family) {
      // 族内预设下拉：name — description（N 层）
      control = (
        <select
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
          value={fieldValue || family.default_preset || ''}
          onChange={(e) => onChange(setNestedValue(value, path, e.target.value))}
          data-testid="adapter-preset-select"
        >
          {(family.presets || []).map((p) => (
            <option key={p.name} value={p.name}>
              {p.name} — {p.description}（{t('preset.layers', { n: p.layers })}）
            </option>
          ))}
        </select>
      );
    } else if (fullPathKey === 'dataset.text_encoding' && family) {
      // 文本编码选项受族 text_modes 约束（krea2 无 online）
      control = (
        <div className="space-y-1">
          <select
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
            value={fieldValue || 'auto'}
            onChange={(e) => onChange(setNestedValue(value, path, e.target.value))}
            data-testid="text-encoding-select"
          >
            {(family.text_modes || []).map((m) => (
              <option key={m} value={m}>{t(`textMode.${m}`, m)}</option>
            ))}
          </select>
          {(family.text_modes || []).length === 2 && !family.text_modes.includes('online') && (
            <p className="text-[11px] text-slate-400">{t('textMode.autoOnly')}</p>
          )}
        </div>
      );
    } else if (prop.enum) {
      control = (
        <select
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
          value={fieldValue || ''}
          onChange={(e) => onChange(setNestedValue(value, path, e.target.value))}
        >
          {prop.enum.map((opt: any) => (
            <option key={opt} value={opt}>{String(opt)}</option>
          ))}
        </select>
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
      if (ui.control === 'slider' && ui.min !== undefined && ui.max !== undefined) {
        control = (
          <input
            type="range"
            className="w-full"
            min={ui.min}
            max={ui.max}
            step={ui.step || 1}
            value={fieldValue ?? ui.min}
            onChange={(e) => onChange(setNestedValue(value, path, Number(e.target.value)))}
          />
        );
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
            min={ui.min}
            max={ui.max}
            step={ui.step}
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

    const label = (
      <div key={fullPathKey} data-testid={`field-${fullPathKey}`} className={`flex flex-col space-y-1 p-2 rounded ${errorItem ? 'bg-red-50 dark:bg-red-950/30 border border-red-300 dark:border-red-800' : ''}`}>
        <div className="flex justify-between items-center">
          <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
            {weightMeta?.label || prop.title || t(`fields.${key}`, key)}
            {ui.unit && <span className="ml-1 text-xs text-slate-500">({ui.unit})</span>}
          </label>
        </div>
        {prop.description && <p className="text-xs text-slate-500 dark:text-slate-400">{prop.description}</p>}
        {weightMeta?.hint && (
          <p className="text-[11px] text-slate-400 dark:text-slate-500" data-testid={`weight-hint-${key}`}>
            {weightMeta.hint}
          </p>
        )}
        {errorItem && <p className="text-xs font-semibold text-red-600 dark:text-red-400">{errorItem.msg}</p>}
        <div className="mt-1">{control}</div>
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

  const groupOrder = schema['x-ui-groups'] || [];
  const sortedGroups = Object.entries(groups).sort(([keyA, groupA], [keyB, groupB]) => {
    const indexA = groupOrder.indexOf(keyA);
    const indexB = groupOrder.indexOf(keyB);
    if (indexA !== -1 && indexB !== -1) return indexA - indexB;
    if (indexA !== -1) return -1;
    if (indexB !== -1) return 1;
    return groupA.order - groupB.order;
  });

  return (
    <div className="space-y-6" data-testid="schema-form">
      {sortedGroups.map(([groupName, groupData]) => (
        <FieldGroup key={groupName} title={t(`groups.${groupName}`, groupName)} count={groupData.fields.length}>
          {groupData.fields}
        </FieldGroup>
      ))}
    </div>
  );
};
