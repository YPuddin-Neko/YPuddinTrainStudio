import React from 'react';
import { evaluateShowWhen } from '../showWhen';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronRight, Plus, Trash2, FolderOpen, ArrowUp, ArrowDown } from 'lucide-react';
import { apiClient } from '../../api/client';
import { FsListResponse } from '../../api/types';

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

// 1. Path 浏览弹窗组件
const PathPickerModal: React.FC<{
  isOpen: boolean;
  initialPath?: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}> = ({ isOpen, initialPath = '/', onSelect, onClose }) => {
  const [currentPath, setCurrentPath] = React.useState(initialPath);
  const [data, setData] = React.useState<FsListResponse | null>(null);

  React.useEffect(() => {
    if (isOpen) {
      apiClient.get<FsListResponse>('/fs/list', { params: { path: currentPath } })
        .then(setData)
        .catch(console.error);
    }
  }, [isOpen, currentPath]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-slate-800 rounded-xl max-w-lg w-full p-6 space-y-4 shadow-xl border border-slate-200 dark:border-slate-700">
        <div className="flex justify-between items-center border-b pb-2 dark:border-slate-700">
          <h3 className="font-semibold text-lg">Browse Server Path</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">✕</button>
        </div>
        <div className="text-xs font-mono bg-slate-100 dark:bg-slate-900 p-2 rounded truncate">
          Current: {data?.path || currentPath}
        </div>
        <div className="max-h-60 overflow-y-auto divide-y divide-slate-100 dark:divide-slate-700">
          {data?.parent && (
            <div
              onClick={() => setCurrentPath(data.parent!)}
              className="p-2 text-sm hover:bg-slate-50 dark:hover:bg-slate-700 cursor-pointer font-medium text-blue-500"
            >
              📁 .. (Parent Directory)
            </div>
          )}
          {data?.entries.map((entry) => (
            <div
              key={entry.name}
              onClick={() => {
                if (entry.is_dir) {
                  setCurrentPath(`${data.path === '/' ? '' : data.path}/${entry.name}`);
                } else {
                  onSelect(`${data.path === '/' ? '' : data.path}/${entry.name}`);
                  onClose();
                }
              }}
              className="p-2 text-sm hover:bg-slate-50 dark:hover:bg-slate-700 flex justify-between items-center cursor-pointer"
            >
              <span>{entry.is_dir ? '📁' : '📄'} {entry.name}</span>
              <span className="text-xs text-slate-400">{entry.is_dir ? 'dir' : `${entry.size} B`}</span>
            </div>
          ))}
        </div>
        <div className="flex justify-end space-x-2 pt-2 border-t dark:border-slate-700">
          <button onClick={onClose} className="px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">Cancel</button>
          <button
            onClick={() => {
              onSelect(currentPath);
              onClose();
            }}
            className="px-3 py-1.5 text-sm rounded bg-blue-600 text-white"
          >
            Select Current Dir
          </button>
        </div>
      </div>
    </div>
  );
};

// 2. Rules 专用组件 (AdapterRule 列表)
const RulesEditor: React.FC<{
  value: any[];
  onChange: (val: any[]) => void;
}> = ({ value = [], onChange }) => {
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
            placeholder="Match regex/glob"
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
        className="flex items-center space-x-1 px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded border border-slate-300 dark:border-slate-600"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>Add Rule</span>
      </button>
    </div>
  );
};

// 3. Key-Value 自由对象编辑器 (additionalProperties)
const KeyValueEditor: React.FC<{
  value: Record<string, any>;
  onChange: (val: Record<string, any>) => void;
}> = ({ value = {}, onChange }) => {
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
            placeholder="Key"
          />
          <input
            type="text"
            value={typeof v === 'object' ? JSON.stringify(v) : v ?? ''}
            onChange={(e) => {
              const num = Number(e.target.value);
              updateVal(k, isNaN(num) || e.target.value === '' ? e.target.value : num);
            }}
            className="flex-1 px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600 font-mono"
            placeholder="Value"
          />
          <button type="button" onClick={() => removeEntry(k)} className="text-red-500 hover:text-red-700 p-1">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
      <button
        type="button"
        onClick={addEntry}
        className="flex items-center space-x-1 px-2 py-1 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded border border-slate-300 dark:border-slate-600"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>Add Property</span>
      </button>
    </div>
  );
};

// 4. Data Sources 列表控件 (DatasetSourceConfig 列表)
const SourcesEditor: React.FC<{
  value: any[];
  onChange: (val: any[]) => void;
}> = ({ value = [], onChange }) => {
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
            <span className="font-semibold text-slate-700 dark:text-slate-300">Source #{idx + 1}</span>
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
              <span>Browse</span>
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            <div>
              <label className="text-[10px] text-slate-400">Repeats</label>
              <input
                type="number"
                value={src.repeats ?? 1}
                onChange={(e) => updateSource(idx, 'repeats', Number(e.target.value))}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400">Caption Ext</label>
              <input
                type="text"
                value={src.caption_ext || '.txt'}
                onChange={(e) => updateSource(idx, 'caption_ext', e.target.value)}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400">Prior Weight</label>
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
              <span className="text-[11px]">Regularization</span>
            </div>
          </div>
        </div>
      ))}
      <button
        type="button"
        onClick={addSource}
        className="flex items-center space-x-1 px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded border border-slate-300 dark:border-slate-600"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>Add Dataset Source</span>
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
            <span className="font-semibold text-slate-700 dark:text-slate-300">Prompt #{idx + 1}</span>
            <button type="button" onClick={() => removePrompt(idx)} className="text-red-500 hover:text-red-700">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
          <textarea
            rows={2}
            placeholder="Prompt text"
            value={p.prompt || ''}
            onChange={(e) => updatePrompt(idx, 'prompt', e.target.value)}
            className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <textarea
            rows={1}
            placeholder="Negative prompt (optional)"
            value={p.negative || ''}
            onChange={(e) => updatePrompt(idx, 'negative', e.target.value)}
            className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
          />
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="text-[10px] text-slate-400">Seed</label>
              <input
                type="number"
                value={p.seed ?? 42}
                onChange={(e) => updatePrompt(idx, 'seed', Number(e.target.value))}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400">Width</label>
              <input
                type="number"
                value={p.width ?? 1024}
                onChange={(e) => updatePrompt(idx, 'width', Number(e.target.value))}
                className="w-full px-2 py-1 border rounded dark:bg-slate-800 dark:border-slate-600"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400">Height</label>
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
        className="flex items-center space-x-1 px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded border border-slate-300 dark:border-slate-600"
      >
        <Plus className="w-3.5 h-3.5" />
        <span>Add Sample Prompt</span>
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

// 7. Path 控件 (输入框 + 浏览按钮)
const PathInput: React.FC<{
  value: string;
  onChange: (val: string) => void;
}> = ({ value = '', onChange }) => {
  const [modalOpen, setModalOpen] = React.useState(false);

  return (
    <div className="flex space-x-2">
      <input
        type="text"
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        className="flex-1 px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600 font-mono"
      />
      <button
        type="button"
        onClick={() => setModalOpen(true)}
        className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-md hover:bg-slate-300 dark:hover:bg-slate-600 text-sm flex items-center space-x-1"
      >
        <FolderOpen className="w-4 h-4" />
        <span>Browse</span>
      </button>
      <PathPickerModal
        isOpen={modalOpen}
        initialPath={value || '/'}
        onSelect={onChange}
        onClose={() => setModalOpen(false)}
      />
    </div>
  );
};

// 分组组件
const FieldGroup: React.FC<{
  title: string;
  children: React.ReactNode;
}> = ({ title, children }) => {
  const [isOpen, setIsOpen] = React.useState(true);
  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden bg-white dark:bg-slate-800">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
      >
        <span className="font-medium text-slate-700 dark:text-slate-200">{title}</span>
        {isOpen ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
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
      control = (
        <PathInput
          value={fieldValue || ''}
          onChange={(val) => onChange(setNestedValue(value, path, val))}
        />
      );
    } else if (prop.anyOf) {
      const stringConst = prop.anyOf.find((p) => p.const !== undefined);
      const isConstSelected = fieldValue === stringConst?.const;

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
        control = (
          <input
            type="number"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
            value={fieldValue ?? ''}
            min={ui.min}
            max={ui.max}
            step={ui.step}
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

    const label = (
      <div key={fullPathKey} data-testid={`field-${fullPathKey}`} className={`flex flex-col space-y-1 p-2 rounded ${errorItem ? 'bg-red-50 dark:bg-red-950/30 border border-red-300 dark:border-red-800' : ''}`}>
        <div className="flex justify-between items-center">
          <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
            {prop.title || t(`fields.${key}`, key)}
            {ui.unit && <span className="ml-1 text-xs text-slate-500">({ui.unit})</span>}
          </label>
        </div>
        {prop.description && <p className="text-xs text-slate-500 dark:text-slate-400">{prop.description}</p>}
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
        <FieldGroup key={groupName} title={t(`groups.${groupName}`, groupName)}>
          {groupData.fields}
        </FieldGroup>
      ))}
    </div>
  );
};
