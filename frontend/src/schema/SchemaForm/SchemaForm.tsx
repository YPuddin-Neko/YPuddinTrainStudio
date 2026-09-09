import React from 'react';
import { evaluateShowWhen } from '../showWhen';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronRight } from 'lucide-react';

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

interface SchemaFormProps {
  schema: any;
  value: Record<string, any>;
  onChange: (newValue: Record<string, any>) => void;
  showAdvanced?: boolean;
}

// 帮助函数: 解析 $ref
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

// 获取嵌套值
const getNestedValue = (obj: any, path: string[]) => {
  let current = obj;
  for (const key of path) {
    if (current === undefined || current === null) return undefined;
    current = current[key];
  }
  return current;
};

// 更新嵌套值
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

// 分组渲染器
const FieldGroup: React.FC<{
  title: string;
  children: React.ReactNode;
  order: number;
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
}) => {
  const { t } = useTranslation();

  const groups: Record<string, { order: number; fields: React.ReactNode[] }> = {};

  const renderField = (key: string, prop: SchemaProperty, parentPath: string[] = []) => {
    const path = [...parentPath, key];
    const fullPathKey = path.join('.');
    const ui = prop['x-ui'] || {};

    if (ui.advanced && !showAdvanced) return null;
    if (ui.show_when && !evaluateShowWhen(ui.show_when, value)) return null;

    const fieldValue = getNestedValue(value, path) !== undefined ? getNestedValue(value, path) : prop.default;

    let control = null;
    const groupName = ui.group || 'default';

    // 简单子对象递归渲染
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
    } else if (prop.anyOf) {
      // anyOf 处理，比如 rank: integer | "full"
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
                  onChange(setNestedValue(value, path, e.target.checked ? stringConst.const : 1));
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
      // 简单的数组文本框编辑
      control = (
        <textarea
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600 font-mono"
          rows={3}
          value={JSON.stringify(fieldValue || [])}
          onChange={(e) => {
            try {
              const val = JSON.parse(e.target.value);
              onChange(setNestedValue(value, path, val));
            } catch {}
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
      <div key={fullPathKey} className="flex flex-col space-y-1">
        <div className="flex justify-between items-center">
          <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
            {prop.title || t(`fields.${key}`, key)}
            {ui.unit && <span className="ml-1 text-xs text-slate-500">({ui.unit})</span>}
          </label>
        </div>
        {prop.description && <p className="text-xs text-slate-500 dark:text-slate-400">{prop.description}</p>}
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
      renderField(key, prop as SchemaProperty, []);
    });
  }

  // 按 x-ui-groups 或 order 排序分组
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
    <div className="space-y-6">
      {sortedGroups.map(([groupName, groupData]) => (
        <FieldGroup key={groupName} title={t(`groups.${groupName}`, groupName)} order={groupData.order}>
          {groupData.fields}
        </FieldGroup>
      ))}
    </div>
  );
};
