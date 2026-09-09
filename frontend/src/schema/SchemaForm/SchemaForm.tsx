import React from 'react';
import { evaluateShowWhen } from '../showWhen';
import { useTranslation } from 'react-i18next';

interface SchemaField {
  type?: string;
  title?: string;
  description?: string;
  default?: any;
  enum?: any[];
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
  schema: {
    properties?: Record<string, SchemaField>;
  };
  value: Record<string, any>;
  onChange: (newValue: Record<string, any>) => void;
  showAdvanced?: boolean;
}

export const SchemaForm: React.FC<SchemaFormProps> = ({
  schema,
  value,
  onChange,
  showAdvanced = false,
}) => {
  const { t } = useTranslation();
  
  // 仅做简单渲染的引擎雏形
  if (!schema || !schema.properties) return <div className="text-slate-500">Invalid Schema</div>;

  const fields = Object.entries(schema.properties);

  return (
    <div className="space-y-4">
      {fields.map(([key, field]) => {
        const ui = field['x-ui'] || {};
        
        // 1. 检查高级选项是否显示
        if (ui.advanced && !showAdvanced) return null;
        
        // 2. 检查 show_when 表达式
        if (ui.show_when && !evaluateShowWhen(ui.show_when, value)) return null;

        const fieldValue = value[key] !== undefined ? value[key] : field.default;

        return (
          <div key={key} className="flex flex-col space-y-1 p-3 bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                {field.title || t(`fields.${key}`, key)}
                {ui.unit && <span className="ml-1 text-xs text-slate-500">({ui.unit})</span>}
              </label>
            </div>
            
            {ui.help && <p className="text-xs text-slate-500 dark:text-slate-400">{ui.help}</p>}
            
            {/* 简单控制渲染 */}
            <div className="mt-1">
              {field.enum ? (
                <select
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
                  value={fieldValue || ''}
                  onChange={(e) => onChange({ ...value, [key]: e.target.value })}
                >
                  {field.enum.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : field.type === 'boolean' ? (
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  checked={!!fieldValue}
                  onChange={(e) => onChange({ ...value, [key]: e.target.checked })}
                />
              ) : field.type === 'integer' || field.type === 'number' ? (
                <input
                  type="number"
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
                  value={fieldValue ?? ''}
                  min={ui.min}
                  max={ui.max}
                  step={ui.step}
                  onChange={(e) => {
                    const val = e.target.value === '' ? undefined : Number(e.target.value);
                    onChange({ ...value, [key]: val });
                  }}
                />
              ) : (
                <input
                  type="text"
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
                  value={fieldValue || ''}
                  onChange={(e) => onChange({ ...value, [key]: e.target.value })}
                />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
