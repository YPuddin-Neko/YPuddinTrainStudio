import React from 'react';
import { SchemaForm } from '../../schema/SchemaForm/SchemaForm';
import { apiClient } from '../../api/client';
import { Plan, Preset } from '../../api/types';
import trainSchema from '../../schema/train-schema.json';

export default function TrainConfig() {
  const [config, setConfig] = React.useState<Record<string, any>>({
    model: { name: 'Anima-Default', dtype: 'bf16' },
    adapter: { algo: 'lokr', rank: 16 },
    optimizer: { lr: 0.0001 },
  });

  const [presets, setPresets] = React.useState<Preset[]>([]);
  const [plan, setPlan] = React.useState<Plan | null>(null);

  // 加载真实 schema + 预设计划
  React.useEffect(() => {
    apiClient.get<Preset[]>('/presets').then(setPresets).catch(console.error);
  }, []);

  React.useEffect(() => {
    // 500ms 防抖更新 Plan
    const timer = setTimeout(() => {
      apiClient.post<Plan>('/plan', { config }).then(setPlan).catch(console.error);
    }, 500);

    return () => clearTimeout(timer);
  }, [config]);

  const handleApplyPreset = (preset: Preset) => {
    setConfig({ ...config, ...preset.config });
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* 左侧 Schema 表单 */}
      <div className="lg:col-span-2 space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold">Training Configuration</h2>
          <div className="flex space-x-2">
            <select
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm dark:bg-slate-800 dark:border-slate-700"
              onChange={(e) => {
                const selected = presets.find((p) => p.name === e.target.value);
                if (selected) handleApplyPreset(selected);
              }}
            >
              <option value="">-- Load Preset --</option>
              {presets.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
            <button className="px-3 py-1.5 text-sm bg-slate-200 dark:bg-slate-700 rounded-md hover:bg-slate-300 dark:hover:bg-slate-600">
              Save as Preset
            </button>
          </div>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 max-h-[75vh] overflow-y-auto">
          <SchemaForm schema={trainSchema} value={config} onChange={setConfig} />
        </div>
      </div>

      {/* 右侧 Plan 面板 */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">Execution Plan</h2>
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 space-y-4">
          <div>
            <div className="text-xs text-slate-400">Total Steps</div>
            <div className="text-xl font-bold">{plan?.total_steps ?? '--'}</div>
          </div>
          <div>
            <div className="text-xs text-slate-400">VRAM Peak (Est.)</div>
            <div className="text-xl font-bold">
              {plan?.memory?.peak_mb_estimate ? `${Math.round(plan.memory.peak_mb_estimate / 1024)} GB` : '--'}
            </div>
          </div>
          <button className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg shadow-sm">
            Enqueue Job
          </button>
        </div>
      </div>
    </div>
  );
}
