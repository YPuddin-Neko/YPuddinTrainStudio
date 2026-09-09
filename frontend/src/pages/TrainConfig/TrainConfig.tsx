import React from 'react';
import { SchemaForm } from '../../schema/SchemaForm/SchemaForm';
import { apiClient } from '../../api/client';
import { Plan } from '../../api/types';

// 示例 Schema 兜底
const mockSchema = {
  properties: {
    model_name: {
      type: 'string',
      title: 'Model Name',
      default: 'Anima-Default',
      'x-ui': { group: 'model', order: 1 },
    },
    'adapter.algo': {
      type: 'string',
      title: 'Adapter Algorithm',
      enum: ['lora', 'lokr', 'loha', 'full'],
      default: 'lokr',
      'x-ui': { group: 'adapter', order: 2 },
    },
    'adapter.rank': {
      type: 'integer',
      title: 'Rank / Factor',
      default: 16,
      'x-ui': {
        group: 'adapter',
        order: 3,
        show_when: "adapter.algo != 'full'",
        min: 1,
        max: 1024,
      },
    },
    learning_rate: {
      type: 'number',
      title: 'Learning Rate',
      default: 0.0001,
      'x-ui': { group: 'optimizer', order: 4 },
    },
  },
};

export default function TrainConfig() {
  const [config, setConfig] = React.useState<Record<string, any>>({
    model_name: 'Anima-Default',
    'adapter.algo': 'lokr',
    'adapter.rank': 16,
    learning_rate: 0.0001,
  });

  const [plan, setPlan] = React.useState<Plan | null>(null);

  React.useEffect(() => {
    // 500ms 防抖更新 Plan
    const timer = setTimeout(() => {
      apiClient.post<Plan>('/plan', { config }).then(setPlan).catch(console.error);
    }, 500);

    return () => clearTimeout(timer);
  }, [config]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* 左侧 Schema 表单 */}
      <div className="lg:col-span-2 space-y-6">
        <h2 className="text-2xl font-bold">Training Configuration</h2>
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
          <SchemaForm schema={mockSchema} value={config} onChange={setConfig} />
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
