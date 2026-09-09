import React from 'react';
import { SchemaForm, ValidationError } from '../../schema/SchemaForm/SchemaForm';
import { apiClient } from '../../api/client';
import { Plan, Preset } from '../../api/types';
import trainSchema from '../../schema/train-schema.json';
import { AlertCircle, CheckCircle, Info } from 'lucide-react';

export default function TrainConfig() {
  const [config, setConfig] = React.useState<Record<string, any>>({
    model: { family: 'anima', dit_path: '', text_encoder_path: '', vae_path: '', dtype: 'bf16' },
    adapter: { algo: 'lokr', rank: 16, alpha: 16, factor: -1, rules: [] },
    dataset: { batch_size: 4, sources: [] },
    optimizer: { type: 'adamw8bit', lr: 0.0001, betas: [0.9, 0.999] },
    sampling: { enabled: true, prompts: [] },
  });

  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [presets, setPresets] = React.useState<Preset[]>([]);
  const [plan, setPlan] = React.useState<Plan | null>(null);
  const [validationErrors, setValidationErrors] = React.useState<ValidationError[]>([]);
  const [isEnqueuing, setIsEnqueuing] = React.useState(false);
  const [enqueueSuccess, setEnqueueSuccess] = React.useState(false);

  React.useEffect(() => {
    apiClient.get<Preset[]>('/presets').then(setPresets).catch(console.error);
  }, []);

  React.useEffect(() => {
    // 500ms 防抖更新 Plan & 触发 Validate
    const timer = setTimeout(() => {
      apiClient.post<Plan>('/plan', { config })
        .then(setPlan)
        .catch(console.error);

      apiClient.post<{ ok: boolean; errors: ValidationError[]; warnings: any[] }>('/config/validate', config)
        .then((res) => {
          setValidationErrors(res.errors || []);
        })
        .catch(console.error);
    }, 500);

    return () => clearTimeout(timer);
  }, [config]);

  const handleApplyPreset = (preset: Preset) => {
    setConfig({ ...config, ...preset.config });
  };

  const handleEnqueue = () => {
    setIsEnqueuing(true);
    apiClient.post('/jobs', {
      type: 'train',
      name: `train-${Date.now()}`,
      config,
    })
      .then(() => {
        setEnqueueSuccess(true);
        setTimeout(() => setEnqueueSuccess(false), 3000);
      })
      .catch(console.error)
      .finally(() => setIsEnqueuing(false));
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* 左侧 Schema 表单 */}
      <div className="lg:col-span-2 space-y-6">
        <div className="flex flex-wrap justify-between items-center gap-4">
          <h2 className="text-2xl font-bold">Training Configuration</h2>
          <div className="flex items-center space-x-3">
            <label className="flex items-center space-x-2 text-sm text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={showAdvanced}
                onChange={(e) => setShowAdvanced(e.target.checked)}
                className="rounded text-blue-600"
              />
              <span>Advanced Options</span>
            </label>
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
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 max-h-[75vh] overflow-y-auto">
          <SchemaForm
            schema={trainSchema}
            value={config}
            onChange={setConfig}
            showAdvanced={showAdvanced}
            errors={validationErrors}
          />
        </div>
      </div>

      {/* 右侧 Plan 面板 */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">Execution Plan</h2>
        <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 space-y-5">
          <div className="grid grid-cols-2 gap-4 border-b pb-4 dark:border-slate-700">
            <div>
              <div className="text-xs text-slate-400">Total Steps</div>
              <div className="text-xl font-bold">{plan?.total_steps ?? '--'}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Steps / Epoch</div>
              <div className="text-xl font-bold">{plan?.steps_per_epoch ?? '--'}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Trainable Params</div>
              <div className="text-sm font-semibold mt-1">
                {plan?.params?.trainable ? `${(plan.params.trainable / 1e6).toFixed(2)} M` : '--'}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400">VRAM Peak (Est.)</div>
              <div className="text-sm font-semibold mt-1">
                {plan?.memory?.peak_mb_estimate ? `${Math.round(plan.memory.peak_mb_estimate / 1024)} GB` : '--'}
              </div>
            </div>
          </div>

          {/* Warnings & Suggestions */}
          {plan?.warnings && plan.warnings.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-semibold text-amber-500 flex items-center space-x-1">
                <AlertCircle className="w-3.5 h-3.5" />
                <span>Warnings & Suggestions</span>
              </div>
              {plan.warnings.map((w, idx) => (
                <div key={idx} className="text-xs p-2 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded text-amber-700 dark:text-amber-300">
                  {w.msg}
                </div>
              ))}
            </div>
          )}

          {plan?.memory?.suggestions && plan.memory.suggestions.length > 0 && (
            <div className="space-y-1 text-xs text-slate-500">
              {plan.memory.suggestions.map((sug, idx) => (
                <div key={idx} className="flex items-start space-x-1">
                  <Info className="w-3.5 h-3.5 flex-shrink-0 text-blue-500 mt-0.5" />
                  <span>{sug}</span>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={handleEnqueue}
            disabled={isEnqueuing || validationErrors.length > 0}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded-lg shadow-sm transition-colors flex items-center justify-center space-x-2"
          >
            {enqueueSuccess ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-300" />
                <span>Enqueued Successfully!</span>
              </>
            ) : (
              <span>{isEnqueuing ? 'Enqueuing...' : 'Enqueue Training Job'}</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
