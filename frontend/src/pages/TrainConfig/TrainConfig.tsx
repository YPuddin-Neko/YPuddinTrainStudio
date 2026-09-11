import React from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { SchemaForm, ValidationError } from '../../schema/SchemaForm/SchemaForm';
import { apiClient } from '../../api/client';
import { Job, Plan, Preset, ModelAsset } from '../../api/types';
import { useFamilies, familyByName } from '../../api/hooks/useFamilies';
import { formatBytesMB, formatParams } from '../../utils/format';
import { mergeConfig } from '../../utils/config';
import { formatApiError } from '../../utils/errors';
import { fillDefaultModels, changeModelFamily } from '../../utils/workspaceConfig';
import { useWorkspaceText } from '../../utils/workspaceText';
import { ProjectWorkflow } from '../../components/ProjectWorkflow';
import CommonTrainingFields from './CommonTrainingFields';
import { AlertCircle, CheckCircle, Info } from 'lucide-react';

export default function TrainConfig() {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const { id: projectId } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: families } = useFamilies();
  const [config, setConfig] = React.useState<Record<string, any>>({});
  const [schema, setSchema] = React.useState<any>(null);
  const [defaults, setDefaults] = React.useState<Record<string, any>>({});
  const [registeredModels, setRegisteredModels] = React.useState<ModelAsset[]>([]);
  const [loaded, setLoaded] = React.useState(false);
  const [reload, setReload] = React.useState(0);
  const [error, setError] = React.useState('');
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [presets, setPresets] = React.useState<Preset[]>([]);
  const [plan, setPlan] = React.useState<Plan | null>(null);
  const [validationErrors, setValidationErrors] = React.useState<ValidationError[]>([]);
  const [validating, setValidating] = React.useState(true);
  const [isEnqueuing, setIsEnqueuing] = React.useState(false);
  const [enqueueSuccess, setEnqueueSuccess] = React.useState(false);
  const [savedAt, setSavedAt] = React.useState<string | null>(null);
  const [jobName, setJobName] = React.useState('');
  const [priority, setPriority] = React.useState(0);
  const [scheduledAt, setScheduledAt] = React.useState('');
  const [presetName, setPresetName] = React.useState('');
  const [savingPreset, setSavingPreset] = React.useState(false);
  const [importText, setImportText] = React.useState('');
  const [importOpen, setImportOpen] = React.useState(false);
  const [importing, setImporting] = React.useState(false);
  const lastSavedRef = React.useRef('');
  const saveQueueRef = React.useRef<Promise<unknown>>(Promise.resolve());
  const draftRef = React.useRef({ projectId, config, loaded });
  React.useEffect(() => { draftRef.current = { projectId, config, loaded }; }, [projectId, config, loaded]);
  React.useEffect(() => () => {
    const draft = draftRef.current;
    if (!draft.projectId || !draft.loaded || JSON.stringify(draft.config) === lastSavedRef.current) return;
    // Route navigation must not discard changes still in the autosave debounce.
    saveQueueRef.current = saveQueueRef.current.catch(() => {}).then(() => apiClient.put(`/projects/${draft.projectId}/config`, draft.config));
    void saveQueueRef.current.catch(() => {});
  }, [projectId]);

  React.useEffect(() => {
    let active = true;
    setLoaded(false);
    setError('');
    setSavedAt(null);
    Promise.all([
      apiClient.get<any>('/schema/train', { silent: true }),
      apiClient.get<Record<string, any>>('/config/defaults', { silent: true }),
      projectId ? apiClient.get<Record<string, any>>(`/projects/${projectId}/config`, { silent: true }) : Promise.resolve({}),
      apiClient.get<Preset[]>('/presets', { silent: true }),
      apiClient.get<ModelAsset[]>('/models', { silent: true }),
    ]).then(([nextSchema, nextDefaults, draft, nextPresets, models]) => {
      if (!active) return;
      const next = fillDefaultModels(mergeConfig(nextDefaults, draft), models);
      setSchema(nextSchema);
      setDefaults(nextDefaults);
      setPresets(nextPresets);
      setRegisteredModels(models);
      lastSavedRef.current = JSON.stringify(next);
      setConfig(next);
      setLoaded(true);
    }).catch((err) => { if (active) setError(formatApiError(err)); });
    return () => { active = false; };
  }, [projectId, reload]);

  // Serialise writes so a slow older save cannot overwrite a newer draft.
  React.useEffect(() => {
    const encoded = JSON.stringify(config);
    if (!projectId || !loaded || encoded === lastSavedRef.current) return;
    let active = true;
    setSavedAt(null);
    const timer = setTimeout(() => {
      saveQueueRef.current = saveQueueRef.current.catch(() => {}).then(async () => {
        if (!active) return;
        await apiClient.put(`/projects/${projectId}/config`, config, { silent: true });
        if (active) {
          lastSavedRef.current = encoded;
          setSavedAt(new Date().toLocaleTimeString(undefined, { hour12: false }));
        }
      }).catch((err) => { if (active) setError(formatApiError(err)); });
    }, 1000);
    return () => { active = false; clearTimeout(timer); };
  }, [config, projectId, loaded]);

  // 族联动副作用：切换 model.family 后，adapter.preset 与 dataset.text_encoding 不合法时自动回退
  React.useEffect(() => {
    const family = familyByName(families, config?.model?.family);
    if (!family) return;
    setConfig((prev) => {
      let changed = false;
      const next = { ...prev, adapter: { ...(prev.adapter || {}) }, dataset: { ...(prev.dataset || {}) } };
      // adapter.preset 不在新族列表 → 回到该族 default_preset
      const presets = (family.presets || []).map((p) => p.name);
      if (next.adapter.preset != null && presets.length > 0 && !presets.includes(next.adapter.preset)) {
        next.adapter.preset = family.default_preset || presets[0];
        changed = true;
      }
      // dataset.text_encoding 不在族 text_modes → 回退 auto（krea2 无 online，后端会 400）
      if (
        next.dataset.text_encoding != null &&
        (family.text_modes || []).length > 0 &&
        !family.text_modes.includes(next.dataset.text_encoding)
      ) {
        next.dataset.text_encoding = 'auto';
        changed = true;
      }
      return changed ? next : prev;
    });
  }, [families, config?.model?.family]);

  React.useEffect(() => {
    if (!loaded) return;
    const controller = new AbortController();
    setValidating(true);
    setPlan(null);
    const timer = setTimeout(() => {
      Promise.all([
        apiClient.post<Plan>('/plan', { config }, { signal: controller.signal, silent: true }),
        apiClient.post<{ errors: ValidationError[] }>('/config/validate', { config }, { signal: controller.signal, silent: true }),
      ]).then(([nextPlan, validation]) => {
        if (controller.signal.aborted) return;
        setPlan(nextPlan);
        setValidationErrors([...validation.errors, ...(nextPlan.errors || [])].filter((item, index, all) => all.findIndex((x) => x.loc === item.loc && x.msg === item.msg) === index));
        setValidating(false);
      }).catch((err) => {
        if (!controller.signal.aborted) { setError(formatApiError(err)); setValidating(false); }
      });
    }, 500);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [config, loaded]);

  const handleApplyPreset = (preset: Preset) => {
    setError('');
    setConfig((prev) => fillDefaultModels(mergeConfig(prev, preset.config), registeredModels));
  };

  const handleConfigChange = (next: Record<string, any>) => {
    const family = familyByName(families, next.model?.family);
    setConfig(family && next.model?.family !== config.model?.family ? changeModelFamily(next, family, registeredModels) : next);
  };

  const handleSavePreset = async () => {
    if (!presetName.trim()) return;
    setSavingPreset(true);
    setError('');
    try {
      await apiClient.post('/presets', { name: presetName.trim(), config }, { silent: true });
      setPresets(await apiClient.get<Preset[]>('/presets'));
      setPresetName('');
    } catch (err: unknown) { setError(formatApiError(err)); }
    finally { setSavingPreset(false); }
  };

  const handleImport = async () => {
    setImporting(true);
    setError('');
    try {
      const next = await apiClient.post<Record<string, any>>('/config/import', { text: importText, format: 'toml' }, { silent: true });
      setConfig(fillDefaultModels(next, registeredModels));
      setImportOpen(false);
    } catch (err: unknown) { setError(formatApiError(err)); }
    finally { setImporting(false); }
  };

  const handleExport = async () => {
    setError('');
    try {
      const { text } = await apiClient.post<{ text: string }>('/config/export', { config, format: 'toml' }, { silent: true });
      const url = URL.createObjectURL(new Blob([text], { type: 'application/toml' }));
      const link = document.createElement('a');
      link.href = url; link.download = 'train-config.toml'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err: unknown) { setError(formatApiError(err)); }
  };

  const handleEnqueue = async () => {
    setIsEnqueuing(true);
    setError('');
    try {
      if (projectId) {
        await saveQueueRef.current.catch(() => {});
        await apiClient.put(`/projects/${projectId}/config`, config, { silent: true });
        lastSavedRef.current = JSON.stringify(config);
      }
      const job = await apiClient.post<Job>('/jobs', {
        type: 'train', name: jobName.trim() || `train-${Date.now()}`, project_id: projectId || null,
        config, priority, scheduled_at: scheduledAt ? new Date(scheduledAt).getTime() / 1000 : null,
      }, { silent: true });
      setEnqueueSuccess(true);
      if (job.id) navigate(`/jobs/${job.id}`);
    } catch (err: any) {
      setError(formatApiError(err));
      if (Array.isArray(err.details?.errors)) setValidationErrors(err.details.errors);
    } finally { setIsEnqueuing(false); }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-6">
      {projectId && <div className="lg:col-span-2 space-y-4"><Link to={`/projects/${projectId}`} className="text-sm text-blue-500">← {text('返回项目工作区', 'Back to project workspace')}</Link><ProjectWorkflow projectId={projectId} active="train" /></div>}
      {error && <div role="alert" className="lg:col-span-2 whitespace-pre-line break-words rounded bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">{error}
        {!loaded && <button className="ml-3 underline" onClick={() => setReload((v) => v + 1)}>{t('common.retry')}</button>}
      </div>}
      {/* 左侧 Schema 表单 */}
      <div className="min-w-0 space-y-6 order-2 lg:order-none">
        {loaded && <CommonTrainingFields config={config} onChange={handleConfigChange} />}
        {loaded && !(config.dataset?.sources?.length > 0) && <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-300">{text('还没有训练数据。请先上传图片或导入文件夹。', 'No training data configured. Upload images or import a folder first.')} <Link className="font-semibold underline" to={`/projects/${projectId}?step=data`}>{text('添加训练数据 →', 'Add training data →')}</Link></div>}
        <div className="flex flex-wrap justify-between items-center gap-4">
          <h2 className="text-2xl font-bold">{t('train.title')}</h2>
          <div className="flex items-center space-x-3">
            <label className="flex items-center space-x-2 text-sm text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={showAdvanced}
                onChange={(e) => setShowAdvanced(e.target.checked)}
                className="rounded text-blue-600"
              />
              <span>{t('train.advanced')}</span>
            </label>
            <select
              aria-label={t('train.loadPreset')} disabled={!loaded}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm dark:bg-slate-800 dark:border-slate-700"
              onChange={(e) => {
                const selected = presets.find((p) => p.name === e.target.value);
                if (selected) handleApplyPreset(selected);
              }}
            >
              <option value="">{t('train.loadPreset')}</option>
              {presets.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-sm">
          <input aria-label={t('train.presetName')} placeholder={t('train.presetName')} value={presetName} onChange={(e) => setPresetName(e.target.value)} className="rounded border px-3 py-2 dark:bg-slate-900 dark:border-slate-600" />
          <button disabled={!loaded || savingPreset || !presetName.trim()} onClick={handleSavePreset} className="rounded bg-slate-100 px-3 py-2 dark:bg-slate-700 disabled:opacity-50">{t('train.savePreset')}</button>
          <button disabled={!loaded} onClick={() => { if (window.confirm(t('train.resetConfirm'))) setConfig(structuredClone(defaults)); }} className="rounded bg-slate-100 px-3 py-2 dark:bg-slate-700">{t('train.resetDefaults')}</button>
          <button disabled={!loaded} onClick={() => setImportOpen((v) => !v)} className="rounded bg-slate-100 px-3 py-2 dark:bg-slate-700">{t('train.importToml')}</button>
          <button disabled={!loaded} onClick={handleExport} className="rounded bg-slate-100 px-3 py-2 dark:bg-slate-700">{t('train.exportToml')}</button>
        </div>
        {importOpen && <div className="space-y-2 rounded border p-3 dark:border-slate-600">
          <input type="file" accept=".toml,text/plain" aria-label={t('train.importFile')} onChange={(e) => {
            const file = e.target.files?.[0]; if (file) file.text().then(setImportText).catch((err) => setError(formatApiError(err)));
          }} />
          <textarea aria-label={t('train.importContent')} className="w-full h-48 rounded border p-2 font-mono text-xs dark:bg-slate-900 dark:border-slate-600" value={importText} onChange={(e) => setImportText(e.target.value)} />
          <button disabled={importing || !importText.trim()} onClick={handleImport} className="rounded bg-blue-600 px-3 py-2 text-sm text-white disabled:opacity-50">{t('train.applyImport')}</button>
        </div>}
        <div id="full-training-config" className="scroll-mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
          {!loaded && <p>{t('common.loading')}</p>}
          <SchemaForm
            schema={schema}
            value={config}
            onChange={handleConfigChange}
            showAdvanced={showAdvanced}
            errors={validationErrors}
            family={familyByName(families, config?.model?.family)}
            families={families}
          />
        </div>

        {savedAt && (
          <div className="flex items-center justify-end space-x-1.5 text-xs text-slate-400" data-testid="draft-saved">
            <CheckCircle className="w-3.5 h-3.5 text-green-500" />
            <span>{t('train.draftSaved', { time: savedAt, defaultValue: '草稿已自动保存 {time}' })}</span>
          </div>
        )}
      </div>

      {/* 右侧 Plan 面板（吸顶） */}
      <div id="training-launch" className="min-w-0 space-y-3 order-1 lg:order-none lg:sticky lg:top-6 lg:max-h-[calc(100vh-120px)] lg:overflow-y-auto self-start scroll-mt-6">
        <h2 className="text-lg font-semibold">{text('启动与执行计划', 'Start and execution plan')}</h2>
        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700 space-y-4">
          <div role="status" className={`rounded-lg p-3 text-sm ${!validating && plan?.ok && validationErrors.length === 0 ? 'bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-900 dark:text-slate-300'}`}>{!loaded ? text('正在加载项目配置…', 'Loading project configuration…') : validating ? text('正在检查数据、模型与训练参数…', 'Checking data, model and training parameters…') : plan?.ok && validationErrors.length === 0 ? text('检查通过，可以启动训练', 'Checks passed. Ready to start training.') : text('请先解决下方问题，训练按钮会自动启用。', 'Resolve the issues below to enable training.')}</div>
          <button
            onClick={handleEnqueue}
            disabled={!loaded || validating || plan?.ok !== true || isEnqueuing || validationErrors.length > 0}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded-lg shadow-sm transition-colors flex items-center justify-center space-x-2"
          >
            {enqueueSuccess ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-300" />
                <span>{t('train.enqueued')}</span>
              </>
            ) : (
              <span>{isEnqueuing ? t('train.enqueuing') : text('开始训练', 'Start training')}</span>
            )}
          </button>
          <div className="space-y-3 text-sm">
            <label className="block">{t('train.jobName')}<input aria-label={t('train.jobName')} value={jobName} onChange={(e) => setJobName(e.target.value)} className="mt-1 w-full rounded border px-2 py-1.5 dark:bg-slate-900 dark:border-slate-600" /></label>
            <label className="block">{t('queue.priority')}<input aria-label={t('queue.priority')} type="number" step="1" value={priority} onChange={(e) => setPriority(Number(e.target.value))} className="mt-1 w-full rounded border px-2 py-1.5 dark:bg-slate-900 dark:border-slate-600" /></label>
            <label className="block">{t('train.scheduledAt')}<input aria-label={t('train.scheduledAt')} type="datetime-local" value={scheduledAt} onChange={(e) => setScheduledAt(e.target.value)} className="mt-1 w-full rounded border px-2 py-1.5 dark:bg-slate-900 dark:border-slate-600" /></label>
          </div>
          {validationErrors.length > 0 && <div role="alert" className="text-sm text-red-600 space-y-1">{validationErrors.map((item, index) => <p key={index}>{item.loc}: {item.msg}</p>)}</div>}
          {projectId && <div className="flex flex-wrap gap-3 text-xs"><Link className="text-blue-500 underline" to={`/projects/${projectId}?step=data`}>{text('补充训练数据', 'Add training data')}</Link><Link className="text-blue-500 underline" to={`/projects/${projectId}?step=models`}>{text('准备或更换模型', 'Prepare or change model')}</Link></div>}
          <div className="grid grid-cols-2 gap-4 border-b pb-4 dark:border-slate-700">
            <div>
              <div className="text-xs text-slate-400">{t('train.totalSteps')}</div>
              <div className="text-xl font-bold font-mono">{plan?.total_steps ?? '--'}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">{t('train.stepsPerEpoch')}</div>
              <div className="text-xl font-bold font-mono">{plan?.steps_per_epoch ?? '--'}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">{t('train.epochs', '训练轮数')}</div>
              <div className="text-xl font-bold font-mono">{plan?.epochs ?? '--'}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">{t('train.trainableParams')}</div>
              <div className="text-sm font-semibold font-mono mt-1">{formatParams(plan?.params?.trainable)}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">{t('train.vramPeak')}</div>
              <div className="text-sm font-semibold font-mono mt-1">{formatBytesMB(plan?.memory?.peak_mb_estimate)}</div>
            </div>
          </div>

          {/* 分桶小表（items 优先） */}
          {plan?.buckets && plan.buckets.length > 0 && (
            <details className="border-b pb-4 dark:border-slate-700" data-testid="plan-buckets">
              <summary className="cursor-pointer text-xs font-semibold text-slate-500 dark:text-slate-400 mb-2">
                {t('train.bucketsTitle', '分桶分布')}
              </summary>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400">
                    <th className="text-left font-medium pb-1">{t('train.bucketSize', '分辨率 (W×H)')}</th>
                    <th className="text-right font-medium pb-1">{t('train.bucketItems', '样本数')}</th>
                    <th className="text-right font-medium pb-1">{t('train.batches', '批次数')}</th>
                  </tr>
                </thead>
                <tbody>
                  {plan.buckets.map((b, idx) => {
                    const itemCount = b.items ?? (b as { count?: number }).count;
                    return (
                      <tr key={idx} className="border-t border-slate-100 dark:border-slate-700/60">
                        <td className="py-1 font-mono">{b.w}×{b.h}</td>
                        <td className="py-1 text-right font-mono">{itemCount ?? '--'}</td>
                        <td className="py-1 text-right font-mono">{b.batches ?? '--'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </details>
          )}

          {/* Warnings & Suggestions */}
          {plan?.warnings && plan.warnings.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-semibold text-amber-500 flex items-center space-x-1">
                <AlertCircle className="w-3.5 h-3.5" />
                <span>{t('train.warningsTitle')}</span>
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


        </div>
      </div>
    </div>
  );
}
