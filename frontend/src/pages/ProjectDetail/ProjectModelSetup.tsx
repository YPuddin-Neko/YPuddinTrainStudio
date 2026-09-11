import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { CheckCircle2, Loader2, Download } from 'lucide-react';
import { projectUrl, versionConfigUrl } from '../../utils/projectVersions';
import { apiClient } from '../../api/client';
import type { ModelAsset } from '../../api/types';
import { useFamilies } from '../../api/hooks/useFamilies';
import { PathInput } from '../../components/PathBrowser';
import { useWorkspaceText } from '../../utils/workspaceText';
import { fillDefaultModels, changeModelFamily, MODEL_PATH_FIELDS } from '../../utils/workspaceConfig';
import { formatApiError } from '../../utils/errors';

export default function ProjectModelSetup({ projectId, versionId, config, onSaved }: {
  projectId: string; versionId?: string; config: Record<string, any>; onSaved: (config: Record<string, any>) => void;
}) {
  const text = useWorkspaceText();
  const navigate = useNavigate();
  const { data: families = [] } = useFamilies();
  const [assets, setAssets] = React.useState<ModelAsset[]>([]);
  const [draft, setDraft] = React.useState(config);
  const [loading, setLoading] = React.useState(true);
  const [saving, setSaving] = React.useState(false);
  const [saved, setSaved] = React.useState(false);
  const [error, setError] = React.useState('');
  React.useEffect(() => {
    let active = true;
    apiClient.get<ModelAsset[]>('/models', { silent: true }).then((models) => {
      if (active) { setAssets(models); setDraft((current) => fillDefaultModels(current, models)); setLoading(false); }
    }).catch((error) => { if (active) { setError(formatApiError(error)); setLoading(false); } });
    return () => { active = false; };
  }, [projectId, versionId]);
  const family = families.find((item) => item.name === draft.model?.family);
  const inputClass = 'w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600';
  const fields = family?.weights.length ? family.weights : [
    { field: 'dit_path', label: text('底模 / DiT 主干', 'Base model / DiT'), hint: '' },
    { field: 'text_encoder_path', label: text('文本编码器', 'Text encoder'), hint: '' },
    { field: 'vae_path', label: 'VAE', hint: '' },
    { field: 'tokenizer_path', label: text('分词器（可选）', 'Tokenizer (optional)'), hint: '' },
  ];
  const updatePath = (field: string, value: string) => {
    setDraft((previous) => ({ ...previous, model: { ...previous.model, [field]: value || null } })); setSaved(false);
  };
  const save = async (continueToTraining = false) => {
    setSaving(true); setError('');
    try {
      // Keep recently imported sources and edits outside this model section.
      const latest = await apiClient.get<Record<string, any>>(versionConfigUrl(projectId, versionId), { silent: true });
      const changedFamily = latest.model?.family !== draft.model?.family;
      const next = { ...latest, model: draft.model,
        ...(changedFamily ? { adapter: { ...latest.adapter, preset: family?.default_preset ?? draft.adapter?.preset }, dataset: { ...latest.dataset, text_encoding: 'auto' } } : {}),
      };
      await apiClient.put(versionConfigUrl(projectId, versionId), next, { silent: true });
      onSaved(next); setSaved(true);
      if (continueToTraining) navigate(projectUrl(projectId, versionId, 'train'));
    } catch (error) { setError(formatApiError(error)); }
    finally { setSaving(false); }
  };
  return <div className="space-y-5" data-testid="project-model-setup">
    <div className="flex flex-wrap items-start justify-between gap-3"><div><h3 className="text-lg font-semibold">{text('选择训练使用的模型', 'Choose the model to train')}</h3><p className="mt-1 text-sm text-slate-500">{text('先准备底模、文本编码器与 VAE。可以从模型库选择，也可以填写训练机上的已有路径。', 'Prepare the base model, text encoder and VAE. Select registered assets or enter existing paths on the training machine.')}</p></div>
      <Link to={`/models?project=${projectId}&family=${draft.model?.family || 'anima'}`} className="inline-flex items-center gap-2 rounded-lg border border-blue-400 px-4 py-2 text-sm text-blue-600 dark:text-blue-300"><Download className="h-4 w-4" />{text('下载 / 管理模型', 'Download / manage models')}</Link>
    </div>
    {error && <div role="alert" className="whitespace-pre-line rounded bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">{error}</div>}
    <div className="rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-700 dark:bg-slate-800 space-y-5">
      <div className="grid gap-4 sm:grid-cols-2"><label className="block text-sm font-medium">{text('模型系列', 'Model family')}<select aria-label={text('模型系列', 'Model family')} value={draft.model?.family || 'anima'} onChange={(e) => {
        const selected = families.find((item) => item.name === e.target.value);
        if (selected) { setDraft(changeModelFamily(draft, selected, assets)); setSaved(false); }
      }} className={`${inputClass} mt-1`}>
        {families.length === 0 && <option value={draft.model?.family || 'anima'}>{draft.model?.family || 'Anima'}</option>}
        {families.map((item) => <option key={item.name} value={item.name}>{item.name === 'toy' ? text('Toy（仅用于流程测试）', 'Toy (pipeline testing only)') : item.label}</option>)}
      </select></label>
      <label className="block text-sm font-medium">{text('计算精度', 'Compute precision')}<select value={draft.model?.dtype || 'bf16'} onChange={(e) => { setDraft({ ...draft, model: { ...draft.model, dtype: e.target.value } }); setSaved(false); }} className={`${inputClass} mt-1`}><option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp32">FP32</option></select></label></div>
      {draft.model?.family === 'toy' ? <p className="rounded bg-amber-50 p-3 text-sm text-amber-800 dark:bg-amber-950 dark:text-amber-300">{text('Toy 是流程测试用的小模型，不能用来训练实际图像 LoRA。正式训练请选择 Anima 或 Krea 2。', 'Toy is a small pipeline test model and cannot train a production image LoRA. Choose Anima or Krea 2 for actual training.')}</p> : fields.map((field) => {
        const kind = MODEL_PATH_FIELDS[field.field as keyof typeof MODEL_PATH_FIELDS];
        const matched = assets.filter((asset) => asset.family === draft.model?.family && asset.kind === kind && asset.exists);
        return <div key={field.field} className="space-y-2" role="group" aria-label={field.label}>
          <div className="flex items-center justify-between gap-2"><label className="text-sm font-medium">{field.label}</label>{draft.model?.[field.field] && <span className="text-xs text-slate-500">{text('已填写路径', 'Path configured')}</span>}</div>
          <PathInput ariaLabel={field.label} value={draft.model?.[field.field] || ''} onChange={(path) => updatePath(field.field, path)} placeholder={text('训练机上的文件或目录路径', 'File or directory path on the training machine')} />
          <select aria-label={`${text('从模型库选择', 'Choose registered model')} ${field.label}`} className={inputClass} value="" onChange={(e) => { if (e.target.value) updatePath(field.field, e.target.value); }}>
            <option value="">{matched.length ? text('从已安装的模型中选择…', 'Choose an installed model…') : text('模型库中暂无对应组件，请先下载或注册', 'No matching component registered — download or register one')}</option>
            {matched.map((asset) => <option key={asset.id} value={asset.path}>{asset.is_default ? '★ ' : ''}{asset.path}</option>)}
          </select>
          {field.hint && <p className="text-xs text-slate-500">{field.hint}</p>}
        </div>;
      })}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4 dark:border-slate-700"><button type="button" onClick={() => void save()} disabled={loading || saving} className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-50">{saving && <Loader2 className="h-4 w-4 animate-spin" />}{saving ? text('正在保存…', 'Saving…') : text('保存模型选择', 'Save model selection')}</button>
        {saved && <p role="status" className="flex items-center gap-2 text-sm text-green-600"><CheckCircle2 className="h-4 w-4" />{text('模型配置已保存，参数页将检查文件与训练条件。', 'Model configuration saved. The training page will validate files and requirements.')}</p>}
      </div>
    </div>
    <div className="flex justify-end"><button disabled={loading || saving} onClick={() => void save(true)} className="rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-50">{text('保存并进入训练参数', 'Save and continue to training')}</button></div>
  </div>;
}
