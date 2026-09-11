import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { CheckCircle2, Loader2, Download, Save, ArrowRight } from 'lucide-react';
import { projectUrl, versionConfigUrl } from '../../utils/projectVersions';
import { apiClient } from '../../api/client';
import type { ModelAsset } from '../../api/types';
import { useFamilies } from '../../api/hooks/useFamilies';
import { PathInput } from '../../components/PathBrowser';
import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';
import { fillDefaultModels, changeModelFamily, MODEL_PATH_FIELDS } from '../../utils/workspaceConfig';
import { formatApiError } from '../../utils/errors';
import './project-model-setup.css';
import '../Models/models.css';

export default function ProjectModelSetup({ projectId, versionId, config, onSaved, registerSave }: {
  projectId: string; versionId?: string; config: Record<string, any>; onSaved: (config: Record<string, any>) => void;
  registerSave?: (save: (() => Promise<void>) | null) => void;
}) {
  const text = useWorkspaceText();
  const navigate = useNavigate();
  const location = useLocation();
  const { data: families = [] } = useFamilies();
  const draftKey = `model-draft:${projectId}:${versionId || 'legacy'}`;
  const [assets, setAssets] = React.useState<ModelAsset[]>([]);
  const [draft, setDraft] = React.useState(() => {
    try { const saved = sessionStorage.getItem(draftKey); return saved ? { ...config, model: JSON.parse(saved) } : config; }
    catch { return config; }
  });
  const [loading, setLoading] = React.useState(true);
  const [saving, setSaving] = React.useState(false);
  const [saved, setSaved] = React.useState(false);
  const [error, setError] = React.useState('');
  const [savedModel, setSavedModel] = React.useState(JSON.stringify(config.model));
  const pending = React.useRef<Promise<void> | null>(null);
  const dirty = JSON.stringify(draft.model) !== savedModel;
  const family = families.find(item => item.name === draft.model?.family);
  React.useEffect(() => {
    let active = true; let initial = true;
    const refresh = () => apiClient.get<ModelAsset[]>('/models', { silent: true }).then(models => {
      if (!active) return;
      setAssets(models); setLoading(false);
      if (initial) { initial = false; setDraft(current => fillDefaultModels(current, models)); }
    }).catch(error => { if (active) { setError(formatApiError(error)); setLoading(false); } });
    void refresh(); window.addEventListener('studio-models-changed', refresh); window.addEventListener('focus', refresh);
    return () => { active = false; window.removeEventListener('studio-models-changed', refresh); window.removeEventListener('focus', refresh); };
  }, [projectId, versionId]);
  React.useEffect(() => {
    try { if (dirty) sessionStorage.setItem(draftKey, JSON.stringify(draft.model)); else sessionStorage.removeItem(draftKey); } catch { /* Storage can be disabled. Navigation still waits for server save. */ }
  }, [draftKey, draft.model, dirty]);
  const saveRef = React.useRef<() => Promise<void>>(async () => {});
  const save = (): Promise<void> => {
    if (pending.current) return pending.current;
    if (!dirty) return Promise.resolve();
    const operation = async () => {
      setSaving(true); setError('');
      try {
        const latest = await apiClient.get<Record<string, any>>(versionConfigUrl(projectId, versionId), { silent: true });
        const changedFamily = latest.model?.family !== draft.model?.family;
        const next = { ...latest, model: draft.model,
          ...(changedFamily ? { adapter: { ...latest.adapter, preset: family?.default_preset ?? draft.adapter?.preset }, dataset: { ...latest.dataset, text_encoding: 'auto' } } : {}),
        };
        await apiClient.put(versionConfigUrl(projectId, versionId), next, { silent: true });
        setSavedModel(JSON.stringify(next.model)); setSaved(true); onSaved(next);
        try { sessionStorage.removeItem(draftKey); } catch { /* optional draft cache */ }
      } catch (error) { setError(`${text('模型选择未能保存，修改已保留。', 'Model selection could not be saved. Your changes are retained.')}\n${formatApiError(error)}`); throw error; }
      finally { setSaving(false); pending.current = null; }
    };
    pending.current = operation(); return pending.current;
  };
  saveRef.current = save;
  React.useEffect(() => { registerSave?.(() => saveRef.current()); return () => registerSave?.(null); }, [registerSave]);
  React.useEffect(() => {
    const leave = (event: MouseEvent) => {
      if (!dirty || location.pathname.startsWith('/settings') || event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      const anchor = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href]') : null;
      if (!anchor || anchor.hasAttribute('download') || (anchor.target && anchor.target !== '_self')) return;
      const target = new URL(anchor.href, window.location.href);
      if (target.origin !== window.location.origin || target.pathname.startsWith('/settings')) return;
      event.preventDefault(); event.stopPropagation();
      void saveRef.current().then(() => navigate(`${target.pathname}${target.search}${target.hash}`)).catch(() => {});
    };
    const beforeUnload = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = ''; } };
    document.addEventListener('click', leave, true); window.addEventListener('beforeunload', beforeUnload);
    return () => { document.removeEventListener('click', leave, true); window.removeEventListener('beforeunload', beforeUnload); };
  }, [dirty, location.pathname, navigate]);
  const fields = family?.weights.length ? family.weights : [
    { field: 'dit_path', label: text('主模型 / DiT', 'Base model / DiT'), hint: '' },
    { field: 'text_encoder_path', label: text('文本编码器', 'Text encoder'), hint: '' },
    { field: 'vae_path', label: 'VAE', hint: '' },
    { field: 'tokenizer_path', label: text('分词器（可选）', 'Tokenizer (optional)'), hint: '' },
  ];
  const updatePath = (field: string, value: string) => {
    setDraft(previous => ({ ...previous, model: { ...previous.model, [field]: value || null } })); setSaved(false);
  };
  const manageLink = `/settings/environment?tab=models&family=${draft.model?.family || 'anima'}`;
  return <section className="project-model-setup" data-testid="project-model-setup">
    <div className="project-model-toolbar"><div><span role="status">{dirty ? text('有未保存的修改', 'Unsaved changes') : saved ? text('已保存', 'Saved') : text('选择本版本使用的模型', 'Models used by this version')}</span></div><div className="project-model-actions">
      <Link to={manageLink} state={{ backgroundLocation: location }} className="model-button"><Download size={14}/>{text('下载 / 管理模型', 'Download / manage models')}</Link>
      <button className="model-button" disabled={loading || saving || !dirty} onClick={() => void save().catch(() => {})}>{saving ? <Loader2 size={14} className="animate-spin"/> : <Save size={14}/>} {text('保存选择', 'Save selection')}</button>
      <button className="model-button model-button-primary" disabled={loading || saving} onClick={() => void save().then(() => navigate(projectUrl(projectId, versionId, 'train'))).catch(() => {})}>{text('训练参数', 'Training parameters')}<ArrowRight size={14}/></button>
    </div></div>
    {error && <div role="alert" className="settings-alert">{error}</div>}
    <fieldset disabled={loading || saving}>
      <div className="project-model-family"><label>{text('模型系列', 'Model family')}<StudioSelect aria-label={text('模型系列', 'Model family')} value={draft.model?.family || 'anima'} options={(families.length ? families : [{ name: draft.model?.family || 'anima', label: 'Anima' }]).map(item => ({ value: item.name, label: item.name === 'toy' ? text('Toy（流程测试）', 'Toy (pipeline testing)') : item.label }))} onValueChange={value => { const selected = families.find(item => item.name === value); if (selected) { setDraft(changeModelFamily(draft, selected, assets)); setSaved(false); } }}/></label><label>{text('计算精度', 'Compute precision')}<StudioSelect aria-label={text('计算精度', 'Compute precision')} value={draft.model?.dtype || 'bf16'} options={['bf16', 'fp16', 'fp32'].map(value => ({ value, label: value.toUpperCase() }))} onValueChange={value => { setDraft({ ...draft, model: { ...draft.model, dtype: value } }); setSaved(false); }}/></label></div>
      {draft.model?.family === 'toy' ? <p className="project-model-note">{text('Toy 是流程测试模型。实际图像 LoRA 训练请选择 Anima 或 Krea 2。', 'Toy is for pipeline testing. Use Anima or Krea 2 for image LoRA training.')}</p> : fields.map(field => {
        const kind = MODEL_PATH_FIELDS[field.field as keyof typeof MODEL_PATH_FIELDS];
        const matched = assets.filter(asset => asset.family === draft.model?.family && asset.kind === kind && asset.exists);
        const current = draft.model?.[field.field] || '';
        const registered = matched.find(asset => asset.path === current);
        return <div key={field.field} className="project-model-field" role="group" aria-label={field.label}>
          <div className="project-model-label"><label>{field.label}</label>{registered ? <span className="model-ready"><CheckCircle2 size={12}/>{text('本地可用', 'Available locally')}</span> : current ? <span>{text('自定义路径', 'Custom path')}</span> : <span>{field.field === 'tokenizer_path' ? text('使用内置分词器', 'Bundled tokenizer') : text('待选择', 'Not selected')}</span>}</div>
          <div className="project-model-inputs"><StudioSelect aria-label={`${text('从模型库选择', 'Choose registered model')} ${field.label}`} value={registered?.path || ''} options={[{ value: '', label: current ? text('已使用自定义路径', 'Custom path configured') : matched.length ? text('选择本地模型…', 'Choose a local model…') : text('暂无对应组件，可下载或填写已有路径', 'No component yet — download or enter a local path') }, ...matched.map(asset => ({ value: asset.path, label: `${asset.path.split(/[\\/]/).pop()}${asset.is_default ? text(' · 默认', ' · default') : ''}` }))]} onValueChange={value => { if (value) updatePath(field.field, value); }}/>
            <details open={current && !registered ? true : undefined}><summary>{text('文件路径 / 自定义', 'File path / custom')}</summary><PathInput ariaLabel={field.label} value={current} onChange={value => updatePath(field.field, value)} placeholder={text('训练机上的文件或目录路径', 'File or directory on the training computer')}/>{field.hint && <p>{field.hint}</p>}</details>
          </div>
        </div>;
      })}
    </fieldset>
  </section>;
}
