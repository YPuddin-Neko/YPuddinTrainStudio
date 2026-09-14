import React from 'react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import { Check, Download, ExternalLink, FolderSearch, KeyRound, Loader2, Plus, RefreshCw, Search, Star, Trash2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { ModelAsset, ModelDownload as BaseDownload, ModelDownloadRequest, Settings } from '../../api/types';
import { useFamilies } from '../../api/hooks/useFamilies';
import LocalModelRegistration from './LocalModelRegistration';
import StudioSelect from '../../components/StudioSelect';
import Dialog from '../../components/Dialog';
import { formatBytes } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { modelAssetUnsupportedReason, modelFamilyWeights, trainingFamilyOptions } from '../../utils/trainingFamilies';
import './models.css';

type ModelDownload = BaseDownload & {bytes_per_second?: number; eta_seconds?: number | null; progress_at?: number | null};
type Provider = 'huggingface' | 'modelscope';
type Recommendation = {
  id: string; family: string; kind: string; name: string; dtype: string; size: number;
  purpose?: 'training' | 'inference'; variant?: 'raw' | 'turbo' | null;
  recommended: boolean; model_id: string | null; available_path: string | null; is_default: boolean;
  sources: { provider: Provider; repo_id: string; filename: string; revision: string; url: string }[];
};
const isActive = (task: ModelDownload) => ['queued', 'downloading'].includes(task.status);
const basename = (path: string) => path.split(/[\\/]/).pop() || path;
const primary = 'model-button model-button-primary';
const secondary = 'model-button';

export default function Models({ embedded = false }: { embedded?: boolean }) {
  const text = useWorkspaceText();
  const location = useLocation();
  const [params, setParams] = useSearchParams();
  const { data: families = [], isPending: familiesLoading, error: familiesError, refetch: refreshFamilies } = useFamilies();
  const family = params.get('family') || families.find(item => item.name !== 'toy')?.name || '';
  const selectedFamily = families.find(item => item.name === family);
  const weights = modelFamilyWeights(selectedFamily);
  const kinds = weights.map(weight => weight.kind);
  const downloadableKinds = weights.filter(weight => weight.downloadable).map(weight => weight.kind);
  const view = ['library', 'downloads'].includes(params.get('view') || '') ? params.get('view')! : 'prepare';
  const [models, setModels] = React.useState<ModelAsset[]>([]);
  const [downloads, setDownloads] = React.useState<ModelDownload[]>([]);
  const [observedAt, setObservedAt] = React.useState(() => Date.now() / 1000);
  const transferring = downloads.some(isActive);
  React.useEffect(() => {
    if (!transferring) return;
    const timer = window.setInterval(() => setObservedAt(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [transferring]);
  const [catalog, setCatalog] = React.useState<Recommendation[]>([]);
  const [settings, setSettings] = React.useState<Settings | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [loadErrors, setLoadErrors] = React.useState<Record<string, string>>({});
  const [notice, setNotice] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [provider, setProvider] = React.useState<Provider>('huggingface');
  const [query, setQuery] = React.useState('');
  const [page, setPage] = React.useState(1);
  const [dialog, setDialog] = React.useState<'local' | 'download' | null>(null);
  const [formError, setFormError] = React.useState('');
  const [kind, setKind] = React.useState('dit');
  const [dtype, setDtype] = React.useState('bf16');
  const [variant, setVariant] = React.useState('');
  const [isDefault, setIsDefault] = React.useState(true);
  const [sourceMode, setSourceMode] = React.useState<'repo' | 'url'>('repo');
  const [url, setUrl] = React.useState('');
  const [repo, setRepo] = React.useState('');
  const [filename, setFilename] = React.useState('');
  const [revision, setRevision] = React.useState('');
  const [mirror, setMirror] = React.useState<'official' | 'hf-mirror'>('official');
  const [remove, setRemove] = React.useState<ModelAsset | null>(null);
  const label = (kind: string) => weights.find(weight => weight.kind === kind)?.label || ({ dit: text('主模型 / DiT', 'Base model / DiT'), text_encoder: text('文本编码器', 'Text encoder'), text_encoder_2: text('第二文本编码器', 'Second text encoder'), vae: 'VAE', tokenizer: text('分词器目录', 'Tokenizer directory') }[kind] || kind);
  const transfer = (task: ModelDownload) => {
    const recent = task.progress_at && observedAt - task.progress_at < 10;
    const rate = recent ? task.bytes_per_second || 0 : 0;
    const seconds = recent ? task.eta_seconds : null;
    return `${formatBytes(task.downloaded_bytes)}${task.total_bytes ? ` / ${formatBytes(task.total_bytes)}` : ''} · ${rate > 0 ? `${formatBytes(rate)}/s` : text('等待数据', 'Waiting for data')}${seconds != null && rate > 0 ? ` · ${text('约剩', 'About')} ${seconds >= 60 ? `${Math.ceil(seconds / 60)} ${text('分钟','min')}` : `${Math.ceil(seconds)} ${text('秒','sec')}`}` : ''}`;
  };
  const updateParams = (values: Record<string, string>) => {
    const next = new URLSearchParams(params); Object.entries(values).forEach(([key, value]) => next.set(key, value));
    setParams(next, { replace: true, state: location.state }); setPage(1); setQuery('');
  };
  const refresh = React.useCallback(async (silent = false) => {
      const [assets, tasks, config, recommendations] = await Promise.allSettled([
        apiClient.get<ModelAsset[]>('/models', { silent }), apiClient.get<ModelDownload[]>('/models/downloads', { silent }),
        apiClient.get<Settings>('/settings', { silent }), apiClient.get<Recommendation[]>('/models/recommendations', { silent }),
      ]);
      if (assets.status === 'fulfilled') setModels(assets.value);
      if (tasks.status === 'fulfilled') setDownloads(tasks.value);
      if (config.status === 'fulfilled') setSettings(config.value);
      if (recommendations.status === 'fulfilled') setCatalog(recommendations.value);
      const failures: Record<string, string> = {};
      for (const [key, result] of [['library', assets], ['downloads', tasks], ['settings', config], ['prepare', recommendations]] as const) {
        if (result.status === 'rejected') failures[key] = formatApiError(result.reason);
      }
      setLoadErrors(failures); setLoading(false);
  }, []);
  React.useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(true), 2500); return () => clearInterval(timer); }, [refresh]);
  const action = async (operation: () => Promise<void>, inForm = false) => {
    setBusy(true); setError(''); setFormError(''); setNotice('');
    try { await operation(); await refresh(); window.dispatchEvent(new Event('studio-models-changed')); }
    catch (error) { (inForm ? setFormError : setError)(formatApiError(error)); }
    finally { setBusy(false); }
  };
  const openForm = (mode: 'local' | 'download', component = 'dit') => {
    if (!selectedFamily || family === 'toy' || !kinds.includes(component)) { setError(text('请选择训练服务支持的模型系列和组件。','Choose a model family and component supported by the training service.')); return; }
    setKind(mode === 'download' && !downloadableKinds.includes(component) ? downloadableKinds[0] || 'dit' : component); setUrl(''); setRepo(''); setFilename(''); setRevision('');
    setDtype('bf16'); setVariant(''); setIsDefault(!models.some(item => item.family === family && item.kind === component && item.is_default && !modelAssetUnsupportedReason(item)));
    setDialog(mode); setFormError(''); setSourceMode('repo'); setMirror('official');
  };
  const closeForm = () => { if (!busy) setDialog(null); };
  const selected = models.filter(model => model.family === family);
  const supportedSelected = selected.filter(model => !modelAssetUnsupportedReason(model) && (model as ModelAsset & {purpose?:string}).purpose !== 'inference');
  const entries = catalog.filter(entry => entry.family === family && !models.some(model => modelAssetUnsupportedReason(model) && (model.id === entry.model_id || model.path === entry.available_path)));
  const catalogProviders = (['huggingface', 'modelscope'] as Provider[]).filter(item => entries.some(entry => entry.sources.some(source => source.provider === item)));
  const catalogProvider = catalogProviders.includes(provider) ? provider : catalogProviders[0] || provider;
  const tasks = downloads.filter(task => task.family === family);
  const ready = kinds.filter(kind => supportedSelected.some(model => model.kind === kind && model.is_default && model.exists));
  const required = weights.filter(weight => weight.required).map(weight => weight.kind);
  const readyRequired = required.filter(kind => ready.includes(kind));
  const missing = required.filter(kind => !ready.includes(kind)).map(kind => entries.find(entry => entry.kind === kind && entry.recommended && entry.purpose !== 'inference')).filter((entry): entry is Recommendation => Boolean(entry));
  const taskFor = (entry: Recommendation) => tasks.find(task => isActive(task) && (task.recommendation_id === entry.id || entry.sources.some(source => basename(source.filename) === task.filename)));
  const startEntry = async (entry: Recommendation) => {
    if (entry.available_path) await apiClient.post(`/models/recommendations/${entry.id}/use`, {});
    else await apiClient.post(`/models/recommendations/${entry.id}/download`, { provider: catalogProvider, is_default: entry.purpose !== 'inference' && !ready.includes(entry.kind) });
  };
  const searchValue = query.trim().toLocaleLowerCase();
  const filteredModels = selected.filter(model => `${model.path} ${label(model.kind)}`.toLocaleLowerCase().includes(searchValue));
  const filteredTasks = tasks.filter(task => `${task.filename} ${task.id}`.toLocaleLowerCase().includes(searchValue));
  const count = view === 'library' ? filteredModels.length : filteredTasks.length;
  const pages = Math.max(1, Math.ceil(count / 12));
  const currentPage = Math.min(page, pages);
  const slice = <T,>(items: T[]) => items.slice((currentPage - 1) * 12, currentPage * 12);
  const settingsLink = (tab: string) => `/settings/environment?tab=${tab}&family=${family}`;
  const tabs = [{ key: 'prepare', label: text('准备模型', 'Prepare models') }, { key: 'library', label: `${text('本地模型', 'Local models')} · ${selected.length}` }, { key: 'downloads', label: `${text('下载记录', 'Downloads')}${tasks.some(isActive) ? ` · ${tasks.filter(isActive).length} ${text('进行中', 'active')}` : ''}` }];
  const statusLabel = (task: ModelDownload) => ({ queued: text('排队中', 'Queued'), downloading: text('下载中', 'Downloading'), completed: text('已就绪', 'Ready'), failed: text('下载失败', 'Failed'), cancelled: text('已取消', 'Cancelled') }[task.status]);

  return <div className="models-workspace" data-testid="models-page">
    <div className="models-toolbar">
      <div className="models-heading"><div><h2>{text('模型权重', 'Model weights')}</h2>{!embedded && <p>{text('准备模型组件，供项目选择。', 'Prepare components for your projects.')}</p>}</div>
        <div className="model-actions"><Link to={settingsLink('credentials')} replace state={location.state} className={secondary}><KeyRound size={14}/>{text('访问密钥', 'Access keys')}</Link><button className={secondary} onClick={() => void refresh()} aria-label={text('刷新模型', 'Refresh models')}><RefreshCw size={14}/></button></div>
      </div>
      <div className="models-filters"><StudioSelect aria-label={text('模型系列', 'Model family')} value={family} disabled={familiesLoading || !!familiesError} options={trainingFamilyOptions(families)} onValueChange={family => updateParams({ family })}/>
        <div className="models-view-tabs" role="tablist" aria-label={text('模型管理视图', 'Model management views')}>{tabs.map(tab => <button key={tab.key} role="tab" aria-selected={view === tab.key} onClick={() => updateParams({ view: tab.key })}>{tab.label}</button>)}</div>
      </div>
    </div>
    <p className="models-storage-note">{text('模型目录', 'Model directory')}：<span title={settings?.paths.models_dir}>{settings?.paths.models_dir}</span><Link to="/settings/preferences?section=storage" replace state={location.state}>{text('更改', 'Change')}</Link></p>
    {error && <div role="alert" className="settings-alert">{error}<button className={secondary} onClick={() => void refresh()}>{text('重试', 'Retry')}</button></div>}
    {Object.keys(loadErrors).length > 0 && <div role="alert" className="settings-alert"><div>{Object.entries(loadErrors).map(([key, message]) => <p key={key}>{({ library: text('本地模型', 'Local models'), downloads: text('下载记录', 'Downloads'), settings: text('存储设置', 'Storage settings'), prepare: text('推荐模型', 'Recommended models') })[key]}：{message}</p>)}</div><button className={secondary} onClick={() => void refresh()}>{text('重新读取', 'Reload')}</button></div>}
    {notice && <p className="model-notice" role="status">{notice}</p>}
    {familiesError && <div role="alert" className="settings-alert">{text('无法读取支持的模型系列。', 'Could not load supported model families.')}<button className={secondary} onClick={() => void refreshFamilies()}>{text('重试', 'Retry')}</button></div>}
    {family === 'flux' && <p role="alert" className="settings-alert">{text('FLUX.1 已停用。已有模型文件保持原样，请选择受支持的模型系列。', 'FLUX.1 is retired. Existing model files are preserved; choose a supported model family.')}</p>}
    {loading || familiesLoading ? <p role="status" className="model-empty">{text('正在读取模型…', 'Loading models…')}</p> : !selectedFamily ? <p className="model-empty">{text('请选择训练服务支持的模型系列。', 'Choose a model family supported by the training service.')}</p> : loadErrors[view] ? <p className="model-empty">{text('此列表暂时无法读取，请重试。其他视图仍可查看。', 'This list is unavailable. Retry or open another view.')}</p> : view === 'prepare' ? <>
      {family === 'toy' ? <p className="model-empty">{text('Toy 测试模型已内置，无需下载权重。', 'The Toy test model is built in; no weights required.')}</p> : <>
        <div className="models-source-bar"><label>{text('下载来源', 'Download source')}<StudioSelect value={catalogProvider} aria-label={text('下载来源', 'Download source')} data-testid="model-provider" options={(catalogProviders.length ? catalogProviders : ['huggingface', 'modelscope']).map(value => ({ value, label: value === 'huggingface' ? 'Hugging Face' : '魔搭 ModelScope' }))} onValueChange={value => setProvider(value as Provider)}/></label><span>{text('必需组件', 'Required components')} {readyRequired.length} / {required.length}</span><button className={primary} disabled={busy || !!loadErrors.library || !!loadErrors.downloads || missing.length === 0 || missing.every(entry => taskFor(entry) || (!entry.available_path && !entry.sources.some(source => source.provider === catalogProvider)))} onClick={() => void action(async () => { for (const entry of missing) if (!taskFor(entry) && (entry.available_path || entry.sources.some(source => source.provider === catalogProvider))) await startEntry(entry); setNotice(text('缺失组件已开始准备，可在下载记录查看进度。', 'Missing components are being prepared. See Downloads for progress.')); })}><Download size={14}/>{text('准备缺失组件', 'Prepare missing components')}</button></div>
        {entries.length === 0 && <p className="model-help-text">{text('添加已有模型，或通过自定义下载填写模型的来源。', 'Add an existing model or enter its source in Custom download.')}</p>}
        {kinds.map(kind => <section className="model-component" key={kind} data-testid={`model-component-${kind}`}>
          <header><h3>{label(kind)}{!required.includes(kind) && <span className="model-tag">{text('可选', 'Optional')}</span>}</h3><button className={secondary} onClick={() => openForm('local', kind)}><Plus size={13}/>{text('已有文件', 'Local file')}</button></header>
          {entries.filter(entry => entry.kind === kind).map(entry => {
            const task = taskFor(entry); const source = entry.sources.find(source => source.provider === catalogProvider);
            return <div className="model-catalog-row" key={entry.id}><div className="model-catalog-description"><strong>{entry.name}</strong><div><span>{formatBytes(entry.size)}</span>{entry.recommended && <span className="model-tag">{text('推荐', 'Recommended')}</span>}<a href={(source || entry.sources[0])?.url} target="_blank" rel="noreferrer">{text('发布页', 'Source')}<ExternalLink size={11}/></a></div>{task && <><p className="model-transfer-status">{transfer(task)}</p><progress aria-label={`${entry.name} ${text('下载进度', 'download progress')}`} max={task.total_bytes || undefined} value={task.total_bytes ? task.downloaded_bytes : undefined}/></>}</div>
              <div className="model-catalog-action">{entry.is_default && entry.available_path ? <span className="model-ready"><Check size={14}/>{text('当前默认', 'Current default')}</span> : task ? <button className={secondary} onClick={() => updateParams({ view: 'downloads' })}>{statusLabel(task)}{task.total_bytes ? ` ${Math.min(100, Math.round(task.downloaded_bytes / task.total_bytes * 100))}%` : ''}</button> : <button className={entry.available_path ? secondary : primary} disabled={busy || (!source && !entry.available_path) || !!loadErrors.library || !!loadErrors.downloads} onClick={() => void action(async () => { await startEntry(entry); setNotice(entry.available_path ? entry.purpose === 'inference' ? text('已登记为仅采样模型。', 'Registered for sampling only.') : text('已设为默认组件。', 'Default component selected.') : text('已开始下载，完成后自动登记。', 'Download started. The file will be registered when complete.')); })}>{entry.available_path ? <Check size={14}/> : <Download size={14}/>} {entry.available_path ? entry.purpose === 'inference' ? text('仅采样 · 已就绪', 'Sampling only · ready') : text('设为默认', 'Use as default') : text('下载', 'Download')}</button>}</div>
            </div>;
          })}
          {supportedSelected.some(model => model.kind === kind) && <div className="model-default-row"><label>{text('本地默认', 'Local default')}</label><StudioSelect aria-label={`${label(kind)} ${text('默认模型', 'default model')}`} value={supportedSelected.find(model => model.kind === kind && model.is_default)?.id || ''} options={[{ value: '', label: text('清除默认模型', 'Clear default model'), displayLabel: text('未设置默认模型', 'No default model') }, ...supportedSelected.filter(model => model.kind === kind).map(model => ({ value: model.id, label: `${basename(model.path)}${!model.exists ? text('（文件缺失）', ' (missing)') : ''}`, disabled: !model.exists }))]} disabled={busy} onValueChange={id => void action(async () => { const previous = selected.find(model => model.kind === kind && model.is_default); if (id) await apiClient.patch(`/models/${id}`, { is_default: true }); else if (previous) await apiClient.patch(`/models/${previous.id}`, { is_default: false }); })}/></div>}
        </section>)}
        <div className="model-actions"><button className={secondary} data-testid="download-model-btn" onClick={() => openForm('download')}><Plus size={14}/>{text('自定义下载', 'Custom download')}</button><button className={secondary} data-testid="add-model-btn" onClick={() => openForm('local')}><Plus size={14}/>{text('添加本地模型', 'Add local model')}</button></div>
        <details className="model-help"><summary>{text('默认组件与下载说明', 'Defaults and downloads')}</summary><p>{text('默认组件用于新配置，已有项目的明确路径会保留。标准单文件的配置与分词器已内置；自定义编码器可登记完整 HF 目录。Anima 与 Krea 2 的 VAE 可共用已有文件。', 'Defaults fill new configurations; existing explicit paths are retained. Standard single-file configurations and tokenizers are bundled. Custom encoders can use a complete local HF directory. Anima and Krea 2 can share an existing VAE.')}</p><p>{text('目录中的文件会校验大小、SHA-256 和组件类型，再加入模型库。中断后的重新下载从头开始。', 'Catalog downloads are checked for size, SHA-256 and component type before registration. Retrying an interrupted download restarts from the beginning.')}</p></details>
      </>}
    </> : <>
      <div className="models-list-toolbar"><label className="model-search"><Search size={15}/><input value={query} aria-label={text('搜索模型或下载', 'Search models or downloads')} placeholder={text('搜索名称、路径或任务 ID', 'Search name, path or task ID')} onChange={event => { setQuery(event.target.value); setPage(1); }}/></label><div className="model-actions">{view === 'library' && <><button className={secondary} onClick={() => openForm('local')}><Plus size={14}/>{text('本地文件', 'Local file')}</button><button className={secondary} disabled={busy} onClick={() => void action(async () => { const found = await apiClient.post<ModelAsset[]>('/models/scan', { family }); setNotice(`${text('新登记文件', 'Newly registered files')}：${found.length}。${text('未识别文件请使用“本地文件”逐个确认。','Review unidentified files through Local file.')}`); })}><FolderSearch size={14}/>{text('扫描模型目录', 'Scan model directory')}</button></>}</div></div>
      <div className="models-pagination"><span>{count} {text('项', 'items')}</span><button className={secondary} disabled={currentPage === 1} onClick={() => setPage(currentPage - 1)}>{text('上一页', 'Previous')}</button><span>{currentPage} / {pages}</span><button className={secondary} disabled={currentPage === pages} onClick={() => setPage(currentPage + 1)}>{text('下一页', 'Next')}</button></div>
      {count === 0 ? <p className="model-empty">{text('没有匹配的记录。', 'No matching records.')}</p> : view === 'library' ? <div className="model-library-list">{slice(filteredModels).map(model => <div className="model-library-row" key={model.id}><div><strong>{basename(model.path)}</strong>{(model as ModelAsset & {purpose?:string}).purpose==='inference'&&<span className="model-ready">{text('仅采样','Sampling only')}</span>}<p>{label(model.kind)} · {model.dtype || '—'} · {formatBytes(model.size)} · {modelAssetUnsupportedReason(model) ? text('已停用', 'Retired') : model.exists ? text('可用', 'Available') : text('文件缺失', 'Missing file')}</p>{modelAssetUnsupportedReason(model) && <p className="model-help-text">{modelAssetUnsupportedReason(model)}</p>}<details><summary>{text('文件路径', 'File path')}</summary><code>{model.path}</code></details></div><div className="model-actions"><button className={secondary} disabled={busy || (model as ModelAsset & {purpose?:string}).purpose === 'inference' || !model.is_default && (!model.exists || !!modelAssetUnsupportedReason(model))} aria-label={`${model.is_default && modelAssetUnsupportedReason(model) ? text('取消默认', 'Clear default') : text('设为默认', 'Set default')} ${basename(model.path)}`} onClick={() => void action(async () => { await apiClient.patch(`/models/${model.id}`, { is_default: !model.is_default }); })}><Star size={14} fill={model.is_default ? 'currentColor' : 'none'}/>{model.is_default ? modelAssetUnsupportedReason(model) ? text('取消默认', 'Clear default') : text('默认', 'Default') : text('设为默认', 'Set default')}</button><button className={secondary} disabled={busy} onClick={() => setRemove(model)} aria-label={`${text('移除登记', 'Remove registration')} ${basename(model.path)}`}><Trash2 size={14}/></button></div></div>)}</div> : <div data-testid="model-downloads" className="model-download-list">{slice(filteredTasks).map(task => <div className="model-download-row" key={task.id}><div className="model-download-heading"><div><strong>{task.filename}</strong><p>{task.provider === 'modelscope' ? 'ModelScope' : task.mirror === 'hf-mirror' ? 'HF-Mirror' : 'Hugging Face'} · {statusLabel(task)} · {formatBytes(task.downloaded_bytes)}{task.total_bytes ? ` / ${formatBytes(task.total_bytes)}` : ''}</p></div><div className="model-actions">{isActive(task) ? <button className={secondary} disabled={busy} onClick={() => void action(async () => { await apiClient.post(`/models/downloads/${task.id}/cancel`, {}); })}>{text('取消', 'Cancel')}</button> : task.status !== 'completed' && <><button className={secondary} disabled={busy} onClick={() => void action(async () => { await apiClient.post(`/models/downloads/${task.id}/retry`, {}); })}>{text('重新下载', 'Restart download')}</button><button className={secondary} onClick={() => { openForm('download', task.kind); setProvider(task.provider); setMirror(task.mirror); setSourceMode('url'); setUrl(task.source_url); setDtype(task.dtype || ''); setVariant((task as ModelDownload & {variant?:string}).variant || ''); if ((task as ModelDownload & {purpose?:string}).purpose === 'inference') setIsDefault(false); }}>{text('更换来源', 'Change source')}</button></>}</div></div>{isActive(task) && <progress aria-label={text('下载进度', 'Download progress')} max={task.total_bytes || undefined} value={task.total_bytes ? task.downloaded_bytes : undefined}/>}{isActive(task) && <p className="model-transfer-status">{transfer(task)}</p>}{task.error && <p className="model-download-error">{task.error}</p>}<details><summary>{text('保存位置与任务信息', 'Destination and task details')}</summary><code>{task.target_path}</code><p>{task.id}</p></details></div>)}</div>}
    </>}

    {dialog && <Dialog title={dialog === 'local' ? text('添加本地模型', 'Add local model') : text('自定义下载', 'Custom download')} onClose={closeForm} closeDisabled={busy} wide>{dialog === 'local' ? <LocalModelRegistration initialFamily={family} families={families} onClose={closeForm} onBusyChange={setBusy} onRegistered={async(family)=>{await refresh();window.dispatchEvent(new Event('studio-models-changed'));setDialog(null);updateParams({view:'library',family});setNotice(text('已登记模型。','Model registered.'));}}/> : <form className="model-source-form" data-testid="download-model-form" onSubmit={event => { event.preventDefault(); if(family==='krea2'&&kind==='dit'&&!variant){setFormError(text('请确认 Raw 或 Turbo。','Confirm Raw or Turbo.'));return;} void action(async () => {
      await apiClient.post('/models/downloads', { family, kind, provider, mirror, dtype: dtype || null, is_default: variant === 'turbo' ? false : isDefault, ...(family === 'krea2' && kind === 'dit' ? {variant, purpose: variant === 'turbo' ? 'inference' : 'training'} : {}), ...(sourceMode === 'url' ? { url: url.trim() } : { repo_id: repo.trim(), filename: filename.trim(), revision: revision.trim() || (provider === 'modelscope' ? 'master' : 'main') }) } as ModelDownloadRequest);
      setDialog(null); updateParams({ view: 'downloads' }); setNotice(text('已提交。', 'Submitted.'));
    }, true); }}>
      <fieldset disabled={busy}>{family==='krea2'&&kind==='dit'&&<label>{text('Krea 2 版本 / 用途','Krea 2 variant / purpose')}<StudioSelect aria-label={text('Krea 2 版本 / 用途','Krea 2 variant / purpose')} value={variant} onValueChange={value=>{setVariant(value);if(value==='turbo')setIsDefault(false);}} placeholder={text('请按发布说明选择','Choose from publisher description')} options={[{value:'raw',label:'Raw'},{value:'turbo',label:text('Turbo · 仅采样','Turbo · sampling only')}]}/></label>}<p className="model-help-text">{text('从对应平台填写仓库和完整文件路径，或粘贴单文件链接。', 'Enter a repository and complete file path on the selected platform, or paste a single-file URL.')}</p>
        <div className="model-form-grid"><label>{text('组件', 'Component')}<StudioSelect data-testid="model-kind-select" aria-label={text('组件', 'Component')} value={kind} onValueChange={value => { setKind(value); setVariant(''); }} options={weights.filter(weight => weight.kind !== 'tokenizer' && weight.downloadable).map(weight => ({ value: weight.kind, label: label(weight.kind) }))}/></label><label>{text('权重精度', 'Weight precision')}<StudioSelect aria-label={text('权重精度', 'Weight precision')} value={dtype} onValueChange={setDtype} options={['bf16', 'fp16', 'fp32', 'fp8', ''].map(value => ({ value, label: value.toUpperCase() || text('未知', 'Unknown') }))}/></label></div>
        < >
          <div className="model-form-grid"><label>{text('下载平台', 'Download platform')}<StudioSelect aria-label={text('下载平台', 'Download platform')} value={provider} onValueChange={value => { setProvider(value as Provider); setMirror('official'); setRevision(''); }} options={[{ value: 'huggingface', label: 'Hugging Face' }, { value: 'modelscope', label: '魔搭 ModelScope' }]}/></label><label>{text('来源格式', 'Source format')}<StudioSelect aria-label={text('来源格式', 'Source format')} value={sourceMode} onValueChange={value => setSourceMode(value as 'repo' | 'url')} options={[{ value: 'repo', label: text('仓库与文件', 'Repository and file') }, { value: 'url', label: text('文件链接', 'File URL') }]}/></label></div>
          {sourceMode === 'url' ? <label>{text('文件下载链接', 'File URL')}<input required data-testid="model-download-url" value={url} onChange={event => setUrl(event.target.value)} placeholder="https://…"/></label> : <><label>{text('仓库 ID', 'Repository ID')}<input required value={repo} onChange={event => setRepo(event.target.value)} placeholder="owner/repository"/></label><label>{text('仓库内文件路径', 'File path in repository')}<input required value={filename} onChange={event => setFilename(event.target.value)} placeholder="folder/model.safetensors"/></label><label>{text('分支 / Revision', 'Branch / revision')}<input value={revision} onChange={event => setRevision(event.target.value)} placeholder={provider === 'modelscope' ? 'master' : 'main'}/></label></>}
          {provider === 'huggingface' && <details><summary>{text('连接选项', 'Connection options')}</summary><StudioSelect aria-label={text('连接方式', 'Connection mode')} value={mirror} onValueChange={value => setMirror(value as 'official' | 'hf-mirror')} options={[{ value: 'official', label: text('官方（使用已保存令牌）', 'Official (saved token)') }, { value: 'hf-mirror', label: 'HF-Mirror · ' + text('仅匿名', 'anonymous only') }]}/></details>}
          <p className="model-help-text">{text('仅下载完整 safetensors 单文件；分片模型请登记完整本地目录。受限仓库使用设置里保存的访问密钥。', 'Only complete safetensors files are downloaded; register a full local directory for sharded models. Restricted repositories use the access keys saved in Settings.')}</p>
        </>
        <label className="model-checkbox"><input type="checkbox" checked={variant==='turbo'?false:isDefault} disabled={variant==='turbo'} onChange={event => setIsDefault(event.target.checked)}/>{text('完成后设为本系列默认组件', 'Set as the default component when ready')}</label>
      </fieldset>
      {formError && <div role="alert" className="settings-alert">{formError}</div>}
      <footer><button type="button" className={secondary} disabled={busy} onClick={closeForm}>{text('取消', 'Cancel')}</button><button className={primary} data-testid="model-download-start" disabled={busy} type="submit">{busy && <Loader2 size={14} className="animate-spin"/>}{text('开始下载', 'Start download')}</button></footer>
    </form>}</Dialog>}
    {remove && <Dialog title={text('移除模型登记', 'Remove model registration')} onClose={() => !busy && setRemove(null)} closeDisabled={busy}><div className="model-source-form"><p>{text('移除后不再显示在模型库，磁盘文件会保留。已有项目路径不会改变。', 'The entry will be removed from the library. Its file and existing project paths are retained.')}</p><code>{remove.path}</code><footer><button className={secondary} disabled={busy} onClick={() => setRemove(null)}>{text('取消', 'Cancel')}</button><button className={primary} disabled={busy} onClick={() => void action(async () => { await apiClient.delete(`/models/${remove.id}`); setRemove(null); })}>{text('移除登记', 'Remove registration')}</button></footer></div></Dialog>}
  </div>;
}
