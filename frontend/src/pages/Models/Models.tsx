import React from 'react';
import { useLocation, useSearchParams } from 'react-router-dom';
import { Check, Download, ExternalLink, FolderSearch, Loader2, Plus, Search, Star, Trash2, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import { useImageModelResources, type ImageRecommendation as Recommendation } from '../../api/hooks/useImageModelResources';
import { usePageVisible, useSharedSettings } from '../../api/resourcePolicy';
import type { ModelAsset, ModelDownload as BaseDownload, ModelDownloadRequest } from '../../api/types';
import { useFamilies } from '../../api/hooks/useFamilies';
import LocalModelRegistration from './LocalModelRegistration';
import StudioSelect from '../../components/StudioSelect';
import Dialog from '../../components/Dialog';
import { formatBytes } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { modelAssetUnsupportedReason, modelFamilyWeights, trainingFamilyOptions } from '../../utils/trainingFamilies';
import { modelOrigin } from '../../utils/modelOrigin';
import './models.css';
import Switch from '../../components/Switch';
import ProgressBar from '../../components/ProgressBar';
import { SlidingIndicator } from '../../components/motion';
import { LoadingNote } from '../../components/Loading';
import TtsModels from './TtsModels';
import { ModelToolbar, ModelCatalogSection, ModelCatalogRow } from './ModelDownloadLayout';

type ModelDownload = BaseDownload & {bytes_per_second?: number; eta_seconds?: number | null; progress_at?: number | null};
type Provider = 'huggingface' | 'modelscope';
const isActive = (task: ModelDownload) => ['queued', 'downloading'].includes(task.status);
const basename = (path: string) => path.split(/[\\/]/).pop() || path;
// Each list arrives on its own, so one slow or failed read holds back only the part that shows it.
type Resource = 'library' | 'downloads' | 'settings' | 'prepare';
const primary = 'ui-btn ui-btn-primary';
const secondary = 'ui-btn';

export default function Models({ embedded = false }: { embedded?: boolean }) {
  const text = useWorkspaceText(), location = useLocation();
  const [params, setParams] = useSearchParams();
  const speech = params.get('type') === 'tts';
  const typeSelector = <StudioSelect aria-label={text('模型类型', 'Model type')} value={speech ? 'tts' : 'image'} options={[{ value: 'image', label: text('图像模型', 'Image models') }, { value: 'tts', label: text('语音模型', 'Speech models') }]}
    onValueChange={type => { const next = new URLSearchParams(params); next.set('type', type); setParams(next, { replace: true, state: location.state }); }}/>;
  return speech ? <TtsModels embedded={embedded} typeSelector={typeSelector}/> : <ImageModels embedded={embedded} typeSelector={typeSelector}/>;
}

function ImageModels({ embedded, typeSelector }: { embedded: boolean; typeSelector: React.ReactNode }) {
  const text = useWorkspaceText();
  const location = useLocation();
  const [params, setParams] = useSearchParams();
  const { data: families = [], isPending: familiesLoading, error: familiesError, refetch: refreshFamilies } = useFamilies();
  const family = params.get('family') || families.find(item => item.name !== 'toy')?.name || '';
  const selectedFamily = families.find(item => item.name === family);
  const weights = modelFamilyWeights(selectedFamily);
  const kinds = weights.map(weight => weight.kind);
  const downloadableKinds = weights.filter(weight => weight.downloadable).map(weight => weight.kind);
  const view = params.get('view') === 'library' ? 'library' : 'prepare';
  const resources = useImageModelResources();
  const settingsQuery = useSharedSettings();
  const models = resources.assets.data ?? [];
  const downloads = React.useMemo(() => resources.tasks.data ?? [], [resources.tasks.data]);
  const catalog = resources.catalog.data ?? [];
  const settings = settingsQuery.data ?? null;
  const queries = { library: resources.assets, downloads: resources.tasks, settings: settingsQuery, prepare: resources.catalog };
  const loaded = Object.fromEntries(Object.entries(queries).map(([key, query]) => [key, query.data !== undefined])) as Record<Resource, boolean>;
  const loadErrors = Object.fromEntries(Object.entries(queries).filter(([, query]) => query.error).map(([key, query]) => [key, formatApiError(query.error)])) as Record<Resource, string>;
  const visible = usePageVisible();
  const [observedAt, setObservedAt] = React.useState(() => Date.now() / 1000);
  const transferring = downloads.some(isActive);
  React.useEffect(() => {
    if (!transferring || !visible) return;
    const timer = window.setInterval(() => setObservedAt(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [transferring, visible]);
  const [error, setError] = React.useState('');
  const [notice, setNotice] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const pendingAction = React.useRef(false);
  const [provider, setProvider] = React.useState<Provider>(() => {
    try { return localStorage.getItem('studio.model-download-provider') === 'modelscope' ? 'modelscope' : 'huggingface'; } catch { return 'huggingface'; }
  });
  React.useEffect(() => { try { localStorage.setItem('studio.model-download-provider', provider); } catch { /* Storage may be disabled by the browser. */ } }, [provider]);
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
  const lifetime = React.useRef<AbortController | null>(null);
  React.useEffect(() => {
    const controller = new AbortController(); lifetime.current = controller;
    return () => controller.abort();
  }, []);
  const refresh = () => Promise.all([resources.refresh(), settingsQuery.refetch()]);
  const acceptDownload = (task: ModelDownload) => { if (!lifetime.current?.signal.aborted) resources.acceptDownload(task); };
  const acceptModel = (model: ModelAsset, recommendationId?: string) => { if (!lifetime.current?.signal.aborted) resources.acceptModel(model, recommendationId); };
  const setModelDefault = async (id: string, isDefault: boolean) => acceptModel(await apiClient.patch<ModelAsset>(`/models/${id}`, { is_default: isDefault }));
  const action = async (operation: () => Promise<void>, inForm = false) => {
    if (pendingAction.current) return;
    pendingAction.current = true; setBusy(true); setError(''); setFormError(''); setNotice('');
    try { await operation(); window.dispatchEvent(new Event('studio-models-changed')); }
    catch (error) { if (!lifetime.current?.signal.aborted) (inForm ? setFormError : setError)(formatApiError(error)); }
    finally { pendingAction.current = false; if (!lifetime.current?.signal.aborted) setBusy(false); }
  };
  const openForm = (mode: 'local' | 'download', component = mode === 'local' ? '' : 'dit') => {
    if (!selectedFamily || family === 'toy' || component && !kinds.includes(component)) { setError(text('请选择训练服务支持的模型系列和组件。','Choose a model family and component supported by the training service.')); return; }
    setKind(mode === 'download' && !downloadableKinds.includes(component) ? downloadableKinds[0] || 'dit' : component); setUrl(''); setRepo(''); setFilename(''); setRevision('');
    setDtype('bf16'); setVariant(''); setIsDefault(!models.some(item => item.family === family && item.kind === component && item.is_default && !modelAssetUnsupportedReason(item)));
    setDialog(mode); setFormError(''); setSourceMode('repo');
  };
  const closeForm = () => { if (!busy) setDialog(null); };
  const selected = models.filter(model => !family || model.family === family);
  const familyOptions = trainingFamilyOptions(families);
  const localFamilyOptions = families.length ? familyOptions : [
    { value: '', label: text('全部模型系列', 'All model families') },
    ...[...new Set(models.map(model => model.family))].map(value => ({ value, label: value })),
  ];
  const supportedSelected = selected.filter(model => !modelAssetUnsupportedReason(model) && (model as ModelAsset & {purpose?:string}).purpose !== 'inference');
  const entries = catalog.filter(entry => entry.family === family && !models.some(model => modelAssetUnsupportedReason(model) && (model.id === entry.model_id || model.path === entry.available_path)));
  const catalogProviders = (['huggingface', 'modelscope'] as Provider[]).filter(item => entries.some(entry => entry.sources.some(source => source.provider === item)));
  const catalogProvider = catalogProviders.includes(provider) ? provider : catalogProviders[0] || provider;
  // A retry creates a new server task and preserves the earlier attempt.
  // Include completed attempts when choosing the latest state so success cannot
  // uncover an older failure. Custom downloads are identified by destination,
  // never by filename alone: unrelated repositories may use the same filename.
  const latestDownloads = React.useMemo(() => {
    const seen = new Set<string>();
    return downloads.filter(task => task.family === family).sort((a, b) => b.created_at - a.created_at).filter(task => {
      const scope = [task.family, task.kind];
      const keys = [JSON.stringify([...scope, 'destination', task.target_path || task.source_url || task.id])];
      if (task.recommendation_id) keys.push(JSON.stringify([...scope, 'recommendation', task.recommendation_id]));
      const superseded = keys.some(key => seen.has(key));
      keys.forEach(key => seen.add(key));
      return !superseded;
    });
  }, [downloads, family]);
  const tasks = latestDownloads.filter(task => task.status !== 'completed');
  const ready = kinds.filter(kind => supportedSelected.some(model => model.kind === kind && model.is_default && model.exists));
  const required = weights.filter(weight => weight.required).map(weight => weight.kind);
  const availableRequired = required.filter(kind => supportedSelected.some(model => model.kind === kind && model.exists));
  const missing = required.filter(kind => !availableRequired.includes(kind)).map(kind => entries.find(entry => entry.kind === kind && entry.recommended && entry.purpose !== 'inference' && !entry.available_path)).filter((entry): entry is Recommendation => Boolean(entry));
  const taskFor = (entry: Recommendation) => tasks.find(task => task.recommendation_id === entry.id);
  const addedLabel = (entry: Recommendation) => {
    if (!entry.model_id) return text('待检查', 'Not checked');
    const model = models.find(model => model.id === entry.model_id) || models.find(model => model.path === entry.available_path && model.kind === entry.kind);
    const origin = modelOrigin(model, resources.tasks.error ? undefined : resources.tasks.data);
    return origin === 'downloaded' ? text('已下载', 'Downloaded') : origin === 'imported' ? text('已导入', 'Imported') : text('已添加', 'Added');
  };
  const startEntry = async (entry: Recommendation) => {
    if (entry.available_path) acceptModel(await apiClient.post<ModelAsset>(`/models/recommendations/${entry.id}/use`, {}), entry.id);
    else acceptDownload(await apiClient.post<ModelDownload>(`/models/recommendations/${entry.id}/download`, { provider: catalogProvider, is_default: entry.purpose !== 'inference' && !ready.includes(entry.kind) }));
  };
  const retryTask = (task: ModelDownload) => void action(async () => {
    acceptDownload(await apiClient.post<ModelDownload>(`/models/downloads/${task.id}/retry`, task.recommendation_id ? { provider: catalogProvider } : {}));
  });
  const cancelButton = (task: ModelDownload) => <button type="button" className={`${secondary} model-button-downloading`} disabled={busy} aria-label={`${text('取消下载', 'Cancel download')} ${task.filename}`} title={text('取消下载', 'Cancel download')} onClick={() => void action(async () => { acceptDownload(await apiClient.post<ModelDownload>(`/models/downloads/${task.id}/cancel`, {})); })}><span className="model-download-active-label">{statusLabel(task)}</span><span className="model-download-cancel-label"><X size={13}/>{text('取消', 'Cancel')}</span></button>;
  const searchValue = query.trim().toLocaleLowerCase();
  const filteredModels = selected.filter(model => `${model.path} ${label(model.kind)}`.toLocaleLowerCase().includes(searchValue));
  const discovered = entries.filter((entry, index) => entry.available_path && !entry.model_id && models.some(model => model.path === entry.available_path && model.family !== family)
    && !selected.some(model => model.path === entry.available_path && model.kind === entry.kind)
    && entries.findIndex(other => other.available_path === entry.available_path && other.kind === entry.kind) === index);
  const filteredDiscovered = discovered.filter(entry => `${entry.available_path} ${entry.name} ${label(entry.kind)}`.toLocaleLowerCase().includes(searchValue));
  const libraryItems: ({ type: 'registered'; model: ModelAsset } | { type: 'discovered'; entry: Recommendation })[] = [
    ...filteredModels.map(model => ({ type: 'registered' as const, model })), ...filteredDiscovered.map(entry => ({ type: 'discovered' as const, entry })),
  ];
  const libraryKind = (item: typeof libraryItems[number]) => item.type === 'registered' ? item.model.kind : item.entry.kind;
  const libraryKinds = [...new Set([...kinds, ...libraryItems.map(libraryKind)])];
  const orderedLibraryItems = libraryKinds.flatMap(kind => libraryItems.filter(item => libraryKind(item) === kind));
  const count = libraryItems.length;
  const pages = Math.max(1, Math.ceil(count / 12));
  const currentPage = Math.min(page, pages);
  const slice = <T,>(items: T[]) => items.slice((currentPage - 1) * 12, currentPage * 12);
  const pageItems = slice(orderedLibraryItems);
  const visibleLibraryKinds = libraryKinds.filter(kind => pageItems.some(item => libraryKind(item) === kind)
    || !searchValue && !libraryItems.some(item => libraryKind(item) === kind));
  const settingsLink = (tab: string) => `/settings/environment?tab=${tab}&family=${family}`;
  const tabs = [{ key: 'prepare', label: text('在线模型', 'Online models') }, { key: 'library', label: `${text('本地模型', 'Local models')}${loaded.library ? ` · ${selected.length + discovered.length}` : ''}` }];
  const statusLabel = (task: ModelDownload) => ({ queued: text('排队中', 'Queued'), downloading: text('下载中', 'Downloading'), completed: text('已就绪', 'Ready'), failed: text('下载失败', 'Failed'), cancelled: text('已取消', 'Cancelled') }[task.status]);
  // A view shows once its own list has arrived; actions wait for the lists they depend on.
  const viewReady = !!loaded[view];
  // Downloads join their catalog rows once both lists are known (or the catalog cannot be read).
  const listsSettled = !!loaded.downloads && (!!loaded.prepare || !!loadErrors.prepare);
  const linkedTaskIds = new Set(entries.map(taskFor).filter((task): task is ModelDownload => Boolean(task)).map(task => task.id));
  const standaloneTasks = tasks.filter(task => !linkedTaskIds.has(task.id));
  const downloadTaskRow = (task: ModelDownload) => <div className="model-download-inline-row" key={task.id}>
    <div className="model-download-heading"><div><strong>{task.filename}</strong><p>{task.provider === 'modelscope' ? 'ModelScope' : task.mirror === 'hf-mirror' ? 'HF-Mirror' : 'Hugging Face'} · {statusLabel(task)} · {formatBytes(task.downloaded_bytes)}{task.total_bytes ? ` / ${formatBytes(task.total_bytes)}` : ''}</p></div><div className="model-actions">{isActive(task) ? cancelButton(task) : <button className={secondary} disabled={busy} onClick={() => retryTask(task)}>{text('重试', 'Retry')}</button>}</div></div>
    {isActive(task) && <ProgressBar className="model-download-progress" label={text('下载进度', 'Download progress')} max={task.total_bytes || undefined} value={task.total_bytes ? task.downloaded_bytes : undefined}/>} {isActive(task) ? <p className="model-transfer-status">{transfer(task)}</p> : task.status === 'failed' ? <p role="alert" className="model-download-error">{text('下载失败', 'Download failed')}：{task.error || text('未返回详细错误，请重试。', 'No detailed error was returned. Retry the download.')}</p> : <p role="status" className="model-transfer-status">{text('下载已取消，可以重新下载。', 'Download cancelled. You can retry it.')}</p>}
  </div>;

  return <div className="models-workspace" data-testid="models-page">
    <ModelToolbar embedded={embedded} directory={settings?.paths.models_dir} credentialsLink={settingsLink('credentials')} onRefresh={refresh}>
      {typeSelector}<StudioSelect aria-label={text('模型系列', 'Model family')} value={family} disabled={view === 'prepare' && (familiesLoading || (!!familiesError && !families.length))} options={view === 'library' ? localFamilyOptions : familyOptions} onValueChange={family => updateParams({ family })}/>
      <div className="models-view-tabs ui-segmented" role="tablist" aria-label={text('模型管理视图', 'Model management views')}>{tabs.map(tab => <button key={tab.key} type="button" role="tab" aria-selected={view === tab.key} onClick={() => updateParams({ view: tab.key })}>{tab.label}</button>)}<SlidingIndicator className="ui-segmented-thumb"/></div>
    </ModelToolbar>
    {error && <div role="alert" className="settings-alert">{error}<button className={secondary} onClick={() => void refresh()}>{text('重试', 'Retry')}</button></div>}
    {Object.keys(loadErrors).length > 0 && <div role="alert" className="settings-alert"><div>{Object.entries(loadErrors).map(([key, message]) => <p key={key}>{({ library: text('本地模型', 'Local models'), downloads: text('下载记录', 'Downloads'), settings: text('存储设置', 'Storage settings'), prepare: text('推荐模型', 'Recommended models') })[key as Resource]}：{message}{' '}<button className="ui-link" onClick={() => void queries[key as Resource].refetch()}>{text('重新读取', 'Reload')}</button></p>)}</div></div>}
    {notice && <p className="model-notice" role="status">{notice}</p>}
    {view === 'prepare' && listsSettled && standaloneTasks.length > 0 && <section className="model-download-inline" aria-label={text('下载状态', 'Download status')}><h3>{text('下载状态', 'Download status')}</h3>{standaloneTasks.map(downloadTaskRow)}</section>}
    {familiesError && <div role="alert" className="settings-alert">{text('无法读取支持的模型系列。', 'Could not load supported model families.')}<button className={secondary} onClick={() => void refreshFamilies()}>{text('重试', 'Retry')}</button></div>}
    {family === 'flux' && <p role="alert" className="settings-alert">{text('FLUX.1 已停用。已有模型文件保持原样，请选择受支持的模型系列。', 'FLUX.1 is retired. Existing model files are preserved; choose a supported model family.')}</p>}
    {view === 'prepare' && familiesLoading ? <LoadingNote block className="model-empty" label={text('正在读取模型系列…', 'Loading model families…')}/> : view === 'prepare' && !selectedFamily ? <p className="model-empty">{text('请选择训练服务支持的模型系列。', 'Choose a model family supported by the training service.')}</p> : loadErrors[view] && !viewReady ? <p className="model-empty">{text('此列表暂时无法读取，请重试。其他视图仍可查看。', 'This list is unavailable. Retry or open another view.')}</p> : !viewReady ? <LoadingNote block className="model-empty" label={view === 'prepare' ? text('正在读取推荐模型…', 'Loading recommended models…') : text('正在读取本地模型…', 'Loading local models…')}/> : view === 'prepare' ? <>
      {family === 'toy' ? <p className="model-empty">{text('Toy 测试模型已内置，无需下载权重。', 'The Toy test model is built in; no weights required.')}</p> : <>
        <div className="models-source-bar"><label>{text('下载来源', 'Download source')}<StudioSelect value={catalogProvider} aria-label={text('下载来源', 'Download source')} data-testid="model-provider" options={(catalogProviders.length ? catalogProviders : ['huggingface', 'modelscope']).map(value => ({ value, label: value === 'huggingface' ? 'Hugging Face' : '魔搭 ModelScope' }))} onValueChange={value => setProvider(value as Provider)}/></label>{loaded.library && <span>{text('已添加', 'Added')} {availableRequired.length} / {required.length}</span>}<button className={primary} disabled={busy || !loaded.downloads || !loaded.library || !!loadErrors.library || !!loadErrors.downloads || missing.length === 0 || missing.every(entry => taskFor(entry) || (!entry.available_path && !entry.sources.some(source => source.provider === catalogProvider)))} onClick={() => void action(async () => { for (const entry of missing) if (!taskFor(entry) && (entry.available_path || entry.sources.some(source => source.provider === catalogProvider))) await startEntry(entry); })}><Download size={14}/>{text('下载缺失组件', 'Download missing components')}</button></div>
        {entries.length === 0 && <p className="model-help-text">{text('当前系列暂无推荐模型，可通过自定义下载添加。', 'No recommended models are available for this family. Use Custom download.')}</p>}
        {kinds.filter(kind => entries.some(entry => entry.kind === kind)).map(kind => <ModelCatalogSection key={kind} testId={`model-component-${kind}`} heading={<>{label(kind)}{!required.includes(kind) && <span className="model-tag">{text('可选', 'Optional')}</span>}</>}>
          {entries.filter(entry => entry.kind === kind).map(entry => {
            const task = taskFor(entry); const source = entry.sources.find(source => source.provider === catalogProvider);
            return <ModelCatalogRow key={entry.id} name={entry.name} metadata={<><span>{formatBytes(entry.size)}</span>{entry.recommended && <span className="model-tag">{text('推荐', 'Recommended')}</span>}<a className="ui-link" href={(source || entry.sources[0])?.url} target="_blank" rel="noreferrer">{text('发布页', 'Source')}<ExternalLink size={11}/></a></>} action={task ? isActive(task) ? cancelButton(task) : <button className={secondary} disabled={busy} onClick={() => retryTask(task)}>{text('重试', 'Retry')}</button> : entry.available_path ? <span className="model-ready"><Check size={14}/>{addedLabel(entry)}</span> : <button className={primary} disabled={busy || !loaded.downloads || !loaded.library || !source || !!loadErrors.library || !!loadErrors.downloads} onClick={() => void action(async () => { await startEntry(entry); })}><Download size={14}/>{text('下载', 'Download')}</button>}>
              {task && (isActive(task) ? <><p className="model-transfer-status">{transfer(task)}</p><ProgressBar className="model-download-progress" label={`${entry.name} ${text('下载进度', 'download progress')}`} max={task.total_bytes || undefined} value={task.total_bytes ? task.downloaded_bytes : undefined}/></> : task.status === 'failed' ? <p role="alert" className="model-download-error">{text('下载失败', 'Download failed')}：{task.error || text('未返回详细错误，请重试。', 'No detailed error was returned. Retry the download.')}</p> : <p role="status" className="model-transfer-status">{text('下载已取消，可以重新下载。', 'Download cancelled. You can retry it.')}</p>)}
            </ModelCatalogRow>;
          })}

        </ModelCatalogSection>)}
        <div className="model-actions"><button className={secondary} data-testid="download-model-btn" onClick={() => openForm('download')}><Plus size={14}/>{text('自定义下载', 'Custom download')}</button></div>
      </>}
    </> : <>
      <div className="models-list-toolbar"><label className="model-search"><Search size={15}/><input value={query} aria-label={text('搜索模型', 'Search models')} placeholder={text('搜索名称或路径', 'Search name or path')} onChange={event => { setQuery(event.target.value); setPage(1); }}/></label><div className="model-actions"><button className={secondary} data-testid="add-model-btn" disabled={busy || !selectedFamily} onClick={() => openForm('local')}><Plus size={14}/>{text('本地文件', 'Local file')}</button><button className={secondary} disabled={busy || !selectedFamily} onClick={() => void action(async () => { const found = await apiClient.post<ModelAsset[]>('/models/scan', { family }); found.forEach(model => acceptModel(model)); setNotice(`${text('新登记文件', 'Newly registered files')}：${found.length}。${text('未识别文件请使用“本地文件”逐个确认。','Review unidentified files through Local file.')}`); })}><FolderSearch size={14}/>{text('扫描模型目录', 'Scan model directory')}</button></div></div>
      <div className="models-pagination"><span>{count} {text('项', 'items')}</span><button className={secondary} disabled={currentPage === 1} onClick={() => setPage(currentPage - 1)}>{text('上一页', 'Previous')}</button><span>{currentPage} / {pages}</span><button className={secondary} disabled={currentPage === pages} onClick={() => setPage(currentPage + 1)}>{text('下一页', 'Next')}</button></div>
      {visibleLibraryKinds.length === 0 ? <p className="model-empty">{text('没有匹配的记录。', 'No matching records.')}</p> : <div className="model-library-list">{visibleLibraryKinds.map(component => <ModelCatalogSection key={component} testId={`model-library-component-${component}`} heading={label(component)} action={kinds.includes(component) && <button type="button" className={secondary} disabled={busy || !selectedFamily} aria-label={`${text('导入', 'Import')} ${label(component)}`} onClick={() => openForm('local', component)}><Plus size={14}/>{text('导入文件', 'Import file')}</button>}>
        {!pageItems.some(item => libraryKind(item) === component) && <p className="model-component-empty">{text('尚未添加', 'No models added')}</p>}
        {pageItems.filter(item => libraryKind(item) === component).map(item => {
          if (item.type === 'discovered') {
            const entry = item.entry;
            return <ModelCatalogRow className="model-library-row" key={`discovered-${entry.id}`} name={basename(entry.available_path!)} nameTitle={entry.available_path!} metadata={<><span>{entry.dtype || '—'}</span><span>{text('待检查', 'Not checked')}</span></>} action={<button type="button" className={secondary} disabled={busy || !selectedFamily} onClick={() => void action(async () => { await startEntry(entry); })}>{text('检查并登记', 'Check and register')}</button>}/>;
          }
          const model = item.model, unsupported = modelAssetUnsupportedReason(model), inference = (model as ModelAsset & { purpose?: string }).purpose === 'inference';
          return <ModelCatalogRow className="model-library-row" key={model.id} name={basename(model.path)} nameTitle={model.path} metadata={<>{!selectedFamily && <span>{model.family}</span>}<span>{model.dtype || '—'}</span><span>{formatBytes(model.size)}</span><span className={model.exists && !unsupported ? 'model-ready' : undefined}>{unsupported ? text('已停用', 'Retired') : model.exists ? text('可用', 'Available') : text('文件缺失', 'Missing file')}</span>{inference && <span>{text('仅采样', 'Sampling only')}</span>}{model.is_default && <span className="model-tag">{text('默认', 'Default')}</span>}</>} action={<div className="model-actions"><button type="button" className={secondary} disabled={busy || !selectedFamily || inference || !model.is_default && (!model.exists || !!unsupported)} aria-label={`${model.is_default ? text('取消默认', 'Clear default') : text('设为默认', 'Set default')} ${basename(model.path)}`} onClick={() => void action(async () => { await setModelDefault(model.id, !model.is_default); })}><Star size={14} fill={model.is_default ? 'currentColor' : 'none'}/>{model.is_default ? text('取消默认', 'Clear default') : text('设为默认', 'Set default')}</button><button type="button" className={`${secondary} ui-btn-icon ui-btn-danger`} disabled={busy} onClick={() => setRemove(model)} aria-label={`${text('移除登记', 'Remove registration')} ${basename(model.path)}`}><Trash2 size={14}/></button></div>}>
            {unsupported && <p className="model-help-text">{unsupported}</p>}
          </ModelCatalogRow>;
        })}
      </ModelCatalogSection>)}</div>}

    </>}

    {dialog && <Dialog title={dialog === 'local' ? text('添加本地模型', 'Add local model') : text('自定义下载', 'Custom download')} onClose={closeForm} closeDisabled={busy} wide>{dialog === 'local' ? <LocalModelRegistration initialFamily={family} initialKind={kind} families={families} onClose={closeForm} onBusyChange={setBusy} onRegistered={async(family)=>{window.dispatchEvent(new Event('studio-models-changed'));setDialog(null);updateParams({view:'library',family});setNotice(text('已登记模型。','Model registered.'));}}/> : <form className="model-source-form" data-testid="download-model-form" onSubmit={event => { event.preventDefault(); if(family==='krea2'&&kind==='dit'&&!variant){setFormError(text('请确认 Raw 或 Turbo。','Confirm Raw or Turbo.'));return;} void action(async () => {
      acceptDownload(await apiClient.post<ModelDownload>('/models/downloads', { family, kind, provider, dtype: dtype || null, is_default: variant === 'turbo' ? false : isDefault, ...(family === 'krea2' && kind === 'dit' ? {variant, purpose: variant === 'turbo' ? 'inference' : 'training'} : {}), ...(sourceMode === 'url' ? { url: url.trim() } : { repo_id: repo.trim(), filename: filename.trim(), revision: revision.trim() || (provider === 'modelscope' ? 'master' : 'main') }) } as ModelDownloadRequest));
      setDialog(null); updateParams({ view: 'prepare' });
    }, true); }}>
      <fieldset disabled={busy}><div className="model-form-intro"><p className="model-help-text">{text('从对应平台填写仓库和完整文件路径，或粘贴单文件链接。', 'Enter a repository and complete file path on the selected platform, or paste a single-file URL.')}</p><Switch className="model-switch" checked={variant==='turbo'?false:isDefault} disabled={variant==='turbo'} onCheckedChange={setIsDefault}>{text('完成后设为本系列默认组件', 'Set as the default component when ready')}</Switch></div>{family==='krea2'&&kind==='dit'&&<label>{text('Krea 2 版本 / 用途','Krea 2 variant / purpose')}<StudioSelect aria-label={text('Krea 2 版本 / 用途','Krea 2 variant / purpose')} value={variant} onValueChange={value=>{setVariant(value);if(value==='turbo')setIsDefault(false);}} placeholder={text('请按发布说明选择','Choose from publisher description')} options={[{value:'raw',label:text('Raw (训练与采样)','Raw')},{value:'turbo',label:text('Turbo (仅采样)','Turbo')}]}/></label>}
        <div className="model-form-grid"><label>{text('组件', 'Component')}<StudioSelect data-testid="model-kind-select" aria-label={text('组件', 'Component')} value={kind} onValueChange={value => { setKind(value); setVariant(''); }} options={weights.filter(weight => weight.kind !== 'tokenizer' && weight.downloadable).map(weight => ({ value: weight.kind, label: label(weight.kind) }))}/></label><label>{text('权重精度', 'Weight precision')}<StudioSelect aria-label={text('权重精度', 'Weight precision')} value={dtype} onValueChange={setDtype} options={['bf16', 'fp16', 'fp32', 'fp8', ''].map(value => ({ value, label: value.toUpperCase() || text('未知', 'Unknown') }))}/></label></div>
        < >
          <div className="model-form-grid"><label>{text('下载平台', 'Download platform')}<StudioSelect aria-label={text('下载平台', 'Download platform')} value={provider} onValueChange={value => { setProvider(value as Provider); setRevision(''); }} options={[{ value: 'huggingface', label: 'Hugging Face' }, { value: 'modelscope', label: '魔搭 ModelScope' }]}/></label><label>{text('来源格式', 'Source format')}<StudioSelect aria-label={text('来源格式', 'Source format')} value={sourceMode} onValueChange={value => setSourceMode(value as 'repo' | 'url')} options={[{ value: 'repo', label: text('仓库与文件', 'Repository and file') }, { value: 'url', label: text('文件链接', 'File URL') }]}/></label></div>
          {sourceMode === 'url' ? <label>{text('文件下载链接', 'File URL')}<input required data-testid="model-download-url" value={url} onChange={event => setUrl(event.target.value)} placeholder="https://…"/></label> : <><label>{text('仓库 ID', 'Repository ID')}<input required value={repo} onChange={event => setRepo(event.target.value)} placeholder="owner/repository"/></label><label>{text('仓库内文件路径', 'File path in repository')}<input required value={filename} onChange={event => setFilename(event.target.value)} placeholder="folder/model.safetensors"/></label><label>{text('分支 / Revision', 'Branch / revision')}<input value={revision} onChange={event => setRevision(event.target.value)} placeholder={provider === 'modelscope' ? 'master' : 'main'}/></label></>}
          <p className="model-help-text">{text('仅下载完整 safetensors 单文件；分片模型请登记完整本地目录。受限仓库使用设置里保存的访问密钥。', 'Only complete safetensors files are downloaded; register a full local directory for sharded models. Restricted repositories use the access keys saved in Settings.')}</p>
        </>
      </fieldset>
      {formError && <div role="alert" className="settings-alert">{formError}</div>}
      <footer><button type="button" className={secondary} disabled={busy} onClick={closeForm}>{text('取消', 'Cancel')}</button><button className={primary} data-testid="model-download-start" disabled={busy} type="submit">{busy && <Loader2 size={14} className="animate-spin"/>}{text('开始下载', 'Start download')}</button></footer>
    </form>}</Dialog>}
    {remove && <Dialog title={text('移除模型登记', 'Remove model registration')} onClose={() => !busy && setRemove(null)} closeDisabled={busy}><div className="model-source-form"><p>{text('移除后不再显示在模型库，磁盘文件会保留。已有项目路径不会改变。', 'The entry will be removed from the library. Its file and existing project paths are retained.')}</p><code>{remove.path}</code><footer><button className={secondary} disabled={busy} onClick={() => setRemove(null)}>{text('取消', 'Cancel')}</button><button type="button" className={`${primary} ui-btn-danger`} disabled={busy} onClick={() => void action(async () => { await apiClient.delete(`/models/${remove.id}`); resources.removeModel(remove.id); setRemove(null); })}>{text('移除登记', 'Remove registration')}</button></footer></div></Dialog>}
  </div>;
}
