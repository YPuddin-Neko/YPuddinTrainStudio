import React from 'react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Check, Download, ExternalLink, KeyRound, RefreshCw, Search, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Settings } from '../../api/types';
import { latestTtsDownloads, ttsDownloadActive, ttsModelIssueText, ttsModelKeys, ttsModelsApi, type TtsInstalledModel, type TtsModelDownload, type TtsModelPackage } from '../../api/ttsModels';
import StudioSelect from '../../components/StudioSelect';
import ProgressBar from '../../components/ProgressBar';
import { SlidingIndicator } from '../../components/motion';
import { LoadingNote } from '../../components/Loading';
import { formatBytes, formatEta } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { ttsEngineLabel } from '../../utils/ttsEngines';
import { useWorkspaceText } from '../../utils/workspaceText';
import './models.css';

const activeInterval = 2500;
const primary = 'ui-btn ui-btn-primary', secondary = 'ui-btn';
const comparablePath = (path: string) => {
  const normalized = path.replace(/\\/g, '/').replace(/\/+$/, '');
  return /^[A-Za-z]:\//.test(normalized) || normalized.startsWith('//') ? normalized.toLowerCase() : normalized;
};

export default function TtsModels({ embedded = false, typeSelector }: { embedded?: boolean; typeSelector?: React.ReactNode }) {
  const text = useWorkspaceText(), location = useLocation(), client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const engine = params.get('engine') === 'gpt-sovits-v5' ? 'gpt-sovits-v5' : 'voxcpm1.5';
  const view = params.get('view') === 'library' ? 'library' : 'prepare';
  const catalog = useQuery({ queryKey: ttsModelKeys.catalog, queryFn: ({ signal }) => ttsModelsApi.catalog(signal) });
  const models = useQuery({ queryKey: ttsModelKeys.installed, queryFn: ({ signal }) => ttsModelsApi.installed(signal), refetchInterval: activeInterval });
  const downloads = useQuery({ queryKey: ttsModelKeys.downloads, queryFn: ({ signal }) => ttsModelsApi.downloads(signal), refetchInterval: activeInterval });
  const settings = useQuery({ queryKey: ['tts-model-settings'], queryFn: ({ signal }) => apiClient.get<Settings>('/settings', { silent: true, signal }) });
  const [error, setError] = React.useState(''), [busy, setBusy] = React.useState(false);
  const [query, setQuery] = React.useState(''), [page, setPage] = React.useState(1);
  const [observedAt, setObservedAt] = React.useState(() => Date.now() / 1000);
  const transferring = downloads.data?.some(ttsDownloadActive) || false;
  React.useEffect(() => {
    if (!transferring) return;
    const timer = window.setInterval(() => setObservedAt(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [transferring]);
  const pending = React.useRef(false), alive = React.useRef(true);
  React.useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const refresh = () => Promise.all([catalog.refetch(), models.refetch(), downloads.refetch(), settings.refetch()]);
  const updateParams = (patch: Record<string, string>) => {
    const next = new URLSearchParams(params);
    Object.entries(patch).forEach(([key, value]) => next.set(key, value));
    setParams(next, { replace: true, state: location.state }); setPage(1); setQuery('');
  };
  const action = async (operation: () => Promise<TtsModelDownload>) => {
    if (pending.current) return;
    pending.current = true; setBusy(true); setError('');
    try {
      const task = await operation();
      client.setQueryData<TtsModelDownload[]>(ttsModelKeys.downloads, previous => [task, ...(previous || []).filter(item => item.id !== task.id)]);
      await Promise.all([client.invalidateQueries({ queryKey: ttsModelKeys.downloads }), client.invalidateQueries({ queryKey: ttsModelKeys.installed })]);
      window.dispatchEvent(new Event('studio-tts-models-changed'));
    } catch (failure) { if (alive.current) setError(formatApiError(failure)); }
    finally { pending.current = false; if (alive.current) setBusy(false); }
  };
  const statusLabel = (task: TtsModelDownload) => ({ queued: text('排队中', 'Queued'), downloading: text('下载中', 'Downloading'), verifying: text('正在校验', 'Verifying'), completed: text('下载完成', 'Download complete'), failed: text('下载失败', 'Download failed'), cancelled: text('已取消', 'Cancelled') })[task.status];
  const phaseLabel = (task: TtsModelDownload) => ({ queued: text('排队中', 'Queued'), download: text('正在下载', 'Downloading'), reuse: text('正在复用已有文件', 'Reusing verified files'), extract: text('正在解压资源', 'Extracting resources'), verify: text('正在校验文件', 'Verifying files'), publish: text('正在完成安装', 'Finishing installation'), completed: text('下载完成', 'Download complete'), failed: text('下载失败', 'Download failed'), cancelled: text('已取消', 'Cancelled') })[task.phase];
  const installationStatus = (model: TtsInstalledModel) => model.status === 'ready' && model.ready ? text('可用', 'Available') : model.status === 'missing' ? text('文件缺失', 'Missing files') : model.status === 'changed' ? text('文件已变化', 'Files changed') : text('不可用', 'Unavailable');
  const selected = (models.data || []).filter(model => model.engine === engine);
  const entries = (catalog.data || []).filter(entry => entry.engine === engine);
  const tasks = latestTtsDownloads(downloads.data || []).filter(task => task.engine === engine);
  // Download starts use the current directory; retries retain the original task's destination.
  const targetFor = (entry: TtsModelPackage) => settings.data?.paths.models_dir ? comparablePath(`${settings.data.paths.models_dir.replace(/[\\/]+$/, '')}/tts/${entry.id}/${entry.revision.slice(0, 16)}`) : null;
  const taskFor = (entry: TtsModelPackage) => tasks.find(task => task.package_id === entry.id && task.package_revision === entry.revision && comparablePath(task.target_path) === targetFor(entry));
  const linked = new Set(entries.map(taskFor).filter((task): task is TtsModelDownload => !!task).map(task => task.id));
  const standalone = tasks.filter(task => !linked.has(task.id) && task.status !== 'completed');
  const installationsFor = (entry: TtsModelPackage) => selected.filter(model => model.package_id === entry.id && model.package_revision === entry.revision);
  const readyFor = (entry: TtsModelPackage) => installationsFor(entry).find(model => model.status === 'ready' && model.ready && comparablePath(model.path) === targetFor(entry));
  const listsReady = !!models.data && !!downloads.data && !models.error && !downloads.error;
  const refreshing = [catalog, models, downloads, settings].some(item => item.isFetching);
  const loadErrors = [
    { label: text('准备模型', 'Prepare models'), query: catalog }, { label: text('本地模型', 'Local models'), query: models },
    { label: text('下载记录', 'Downloads'), query: downloads }, { label: text('存储设置', 'Storage settings'), query: settings },
  ].filter(item => item.query.error);
  const viewQuery = view === 'prepare' ? catalog : models;
  const searchValue = query.trim().toLocaleLowerCase();
  const filtered = selected.filter(model => `${model.name} ${model.path}`.toLocaleLowerCase().includes(searchValue));
  const pages = Math.max(1, Math.ceil(filtered.length / 12)), currentPage = Math.min(page, pages);
  const cancelButton = (task: TtsModelDownload) => <button type="button" className={`${secondary} model-button-downloading`} disabled={busy} aria-label={`${text('取消下载', 'Cancel download')} ${task.name}`} title={text('取消下载', 'Cancel download')} onClick={() => void action(() => ttsModelsApi.cancel(task.id))}><span className="model-download-active-label">{statusLabel(task)}</span><span className="model-download-cancel-label"><X size={13}/>{text('取消', 'Cancel')}</span></button>;
  const taskAction = (task: TtsModelDownload) => ttsDownloadActive(task) ? cancelButton(task) : task.installed_id && models.data?.some(model => model.id === task.installed_id && model.status === 'ready' && model.ready) ? <button type="button" className={secondary} onClick={() => updateParams({ view: 'library' })}>{text('查看本地模型', 'View local models')}</button> : ['failed', 'cancelled'].includes(task.status) ? <button type="button" className={secondary} disabled={busy || !listsReady} onClick={() => void action(() => ttsModelsApi.retry(task.id))}>{text('重试', 'Retry')}</button> : null;
  const startButton = (entry: TtsModelPackage, again: boolean) => <button type="button" className={primary} disabled={busy || !listsReady || !settings.data || !!settings.error || !entry.providers?.includes('huggingface')} onClick={() => void action(() => ttsModelsApi.start(entry.id))}><Download size={14}/>{again ? text('重新下载', 'Download again') : text('下载', 'Download')}</button>;
  const recoveryHelp = () => <p className="model-help-text">{text('下载到当前模型目录。原目录仍存在时，请先', 'Downloads use the current model directory. If the original folder still exists, ')}<Link className="ui-link" to="/settings/preferences?section=storage" replace state={location.state}>{text('更改模型目录', 'change the model directory')}</Link>{text('。原文件会保留。', '. Existing files are preserved.')}</p>;
  const taskStatus = (task: TtsModelDownload) => {
    if (!ttsDownloadActive(task)) return task.status === 'failed' ? <p role="alert" className="model-download-error">{task.error || text('下载失败，请重试。', 'Download failed. Retry the download.')}</p> : <p className="model-transfer-status" role="status">{statusLabel(task)}</p>;
    const transferring = task.phase === 'download' && task.status === 'downloading';
    const recent = task.progress_at != null && observedAt - task.progress_at < 10;
    const rate = transferring && recent && Number.isFinite(task.bytes_per_second) ? task.bytes_per_second : 0;
    const eta = rate > 0 && task.eta_seconds != null && Number.isFinite(task.eta_seconds) && task.eta_seconds >= 0 ? task.eta_seconds : null;
    return <><p className="model-transfer-status" role="status">{phaseLabel(task)}{` · ${formatBytes(task.downloaded_bytes)} / ${formatBytes(task.total_bytes)}`}{rate > 0 && ` · ${formatBytes(rate)}/s`}{eta != null && ` · ${text('约剩', 'About')} ${formatEta(eta)}`}</p>
      {task.current_file && <p className="tts-model-current-file" title={task.current_file}>{task.current_file}</p>}
      <ProgressBar className="model-download-progress" label={`${task.name} ${transferring ? text('下载进度', 'download progress') : phaseLabel(task)}`} max={task.total_bytes || undefined} value={transferring && task.total_bytes > 0 ? task.downloaded_bytes : undefined}/></>;
  };
  return <div className="models-workspace tts-models-workspace" data-testid="tts-models-page">
    <div className="models-toolbar"><div className="models-heading"><div><h2>{text('模型权重', 'Model weights')}</h2>{!embedded && <p>{text('准备模型组件，供项目选择。', 'Prepare components for your projects.')}</p>}</div>
      {settings.data?.paths.models_dir && <div className="models-heading-path" title={settings.data.paths.models_dir}><span>{text('模型目录', 'Model directory')}</span><strong>{settings.data.paths.models_dir}</strong><Link className="ui-link" to="/settings/preferences?section=storage" replace state={location.state}>{text('更改', 'Change')}</Link></div>}
      <div className="model-actions"><Link to="/settings/environment?tab=credentials&type=tts" replace state={location.state} className={secondary}><KeyRound size={14}/>{text('访问密钥', 'Access keys')}</Link><button type="button" className={`${secondary} ui-btn-icon`} disabled={refreshing} onClick={() => void refresh()} aria-label={text('刷新模型', 'Refresh models')} title={text('刷新模型', 'Refresh models')}><RefreshCw size={14}/></button></div>
    </div><div className="models-filters">{typeSelector}<StudioSelect aria-label={text('模型系列', 'Model family')} value={engine} options={[{ value: 'voxcpm1.5', label: 'VoxCPM 1.5' }, { value: 'gpt-sovits-v5', label: 'GPT-SoVITS' }]} onValueChange={value => updateParams({ engine: value })}/>
      <div className="models-view-tabs ui-segmented" role="tablist" aria-label={text('模型管理视图', 'Model management views')}>{[{ key: 'prepare', label: text('准备模型', 'Prepare models') }, { key: 'library', label: `${text('本地模型', 'Local models')}${models.data ? ` · ${selected.length}` : ''}` }].map(tab => <button type="button" key={tab.key} role="tab" aria-selected={view === tab.key} onClick={() => updateParams({ view: tab.key })}>{tab.label}</button>)}<SlidingIndicator className="ui-segmented-thumb"/></div>
    </div></div>
    {error && <div role="alert" className="settings-alert">{error}</div>}
    {!!loadErrors.length && <div role="alert" className="settings-alert"><div>{loadErrors.map(item => <p key={item.label}>{item.label}：{formatApiError(item.query.error)}</p>)}</div><button type="button" className={secondary} disabled={refreshing} onClick={() => void refresh()}>{text('重新读取', 'Reload')}</button></div>}
    {standalone.length > 0 && <section className="model-download-inline" aria-label={text('下载状态', 'Download status')}><h3>{text('下载状态', 'Download status')}</h3>{standalone.map(task => <div className="model-download-inline-row" key={task.id}><div className="model-download-heading"><div><strong>{task.name}</strong><p className="model-help-text">{task.target_path}</p></div><div className="model-actions">{taskAction(task)}</div></div>{taskStatus(task)}</div>)}</section>}
    {viewQuery.error ? <p className="model-empty">{text('此列表暂时无法读取，请重试。其他视图仍可查看。', 'This list is unavailable. Retry or open another view.')}</p> : viewQuery.isPending ? <LoadingNote block className="model-empty" label={view === 'prepare' ? text('正在读取模型包…', 'Loading model packages…') : text('正在读取本地模型…', 'Loading local models…')}/> : view === 'prepare' ? <>
      <div className="models-source-bar"><label>{text('下载来源', 'Download source')}<StudioSelect value="huggingface" aria-label={text('下载来源', 'Download source')} options={[{ value: 'huggingface', label: 'Hugging Face' }]} disabled onValueChange={() => {}}/></label><span>{text('包含训练与试听所需的模型资源', 'Includes model resources for training and previews')}</span></div>
      {entries.length ? <section className="model-component" aria-label={text('完整模型包', 'Complete model packages')}><header><h3>{text('完整模型包', 'Complete model packages')}</h3></header>{entries.map(entry => {
        const task = taskFor(entry), ready = readyFor(entry), installations = installationsFor(entry);
        const unavailable = installations.find(model => !model.ready || model.status !== 'ready');
        const elsewhere = installations.find(model => model.status === 'ready' && model.ready && comparablePath(model.path) !== targetFor(entry));
        const recovering = !!unavailable || task?.status === 'completed';
        return <div className="model-catalog-row" key={entry.id} data-testid={`tts-model-package-${entry.id}`}><div className="model-catalog-description"><strong>{entry.name}</strong><div><span>{formatBytes(entry.size)}</span><span>{entry.license}</span><a className="ui-link" href={entry.url} target="_blank" rel="noreferrer">{text('发布页', 'Source')}<ExternalLink size={11}/></a></div>
          {task && task.status !== 'completed' && taskStatus(task)}
          {unavailable && !ready && <p className="model-download-error">{installationStatus(unavailable)}{unavailable.issues?.map(issue => ` · ${ttsModelIssueText(issue, text)}`).join('')}</p>}
          {!ready && elsewhere && <p className="model-help-text">{text('其他目录已有可用模型，仍可在项目中选择。', 'A model in another directory remains available for projects.')}</p>}
          {!ready && recovering && recoveryHelp()}
          {!ready && (installations.length > 0 || task?.status === 'completed') && <button type="button" className="ui-link" onClick={() => updateParams({ view: 'library' })}>{text('查看本地模型', 'View local models')}</button>}
        </div><div className="model-catalog-action">{task && ttsDownloadActive(task) ? cancelButton(task) : ready ? <span className="model-ready"><Check size={14}/>{text('已就绪', 'Ready')}</span> : task && ['failed', 'cancelled'].includes(task.status) && !recovering ? taskAction(task) : startButton(entry, recovering)}</div></div>;
      })}</section> : <p className="model-empty">{text('当前服务没有此系列的可下载模型包。', 'No downloadable packages are available for this model family.')}</p>}
      <p className="model-help-text">{text('已有模型可在项目参数中填写本地路径。Python 与训练器路径也在项目内配置。', 'Enter existing local model, Python and trainer paths in the project parameters.')}</p>
    </> : <><div className="models-list-toolbar"><label className="model-search"><Search size={15}/><input value={query} aria-label={text('搜索模型', 'Search models')} placeholder={text('搜索名称或路径', 'Search name or path')} onChange={event => { setQuery(event.target.value); setPage(1); }}/></label></div>
      <div className="models-pagination"><span>{filtered.length} {text('项', 'items')}</span><button type="button" className={secondary} disabled={currentPage === 1} onClick={() => setPage(currentPage - 1)}>{text('上一页', 'Previous')}</button><span>{currentPage} / {pages}</span><button type="button" className={secondary} disabled={currentPage === pages} onClick={() => setPage(currentPage + 1)}>{text('下一页', 'Next')}</button></div>
      {!filtered.length ? <p className="model-empty">{query ? text('没有匹配的记录。', 'No matching records.') : text('尚无已下载的模型包，可在“准备模型”中下载，或在项目参数中填写本地路径。', 'No model packages are installed. Download one in Prepare models or enter a local path in the project parameters.')}</p> : <div className="model-library-list">{filtered.slice((currentPage - 1) * 12, currentPage * 12).map(model => {
        const ready = model.status === 'ready' && model.ready;
        const entry = entries.find(item => item.id === model.package_id);
        const task = entry ? taskFor(entry) : undefined, currentReady = entry ? readyFor(entry) : undefined;
        return <div className="model-library-row" key={model.id}><div><strong>{model.name}</strong><p>{ttsEngineLabel(model.engine)}{model.variant && ` · ${model.variant}`} · {installationStatus(model)}</p>{model.issues?.map((issue, index) => <p className="model-download-error" key={`${issue.code}-${index}`}>{ttsModelIssueText(issue, text)}</p>)}<details><summary>{text('文件路径', 'File path')}</summary><code>{model.path}</code></details>
          {!ready && entry && !currentReady && recoveryHelp()}
          {!ready && task && ttsDownloadActive(task) && taskStatus(task)}
          {!ready && !entry && !catalog.isPending && <p className="model-help-text">{text('此模型包暂不提供下载，可检查文件路径或选择其他模型。', 'This package is not currently available to download. Check its file path or choose another model.')}</p>}
        </div><div className="model-actions">{ready ? <span className="model-ready"><Check size={14}/>{text('可在项目中选择', 'Available in projects')}</span> : task && ttsDownloadActive(task) ? cancelButton(task) : currentReady ? <span className="model-ready"><Check size={14}/>{text('当前目录已有可用模型', 'A model is available in the current directory')}</span> : entry ? startButton(entry, true) : null}</div></div>;
      })}</div>}
    </>}
  </div>;
}
