import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Check, Download, ExternalLink, Loader2, RefreshCw, X } from 'lucide-react';
import { apiClient, READ_TIMEOUT_MS } from '../../api/client';
import type { Settings } from '../../api/types';
import { latestTtsDownloads, ttsDownloadActive, ttsModelIssueText, ttsModelKeys, ttsModelsApi, type TtsInstalledModel, type TtsModelDownload, type TtsModelPackage } from '../../api/ttsModels';
import StudioSelect from '../../components/StudioSelect';
import ProgressBar from '../../components/ProgressBar';
import { formatBytes, formatEta } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';

const comparablePath = (path: string) => {
  const normalized = path.replace(/\\/g, '/').replace(/\/+$/, '');
  return /^[A-Za-z]:\//.test(normalized) || normalized.startsWith('//') ? normalized.toLowerCase() : normalized;
};
const isReady = (model: TtsInstalledModel) => model.ready && model.status === 'ready';

export default function SetupTtsModels({ onContinueChange }: { onContinueChange?: (ready: boolean) => void }) {
  const text = useWorkspaceText(), client = useQueryClient();
  const catalog = useQuery({ queryKey: ttsModelKeys.catalog, queryFn: ({ signal }) => ttsModelsApi.catalog(signal), retry: false });
  const installed = useQuery({ queryKey: ttsModelKeys.installed, queryFn: ({ signal }) => ttsModelsApi.installed(signal), refetchInterval: 3000, retry: false });
  const downloads = useQuery({ queryKey: ttsModelKeys.downloads, queryFn: ({ signal }) => ttsModelsApi.downloads(signal), refetchInterval: 3000, retry: false });
  const settings = useQuery({ queryKey: ['tts-model-settings'], queryFn: ({ signal }) => apiClient.get<Settings>('/settings', { silent: true, signal, timeout: READ_TIMEOUT_MS }), refetchInterval: 3000, retry: false });
  const [engine, setEngine] = React.useState('voxcpm1.5');
  const [busy, setBusy] = React.useState(false), [error, setError] = React.useState('');
  const [observedAt, setObservedAt] = React.useState(() => Date.now() / 1000);
  const pending = React.useRef(false), alive = React.useRef(true);
  React.useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const active = downloads.data?.some(ttsDownloadActive) || false;
  React.useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setObservedAt(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [active]);
  const entries = (catalog.data || []).filter(entry => entry.engine === engine);
  const tasks = latestTtsDownloads(downloads.data || []);
  const targetFor = (entry: TtsModelPackage) => settings.data?.paths.models_dir
    ? comparablePath(`${settings.data.paths.models_dir.replace(/[\\/]+$/, '')}/tts/${entry.id}/${entry.revision.slice(0, 16)}`) : null;
  const taskFor = (entry: TtsModelPackage) => tasks.find(task => task.package_id === entry.id && task.package_revision === entry.revision && comparablePath(task.target_path) === targetFor(entry));
  const installationsFor = (entry: TtsModelPackage) => (installed.data || []).filter(model => model.package_id === entry.id && model.package_revision === entry.revision);
  const readyFor = (entry: TtsModelPackage) => installationsFor(entry).find(isReady);
  const linked = new Set(entries.map(taskFor).filter((task): task is TtsModelDownload => !!task).map(task => task.id));
  const standalone = tasks.filter(task => !linked.has(task.id) && task.status !== 'completed');
  const loadErrors = [
    { label: text('语音模型包', 'Speech model packages'), query: catalog }, { label: text('本地模型', 'Local models'), query: installed },
    { label: text('下载记录', 'Downloads'), query: downloads }, { label: text('存储设置', 'Storage settings'), query: settings },
  ].filter(item => item.query.error);
  const listsReady = !!catalog.data && !!installed.data && !!downloads.data && !!settings.data && !loadErrors.length;
  const refreshing = catalog.isFetching || installed.isFetching || downloads.isFetching || settings.isFetching;
  const canContinue = listsReady && entries.some(entry => readyFor(entry)
    || tasks.some(task => task.package_id === entry.id && task.package_revision === entry.revision && ttsDownloadActive(task)));
  React.useEffect(() => { onContinueChange?.(canContinue); }, [canContinue, onContinueChange]);
  const refresh = () => Promise.all([catalog.refetch(), installed.refetch(), downloads.refetch(), settings.refetch()]);
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
  const taskStatus = (task: TtsModelDownload) => {
    if (!ttsDownloadActive(task)) return <p className={task.status === 'failed' ? 'setup-error' : 'setup-tts-transfer'} role={task.status === 'failed' ? 'alert' : 'status'}>{task.status === 'failed' ? task.error || text('下载失败，请重试。', 'Download failed. Retry the download.') : statusLabel(task)}</p>;
    const transferring = task.status === 'downloading' && task.phase === 'download';
    const recent = task.progress_at != null && observedAt - task.progress_at < 10;
    const rate = transferring && recent && Number.isFinite(task.bytes_per_second) ? task.bytes_per_second : 0;
    const eta = rate > 0 && task.eta_seconds != null && Number.isFinite(task.eta_seconds) && task.eta_seconds >= 0 ? task.eta_seconds : null;
    return <><p className="setup-tts-transfer" role="status">{phaseLabel(task)}{` · ${formatBytes(task.downloaded_bytes)} / ${formatBytes(task.total_bytes)}`}{rate > 0 && ` · ${formatBytes(rate)}/s`}{eta != null && ` · ${text('约剩', 'About')} ${formatEta(eta)}`}</p>
      {task.current_file && <code className="setup-tts-path" title={task.current_file}>{task.current_file}</code>}
      <ProgressBar className="setup-tts-progress" label={`${task.name} ${transferring ? text('下载进度', 'download progress') : phaseLabel(task)}`} max={task.total_bytes || undefined} value={transferring && task.total_bytes > 0 ? task.downloaded_bytes : undefined}/></>;
  };
  const cancelButton = (task: TtsModelDownload) => <button type="button" className="ui-btn ui-btn-sm" disabled={busy} aria-label={`${text('取消下载', 'Cancel download')} ${task.name}`} onClick={() => void action(() => ttsModelsApi.cancel(task.id))}><X size={13}/>{text('取消', 'Cancel')}</button>;
  const retryButton = (task: TtsModelDownload) => <button type="button" className="ui-btn ui-btn-sm" disabled={busy || !listsReady} aria-label={`${text('重试下载', 'Retry download')} ${task.name}`} onClick={() => void action(() => ttsModelsApi.retry(task.id))}><RefreshCw size={13}/>{text('重试', 'Retry')}</button>;
  const readyLabel = () => <span className="setup-model-status"><Check size={14}/>{text('已就绪', 'Ready')}</span>;
  return <>
    <div className="setup-model-selects"><label>{text('模型类型', 'Model family')}<StudioSelect aria-label={text('模型类型', 'Model family')} value={engine} onValueChange={value => { setEngine(value); setError(''); }} options={[{ value: 'voxcpm1.5', label: 'VoxCPM 1.5' }, { value: 'gpt-sovits-v5', label: 'GPT-SoVITS' }]}/></label>
      <label>{text('模型下载来源', 'Model source')}<StudioSelect aria-label={text('模型下载来源', 'Model source')} value="huggingface" disabled options={[{ value: 'huggingface', label: 'Hugging Face' }]} onValueChange={() => {}}/></label></div>
    {!!loadErrors.length && <div className="setup-error" role="alert"><div>{loadErrors.map(item => <p key={item.label}>{item.label}：{formatApiError(item.query.error)}</p>)}</div><button type="button" className="ui-link" disabled={refreshing} onClick={() => void refresh()}>{text('重新读取', 'Reload')}</button></div>}
    {error && <p className="setup-error" role="alert">{error}</p>}
    {(catalog.isPending || installed.isPending || downloads.isPending || settings.isPending) && !loadErrors.length && <div className="setup-resource-loading" role="status"><Loader2 size={18} className="animate-spin"/>{text('正在读取语音模型…', 'Loading speech models…')}</div>}
      <div className="setup-model-list">{entries.map(entry => {
        const task = taskFor(entry), ready = readyFor(entry);
        const unavailable = installationsFor(entry).find(model => !isReady(model));
        const recovering = !!unavailable || task?.status === 'completed';
        return <div className="setup-model-row setup-tts-model" key={entry.id} data-testid={`setup-tts-package-${entry.id}`}>
          <div className="setup-tts-description"><strong>{entry.name}</strong><small>{formatBytes(entry.size)} · {entry.license} · <a className="ui-link" href={entry.url} target="_blank" rel="noreferrer">{text('发布页', 'Source')}<ExternalLink size={11}/></a></small>
            {task && task.status !== 'completed' && taskStatus(task)}
            {!ready && unavailable && <p className="setup-error" role="alert">{unavailable.issues?.length ? unavailable.issues.map(issue => ttsModelIssueText(issue, text)).join(' · ') : text('模型文件不可用，请检查文件或重新下载。', 'Model files are unavailable. Check the files or download again.')}</p>}
            {ready && <code className="setup-tts-path" title={ready.path}>{ready.path}</code>}
            {!ready && recovering && <p className="setup-tts-transfer">{text('原目录仍存在时，请在存储设置中更换模型目录后重新下载。原文件会保留。', 'If the original folder still exists, change the model directory in Storage settings before downloading again. Existing files are preserved.')}</p>}
          </div><div className="setup-tts-actions">{task && ttsDownloadActive(task) ? cancelButton(task) : ready ? readyLabel() : task && ['failed', 'cancelled'].includes(task.status) && !recovering ? retryButton(task)
            : <button type="button" className="ui-btn ui-btn-sm" disabled={busy || !listsReady || !settings.data?.paths.models_dir || !entry.providers?.includes('huggingface')} aria-label={`${recovering ? text('重新下载', 'Download again') : text('下载', 'Download')} ${entry.name}`} onClick={() => void action(() => ttsModelsApi.start(entry.id))}><Download size={13}/>{recovering ? text('重新下载', 'Download again') : text('下载', 'Download')}</button>}</div>
        </div>;
      })}{catalog.data && !entries.length && <p className="setup-note">{text('当前服务没有此系列的可下载模型包。', 'No downloadable packages are available for this model family.')}</p>}</div>
      {!!standalone.length && <section className="setup-tts-other" aria-label={text('其他下载', 'Other downloads')}><h3>{text('其他下载', 'Other downloads')}</h3>{standalone.map(task => <div className="setup-model-row setup-tts-model" key={task.id}><div className="setup-tts-description"><strong>{task.name}</strong><code className="setup-tts-path" title={task.target_path}>{task.target_path}</code>{taskStatus(task)}</div><div className="setup-tts-actions">{ttsDownloadActive(task) ? cancelButton(task) : installed.data?.some(model => model.id === task.installed_id && isReady(model)) ? readyLabel() : ['failed', 'cancelled'].includes(task.status) ? retryButton(task) : null}</div></div>)}</section>}
    <p className="setup-note">{text('模型包包含训练与试听所需的配套资源。开始下载后即可进入下一步，任务会在后台继续。已有本地模型可稍后在项目参数中填写路径。', 'Packages include the resources needed for training and previews. Once a download starts, you can move to the next step while it continues in the background. You can enter an existing local model path in the project parameters later.')}</p>
  </>;
}
