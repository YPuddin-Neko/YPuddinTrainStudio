import React from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { invalidateResource, useResourceCacheEvents, useResourceQuery } from '../../api/resourcePolicy';
import { readCredentialStatus } from '../../api/hooks/useCredentialsStatus';
import { useImageModelResources } from '../../api/hooks/useImageModelResources';
import { Check, Download, ExternalLink, Loader2 } from 'lucide-react';
import { apiClient, READ_TIMEOUT_MS } from '../../api/client';
import type { Settings } from '../../api/types';
import type { EnvironmentStatus } from '../../components/EnvironmentManagerPanel';
import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { modelAssetUnsupportedReason, trainingFamilyOptions } from '../../utils/trainingFamilies';
import SetupTtsModels from './SetupTtsModels';
import TtsEnvironmentPanel from '../../components/TtsEnvironmentPanel';

const AttentionSetup = React.lazy(() => import('../../components/EnvironmentManagerPanel').then(module => ({ default: module.EnvironmentManagerPanel })));

type Paths = Record<string, { path: string }>;
function useResource<T>(path: string) {
  const client = useQueryClient();
  useResourceCacheEvents();
  const queryKey = path === '/families' ? ['families'] : path === '/environment' ? ['environment', path] : path === '/credentials' ? ['credentials'] : ['setup-resource', path];
  const query = useResourceQuery<T>({ queryKey,
    queryFn: ({ signal }) => path === '/credentials' ? readCredentialStatus(signal) as Promise<T> : apiClient.get<T>(path, { silent: true, signal, timeout: READ_TIMEOUT_MS }),
  });
  React.useEffect(() => {
    const event = path === '/settings/storage-defaults' ? 'studio.settings.changed' : null;
    if (!event) return;
    const changed = () => invalidateResource(client, path === '/families' ? ['families'] : path === '/environment' ? ['environment', path] : path === '/credentials' ? ['credentials'] : ['setup-resource', path]);
    window.addEventListener(event, changed);
    return () => window.removeEventListener(event, changed);
  }, [client, path]);
  return { data: query.data ?? null, error: query.error ? formatApiError(query.error) : '', reload: () => void query.refetch(),
    setData: (data: T) => { void client.cancelQueries({ queryKey, exact: true }); client.setQueryData(queryKey, data); } };
}
function ResourceState({ error, retry, label }: { error: string; retry: () => void; label?: string }) {
  const text = useWorkspaceText();
  return error ? <div role="alert" className="setup-error">{error}<button className="ui-link" onClick={retry}>{text('重试', 'Retry')}</button></div>
    : <div className="setup-resource-loading" role="status"><Loader2 size={18} className="animate-spin"/>{label ?? text('正在读取…', 'Loading…')}</div>;
}
export function StorageStep({ settings }: { settings: Settings }) {
  const text = useWorkspaceText();
  const { data, error, reload } = useResource<Paths>('/settings/storage-defaults');
  return <><dl className="setup-paths">{([
    ['data_root', text('数据目录', 'Data directory'), text('项目、数据集与任务记录', 'Projects, datasets and job history')],
    ['models_dir', text('模型目录', 'Model directory'), text('下载的基础模型与组件', 'Downloaded base models and components')],
    ['cache_dir', text('缓存目录', 'Cache directory'), text('训练缓存、缩略图与安装包', 'Training caches, thumbnails and packages')],
    ['output_dir', text('训练产物', 'Training outputs'), text('按项目、版本和任务分别保存', 'Organized by project, version and job')],
  ] as const).map(([key, label, hint]) => <div key={key}><dt>{label}<span>{hint}</span></dt><dd>{key === 'output_dir' && settings.paths.output_mode !== 'custom'
    ? data?.output_dir.path ?? '…' : settings.paths[key]}</dd></div>)}</dl>
    {!data && <ResourceState error={error} retry={reload}/>}
    <p className="setup-note">{text('这些目录位于运行训练器的机器上，可稍后在设置中修改。', 'These folders are on the machine running Studio. You can change them later in Settings.')}</p></>;
}

type Keys = Record<string, { configured: boolean }>;
export type KeyDraft = { huggingface: string; modelscope: string };
export function KeysStep({ draft, onChange, disabled }: { draft: KeyDraft; onChange: (draft: KeyDraft) => void; disabled: boolean }) {
  const text = useWorkspaceText();
  const { data, error, reload } = useResource<Keys>('/credentials');
  return <><div className="setup-key-fields">{([
    ['huggingface', 'Hugging Face', 'https://huggingface.co/settings/tokens'],
    ['modelscope', 'ModelScope', 'https://modelscope.cn/my/myaccesstoken'],
  ] as const).map(([key, name, url]) => <div className="setup-key-field" key={key}>
    <div><label htmlFor={`setup-${key}`}>{name}</label><span>{data ? data[key]?.configured ? text('已配置', 'Configured') : text('可选', 'Optional') : error ? text('状态未知', 'Status unavailable') : text('读取状态中', 'Loading status')}</span></div>
    <input id={`setup-${key}`} type="password" autoComplete="new-password" spellCheck={false} maxLength={4096} className="settings-input" disabled={disabled} value={draft[key]} onChange={event => onChange({ ...draft, [key]: event.target.value })} placeholder={data?.[key]?.configured ? text('留空保留现有密钥', 'Leave blank to keep the current key') : text('输入访问令牌', 'Enter an access token')}/>
    <a href={url} target="_blank" rel="noreferrer" className="ui-link">{text('获取访问令牌', 'Get an access token')}<ExternalLink size={12}/></a>
  </div>)}</div>{error && <ResourceState error={text('无法读取密钥配置状态。', 'Could not load access-key status.')} retry={reload}/>}
    <p className="setup-note">{text('公开模型通常无需密钥；受限模型需先在对应站点申请访问权限。', 'Public models usually need no key. Gated models require access approval on their hosting site.')}</p></>;
}

type Recommendation = import('../../api/generated').components['schemas']['RecommendedModel'];
type DownloadTask = import('../../api/types').ModelDownload;
type ModelAsset = import('../../api/types').ModelAsset;
type ModelSnapshot = { catalog: Recommendation[]; assets: ModelAsset[]; tasks: DownloadTask[] };
type ModelList = keyof ModelSnapshot;
/** A list that could not be read, so the step can say which one. */
class ListFailure extends Error {
  constructor(readonly list: ModelList, reason: unknown) { super(formatApiError(reason)); }
}
const activeDownload = (task: DownloadTask) => ['queued', 'downloading'].includes(task.status);
const kleinEncoders: Record<string, string> = { 'flux2-klein-base-4b': 'flux2-qwen3-4b', 'flux2-klein-base-9b': 'flux2-qwen3-8b' };
const matchingAsset = (entry: Recommendation, asset: ModelAsset) => asset.family === entry.family && asset.kind === entry.kind
  && (asset.id === entry.model_id || asset.path === entry.available_path);
function modelBundle(catalog: Recommendation[], family: string, main: string) {
  return catalog.filter(entry => entry.family === family && entry.recommended && entry.purpose !== 'inference'
    && (entry.kind !== 'dit' || entry.id === main)
    && (family !== 'flux2' || entry.kind !== 'text_encoder' || entry.id === kleinEncoders[main]));
}
function latestModelTasks(tasks: DownloadTask[]) {
  const latest = new Map<string, DownloadTask>();
  for (const task of [...tasks].sort((a, b) => b.created_at - a.created_at)) {
    if (task.recommendation_id && !latest.has(task.recommendation_id)) latest.set(task.recommendation_id, task);
  }
  return latest;
}
export function ModelsStep({ onContinueChange }: { onContinueChange?: (ready: boolean) => void } = {}) {
  const text = useWorkspaceText();
  const [type, setType] = React.useState('image');
  const currentType = React.useRef(type);
  const imageContinue = React.useCallback((ready: boolean) => { if (currentType.current === 'image') onContinueChange?.(ready); }, [onContinueChange]);
  const ttsContinue = React.useCallback((ready: boolean) => { if (currentType.current === 'tts') onContinueChange?.(ready); }, [onContinueChange]);
  return <><label className="setup-model-type">{text('训练类型', 'Training type')}<StudioSelect aria-label={text('训练类型', 'Training type')} value={type}
    options={[{ value: 'image', label: text('图像', 'Image') }, { value: 'tts', label: text('语音', 'Speech') }]}
    onValueChange={value => { currentType.current = value; onContinueChange?.(false); setType(value); }}/></label>
    {type === 'tts' ? <SetupTtsModels onContinueChange={ttsContinue}/> : <ImageModelsStep onContinueChange={imageContinue}/>}</>;
}
function ImageModelsStep({ onContinueChange }: { onContinueChange?: (ready: boolean) => void }) {
  const text = useWorkspaceText();
  const families = useResource<import('../../api/types').FamilyInfo[]>('/families');
  const [family, setFamily] = React.useState('anima');
  const [main, setMain] = React.useState('');
  const [preparedPackage, setPreparedPackage] = React.useState('');
  const [provider, setProvider] = React.useState<'huggingface' | 'modelscope'>('huggingface');
  const [busy, setBusy] = React.useState(false);
  const busyRef = React.useRef(false);
  const [error, setError] = React.useState('');
  const resources = useImageModelResources(!busy);
  const snapshot: ModelSnapshot = { catalog: resources.catalog.data ?? [], assets: resources.assets.data ?? [], tasks: resources.tasks.data ?? [] };
  const loaded = { catalog: resources.catalog.data !== undefined, assets: resources.assets.data !== undefined, tasks: resources.tasks.data !== undefined };
  const loadErrors = Object.fromEntries((['catalog', 'assets', 'tasks'] as const).map(name => [name, resources[name].error ? formatApiError(resources[name].error) : ''])) as Record<ModelList, string>;
  const reload = () => { void resources.refresh(); };
  const prepareController = React.useRef<AbortController | null>(null);
  const [selected, setSelected] = React.useState<string[] | null>(null);
  const read = async (signal?: AbortSignal): Promise<ModelSnapshot> => {
    const list = <T,>(name: ModelList, path: string) => apiClient.get<T>(path, { silent: true, signal, timeout: READ_TIMEOUT_MS })
      .catch((failure: unknown) => { throw failure instanceof Error && failure.name === 'AbortError' ? failure : new ListFailure(name, failure); });
    const results = await Promise.allSettled([
      list<Recommendation[]>('catalog', '/models/recommendations'), list<ModelAsset[]>('assets', '/models'), resources.readTasks(signal).catch(failure => { throw failure instanceof Error && failure.name === 'AbortError' ? failure : new ListFailure('tasks', failure); }),
    ]);
    const failure = results.find(result => result.status === 'rejected');
    if (failure?.status === 'rejected') throw failure.reason;
    return { catalog: (results[0] as PromiseFulfilledResult<Recommendation[]>).value, assets: (results[1] as PromiseFulfilledResult<ModelAsset[]>).value, tasks: (results[2] as PromiseFulfilledResult<DownloadTask[]>).value };
  };
  React.useEffect(() => () => prepareController.current?.abort(), []);
  const loadError = Object.values(loadErrors).some(Boolean);
  const listsReady = loaded.catalog && loaded.assets && loaded.tasks;
  const availableFamilies = (families.data ? trainingFamilyOptions(families.data) : [])
    .filter(option => snapshot?.catalog.some(entry => entry.family === option.value && entry.recommended && entry.purpose !== 'inference'));
  const currentFamily = availableFamilies.some(option => option.value === family) ? family : availableFamilies[0]?.value ?? family;
  const mains = snapshot?.catalog.filter(entry => entry.family === currentFamily && entry.kind === 'dit' && entry.recommended && entry.purpose !== 'inference') ?? [];
  const latest = latestModelTasks(snapshot?.tasks ?? []);
  const currentMain = mains.some(entry => entry.id === main) ? main
    : mains.find(entry => entry.is_default || snapshot?.assets.some(asset => asset.is_default && asset.exists && matchingAsset(entry, asset)))?.id
      ?? mains.find(entry => { const task = latest.get(entry.id); return task && activeDownload(task); })?.id ?? mains[0]?.id ?? '';
  const entries = modelBundle(snapshot?.catalog ?? [], currentFamily, currentMain);
  const validAsset = (asset: ModelAsset) => asset.exists && asset.purpose !== 'inference' && !modelAssetUnsupportedReason(asset);
  const canSetDefaults = (value: ModelSnapshot, bundle: Recommendation[]) => value.assets.filter(asset => validAsset(asset) && asset.is_default && asset.family === currentFamily)
    .every(asset => bundle.some(entry => matchingAsset(entry, asset)))
    && !value.tasks.some(task => task.family === currentFamily && task.is_default && activeDownload(task) && !bundle.some(entry => entry.id === task.recommendation_id));
  const setDefaults = snapshot ? canSetDefaults(snapshot, entries) : false;
  const needsPreparation = (entry: Recommendation, value: ModelSnapshot, defaults: boolean) => {
    if (value.tasks.some(task => activeDownload(task) && task.recommendation_id === entry.id)) return false;
    if (!entry.available_path) return true;
    return defaults && !entry.is_default;
  };
  const chosen = selected ?? entries.filter(entry => snapshot && needsPreparation(entry, snapshot, setDefaults)).map(entry => entry.id);
  const familyDownloading = snapshot?.tasks.some(task => task.family === currentFamily && activeDownload(task));
  const packageKey = `${currentFamily}/${currentMain}`;
  const completePackage = entries.length > 0 && entries.every(entry => snapshot?.assets.some(asset => validAsset(asset) && matchingAsset(entry, asset)));
  const startedPackage = entries.some(entry => snapshot?.tasks.some(task => task.recommendation_id === entry.id && activeDownload(task)));
  const continueReady = !!snapshot && !!listsReady && !loadError && (completePackage || startedPackage || preparedPackage === packageKey);
  React.useEffect(() => { onContinueChange?.(continueReady); }, [continueReady, onContinueChange]);
  const prepare = async (ids: string[]) => {
    if (busyRef.current) return;
    busyRef.current = true; void resources.cancelReads();
    const controller = new AbortController(); prepareController.current = controller;
    setBusy(true); setError('');
    try {
      // Re-read before submission: another page may have started a download or changed the defaults.
      let current = await read(controller.signal);
      if (controller.signal.aborted) return;
      current = resources.acceptSnapshot(current);
      for (const id of ids) {
        const bundle = modelBundle(current.catalog, currentFamily, currentMain);
        const entry = bundle.find(item => item.id === id);
        const defaults = canSetDefaults(current, bundle);
        if (!entry || !needsPreparation(entry, current, defaults)) continue;
        if (entry.available_path) {
          const model = await apiClient.post<ModelAsset>(`/models/recommendations/${encodeURIComponent(entry.id)}/use`, {}, { silent: true });
          current = { ...current,
            assets: [...current.assets.filter(item => item.id !== model.id).map(item => model.is_default && item.family === model.family && item.kind === model.kind ? { ...item, is_default: false } : item), model],
            catalog: current.catalog.map(item => item.id === entry.id ? { ...item, model_id: model.id, available_path: model.path, is_default: model.is_default }
              : model.is_default && item.family === model.family && item.kind === model.kind ? { ...item, is_default: false } : item),
          };
        } else {
          if (!entry.sources.some(source => source.provider === provider)) continue;
          // A fresh attempt uses today's default policy; old retries can carry a stale is_default flag.
          const task = await apiClient.post<DownloadTask>(`/models/recommendations/${encodeURIComponent(entry.id)}/download`, { provider, is_default: defaults }, { silent: true });
          current = { ...current, tasks: [task, ...current.tasks.filter(item => item.id !== task.id)] };
        }
        if (controller.signal.aborted) return;
        setPreparedPackage(packageKey);
        current = resources.acceptSnapshot(current);
      }
    } catch (failure) { if (!controller.signal.aborted) setError(formatApiError(failure)); }
    finally { busyRef.current = false; if (!controller.signal.aborted) { setBusy(false); window.dispatchEvent(new Event('studio-models-changed')); } }
  };
  const listNames: Record<ModelList, string> = { catalog: text('推荐模型', 'Recommended models'), assets: text('本地模型', 'Local models'), tasks: text('下载记录', 'Downloads') };
  const loadMessage = (Object.entries(loadErrors) as [ModelList, string][]).filter(([, message]) => message).map(([list, message]) => `${listNames[list]}：${message}`).join('；');
  const familiesMessage = families.error ? `${text('模型系列', 'Model families')}：${families.error}` : '';
  if (!snapshot || !loaded.catalog || !families.data) return <>
    {loaded.assets && !!snapshot?.assets.length && <div className="setup-model-list" aria-label={text('本地模型', 'Local models')}>{snapshot.assets.map(asset => <div className="setup-model-row" key={asset.id}><span><strong>{asset.path.split(/[\\/]/).pop() || asset.id}</strong><small>{asset.family}</small></span><span className="setup-model-status">{asset.exists ? text('本地文件', 'Local file') : text('文件不存在', 'File missing')}</span></div>)}</div>}
    <ResourceState error={loadMessage || familiesMessage} retry={() => { reload(); families.reload(); }}
      label={!loaded.catalog ? text('正在读取推荐模型…', 'Loading recommended models…') : text('正在读取模型系列…', 'Loading model families…')}/></>;
  return <><div className="setup-model-selects"><label>{text('模型类型', 'Model family')}<StudioSelect aria-label={text('模型类型', 'Model family')} disabled={busy} value={currentFamily} onValueChange={value => { setFamily(value); setMain(''); setSelected(null); setError(''); }} options={availableFamilies}/></label>
    <label>{text('模型下载来源', 'Model source')}<StudioSelect aria-label={text('模型下载来源', 'Model source')} value={provider} disabled={busy} onValueChange={value => setProvider(value as typeof provider)} options={[{ value: 'huggingface', label: 'Hugging Face' }, { value: 'modelscope', label: 'ModelScope' }]}/></label>
    {mains.length > 1 && <label>{text('主模型', 'Main model')}<StudioSelect aria-label={text('主模型', 'Main model')} value={currentMain} disabled={busy || familyDownloading} onValueChange={value => { setMain(value); setSelected(null); setError(''); }} options={mains.map(entry => ({ value: entry.id, label: entry.name }))}/></label>}</div>
    <div className="setup-model-list">{entries.map(entry => {
      const task = latest.get(entry.id); const active = task && activeDownload(task);
      const local = !!entry.available_path;
      const needs = needsPreparation(entry, snapshot, setDefaults);
      const supported = local || entry.sources.some(source => source.provider === provider);
      const failed = !local && task?.status === 'failed';
      const cancelled = !local && task?.status === 'cancelled';
      return <div key={entry.id} className="setup-model-row"><input type="checkbox" aria-label={entry.name} checked={chosen.includes(entry.id)} disabled={busy || !listsReady || loadError || !needs || !supported} onChange={event => setSelected(event.target.checked ? [...chosen, entry.id] : chosen.filter(id => id !== entry.id))}/>
        <span><strong>{entry.name}</strong><small>{entry.kind === 'dit' ? text('主模型', 'Main model') : entry.kind === 'vae' ? 'VAE' : text('文本编码器', 'Text encoder')}{entry.size > 0 ? ` · ${(entry.size / 1024 ** 3).toFixed(1)} GiB` : ''}</small>
          {failed && <span role="alert" className="setup-error">{text('下载失败：', 'Download failed: ')}{task.error || text('未返回详细错误，请重试。', 'No detailed error was returned. Retry the download.')}</span>}
        </span>
        <span className="setup-model-status">{active ? <><Loader2 size={14} className="animate-spin"/>{text('下载中', 'Downloading')}</> : local ? <><Check size={14}/>{entry.is_default ? text('当前默认', 'Current default') : needs ? text('本地文件待确认', 'Local file to verify') : text('已下载', 'Downloaded')}</> : failed || cancelled ? <><span>{cancelled ? text('已取消', 'Cancelled') : ''}</span><button className="ui-link" disabled={busy || !listsReady || !supported || loadError} onClick={() => void prepare([entry.id])}>{text('重试', 'Retry')}</button></> : !supported ? text('此来源不可用', 'Unavailable here') : ''}</span></div>;
    })}{!entries.length && <p className="setup-note">{text('暂无推荐组件，可稍后在模型设置中添加。', 'No recommended components. Add models later in Settings.')}</p>}</div>
    {loadError && <ResourceState error={loadMessage} retry={reload}/>}
    {!listsReady && !loadError && <ResourceState error="" retry={reload} label={text('正在读取本地模型与下载记录…', 'Loading local models and downloads…')}/>}{error && <div role="alert" className="setup-error">{error}</div>}
    <button className="ui-btn" disabled={busy || !listsReady || loadError || !entries.some(entry => chosen.includes(entry.id) && needsPreparation(entry, snapshot, setDefaults) && (entry.available_path || entry.sources.some(source => source.provider === provider)))} onClick={() => void prepare(chosen)}>{busy ? <Loader2 size={15} className="animate-spin"/> : <Download size={15}/>} {text('准备所选组件', 'Prepare selected components')}</button>
    {!setDefaults && entries.length > 0 && <p className="setup-note">{text('保留已有默认模型。新组件下载后可在模型设置中选用。', 'Existing defaults are kept. Select the new components in Model settings after downloading.')}</p>}
    <p className="setup-note">{text('下载会在后台继续，也可稍后在模型设置中导入本地文件。', 'Downloads continue in the background. You can also import local files later in Model settings.')}</p></>;
}

export function RuntimeStep() {
  const text = useWorkspaceText();
  const { data, error, reload } = useResource<EnvironmentStatus>('/environment');
  const speech = <section className="setup-tts-environments"><div className="setup-runtime-heading"><span>{text('语音训练环境', 'Speech training environments')}</span></div><TtsEnvironmentPanel/></section>;
  if (!data) return <><ResourceState error={error} retry={reload}/>{speech}</>;
  const runtime = data.runtime;
  const backend = runtime.compute_backend || 'cpu';
  const profileName = ({ 'linux-dtk': 'Linux DTK', 'linux-cuda': 'Linux CUDA', 'windows-cuda': 'Windows CUDA', 'macos-mps': 'macOS MPS', 'linux-cpu': 'Linux CPU', 'windows-cpu': 'Windows CPU', 'macos-cpu': 'macOS CPU' } as Record<string, string>)[runtime.environment_profile ?? ''] || runtime.environment_profile || runtime.platform;
  const backendName = ({ cuda: 'CUDA', hip: 'DTK / HIP', mps: 'Metal', cpu: 'CPU' } as Record<string, string>)[backend] || backend;
  const gpuAvailable = (gpu: EnvironmentStatus['runtime']['gpus'][number]) => backend === 'mps'
    ? runtime.mps_available === true && gpu.kind === 'mps'
    : ['cuda', 'hip'].includes(backend) && runtime.cuda_available === true && gpu.cuda_available === true;
  return <><div className="setup-runtime-heading"><span>{text('当前运行环境', 'Current environment')}</span></div>
    <dl className="setup-runtime-grid">{[[text('部署环境', 'Deployment'), profileName], ['Python', runtime.python], ['PyTorch', runtime.torch || text('未安装', 'Not installed')], [text('计算后端', 'Compute backend'), backendName]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
    <div className="setup-gpus">{runtime.gpus?.length ? runtime.gpus.map((gpu, index) => <div key={gpu.device ?? index}><span className="setup-gpu-index">{index + 1}</span><span><strong>{gpu.name}</strong><small>{gpu.device || `GPU ${index}`}{gpu.mem_total_mb ? ` · ${(gpu.mem_total_mb / 1024).toFixed(0)} GiB` : ''}</small></span>{gpuAvailable(gpu) ? <Check size={17} aria-label={text('设备可用', 'Device available')}/> : <small>{text('当前环境不可用', 'Unavailable in this environment')}</small>}</div>) : <p className="setup-note">{text('未检测到可用 GPU。', 'No available GPU detected.')}</p>}</div>
    <React.Suspense fallback={<div className="setup-resource-loading"><Loader2 size={16} className="animate-spin"/></div>}><AttentionSetup mode="onboarding-attention" initialStatus={data}/></React.Suspense>
    {speech}
    <p className="setup-note">{text('完成后即可创建项目，添加训练数据。', 'Create a project and add your training data to get started.')}</p>
    {error && <ResourceState error={error} retry={reload}/>}</>;
}
