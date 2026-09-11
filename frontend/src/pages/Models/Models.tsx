import React from 'react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Download, ExternalLink, FolderSearch, HardDrive, Loader2, Plus, RefreshCw, Settings2, Star, Trash2, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import { ModelAsset, ModelDownload, ModelDownloadRequest, Settings } from '../../api/types';
import { useFamilies, familyByName } from '../../api/hooks/useFamilies';
import { PathInput } from '../../components/PathBrowser';
import { formatBytes } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { SettingsSections } from '../Settings/SettingsSections';

const FIELD_KIND: Record<string, string> = { dit_path: 'dit', text_encoder_path: 'text_encoder', vae_path: 'vae', tokenizer_path: 'tokenizer' };
const KIND_LABEL: Record<string, string> = { dit: '主模型 / DiT', text_encoder: '文本编码器', vae: 'VAE', tokenizer: '分词器目录' };
const input = 'w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-900';
const panel = 'settings-model-panel';
const secondary = 'inline-flex items-center justify-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs hover:bg-slate-50 disabled:opacity-50 dark:border-slate-600 dark:hover:bg-slate-700';
const primary = 'inline-flex items-center justify-center gap-1.5 rounded-md bg-blue-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50';
const active = (d: ModelDownload) => d.status === 'queued' || d.status === 'downloading';
// Publisher file pages verified 2026-09-11. Selecting a source only fills the download form.
const SOURCES: Record<string, Record<string, { label: string; url: string; size: string; dtype: ModelDownloadRequest['dtype'] }>> = {
  anima: {
    dit: { label: 'Anima Base 1.0', url: 'https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-base-v1.0.safetensors', size: '4.18 GB', dtype: 'bf16' },
    text_encoder: { label: 'Qwen3 0.6B Base', url: 'https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/text_encoders/qwen_3_06b_base.safetensors', size: '1.19 GB', dtype: 'bf16' },
    vae: { label: 'Qwen Image VAE', url: 'https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/vae/qwen_image_vae.safetensors', size: '254 MB', dtype: 'bf16' },
  },
  krea2: {
    dit: { label: 'Krea 2 Raw · FP8 scaled', url: 'https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_fp8_scaled.safetensors', size: '13.1 GB', dtype: 'fp8' },
    text_encoder: { label: 'Qwen3-VL 4B · BF16', url: 'https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors', size: '8.88 GB', dtype: 'bf16' },
    vae: { label: 'Qwen Image VAE', url: 'https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/vae/qwen_image_vae.safetensors', size: '254 MB', dtype: 'bf16' },
  },
};

export default function Models({ embedded = false }: { embedded?: boolean }) {
  const { t } = useTranslation();
  const location = useLocation();
  const [params] = useSearchParams();
  const { data: families } = useFamilies();
  const [family, setFamily] = React.useState(params.get('family') || 'anima');
  const [models, setModels] = React.useState<ModelAsset[]>([]);
  const [downloads, setDownloads] = React.useState<ModelDownload[]>([]);
  const [settings, setSettings] = React.useState<Settings | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [notice, setNotice] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [addOpen, setAddOpen] = React.useState(false);
  const [downloadOpen, setDownloadOpen] = React.useState(false);
  const sourceRef = React.useRef<HTMLElement>(null);
  const errorRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    if (!addOpen && !downloadOpen) return;
    sourceRef.current?.scrollIntoView?.({ block: 'start', behavior: 'smooth' });
    sourceRef.current?.querySelector<HTMLElement>('input:not([type=checkbox]),select')?.focus({ preventScroll: true });
  }, [addOpen, downloadOpen]);
  React.useEffect(() => { if (error) errorRef.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' }); }, [error]);
  const [kind, setKind] = React.useState('dit');
  const [path, setPath] = React.useState('');
  const [dtype, setDtype] = React.useState('bf16');
  const [isDefault, setIsDefault] = React.useState(true);
  const [sourceMode, setSourceMode] = React.useState<'url' | 'repo'>('url');
  const [url, setUrl] = React.useState('');
  const [repo, setRepo] = React.useState('');
  const [filename, setFilename] = React.useState('');
  const [revision, setRevision] = React.useState('main');
  const [cancelling, setCancelling] = React.useState<string[]>([]);
  const currentFamily = familyByName(families, family);
  const kinds = (currentFamily?.weights || []).map(w => FIELD_KIND[w.field]).filter(Boolean);
  const kindOptions = kinds.length ? kinds : ['dit', 'text_encoder', 'vae', 'tokenizer'];
  const label = (k: string) => t(`models.kind_${k}`, KIND_LABEL[k] || k);

  const refresh = React.useCallback(async (silent = false) => {
    try {
      const [assets, tasks, config] = await Promise.all([
        apiClient.get<ModelAsset[]>('/models', { silent }),
        apiClient.get<ModelDownload[]>('/models/downloads', { silent }),
        apiClient.get<Settings>('/settings', { silent }),
      ]);
      setModels(assets); setDownloads(tasks); setSettings(config); if (!silent) setError('');
    } catch (e) { if (!silent) setError(formatApiError(e)); }
    finally { setLoading(false); }
  }, []);
  React.useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => { void refresh(true); }, 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  const action = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError(''); setNotice('');
    try { await fn(); await refresh(); }
    catch (e) { setError(formatApiError(e)); }
    finally { setBusy(false); }
  };
  const openDownload = (k = 'dit') => {
    setKind(k); setDownloadOpen(true); setAddOpen(false); setError('');
    const source = SOURCES[family]?.[k];
    if (source) { setUrl(source.url); setSourceMode('url'); setDtype(source.dtype || ''); }
  };
  const local = (k = 'dit') => { setKind(k); setAddOpen(true); setDownloadOpen(false); setPath(''); setError(''); };
  const changeDownloadKind = (next: string) => {
    const wasSuggested = sourceMode === 'url' && (!url.trim() || url === SOURCES[family]?.[kind]?.url);
    setKind(next);
    if (wasSuggested) {
      const source = SOURCES[family]?.[next];
      setUrl(source?.url || ''); setDtype(source?.dtype || '');
    } else {
      setNotice(t('models.customSourceKept', '已保留你填写的来源，请确认它属于当前选择的组件。'));
    }
  };
  const start = () => action(async () => {
    const body: ModelDownloadRequest = {
      family: family as ModelDownloadRequest['family'], kind: kind as ModelDownloadRequest['kind'],
      is_default: isDefault, dtype: (dtype || null) as ModelDownloadRequest['dtype'], revision: revision.trim() || 'main',
      ...(sourceMode === 'url' ? { url: url.trim() } : { repo_id: repo.trim(), filename: filename.trim(), revision: revision.trim() || 'main' }),
    };
    await apiClient.post<ModelDownload>('/models/downloads', body);
    setDownloadOpen(false); setNotice(t('models.downloadStarted', '下载已加入列表，完成后自动注册到本族模型库。'));
  });
  const selected = models.filter(m => m.family === family);
  const tasks = downloads.filter(d => d.family === family);
  const ready = ['dit', 'text_encoder', 'vae'].filter(k => selected.some(m => m.kind === k && m.exists && m.is_default)).length;

  return <div data-testid="models-page"><SettingsSections sections={[
    { id: 'models-components', label: t('models.components', '模型组件') },
    ...(addOpen || downloadOpen ? [{ id: 'models-source', label: t('models.source', '添加来源') }] : []),
    ...(tasks.length ? [{ id: 'models-downloads', label: t('models.downloads', '下载列表') }] : []),
    { id: 'models-library', label: t('models.registeredFiles', '已登记模型') },
  ]}>
    <section id="models-components" data-settings-section tabIndex={-1} className="settings-section">
    <div className="settings-section-heading">
      <div><h2 className={`flex items-center gap-2 font-bold ${embedded ? 'text-base' : 'text-2xl'}`}>{!embedded && <HardDrive className="h-6 w-6 text-blue-500" />}{t('models.title')}</h2></div>
      <div className="flex flex-wrap gap-2">
        {params.get('project') && <Link className={secondary} to={`/projects/${encodeURIComponent(params.get('project')!)}`}>{t('models.backProject', '返回项目')}</Link>}
        {!embedded && <Link to="/settings/preferences?section=storage" replace state={location.state} className={secondary}><Settings2 size={16} />{t('models.storageSettings', '目录与默认设置')}</Link>}
        <button className={secondary} onClick={() => local()} data-testid="add-model-btn"><Plus size={16} />{t('models.addModel')}</button>
        {family !== 'toy' && <button className={primary} onClick={() => openDownload()} data-testid="download-model-btn"><Download size={16} />{t('models.download', '下载模型')}</button>}
      </div>
    </div>
    <div className="settings-model-family">{(families || []).map(f => <button key={f.name} aria-pressed={family === f.name} onClick={() => { setFamily(f.name); setDownloadOpen(false); setAddOpen(false); }}>{f.label}</button>)}{family !== 'toy' && <span className="ml-auto text-xs text-slate-500">{t('models.defaultReady', '已设默认组件')} {ready} / 3</span>}</div>
    {error && <div ref={errorRef} role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/20 dark:text-red-300">{error}<button onClick={() => void refresh()} className="ml-3 underline">{t('common.retry', '重试')}</button></div>}
    {notice && <div role="status" className="rounded-lg bg-blue-50 p-3 text-sm text-blue-700 dark:bg-blue-950/30 dark:text-blue-300">{notice}</div>}

    {family !== 'toy' && <><div className="settings-model-list">{['dit', 'text_encoder', 'vae'].map(k => {
      const chosen = selected.find(m => m.kind === k && m.is_default); const source = SOURCES[family]?.[k];
      return <section key={k} className="settings-model-component" data-testid={`model-component-${k}`}>
        <div><h3>{label(k)}</h3><span className={`text-xs ${chosen?.exists ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-500'}`}>{chosen?.exists ? t('models.defaultSelected', '已选默认') : t('models.needsSetup', '待设置')}</span></div>
        <div className="settings-model-choice">
          <select className={input} aria-label={`${label(k)} ${t('models.setDefault')}`} value={chosen?.id || ''} disabled={busy} onChange={e => { const id = e.target.value; if (id) void action(() => apiClient.patch(`/models/${id}`, { is_default: true })); else if (chosen) void action(() => apiClient.patch(`/models/${chosen.id}`, { is_default: false })); }}><option value="">{t('models.chooseDefault', '选择本地默认模型')}</option>{selected.filter(m => m.kind === k).map(m => <option key={m.id} value={m.id} disabled={!m.exists}>{m.path.split(/[\\/]/).pop()} {!m.exists ? `(${t('models.missing')})` : ''}</option>)}</select>
          {chosen && <p className="settings-model-path">{chosen.dtype || '—'} · {chosen.path}</p>}
          {source && <p className="settings-note">{source.label} · {source.size}</p>}
          <div className="settings-model-actions"><button className={secondary} onClick={() => local(k)}><Plus size={13} />{t('models.useLocal', '已有文件')}</button><button className={secondary} onClick={() => openDownload(k)}><Download size={13} />{t('models.getComponent', '获取组件')}</button>{source && <a href={source.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-xs text-blue-600">{t('models.publisher', '发布页')}<ExternalLink size={12} /></a>}</div>
        </div>
      </section>;
    })}</div><details className="settings-inline-details"><summary>{t('models.componentDetails', '组件与分词器说明')}</summary><p>{t('models.componentHelp', '标准单文件配套的默认配置与分词器已内置，无需另下 tokenizer。自定义文本编码器可登记完整 HF 目录（含 config、tokenizer 和全部权重分片）；可选分词器目录可通过“添加本地模型”登记。Anima 与 Krea 2 可登记同一个 Qwen Image VAE 文件。默认路径用于后续项目/任务，已有显式配置保留。')}</p></details></>}
    </section>

    {addOpen && <section ref={sourceRef} id="models-source" data-settings-section tabIndex={-1} className={`${panel} space-y-4`} data-testid="add-model-modal"><h3 className="text-lg font-semibold">{t('models.addModel')}</h3><p className="text-xs text-slate-500">{t('models.localPathHelp', '选择运行训练服务的电脑上的文件或完整文本编码器目录。')}</p>
      <div className="settings-model-form"><label className="space-y-1 text-xs">{t('models.family')}<select className={input} value={family} onChange={e => setFamily(e.target.value)} data-testid="model-family-select">{(families || []).map(f => <option key={f.name} value={f.name}>{f.label}</option>)}</select></label><label className="space-y-1 text-xs">{t('models.kind')}<select className={input} value={kind} onChange={e => setKind(e.target.value)} data-testid="model-kind-select">{[...new Set([...kindOptions, 'tokenizer'])].map(k => <option key={k} value={k}>{label(k)}</option>)}</select></label></div>
      <div className="settings-field"><p className="settings-field-label">{t('models.pathLabel')}</p><PathInput ariaLabel={t('models.pathLabel')} value={path} onChange={setPath} placeholder={t('models.pathPlaceholder')} /></div><div className="flex items-center gap-3"><label className="text-xs">{t('models.dtype')}<select className={input} value={dtype} onChange={e => setDtype(e.target.value)}>{['bf16', 'fp16', 'fp32', 'fp8', ''].map(d => <option key={d} value={d}>{d || t('models.dtypeUnknown', '未知')}</option>)}</select></label><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={isDefault} onChange={e => setIsDefault(e.target.checked)} />{t('models.setDefault')}</label></div>
      <div className="flex justify-end gap-2"><button className={secondary} onClick={() => setAddOpen(false)}>{t('common.cancel')}</button><button className={primary} disabled={busy || !path.trim()} data-testid="add-model-submit" onClick={() => void action(async () => { await apiClient.post('/models', { family, kind, path: path.trim(), dtype: dtype || null, is_default: isDefault }); setAddOpen(false); })}>{busy ? t('models.adding') : t('models.add')}</button></div>
    </section>}

    {downloadOpen && <section ref={sourceRef} id="models-source" data-settings-section tabIndex={-1} className={panel} data-testid="download-model-form"><div className="mb-4 flex items-center justify-between"><h3 className="font-semibold">{t('models.download', '下载模型')} · {label(kind)}</h3><button onClick={() => setDownloadOpen(false)} aria-label={t('common.close', '关闭')}><X size={18} /></button></div>
      <form className="space-y-4 settings-model-form" onSubmit={e => { e.preventDefault(); void start(); }}>
        <button type="button" className="text-xs text-blue-600 hover:underline" onClick={() => openDownload(kind)}>{t('models.fillSuggestedSource', '填入当前组件的标准来源')}</button>
        <div className="settings-model-form"><label className="space-y-1 text-xs">{t('models.kind')}<select className={input} value={kind} onChange={e => changeDownloadKind(e.target.value)}>{['dit', 'text_encoder', 'vae'].map(k => <option key={k} value={k}>{label(k)}</option>)}</select></label><label className="space-y-1 text-xs">{t('models.sourceMode', '来源格式')}<select className={input} value={sourceMode} onChange={e => setSourceMode(e.target.value as 'url' | 'repo')}><option value="url">Hugging Face URL</option><option value="repo">{t('models.repoAndFile', '仓库 + 文件名')}</option></select></label><label className="space-y-1 text-xs">{t('models.dtype')}<select className={input} value={dtype} onChange={e => setDtype(e.target.value)}>{['bf16', 'fp16', 'fp32', 'fp8', ''].map(d => <option key={d} value={d}>{d || t('models.dtypeUnknown', '未知')}</option>)}</select></label></div>
        {sourceMode === 'url' ? <label className="block space-y-1 text-xs">{t('models.fileUrl', '文件下载链接')}<input autoFocus className={input} value={url} onChange={e => setUrl(e.target.value)} placeholder="https://huggingface.co/owner/repo/blob/main/model.safetensors" required data-testid="model-download-url" /></label> : <div className="settings-model-form"><label className="space-y-1 text-xs">Repository<input className={input} value={repo} onChange={e => setRepo(e.target.value)} placeholder="owner/repository" required /></label><label className="space-y-1 text-xs">{t('models.repositoryFile', '仓库内文件名')}<input className={input} value={filename} onChange={e => setFilename(e.target.value)} placeholder="folder/model.safetensors" required /></label><label className="space-y-1 text-xs">Revision<input className={input} value={revision} onChange={e => setRevision(e.target.value)} placeholder="main" required /></label></div>}
        <p className="break-all text-xs leading-6 text-slate-500">{t('models.downloadDestination', '保存目录')}：{settings?.paths.models_dir} <Link className="ml-2 text-blue-600 underline dark:text-blue-400" to="/settings/preferences?section=storage" replace state={location.state}>{t('models.changeDownloadDir', '更改保存目录')}</Link><br />{t('models.downloadHelp', '支持完整 safetensors 单文件。下载并检查文件完整性后才注册；模型架构与训练兼容性仍由训练准备阶段校验。中断后可重新下载，已有文件不会覆盖。私有/授权仓库请先在运行服务的电脑上执行 hf auth login 并取得访问权限。')}</p>
        <div className="flex flex-wrap items-center justify-between gap-3"><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={isDefault} onChange={e => setIsDefault(e.target.checked)} />{t('models.defaultAfterDownload', '完成后设为本族默认组件')}</label><button className={primary} disabled={busy} type="submit" data-testid="model-download-start">{busy ? <Loader2 className="animate-spin" size={16} /> : <Download size={16} />}{t('models.startDownload', '开始下载')}</button></div>
      </form>
    </section>}

    {tasks.length > 0 && <section id="models-downloads" data-settings-section tabIndex={-1} className="settings-section" data-testid="model-downloads"><div className="mb-4 flex items-center justify-between"><h3 className="font-semibold">{t('models.downloads', '下载列表')}</h3><button onClick={() => void refresh()} className="text-slate-500" aria-label={t('common.refresh', '刷新')}><RefreshCw size={16} /></button></div><div className="divide-y dark:divide-slate-700">{tasks.slice(0, 15).map(d => <div key={d.id} className="space-y-2 py-3">
      <div className="flex items-start justify-between gap-3"><div className="min-w-0"><p className="break-all text-sm font-medium">{d.filename}</p><p className="mt-1 text-xs text-slate-500">{label(d.kind)} · {t(`models.downloadStatus_${d.status}`, { queued: '等待下载', downloading: '正在下载', completed: '已下载并注册', failed: '下载失败', cancelled: '已取消' }[d.status])} · {formatBytes(d.downloaded_bytes)}{d.total_bytes ? ` / ${formatBytes(d.total_bytes)}` : ''}</p></div>
        {active(d) ? <button className={secondary} disabled={cancelling.includes(d.id)} onClick={() => { setCancelling(v => [...v, d.id]); void action(() => apiClient.post(`/models/downloads/${d.id}/cancel`, {})).finally(() => setCancelling(v => v.filter(id => id !== d.id))); }}>{t('common.cancel')}</button> : d.status !== 'completed' && <button className={secondary} disabled={busy} onClick={() => { setKind(d.kind); setUrl(d.source_url); setDtype(d.dtype || ''); setIsDefault(d.is_default); setSourceMode('url'); setDownloadOpen(true); }}>{t('models.retryDownload', '重新下载')}</button>}</div>
      {active(d) && <progress className="h-2 w-full accent-blue-600" value={d.total_bytes ? d.downloaded_bytes : undefined} max={d.total_bytes || undefined} aria-label={t('models.downloadProgress', '下载进度')} />}<p className="break-all text-xs text-slate-400">{d.target_path}</p>{d.error && <p role="alert" className="break-words text-xs text-red-600 dark:text-red-400">{d.error}</p>}
    </div>)}</div></section>}

    <section id="models-library" data-settings-section tabIndex={-1} className="settings-section settings-model-library"><div className="mb-4 flex flex-wrap items-center justify-between gap-2"><h3 className="font-semibold">{t('models.registeredFiles', '已登记模型')} · {selected.length}</h3><button className={secondary} disabled={busy} onClick={() => void action(async () => { const found = await apiClient.post<ModelAsset[]>('/models/scan', { family }); setNotice(`${t('models.scanAdded', '新登记文件')}：${found.length}`); })}><FolderSearch size={16} />{t('models.scanDirectory')}</button></div>
      {loading ? <p className="py-8 text-center text-sm text-slate-500">{t('common.loading')}</p> : selected.length === 0 ? <div className="space-y-3 py-8 text-center"><p className="text-sm text-slate-500">{family === 'toy' ? t('models.toyNoFiles', 'Toy 测试模型已内置，无需下载权重。') : t('models.emptySetup', '还没有本地模型。准备好三个组件即可在项目里使用。')}</p><button className={secondary} onClick={() => local()}><Plus size={16} />{t('models.useLocal', '已有文件')}</button></div> : <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="text-xs text-slate-400"><tr><th className="pb-3 pr-3">{t('models.kind')}</th><th className="pb-3">{t('models.pathLabel')}</th><th className="pb-3 px-3">{t('models.dtype')}</th><th className="pb-3 text-right">{t('models.status', '状态 / 操作')}</th></tr></thead><tbody className="divide-y dark:divide-slate-700">{selected.map(m => <tr key={m.id}>
        <td className="py-3 pr-3 whitespace-nowrap">{label(m.kind)}</td><td className="py-3"><div className="max-w-xl break-all font-mono text-xs">{m.path}</div><p className="mt-1 text-xs text-slate-400">{formatBytes(m.size)}</p></td><td className="px-3 text-xs">{m.dtype || '—'}</td><td className="py-3"><div className="flex items-center justify-end gap-2 whitespace-nowrap"><span className={`text-xs ${m.exists ? 'text-emerald-600' : 'text-red-500'}`}>{m.exists ? t('models.exists') : t('models.missing')}</span><button disabled={busy || !m.exists} className={`rounded p-2 disabled:opacity-40 ${m.is_default ? 'text-amber-500' : 'text-slate-400'}`} title={m.is_default ? t('models.default') : t('models.setDefault')} aria-label={`${t('models.setDefault')} ${m.path}`} onClick={() => void action(() => apiClient.patch(`/models/${m.id}`, { is_default: !m.is_default }))}><Star size={16} fill={m.is_default ? 'currentColor' : 'none'} /></button><button disabled={busy} className="rounded p-2 text-slate-400 hover:text-red-500" title={t('common.remove')} onClick={() => { if (window.confirm(t('models.removeRecord', '只移除这条模型登记，磁盘文件会保留。继续吗？'))) void action(() => apiClient.delete(`/models/${m.id}`)); }}><Trash2 size={16} /></button></div></td>
      </tr>)}</tbody></table></div>}
    </section>


  </SettingsSections></div>;
}
