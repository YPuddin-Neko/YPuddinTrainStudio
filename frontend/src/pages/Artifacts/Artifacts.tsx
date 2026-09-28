import React from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { apiClient, apiUrl } from '../../api/client';
import { Artifact } from '../../api/types';
import { formatBytes, formatTime } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { projectUrl } from '../../utils/projectVersions';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import StudioSelect from '../../components/StudioSelect';
import '../Queue/queue.css';
import './artifacts.css';
import { Box, Download, Trash2, FileJson, PackageOpen, RefreshCw, Search, Wrench, X } from 'lucide-react';
import { LoadingNote } from '../../components/Loading';

const PAGE_SIZES = [25, 50, 100];
const PAGE_SIZE_KEY = 'studio.artifacts.pageSize';
const button = 'ui-btn ui-btn-sm ui-btn-icon';

type VersionedArtifact = Artifact & { version_id?: string | null };
interface ArtifactsProps { embedded?: boolean; projectId?: string; versionId?: string; jobId?: string; readOnly?: boolean }

export default function Artifacts({ embedded = false, projectId: projectScope, versionId: versionScope, jobId: jobScope, readOnly = false }: ArtifactsProps) {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const [params, setParams] = useSearchParams();
  const scoped = projectScope !== undefined || versionScope !== undefined || jobScope !== undefined;
  const projectId = scoped ? projectScope : params.get('project_id') || params.get('project') || undefined;
  const versionId = scoped ? versionScope : params.get('version_id') || undefined;
  const jobId = scoped ? jobScope : params.get('job_id') || params.get('job') || undefined;
  const [localQuery, setLocalQuery] = React.useState('');
  const query = scoped ? localQuery : params.get('q') || '';
  const [page, setPage] = React.useState(1);
  const [pageSize, setPageSize] = React.useState(() => {
    try { const saved = Number(localStorage.getItem(PAGE_SIZE_KEY)); return PAGE_SIZES.includes(saved) ? saved : PAGE_SIZES[0]; } catch { return PAGE_SIZES[0]; }
  });
  const changePageSize = (value: string) => {
    setPageSize(Number(value)); setPage(1);
    try { localStorage.setItem(PAGE_SIZE_KEY, value); } catch { /* The choice lasts for this page only. */ }
  };
  const [sort, setSort] = React.useState('newest');
  const [artifacts, setArtifacts] = React.useState<VersionedArtifact[]>([]);
  const scopeKey = JSON.stringify([projectId, versionId, jobId]);
  const [loadedScope, setLoadedScope] = React.useState('');
  const scopeRef = React.useRef(scopeKey);
  React.useLayoutEffect(() => { scopeRef.current = scopeKey; }, [scopeKey]);
  const [loading, setLoading] = React.useState(true);
  const [busy, setBusy] = React.useState<string | null>(null);
  const [error, setError] = React.useState('');
  const [metadataFor, setMetadataFor] = React.useState<VersionedArtifact | null>(null);
  const request = React.useRef<AbortController | null>(null);
  const metadataClose = React.useRef<HTMLButtonElement>(null);
  const fetchArtifacts = React.useCallback(async () => {
    request.current?.abort(); const controller = new AbortController(); request.current = controller;
    setLoading(true); setError(''); setArtifacts([]);
    try {
      const data = await apiClient.get<VersionedArtifact[]>('/artifacts', { params: { project_id: projectId, ...(versionId ? { version_id: versionId } : {}), ...(jobId ? { job_id: jobId } : {}) }, signal: controller.signal, silent: true });
      if (!controller.signal.aborted) { setArtifacts(Array.isArray(data) ? data : []); setLoadedScope(scopeKey); }
    } catch (error) { if (!controller.signal.aborted) setError(formatApiError(error)); }
    finally { if (!controller.signal.aborted) setLoading(false); }
  }, [projectId, versionId, jobId, scopeKey]);
  React.useEffect(() => { setLocalQuery(''); setMetadataFor(null); void fetchArtifacts(); return () => request.current?.abort(); }, [fetchArtifacts]);
  useEventStream(EVENT_TYPES.ARTIFACT_CREATED, (event: { project_id?: string; version_id?: string; job_id?: string }) => {
    if (event.project_id && projectId && event.project_id !== projectId || event.version_id && versionId && event.version_id !== versionId || jobId && event.job_id !== jobId) return;
    void fetchArtifacts();
  });
  React.useEffect(() => {
    if (!metadataFor) return;
    const previous = document.activeElement as HTMLElement | null; metadataClose.current?.focus();
    return () => previous?.focus();
  }, [metadataFor]);
  const action = async (id: string, run: () => Promise<unknown>) => {
    if (readOnly) return;
    const originalScope = scopeKey;
    setBusy(id); setError('');
    try { await run(); if (originalScope === scopeRef.current) await fetchArtifacts(); }
    catch (error) { if (originalScope === scopeRef.current) setError(formatApiError(error)); }
    finally { setBusy(null); }
  };
  React.useEffect(() => { setPage(1); }, [query, scopeKey, sort]);
  const updateQuery = (value: string) => {
    if (scoped) { setLocalQuery(value); return; }
    const next = new URLSearchParams(params); if (value) next.set('q', value); else next.delete('q'); setParams(next, { replace: true });
  };
  const clearFilter = () => {
    if (scoped) { setLocalQuery(''); return; }
    const next = new URLSearchParams(params); ['project', 'project_id', 'version_id', 'job', 'job_id', 'q'].forEach(key => next.delete(key)); setParams(next);
  };
  const filtered = (loadedScope === scopeKey ? artifacts : []).filter(artifact => (!projectId || artifact.project_id === projectId) && (!versionId || artifact.version_id === versionId) && (!jobId || artifact.job_id === jobId) && (!query || `${artifact.name} ${artifact.job_id || ''} ${artifact.algo || ''}`.toLowerCase().includes(query.toLowerCase())));

  const fullModelComponents = (artifact: Artifact) => {
    const values = artifact.metadata?.components;
    const labels: Record<string, string> = { backbone: 'UNet / DiT', text_encoder: text('文本编码器', 'Text encoder'), text_encoder_2: text('文本编码器 2', 'Text encoder 2') };
    return Array.isArray(values) && values.length ? values.map(value => labels[String(value)] || String(value)).join(' + ') : text('查看元数据', 'View metadata');
  };
  const sizeDetail = (artifact: Artifact) => artifact.kind === 'model' ? fullModelComponents(artifact) : `${artifact.rank ?? '—'} / ${artifact.alpha ?? '—'} / ${artifact.factor ?? '—'}`;
  const sizeLabel = (artifact: Artifact) => artifact.kind === 'model' ? text('模型组件', 'Model components') : t('artifacts.rankAlphaFactor');
  const onlyFullModels = filtered.length > 0 && filtered.every(artifact => artifact.kind === 'model');
  const sorted = [...filtered].sort((a, b) => sort === 'step' ? (b.step ?? 0) - (a.step ?? 0) : b.created_at - a.created_at);
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const currentPage = Math.min(page, pages);

  return <div className={`artifacts-page${embedded ? ' artifacts-embedded' : ''}`} data-testid="artifacts-page">
    {!embedded && <header className="artifact-heading"><h2><Box size={20}/>{text('训练产物库', 'Training output library')}</h2></header>}
    {readOnly && <p className="artifact-readonly">{text('已归档：可以下载已有权重和查看元数据，恢复版本后才能修改或移除产物。', 'Archived: download weights and inspect metadata. Restore the version to change or remove outputs.')}</p>}
    <div className="artifact-controls">
      <label className="artifact-search"><Search size={15}/><input aria-label={text('搜索训练产物', 'Search training outputs')} value={query} onChange={event => updateQuery(event.target.value)} placeholder={text('搜索文件名、任务或算法', 'Search filename, job or algorithm')} /></label>
      <StudioSelect className="artifact-sort" aria-label={text('产物排序', 'Output order')} value={sort} options={[{ value: 'newest', label: text('最新产物优先', 'Newest first') }, { value: 'step', label: text('训练步数降序', 'Highest step first') }]} onValueChange={setSort}/><span className="artifact-count">{filtered.length} {text('个产物', 'outputs')}</span>
      <button type="button" className="ui-btn artifact-refresh" disabled={loading} onClick={() => void fetchArtifacts()} aria-label={text('刷新产物', 'Refresh outputs')}><RefreshCw size={14} className={loading ? 'animate-spin' : ''}/>{text('刷新', 'Refresh')}</button>
      {(query || !scoped && (projectId || versionId || jobId)) && <button type="button" className="ui-link" onClick={clearFilter}>{text('清除筛选', 'Clear filters')}</button>}
      {projectId && !scoped && <Link className="ui-link" to={projectUrl(projectId, versionId, 'results')}>{text('项目', 'Project')} · {projectId}</Link>}
    {filtered.length > PAGE_SIZES[0] && <div className="task-pagination artifact-pagination"><div><StudioSelect aria-label={text('每页产物数', 'Outputs per page')} value={String(pageSize)} options={PAGE_SIZES.map(value => ({ value: String(value), label: text(`${value} 个 / 页`, `${value} / page`) }))} onValueChange={changePageSize}/><button type="button" className="ui-btn" disabled={currentPage <= 1} onClick={() => setPage(currentPage - 1)}>{text('上一页', 'Previous')}</button><span>{currentPage} / {pages}</span><button type="button" className="ui-btn" disabled={currentPage >= pages} onClick={() => setPage(currentPage + 1)}>{text('下一页', 'Next')}</button></div></div>}
    </div>
    {error && <div role="alert" className="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-300"><p className="whitespace-pre-wrap break-words">{error}</p><button type="button" className="ui-link mt-2" onClick={() => void fetchArtifacts()}>{t('common.retry')}</button></div>}
    <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">
      {loading ? <LoadingNote block label={text('正在读取训练产物…', 'Loading outputs…')}/> : filtered.length === 0 ? <div className="flex flex-col items-center gap-2 p-10 text-center"><PackageOpen className="h-9 w-9 text-slate-400" /><p className="text-sm font-medium text-slate-500">{query || projectId || versionId || jobId ? text('没有匹配的训练产物', 'No matching training outputs') : t('artifacts.empty')}</p><Link to={projectId ? `${projectUrl(projectId, versionId, 'results')}&result_tab=jobs` : '/queue'} className="ui-link mt-2">{text('查看训练任务', 'View training jobs')}</Link></div> : <table className="artifact-table w-full text-left text-sm">
        <thead className="border-b border-slate-200 bg-slate-50 text-xs text-slate-500 dark:border-slate-700 dark:bg-slate-900/50"><tr><th className="px-3 py-2.5">{t('artifacts.name')}</th><th className="px-3 py-2.5">{t('artifacts.job')}</th><th className="px-3 py-2.5">{t('artifacts.algo')}</th><th className="artifact-secondary px-3 py-2.5">{onlyFullModels ? text('模型组件', 'Model components') : text('参数规模 / 组件', 'Adapter size / components')}</th><th className="px-3 py-2.5">{t('artifacts.size')}</th><th className="artifact-secondary px-3 py-2.5">{t('artifacts.created')}</th><th className="px-3 py-2.5 text-right">{t('artifacts.actions')}</th></tr></thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-700">{sorted.slice((currentPage - 1) * pageSize, currentPage * pageSize).map(artifact => <tr key={artifact.id} className="hover:bg-slate-50 dark:hover:bg-slate-900/30" data-testid={`artifact-row-${artifact.id}`}>
          <td className="max-w-60 break-words px-3 py-3 font-mono text-xs font-medium">{artifact.name}<small className="artifact-compact-meta">{sizeLabel(artifact)}: {sizeDetail(artifact)} · {formatTime(artifact.created_at)}</small>{!scoped && artifact.project_id && <Link className="artifact-project-link" to={projectUrl(artifact.project_id, artifact.version_id, 'results')}>{text('项目', 'Project')} {artifact.project_id} · {text('所属版本结果', 'Version results')}</Link>}</td><td className="px-3 py-3 font-mono text-xs text-slate-500">{artifact.job_id ? <Link to={`/jobs/${encodeURIComponent(artifact.job_id)}`} className="ui-link">{artifact.job_id}</Link> : '—'}</td><td className="px-3 py-3 text-xs">{artifact.kind === 'model' ? <>{text('全量模型', 'Full model')}<span className="block text-slate-500">{text('组件目录 · ZIP', 'Component directory · ZIP')}</span></> : <>{artifact.algo || text('适配器', 'Adapter')}{artifact.kind && artifact.kind !== 'weights' && <span className="block text-slate-400">{artifact.kind}</span>}</>}</td><td className="artifact-secondary whitespace-nowrap px-3 py-3 font-mono text-xs">{sizeDetail(artifact)}</td><td className="whitespace-nowrap px-3 py-3 font-mono text-xs">{formatBytes(artifact.size)}</td><td className="artifact-secondary whitespace-nowrap px-3 py-3 text-xs text-slate-500">{formatTime(artifact.created_at)}</td>
          <td className="artifact-action-cell px-3 py-3"><div className="artifact-row-actions flex items-center justify-end gap-1.5"><a href={apiUrl(`/artifacts/${encodeURIComponent(artifact.id)}/download`)} download={artifact.kind === 'model' ? `${artifact.name}.zip` : undefined} className={button} aria-label={`${t('common.download')}: ${artifact.name}`} title={t('common.download')}><Download className="h-4 w-4" /></a><button className={button} aria-label={`${text('查看元数据', 'View metadata')}: ${artifact.name}`} title={text('查看元数据', 'View metadata')} onClick={() => setMetadataFor(artifact)}><FileJson className="h-4 w-4" /></button>{artifact.legacy_text_keys && <button type="button" className="ui-btn ui-btn-sm" aria-label={`${text('修复键名', 'Fix key names')}: ${artifact.name}`} title={text('把文本编码器的键名改成 ComfyUI 能读取的写法，权重不变。', 'Renames the text encoder keys to the names ComfyUI reads; the weights stay the same.')} disabled={readOnly || !!busy} onClick={() => void action(artifact.id, () => apiClient.post(`/artifacts/${artifact.id}/fix-text-keys`, undefined, { silent: true }))}><Wrench className="h-3.5 w-3.5"/>{busy === artifact.id ? text('修复中…', 'Fixing…') : text('修复键名', 'Fix key names')}</button>}<button type="button" className={`${button} ui-btn-danger`} aria-label={`${text('移除产物记录', 'Remove output record')}: ${artifact.name}`} disabled={readOnly || !!busy} onClick={() => { if (window.confirm(text(`从产物列表移除 ${artifact.name}？磁盘中的权重文件会保留。`, `Remove ${artifact.name} from the output list? Its weight file will remain on disk.`))) void action(artifact.id, () => apiClient.delete(`/artifacts/${artifact.id}`, { silent: true })); }}><Trash2 className="h-4 w-4" /></button></div></td>
        </tr>)}</tbody>
      </table>}
    </div>
    {metadataFor && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={() => setMetadataFor(null)}><div role="dialog" aria-modal="true" aria-labelledby="artifact-metadata-title" onKeyDown={event => { if (event.key === 'Escape') setMetadataFor(null); }} onClick={event => event.stopPropagation()} className="max-h-[80vh] w-full max-w-2xl min-w-0 space-y-4 overflow-auto rounded-xl bg-white p-4 shadow-xl dark:bg-slate-800"><div className="flex items-start justify-between gap-3 border-b border-slate-200 pb-3 dark:border-slate-700"><h3 id="artifact-metadata-title" className="break-all text-sm font-semibold">{t('artifacts.metadata')} · {metadataFor.name}</h3><button ref={metadataClose} className={button} aria-label={t('common.close')} onClick={() => setMetadataFor(null)}><X className="h-4 w-4" /></button></div><pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 font-mono text-xs dark:bg-slate-900">{JSON.stringify(metadataFor.metadata || {}, null, 2)}</pre></div></div>}
  </div>;
}
