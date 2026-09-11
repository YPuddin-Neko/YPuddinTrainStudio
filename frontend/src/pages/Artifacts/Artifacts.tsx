import React from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { apiClient, apiUrl } from '../../api/client';
import { Artifact } from '../../api/types';
import { formatBytes, formatTime } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { Box, Download, Trash2, FileJson, PackageOpen, RefreshCw, Search, X } from 'lucide-react';

// These are the formats currently implemented by the conversion API.
const CONVERT_FORMATS = ['comfyui', 'kohya'] as const;
const button = 'inline-flex min-h-9 items-center justify-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs hover:bg-slate-50 disabled:opacity-40 dark:border-slate-600 dark:hover:bg-slate-700';

export default function Artifacts({ embedded = false }: { embedded?: boolean }) {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const [params, setParams] = useSearchParams();
  const projectId = params.get('project_id') || params.get('project') || undefined;
  const jobId = params.get('job_id') || params.get('job') || undefined;
  const query = params.get('q') || '';
  const [artifacts, setArtifacts] = React.useState<Artifact[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [busy, setBusy] = React.useState<string | null>(null);
  const [error, setError] = React.useState('');
  const [metadataFor, setMetadataFor] = React.useState<Artifact | null>(null);
  const request = React.useRef<AbortController | null>(null);
  const metadataClose = React.useRef<HTMLButtonElement>(null);
  const fetchArtifacts = React.useCallback(async () => {
    request.current?.abort(); const controller = new AbortController(); request.current = controller;
    setLoading(true); setError(''); setArtifacts([]);
    try {
      const data = await apiClient.get<Artifact[]>('/artifacts', { params: { project_id: projectId }, signal: controller.signal, silent: true });
      if (!controller.signal.aborted) setArtifacts(Array.isArray(data) ? data : []);
    } catch (error) { if (!controller.signal.aborted) setError(formatApiError(error)); }
    finally { if (!controller.signal.aborted) setLoading(false); }
  }, [projectId]);
  React.useEffect(() => { void fetchArtifacts(); return () => request.current?.abort(); }, [fetchArtifacts]);
  React.useEffect(() => {
    if (!metadataFor) return;
    const previous = document.activeElement as HTMLElement | null; metadataClose.current?.focus();
    return () => previous?.focus();
  }, [metadataFor]);
  const action = async (id: string, run: () => Promise<unknown>) => {
    setBusy(id); setError('');
    try { await run(); await fetchArtifacts(); }
    catch (error) { setError(formatApiError(error)); }
    finally { setBusy(null); }
  };
  const clearFilter = () => { const next = new URLSearchParams(params); ['project', 'project_id', 'job', 'job_id', 'q'].forEach(key => next.delete(key)); setParams(next); };
  const filtered = artifacts.filter(artifact => (!jobId || artifact.job_id === jobId) && (!query || `${artifact.name} ${artifact.job_id || ''} ${artifact.algo || ''}`.toLowerCase().includes(query.toLowerCase())));

  return <div className={embedded ? 'min-w-0 space-y-4' : 'min-w-0 space-y-6'} data-testid="artifacts-page">
    <header className="flex flex-wrap items-start justify-between gap-3">
      <div><h2 className={`flex items-center gap-2 font-semibold ${embedded ? 'text-base' : 'text-2xl'}`}>{!embedded && <Box className="h-5 w-5 text-indigo-500" />}{text('训练产物', 'Training outputs')}</h2><p className="mt-1 text-xs text-slate-500">{text('下载训练权重，查看元数据，或转换已支持的格式。', 'Download trained weights, inspect metadata, or convert to supported formats.')}</p></div>
      <button className={button} disabled={loading} onClick={() => void fetchArtifacts()}><RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />{text('刷新产物', 'Refresh outputs')}</button>
    </header>
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
      <label className="flex min-w-0 flex-1 items-center gap-2"><Search className="h-4 w-4 shrink-0 text-slate-400" /><input aria-label={text('搜索训练产物', 'Search training outputs')} value={query} onChange={event => { const next = new URLSearchParams(params); if (event.target.value) next.set('q', event.target.value); else next.delete('q'); setParams(next, { replace: true }); }} placeholder={text('按文件名、任务或算法搜索', 'Search filename, job or algorithm')} className="min-w-0 flex-1 bg-transparent text-sm outline-offset-4" /></label>
      <span className="text-xs text-slate-500">{filtered.length} {text('个产物', 'outputs')}</span>
      {(projectId || jobId || query) && <button className="text-xs text-blue-600 hover:underline" onClick={clearFilter}>{text('清除筛选', 'Clear filters')}</button>}
      {projectId && <Link className="break-all text-xs text-blue-600" to={`/projects/${encodeURIComponent(projectId)}`}>{text('项目', 'Project')} · {projectId}</Link>}
    </div>
    {error && <div role="alert" className="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-300"><p className="whitespace-pre-wrap break-words">{error}</p><button className="mt-2 underline" onClick={() => void fetchArtifacts()}>{t('common.retry')}</button></div>}
    <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">
      {loading ? <p role="status" className="p-6 text-sm text-slate-500">{t('common.loading')}</p> : filtered.length === 0 ? <div className="flex flex-col items-center gap-2 p-10 text-center"><PackageOpen className="h-9 w-9 text-slate-400" /><p className="text-sm font-medium text-slate-500">{query || projectId || jobId ? text('没有匹配的训练产物', 'No matching training outputs') : t('artifacts.empty')}</p><p className="text-xs text-slate-500">{text('训练产生权重文件后会显示在这里，也可在任务详情查看检查点。', 'Generated weights appear here. Checkpoints are also available in job details.')}</p><Link to="/queue" className="mt-2 text-sm text-blue-600">{text('查看训练任务', 'View training jobs')}</Link></div> : <table className="w-full text-left text-sm">
        <thead className="border-b border-slate-200 bg-slate-50 text-xs text-slate-500 dark:border-slate-700 dark:bg-slate-900/50"><tr><th className="px-3 py-2.5">{t('artifacts.name')}</th><th className="px-3 py-2.5">{t('artifacts.job')}</th><th className="px-3 py-2.5">{t('artifacts.algo')}</th><th className="px-3 py-2.5">{t('artifacts.rankAlphaFactor')}</th><th className="px-3 py-2.5">{t('artifacts.size')}</th><th className="px-3 py-2.5">{t('artifacts.created')}</th><th className="px-3 py-2.5 text-right">{t('artifacts.actions')}</th></tr></thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-700">{filtered.map(artifact => <tr key={artifact.id} className="hover:bg-slate-50 dark:hover:bg-slate-900/30" data-testid={`artifact-row-${artifact.id}`}>
          <td className="max-w-60 break-words px-3 py-3 font-mono text-xs font-medium">{artifact.name}</td><td className="px-3 py-3 font-mono text-xs text-slate-500">{artifact.job_id ? <Link to={`/jobs/${encodeURIComponent(artifact.job_id)}`} className="text-blue-600 hover:underline">{artifact.job_id}</Link> : '—'}</td><td className="px-3 py-3 text-xs">{artifact.algo || '—'}{artifact.kind && <span className="ml-1 text-slate-400">({artifact.kind})</span>}</td><td className="whitespace-nowrap px-3 py-3 font-mono text-xs">{artifact.rank ?? '—'} / {artifact.alpha ?? '—'} / {artifact.factor ?? '—'}</td><td className="whitespace-nowrap px-3 py-3 font-mono text-xs">{formatBytes(artifact.size)}</td><td className="whitespace-nowrap px-3 py-3 text-xs text-slate-500">{formatTime(artifact.created_at)}</td>
          <td className="px-3 py-3"><div className="flex items-center justify-end gap-1.5"><a href={apiUrl(`/artifacts/${encodeURIComponent(artifact.id)}/download`)} className={button} aria-label={`${t('common.download')}: ${artifact.name}`} title={t('common.download')}><Download className="h-4 w-4" /></a><button className={button} aria-label={`${text('查看元数据', 'View metadata')}: ${artifact.name}`} title={text('查看元数据', 'View metadata')} onClick={() => setMetadataFor(artifact)}><FileJson className="h-4 w-4" /></button><select className="min-h-9 rounded-md border border-slate-200 px-2 py-1 text-xs disabled:opacity-40 dark:border-slate-600 dark:bg-slate-900" aria-label={`${text('转换格式', 'Convert format')}: ${artifact.name}`} disabled={!!busy} defaultValue="" onChange={event => { const format = event.target.value; if (format) void action(artifact.id, () => apiClient.post(`/artifacts/${artifact.id}/convert`, { format }, { silent: true })); event.target.value = ''; }}><option value="" disabled>{busy === artifact.id ? t('artifacts.converting') : t('artifacts.convert')}</option>{CONVERT_FORMATS.map(format => <option key={format} value={format}>{format}</option>)}</select><button className={`${button} text-red-600`} aria-label={`${text('移除产物记录', 'Remove output record')}: ${artifact.name}`} disabled={!!busy} onClick={() => { if (window.confirm(text(`从产物列表移除 ${artifact.name}？磁盘中的权重文件会保留。`, `Remove ${artifact.name} from the output list? Its weight file will remain on disk.`))) void action(artifact.id, () => apiClient.delete(`/artifacts/${artifact.id}`, { silent: true })); }}><Trash2 className="h-4 w-4" /></button></div></td>
        </tr>)}</tbody>
      </table>}
    </div>
    {metadataFor && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={() => setMetadataFor(null)}><div role="dialog" aria-modal="true" aria-labelledby="artifact-metadata-title" onKeyDown={event => { if (event.key === 'Escape') setMetadataFor(null); }} onClick={event => event.stopPropagation()} className="max-h-[80vh] w-full max-w-2xl min-w-0 space-y-4 overflow-auto rounded-xl bg-white p-4 shadow-xl dark:bg-slate-800"><div className="flex items-start justify-between gap-3 border-b border-slate-200 pb-3 dark:border-slate-700"><h3 id="artifact-metadata-title" className="break-all text-sm font-semibold">{t('artifacts.metadata')} · {metadataFor.name}</h3><button ref={metadataClose} className={button} aria-label={t('common.close')} onClick={() => setMetadataFor(null)}><X className="h-4 w-4" /></button></div><pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 font-mono text-xs dark:bg-slate-900">{JSON.stringify(metadataFor.metadata || {}, null, 2)}</pre></div></div>}
  </div>;
}
