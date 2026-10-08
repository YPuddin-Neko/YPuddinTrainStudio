import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';
import { Activity, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { JobListResponse } from '../../api/types';
import { EVENT_TYPES } from '../../events/eventTypes';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';
import { formatApiError } from '../../utils/errors';
import { formatTime } from '../../utils/format';
import { projectUrl } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import { JobProgressSummary, JobStatus } from '../Queue/jobPresentation';
import StudioSelect from '../../components/StudioSelect';
import TtsResultsView from './TtsResultsView';

type Props = { projectId: string; versionId: string; engine?: string; readOnly?: boolean };
const PAGE_SIZE = 20;

export default function TtsVersionResults(props: Props) {
  return <VersionResults key={`${props.projectId}:${props.versionId}`} {...props}/>;
}

function VersionResults({ projectId, versionId, engine, readOnly = false }: Props) {
  const text = useWorkspaceText();
  const [, setParams] = useSearchParams();
  const [page, setPage] = React.useState(1);
  const [sourceJob, setSourceJob] = React.useState<{ id: string; name: string } | null>(null);
  const jobs = useQuery({
    queryKey: ['tts-version-result-jobs', projectId, versionId, page],
    queryFn: ({ signal }) => apiClient.get<JobListResponse>('/jobs', { params: { project_id: projectId, version_id: versionId, type: 'tts_train', page, page_size: PAGE_SIZE }, signal, silent: true }),
    refetchInterval: query => query.state.data?.items.some(job => ['queued', 'scheduled', 'running', 'cancelling'].includes(job.status)) ? 3000 : false,
  });
  const refresh = () => { void jobs.refetch({ cancelRefetch: false }); };
  useEventStream<{ project_id?: string; version_id?: string; job_id?: string }>(EVENT_TYPES.JOB_STATE, event => {
    if (!event.project_id || event.project_id === projectId && (!event.version_id || event.version_id === versionId)) refresh();
  });
  useEventStream<{ project_id?: string; version_id?: string }>(EVENT_TYPES.QUEUE_CHANGED, event => {
    if (!event.project_id || event.project_id === projectId && (!event.version_id || event.version_id === versionId)) refresh();
  });
  const connection = useEventStreamStatus(), previousConnection = React.useRef(connection);
  React.useEffect(() => { if (connection === 'connected' && previousConnection.current !== 'connected') refresh(); previousConnection.current = connection; });
  const items = jobs.data?.items || [];
  const total = jobs.data?.total;
  const pages = Math.max(1, Math.ceil((total || 0) / PAGE_SIZE));
  React.useEffect(() => { if (total !== undefined && page > pages) setPage(pages); }, [page, pages, total]);
  const selectOutputs = (job: { id: string; name: string }) => {
    setSourceJob({ id: job.id, name: job.name });
    setParams(previous => { const next = new URLSearchParams(previous); next.set('result_tab', 'artifacts'); return next; });
  };
  const choices = sourceJob && !items.some(job => job.id === sourceJob.id) ? [sourceJob, ...items] : items;
  const pagination = (page > 1 || pages > 1) && <nav className="results-pagination" aria-label={text('训练任务分页', 'Training run pages')}><span>{total == null ? text('正在读取训练记录…', 'Loading training records…') : text(`共 ${total} 个任务 · 每页 ${PAGE_SIZE} 个`, `${total} jobs · ${PAGE_SIZE} per page`)}</span><div><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一页任务', 'Previous jobs page')} disabled={jobs.isFetching || page <= 1} onClick={() => setPage(value => value - 1)}><ChevronLeft size={14}/></button><span>{page} / {Math.max(page, pages)}</span><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一页任务', 'Next jobs page')} disabled={jobs.isFetching || page >= pages} onClick={() => setPage(value => value + 1)}><ChevronRight size={14}/></button></div></nav>;
  const failure = jobs.error && <div className="results-error" role="alert"><span>{formatApiError(jobs.error)}</span><button type="button" className="ui-btn ui-btn-sm" disabled={jobs.isFetching} onClick={refresh}>{text('重试', 'Retry')}</button></div>;
  const selector = <>{failure}<div className="results-sample-toolbar results-output-selector"><label><span>{text('训练任务', 'Training job')}</span><StudioSelect aria-label={text('结果所属任务', 'Result source job')} value={sourceJob?.id || ''} disabled={jobs.isPending} onValueChange={value => { const job = choices.find(item => item.id === value); setSourceJob(job ? { id: job.id, name: job.name } : null); }} options={[{ value: '', label: text('此版本全部训练', 'All training runs in this version') }, ...choices.map(job => ({ value: job.id, label: job.name }))]}/></label>{sourceJob && <Link className="ui-link" to={`/jobs/${encodeURIComponent(sourceJob.id)}`}>{text('打开任务详情', 'Open job details')}</Link>}{pagination}</div></>;
  const content = <>
    {failure}{pagination}
    {jobs.isPending ? <p className="results-empty" role="status"><Loader2 size={16} className="animate-spin"/>{text('正在读取训练记录…', 'Loading training records…')}</p> : !items.length && !jobs.error ? <div className="results-empty"><Activity size={22}/><p>{text('此版本还没有训练记录', 'This version has no training records yet')}</p>{!readOnly && <Link className="ui-link" to={projectUrl(projectId, versionId, 'train')}>{text('配置并启动训练', 'Configure and start training')}</Link>}</div> : !!items.length && <div className="results-table-wrap" tabIndex={0} role="region" aria-label={text('当前版本训练记录', 'Training records in this version')}><table className="results-job-table tts-training-records" aria-label={text('当前版本训练记录', 'Training records in this version')}><thead><tr><th scope="col">{text('任务', 'Job')}</th><th scope="col">{text('状态', 'Status')}</th><th scope="col">{text('进度', 'Progress')}</th><th scope="col">{text('创建时间', 'Created')}</th><th scope="col">{text('结果', 'Results')}</th></tr></thead><tbody>{items.map(job => <tr key={job.id}>
      <td><Link to={`/jobs/${encodeURIComponent(job.id)}`}>{job.name}</Link><small>{job.id}</small></td>
      <td><JobStatus status={job.status}/></td><td className="results-progress"><JobProgressSummary job={job}/></td><td><time>{formatTime(job.created_at)}</time></td>
      <td><div className="results-row-actions"><Link className="ui-link" to={`/jobs/${encodeURIComponent(job.id)}`}>{text('监控与日志', 'Monitor & logs')}</Link><button type="button" className="ui-link" onClick={() => selectOutputs(job)}>{text('查看产物', 'View outputs')}</button></div></td>
    </tr>)}</tbody></table></div>}
  </>;
  return <TtsResultsView key={sourceJob?.id || 'version'} scope={sourceJob ? { jobId: sourceJob.id } : { projectId, versionId }} engine={engine} trainingRecords={{ content, selector, projectId, total, loading: jobs.isFetching, refresh }}/>;
}
