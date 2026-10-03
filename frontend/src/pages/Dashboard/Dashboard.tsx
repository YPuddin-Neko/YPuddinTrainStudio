import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, ArrowRight, ChevronLeft, ChevronRight, FolderOpen, Plus, RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { JobListResponse, Project } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatTime } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { mergeJobEvent } from '../../utils/jobs';
import { projectUrl } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import { JobContext, JobStatus, type ContextJob } from '../Queue/jobPresentation';
import '../Queue/queue.css';
import './dashboard.css';
import PageLocation from '../../components/PageLocation';

const PAGE_SIZE = 5;
/** Steps can arrive many times a second; the job cards take them at most this often. */
const PROGRESS_INTERVAL = 500;

function DashboardPagination({ label, page, pages, disabled, onChange }: { label: string; page: number; pages: number; disabled?: boolean; onChange: (page: number) => void }) {
  const text = useWorkspaceText();
  if (pages <= 1) return null;
  return <nav className="dashboard-pagination" aria-label={label}>
    <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一页', 'Previous page')} disabled={disabled || page <= 1} onClick={() => onChange(page - 1)}><ChevronLeft size={14}/></button>
    <span>{page} / {pages}</span>
    <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一页', 'Next page')} disabled={disabled || page >= pages} onClick={() => onChange(page + 1)}><ChevronRight size={14}/></button>
  </nav>;
}

export default function Dashboard() {
  const text = useWorkspaceText();
  const textRef = React.useRef(text); textRef.current = text;
  const [projects, setProjects] = React.useState<Project[]>([]);
  const [current, setCurrent] = React.useState<ContextJob[]>([]);
  const [activePage, setActivePage] = React.useState(1);
  const [projectPage, setProjectPage] = React.useState(1);
  const requestedPage = React.useRef(1);
  const [counts, setCounts] = React.useState<{ active: number | null; waiting: number | null; history: number | null }>({ active: null, waiting: null, history: null });
  const [projectsAvailable, setProjectsAvailable] = React.useState(false);
  const [activeAvailable, setActiveAvailable] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const request = React.useRef<AbortController | null>(null);
  const refreshTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const fetchOverview = React.useCallback(async () => {
    request.current?.abort(); const controller = new AbortController(); request.current = controller;
    setLoading(true); setError('');
    try {
      const fetchActive = async () => {
        let page = requestedPage.current;
        let result = await apiClient.get<JobListResponse>('/jobs', { params: { group: 'active', page, page_size: PAGE_SIZE }, signal: controller.signal, silent: true });
        while (page > Math.max(1, Math.ceil(result.total / PAGE_SIZE)) && !controller.signal.aborted) {
          page = Math.max(1, Math.ceil(result.total / PAGE_SIZE));
          result = await apiClient.get<JobListResponse>('/jobs', { params: { group: 'active', page, page_size: PAGE_SIZE }, signal: controller.signal, silent: true });
        }
        return { ...result, page };
      };
      const [projectResult, activeResult, waitingResult, historyResult] = await Promise.allSettled([
        apiClient.get<Project[] | { items: Project[] }>('/projects', { signal: controller.signal, silent: true }),
        fetchActive(),
        apiClient.get<JobListResponse>('/jobs', { params: { group: 'waiting', page_size: 1 }, signal: controller.signal, silent: true }),
        apiClient.get<JobListResponse>('/jobs', { params: { group: 'history', page_size: 1 }, signal: controller.signal, silent: true }),
      ]);
      if (controller.signal.aborted) return;
      const failures: string[] = [];
      function value<T>(result: PromiseSettledResult<T>, label: string): T | undefined {
        if (result.status === 'fulfilled') return result.value;
        failures.push(`${label}: ${formatApiError(result.reason)}`); return undefined;
      }
      const projectRows = value(projectResult, textRef.current('项目列表', 'Projects'));
      const active = value(activeResult, textRef.current('当前任务', 'Active jobs'));
      const waiting = value(waitingResult, textRef.current('等待调度', 'Waiting jobs'));
      const history = value(historyResult, textRef.current('历史记录', 'Job history'));
      setProjectsAvailable(projectRows !== undefined); setActiveAvailable(active !== undefined);
      const visibleProjects = [...(Array.isArray(projectRows) ? projectRows : projectRows?.items || [])].filter(project => !project.archived).sort((a, b) => b.updated_at - a.updated_at);
      setProjects(visibleProjects);
      if (projectRows !== undefined) setProjectPage(page => Math.min(page, Math.max(1, Math.ceil(visibleProjects.length / PAGE_SIZE))));
      setCurrent(active?.items.slice(0, PAGE_SIZE) || []);
      if (active) { requestedPage.current = active.page; setActivePage(active.page); }
      setCounts({ active: active?.total ?? null, waiting: waiting?.total ?? null, history: history?.total ?? null });
      setError(failures.join('\n'));
    } catch (failure) { if (!controller.signal.aborted) setError(formatApiError(failure)); }
    finally { if (!controller.signal.aborted) setLoading(false); }
  }, []);
  const currentRef = React.useRef(current); currentRef.current = current;
  const pendingEvents = React.useRef<Record<string, unknown>[]>([]);
  const progressTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastProgress = React.useRef(0);
  const applyProgress = React.useCallback(() => {
    progressTimer.current = null; lastProgress.current = Date.now();
    const events = pendingEvents.current; pendingEvents.current = [];
    setCurrent(jobs => {
      const next = jobs.map(job => events.reduce<ContextJob>((merged, event) => mergeJobEvent(merged, event), job));
      return next.some((job, index) => job !== jobs[index]) ? next : jobs;
    });
  }, []);
  React.useEffect(() => { void fetchOverview(); return () => { request.current?.abort(); if (refreshTimer.current) clearTimeout(refreshTimer.current); if (progressTimer.current) clearTimeout(progressTimer.current); }; }, [fetchOverview]);
  const queueRefresh = () => { if (refreshTimer.current) clearTimeout(refreshTimer.current); refreshTimer.current = setTimeout(() => void fetchOverview(), 200); };
  // Only jobs on this page are updated, and a burst of steps renders once: the first at once, the rest together.
  const liveProgress = (event: Record<string, unknown>) => {
    if (!currentRef.current.some(job => job.id === event.job_id)) return;
    pendingEvents.current.push(event);
    if (progressTimer.current) return;
    const wait = PROGRESS_INTERVAL - (Date.now() - lastProgress.current);
    if (wait <= 0) applyProgress(); else progressTimer.current = setTimeout(applyProgress, wait);
  };
  useEventStream(EVENT_TYPES.JOB_STATE, queueRefresh);
  useEventStream(EVENT_TYPES.QUEUE_CHANGED, queueRefresh);
  useEventStream(EVENT_TYPES.JOB_STEP, liveProgress);
  useEventStream(EVENT_TYPES.JOB_PHASE, liveProgress);
  const last = projects[0];
  const activePages = Math.max(1, Math.ceil((counts.active ?? 0) / PAGE_SIZE));
  const projectPages = Math.max(1, Math.ceil(projects.length / PAGE_SIZE));
  return <section className="task-workspace dashboard-overview" data-testid="dashboard-page">
    <PageLocation trail={[{ label: text('仪表盘', 'Dashboard') }]}/><header className="task-page-heading"><div><h1>{text('训练工作台', 'Training workspace')}</h1></div><div className="dashboard-page-actions"><button type="button" className="ui-btn ui-btn-icon" aria-label={text('刷新概览', 'Refresh overview')} title={text('刷新概览', 'Refresh overview')} disabled={loading} onClick={() => void fetchOverview()}><RefreshCw size={15} className={loading ? 'animate-spin' : undefined}/></button><Link className="ui-btn ui-btn-primary" to="/projects"><Plus size={15}/>{text('新建或打开项目', 'Create or open a project')}</Link></div></header>
    {error && <div role="alert" className="task-error">{error}<button type="button" className="ui-btn ui-btn-sm" onClick={() => void fetchOverview()}>{text('重试', 'Retry')}</button></div>}
    <div className="dashboard-summary" aria-label={text('任务总览', 'Job overview')}>{([{ view: 'active', label: text('进行中与暂停', 'Active & paused'), value: counts.active }, { view: 'waiting', label: text('等待调度', 'Waiting'), value: counts.waiting }, { view: 'history', label: text('历史记录', 'History'), value: counts.history }] as const).map(item => <Link className="ui-card-interactive" key={item.view} to={`/queue?view=${item.view}`}><strong>{loading ? '—' : item.value ?? '—'}</strong><span>{item.label}</span><ArrowRight size={14}/></Link>)}</div>
    <div className="dashboard-content-grid">
      <section className="dashboard-panel" aria-label={text('当前任务', 'Current jobs')}>
        <header>
          <h2><Activity size={16}/>{text('当前任务', 'Current jobs')}</h2>
          <div className="dashboard-panel-actions">
            <Link className="ui-link" to="/queue">{text('打开队列', 'Open queue')}<ArrowRight size={12}/></Link>
            <DashboardPagination label={text('当前任务分页', 'Current jobs pagination')} page={activePage} pages={activePages} disabled={loading} onChange={page => { requestedPage.current = page; void fetchOverview(); }}/>
          </div>
        </header>
        {current.length ? <div className="dashboard-current-list" aria-busy={loading}>{current.map(job => <article key={job.id} className="dashboard-current">
          <div className="dashboard-job-heading"><JobStatus status={job.status}/><h3><Link to={`/jobs/${job.id}`}>{job.name}<ArrowRight size={14}/></Link></h3></div>
          <JobContext job={job}/>
          {job.progress?.total_steps ? <div className="dashboard-run-progress"><span>{job.progress.step ?? 0} / {job.progress.total_steps} {text('步', 'steps')}</span><progress aria-label={text(`${job.name}训练进度`, `${job.name} training progress`)} value={job.progress.step ?? 0} max={job.progress.total_steps}/></div> : null}
        </article>)}</div> : <div className="task-empty">
          <Activity size={26}/><strong>{loading ? text('读取任务…', 'Loading jobs…') : !activeAvailable ? text('当前任务暂时无法读取', 'Active jobs are temporarily unavailable') : text('现在没有运行或暂停的任务', 'No active or paused jobs')}</strong>
          {(!activeAvailable || (counts.waiting ?? 0) > 0) && <span>{!activeAvailable ? text('请重试，或直接打开队列。', 'Retry or open the queue directly.') : text('有任务等待调度，可在队列查看原因。', 'Jobs are waiting. Check the queue for details.')}</span>}
        </div>}
      </section>
      <section className="dashboard-panel" aria-label={text('最近项目', 'Recent projects')}>
        <header>
          <h2><FolderOpen size={16}/>{text('最近项目', 'Recent projects')}</h2>
          <div className="dashboard-panel-actions">
            <Link className="ui-link" to="/projects">{text('所有项目', 'All projects')}<ArrowRight size={12}/></Link>
            <DashboardPagination label={text('最近项目分页', 'Recent projects pagination')} page={projectPage} pages={projectPages} onChange={setProjectPage}/>
          </div>
        </header>
        {!projects.length ? <div className="task-empty"><FolderOpen size={26}/><strong>{loading ? text('读取项目…', 'Loading projects…') : projectsAvailable ? text('从第一个项目开始', 'Start with your first project') : text('项目列表暂时无法读取', 'Projects are temporarily unavailable')}</strong><Link className="ui-btn" to="/projects">{projectsAvailable ? text('创建项目与版本', 'Create a project and version') : text('打开项目列表', 'Open projects')}</Link></div> : <div className="dashboard-project-list">{projects.slice((projectPage - 1) * PAGE_SIZE, projectPage * PAGE_SIZE).map(project => <Link key={project.id} to={projectUrl(project.id, project.active_version_id, 'overview')}><div><strong>{project.name}</strong><small>{project.id} · {project.version_count ?? 1} {text('个版本', 'versions')}</small></div><time>{formatTime(project.updated_at)}</time><ArrowRight size={14}/></Link>)}</div>}
      </section>
    </div>
    <section className="dashboard-next"><h2>{last ? text(`继续“${last.name}”`, `Continue “${last.name}”`) : text('准备一次训练', 'Prepare a training run')}</h2><div>{[{ step: 'data', zh: '准备训练数据', en: 'Prepare training data' }, { step: 'train', zh: '检查参数并启动', en: 'Configure and start' }, { step: 'results', zh: '检查训练结果', en: 'Review training results' }].map((item, index) => <Link className="ui-card-interactive" key={item.step} to={last ? projectUrl(last.id, last.active_version_id, item.step) : '/projects'}><b>{index + 1}</b><span><strong>{text(item.zh, item.en)}</strong></span><ArrowRight size={14}/></Link>)}</div></section>
  </section>;
}
