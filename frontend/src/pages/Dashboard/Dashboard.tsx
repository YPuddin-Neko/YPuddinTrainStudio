import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, ArrowRight, FolderOpen, Plus, RefreshCw } from 'lucide-react';
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
export default function Dashboard() {
  const text = useWorkspaceText();
  const textRef = React.useRef(text); textRef.current = text;
  const [projects, setProjects] = React.useState<Project[]>([]);
  const [current, setCurrent] = React.useState<ContextJob | null>(null);
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
      const [projectResult, activeResult, waitingResult, historyResult] = await Promise.allSettled([
        apiClient.get<Project[] | { items: Project[] }>('/projects', { signal: controller.signal, silent: true }),
        apiClient.get<JobListResponse>('/jobs', { params: { group: 'active', page_size: 1 }, signal: controller.signal, silent: true }),
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
      setProjects([...(Array.isArray(projectRows) ? projectRows : projectRows?.items || [])].filter(project => !project.archived).sort((a, b) => b.updated_at - a.updated_at));
      setCurrent(active?.items[0] || null); setCounts({ active: active?.total ?? null, waiting: waiting?.total ?? null, history: history?.total ?? null });
      setError(failures.join('\n'));
    } catch (failure) { if (!controller.signal.aborted) setError(formatApiError(failure)); }
    finally { if (!controller.signal.aborted) setLoading(false); }
  }, []);
  React.useEffect(() => { void fetchOverview(); return () => { request.current?.abort(); if (refreshTimer.current) clearTimeout(refreshTimer.current); }; }, [fetchOverview]);
  const queueRefresh = () => { if (refreshTimer.current) clearTimeout(refreshTimer.current); refreshTimer.current = setTimeout(() => void fetchOverview(), 200); };
  useEventStream(EVENT_TYPES.JOB_STATE, queueRefresh);
  useEventStream(EVENT_TYPES.QUEUE_CHANGED, queueRefresh);
  useEventStream(EVENT_TYPES.JOB_STEP, event => setCurrent(job => job ? mergeJobEvent(job, event) : null));
  useEventStream(EVENT_TYPES.JOB_PHASE, event => setCurrent(job => job ? mergeJobEvent(job, event) : null));
  const last = projects[0];
  return <section className="task-workspace dashboard-overview" data-testid="dashboard-page">
    <header className="task-page-heading"><div><h1>{text('训练工作台', 'Training workspace')}</h1><p>{text('继续一个项目，或查看当前训练的进展。', 'Continue a project or check how your training is progressing.')}</p></div><Link className="task-button dashboard-primary" to="/projects"><Plus size={15}/>{text('新建或打开项目', 'Create or open a project')}</Link></header>
    {error && <div role="alert" className="task-error">{error}<button onClick={() => void fetchOverview()}>{text('重试', 'Retry')}</button></div>}
    <div className="dashboard-summary" aria-label={text('任务总览', 'Job overview')}>{([{ view: 'active', label: text('进行中与暂停', 'Active & paused'), value: counts.active }, { view: 'waiting', label: text('等待调度', 'Waiting'), value: counts.waiting }, { view: 'history', label: text('历史记录', 'History'), value: counts.history }] as const).map(item => <Link key={item.view} to={`/queue?view=${item.view}`}><strong>{loading ? '—' : item.value ?? '—'}</strong><span>{item.label}</span><ArrowRight size={14}/></Link>)}</div>
    <div className="dashboard-content-grid"><section className="dashboard-panel"><header><h2><Activity size={16}/>{text('当前任务', 'Current job')}</h2><Link to="/queue">{text('打开队列', 'Open queue')}<ArrowRight size={12}/></Link></header>{current ? <div className="dashboard-current"><JobStatus status={current.status}/><h3><Link to={`/jobs/${current.id}`}>{current.name}</Link></h3><JobContext job={current}/>{current.progress?.total_steps ? <div className="dashboard-run-progress"><span>{current.progress.step ?? 0} / {current.progress.total_steps} {text('步', 'steps')}</span><progress value={current.progress.step ?? 0} max={current.progress.total_steps}/></div> : <p>{text('详细阶段与日志可在任务监控中查看。', 'Open monitoring for detailed progress and logs.')}</p>}<Link className="task-button" to={`/jobs/${current.id}`}>{text('打开运行监控', 'Open monitoring')}<ArrowRight size={14}/></Link></div> : <div className="task-empty"><Activity size={26}/><strong>{loading ? text('读取任务…', 'Loading jobs…') : !activeAvailable ? text('当前任务暂时无法读取', 'Active jobs are temporarily unavailable') : text('现在没有运行或暂停的任务', 'No active or paused jobs')}</strong><span>{!activeAvailable ? text('请重试，或直接打开队列。', 'Retry or open the queue directly.') : counts.waiting ? text('有任务等待调度，可在队列查看原因。', 'Jobs are waiting. Check the queue for details.') : text('在项目中完成数据和参数准备后即可开始。', 'Start after preparing project data and settings.')}</span></div>}</section>
      <section className="dashboard-panel"><header><h2><FolderOpen size={16}/>{text('最近项目', 'Recent projects')}</h2><Link to="/projects">{text('所有项目', 'All projects')}<ArrowRight size={12}/></Link></header>{!projects.length ? <div className="task-empty"><FolderOpen size={26}/><strong>{loading ? text('读取项目…', 'Loading projects…') : projectsAvailable ? text('从第一个项目开始', 'Start with your first project') : text('项目列表暂时无法读取', 'Projects are temporarily unavailable')}</strong><Link to="/projects">{projectsAvailable ? text('创建项目与版本', 'Create a project and version') : text('打开项目列表', 'Open projects')}</Link></div> : <div className="dashboard-project-list">{projects.slice(0, 4).map(project => <Link key={project.id} to={projectUrl(project.id, project.active_version_id, 'overview')}><div><strong>{project.name}</strong><small>{project.id} · {project.version_count ?? 1} {text('个版本', 'versions')}</small></div><time>{formatTime(project.updated_at)}</time><ArrowRight size={14}/></Link>)}</div>}</section></div>
    <section className="dashboard-next"><h2>{last ? text(`继续“${last.name}”`, `Continue “${last.name}”`) : text('准备一次训练', 'Prepare a training run')}</h2><div>{[{ step: 'data', zh: '准备训练数据', en: 'Prepare training data', hintZh: '导入、查看标签、处理遮罩', hintEn: 'Import, inspect captions and edit masks' }, { step: 'train', zh: '检查参数并启动', en: 'Configure and start', hintZh: '选择模型、检查参数、开始训练', hintEn: 'Choose models, review the plan and launch' }, { step: 'results', zh: '检查训练结果', en: 'Review training results', hintZh: '选择训练记录，下载权重与采样', hintEn: 'Choose a run, download weights and samples' }].map((item, index) => <Link key={item.step} to={last ? projectUrl(last.id, last.active_version_id, item.step) : '/projects'}><b>{index + 1}</b><span><strong>{text(item.zh, item.en)}</strong><small>{text(item.hintZh, item.hintEn)}</small></span><ArrowRight size={14}/></Link>)}</div></section>
    <button className="task-link dashboard-refresh" disabled={loading} onClick={() => void fetchOverview()}><RefreshCw size={12}/>{text('刷新概览', 'Refresh overview')}</button>
  </section>;
}
