import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Activity, AlertTriangle, ArrowRight, Ban, CheckCircle2, Circle, CircleAlert, Clock, Download, FlaskConical, ListChecks, Pause, Rocket } from 'lucide-react';
import { apiUrl } from '../../api/client';
import type { Artifact, Job } from '../../api/types';
import ProgressBar from '../../components/ProgressBar';
import { JobActions } from '../Queue/jobPresentation';
import { formatBytes, formatEta } from '../../utils/format';
import { artifactKindLabel, shortTime } from '../../utils/jobs';
import { samplingUrl } from '../../utils/samplingRoutes';
import { useWorkspaceText } from '../../utils/workspaceText';

export interface ReadinessCheck { key: string; label: string; detail: string; state: 'done' | 'todo' | 'warn'; href: string }

const RUNNING = ['running', 'pausing', 'cancelling'];
const WAITING = ['queued', 'scheduled'];
const finite = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;

function Shell({ tone, icon, title, subtitle, actions, children }: {
  tone: 'active' | 'paused' | 'danger' | 'success' | 'neutral' | 'prepare' | 'ready';
  icon: React.ReactNode; title: string; subtitle?: React.ReactNode; actions?: React.ReactNode; children?: React.ReactNode;
}) {
  return <section className="overview-banner" data-tone={tone} data-testid="overview-banner" aria-label={title}>
    <div className="overview-banner-head">
      <span className="overview-banner-icon" aria-hidden="true">{icon}</span>
      <div className="overview-banner-title"><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>
      {actions && <div className="overview-banner-actions">{actions}</div>}
    </div>
    {children}
  </section>;
}

export default function OverviewBanner({ versionName, job, artifact, checks, ready, next, archived, projectId, versionId, trainUrl, resultsUrl, loading, onJobUpdated }: {
  versionName: string; job?: Job; artifact?: Artifact; checks: ReadinessCheck[]; ready: boolean;
  next: { href: string; label: string }; archived: boolean; projectId: string; versionId?: string;
  trainUrl: string; resultsUrl: string; loading: boolean; onJobUpdated: () => void;
}) {
  const text = useWorkspaceText();
  const { t } = useTranslation();
  const status = job?.status || '';
  const jobUrl = job ? `/jobs/${encodeURIComponent(job.id)}` : '';
  const jobLink = job && <Link className="overview-banner-job" to={jobUrl}>{job.name}</Link>;
  const progress = job?.progress || {};
  const step = finite(progress.step);
  const total = finite(progress.total_steps);
  const steps = step !== null ? total ? `${step} / ${total} ${text('步', 'steps')}` : `${step} ${text('步', 'steps')}` : '';
  const actions = job && !archived && <JobActions job={job} onUpdated={onJobUpdated}/>;
  const progressRow = total ? <div className="overview-banner-progress">
    <ProgressBar label={text('训练进度', 'Training progress')} value={step ?? 0} max={total}/>
    <span>{step ?? 0} / {total} {text('步', 'steps')} · {Math.min(100, Math.floor((step ?? 0) / total * 100))}%</span>
  </div> : null;

  if (job && RUNNING.includes(status)) {
    const loss = finite(job.latest?.loss);
    const mean = finite(job.latest?.loss_mean);
    const speed = finite(progress.it_s);
    const perEpoch = finite(progress.steps_per_epoch);
    const epochs = total && perEpoch ? Math.ceil(total / perEpoch) : null;
    const metrics = [
      [text('轮数', 'Epoch'), progress.epoch != null ? `${progress.epoch}${epochs ? ` / ${epochs}` : ''}` : '—'],
      [text('当前损失', 'Loss'), loss === null ? '—' : loss.toFixed(4)],
      [job.latest?.loss_mean_scope === 'since_resume' ? text('恢复后平均损失', 'Mean loss since resume') : text('平均损失', 'Mean loss'), mean === null ? '—' : mean.toFixed(4)],
      [text('速度', 'Speed'), speed === null ? '—' : `${speed.toFixed(2)} ${text('步/秒', 'steps/s')}`],
      [text('预计剩余', 'Remaining'), status === 'running' && progress.eta_s != null ? formatEta(progress.eta_s) : '—'],
    ];
    const label = status === 'pausing' ? text('正在暂停', 'Pausing') : status === 'cancelling' ? text('正在取消', 'Cancelling') : text('训练中', 'Training');
    return <Shell tone="active" icon={<Activity size={18}/>} title={`${versionName} · ${label}`}
      subtitle={<>{jobLink}{job.started_at != null && <> · {text('开始于', 'Started')} {shortTime(job.started_at)}</>}</>}
      actions={<>{actions}<Link className="ui-btn ui-btn-sm ui-btn-primary" to={jobUrl}>{text('打开训练监控', 'Open monitor')}<ArrowRight size={13}/></Link></>}>
      {progressRow || <p className="overview-banner-note">{progress.phase ? t(`phase.${progress.phase}`, t('job.phaseInProgress')) : text('正在准备训练…', 'Preparing training…')}</p>}
      {total ? <dl className="overview-banner-metrics">{metrics.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl> : null}
    </Shell>;
  }
  if (job && status === 'paused') {
    return <Shell tone="paused" icon={<Pause size={18}/>} title={`${versionName} · ${text('已暂停', 'Paused')}`}
      subtitle={<>{jobLink}{steps && <> · {text('已完成', 'Reached')} {steps}</>}</>}
      actions={<>{actions}<Link className="ui-btn ui-btn-sm" to={jobUrl}>{text('查看任务', 'View job')}<ArrowRight size={13}/></Link></>}>
      {progressRow}
    </Shell>;
  }
  if (job && WAITING.includes(status)) {
    const reason = status === 'scheduled' && job.scheduled_at != null ? text(`${shortTime(job.scheduled_at)} 开始`, `Starts ${shortTime(job.scheduled_at)}`)
      : progress.wait_reason || text('设备空闲后自动开始', 'Starts when a device is free');
    return <Shell tone="active" icon={<Clock size={18}/>} title={`${versionName} · ${status === 'scheduled' ? text('已排期', 'Scheduled') : text('排队中', 'Queued')}`}
      subtitle={<>{jobLink} · {reason}</>}
      actions={<>{actions}<Link className="ui-btn ui-btn-sm" to="/queue">{text('任务队列', 'Queue')}<ArrowRight size={13}/></Link></>}/>;
  }
  if (job && status === 'failed') {
    const error = (job.error || '').trim();
    return <Shell tone="danger" icon={<AlertTriangle size={18}/>} title={`${versionName} · ${text('上次训练失败', 'Last run failed')}`}
      subtitle={<>{jobLink} · {shortTime(job.finished_at ?? job.created_at)}{steps && <> · {text('停在', 'Stopped at')} {steps}</>}</>}
      actions={<><Link className="ui-btn ui-btn-sm" to={`${jobUrl}?tab=logs`}>{text('查看日志', 'View log')}</Link><Link className="ui-btn ui-btn-sm" to={trainUrl}>{text('检查训练参数', 'Review parameters')}</Link>{actions}</>}>
      {error && <pre className="overview-banner-error" title={error}>{error}</pre>}
    </Shell>;
  }
  if (!job) {
    // Until the runs arrive the version's state is unknown; a readiness banner here would flash.
    if (loading) return <div className="overview-banner overview-banner-loading" data-tone="neutral" role="status" aria-label={text('正在读取版本状态…', 'Loading version status…')}><span className="ui-skeleton"/><span className="ui-skeleton"/></div>;
    if (archived) return null;
    const pending = checks.filter(check => check.state === 'todo').length;
    return <Shell tone={ready ? 'ready' : 'prepare'} icon={ready ? <Rocket size={18}/> : <ListChecks size={18}/>}
      title={`${versionName} · ${ready ? text('可以开始训练', 'Ready to train') : text(`还差 ${pending} 项准备`, `${pending} ${pending === 1 ? 'step' : 'steps'} before training`)}`}
      subtitle={ready ? text('数据和模型都已就绪，检查参数后即可开始训练。', 'Data and model are ready. Review the parameters, then start training.') : text('完成以下准备后即可开始训练。', 'Finish these steps to start training.')}
      actions={<Link className="ui-btn ui-btn-sm ui-btn-primary" to={next.href}>{next.label}<ArrowRight size={13}/></Link>}>
      <ul className="overview-checklist">{checks.map(check => <li key={check.key}><Link to={check.href} data-state={check.state}>
        {check.state === 'done' ? <CheckCircle2 size={16}/> : check.state === 'warn' ? <CircleAlert size={16}/> : <Circle size={16}/>}
        <span><strong>{check.label}</strong><small>{check.detail}</small></span>
      </Link></li>)}</ul>
    </Shell>;
  }
  const elapsed = job.started_at != null && job.finished_at != null ? job.finished_at - job.started_at : null;
  const output = artifact && <div className="overview-banner-output">
    <span><strong title={artifact.name}>{artifact.name}</strong><small>{[artifactKindLabel(artifact.kind, text), formatBytes(artifact.size), artifact.step != null ? `${artifact.step} ${text('步', 'steps')}` : ''].filter(Boolean).join(' · ')}</small></span>
    <a className="ui-btn ui-btn-sm ui-btn-primary" href={apiUrl(`/artifacts/${encodeURIComponent(artifact.id)}/download`)} download aria-label={text(`下载 ${artifact.name}`, `Download ${artifact.name}`)}><Download size={13}/>{text('下载', 'Download')}</a>
  </div>;
  if (status === 'completed') {
    return <Shell tone="success" icon={<CheckCircle2 size={18}/>} title={`${versionName} · ${text('训练完成', 'Training complete')}`}
      subtitle={<>{jobLink} · {shortTime(job.finished_at)}{steps && <> · {steps}</>}{elapsed != null && <> · {text('用时', 'took')} {formatEta(elapsed)}</>}</>}
      actions={<>{!archived && <Link className="ui-btn ui-btn-sm" to={samplingUrl(job.id, null, projectId, versionId)}><FlaskConical size={13}/>{text('测试模型', 'Test model')}</Link>}<Link className="ui-btn ui-btn-sm" to={resultsUrl}>{text('查看训练结果', 'View results')}<ArrowRight size={13}/></Link></>}>
      {output}
    </Shell>;
  }
  return <Shell tone="neutral" icon={<Ban size={18}/>} title={`${versionName} · ${text('训练已取消', 'Training cancelled')}`}
    subtitle={<>{jobLink} · {shortTime(job.finished_at ?? job.created_at)}{steps && <> · {text('停在', 'Stopped at')} {steps}</>}</>}
    actions={<><Link className="ui-btn ui-btn-sm" to={trainUrl}>{text('检查训练参数', 'Review parameters')}</Link>{actions}</>}>
    {output}
  </Shell>;
}
