import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import { Activity, ArrowRight, CheckCircle2, Circle, Database, Image, Layers, Loader2, SlidersHorizontal, Tag, TriangleAlert } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { DatasetInfo, DatasetSource, JobListResponse } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatApiError } from '../../utils/errors';
import { mergeConfig } from '../../utils/config';
import { modelConfigUrl, projectUrl, type ProjectVersion, type VersionedProject } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import { inactiveTrainingReason } from '../../utils/trainingFamilies';
import './project-overview.css';

export interface OverviewDataset {
  source: DatasetSource;
  stats?: DatasetInfo['stats'];
  index_status?: string;
}

export interface ProjectOverviewProps {
  project: VersionedProject;
  version?: ProjectVersion | null;
  versionId?: string;
  config: Record<string, any>;
  datasets: OverviewDataset[];
}

const number = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const fileName = (value: unknown) => typeof value === 'string' ? value.trim().replace(/\\/g, '/').split('/').filter(Boolean).pop() || '' : '';
const familyName = (family: string) => ({ anima: 'Anima', krea2: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2: 'FLUX.2 Klein', toy: 'Toy' }[family] || family);

/** Missing, failed and in-progress indexes must never look like an empty dataset. */
function overviewDatasetStats(datasets: OverviewDataset[]) {
  const ready = datasets.every(row => row.stats && !row.stats.error && !['indexing', 'failed', 'stale'].includes(row.index_status || '')
    && ['images', 'captioned', 'masks'].every(key => number(row.stats?.[key]) !== null));
  const total = (key: 'images' | 'captioned' | 'masks', rows = datasets) => ready ? rows.reduce((sum, row) => sum + (number(row.stats?.[key]) ?? 0), 0) : null;
  return { ready, images: total('images'), captions: total('captioned'), masks: total('masks'), training: total('images', datasets.filter(row => !row.source.is_reg)), regularization: total('images', datasets.filter(row => row.source.is_reg)) };
}

export default function ProjectOverview({ project, version, versionId, config: savedConfig, datasets }: ProjectOverviewProps) {
  const text = useWorkspaceText();
  const { t, i18n } = useTranslation();
  const scopedVersionId = versionId || version?.id;
  const defaultsQuery = useQuery({
    queryKey: ['config-defaults'],
    queryFn: () => apiClient.get<Record<string, any>>('/config/defaults', { silent: true }),
    staleTime: 5 * 60 * 1000,
  });
  const config = mergeConfig(defaultsQuery.data || {}, savedConfig);
  const jobsQuery = useQuery({
    queryKey: ['project-overview-jobs', project.id, scopedVersionId],
    queryFn: () => apiClient.get<JobListResponse>('/jobs', { params: { project_id: project.id, version_id: scopedVersionId, type: 'train', page: 1, page_size: 3 }, silent: true }),
    refetchInterval: query => query.state.data?.items.some(job => ['queued', 'scheduled', 'running', 'pausing', 'cancelling'].includes(job.status)) ? 5000 : false,
  });
  useEventStream(EVENT_TYPES.JOB_STATE, () => { void jobsQuery.refetch(); });
  const stats = overviewDatasetStats(datasets);
  const jobs = jobsQuery.data?.items || [];
  const latest = jobs[0];
  const family = config.model?.family || version?.family || project.active_family || '';
  const inactiveReason = inactiveTrainingReason(config, i18n.language.startsWith('en'));
  const modelLabel = inactiveReason ? `${family === 'flux' ? 'FLUX.1' : 'FLUX.2 dev'} · ${text('已停用', 'Retired')}` : familyName(family);
  const baseModel = fileName(config.model?.dit_path);
  const hasTrainingImages = stats.training !== null && stats.training > 0;
  const indexing = datasets.some(row => row.index_status === 'indexing');
  const failedIndex = datasets.some(row => row.index_status === 'failed' || row.stats?.error);
  const missingCaptions = stats.images !== null && stats.captions !== null ? Math.max(0, stats.images - stats.captions) : null;
  const dataWorkspaceUrl = projectUrl(project.id, scopedVersionId, 'data');
  const dataUrl = `${dataWorkspaceUrl}&data_step=import#version-datasets`;
  const captionsUrl = `${dataWorkspaceUrl}&data_step=captions`;
  const masksUrl = `${dataWorkspaceUrl}&data_step=paint`;
  const trainUrl = projectUrl(project.id, scopedVersionId, 'train');
  const resultsUrl = projectUrl(project.id, scopedVersionId, 'results');
  const modelsUrl = modelConfigUrl(project.id, scopedVersionId);
  const archived = project.archived || version?.archived;
  const nextUrl = !hasTrainingImages || failedIndex ? dataUrl : !baseModel ? modelsUrl : trainUrl;
  const nextLabel = !hasTrainingImages || failedIndex ? text('整理训练数据', 'Prepare training data') : !baseModel ? text('配置训练模型', 'Configure models') : text('检查训练参数', 'Review parameters');
  const date = (value: unknown) => number(value) && Number(value) > 0 ? new Date(Number(value) * 1000).toLocaleDateString(i18n.resolvedLanguage || 'zh-CN', { month: 'short', day: 'numeric' }) : '—';
  const checks = [
    { label: text('训练图片', 'Training images'), detail: !stats.ready ? text('等待索引完成', 'Waiting for indexing') : hasTrainingImages ? text(`${stats.training} 张训练图片`, `${stats.training} training images`) : text('尚未导入训练图片', 'No training images imported'), done: hasTrainingImages, href: dataUrl },
    { label: text('训练底模', 'Base model'), detail: baseModel ? text('已配置', 'Configured') : text('尚未选择底模', 'No base model selected'), done: !!baseModel, href: modelsUrl },
    { label: text('标签覆盖', 'Caption coverage'), detail: missingCaptions === null ? text('等待索引完成', 'Waiting for indexing') : missingCaptions > 0 ? text(`${missingCaptions} 张图片没有标签文件`, `${missingCaptions} images have no caption file`) : hasTrainingImages ? text('所有图片都有标签文件', 'Every image has a caption file') : text('导入后可检查标签', 'Check captions after import'), done: hasTrainingImages && missingCaptions === 0, href: captionsUrl },
  ];
  const parameters = [
    [text('算法', 'Algorithm'), config.adapter?.algo ? ({ lora: 'LoRA', lokr: 'LoKr', loha: 'LoHa', full: text('完整权重', 'Full weights') }[String(config.adapter.algo)] || String(config.adapter.algo)) : '—'],
    ['Rank', config.adapter?.rank ?? '—'],
    [text('批量大小', 'Batch size'), config.dataset?.batch_size ?? '—'],
    [text('学习率', 'Learning rate'), config.optimizer?.lr ?? '—'],
    [text('训练轮数', 'Epochs'), config.loop?.epochs ?? (config.loop?.epochs === null || defaultsQuery.isSuccess ? text('未设置', 'Not set') : '—')],
    [text('最大步数', 'Step limit'), config.loop?.max_steps ?? (config.loop?.max_steps === null || defaultsQuery.isSuccess ? text('未设置', 'Not set') : '—')],
  ];

  return <section className="project-overview" data-testid="project-overview">
    <header className="overview-heading">
      <div><span className="overview-eyebrow">{text('版本概览', 'VERSION OVERVIEW')}</span><h2>{version?.name || text('当前版本', 'Current version')}{archived && <span className="overview-badge">{text('已归档', 'Archived')}</span>}</h2>{(version?.note?.trim() || project.note?.trim()) && <p>{version?.note?.trim() || project.note?.trim()}</p>}</div>
      <div className="overview-version-meta"><span>{text('最后更新', 'Updated')} {date(version?.updated_at || project.updated_at)}</span>{number(project.version_count) !== null && <span>{project.version_count} {text('个版本', 'versions')}</span>}</div>
    </header>

    <div className="overview-metrics" aria-label={text('当前版本数据统计', 'Current version data statistics')}>
      {[
        { label: text('图片总数', 'Images'), value: stats.images, icon: <Image size={17}/>, href: `${dataWorkspaceUrl}&data_step=inspect` },
        { label: text('标签文件', 'Captions'), value: stats.captions, icon: <Tag size={17}/>, href: captionsUrl },
        { label: text('遮罩文件', 'Masks'), value: stats.masks, icon: <Layers size={17}/>, href: masksUrl },
        { label: text('数据来源', 'Sources'), value: datasets.length, icon: <Database size={17}/>, href: dataUrl },
      ].map(metric => <Link key={metric.label} to={metric.href} className="overview-metric"><span>{metric.icon}{metric.label}</span><strong>{metric.value ?? '—'}</strong><small>{metric.value === null ? text('待索引', 'Awaiting index') : metric.label === text('数据来源', 'Sources') ? text('查看数据集', 'View datasets') : text('当前版本', 'Current version')}<ArrowRight size={12}/></small></Link>)}
    </div>

    <div className="overview-main-grid">
      <section className="overview-panel overview-readiness">
        <div className="overview-panel-heading"><h3>{text('数据准备', 'Data preparation')}</h3><span className={`overview-badge${indexing || failedIndex ? ' attention' : ''}`}>{!datasets.length ? text('尚未导入', 'Not imported') : indexing ? text('正在索引', 'Indexing') : failedIndex ? text('需要检查', 'Needs attention') : !stats.ready ? text('等待统计', 'Awaiting statistics') : text('索引已更新', 'Index up to date')}</span></div>
        <div className="overview-data-balance"><div><strong>{stats.training ?? '—'}</strong><span>{text('训练图片', 'Training images')}</span></div><div><strong>{stats.regularization ?? '—'}</strong><span>{text('正则图片', 'Regularization images')}</span></div></div>
        <ul className="overview-checks">{checks.map(check => <li key={check.label}><Link to={check.href}>{check.done ? <CheckCircle2 size={16} className="overview-check-done"/> : <Circle size={16}/>}<span><strong>{check.label}</strong><small>{check.detail}</small></span><ArrowRight size={14}/></Link></li>)}</ul>
        {config.dataset?.masked_loss && stats.masks === 0 && hasTrainingImages && <p className="overview-inline-note"><TriangleAlert size={15}/>{text('已启用遮罩训练 · 未索引到遮罩文件', 'Masked training enabled · No mask files indexed')}</p>}
        <footer className="overview-panel-footer"><Link to={archived ? dataUrl : nextUrl} className="overview-primary-link">{archived ? text('查看版本数据', 'View version data') : nextLabel}<ArrowRight size={14}/></Link></footer>
      </section>

      <section className="overview-panel overview-configuration">
        <div className="overview-panel-heading"><h3>{text('模型与训练配置', 'Model and training configuration')}</h3><Link to={trainUrl}>{text('查看参数', 'View parameters')}<ArrowRight size={13}/></Link></div>
        <Link to={modelsUrl} className="overview-model"><div className="overview-model-icon"><Layers size={22}/></div><div><span>{family ? modelLabel : text('未选择模型族', 'No model family selected')}</span><strong>{baseModel || text('尚未配置训练底模', 'Base model not configured')}</strong><small>{inactiveReason || (baseModel ? text('已保存的模型选择', 'Saved model selection') : text('选择底模、文本编码器和 VAE', 'Choose a base model, text encoder and VAE'))}</small></div><ArrowRight size={16}/></Link>
        <dl className="overview-parameters">{parameters.map(([label, value]) => <div key={String(label)}><dt>{label}</dt><dd>{String(value)}</dd></div>)}</dl>
        {defaultsQuery.error && <div role="alert" className="overview-job-error"><span>{text('默认参数读取失败', 'Could not load default parameters')}</span><button type="button" onClick={() => void defaultsQuery.refetch()}>{text('重新读取默认参数', 'Reload default parameters')}</button></div>}
      </section>
    </div>

    <section className="overview-panel overview-training">
      <div className="overview-panel-heading"><h3>{text('训练动态', 'Training activity')}</h3><Link to={resultsUrl}>{text('查看版本结果', 'View version results')}<ArrowRight size={13}/></Link></div>
      {jobsQuery.isPending ? <div role="status" className="overview-job-empty"><Loader2 size={17} className="animate-spin"/>{text('正在读取训练记录…', 'Loading training history…')}</div>
        : jobsQuery.error ? <div role="alert" className="overview-job-error"><span>{formatApiError(jobsQuery.error)}</span><button type="button" onClick={() => void jobsQuery.refetch()}>{text('重试', 'Retry')}</button></div>
          : !latest ? <div className="overview-job-empty"><Activity size={23}/><div><strong>{text('这个版本还没有训练记录', 'No training runs for this version')}</strong><p>{text('准备好数据和模型后，在训练参数页检查计划并创建任务。', 'Prepare the data and model, then review the plan and create a training job.')}</p></div><Link to={trainUrl}>{text('查看训练参数', 'View training parameters')}<ArrowRight size={14}/></Link></div>
            : <div className="overview-jobs">{jobs.map(job => <Link key={job.id} to={`/jobs/${encodeURIComponent(job.id)}`} className="overview-job-row"><Activity size={16}/><div><strong>{job.name}</strong><small>{date(job.created_at)}{job.progress?.step != null ? ` · ${job.progress.step} / ${job.progress.total_steps ?? '—'} ${text('步', 'steps')}` : ''}</small></div><span className="overview-job-status" data-status={job.status}>{t(`queue.status.${job.status}`, job.status)}</span><span className="overview-job-action">{job.status === 'failed' ? text('查看原因', 'View error') : text('查看任务', 'View job')}<ArrowRight size={14}/></span></Link>)}</div>}
    </section>

    <nav className="overview-shortcuts" aria-label={text('版本工作入口', 'Version workspace links')}>
      {[{ href: dataUrl, icon: <Database size={19}/>, title: text('训练数据', 'Training data'), note: text('导入、标签与遮罩', 'Import, captions and masks') }, { href: trainUrl, icon: <SlidersHorizontal size={19}/>, title: text('训练参数', 'Training parameters'), note: text('配置模型、检查计划与启动训练', 'Configure, plan and start training') }, { href: resultsUrl, icon: <Layers size={19}/>, title: text('训练结果', 'Training results'), note: text('查看权重、采样图与训练记录', 'Weights, samples and training history') }].map(item => <Link key={item.href} to={item.href}>{item.icon}<span><strong>{item.title}</strong><small>{item.note}</small></span><ArrowRight size={15}/></Link>)}
    </nav>
  </section>;
}
