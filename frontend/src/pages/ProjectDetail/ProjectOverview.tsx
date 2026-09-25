import OverviewDataPanel from './OverviewDataPanel';
import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import { ArrowRight, Download, FolderPlus, Images, Loader2, Pencil } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { Artifact, DatasetInfo, DatasetSource, JobListResponse } from '../../api/types';
import { useFamilies } from '../../api/hooks/useFamilies';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatBytes } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { mergeConfig } from '../../utils/config';
import { configOptionLabel } from '../../utils/configPresentation';
import { artifactKindLabel, focusJob, mergeJobEvent, shortTime } from '../../utils/jobs';
import { modelConfigUrl, projectUrl, type ProjectVersion, type VersionedProject } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import { inactiveTrainingReason } from '../../utils/trainingFamilies';
import { JobProgressSummary, JobStatus } from '../Queue/jobPresentation';
import { ProjectArtwork } from '../Projects/ProjectCardParts';
import ProjectEditor from '../Projects/ProjectEditor';
import { categoryLabel, type GalleryProject } from '../Projects/projectGallery';
import OverviewBanner, { type ReadinessCheck } from './OverviewBanner';
import '../Queue/queue.css';
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

const LIVE = ['queued', 'scheduled', 'running', 'pausing', 'cancelling'];
const number = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const fileName = (value: unknown) => typeof value === 'string' ? value.trim().replace(/\\/g, '/').split('/').filter(Boolean).pop() || '' : '';
const FAMILY_NAMES: Record<string, string> = { anima: 'Anima', krea2: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2: 'FLUX.2 Klein', toy: 'Toy' };
const learningRate = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? (value !== 0 && Math.abs(value) < 0.001 ? value.toExponential() : String(value)) : '—';

/** Missing, failed and in-progress indexes must never look like an empty dataset. */
function overviewDatasetStats(datasets: OverviewDataset[]) {
  const ready = datasets.every(row => row.stats && !row.stats.error && row.index_status === 'ready'
    && ['images', 'captioned', 'masks'].every(key => number(row.stats?.[key]) !== null));
  const total = (key: 'images' | 'captioned' | 'masks', rows = datasets) => ready ? rows.reduce((sum, row) => sum + (number(row.stats?.[key]) ?? 0), 0) : null;
  const training = datasets.filter(row => !row.source.is_reg);
  return { ready, images: total('images'), captions: total('captioned', training), masks: total('masks', training), training: total('images', training), regularization: total('images', datasets.filter(row => row.source.is_reg)) };
}

export default function ProjectOverview({ project, version, versionId, config: savedConfig, datasets: providedDatasets }: ProjectOverviewProps) {
  const text = useWorkspaceText();
  const { i18n } = useTranslation();
  const queryClient = useQueryClient();
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  const [editing, setEditing] = React.useState(false);
  const scopedVersionId = versionId || version?.id;
  const belongsToVersion = (item: { project_id?: string | null; version_id?: string | null }) => (item.project_id === undefined || item.project_id === project.id) && (item.version_id === undefined || item.version_id === (scopedVersionId || null));
  const datasets = providedDatasets.filter(row => belongsToVersion(row.source));
  const families = useFamilies();
  const defaultsQuery = useQuery({
    queryKey: ['config-defaults'],
    queryFn: () => apiClient.get<Record<string, any>>('/config/defaults', { silent: true }),
    staleTime: 5 * 60 * 1000,
  });
  const config = mergeConfig(defaultsQuery.data || {}, savedConfig);
  const jobsKey = ['project-overview-jobs', project.id, scopedVersionId];
  const activeKey = ['project-overview-active', project.id, scopedVersionId];
  const jobsQuery = useQuery({
    queryKey: jobsKey,
    queryFn: () => apiClient.get<JobListResponse>('/jobs', { params: { project_id: project.id, version_id: scopedVersionId, type: 'train', page: 1, page_size: 4 }, silent: true }),
    refetchInterval: query => query.state.data?.items.some(job => LIVE.includes(job.status)) ? 5000 : false,
  });
  const activeQuery = useQuery({
    queryKey: activeKey,
    queryFn: () => apiClient.get<JobListResponse>('/jobs', { params: { project_id: project.id, version_id: scopedVersionId, type: 'train', group: 'active', page: 1, page_size: 20 }, silent: true }),
    refetchInterval: query => query.state.data?.items.some(job => job.status !== 'paused') ? 5000 : false,
  });
  const artifactsQuery = useQuery({
    queryKey: ['project-overview-artifacts', project.id, scopedVersionId],
    queryFn: () => apiClient.get<Artifact[]>('/artifacts', { params: { project_id: project.id, version_id: scopedVersionId }, silent: true }),
  });
  const refreshJobs = () => { void jobsQuery.refetch(); void activeQuery.refetch(); };
  // Step events keep the banner live between the slower list refreshes.
  const applyEvent = (event: Record<string, any>) => {
    if (typeof event.job_id !== 'string') return;
    for (const key of [jobsKey, activeKey]) queryClient.setQueryData<JobListResponse>(key, data => data && { ...data, items: data.items.map(job => mergeJobEvent(job, event)) });
  };
  useEventStream(EVENT_TYPES.JOB_STEP, applyEvent);
  useEventStream(EVENT_TYPES.JOB_PHASE, applyEvent);
  useEventStream(EVENT_TYPES.JOB_STATE, event => { if (event.project_id && event.project_id !== project.id) return; applyEvent(event); refreshJobs(); });
  useEventStream(EVENT_TYPES.ARTIFACT_CREATED, event => { if (!event.project_id || event.project_id === project.id) void artifactsQuery.refetch(); });

  const stats = overviewDatasetStats(datasets);
  const jobs = [...new Map([...(activeQuery.data?.items || []), ...(jobsQuery.data?.items || [])].filter(belongsToVersion).map(job => [job.id, job])).values()]
    .sort((left, right) => right.created_at - left.created_at);
  const focus = focusJob(jobs);
  const history = jobs.slice(0, 4);
  const runTotal = Math.max(jobsQuery.data?.total ?? 0, jobs.length);
  const artifacts = (artifactsQuery.data || []).filter(belongsToVersion).sort((a, b) => b.created_at - a.created_at);
  const focusArtifact = focus && artifacts.filter(item => item.job_id === focus.id).sort((a, b) => (b.step ?? -1) - (a.step ?? -1) || b.created_at - a.created_at)[0];

  const family = config.model?.family || version?.family || project.active_family || '';
  const familyLabel = families.data?.find(item => item.name === family)?.label || FAMILY_NAMES[family] || family;
  const inactiveReason = inactiveTrainingReason(config, english);
  const baseModel = fileName(config.model?.dit_path);
  const hasTrainingImages = stats.training !== null && stats.training > 0;
  const indexing = datasets.some(row => row.index_status === 'indexing');
  const failedIndex = datasets.some(row => row.index_status === 'failed' || row.stats?.error);
  const missingCaptions = stats.training !== null && stats.captions !== null ? Math.max(0, stats.training - stats.captions) : null;
  const dataWorkspaceUrl = projectUrl(project.id, scopedVersionId, 'data');
  const dataUrl = `${dataWorkspaceUrl}&data_step=datasets#version-datasets`;
  const captionsUrl = `${dataWorkspaceUrl}&data_step=captions`;
  const masksUrl = `${dataWorkspaceUrl}&data_step=paint`;
  const trainUrl = projectUrl(project.id, scopedVersionId, 'train');
  const resultsUrl = projectUrl(project.id, scopedVersionId, 'results');
  const modelsUrl = modelConfigUrl(project.id, scopedVersionId);
  const archived = !!(project.archived || version?.archived);
  const versionName = version?.name || text('当前版本', 'Current version');

  const checks: ReadinessCheck[] = [
    { key: 'images', label: text('训练图片', 'Training images'), href: dataUrl,
      state: hasTrainingImages && !failedIndex ? 'done' : 'todo',
      detail: failedIndex ? text('数据集索引失败，需要检查', 'A dataset failed to index') : indexing || (!stats.ready && datasets.length) ? text('正在索引…', 'Indexing…') : hasTrainingImages ? text(`${stats.training} 张`, `${stats.training} images`) : text('尚未导入', 'None imported') },
    { key: 'model', label: text('训练底模', 'Base model'), href: modelsUrl,
      state: baseModel && !inactiveReason ? 'done' : 'todo',
      detail: inactiveReason ? text('模型类型已停用', 'Model type retired') : baseModel || text('尚未选择', 'Not selected') },
    { key: 'captions', label: text('标签', 'Captions'), href: captionsUrl,
      state: missingCaptions === null || !hasTrainingImages ? 'todo' : missingCaptions > 0 ? 'warn' : 'done',
      detail: missingCaptions === null ? text('等待索引', 'Waiting for index') : !hasTrainingImages ? text('导入图片后检查', 'Check after import') : missingCaptions > 0 ? text(`${missingCaptions} 张没有标签`, `${missingCaptions} without captions`) : text('全部已标注', 'All captioned') },
    ...(config.dataset?.masked_loss ? [{ key: 'masks', label: text('遮罩', 'Masks'), href: masksUrl,
      state: (stats.masks ?? 0) > 0 ? 'done' as const : 'warn' as const,
      detail: (stats.masks ?? 0) > 0 ? text(`${stats.masks} 个遮罩`, `${stats.masks} masks`) : text('已启用遮罩训练，但没有遮罩', 'Masked training on, no masks') }] : []),
  ];
  const ready = checks.every(check => check.state !== 'todo');
  const next = !hasTrainingImages || failedIndex ? { href: dataUrl, label: text('整理训练数据', 'Prepare training data') }
    : !baseModel || inactiveReason ? { href: modelsUrl, label: text('配置训练模型', 'Configure models') }
      : !stats.ready ? { href: dataUrl, label: text('查看数据集状态', 'View dataset status') }
        : { href: trainUrl, label: text('检查参数并开始训练', 'Review and start training') };

  const full = config.training?.mode === 'full';
  const epochs = number(config.loop?.epochs);
  const maxSteps = number(config.loop?.max_steps);
  const resolutions = Array.isArray(config.dataset?.resolutions) ? config.dataset.resolutions.filter((value: unknown) => typeof value === 'number') : [];
  const parameters: [string, React.ReactNode][] = [
    [text('算法', 'Algorithm'), full ? text('全量微调', 'Full fine-tuning') : config.adapter?.algo ? configOptionLabel('adapter.algo', String(config.adapter.algo), english) : '—'],
    full ? [text('训练组件', 'Trained parts'), [config.training?.train_backbone && 'UNet / DiT', config.training?.train_text_encoder && text('文本编码器', 'Text encoder')].filter(Boolean).join(' + ') || '—']
      : ['Rank / Alpha', config.adapter?.rank != null ? `${config.adapter.rank} / ${config.adapter.alpha ?? '—'}` : '—'],
    [text('学习率', 'Learning rate'), learningRate(config.optimizer?.lr)],
    [text('优化器', 'Optimizer'), config.optimizer?.type ? configOptionLabel('optimizer.type', String(config.optimizer.type), true) : '—'],
    [text('批量大小', 'Batch size'), config.dataset?.batch_size != null ? `${config.dataset.batch_size}${number(config.loop?.grad_accum) && config.loop.grad_accum > 1 ? ` × ${config.loop.grad_accum}` : ''}` : '—'],
    [text('训练长度', 'Length'), [epochs !== null && text(`${epochs} 轮`, `${epochs} epochs`), maxSteps !== null && text(`最多 ${maxSteps} 步`, `≤ ${maxSteps} steps`)].filter(Boolean).join(' · ') || (defaultsQuery.isSuccess ? text('未设置', 'Not set') : '—')],
    [text('分辨率', 'Resolution'), resolutions.length ? resolutions.join(' / ') : '—'],
    [text('混合精度', 'Precision'), config.loop?.mixed_precision ? configOptionLabel('loop.mixed_precision', String(config.loop.mixed_precision), true) : '—'],
  ];
  const counts = !stats.ready ? text('正在读取数据统计…', 'Reading dataset statistics…')
      : [text(`${stats.training} 张训练图片`, `${stats.training} training images`), text(`${stats.regularization} 张正则图片`, `${stats.regularization} regularization images`),
        text(`${stats.captions} / ${stats.training} 已标注`, `${stats.captions} / ${stats.training} captioned`), text(`${stats.masks} 个遮罩`, `${stats.masks} masks`),
        text(`${datasets.length} 个数据集`, `${datasets.length} datasets`)].join(' · ');
  const date = (value: unknown) => number(value) ? new Date(Number(value) * 1000).toLocaleDateString(i18n.resolvedLanguage || 'zh-CN', { year: 'numeric', month: 'short', day: 'numeric' }) : '—';
  const loading = jobsQuery.isPending || activeQuery.isPending;
  const jobError = jobsQuery.error || activeQuery.error;

  const runs = <section className="overview-panel overview-runs" aria-labelledby="overview-runs-title">
    <header className="overview-panel-heading"><div className="overview-panel-title"><h3 id="overview-runs-title">{text('训练记录', 'Training runs')}</h3>{runTotal > 0 && <span className="overview-count">{runTotal}</span>}</div>{runTotal > 0 && <Link className="ui-link" to={`${resultsUrl}&result_tab=jobs`}>{text('全部记录', 'All runs')}<ArrowRight size={13}/></Link>}</header>
    {loading ? <p role="status" className="overview-muted"><Loader2 size={14} className="animate-spin"/>{text('正在读取训练记录…', 'Loading training runs…')}</p>
      : jobError ? <div role="alert" className="overview-inline-error"><span>{formatApiError(jobError)}</span><button type="button" className="ui-btn ui-btn-sm" onClick={refreshJobs}>{text('重试', 'Retry')}</button></div>
        : history.length ? <ul className="overview-run-list">{history.map(job => <li key={job.id}><Link to={`/jobs/${encodeURIComponent(job.id)}`}>
          <span className="overview-run-top"><JobStatus status={job.status}/><time>{shortTime(job.created_at)}</time></span>
          <strong title={job.name}>{job.name}</strong>
          <JobProgressSummary job={job}/>
        </Link></li>)}</ul>
          : <p className="overview-muted">{text('还没有训练记录。开始训练后，这里会显示每次训练的进度和保存的模型权重。', 'No runs yet. Training progress and saved weights will appear here.')}</p>}
  </section>;
  const outputs = <section className="overview-panel overview-outputs" aria-labelledby="overview-outputs-title">
    <header className="overview-panel-heading"><div className="overview-panel-title"><h3 id="overview-outputs-title">{text('训练产物', 'Outputs')}</h3>{artifacts.length > 0 && <span className="overview-count">{artifacts.length}</span>}</div>{artifacts.length > 0 && <Link className="ui-link" to={`${resultsUrl}&result_tab=artifacts`}>{text('全部产物', 'All outputs')}<ArrowRight size={13}/></Link>}</header>
    {artifactsQuery.isPending ? <p role="status" className="overview-muted"><Loader2 size={14} className="animate-spin"/>{text('正在读取训练产物…', 'Loading outputs…')}</p>
      : artifactsQuery.error ? <div role="alert" className="overview-inline-error"><span>{formatApiError(artifactsQuery.error)}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => void artifactsQuery.refetch()}>{text('重新读取产物', 'Reload outputs')}</button></div>
        : artifacts.length ? <ul className="overview-output-list">{artifacts.slice(0, 3).map(artifact => <li key={artifact.id}>
          <span><strong title={artifact.name}>{artifact.name}</strong><small>{[artifactKindLabel(artifact.kind, text), formatBytes(artifact.size), artifact.step != null ? `${artifact.step} ${text('步', 'steps')}` : '', shortTime(artifact.created_at)].filter(Boolean).join(' · ')}</small></span>
          <a className="ui-btn ui-btn-sm ui-btn-icon ui-btn-quiet" href={apiUrl(`/artifacts/${encodeURIComponent(artifact.id)}/download`)} download aria-label={text(`下载 ${artifact.name}`, `Download ${artifact.name}`)} title={text('下载', 'Download')}><Download size={14}/></a>
        </li>)}</ul>
          : <p className="overview-muted">{text('训练保存的模型权重会显示在这里。', 'Saved model weights appear here.')}</p>}
  </section>;

  return <section className="project-overview" data-testid="project-overview">
    <OverviewBanner versionName={versionName} job={focus} artifact={focusArtifact} checks={checks} ready={ready} next={next} archived={archived}
      projectId={project.id} versionId={scopedVersionId} trainUrl={trainUrl} resultsUrl={resultsUrl} loading={loading} onJobUpdated={refreshJobs}/>
    <div className="overview-layout">
      <section className="overview-main" aria-labelledby="overview-data-title">
        <header className="overview-section-heading">
          <div><h2 id="overview-data-title">{text('训练数据', 'Training data')}</h2>{datasets.length > 0 && <p>{counts}</p>}</div>
          {datasets.length > 0 && <Link className="ui-btn ui-btn-sm" to={dataUrl}>{archived ? text('查看训练数据', 'View training data') : text('管理训练数据', 'Manage data')}<ArrowRight size={13}/></Link>}
        </header>
        {datasets.length ? <OverviewDataPanel key={`${project.id}/${scopedVersionId || 'legacy'}`} datasets={datasets} workspaceUrl={dataWorkspaceUrl} projectId={project.id} versionId={scopedVersionId}/>
          : <div className="overview-data-empty" data-testid="overview-data-empty">
            <span className="overview-data-empty-icon" aria-hidden="true"><Images size={22}/></span>
            <strong>{text('这个版本还没有训练数据', 'This version has no training data')}</strong>
            <p>{text('导入图片文件夹后，这里会显示图片预览、标签分布和尺寸分布。', 'Import an image folder to see previews, tag frequency and size distribution here.')}</p>
            {!archived && <Link className="ui-btn ui-btn-primary" to={`${dataWorkspaceUrl}&data_step=datasets`}><FolderPlus size={15}/>{text('导入训练数据', 'Import training data')}</Link>}
          </div>}
      </section>
      <aside className="overview-aside" aria-label={text('版本信息', 'Version details')}>
        <section className="overview-panel overview-configuration" aria-labelledby="overview-config-title">
          <header className="overview-panel-heading"><h3 id="overview-config-title">{text('模型与参数', 'Model & parameters')}</h3><Link className="ui-link" to={trainUrl}>{text('训练参数', 'Parameters')}<ArrowRight size={13}/></Link></header>
          <Link to={modelsUrl} className="overview-model" data-missing={!baseModel || !!inactiveReason || undefined}>
            <span>{family ? `${familyLabel}${inactiveReason ? ` · ${text('已停用', 'Retired')}` : ''}` : text('未选择模型类型', 'No model type')}</span>
            <strong title={baseModel || undefined}>{baseModel || text('尚未配置训练底模', 'Base model not configured')}</strong>
            {inactiveReason && <small>{inactiveReason}</small>}
          </Link>
          <dl className="overview-parameters">{parameters.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
          {defaultsQuery.error && <div role="alert" className="overview-inline-error"><span>{text('默认参数读取失败', 'Could not load default parameters')}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => void defaultsQuery.refetch()}>{text('重新读取默认参数', 'Reload default parameters')}</button></div>}
        </section>
        {runs}
        {(runTotal > 0 || artifacts.length > 0 || artifactsQuery.isPending || !!artifactsQuery.error) && outputs}
        <section className="overview-panel overview-about" aria-labelledby="overview-about-title">
          <header className="overview-panel-heading"><h3 id="overview-about-title">{text('关于项目', 'About')}</h3>{!project.archived && <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" onClick={() => setEditing(true)}><Pencil size={13}/>{text('编辑', 'Edit')}</button>}</header>
          <div className="overview-about-body">
            <div className="overview-about-art"><ProjectArtwork name={project.name} coverUrl={(project as GalleryProject).cover_url}/></div>
            <p data-empty={!project.note?.trim() || undefined}>{project.note?.trim() || text('没有项目备注', 'No project note')}</p>
          </div>
          <dl className="overview-facts">
            <div><dt>{text('分类', 'Category')}</dt><dd>{project.category ? categoryLabel(project.category, english) : text('未分类', 'Uncategorized')}</dd></div>
            <div><dt>{text('版本', 'Versions')}</dt><dd>{text(`${project.version_count ?? 1} 个`, `${project.version_count ?? 1}`)}</dd></div>
            {version?.note?.trim() && <div className="overview-fact-wide"><dt>{text('版本说明', 'Version note')}</dt><dd>{version.note.trim()}</dd></div>}
            <div><dt>{text('创建于', 'Created')}</dt><dd>{date(project.created_at)}</dd></div>
            <div><dt>{text('最后更新', 'Updated')}</dt><dd>{date(version?.updated_at || project.updated_at)}</dd></div>
          </dl>
        </section>
      </aside>
    </div>
    {editing && <ProjectEditor project={project as GalleryProject} categories={[]} onClose={() => setEditing(false)}
      onPartial={updated => queryClient.setQueryData(['project', project.id], updated)}
      onSaved={updated => { queryClient.setQueryData(['project', project.id], updated); setEditing(false); }}/>}
  </section>;
}
