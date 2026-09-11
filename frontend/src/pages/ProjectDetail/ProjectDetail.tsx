import React from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Activity, ArrowLeft, ArrowRight, Database, Download, Image, Loader2, RefreshCw } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { Artifact, DatasetInfo, DatasetSource, Job, JobListResponse, Project } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { ProjectWorkflow, NextStepLink } from '../../components/ProjectWorkflow';
import { useWorkspaceText } from '../../utils/workspaceText';
import { mergeJobEvent } from '../../utils/jobs';
import { formatApiError } from '../../utils/errors';
import { formatBytes, formatTime } from '../../utils/format';
import ProjectDataImport from './ProjectDataImport';
import ProjectModelSetup from './ProjectModelSetup';

type WorkspaceDataset = { source: DatasetSource; stats?: DatasetInfo['stats']; index_status?: string };

export default function ProjectDetail() {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const { id } = useParams<{ id: string }>();
  const [params] = useSearchParams();
  const step = params.get('step') === 'models' ? 'models' : params.get('step') === 'results' ? 'results' : 'data';
  const [project, setProject] = React.useState<Project | null>(null);
  const [config, setConfig] = React.useState<Record<string, any>>({});
  const [datasets, setDatasets] = React.useState<WorkspaceDataset[]>([]);
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [jobPage, setJobPage] = React.useState(1);
  const [jobTotal, setJobTotal] = React.useState(0);
  const [artifacts, setArtifacts] = React.useState<Artifact[]>([]);
  const [resultTab, setResultTab] = React.useState<'jobs' | 'artifacts'>('jobs');
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [reload, setReload] = React.useState(0);

  const fetchDatasets = React.useCallback(async () => {
    if (!id) return;
    const data = await apiClient.get<Array<DatasetInfo | DatasetSource>>(`/projects/${id}/datasets`, { silent: true });
    setDatasets(data.map((item) => 'source' in item ? item as DatasetInfo : { source: item as DatasetSource }));
  }, [id]);
  const fetchConfig = React.useCallback(async () => {
    if (id) setConfig(await apiClient.get<Record<string, any>>(`/projects/${id}/config`, { silent: true }));
  }, [id]);
  const fetchResults = React.useCallback(async () => {
    if (!id) return;
    const [page, output] = await Promise.all([
      apiClient.get<JobListResponse>('/jobs', { params: { project_id: id, page: jobPage, page_size: 50 }, silent: true }),
      apiClient.get<Artifact[]>('/artifacts', { params: { project_id: id }, silent: true }),
    ]);
    setJobs(page.items); setJobTotal(page.total); setArtifacts(output);
  }, [id, jobPage]);
  React.useEffect(() => {
    if (!id) return;
    let active = true;
    setLoading(true); setError('');
    Promise.all([apiClient.get<Project>(`/projects/${id}`, { silent: true }), fetchDatasets(), fetchConfig()])
      .then(([project]) => { if (active) setProject(project); })
      .catch((error) => { if (active) setError(formatApiError(error)); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [id, fetchDatasets, fetchConfig, reload]);
  React.useEffect(() => { void fetchResults().catch((error) => setError(formatApiError(error))); }, [fetchResults]);
  const indexing = datasets.some((dataset) => dataset.index_status === 'indexing');
  React.useEffect(() => {
    if (!indexing) return;
    const timer = setInterval(() => { void fetchDatasets().catch((error) => setError(formatApiError(error))); }, 2000);
    return () => clearInterval(timer);
  }, [indexing, fetchDatasets]);
  const imported = () => { void Promise.all([fetchDatasets(), fetchConfig()]).catch((error) => setError(formatApiError(error))); };
  useEventStream(EVENT_TYPES.DATASET_CHANGED, () => { imported(); });
  useEventStream(EVENT_TYPES.JOB_STEP, (event: any) => setJobs((rows) => rows.map((job) => mergeJobEvent(job, event))));
  useEventStream(EVENT_TYPES.JOB_PHASE, (event: any) => setJobs((rows) => rows.map((job) => mergeJobEvent(job, event))));
  useEventStream(EVENT_TYPES.JOB_STATE, (event: any) => {
    setJobs((rows) => rows.map((job) => mergeJobEvent(job, event)));
    if (['completed', 'failed', 'cancelled'].includes(event.status)) void fetchResults().catch((error) => setError(formatApiError(error)));
  });

  if (loading) return <div className="flex items-center gap-2 py-12 text-slate-500"><Loader2 className="h-4 w-4 animate-spin" />{text('正在打开项目工作区…', 'Opening project workspace…')}</div>;
  if (!project || !id) return <div role="alert" className="space-y-3 rounded border border-red-200 p-5 text-red-700"><p>{error || text('项目不存在', 'Project not found')}</p><button onClick={() => setReload((value) => value + 1)} className="underline">{t('common.retry')}</button></div>;
  const imageCount = datasets.reduce((count, dataset) => count + (dataset.stats?.images || 0), 0);
  const captionCount = datasets.reduce((count, dataset) => count + (dataset.stats?.captioned || 0), 0);
  const activeJob = jobs.find((job) => ['running', 'pausing', 'cancelling'].includes(job.status));
  const modelFamily = config.model?.family || 'anima';

  return <div className="space-y-6" data-testid="project-detail-page">
    <div className="flex flex-wrap items-start justify-between gap-4"><div><Link to="/projects" className="mb-2 inline-flex items-center gap-1 text-xs text-slate-500 hover:text-blue-500"><ArrowLeft className="h-3 w-3" />{t('projects.title')}</Link><h1 className="text-2xl font-bold">{project.name}</h1><p className="mt-1 text-sm text-slate-500">{project.note || text('项目训练工作区 · 从图片到 LoRA', 'Training workspace · from images to LoRA')}</p></div>
      <NextStepLink to={`/projects/${id}/train`}>{text('配置并启动训练', 'Configure and start training')}</NextStepLink>
    </div>
    <ProjectWorkflow projectId={id} active={step} />
    {error && <div role="alert" className="whitespace-pre-line rounded bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">{error}<button className="ml-3 underline" onClick={() => setError('')}>{t('common.close')}</button></div>}
    {activeJob && <Link to={`/jobs/${activeJob.id}`} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm dark:border-blue-900 dark:bg-blue-950/30"><span className="flex items-center gap-2"><Activity className="h-4 w-4 text-blue-500" />{activeJob.name} · {t(`queue.status.${activeJob.status}`, activeJob.status)} · {activeJob.progress?.step ?? 0} / {activeJob.progress?.total_steps ?? '--'}</span><span className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-300">{text('打开训练监控', 'Open training monitor')}<ArrowRight className="h-4 w-4" /></span></Link>}
    {step === 'data' && <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3">
        {[{ label: text('训练图片', 'Training images'), value: imageCount }, { label: text('已有标签', 'Captioned images'), value: captionCount }, { label: text('数据源', 'Data sources'), value: datasets.length }].map((stat) => <div key={stat.label} className="rounded-lg border border-slate-200 bg-white px-4 py-3 dark:border-slate-700 dark:bg-slate-800"><p className="text-xs text-slate-500">{stat.label}</p><p className="mt-1 text-2xl font-semibold font-mono">{stat.value}</p></div>)}
      </div>
      <ProjectDataImport key={id} projectId={id} onImported={imported} />
      <div className="space-y-3"><div className="flex items-center justify-between"><h3 className="font-semibold">{text('项目中的图片与标签', 'Project images and captions')}</h3><button onClick={() => void fetchDatasets().catch((error) => setError(formatApiError(error)))} className="inline-flex items-center gap-1 text-xs text-slate-500"><RefreshCw className="h-3.5 w-3.5" />{text('刷新索引状态', 'Refresh index status')}</button></div>
        {datasets.length === 0 ? <div className="rounded-lg border border-dashed border-slate-300 p-5 text-sm text-slate-500 dark:border-slate-600" data-testid="datasets-empty">{text('还没有训练图片。用上方上传区添加图片，或导入训练机上的素材文件夹。', 'No training images yet. Upload images above or import a folder from the training machine.')}</div> : <div className="grid gap-3 lg:grid-cols-2">{datasets.map((dataset) => <Link key={dataset.source.id} to={`/datasets/${dataset.source.id}`} className="rounded-xl border border-slate-200 bg-white p-4 hover:border-blue-500 dark:border-slate-700 dark:bg-slate-800" data-testid={`dataset-card-${dataset.source.id}`}>
          <div className="flex items-center justify-between gap-2"><span className="inline-flex items-center gap-2 font-semibold text-sm"><Image className="h-4 w-4 text-blue-500" />{dataset.source.path.replace(/\\/g, '/').split('/').filter(Boolean).pop()?.replace(/^d_[0-9a-f]+-/, '')}</span><span className={`text-xs ${dataset.index_status === 'failed' ? 'text-red-500' : 'text-slate-500'}`}>{dataset.index_status ? t(`dataset.status${dataset.index_status[0].toUpperCase()}${dataset.index_status.slice(1)}`, dataset.index_status) : text('已登记', 'Registered')}</span></div>
          <p className="mt-2 break-all font-mono text-xs text-slate-400">{dataset.source.path}</p><div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-500"><span>{dataset.stats?.images ?? '--'} {text('张图片', 'images')}</span><span>{dataset.stats?.captioned ?? '--'} {text('份标签', 'captions')}</span><span>×{dataset.source.repeats} {text('重复', 'repeats')}</span>{dataset.source.is_reg && <span>{text('正则集', 'Regularization')}</span>}</div>
          {dataset.stats?.error && <p className="mt-2 text-xs text-red-500">{dataset.stats.error}</p>}<span className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-blue-600 dark:text-blue-300">{text('检查图片 / 编辑标签', 'Review images / edit captions')}<ArrowRight className="h-3.5 w-3.5" /></span>
        </Link>)}</div>}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-slate-100 p-4 dark:bg-slate-800"><p className="text-sm text-slate-500">{indexing ? text('图片正在索引，你可以先准备模型。', 'Images are indexing. You can prepare the model meanwhile.') : text('确认图片与标签后，选择训练用的底模。', 'After reviewing images and captions, choose your base model.')}</p><NextStepLink to={`/projects/${id}?step=models`}>{text('下一步：模型准备', 'Next: model setup')}</NextStepLink></div>
    </div>}
    {step === 'models' && <ProjectModelSetup key={id} projectId={id} config={config} onSaved={setConfig} />}
    {step === 'results' && <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3"><div className="flex gap-2">{(['jobs', 'artifacts'] as const).map((key) => <button key={key} onClick={() => setResultTab(key)} aria-pressed={resultTab === key} className={`rounded-lg px-3 py-2 text-sm font-medium ${resultTab === key ? 'bg-blue-600 text-white' : 'bg-slate-100 dark:bg-slate-800'}`}>{key === 'jobs' ? `${text('训练任务', 'Training jobs')} (${jobTotal})` : `${text('模型产物', 'Model outputs')} (${artifacts.length})`}</button>)}</div><button onClick={() => void fetchResults().catch((error) => setError(formatApiError(error)))} className="text-sm text-blue-500">{text('刷新', 'Refresh')}</button></div>
      {resultTab === 'jobs' && <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">
        {jobs.length === 0 ? <div className="space-y-3 p-8 text-center"><Database className="mx-auto h-7 w-7 text-slate-400" /><p className="text-slate-500">{text('还没有训练任务。准备数据和模型后，在参数页检查并启动。', 'No training jobs yet. Prepare your data and model, then validate and start from the configuration page.')}</p><NextStepLink to={`/projects/${id}/train`}>{text('配置并启动训练', 'Configure and start training')}</NextStepLink></div> : <table className="w-full text-left text-sm"><thead className="bg-slate-50 text-xs text-slate-500 dark:bg-slate-900"><tr><th className="p-3">{t('queue.name')}</th><th className="p-3">{t('common.status')}</th><th className="p-3">{t('queue.progress')}</th><th className="p-3">{t('queue.created')}</th><th className="p-3">{text('操作', 'Actions')}</th></tr></thead><tbody className="divide-y divide-slate-100 dark:divide-slate-700">{jobs.map((job) => <tr key={job.id}><td className="p-3"><Link to={`/jobs/${job.id}`} className="font-medium text-blue-600 dark:text-blue-300">{job.name}</Link></td><td className="p-3">{t(`queue.status.${job.status}`, job.status)}</td><td className="p-3 font-mono">{job.progress?.step ?? 0} / {job.progress?.total_steps ?? '--'}</td><td className="p-3 text-xs text-slate-500">{formatTime(job.created_at)}</td><td className="p-3"><Link to={`/jobs/${job.id}`} className="text-blue-500">{text('监控 / 预览 / 检查点', 'Monitor / samples / checkpoints')}</Link></td></tr>)}</tbody></table>}
        {jobTotal > 0 && <div className="flex items-center justify-end gap-3 border-t p-3 text-sm dark:border-slate-700"><span>{jobPage} / {Math.max(1, Math.ceil(jobTotal / 50))}</span><button disabled={jobPage <= 1} className="disabled:opacity-40" onClick={() => setJobPage((page) => page - 1)}>{t('common.previous')}</button><button disabled={jobPage * 50 >= jobTotal} className="disabled:opacity-40" onClick={() => setJobPage((page) => page + 1)}>{t('common.next')}</button></div>}
      </div>}
      {resultTab === 'artifacts' && <div className="rounded-xl border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">{artifacts.length === 0 ? <p className="p-8 text-center text-slate-500">{text('训练保存的 LoRA 权重会出现在这里。完整续训状态可在任务的检查点页使用。', 'Saved LoRA weights will appear here. Full training states can be resumed from the job’s checkpoints tab.')}</p> : <ul className="divide-y divide-slate-100 dark:divide-slate-700">{artifacts.map((artifact) => <li key={artifact.id} className="flex items-center justify-between gap-3 p-4"><div><p className="font-mono text-sm font-medium">{artifact.name}</p><p className="mt-1 text-xs text-slate-500">{artifact.family || modelFamily} · {artifact.algo || 'LoRA'} · {formatBytes(artifact.size)} · step {artifact.step ?? '--'}</p></div><a href={apiUrl(`/artifacts/${artifact.id}/download`)} className="inline-flex items-center gap-1 rounded-lg bg-blue-600 px-3 py-2 text-sm text-white"><Download className="h-4 w-4" />{t('common.download')}</a></li>)}</ul>}</div>}
    </div>}
  </div>;
}
