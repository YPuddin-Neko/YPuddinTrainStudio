import React from 'react';
import { Link, Navigate, useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowRight, AudioLines, Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { JobListResponse, Project } from '../../api/types';
import { ttsApi } from '../../api/tts';
import ConfigHelp from '../../components/ConfigHelp';
import TtsSourcesPanel from './TtsSourcesPanel';
import TtsVersionResults from './TtsVersionResults';
import TtsLegacyImport from './TtsLegacyImport';
import TtsTrainingActions from './TtsTrainingActions';
import { resolveTrainingSchema } from './ttsVersionFields';
import { resolveGptSovitsSchema } from './gptSovitsVersionFields';
import TtsOverviewParameters from './TtsOverviewParameters';
import { ttsEngineLabel } from '../../utils/ttsEngines';
import ProjectWorkspaceHeader from '../../components/projects/ProjectWorkspaceHeader';
import { useProjectVersions } from '../../components/projects/useProjectVersions';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { projectUrl } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { formatTime } from '../../utils/format';
import { JobProgressSummary, JobStatus } from '../Queue/jobPresentation';
import TtsVersionEditor, { type TtsEditorHandle } from './TtsVersionEditor';
import '../../styles/project-workspace.css';
import '../ProjectDetail/project-overview.css';
import './tts-project.css';

export default function TtsProjectWorkspace({ project, versionId, training }: { project: Project; versionId?: string; training: boolean }) {
  const text = useWorkspaceText(), client = useQueryClient();
  const location = useLocation(), navigate = useNavigate();
  const [draftDirty, setDraftDirty] = React.useState(false);
  const [params] = useSearchParams();
  const step = training ? 'train' : params.get('step') === 'data' ? 'data' : params.get('step') === 'results' ? 'results' : 'overview';
  const versions = useProjectVersions(project, versionId);
  const current = versions.current;
  const ready = !!versionId && current?.id === versionId && current.status === 'ready';
  const readOnly = !!project.archived || !!current?.archived || !!current?.busy || !ready;
  const editor = React.useRef<TtsEditorHandle>(null);
  const configKey = ['tts-version-config', project.id, versionId];
  const config = useQuery({ queryKey: configKey, queryFn: ({ signal }) => ttsApi.versionConfig(project.id, versionId!, signal), enabled: ready, refetchOnWindowFocus: false });
  const capabilities = useQuery({ queryKey: ['tts-capabilities'], queryFn: ({ signal }) => ttsApi.capabilities(signal), enabled: ready && training });
  const engine = config.data?.config.engine || 'voxcpm1.5';
  const gptSovits = engine === 'gpt-sovits-v5';
  const schema = useQuery({ queryKey: ['tts-train-schema', engine], queryFn: ({ signal }) => ttsApi.trainSchema(signal, engine), enabled: ready && training && !!config.data });
  const supported = capabilities.data?.engines.some(item => item.id === engine && (item.id === 'gpt-sovits-v5' ? item.training_modes?.includes('gpt_finetune') && item.training_modes?.includes('sovits_lora') : item.training_modes?.includes('lora')));
  const schemaReady = !!schema.data && (gptSovits ? !!resolveGptSovitsSchema(schema.data) : !!resolveTrainingSchema(schema.data));
  const jobs = useQuery({ queryKey: ['tts-version-jobs', project.id, versionId], queryFn: ({ signal }) => apiClient.get<JobListResponse>('/jobs', { params: { project_id: project.id, version_id: versionId, type: 'tts_train', page: 1, page_size: 20 }, signal, silent: true }), enabled: ready && (step === 'overview' || step === 'results') });
  useEventStream(EVENT_TYPES.JOB_STATE, () => { if (ready) { void jobs.refetch(); void versions.refresh(); } });
  if (!versionId && project.active_version_id) return <Navigate replace to={{ pathname: projectUrl(project.id, project.active_version_id, training ? 'train' : undefined), search: location.search, hash: location.hash }} state={location.state}/>;
  const model = config.data?.config.model_path;
  const unavailable = !versions.loading && !current;
  const summaries = current?.audio_stats;
  const sourceSummary = (split: 'train' | 'validation') => {
    const value = summaries?.[split];
    if (!value || value.state === 'missing') return text('未登记', 'Not registered');
    if (value.clips_count == null) return text('尚无检查结果', 'No check results');
    return text(`${value.clips_count.toLocaleString()} 条音频`, `${value.clips_count.toLocaleString()} clips`);
  };
  const runs = <section className="overview-panel"><header className="overview-panel-heading"><h3>{text('训练记录', 'Training runs')}</h3><button className="ui-btn ui-btn-sm ui-btn-quiet" onClick={() => void jobs.refetch()} disabled={jobs.isFetching}>{text('刷新', 'Refresh')}</button></header>
    {jobs.isPending ? <p className="overview-muted" role="status">{text('正在读取训练记录…', 'Loading training runs…')}</p> : jobs.error ? <p role="alert" className="workspace-message error">{formatApiError(jobs.error)}</p> : jobs.data?.items.length ? <ul className="overview-run-list">{jobs.data.items.map(job => <li key={job.id}><Link to={`/jobs/${encodeURIComponent(job.id)}`}><span className="overview-run-top"><JobStatus status={job.status}/><time>{formatTime(job.created_at)}</time></span><strong>{job.name}</strong><JobProgressSummary job={job}/></Link></li>)}</ul> : <p className="overview-muted">{text('本版本还没有训练记录。', 'No training runs in this version.')}</p>}
    {(jobs.data?.total || 0) > 20 && <Link className="ui-link" to={`/queue?project_id=${encodeURIComponent(project.id)}&type=tts_train`}>{text('查看全部训练记录', 'View all training runs')}<ArrowRight size={13}/></Link>}
  </section>;
  return <div className="project-workspace tts-project-workspace" data-testid="tts-project-workspace" data-step={step}>
    <ProjectWorkspaceHeader project={project} versionId={versionId} versions={versions.versions} current={current} active={step} refresh={versions.refresh} error={versions.error} beforeAction={() => editor.current?.beforeAction() || Promise.resolve()} title={step === 'overview' ? project.name : undefined}/>
    {versions.loading && <p className="workspace-loading" role="status"><Loader2 size={16} className="animate-spin"/>{text('正在读取版本…', 'Loading version…')}</p>}
    {unavailable && <div className="workspace-message error" role="alert">{text('该版本不存在或不属于当前项目。', 'This version does not belong to this project.')}<Link className="ui-link" to={projectUrl(project.id)}>{text('返回当前版本', 'Return to current version')}</Link></div>}
    {ready && config.error && <div role="alert" className="workspace-message error">{formatApiError(config.error)}<button className="ui-btn ui-btn-sm" onClick={() => void config.refetch()}>{text('重新读取参数', 'Reload parameters')}</button></div>}
    {ready && config.isPending && <p role="status" className="workspace-loading"><Loader2 size={16} className="animate-spin"/>{text('正在读取版本参数…', 'Loading version parameters…')}</p>}
    {ready && config.data && (training ? <>
      {(capabilities.isPending || schema.isPending) && <p role="status" className="workspace-loading"><Loader2 size={16} className="animate-spin"/>{text('正在读取训练能力…', 'Loading training capabilities…')}</p>}
      {(capabilities.error || schema.error || capabilities.data && !supported || schema.data && !schemaReady) && <div role="alert" className="workspace-message error">{capabilities.error || schema.error ? formatApiError(capabilities.error || schema.error) : text('当前服务的训练参数不兼容，请更新页面后重试。', 'Training parameters are incompatible with this page. Update the page and try again.')}<button className="ui-btn ui-btn-sm" onClick={() => { void capabilities.refetch(); void schema.refetch(); }}>{text('重新读取', 'Reload')}</button></div>}
      {supported && schemaReady && schema.data && <>
        <div className="tts-capability-summary"><span>{gptSovits ? text('单卡 NVIDIA CUDA · FP16 / FP32', 'Single NVIDIA CUDA GPU · FP16 / FP32') : text('单卡 NVIDIA CUDA · BF16', 'Single NVIDIA CUDA GPU · BF16')}</span><ConfigHelp label={text('训练能力说明', 'Training capabilities')}>{gptSovits ? text('GPT 全量微调与 SoVITS LoRA 可分别训练，或依次训练 SoVITS、GPT。使用所选变体的配套基础权重，不支持多卡、暂停或恢复。', 'Train GPT with full finetuning and SoVITS with LoRA, separately or in the order SoVITS then GPT. Use the base weights for the selected variant. Multiple GPUs, pause and resume are not supported.') : text('支持 VoxCPM 1.5 LoRA，使用 AdamW 和带预热的余弦学习率调度。训练任务不支持全量微调、多卡、暂停或恢复。输出目录由项目、版本和任务自动分配。', 'VoxCPM 1.5 LoRA uses AdamW with a warmup cosine schedule. Full finetuning, multiple GPUs, pause and resume are not supported. Output paths are assigned by project, version and job.')}</ConfigHelp></div>
        <TtsVersionEditor key={`${project.id}:${versionId}:${engine}`} ref={editor} initial={config.data} schema={schema.data} readOnly={readOnly} onDirtyChange={setDraftDirty} onSaved={saved => { client.setQueryData(configKey, saved); void client.invalidateQueries({ queryKey: ['project-versions', project.id] }); void client.invalidateQueries({ queryKey: ['project', project.id] }); if (saved.data_revision !== config.data?.data_revision) void client.invalidateQueries({ queryKey: ['tts-sources', project.id, versionId] }); }} toolbarActions={<>{!gptSovits && <TtsLegacyImport key={`${project.id}:${versionId}`} projectId={project.id} versionId={versionId!} readOnly={readOnly} ensureSaved={() => editor.current?.saveConfig() || Promise.resolve(null)} onImported={saved => { client.setQueryData(configKey, saved); }}/>}<TtsTrainingActions engine={engine} projectId={project.id} versionId={versionId!} readOnly={readOnly} draftDirty={draftDirty} ensureSaved={() => editor.current?.saveConfig() || Promise.resolve(null)} onStarted={job => { void client.invalidateQueries({ queryKey: ['project-versions', project.id] }); void client.invalidateQueries({ queryKey: ['tts-version-jobs', project.id, versionId] }); navigate(`/jobs/${encodeURIComponent(job.id)}`); }}/></>}/>
      </>}
    </> : step === 'results' ? <><TtsVersionResults engine={engine} key={`${project.id}:${versionId}`} projectId={project.id} versionId={versionId!}/>{runs}</> : step === 'data' ? <TtsSourcesPanel engine={engine} projectId={project.id} versionId={versionId!} readOnly={readOnly}/>
        : <section className="project-overview"><div className="overview-layout"><section className="overview-main"><header className="overview-section-heading"><h2>{text('训练数据', 'Training data')}</h2><Link className="ui-link" to={projectUrl(project.id, versionId, 'data')}>{text('查看音频与转写', 'View audio and transcripts')}<ArrowRight size={13}/></Link></header>{summaries?.train && summaries.train.state !== 'missing' ? <dl className="overview-facts"><div><dt>{text('训练音频', 'Training audio')}</dt><dd>{sourceSummary('train')}</dd></div>{!gptSovits && <div><dt>{text('验证音频', 'Validation audio')}</dt><dd>{sourceSummary('validation')}</dd></div>}</dl> : <div className="overview-data-empty"><span className="overview-data-empty-icon" aria-hidden="true"><AudioLines size={22}/></span><strong>{text('训练音频尚未登记', 'Training audio is not registered')}</strong><p>{gptSovits ? text('使用 32 / 44.1 / 48 kHz 单声道 PCM WAV，配合含文本与语言的 JSONL 或四列 .list 清单。', 'Use 32 / 44.1 / 48 kHz mono PCM WAV with transcripts and languages in a JSONL or four-column .list manifest.') : text('VoxCPM 1.5 使用 44100 Hz 单声道 PCM WAV 和 JSONL 转写清单。', 'VoxCPM 1.5 uses 44100 Hz mono PCM WAV audio and a JSONL transcript manifest.')}</p><Link className="ui-btn" to={projectUrl(project.id, versionId, 'data')}>{readOnly ? text('查看数据', 'View data') : text('登记音频清单', 'Register audio manifest')}</Link></div>}</section><aside className="overview-aside" aria-label={text('版本信息', 'Version details')}><section className="overview-panel"><header className="overview-panel-heading"><h3>{text('模型与训练参数', 'Model and training parameters')}</h3><Link className="ui-link" to={projectUrl(project.id, versionId, 'train')}>{readOnly ? text('查看参数', 'View parameters') : text('设置参数', 'Edit parameters')}<ArrowRight size={13}/></Link></header><div className="overview-model"><span>{ttsEngineLabel(engine)} · {gptSovits ? text('GPT 微调 / SoVITS LoRA', 'GPT finetuning / SoVITS LoRA') : 'LoRA'}</span><strong title={model || undefined}>{model || text('尚未配置训练底模', 'Base model not configured')}</strong></div><TtsOverviewParameters config={config.data.config}/></section>{runs}{project.note && <section className="overview-panel"><header className="overview-panel-heading"><h3>{text('项目备注', 'Project note')}</h3></header><p className="overview-muted">{project.note}</p></section>}</aside></div></section>)}
  </div>;
}
