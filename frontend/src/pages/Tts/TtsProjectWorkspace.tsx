import React from 'react';
import { Link, Navigate, useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2 } from 'lucide-react';
import type { Project } from '../../api/types';
import { ttsApi, type TtsScopedConfig } from '../../api/tts';
import { ttsModelSelectable } from '../../api/ttsModels';
import TtsModelPicker from '../Models/TtsModelPicker';
import TtsPresetActions from './TtsPresetActions';
import { TtsParameterHeadingContext } from './TtsParameterHeadingContext';
import ConfigHelp from '../../components/ConfigHelp';
import TtsSourcesPanel from './TtsSourcesPanel';
import TtsDataSummary from './TtsDataSummary';
import TtsVersionResults from './TtsVersionResults';
import TtsLegacyImport from './TtsLegacyImport';
import TtsTrainingActions from './TtsTrainingActions';
import { resolveTrainingSchema } from './ttsVersionFields';
import { resolveGptSovitsSchema } from './gptSovitsVersionFields';
import TtsProjectOverview from './TtsProjectOverview';
import { ttsEngineLabel } from '../../utils/ttsEngines';
import ProjectWorkspaceHeader from '../../components/projects/ProjectWorkspaceHeader';
import { useProjectVersions } from '../../components/projects/useProjectVersions';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { projectUrl } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import TtsVersionEditor, { type TtsEditorHandle } from './TtsVersionEditor';
import '../../styles/project-workspace.css';
import '../ProjectDetail/project-overview.css';
import './tts-project.css';

function ParameterActions({ config, readOnly, editorRef }: { config: TtsScopedConfig; readOnly: boolean; editorRef: React.RefObject<TtsEditorHandle> }) {
  const text = useWorkspaceText();
  const [error, setError] = React.useState('');
  return <><TtsModelPicker engine={config.engine} variant={config.engine === 'gpt-sovits-v5' ? config.variant : undefined} disabled={readOnly} getScope={() => {
    const draft = editorRef.current?.getConfig(); setError('');
    return draft ? { engine: draft.engine, variant: draft.engine === 'gpt-sovits-v5' ? draft.variant : undefined, value: draft.model_path } : null;
  }} onSelect={installation => {
    const editor = editorRef.current, draft = editor?.getConfig();
    if (!readOnly && draft && ttsModelSelectable(installation, draft.engine, draft.engine === 'gpt-sovits-v5' ? draft.variant : undefined) && editor?.loadConfig({ ...draft, model_path: installation.bindings.model_path })) setError('');
    else setError(text('参数已变化或暂时不可编辑，请处理后重新选择模型。', 'Parameters changed or cannot be edited. Resolve this and choose the model again.'));
  }}/><TtsPresetActions engine={config.engine} readOnly={readOnly} editorRef={editorRef}/>{error && <span role="alert" className="tts-preset-feedback">{error}</span>}</>;
}

export default function TtsProjectWorkspace({ project, versionId, training }: { project: Project; versionId?: string; training: boolean }) {
  const text = useWorkspaceText(), client = useQueryClient();
  const location = useLocation(), navigate = useNavigate();
  const [draftDirty, setDraftDirty] = React.useState(false);
  const [headingTarget, setHeadingTarget] = React.useState<HTMLDivElement | null>(null);
  const [params, setParams] = useSearchParams();
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
  useEventStream(EVENT_TYPES.JOB_STATE, () => { if (ready) void versions.refresh(); });
  if (!versionId && project.active_version_id) return <Navigate replace to={{ pathname: projectUrl(project.id, project.active_version_id, training ? 'train' : undefined), search: location.search, hash: location.hash }} state={location.state}/>;
  const unavailable = !versions.loading && !current;
  return <div className="project-workspace tts-project-workspace" data-testid="tts-project-workspace" data-step={step}>
    <ProjectWorkspaceHeader project={project} versionId={versionId} versions={versions.versions} current={current} active={step} refresh={versions.refresh} error={versions.error} beforeAction={() => editor.current?.beforeAction() || Promise.resolve()} title={step === 'overview' ? project.name : undefined} titleBadge={training && config.data ? <><span className="workspace-title-badge">{ttsEngineLabel(engine)}{config.data.config.engine === 'gpt-sovits-v5' ? ` · ${config.data.config.variant}` : ' · LoRA'}</span><ConfigHelp label={text('训练能力说明', 'Training capabilities')}>{gptSovits ? text('单卡 NVIDIA CUDA，支持 FP16 / FP32。GPT 全量微调与 SoVITS LoRA 可分别训练，或依次训练 SoVITS、GPT。使用所选变体的配套基础权重，不支持多卡、暂停或恢复。', 'Single NVIDIA CUDA GPU with FP16 / FP32. Train GPT with full finetuning and SoVITS with LoRA, separately or in the order SoVITS then GPT. Use base weights for the selected variant. Multiple GPUs, pause and resume are not supported.') : text('单卡 NVIDIA CUDA / BF16。VoxCPM 1.5 LoRA 使用 AdamW 和带预热的余弦学习率调度。不支持全量微调、多卡、暂停或恢复。输出目录由项目、版本和任务自动分配。', 'Single NVIDIA CUDA GPU with BF16. VoxCPM 1.5 LoRA uses AdamW with a warmup cosine schedule. Full finetuning, multiple GPUs, pause and resume are not supported. Output paths are assigned by project, version and job.')}</ConfigHelp></> : undefined} status={step === 'data' ? <TtsDataSummary stats={current?.audio_stats}/> : training ? <div ref={setHeadingTarget}/> : undefined}/>
    {versions.loading && <p className="workspace-loading" role="status"><Loader2 size={16} className="animate-spin"/>{text('正在读取版本…', 'Loading version…')}</p>}
    {unavailable && <div className="workspace-message error" role="alert">{text('该版本不存在或不属于当前项目。', 'This version does not belong to this project.')}<Link className="ui-link" to={projectUrl(project.id)}>{text('返回当前版本', 'Return to current version')}</Link></div>}
    {ready && config.error && <div role="alert" className="workspace-message error">{formatApiError(config.error)}<button className="ui-btn ui-btn-sm" onClick={() => void config.refetch()}>{text('重新读取参数', 'Reload parameters')}</button></div>}
    {ready && config.isPending && <p role="status" className="workspace-loading"><Loader2 size={16} className="animate-spin"/>{text('正在读取版本参数…', 'Loading version parameters…')}</p>}
    {ready && config.data && (training ? <>
      {(capabilities.isPending || schema.isPending) && <p role="status" className="workspace-loading"><Loader2 size={16} className="animate-spin"/>{text('正在读取训练能力…', 'Loading training capabilities…')}</p>}
      {(capabilities.error || schema.error || capabilities.data && !supported || schema.data && !schemaReady) && <div role="alert" className="workspace-message error">{capabilities.error || schema.error ? formatApiError(capabilities.error || schema.error) : text('当前服务的训练参数不兼容，请更新页面后重试。', 'Training parameters are incompatible with this page. Update the page and try again.')}<button className="ui-btn ui-btn-sm" onClick={() => { void capabilities.refetch(); void schema.refetch(); }}>{text('重新读取', 'Reload')}</button></div>}
      {supported && schemaReady && schema.data && <>
        <TtsParameterHeadingContext.Provider value={headingTarget}><TtsVersionEditor key={`${project.id}:${versionId}:${engine}`} ref={editor} initial={config.data} schema={schema.data} readOnly={readOnly} onDirtyChange={setDraftDirty} parameterActions={<ParameterActions key={`${project.id}:${versionId}:${engine}`} config={config.data.config} readOnly={readOnly} editorRef={editor}/>} onSaved={saved => { client.setQueryData(configKey, saved); void client.invalidateQueries({ queryKey: ['project-versions', project.id] }); void client.invalidateQueries({ queryKey: ['project', project.id] }); if (saved.data_revision !== config.data?.data_revision) void client.invalidateQueries({ queryKey: ['tts-sources', project.id, versionId] }); }} toolbarActions={<>{!gptSovits && <TtsLegacyImport key={`${project.id}:${versionId}`} projectId={project.id} versionId={versionId!} readOnly={readOnly} ensureSaved={() => editor.current?.saveConfig() || Promise.resolve(null)} onImported={saved => { client.setQueryData(configKey, saved); }}/>}<TtsTrainingActions engine={engine} projectId={project.id} versionId={versionId!} readOnly={readOnly} draftDirty={draftDirty} ensureSaved={() => editor.current?.saveConfig() || Promise.resolve(null)} onStarted={job => { void client.invalidateQueries({ queryKey: ['project-versions', project.id] }); void client.invalidateQueries({ queryKey: ['tts-version-jobs', project.id, versionId] }); navigate(`/jobs/${encodeURIComponent(job.id)}`); }}/></>}/></TtsParameterHeadingContext.Provider>
      </>}
    </> : step === 'results' ? <><TtsVersionResults engine={engine} key={`${project.id}:${versionId}`} projectId={project.id} versionId={versionId!} readOnly={readOnly}/></> : step === 'data' ? <TtsSourcesPanel engine={engine} projectId={project.id} versionId={versionId!} readOnly={readOnly}
      view={{ stage: params.get('tts_data') === 'rows' ? 'rows' : params.get('tts_data') === 'issues' ? 'issues' : 'sources', split: params.get('split') === 'validation' ? 'validation' : 'train' }}
      onViewChange={view => setParams(previous => { const next = new URLSearchParams(previous); next.set('tts_data', view.stage); next.set('split', view.split); return next; })}/>
        : <TtsProjectOverview project={project} version={current} config={config.data.config} readOnly={readOnly}/>)}
  </div>;
}
