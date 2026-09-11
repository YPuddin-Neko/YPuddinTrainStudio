import { useId, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Tags, Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import { PathInput } from '../PathBrowser';
import NumericControl from '../../schema/SchemaForm/NumericControl';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import type { TaggingMount } from './DatasetPipelinePanel';
import './tagging-panel.css';

type LocalModel = { name: string; path: string; model_path: string; tags_path: string };
type TaggerStatus = {
  available: boolean; runtime_available: boolean; runtime_version: string | null;
  providers: string[]; runtime_providers: string[]; provider: string;
  model_exists: boolean; tags_exists: boolean; model_path: string | null; tags_path: string | null;
  input_size: number | null; models: LocalModel[]; recommended_model_dir?: string;
  errors: string[]; notes: string[];
};
type Selection = { model_path: string; tags_path: string; provider: 'cpu' | 'cuda' };
type Props = TaggingMount & { projectId: string; versionId: string };

export default function TaggingPanel({ projectId, versionId, images, busy, onSubmit }: Props) {
  const text = useWorkspaceText();
  const location = useLocation();
  const id = useId();
  const [selection, setSelection] = useState<Selection>({ model_path: '', tags_path: '', provider: 'cpu' });
  const [customPaths, setCustomPaths] = useState(false);
  const [checked, setChecked] = useState(selection);
  const [mode, setMode] = useState('missing');
  const [general, setGeneral] = useState<number | undefined>(0.35);
  const [character, setCharacter] = useState<number | undefined>(0.85);
  const [trigger, setTrigger] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const status = useQuery({
    queryKey: ['dataset-tagger-status', projectId, versionId, checked],
    queryFn: ({signal}) => apiClient.get<TaggerStatus>('/dataset-tagging/status', {params: checked, signal, silent: true}),
    retry: false,
  });
  const data = status.data;
  const models = data?.models || [];
  const current = selection.model_path.trim() === checked.model_path && selection.tags_path.trim() === checked.tags_path && selection.provider === checked.provider;
  const modelPath = customPaths ? selection.model_path.trim() : selection.model_path.trim() || data?.model_path || '';
  const tagsPath = customPaths ? selection.tags_path.trim() : selection.tags_path.trim() || data?.tags_path || '';
  const selectedModel = !customPaths && models.find(model => model.model_path === modelPath && model.tags_path === tagsPath);
  const ready = current && !status.isFetching && data?.available && !!modelPath && !!tagsPath;
  const thresholdsValid = [general, character].every(value => value !== undefined && Number.isFinite(value) && value >= 0 && value <= 1);
  const locked = busy || submitting;
  const settingsState = { backgroundLocation: location.state?.backgroundLocation || location };
  const check = () => {
    const next = {...selection, model_path: selection.model_path.trim(), tags_path: selection.tags_path.trim()};
    setSelection(next);
    if (JSON.stringify(next) === JSON.stringify(checked)) void status.refetch(); else setChecked(next);
  };
  const chooseModel = (path: string) => {
    const model = models.find(item => item.path === path);
    if (!model) { setSelection({...selection, model_path:modelPath, tags_path:tagsPath}); setCustomPaths(true); return; }
    const next = {...selection, model_path:model.model_path, tags_path:model.tags_path};
    setSelection(next); setChecked(next); setCustomPaths(false);
  };
  const updatePath = (field: 'model_path' | 'tags_path', value: string) => {
    setSelection({...selection, model_path:modelPath, tags_path:tagsPath, [field]:value}); setCustomPaths(true);
  };
  const submit = async () => {
    if (locked || !ready || !images.length || !thresholdsValid) return;
    setError(''); setSubmitting(true);
    try {
      await onSubmit({ model_path:data?.model_path || modelPath, tags_path:data?.tags_path || tagsPath, provider:selection.provider,
        general_threshold:general, character_threshold:character, mode, trigger_word:trigger.trim() });
    } catch (reason) { setError(formatApiError(reason)); }
    finally { setSubmitting(false); }
  };

  return <form className="tagging-panel" aria-label={text('本地 WD14 自动标注', 'Local WD14 automatic tagging')} onSubmit={event => {event.preventDefault(); void submit();}}>
    <div className="tagging-heading"><strong><Tags size={15}/>{text('本地 WD14 自动标注', 'Local WD14 automatic tagging')}</strong><button type="button" disabled={locked || status.isFetching} onClick={check}><RefreshCw size={13} className={status.isFetching ? 'animate-spin' : ''}/>{text('检查模型与依赖', 'Check model and runtime')}</button></div>
    <p className="tagging-note">{text('在训练机上生成标签，图片不发送到云端。先在下方选择图片；取消、重试和恢复原标签均在操作记录中。', 'Generates tags on the training machine without uploading images. Select images below; cancellation, retry, and caption recovery use the operation history.')}</p>
    {status.isPending ? <p role="status" className="tagging-note"><Loader2 size={13} className="animate-spin"/>{text('正在读取打标环境…', 'Checking tagging environment…')}</p> : status.isError ? <p role="alert" className="tagging-error">{formatApiError(status.error)}</p> : <>
      {!current && <p role="status" className="tagging-note">{text('模型路径或设备已更改，请重新检查后开始。', 'The model path or device changed. Check it again before starting.')}</p>}
      {current && data?.available && <p className="tagging-note">{text('本地文件与运行库就绪', 'Local files and runtime ready')} · ONNX Runtime {data.runtime_version || '—'} · {data.input_size ? `${data.input_size} × ${data.input_size}` : '—'} · {selection.provider.toUpperCase()}</p>}
      {current && !!data?.errors.length && <div role="alert" className="tagging-error"><strong>{text('暂时无法开始自动标注', 'Automatic tagging is not ready')}</strong><ul>{data.errors.map((message,index) => <li key={index}>{message}</li>)}</ul></div>}
      {data && (!data.runtime_available || !data.model_exists || !data.tags_exists) && <div className="tagging-setup-links">
        {!data.runtime_available && <Link to="/settings/environment?tab=runtime&package=onnxruntime" state={settingsState}>{text('安装 ONNX Runtime', 'Install ONNX Runtime')}</Link>}
        {(!data.model_exists || !data.tags_exists) && <Link to="/settings/environment?tab=models" state={settingsState}>{text('下载 WD14 模型与标签表', 'Download WD14 model and tag table')}</Link>}
      </div>}
    </>}
    <fieldset disabled={locked} className="tagging-controls">
      <label className="tagging-model">{text('本地打标模型', 'Local tagging model')}<select aria-label={text('本地打标模型', 'Local tagging model')} value={selectedModel ? selectedModel.path : ''} onChange={event => chooseModel(event.target.value)}><option value="">{modelPath ? text('自定义本地文件', 'Custom local files') : text('尚无可用模型，请下载或填写路径', 'No model available; download or enter paths')}</option>{models.map(model => <option key={model.path} value={model.path}>{model.name}</option>)}</select></label>
      <label>{text('推理设备', 'Inference device')}<select value={selection.provider} onChange={event => setSelection(previous => ({...previous,provider:event.target.value as Selection['provider']}))}><option value="cpu">CPU</option><option value="cuda" disabled={!data?.providers.includes('cuda')}>{text('CUDA（需要 GPU 版运行库）', 'CUDA (requires GPU runtime)')}</option></select></label>
      <label>{text('写入方式', 'Caption mode')}<select value={mode} onChange={event => setMode(event.target.value)}><option value="missing">{text('仅填写缺失标签', 'Fill missing captions only')}</option><option value="append">{text('追加并去重', 'Append unique tags')}</option><option value="overwrite">{text('替换原标签', 'Replace captions')}</option></select></label>
      <div className="tagging-threshold"><label htmlFor={`${id}-general`}>{text('常规标签阈值', 'General tag threshold')}</label><NumericControl id={`${id}-general`} label={text('常规标签阈值', 'General tag threshold')} sliderLabel={text('滑条','slider')} value={general} onChange={setGeneral} min={0} max={1} step={0.01} percentage/></div>
      <div className="tagging-threshold"><label htmlFor={`${id}-character`}>{text('角色标签阈值', 'Character tag threshold')}</label><NumericControl id={`${id}-character`} label={text('角色标签阈值', 'Character tag threshold')} sliderLabel={text('滑条','slider')} value={character} onChange={setCharacter} min={0} max={1} step={0.01} percentage/></div>
      <label className="tagging-trigger">{text('触发词（可选）', 'Trigger words (optional)')}<input value={trigger} maxLength={1000} onChange={event => setTrigger(event.target.value)} placeholder={text('例如 sora_character', 'e.g. sora_character')}/></label>
      <details className="tagging-paths" open={!selectedModel}>
        <summary>{text('自定义模型文件路径', 'Custom model file paths')}</summary>
        <div><label>{text('ONNX 模型文件', 'ONNX model file')}</label><PathInput value={modelPath} onChange={value => updatePath('model_path',value)} ariaLabel={text('ONNX 模型文件', 'ONNX model file')} placeholder="model.onnx"/></div>
        <div><label>{text('标签表 CSV', 'Tag table CSV')}</label><PathInput value={tagsPath} onChange={value => updatePath('tags_path',value)} ariaLabel={text('标签表 CSV', 'Tag table CSV')} placeholder="selected_tags.csv"/></div>
      </details>
      <div className="tagging-submit"><p className="tagging-note">{mode === 'overwrite' ? text('将替换所选图片的标签；原标签会自动备份，可在操作记录中恢复。', 'Replaces selected captions. Originals are backed up and can be restored from operation history.') : text('标签写入前自动备份；恢复操作会还原原文件，包括原先没有标签的状态。', 'Captions are backed up before writing. Recovery restores original files, including previously missing captions.')}</p><button type="submit" className="pipeline-primary" disabled={!ready || !images.length || !thresholdsValid}>{submitting && <Loader2 size={13} className="animate-spin"/>}{text(`为 ${images.length} 张图片生成标签`, `Generate tags for ${images.length} images`)}</button></div>
    </fieldset>
    {!images.length && <p className="tagging-note">{text('在下方图片列表中勾选要标注的素材。', 'Select images to tag in the list below.')}</p>}
    {error && <p role="alert" className="tagging-error">{error}</p>}
    {!!data?.notes.length && <details className="tagging-notes"><summary>{text('运行说明', 'Runtime notes')}</summary><ul>{data.notes.map((note,index) => <li key={index}>{note}</li>)}</ul></details>}
  </form>;
}
