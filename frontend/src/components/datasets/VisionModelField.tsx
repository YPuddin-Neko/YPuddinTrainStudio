import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useQueryClient, type UseQueryResult } from '@tanstack/react-query';
import { CircleAlert, CircleCheck, Download, Trash2, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { VisionCatalog, VisionModel } from '../../api/types';
import { formatBytes } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { LoadingNote } from '../Loading';
import StudioSelect from '../StudioSelect';
import { ACTIVE_DOWNLOAD } from './visionHooks';

const SOURCE_KEY = 'studio.vision.source';

/** Why a vision run cannot start yet: the catalog failed to load, or ONNX Runtime is missing. */
export function VisionRuntimeNotice({ catalog }: { catalog: UseQueryResult<VisionCatalog> }) {
  const text = useWorkspaceText();
  const location = useLocation();
  if (catalog.error) return <p role="alert" className="vision-runtime-notice"><CircleAlert size={15}/><span>{text('读取模型列表失败：', 'Could not load the model list: ')}{formatApiError(catalog.error)}</span>
    <button type="button" className="ui-link" onClick={() => void catalog.refetch()}>{text('重新读取', 'Reload')}</button></p>;
  if (!catalog.data || catalog.data.runtime.available) return null;
  return <p role="alert" className="vision-runtime-notice"><CircleAlert size={15}/><span>{text('需要先安装 ONNX Runtime。', 'Install ONNX Runtime first.')}</span>
    <Link className="ui-link" to="/settings/environment?package=onnxruntime" state={{ backgroundLocation: location.state?.backgroundLocation ?? location }}>{text('前往安装', 'Install')}</Link></p>;
}

export default function VisionModelField({ role, value, onChange, disabled, catalog, label, hint }: {
  role: VisionModel['role']; value: string; onChange?: (id: string) => void; disabled?: boolean;
  catalog: UseQueryResult<VisionCatalog>; label: string; hint: string;
}) {
  const text = useWorkspaceText();
  const client = useQueryClient();
  const [error, setError] = useState('');
  const [source, setSource] = useState(() => { try { return localStorage.getItem(SOURCE_KEY) || 'huggingface'; } catch { return 'huggingface'; } });
  const models = catalog.data?.models.filter(model => model.role === role) || [];
  const model = models.find(item => item.id === value) || models[0];
  const task = model?.download;
  const active = ACTIVE_DOWNLOAD.includes(task?.status ?? '');
  const refresh = () => client.invalidateQueries({ queryKey: ['vision-models'] });
  const act = async (request: () => Promise<unknown>) => {
    setError('');
    try { await request(); } catch (e) { setError(formatApiError(e)); }
    await refresh();
  };
  const chooseSource = (next: string) => { setSource(next); try { localStorage.setItem(SOURCE_KEY, next); } catch { /* the choice lasts for this page only */ } };
  const sources = model?.sources || [];
  const chosenSource = sources.includes(source as never) ? source : sources[0];
  const sourceLabel: Record<string, string> = { huggingface: 'Hugging Face', 'hf-mirror': text('HF 镜像', 'HF Mirror'), modelscope: text('魔搭社区', 'ModelScope') };
  const percent = task?.total_bytes ? Math.min(100, Math.floor((task.downloaded_bytes || 0) / task.total_bytes * 100)) : 0;
  const fieldId = `vision-model-${role}`;
  const failure = error || (task?.status === 'failed' ? task.error : '');
  return <div className="vision-field vision-model-field" data-testid={`vision-model-${role}`}>
    <label className="vision-field-label" htmlFor={fieldId}>{label}</label>
    <div className="vision-model-control">
      {catalog.isPending ? <LoadingNote label={text('读取模型列表…', 'Loading models…')}/>
        : models.length > 1 ? <StudioSelect id={fieldId} aria-label={label} value={model?.id || ''} disabled={disabled} onValueChange={id => onChange?.(id)}
          options={models.map(item => ({ value: item.id, label: `${item.label}${item.recommended ? text('（推荐）', ' (recommended)') : ''}` }))}/>
          : <span id={fieldId} className="vision-model-name">{model?.label || '—'}</span>}
      {model && (model.ready
        ? <span className="vision-model-state ready"><CircleCheck size={14}/>{text(`已下载 · ${formatBytes(model.size)}`, `Downloaded · ${formatBytes(model.size)}`)}
          <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon ui-btn-sm" aria-label={text(`删除 ${model.label}`, `Delete ${model.label}`)} title={text('删除模型文件', 'Delete model files')} disabled={disabled} onClick={() => { if (window.confirm(text(`删除 ${model.label} 的模型文件？`, `Delete the ${model.label} model files?`))) void act(() => apiClient.delete(`/vision/models/${model.id}`, { silent: true })); }}><Trash2 size={13}/></button></span>
        : active ? <span className="vision-model-progress" role="status">
          <span className="vision-model-bar" aria-hidden="true"><span style={{ width: `${task?.status === 'verifying' ? 100 : percent}%` }}/></span>
          <span className="vision-model-state">{task?.status === 'verifying' ? text('校验文件…', 'Verifying…') : task?.status === 'queued' ? text('等待下载…', 'Waiting…') : `${formatBytes(task?.downloaded_bytes || 0)} / ${formatBytes(model.size)} · ${percent}%`}</span>
          <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon ui-btn-sm" aria-label={text('取消下载', 'Cancel download')} title={text('取消下载', 'Cancel download')} onClick={() => void act(() => apiClient.post(`/vision/models/${model.id}/cancel`, {}, { silent: true }))}><X size={13}/></button></span>
        : <span className="vision-model-download">
          {sources.length > 1 && <StudioSelect aria-label={text('下载来源', 'Download source')} value={chosenSource} disabled={disabled} onValueChange={chooseSource} options={sources.map(item => ({ value: item, label: sourceLabel[item] || item }))}/>}
          <button type="button" className="ui-btn" disabled={disabled} onClick={() => void act(() => apiClient.post(`/vision/models/${model.id}/download`, { source: chosenSource }, { silent: true }))}><Download size={14}/>{text(`下载模型 · ${formatBytes(model.size)}`, `Download · ${formatBytes(model.size)}`)}</button></span>)}
    </div>
    {failure ? <p role="alert" className="vision-model-error">{failure}</p> : <span className="vision-field-hint">{hint}</span>}
  </div>;
}
