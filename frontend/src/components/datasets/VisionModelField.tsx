import { useRef, useState, type ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useQueryClient, type UseQueryResult } from '@tanstack/react-query';
import { Check, CircleAlert, Download, Trash2, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { VisionCatalog, VisionModel } from '../../api/types';
import { formatBytes } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { LoadingNote } from '../Loading';
import StudioSelect from '../StudioSelect';
import { ACTIVE_DOWNLOAD, TAGGER_SERIES, modelName, useTaggingSettings } from './visionHooks';
import { useConfirmation } from '../useConfirmation';
import ProgressBar from '../ProgressBar';
import '../model-download.css';

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

/** Downloaded state, progress or the download button of one model, from the source chosen in 设置 → 打标. */
export function VisionModelState({ model, disabled, catalogRow = false, children }: {
  model: VisionModel; disabled?: boolean; catalogRow?: boolean;
  children?: (action: ReactNode, details: ReactNode) => ReactNode;
}) {
  const text = useWorkspaceText();
  const { confirm, confirmation } = useConfirmation();
  const client = useQueryClient();
  const { tagging } = useTaggingSettings();
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const requestPending = useRef(false);
  const task = model.download;
  const active = ACTIVE_DOWNLOAD.includes(task?.status ?? '');
  const source = model.sources.includes(tagging?.model_source as never) ? tagging!.model_source : 'huggingface';
  const act = async (request: () => Promise<VisionModel>) => {
    if (requestPending.current) return;
    requestPending.current = true; setPending(true); setError('');
    try {
      try {
        const result = await request();
        await client.cancelQueries({ queryKey: ['vision-models'], exact: true });
        client.setQueryData<VisionCatalog>(['vision-models'], current => current && ({
          ...current, models: current.models.map(item => item.id === result.id ? result : item),
        }));
      } catch (e) { setError(formatApiError(e)); }
      void client.invalidateQueries({ queryKey: ['vision-models'] });
    } finally { requestPending.current = false; setPending(false); }
  };
  const remove = async () => {
    if (disabled || requestPending.current) return;
    if (!await confirm({ title: text('删除模型文件', 'Delete model files'), message: text(`删除 ${model.label} 的模型文件？`, `Delete the ${model.label} model files?`), confirmLabel: text('删除', 'Delete'), danger: true })) return;
    await act(() => apiClient.delete<VisionModel>(`/vision/models/${model.id}`, { silent: true }));
  };
  const total = task?.total_bytes || model.size;
  const downloaded = task?.downloaded_bytes || 0;
  const percent = total ? Math.min(100, Math.floor(downloaded / total * 100)) : 0;
  const status = task?.status === 'verifying' ? text('校验文件…', 'Verifying…') : task?.status === 'queued' ? text('排队中', 'Queued') : text('下载中', 'Downloading');
  const rawFailure = error || (task?.status === 'failed' ? task.error || text('未返回详细错误，请重试。', 'No detailed error was returned. Retry the download.') : '');
  const failure = /nodename nor servname provided, or not known/i.test(rawFailure)
    ? text('下载地址解析失败，请检查网络或代理设置。', 'Cannot resolve the download address. Check your network or proxy settings.') : rawFailure;
  const retry = !model.ready && !active && (!!error || task?.status === 'failed');
  const action = model.ready
    ? <span className="vision-model-state model-ready"><Check size={14}/>{catalogRow ? text('已下载', 'Downloaded') : text(`已下载 · ${formatBytes(model.size)}`, `Downloaded · ${formatBytes(model.size)}`)}
      <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon ui-btn-sm" aria-label={text(`删除 ${model.label}`, `Delete ${model.label}`)} title={text('删除模型文件', 'Delete model files')} disabled={disabled || pending} onClick={() => void remove()}><Trash2 size={13}/></button></span>
    : active ? <button type="button" className="ui-btn model-button-downloading" aria-label={`${text('取消下载', 'Cancel download')} ${model.label}`} title={`${text('取消下载', 'Cancel download')} ${model.label}`} disabled={disabled || pending} onClick={() => void act(() => apiClient.post<VisionModel>(`/vision/models/${model.id}/cancel`, {}, { silent: true }))}><span className="model-download-active-label">{status}</span><span className="model-download-cancel-label"><X size={13}/>{text('取消', 'Cancel')}</span></button>
    : <button type="button" className={`ui-btn${retry ? '' : ' ui-btn-primary'}`} disabled={disabled || pending || !!(model.token_required && !model.token_configured)} title={model.token_required && !model.token_configured ? text('请先在设置 → 访问密钥中保存 Hugging Face 令牌', 'Save a Hugging Face token under Settings → Access keys first') : source === 'modelscope' ? text('从魔搭社区下载', 'From ModelScope') : text('从 Hugging Face 下载', 'From Hugging Face')} onClick={() => void act(() => apiClient.post<VisionModel>(`/vision/models/${model.id}/download`, { source }, { silent: true }))}>{!retry && <Download size={14}/>} {retry ? text('重试', 'Retry') : catalogRow ? text('下载', 'Download') : text(`下载模型 · ${formatBytes(model.size)}`, `Download · ${formatBytes(model.size)}`)}</button>;
  const details = <>{active && <>
    <p className="model-transfer-status" role="status">{task?.status === 'downloading' ? `${formatBytes(downloaded)} / ${formatBytes(total)} · ${percent}%${task.bytes_per_second ? ` · ${formatBytes(task.bytes_per_second)}/s` : ''}` : status}</p>
    <ProgressBar className="model-download-progress" label={`${model.label} ${text('下载进度', 'download progress')}`} max={total || undefined} value={task?.status === 'queued' ? undefined : task?.status === 'verifying' ? total : total ? downloaded : undefined}/>
  </>}{failure && <p role="alert" className="model-download-error vision-model-error">{retry && text('下载失败：', 'Download failed: ')}{failure}</p>}</>;
  return <>{children ? children(action, details) : <>{action}{(active || failure) && <div className="vision-model-details">{details}</div>}</>}{confirmation}</>;
}

export default function VisionModelField({ role, value, onChange, disabled, catalog, label, hint }: {
  role: VisionModel['role']; value: string; onChange?: (id: string) => void; disabled?: boolean;
  catalog: UseQueryResult<VisionCatalog>; label: string; hint: string;
}) {
  const text = useWorkspaceText();
  const models = catalog.data?.models.filter(model => model.role === role) || [];
  const model = models.find(item => item.id === value) || models[0];
  const fieldId = `vision-model-${role}`;
  const series = [...new Set(models.map(item => item.family))];
  const inSeries = models.filter(item => item.family === model?.family);
  const pickSeries = (family: string) => {
    const choices = models.filter(item => item.family === family);
    const next = choices.find(item => item.recommended) || choices[0];
    if (next) onChange?.(next.id);
  };
  const gated = !!model?.token_required && !model.ready;
  return <div className="vision-field vision-model-field" data-testid={`vision-model-${role}`}>
    <label className="vision-field-label" htmlFor={fieldId}>{label}</label>
    <div className="vision-model-control">
      {catalog.isPending ? <LoadingNote label={text('读取模型列表…', 'Loading models…')}/>
        : role === 'tagger' && series.length > 1 ? <>
          <StudioSelect className="vision-model-series" aria-label={text('模型系列', 'Model series')} value={model?.family || ''} disabled={disabled} onValueChange={pickSeries}
            options={series.map(family => ({ value: family, label: TAGGER_SERIES[family] || family }))}/>
          <StudioSelect id={fieldId} aria-label={label} value={model?.id || ''} disabled={disabled} onValueChange={id => onChange?.(id)}
            options={inSeries.map(item => { const name = `${modelName(item)}${item.recommended ? text('（推荐）', ' (recommended)') : ''}`; return { value: item.id, label: `${name}${item.ready ? text(' · 已下载', ' · downloaded') : ''}`, displayLabel: name }; })}/>
        </> : models.length > 1 ? <StudioSelect id={fieldId} aria-label={label} value={model?.id || ''} disabled={disabled} onValueChange={id => onChange?.(id)}
          options={models.map(item => ({ value: item.id, label: `${item.label}${item.recommended ? text('（推荐）', ' (recommended)') : ''}` }))}/>
          : <span id={fieldId} className="vision-model-name">{model?.label || '—'}</span>}
      {model && <VisionModelState model={model} disabled={disabled}/>}
    </div>
    <span className="vision-field-hint">{gated
      ? <>{text('作者要求先在 Hugging Face 模型页同意使用条款，再用该账号的令牌下载。', 'The author asks you to accept the terms on the Hugging Face page, then download with that account’s token.')} <a className="ui-link" href={`https://huggingface.co/${model!.repo}`} target="_blank" rel="noreferrer">{text('打开模型页', 'Open the model page')}</a></>
      : hint}</span>
  </div>;
}
