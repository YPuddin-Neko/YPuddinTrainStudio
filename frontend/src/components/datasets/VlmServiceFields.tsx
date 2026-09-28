import { useRef, useState, type FormEvent } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { CircleCheck, Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { VlmService } from '../../api/types';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../StudioSelect';
import { resolveService, useVlmModels, useVlmServices, type VlmServiceSettings } from './visionHooks';

const CUSTOM = '__custom';
const SERVICE_NAMES: Record<string, [string, string]> = {
  openai: ['OpenAI', 'OpenAI'],
  gemini: ['Google Gemini', 'Google Gemini'],
  openrouter: ['OpenRouter', 'OpenRouter'],
  siliconflow: ['SiliconFlow (硅基流动)', 'SiliconFlow'],
  dashscope: ['Model Studio (阿里云百炼)', 'Alibaba Cloud Model Studio'],
  deepseek: ['DeepSeek', 'DeepSeek'],
  ollama: ['Ollama', 'Ollama'],
  lmstudio: ['LM Studio', 'LM Studio'],
  custom: ['Custom (自定义)', 'Custom'],
};

function KeyField({ service, disabled }: { service: VlmService; disabled: boolean }) {
  const text = useWorkspaceText();
  const client = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const refresh = async () => { await client.invalidateQueries({ queryKey: ['vlm-services'] }); await client.invalidateQueries({ queryKey: ['vlm-models', service.id] }); };
  const act = async (request: () => Promise<unknown>) => {
    setBusy(true); setError('');
    try { await request(); setEditing(false); setValue(''); await refresh(); } catch (e) { setError(formatApiError(e)); } finally { setBusy(false); }
  };
  const save = (event: FormEvent) => { event.preventDefault(); if (value.trim()) void act(() => apiClient.put(`/vlm/services/${service.id}/key`, { api_key: value.trim() }, { silent: true })); };
  const local = service.editable && service.id !== 'custom';
  return <div className="vision-field"><span className="vision-field-label">{text('API 密钥', 'API key')}</span>
    {service.key_configured && !editing
      ? <span className="vision-key-saved"><span className="vision-model-state ready"><CircleCheck size={14}/>{text('已保存', 'Saved')}</span>
        <button type="button" className="ui-btn ui-btn-sm" disabled={disabled || busy} onClick={() => setEditing(true)}>{text('更换', 'Replace')}</button>
        <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet ui-btn-danger" disabled={disabled || busy} onClick={() => { if (window.confirm(text('删除保存的 API 密钥？', 'Delete the saved API key?'))) void act(() => apiClient.delete(`/vlm/services/${service.id}/key`, { silent: true })); }}>{text('删除', 'Delete')}</button></span>
      : <form className="vision-key-form" onSubmit={save}>
        <input type="password" autoComplete="off" spellCheck={false} aria-label={text('API 密钥', 'API key')} value={value} disabled={disabled || busy} placeholder={local ? text('本地服务通常不需要', 'Usually not needed locally') : text('粘贴 API 密钥', 'Paste the API key')} onChange={event => setValue(event.target.value)}/>
        <button type="submit" className="ui-btn" disabled={disabled || busy || !value.trim()}>{busy && <Loader2 size={13} className="animate-spin"/>}{text('保存', 'Save')}</button>
        {service.key_configured && <button type="button" className="ui-btn ui-btn-quiet" disabled={busy} onClick={() => { setEditing(false); setValue(''); }}>{text('取消', 'Cancel')}</button>}
      </form>}
    {error ? <span role="alert" className="vision-model-error">{error}</span> : <span className="vision-field-hint">{text('保存在训练服务器上，之后不再显示。', 'Kept on the training server and never shown again.')}</span>}
  </div>;
}

function ModelField({ provider, baseUrl, value, enabled, disabled, onChange }: { provider: string; baseUrl: string; value: string; enabled: boolean; disabled: boolean; onChange: (model: string) => void }) {
  const text = useWorkspaceText();
  const box = useRef<HTMLDivElement>(null);
  const models = useVlmModels(provider, baseUrl, enabled);
  const listed = models.data?.models || [];
  const [typing, setTyping] = useState(false);
  const custom = typing || !listed.length || (!!value && !listed.includes(value));
  const options = [...listed.map(model => ({ value: model, label: model })), { value: CUSTOM, label: text('手动输入…', 'Enter manually…') }];
  const choose = (next: string) => { if (next === CUSTOM) { setTyping(true); onChange(''); } else { setTyping(false); onChange(next); } };
  return <div className="vision-field"><span className="vision-field-label">{text('模型', 'Model')}</span>
    <span className="vision-model-line">
      {custom ? <div ref={box} className="vision-combo">
        <input type="text" aria-label={text('模型', 'Model')} value={value} maxLength={200} disabled={disabled} spellCheck={false} placeholder={text('手动输入，例如 gpt-4o-mini', 'Enter manually, e.g. gpt-4o-mini')} onChange={event => onChange(event.target.value.trim())}/>
        {!!listed.length && <StudioSelect className="vision-combo-toggle" anchorRef={box} searchable aria-label={text('从列表选择模型', 'Choose a model from the list')} disabled={disabled} value={CUSTOM} options={options} onValueChange={choose}/>}
      </div> : <StudioSelect searchable aria-label={text('模型', 'Model')} disabled={disabled} value={value} placeholder={text('选择模型', 'Choose a model')} options={options} onValueChange={choose}/>}
      <button type="button" className="ui-btn ui-btn-icon" aria-label={text('刷新模型列表', 'Refresh the model list')} title={text('刷新模型列表', 'Refresh the model list')} disabled={disabled || !enabled || models.isFetching} onClick={() => void models.refetch()}>{models.isFetching ? <Loader2 size={14} className="animate-spin"/> : <RefreshCw size={14}/>}</button>
    </span>
    {models.error ? <span role="alert" className="vision-model-error">{text('读取模型列表失败：', 'Could not list models: ')}{formatApiError(models.error)}</span>
      : <span className="vision-field-hint">{enabled ? text('需要支持图片输入的模型。', 'Pick a model that accepts images.') : text('保存密钥后列出可用模型。', 'Save a key to list the models.')}</span>}
  </div>;
}

export default function VlmServiceFields({ settings, onChange, disabled }: { settings: VlmServiceSettings; onChange: (patch: Partial<VlmServiceSettings>) => void; disabled: boolean }) {
  const text = useWorkspaceText();
  const services = useVlmServices();
  const resolved = resolveService(settings, services.data?.services);
  if (services.error) return <p role="alert" className="vision-runtime-notice vision-inline-notice"><span>{text('读取模型服务失败：', 'Could not load the model services: ')}{formatApiError(services.error)}</span><button type="button" className="ui-link" onClick={() => void services.refetch()}>{text('重新读取', 'Reload')}</button></p>;
  if (!resolved) return <p className="vision-field-hint"><Loader2 size={13} className="animate-spin"/> {text('读取模型服务…', 'Loading model services…')}</p>;
  const { service, baseUrl, model } = resolved;
  const name = (id: string) => { const pair = SERVICE_NAMES[id] || [id, id]; return text(pair[0], pair[1]); };
  return <div className="vision-row">
    <div className="vision-field"><span className="vision-field-label">{text('服务', 'Service')}</span>
      <StudioSelect aria-label={text('服务', 'Service')} value={service.id} disabled={disabled} onValueChange={provider => onChange({ provider })} options={(services.data?.services || []).map(item => ({ value: item.id, label: name(item.id) }))}/>
      <span className="vision-field-hint">{text('OpenAI 兼容接口；本地模型可用 Ollama 或 LM Studio。', 'Any OpenAI-compatible API; Ollama or LM Studio for local models.')}</span></div>
    <label className="vision-field"><span className="vision-field-label">{text('接口地址', 'Address')}</span>
      <input type="text" aria-label={text('接口地址', 'Address')} value={baseUrl} disabled={disabled || !service.editable} readOnly={!service.editable} spellCheck={false} maxLength={500} placeholder="http://127.0.0.1:8000/v1" onChange={event => onChange({ baseUrls: { ...settings.baseUrls, [service.id]: event.target.value.trim() } })}/>
      <span className="vision-field-hint">{service.editable ? text('由训练服务器发出请求，同一台机器上的服务填 127.0.0.1。', 'The training server sends the requests; use 127.0.0.1 for a service on the same machine.') : text('该服务使用固定地址。', 'This service has a fixed address.')}</span></label>
    <KeyField key={service.id} service={service} disabled={disabled}/>
    <ModelField key={`${service.id}/${baseUrl}`} provider={service.id} baseUrl={baseUrl} value={model} enabled={service.key_configured || service.editable} disabled={disabled} onChange={next => onChange({ models: { ...settings.models, [service.id]: next } })}/>
  </div>;
}
