import React from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { CircleCheck, Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { VlmService } from '../../api/types';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../../components/StudioSelect';
import { LoadingNote } from '../../components/Loading';
import { VisionModelState } from '../../components/datasets/VisionModelField';
import { SERVICE_NAMES, TAGGER_SERIES, modelName, resolveService, useTaggingSettings, useVisionModels, useVlmError, useVlmModels, useVlmServices, type TaggingSettings as Tagging, type VlmSettings } from '../../components/datasets/visionHooks';
import { SettingsSections } from './SettingsSections';
import '../../components/datasets/dataset-vision.css';
import './tagging-settings.css';

const CUSTOM = '__custom';
const ADDRESS = /^https?:\/\/[^\s/]+(\/\S*)?$/i;
const MODEL_DESCRIPTIONS: Record<string, [string, string]> = {
  'wd-eva02-tagger-2026-canary': ['EVA02 架构的新版实验模型，适合尝试更细的标签识别。', 'New EVA02 experimental model for detailed tag recognition.'],
  'wd-eva02-large-tagger-v3': ['EVA02-Large v3，通用动漫标签识别，当前推荐。', 'EVA02-Large v3 for general anime tagging; recommended.'],
  'wd-vit-large-tagger-v3': ['ViT-Large v3，通用动漫标签识别。', 'ViT-Large v3 for general anime tagging.'],
  'wd-swinv2-tagger-v3': ['SwinV2 v3，通用动漫标签识别。', 'SwinV2 v3 for general anime tagging.'],
  'wd-convnext-tagger-v3': ['ConvNeXt v3，通用动漫标签识别。', 'ConvNeXt v3 for general anime tagging.'],
  'wd-vit-tagger-v3': ['ViT v3，体积较小的通用动漫标签模型。', 'ViT v3, a smaller general anime tagger.'],
  'wd-v1-4-moat-tagger-v2': ['MOAT v2，通用动漫标签识别。', 'MOAT v2 for general anime tagging.'],
  'pixai-tagger-v1.0': ['PixAI v1.0，支持通用、角色、作品和画师等分类。', 'PixAI v1.0 with general, character, copyright and artist categories.'],
  'pixai-tagger-v0.9': ['PixAI v0.9，支持通用和角色标签。', 'PixAI v0.9 for general and character tags.'],
  'cl-tagger-v2-01a': ['CL Tagger v2.01a，按类别输出标签。', 'CL Tagger v2.01a with categorized output.'],
  'cl-tagger-1-02': ['CL Tagger v1.02，按类别输出标签。', 'CL Tagger v1.02 with categorized output.'],
};
function Field({ id, label, hint, children }: { id?: string; label: string; hint?: React.ReactNode; children: React.ReactNode }) {
  return <div className="settings-field">{id ? <label htmlFor={id}>{label}</label> : <span className="settings-field-label">{label}</span>}
    <div className="settings-field-control">{children}{hint && <p className="settings-note">{hint}</p>}</div></div>;
}

function KeyField({ service }: { service: VlmService }) {
  const text = useWorkspaceText();
  const client = useQueryClient();
  const [editing, setEditing] = React.useState(false);
  const [value, setValue] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');
  const act = async (request: () => Promise<unknown>) => {
    setBusy(true); setError('');
    try { await request(); setEditing(false); setValue(''); await client.invalidateQueries({ queryKey: ['vlm-services'] }); } catch (e) { setError(formatApiError(e)); } finally { setBusy(false); }
  };
  const save = (event: React.FormEvent) => { event.preventDefault(); if (value.trim()) void act(() => apiClient.put(`/vlm/services/${service.id}/key`, { api_key: value.trim() }, { silent: true })); };
  const local = service.editable && service.id !== 'custom';
  return <Field id="tagging-vlm-key" label={text('API 密钥', 'API key')} hint={error ? <span role="alert" className="tagging-settings-error">{error}</span> : text('保存在训练服务器上，之后不再显示。', 'Kept on the training server and never shown again.')}>
    {service.key_configured && !editing
      ? <span className="tagging-settings-inline"><span className="tagging-settings-saved"><CircleCheck size={14}/>{text('已保存', 'Saved')}</span>
        <button type="button" className="ui-btn ui-btn-sm" disabled={busy} onClick={() => setEditing(true)}>{text('更换', 'Replace')}</button>
        <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet ui-btn-danger" disabled={busy} onClick={() => { if (window.confirm(text('删除保存的 API 密钥？', 'Delete the saved API key?'))) void act(() => apiClient.delete(`/vlm/services/${service.id}/key`, { silent: true })); }}>{text('删除', 'Delete')}</button></span>
      : <form className="tagging-settings-inline" onSubmit={save}>
        <input id="tagging-vlm-key" className="settings-input" type="password" autoComplete="off" spellCheck={false} value={value} disabled={busy} placeholder={local ? text('本地服务通常不需要', 'Usually not needed locally') : text('粘贴 API 密钥', 'Paste the API key')} onChange={event => setValue(event.target.value)}/>
        <button type="submit" className="ui-btn" disabled={busy || !value.trim()}>{busy && <Loader2 size={13} className="animate-spin"/>}{text('保存密钥', 'Save key')}</button>
        {service.key_configured && <button type="button" className="ui-btn ui-btn-quiet" disabled={busy} onClick={() => { setEditing(false); setValue(''); }}>{text('取消', 'Cancel')}</button>}
      </form>}
  </Field>;
}

/** The model to use; the service's list is read only when asked for, never on its own. */
function ModelField({ service, baseUrl, value, onChange }: { service: VlmService; baseUrl: string; value: string; onChange: (model: string) => void }) {
  const text = useWorkspaceText();
  const describe = useVlmError();
  const box = React.useRef<HTMLDivElement>(null);
  const models = useVlmModels(service.id, baseUrl, false);
  const listed = models.data?.models || [];
  const [typing, setTyping] = React.useState(false);
  const addressOk = !service.editable || ADDRESS.test(baseUrl);
  const keyMissing = !service.editable && !service.key_configured;
  const custom = typing || !listed.length || (!!value && !listed.includes(value));
  const options = [...listed.map(model => ({ value: model, label: model })), { value: CUSTOM, label: text('手动输入…', 'Enter manually…') }];
  const choose = (next: string) => { if (next === CUSTOM) { setTyping(true); onChange(''); } else { setTyping(false); onChange(next); } };
  const hint = models.error ? <span role="alert" className="tagging-settings-error">{describe(models.error)}</span>
    : keyMissing ? text('先保存 API 密钥，再获取模型列表。', 'Save the API key before listing models.')
    : !addressOk ? text('先填写接口地址，再获取模型列表。', 'Enter the address before listing models.')
    : listed.length ? text(`已获取 ${listed.length} 个模型；需要支持图片输入的模型。`, `${listed.length} models listed; pick one that accepts images.`)
    : text('需要支持图片输入的模型。', 'Pick a model that accepts images.');
  return <Field id="tagging-vlm-model" label={text('模型', 'Model')} hint={hint}>
    <span className="tagging-settings-inline">
      {custom ? <div ref={box} className="vision-combo">
        <input id="tagging-vlm-model" className="settings-input" type="text" value={value} maxLength={200} spellCheck={false} placeholder={text('手动输入，例如 gpt-4o-mini', 'Enter manually, e.g. gpt-4o-mini')} onChange={event => onChange(event.target.value.trim())}/>
        {!!listed.length && <StudioSelect className="vision-combo-toggle" anchorRef={box} searchable aria-label={text('从列表选择模型', 'Choose a model from the list')} value={CUSTOM} options={options} onValueChange={choose}/>}
      </div> : <StudioSelect id="tagging-vlm-model" searchable aria-label={text('模型', 'Model')} value={value} placeholder={text('选择模型', 'Choose a model')} options={options} onValueChange={choose}/>}
      <button type="button" className="ui-btn" disabled={keyMissing || !addressOk || models.isFetching} onClick={() => void models.refetch()}>{models.isFetching ? <Loader2 size={14} className="animate-spin"/> : <RefreshCw size={14}/>}{text('获取模型列表', 'List models')}</button>
    </span>
  </Field>;
}

function NumberInput({ id, value, min, max, step, optional = false, placeholder, onChange }: { id: string; value: number | null | undefined; min: number; max: number; step: number; optional?: boolean; placeholder?: string; onChange: (value: number | null) => void }) {
  return <input id={id} type="number" className="settings-input tagging-settings-number" value={value ?? ''} min={min} max={max} step={step} placeholder={placeholder} onChange={event => {
    if (event.target.value === '') { if (optional) onChange(null); return; }
    const next = Number(event.target.value);
    if (Number.isFinite(next)) onChange(Math.min(max, Math.max(min, next)));
  }}/>;
}

export default function TaggingSettings() {
  const text = useWorkspaceText();
  const { tagging, query, save } = useTaggingSettings();
  const services = useVlmServices();
  const catalog = useVisionModels();
  const [draft, setDraft] = React.useState<Tagging | null>(null);
  const [taggerFamily, setTaggerFamily] = React.useState('wd');
  const [saving, setSaving] = React.useState(false);
  const [notice, setNotice] = React.useState<{ error?: string; saved?: boolean }>({});
  const current = draft ?? tagging;
  const vlm = (current?.vlm || {}) as VlmSettings;
  const change = (patch: Partial<Tagging>) => { setDraft({ ...(current as Tagging), ...patch }); setNotice({}); };
  const changeVlm = (patch: Partial<VlmSettings>) => change({ vlm: { ...vlm, ...patch } });
  const resolved = resolveService(vlm, services.data?.services);
  const name = (id: string) => { const pair = SERVICE_NAMES[id] || [id, id]; return text(pair[0], pair[1]); };
  const submit = async () => {
    if (!draft) return;
    setSaving(true); setNotice({});
    try { await save(draft); setDraft(null); setNotice({ saved: true }); } catch (e) { setNotice({ error: formatApiError(e) }); } finally { setSaving(false); }
  };
  const sections = [
    { id: 'tagging-vlm', label: text('视觉大模型', 'Vision model') },
    { id: 'tagging-requests', label: text('请求参数', 'Requests') },
    { id: 'tagging-models', label: text('打标与遮罩模型', 'Tagging and mask models') },
  ];
  if (query.isPending || services.isPending) return <LoadingNote block label={text('正在读取打标设置…', 'Loading tagging settings…')}/>;
  if (query.error || services.error || !current) return <div role="alert" className="settings-alert">{formatApiError(query.error || services.error)}<button type="button" className="ui-link ml-2" onClick={() => { void query.refetch(); void services.refetch(); }}>{text('重新读取', 'Reload')}</button></div>;
  const models = catalog.data?.models || [];
  const taggers = models.filter(model => model.role === 'tagger');
  const detectors = models.filter(model => model.role !== 'tagger');
  const modelRow = (model: typeof models[number], series: string) => <div key={model.id} className="tagging-settings-model" data-testid={`tagging-model-${model.id}`}>
    <div className="tagging-settings-model-name"><strong>{series ? `${series} · ${modelName(model)}` : model.label}</strong>
      {MODEL_DESCRIPTIONS[model.id] && <p>{text(MODEL_DESCRIPTIONS[model.id][0], MODEL_DESCRIPTIONS[model.id][1])}</p>}
      <span>{model.license === 'unspecified' ? text('未声明许可', 'No licence stated') : model.license}{model.token_required ? text(' · 需要 Hugging Face 令牌', ' · Hugging Face token needed') : ''}</span></div>
    <div className="tagging-settings-model-state"><VisionModelState model={model}/></div>
  </div>;
  return <div className="tagging-settings" data-testid="tagging-settings"><SettingsSections sections={sections}>
    <section id="tagging-vlm" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><div><h2>{text('视觉大模型', 'Vision model')}</h2><p className="settings-note">{text('视觉大模型打标和辅助打标使用这个服务。', 'Vision model tagging and assisted tagging use this service.')}</p></div></div>
      <Field id="tagging-vlm-service" label={text('服务', 'Service')} hint={text('OpenAI 兼容接口；本地模型可用 Ollama 或 LM Studio。', 'Any OpenAI-compatible API; Ollama or LM Studio for local models.')}>
        <StudioSelect id="tagging-vlm-service" value={resolved?.service.id || ''} onValueChange={provider => changeVlm({ provider: provider as VlmSettings['provider'] })} options={(services.data?.services || []).map(item => ({ value: item.id, label: name(item.id) }))}/>
      </Field>
      {resolved && <>
        <Field id="tagging-vlm-address" label={text('接口地址', 'Address')} hint={resolved.service.editable ? text('由训练服务器发出请求，同一台机器上的服务填 127.0.0.1。', 'The training server sends the requests; use 127.0.0.1 for a service on the same machine.') : text('该服务使用固定地址。', 'This service has a fixed address.')}>
          <input id="tagging-vlm-address" className="settings-input" type="text" value={resolved.baseUrl} disabled={!resolved.service.editable} readOnly={!resolved.service.editable} spellCheck={false} maxLength={500} placeholder="http://127.0.0.1:8000/v1" onChange={event => changeVlm({ base_urls: { ...(vlm.base_urls || {}), [resolved.service.id]: event.target.value.trim() } })}/>
        </Field>
        <KeyField key={resolved.service.id} service={resolved.service}/>
        <ModelField key={`${resolved.service.id}/${resolved.baseUrl}`} service={resolved.service} baseUrl={resolved.baseUrl} value={resolved.model} onChange={model => changeVlm({ models: { ...(vlm.models || {}), [resolved.service.id]: model } })}/>
      </>}
    </section>
    <section id="tagging-requests" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><div><h2>{text('请求参数', 'Requests')}</h2><p className="settings-note">{text('对之后开始的打标生效。', 'Apply to tagging runs started afterwards.')}</p></div></div>
      <Field id="tagging-temperature" label={text('温度', 'Temperature')} hint={text('0 使用模型默认值；大于 0 时发送指定温度。', '0 uses the model default; positive values send a specific temperature.')}><NumberInput id="tagging-temperature" value={vlm.temperature ?? 0} min={0} max={2} step={0.1} onChange={value => changeVlm({ temperature: value ?? 0 })}/></Field>
      <Field id="tagging-max-tokens" label={text('最大输出长度', 'Max output tokens')} hint={text('留空不限制；思考型模型需要留足。', 'Empty for no limit; reasoning models need room.')}><NumberInput id="tagging-max-tokens" optional value={vlm.max_tokens} min={16} max={65536} step={1} placeholder={text('不限制', 'No limit')} onChange={value => changeVlm({ max_tokens: value })}/></Field>
      <Field id="tagging-image-size" label={text('图片长边', 'Image size')} hint={text('发送前缩到这个尺寸，越大越费额度。', 'Images are scaled down to this before sending; larger costs more.')}>
        <StudioSelect id="tagging-image-size" value={String(vlm.image_size ?? 1024)} onValueChange={value => changeVlm({ image_size: Number(value) })} options={[512, 768, 1024, 1536, 2048].map(size => ({ value: String(size), label: `${size} px` }))}/></Field>
      <Field id="tagging-image-detail" label={text('图像细节', 'Image detail')} hint={text('OpenAI 的 detail 参数，接口不支持时选“不发送”。', 'OpenAI’s detail setting; choose “Not sent” when the API rejects it.')}>
        <StudioSelect id="tagging-image-detail" value={vlm.image_detail ?? ''} onValueChange={value => changeVlm({ image_detail: value as VlmSettings['image_detail'] })} options={[{ value: '', label: text('Not sent (不发送)', 'Not sent') }, { value: 'auto', label: text('Auto (自动)', 'Auto') }, { value: 'low', label: text('Low (低)', 'Low') }, { value: 'high', label: text('High (高)', 'High') }]}/></Field>
      <Field id="tagging-concurrency" label={text('并发数', 'Parallel requests')} hint={text('同时发送的请求数，受服务限速约束。', 'Requests in flight at once, within the service’s rate limit.')}><NumberInput id="tagging-concurrency" value={vlm.concurrency} min={1} max={16} step={1} onChange={value => changeVlm({ concurrency: value ?? 1 })}/></Field>
      <Field id="tagging-interval" label={text('请求间隔（秒）', 'Interval (s)')} hint={text('两次请求开始之间的最短间隔。', 'Shortest time between two request starts.')}><NumberInput id="tagging-interval" value={vlm.interval} min={0} max={120} step={0.5} onChange={value => changeVlm({ interval: value ?? 0 })}/></Field>
      <Field id="tagging-timeout" label={text('超时（秒）', 'Timeout (s)')} hint={text('单次请求的最长等待时间。', 'Longest wait for one reply.')}><NumberInput id="tagging-timeout" value={vlm.timeout} min={10} max={900} step={10} onChange={value => changeVlm({ timeout: value ?? 120 })}/></Field>
      <Field id="tagging-retries" label={text('重试次数', 'Retries')} hint={text('限速、服务错误或超时后重试。', 'Retried after rate limits, server errors or timeouts.')}><NumberInput id="tagging-retries" value={vlm.retries} min={0} max={5} step={1} onChange={value => changeVlm({ retries: value ?? 0 })}/></Field>
    </section>
    <section id="tagging-models" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><div><h2>{text('打标与遮罩模型', 'Tagging and mask models')}</h2><p className="settings-note">{text('存放在模型目录的 tagger 和 mask 文件夹，按系列和版本分开。', 'Stored in the tagger and mask folders of the model directory, by series and version.')}</p></div></div>
      <Field id="tagging-source" label={text('下载来源', 'Download source')} hint={text('没有魔搭社区副本的模型从 Hugging Face 下载。', 'Models without a ModelScope copy come from Hugging Face.')}>
        <StudioSelect id="tagging-source" value={current.model_source} onValueChange={value => change({ model_source: value as Tagging['model_source'] })} options={[{ value: 'huggingface', label: 'Hugging Face' }, { value: 'modelscope', label: text('魔搭社区', 'ModelScope') }]}/>
      </Field>
      {catalog.isPending ? <LoadingNote label={text('读取模型列表…', 'Loading models…')}/> : catalog.error ? <p role="alert" className="settings-alert">{formatApiError(catalog.error)}</p> : <>
        <h3 className="tagging-settings-group">{text('Tagger 模型', 'Tagger models')}</h3>
        <div className="tagging-settings-tabs" role="tablist" aria-label={text('Tagger 模型系列', 'Tagger model series')}>
          {['wd', 'cl', 'pixai'].filter(family => taggers.some(model => model.family === family)).map(family => <button key={family} type="button" role="tab" aria-selected={taggerFamily === family} onClick={() => setTaggerFamily(family)}>{TAGGER_SERIES[family] || family}</button>)}
        </div>
        <div className="tagging-settings-models">{taggers.filter(model => model.family === taggerFamily).map(model => modelRow(model, TAGGER_SERIES[model.family] || model.family))}</div>
        <h3 className="tagging-settings-group">{text('遮罩检测模型', 'Mask detectors')}</h3>
        <div className="tagging-settings-models">{detectors.map(model => modelRow(model, ''))}</div>
      </>}
    </section>
  </SettingsSections>
  <div className="settings-save">{notice.error && <span role="alert" className="tagging-settings-error">{notice.error}</span>}<span role="status" className="settings-note">{notice.saved && !draft && text('已保存', 'Saved')}</span>
    <button type="button" className="ui-btn ui-btn-primary" disabled={saving || !draft} onClick={() => void submit()}>{saving && <Loader2 size={14} className="animate-spin"/>}{text('保存', 'Save')}</button></div>
  </div>;
}
