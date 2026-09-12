import React from 'react';
import { useBlocker, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Copy, Plus, Save, Search, Trash2, SlidersHorizontal } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { FamilyInfo, Preset } from '../../api/types';
import Dialog from '../../components/Dialog';
import StudioSelect from '../../components/StudioSelect';
import { SchemaForm, type ValidationError } from '../../schema/SchemaForm/SchemaForm';
import { CONFIG_TAB_GROUPS, type ConfigTab } from '../../utils/configPresentation';
import { mergeConfig } from '../../utils/config';
import { formatApiError } from '../../utils/errors';
import { presetEditorSchema, presetPayload, presetSummary, presetFamily } from '../../utils/presetEditor';
import { reusableTrainingPreset } from '../../utils/trainingPresets';
import { useWorkspaceText } from '../../utils/workspaceText';
import './presets.css';

interface Draft { name: string; description: string; config: Record<string, any>; originalName: string | null; builtin: boolean; }
const KEY = ['standalone-presets'];

export default function Presets() {
  const text = useWorkspaceText();
  const english = text('zh', 'en') === 'en';
  const queryClient = useQueryClient();
  const [params] = useSearchParams();
  const list = useQuery({ queryKey: KEY, queryFn: () => apiClient.get<Preset[]>('/presets', { silent: true }) });
  const schema = useQuery({ queryKey: ['training-schema'], queryFn: () => apiClient.get<any>('/schema/train', { silent: true }) });
  const families = useQuery({ queryKey: ['families'], queryFn: () => apiClient.get<FamilyInfo[]>('/families', { silent: true }) });
  const editorSchema = React.useMemo(() => schema.data ? presetEditorSchema(schema.data) : null, [schema.data]);
  const [draft, setDraft] = React.useState<Draft | null>(null);
  const [saved, setSaved] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');
  const [notice, setNotice] = React.useState('');
  const [errors, setErrors] = React.useState<ValidationError[]>([]);
  const [search, setSearch] = React.useState('');
  const [advanced, setAdvanced] = React.useState(false);
  const [tab, setTab] = React.useState<ConfigTab>('train');
  const [pending, setPending] = React.useState<(() => void) | null>(null);
  const [deleting, setDeleting] = React.useState(false);
  const startingConfig = React.useRef('');
  const userPresets = React.useMemo(() => (list.data || []).filter(item => !item.builtin).sort((a,b) => (b.updated_at || 0) - (a.updated_at || 0) || a.name.localeCompare(b.name)), [list.data]);
  const dirty = !!draft && !draft.builtin && JSON.stringify(presetPayload(draft)) !== saved;
  const blocker = useBlocker(({ currentLocation, nextLocation }) => {
    const background = (nextLocation.state as { backgroundLocation?: { pathname?: string } } | null)?.backgroundLocation;
    return dirty && nextLocation.pathname !== currentLocation.pathname && !(nextLocation.pathname.startsWith('/settings') && background?.pathname === '/presets');
  });
  React.useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [dirty]);

  const begin = async (preset?: Preset, copy = false, family = (preset && presetFamily(preset.config)) || 'anima', newDraft = false) => {
    setBusy(true); setError(''); setNotice('');
    try {
      const defaults = await apiClient.get<Record<string, any>>('/config/defaults', { params: { family }, silent: true });
      let name = preset?.name || '';
      if (copy) {
        const base = `${name}-copy`; name = base;
        for (let i = 2; list.data?.some(item => item.name.toLocaleLowerCase() === name.toLocaleLowerCase()); i += 1) name = `${base}-${i}`;
      }
      const existing = preset && !copy && !newDraft;
      const next: Draft = { name, description: preset?.description || '', config: reusableTrainingPreset(mergeConfig(defaults, preset?.config || {})), originalName: existing ? preset.name : null, builtin: false };
      next.config.model = { ...next.config.model, family };
      startingConfig.current = JSON.stringify(next.config);
      setDraft(next); setSaved(existing ? JSON.stringify(presetPayload(next)) : JSON.stringify(presetPayload({...next,name:'',description:''})));
      setErrors([]); setSearch(''); setTab('train');
    } catch (failure) { setError(formatApiError(failure)); }
    finally { setBusy(false); }
  };
  const requestAction = (action: () => void) => { if (busy) return; if (dirty) setPending(() => action); else action(); };
  const initial = React.useRef(false);
  React.useEffect(() => {
    if (!list.data || !families.data || initial.current) return;
    initial.current = true;
    const name = params.get('name');
    if (name) {
      const selected = userPresets.find(item => item.name === name);
      if (selected) void begin(selected);
      else setError(text('找不到指定预设，请选择已有预设或新建。', 'The requested preset was not found. Choose another or create one.'));
    } else void begin(userPresets[0], false, userPresets[0] ? presetFamily(userPresets[0].config) : families.data.find(item=>item.name==='anima')?.name || families.data[0]?.name || 'anima');
    // A list refresh must never replace an editor's unsaved draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [list.data, families.data]);

  const save = async (): Promise<boolean> => {
    if (!draft || draft.builtin || busy) return false;
    const body = presetPayload(draft);
    if (!/^[\p{L}\p{N}_-]{1,128}$/u.test(body.name) || !/[\p{L}\p{N}]/u.test(body.name)) {
      setError(text('预设名称须为 1–128 个字母、数字、中文、短横线或下划线。', 'Use 1–128 letters or numbers, with hyphens or underscores, for the preset name.'));
      return false;
    }
    setBusy(true); setError(''); setNotice(''); setErrors([]);
    try {
      const item = draft.originalName
        ? await apiClient.put<Preset>(`/presets/${encodeURIComponent(draft.originalName)}`, body, { silent: true })
        : await apiClient.post<Preset>('/presets', body, { silent: true });
      const next = { ...draft, name: item.name, description: item.description, config: item.config, originalName: item.name, builtin: false };
      setDraft(next); setSaved(JSON.stringify(presetPayload(next)));
      queryClient.setQueryData<Preset[]>(KEY, previous => [...(previous || []).filter(row => row.name !== item.name), item]);
      setNotice(text('预设已保存，可在项目训练参数中加载。', 'Preset saved. It is available in project training parameters.'));
      return true;
    } catch (failure) {
      setError(formatApiError(failure));
      const details = (failure as { details?: { errors?: ValidationError[] } })?.details?.errors;
      if (Array.isArray(details)) setErrors(details);
      return false;
    } finally { setBusy(false); }
  };
  const remove = async () => {
    if (!draft?.originalName || draft.builtin || busy) return;
    setBusy(true); setError('');
    try {
      await apiClient.delete(`/presets/${encodeURIComponent(draft.originalName)}`, { silent: true });
      queryClient.setQueryData<Preset[]>(KEY, previous => previous?.filter(item => item.name !== draft.originalName));
      const next = userPresets.find(item => item.name !== draft.originalName);
      setDraft(null); setSaved(''); setDeleting(false);
      await begin(next);
      setNotice(text('预设已删除。', 'Preset deleted.'));
    } catch (failure) { setError(formatApiError(failure)); }
    finally { setBusy(false); }
  };
  const proceed = () => {
    if (blocker.state === 'blocked') blocker.proceed();
    else { const action = pending; setPending(null); action?.(); }
  };
  const cancelLeave = () => { setPending(null); if (blocker.state === 'blocked') blocker.reset(); };
  const familyName = (name: string) => families.data?.find(item => item.name === name)?.label || (name || text('通用', 'General'));
  const family = families.data?.find(item => item.name === draft?.config.model?.family);
  const tabs: [ConfigTab, string][] = [['train', text('训练参数', 'Training')], ['data', text('数据与标签', 'Data and captions')], ['model', text('精度与保存', 'Precision and saving')], ['advanced', text('采样与高级', 'Sampling and advanced')]];

  const options = [
    ...(!draft?.originalName ? [{value:'',label:text('新预设', 'New preset'),disabled:true}] : []),
    ...userPresets.map(item=>({value:item.name,label:`${item.name} · ${familyName(presetFamily(item.config))}`})),
  ];
  return <section className="presets-page">
    <header className="presets-toolbar">
      <h1>{text('参数预设', 'Presets')}</h1>
      <div className="presets-switcher"><StudioSelect searchable aria-label={text('选择预设', 'Choose preset')} value={draft?.originalName || ''} disabled={busy || !userPresets.length} options={options} onValueChange={name=>{const item=userPresets.find(row=>row.name===name);if(item && name!==draft?.originalName)requestAction(()=>void begin(item));}}/></div>
      <div className="presets-actions">
        <button className="studio-secondary" disabled={busy || !schema.data || !families.data || list.isPending || list.isError} onClick={() => requestAction(() => void begin())}><Plus size={15}/>{text('新建预设', 'New preset')}</button>
        {draft?.originalName && <><button className="studio-secondary presets-icon-action" aria-label={text('复制为新预设', 'Duplicate')} title={text('复制为新预设', 'Duplicate')} disabled={busy} onClick={() => requestAction(() => void begin({name:draft.name,description:draft.description,config:draft.config,builtin:false,updated_at:null},true))}><Copy size={15}/></button><button className="studio-secondary presets-icon-action" aria-label={text('删除预设', 'Delete preset')} title={text('删除预设', 'Delete preset')} disabled={busy} onClick={()=>setDeleting(true)}><Trash2 size={15}/></button></>}
        <button className="studio-primary" disabled={busy || !dirty || !editorSchema} onClick={()=>void save()}><Save size={15}/>{busy ? text('保存中…', 'Saving…') : text('保存预设', 'Save preset')}</button>
      </div>
    </header>
    {error && !deleting && <div role="alert" className="studio-error">{error}</div>}
    {notice && <p role="status" className="presets-notice">{notice}</p>}
    {[list, schema, families].some(query => query.isError) && <div role="alert" className="studio-error"><span>{[list, schema, families].filter(query => query.error).map(query => formatApiError(query.error)).join('\n')}</span><button onClick={() => { for (const query of [list, schema, families]) if (query.isError) void query.refetch(); }}>{text('重新加载', 'Reload')}</button></div>}
    {!draft && !error && <p role="status" className="presets-loading">{text('正在读取参数…', 'Loading parameters…')}</p>}
    {draft && <div className="presets-editor">
      <fieldset disabled={busy} className="presets-meta">
        <label>{text('预设名称', 'Preset name')}<input aria-label={text('预设名称', 'Preset name')} placeholder={text('例如：人物_LoKr', 'For example: character_LoKr')} value={draft.name} readOnly={!!draft.originalName} onChange={event=>setDraft({...draft,name:event.target.value})}/></label>
        <label>{text('适用模型', 'Model family')}<StudioSelect aria-label={text('适用模型', 'Model family')} disabled={busy || !!draft.originalName} value={draft.config.model.family} onValueChange={target=>{const change=()=>{void begin({name:draft.name,description:draft.description,config:{model:{family:target}},builtin:false,updated_at:null},false,target,true);};if(JSON.stringify(draft.config)!==startingConfig.current)requestAction(change);else change();}} options={(families.data || []).filter(item=>item.name!=='toy'||draft.config.model.family==='toy').map(item=>({value:item.name,label:item.label}))}/></label>
        <label className="presets-description">{text('用途与说明', 'Description')}<input aria-label={text('用途与说明', 'Description')} placeholder={text('可选，记录用途或参数取舍', 'Optional: purpose or parameter choices')} value={draft.description} onChange={event=>setDraft({...draft,description:event.target.value})}/></label>
      </fieldset>
      <div className="presets-context"><span>{!draft.originalName ? text('填写名称并编辑参数后保存。', 'Name and edit the preset, then save.') : presetSummary(draft.config, english)}</span><span className={dirty ? 'presets-dirty' : ''}>{dirty ? text('有未保存修改', 'Unsaved changes') : draft.originalName ? text('已保存', 'Saved') : text('尚未创建', 'Not created yet')}</span></div>
      <div className="presets-form-toolbar">
        <div role="tablist" aria-label={text('预设参数分区', 'Preset parameter sections')}>{tabs.map(([key,label])=><button key={key} role="tab" aria-selected={tab===key} onClick={()=>setTab(key)}>{label}</button>)}</div>
        <label className="presets-field-search"><Search size={14}/><input aria-label={text('搜索预设参数', 'Search preset parameters')} placeholder={text('搜索参数', 'Search parameters')} value={search} onChange={event=>setSearch(event.target.value)}/></label>
        <label className="presets-advanced"><input type="checkbox" checked={advanced} onChange={event=>setAdvanced(event.target.checked)}/><SlidersHorizontal size={13}/>{text('高级选项', 'Advanced')}</label>
      </div>
      {editorSchema && <div className="presets-schema" role="tabpanel" aria-label={tabs.find(([key])=>key===tab)?.[1]}><SchemaForm readOnly={busy} schema={editorSchema} value={draft.config} onChange={config=>{setDraft({...draft,config});setErrors([]);}} compact showAdvanced={advanced} groupFilter={search ? undefined : CONFIG_TAB_GROUPS[tab]} search={search} family={family} families={families.data} errors={errors}/></div>}
    </div>}
    {(pending || blocker.state === 'blocked') && <Dialog title={text('保存预设修改？', 'Save preset changes?')} onClose={cancelLeave} closeDisabled={busy}><p>{text('当前预设尚未保存。可以先保存，或放弃这些修改。', 'This preset has unsaved changes. Save them or discard the draft.')}</p>{error && <p role="alert" className="studio-error">{error}</p>}<div className="presets-confirm-actions"><button className="studio-secondary" disabled={busy} onClick={cancelLeave}>{text('继续编辑', 'Keep editing')}</button><button className="studio-secondary" disabled={busy} onClick={proceed}>{text('放弃修改', 'Discard changes')}</button><button className="studio-primary" disabled={busy} onClick={()=>{void save().then(ok=>{if(ok)proceed();});}}>{text('保存并继续', 'Save and continue')}</button></div></Dialog>}
    {deleting && <Dialog title={text('删除预设', 'Delete preset')} onClose={()=>setDeleting(false)} closeDisabled={busy}><p>{text('删除后不会改变任何已有项目配置。确定删除', 'Existing project configurations will remain unchanged. Delete')} “{draft?.name}”?</p>{error&&<p role="alert" className="studio-error">{error}</p>}<div className="presets-confirm-actions"><button className="studio-secondary" disabled={busy} onClick={()=>setDeleting(false)}>{text('取消', 'Cancel')}</button><button className="studio-primary" disabled={busy} onClick={()=>void remove()}>{text('确认删除', 'Confirm deletion')}</button></div></Dialog>}
  </section>;
}
