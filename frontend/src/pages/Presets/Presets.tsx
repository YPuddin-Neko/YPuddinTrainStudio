import React from 'react';
import { useBlocker, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Copy, Plus, Save, Search, Trash2 } from 'lucide-react';
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
  const [filter, setFilter] = React.useState('');
  const [familyFilter, setFamilyFilter] = React.useState('');
  const [search, setSearch] = React.useState('');
  const [advanced, setAdvanced] = React.useState(false);
  const [tab, setTab] = React.useState<ConfigTab>('train');
  const [pending, setPending] = React.useState<(() => void) | null>(null);
  const [deleting, setDeleting] = React.useState(false);
  const startingConfig = React.useRef('');
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
      const next: Draft = { name, description: preset?.description || '', config: reusableTrainingPreset(mergeConfig(defaults, preset?.config || {})), originalName: existing ? preset.name : null, builtin: !!preset?.builtin && !copy && !newDraft };
      next.config.model = { ...next.config.model, family };
      startingConfig.current = JSON.stringify(next.config);
      setDraft(next); setSaved(existing ? JSON.stringify(presetPayload(next)) : '');
      setErrors([]); setSearch(''); setTab('train');
    } catch (failure) { setError(formatApiError(failure)); }
    finally { setBusy(false); }
  };
  const requestAction = (action: () => void) => { if (busy) return; if (dirty) setPending(() => action); else action(); };
  const initial = React.useRef(false);
  React.useEffect(() => {
    if (!list.data || initial.current) return;
    initial.current = true;
    const name = params.get('name');
    if (name) {
      const selected = list.data.find(item => item.name === name);
      if (selected) void begin(selected);
      else setError(text('找不到指定预设，请从列表选择。', 'The requested preset was not found. Choose one from the list.'));
    }
    // A list refresh must never replace an editor's unsaved draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [list.data]);

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
      const next = { ...draft, name: item.name, description: item.description, config: item.config, originalName: item.name, builtin: item.builtin };
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
      setDraft(null); setSaved(''); setDeleting(false); setNotice(text('预设已删除。', 'Preset deleted.'));
    } catch (failure) { setError(formatApiError(failure)); }
    finally { setBusy(false); }
  };
  const proceed = () => {
    if (blocker.state === 'blocked') blocker.proceed();
    else { const action = pending; setPending(null); action?.(); }
  };
  const cancelLeave = () => { setPending(null); if (blocker.state === 'blocked') blocker.reset(); };
  const familyName = (name: string) => families.data?.find(item => item.name === name)?.label || (name || text('通用', 'General'));
  const visible = (list.data || []).filter(item => (!familyFilter || presetFamily(item.config) === familyFilter) && `${item.name} ${item.description}`.toLocaleLowerCase().includes(filter.toLocaleLowerCase()));
  const groups = [...new Set(visible.map(item => presetFamily(item.config)))];
  const family = families.data?.find(item => item.name === draft?.config.model?.family);
  const tabs: [ConfigTab, string][] = [['train', text('训练参数', 'Training')], ['data', text('数据与标签', 'Data and captions')], ['model', text('精度与保存', 'Precision and saving')], ['advanced', text('采样与高级', 'Sampling and advanced')]];

  return <section className="presets-page">
    <header className="presets-heading"><div><h1>{text('参数预设', 'Training presets')}</h1><p>{text('独立管理可复用参数，项目的数据、模型文件和输出目录保持独立。', 'Manage reusable parameters independently from project data, model files and output folders.')}</p></div><button className="studio-primary" disabled={busy || !schema.data || !families.data} onClick={() => requestAction(() => void begin())}><Plus size={15}/>{text('新建预设', 'New preset')}</button></header>
    {error && !deleting && <div role="alert" className="studio-error">{error}</div>}
    {notice && <p role="status" className="presets-notice">{notice}</p>}
    {[list, schema, families].some(query => query.isError) && <div role="alert" className="studio-error"><span>{[list, schema, families].filter(query => query.error).map(query => formatApiError(query.error)).join('\n')}</span><button onClick={() => { for (const query of [list, schema, families]) if (query.isError) void query.refetch(); }}>{text('重新加载', 'Reload')}</button></div>}
    <div className="presets-columns">
      <aside className="presets-library" aria-label={text('预设列表', 'Preset library')}>
        <div className="presets-library-tools"><label><Search size={14}/><input aria-label={text('搜索预设', 'Search presets')} placeholder={text('搜索名称或描述', 'Search name or description')} value={filter} onChange={event => setFilter(event.target.value)}/></label><StudioSelect aria-label={text('按模型筛选预设', 'Filter presets by model')} value={familyFilter} onValueChange={setFamilyFilter} options={[{value:'',label:text('全部模型', 'All models')},...(families.data || []).map(item => ({value:item.name,label:item.label}))]}/></div>
        {list.isPending && <p role="status">{text('正在读取预设…', 'Loading presets…')}</p>}
        {!list.isPending && !visible.length && <p className="presets-empty">{text('暂无匹配预设。', 'No matching presets.')}</p>}
        {groups.map(group => <section key={group}><h2>{familyName(group)}</h2>{visible.filter(item => presetFamily(item.config) === group).map(item => <button key={item.name} disabled={busy} className={`preset-list-item ${draft?.originalName === item.name ? 'selected' : ''}`} onClick={() => requestAction(() => void begin(item))} aria-label={`${text('打开预设', 'Open preset')} ${item.name}`}><span><strong>{item.name}</strong><small>{item.builtin ? text('内置', 'Built-in') : text('自定义', 'Custom')}</small></span><p>{item.description || text('暂无描述', 'No description')}</p><small>{presetSummary(item.config, english)}</small></button>)}</section>)}
      </aside>
      <div className="presets-editor">
        {!draft && <div className="presets-empty">{text('从列表打开预设，或新建一份参数配置。内置预设可以查看和复制。', 'Open a preset or create a new configuration. Built-in presets can be viewed and copied.')}</div>}
        {draft && <>
          <div className="presets-editor-actions"><strong>{draft.builtin ? text('内置预设 · 只读', 'Built-in preset · read only') : dirty ? text('有未保存的修改', 'Unsaved changes') : text('已保存', 'Saved')}</strong><span/>{draft.originalName && <button className="studio-secondary" disabled={busy} onClick={() => requestAction(() => void begin({name:draft.name,description:draft.description,config:draft.config,builtin:draft.builtin,updated_at:null},true))}><Copy size={14}/>{text('复制为新预设', 'Duplicate')}</button>}{!draft.builtin && draft.originalName && <button className="studio-secondary" disabled={busy} onClick={() => setDeleting(true)}><Trash2 size={14}/>{text('删除预设', 'Delete preset')}</button>}{!draft.builtin && <button className="studio-primary" disabled={busy || !dirty || !editorSchema} onClick={() => void save()}><Save size={14}/>{busy ? text('正在保存…', 'Saving…') : text('保存预设', 'Save preset')}</button>}</div>
          <fieldset disabled={busy || draft.builtin} className="presets-meta"><label>{text('预设名称', 'Preset name')}<input aria-label={text('预设名称', 'Preset name')} value={draft.name} readOnly={!!draft.originalName} onChange={event => setDraft({...draft,name:event.target.value})}/></label><label>{text('适用模型', 'Model family')}<StudioSelect aria-label={text('适用模型', 'Model family')} disabled={busy || draft.builtin || !!draft.originalName} value={draft.config.model.family} onValueChange={target => { const change = () => { void begin({name:draft.name,description:draft.description,config:{model:{family:target}},builtin:false,updated_at:null},false,target,true); }; if (JSON.stringify(draft.config) !== startingConfig.current) requestAction(change); else change(); }} options={(families.data || []).map(item => ({value:item.name,label:item.label}))}/></label><label className="presets-description">{text('用途与说明', 'Description')}<textarea rows={2} aria-label={text('用途与说明', 'Description')} value={draft.description} onChange={event => setDraft({...draft,description:event.target.value})}/></label></fieldset>
          <p className="preset-parameter-summary">{presetSummary(draft.config, english)}</p>
          <div className="presets-form-toolbar"><div role="tablist" aria-label={text('预设参数分区', 'Preset parameter sections')}>{tabs.map(([key,label])=><button key={key} role="tab" aria-selected={tab===key} onClick={()=>setTab(key)}>{label}</button>)}</div><label><input type="checkbox" checked={advanced} onChange={event=>setAdvanced(event.target.checked)}/>{text('高级选项', 'Advanced')}</label></div>
          <label className="presets-field-search"><Search size={14}/><input aria-label={text('搜索预设参数', 'Search preset parameters')} placeholder={text('搜索参数', 'Search parameters')} value={search} onChange={event=>setSearch(event.target.value)}/></label>
          {editorSchema && <div className="presets-schema"><SchemaForm readOnly={busy || draft.builtin} schema={editorSchema} value={draft.config} onChange={config=>{setDraft({...draft,config});setErrors([]);}} compact showAdvanced={advanced} groupFilter={search ? undefined : CONFIG_TAB_GROUPS[tab]} search={search} family={family} families={families.data} errors={errors}/></div>}
        </>}
      </div>
    </div>
    {(pending || blocker.state === 'blocked') && <Dialog title={text('保存预设修改？', 'Save preset changes?')} onClose={cancelLeave} closeDisabled={busy}><p>{text('当前预设尚未保存。可以先保存，或放弃这些修改。', 'This preset has unsaved changes. Save them or discard the draft.')}</p>{error && <p role="alert" className="studio-error">{error}</p>}<div className="presets-confirm-actions"><button className="studio-secondary" disabled={busy} onClick={cancelLeave}>{text('继续编辑', 'Keep editing')}</button><button className="studio-secondary" disabled={busy} onClick={proceed}>{text('放弃修改', 'Discard changes')}</button><button className="studio-primary" disabled={busy} onClick={()=>{void save().then(ok=>{if(ok)proceed();});}}>{text('保存并继续', 'Save and continue')}</button></div></Dialog>}
    {deleting && <Dialog title={text('删除预设', 'Delete preset')} onClose={()=>setDeleting(false)} closeDisabled={busy}><p>{text('删除后不会改变任何已有项目配置。确定删除', 'Existing project configurations will remain unchanged. Delete')} “{draft?.name}”?</p>{error&&<p role="alert" className="studio-error">{error}</p>}<div className="presets-confirm-actions"><button className="studio-secondary" disabled={busy} onClick={()=>setDeleting(false)}>{text('取消', 'Cancel')}</button><button className="studio-primary" disabled={busy} onClick={()=>void remove()}>{text('确认删除', 'Confirm deletion')}</button></div></Dialog>}
  </section>;
}
