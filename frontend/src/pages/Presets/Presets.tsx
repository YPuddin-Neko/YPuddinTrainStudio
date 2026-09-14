import React from 'react';
import { useBlocker, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Copy, Plus, Save, Search, Trash2, X, ChevronRight, ChevronDown } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Preset } from '../../api/types';
import { useFamilies } from '../../api/hooks/useFamilies';
import Dialog from '../../components/Dialog';
import StudioSelect from '../../components/StudioSelect';
import { SchemaForm, type ValidationError } from '../../schema/SchemaForm/SchemaForm';
import { CONFIG_TAB_GROUPS, presentConfigIssues, type ConfigTab, type ConfigIssue } from '../../utils/configPresentation';
import { mergeConfig } from '../../utils/config';
import { formatApiError } from '../../utils/errors';
import { presetEditorSchema, presetPayload, presetSummary, presetFamily } from '../../utils/presetEditor';
import { reusableTrainingPreset } from '../../utils/trainingPresets';
import { inactiveTrainingReason, trainingFamilyOptions } from '../../utils/trainingFamilies';
import { useWorkspaceText } from '../../utils/workspaceText';
import { useWorkspaceHeight } from '../../components/projects/useWorkspaceHeight';
import './presets.css';
import '../../styles/parameter-workspace.css';
import ParameterModeToggle from '../../components/ParameterModeToggle';
import ParameterSections from '../../components/ParameterSections';

interface Draft { name: string; description: string; config: Record<string, any>; originalName: string | null; builtin: boolean; }
const KEY = ['standalone-presets'];

export default function Presets() {
  const text = useWorkspaceText();
  const english = text('zh', 'en') === 'en';
  const queryClient = useQueryClient();
  const [params] = useSearchParams();
  const list = useQuery({ queryKey: KEY, queryFn: () => apiClient.get<Preset[]>('/presets', { silent: true }) });
  const schema = useQuery({ queryKey: ['training-schema'], queryFn: () => apiClient.get<any>('/schema/train', { silent: true }) });
  const families = useFamilies();
  const editorSchema = React.useMemo(() => schema.data ? presetEditorSchema(schema.data) : null, [schema.data]);
  const [draft, setDraft] = React.useState<Draft | null>(null);
  const inactiveReason = inactiveTrainingReason(draft?.config, english);
  const [saved, setSaved] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');
  const [notice, setNotice] = React.useState('');
  const [errors, setErrors] = React.useState<ValidationError[]>([]);
  const [search, setSearch] = React.useState('');
  const searchRef = React.useRef<HTMLInputElement>(null);
  const parameterScrollRef = React.useRef<HTMLDivElement>(null);
  const nameRef = React.useRef<HTMLInputElement>(null);
  const descriptionRef = React.useRef<HTMLInputElement>(null);
  const [descriptionOpen, setDescriptionOpen] = React.useState(false);
  const [revealVersion, setRevealVersion] = React.useState(0);
  const toolbarRef = useWorkspaceHeight('--presets-toolbar-height');
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
      const retired = inactiveTrainingReason(preset?.config, english);
      if (retired && (copy || newDraft)) { setError(retired); return; }
      const defaults = retired ? {} : await apiClient.get<Record<string, any>>('/config/defaults', { params: { family }, silent: true });
      let name = preset?.name || '';
      if (copy) {
        const base = `${name}-copy`; name = base;
        for (let i = 2; list.data?.some(item => item.name.toLocaleLowerCase() === name.toLocaleLowerCase()); i += 1) name = `${base}-${i}`;
      }
      const existing = preset && !copy && !newDraft;
      const merged = mergeConfig(defaults, preset?.config || {});
      const next: Draft = { name, description: preset?.description || '', config: retired ? merged : reusableTrainingPreset(merged), originalName: existing ? preset.name : null, builtin: false };
      next.config.model = { ...next.config.model, family };
      startingConfig.current = JSON.stringify(next.config);
      setDraft(next); setSaved(existing ? JSON.stringify(presetPayload(next)) : JSON.stringify(presetPayload({...next,name:'',description:''})));
      setErrors([]); setSearch(''); setTab('train'); setDescriptionOpen(false);
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
    if (!draft || draft.builtin || busy || inactiveReason) return false;
    const body = presetPayload(draft);
    if (!/^[\p{L}\p{N}_-]{1,128}$/u.test(body.name) || !/[\p{L}\p{N}]/u.test(body.name)) {
      setError(text('预设名称须为 1–128 个字母、数字、中文、短横线或下划线。', 'Use 1–128 letters or numbers, with hyphens or underscores, for the preset name.'));
      nameRef.current?.focus();
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
  const familyName = (name: string) => name === 'flux' ? 'FLUX.1 · ' + text('已停用', 'Retired') : families.data?.find(item => item.name === name)?.label || (name || text('通用', 'General'));
  const family = families.data?.find(item => item.name === draft?.config.model?.family);
  const tabs: [ConfigTab, string][] = [['train', text('训练参数', 'Training')], ['data', text('数据与标签', 'Data and captions')], ['model', text('精度与保存', 'Precision and saving')], ['advanced', text('采样与高级', 'Sampling and advanced')]];
  const clearSearch = () => { setSearch(''); searchRef.current?.focus(); };
  const issues = presentConfigIssues(errors, english);
  const goToIssue = (issue: ConfigIssue) => {
    setTab(issue.tab); setSearch(''); setAdvanced(true); setRevealVersion(version => version + 1);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      let path = issue.path;
      let field = document.getElementById(`field-${path}`);
      while (!field && path.includes('.')) { path = path.slice(0, path.lastIndexOf('.')); field = document.getElementById(`field-${path}`); }
      field?.scrollIntoView?.({ block: 'center', behavior: window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
      const control = field?.querySelector<HTMLElement>('input:not(:disabled):not([readonly]):not([type="hidden"]),textarea:not(:disabled),[role="combobox"]:not(:disabled)');
      control?.focus({ preventScroll: true });
    }));
  };

  const options = [
    ...(!draft?.originalName ? [{value:'',label:text('新预设', 'New preset'),disabled:true}] : []),
    ...userPresets.map(item=>({value:item.name,label:`${item.name} · ${familyName(presetFamily(item.config))}`})),
  ];
  return <section className="presets-page parameter-workspace">
    <div className="parameter-workspace-header">
    <header className="presets-page-heading"><div><h1>{text('参数预设', 'Presets')}</h1><p>{text('保存常用训练参数，在项目版本中预览后应用。', 'Save reusable training parameters, then preview and apply them in a project version.')}</p></div><span>{text(`${userPresets.length} 个预设`, `${userPresets.length} presets`)}</span></header>
    <div className="presets-toolbar" ref={toolbarRef} role="group" aria-label={text('预设操作', 'Preset actions')}>
      <div className="presets-switcher"><span>{text('当前预设', 'Current preset')}</span><StudioSelect searchable aria-label={text('选择预设', 'Choose preset')} value={draft?.originalName || ''} disabled={busy || !userPresets.length} options={options} onValueChange={name=>{const item=userPresets.find(row=>row.name===name);if(item && name!==draft?.originalName)requestAction(()=>void begin(item));}}/></div>
      <div className="presets-actions">
        <button className="studio-secondary" disabled={busy || !schema.data || !families.data || list.isPending || list.isError} onClick={() => requestAction(() => void begin())}><Plus size={15}/>{text('新建预设', 'New preset')}</button>
        {draft?.originalName && <><button className="studio-secondary" aria-label={text('复制为新预设', 'Duplicate')} disabled={busy || !!inactiveReason} onClick={() => requestAction(() => void begin({name:draft.name,description:draft.description,config:draft.config,builtin:false,updated_at:null},true))}><Copy size={15}/>{text('复制', 'Duplicate')}</button><button className="studio-secondary presets-delete" aria-label={text('删除预设', 'Delete preset')} disabled={busy} onClick={()=>setDeleting(true)}><Trash2 size={15}/>{text('删除', 'Delete')}</button></>}
        <button className="studio-primary" disabled={busy || !!inactiveReason || !dirty || !editorSchema} onClick={()=>void save()}><Save size={15}/>{busy ? text('保存中…', 'Saving…') : text('保存预设', 'Save preset')}</button>
      </div>
    </div>
    {error && !deleting && <div role="alert" className="studio-error presets-validation-error"><p>{error}</p>{issues.length > 0 && <div className="presets-error-links">{issues.map((issue, index) => <button type="button" key={`${issue.path}-${index}`} onClick={() => goToIssue(issue)}>{text('定位', 'Go to')} {issue.label}<ChevronRight size={14}/></button>)}</div>}</div>}
    {notice && <p role="status" className="presets-notice">{notice}</p>}
    {inactiveReason && <p role="alert" className="studio-error" data-testid="retired-preset">{inactiveReason}</p>}
    {[list, schema, families].some(query => query.isError) && <div role="alert" className="studio-error"><span>{[list, schema, families].filter(query => query.error).map(query => formatApiError(query.error)).join('\n')}</span><button onClick={() => { for (const query of [list, schema, families]) if (query.isError) void query.refetch(); }}>{text('重新加载', 'Reload')}</button></div>}
    {!draft && !error && <p role="status" className="presets-loading">{text('正在读取参数…', 'Loading parameters…')}</p>}
    </div>
    {draft && <div className="presets-editor">
      <div className="presets-form-toolbar">
        <div role="tablist" aria-label={text('预设参数分区', 'Preset parameter sections')}>{tabs.map(([key,label], index)=><button key={key} id={`preset-tab-${key}`} role="tab" aria-label={label} tabIndex={tab === key ? 0 : -1} aria-selected={!search && tab===key} aria-controls="preset-parameters" onClick={()=>{setTab(key);setSearch('');}} onKeyDown={event => {
          const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1;
          if (next >= 0) { event.preventDefault(); setTab(tabs[next][0]); setSearch(''); document.getElementById(`preset-tab-${tabs[next][0]}`)?.focus(); }
        }}>{label}{issues.some(issue => issue.tab === key) && <span className="tab-issue-dot" aria-label={text('有待配置项', 'Needs configuration')}/>}</button>)}</div>
        <label className="presets-field-search"><Search size={16}/><input ref={searchRef} aria-label={text('搜索预设参数', 'Search preset parameters')} placeholder={text('搜索所有分区的参数', 'Search all parameter sections')} value={search} onChange={event=>setSearch(event.target.value)}/>{search && <button type="button" aria-label={text('清空预设参数搜索', 'Clear preset parameter search')} onClick={clearSearch}><X size={15}/></button>}</label>
        <ParameterModeToggle advanced={advanced} onChange={setAdvanced}/>
      </div>
      <div className="parameter-workspace-body">
      <ParameterSections rootRef={parameterScrollRef} tab={tab} onTabChange={tab => {setTab(tab);setSearch('');}} issues={issues} preset hasTrainingMode={!!editorSchema?.properties?.training} fullTraining={draft.config.training?.mode === 'full'} onRevealAdvanced={() => setAdvanced(true)}/>
      <div className="parameter-scroll-region" ref={parameterScrollRef}>
      <fieldset disabled={busy || !!inactiveReason} className="presets-meta">
        <label className="presets-family">{text('适用模型', 'Model family')}{inactiveReason ? <span>{draft.config.model.family === 'flux' ? 'FLUX.1' : 'FLUX.2 dev'} · {text('已停用', 'Retired')}</span> : <StudioSelect aria-label={text('适用模型', 'Model family')} disabled={busy || !!draft.originalName} value={draft.config.model.family} onValueChange={target=>{const change=()=>{void begin({name:draft.name,description:draft.description,config:{model:{family:target}},builtin:false,updated_at:null},false,target,true);};if(JSON.stringify(draft.config)!==startingConfig.current)requestAction(change);else change();}} options={trainingFamilyOptions(families.data || [], english, draft.config.model.family)}/>}</label>
        <label>{text('预设名称', 'Preset name')}<input ref={nameRef} aria-label={text('预设名称', 'Preset name')} placeholder={text('例如：人物_LoKr', 'For example: character_LoKr')} value={draft.name} readOnly={!!draft.originalName} onChange={event=>setDraft({...draft,name:event.target.value})}/></label>
        <button type="button" className="presets-description-toggle" aria-label={text('编辑用途与说明', 'Edit description')} aria-expanded={descriptionOpen} aria-controls="preset-description" onClick={() => {
          setDescriptionOpen(open => !open);
          if (!descriptionOpen) requestAnimationFrame(() => descriptionRef.current?.focus());
        }}>{text('说明', 'Description')}<ChevronDown size={14}/></button>
        {descriptionOpen && <label id="preset-description" className="presets-description">{text('用途与说明', 'Description')}<input ref={descriptionRef} aria-label={text('用途与说明', 'Description')} placeholder={text('可选，记录用途或参数取舍', 'Optional: purpose or parameter choices')} value={draft.description} onChange={event=>setDraft({...draft,description:event.target.value})}/></label>}
      </fieldset>
      <div className="presets-context"><span>{!draft.originalName ? text('填写名称并编辑参数后保存。', 'Name and edit the preset, then save.') : presetSummary(draft.config, english)}</span><span className={dirty ? 'presets-dirty' : ''}>{dirty ? text('有未保存修改', 'Unsaved changes') : draft.originalName ? text('已保存', 'Saved') : text('尚未创建', 'Not created yet')}</span></div>
      {search && <p className="presets-search-context">{text('搜索所有分区，包含高级参数', 'Searching every section, including advanced parameters')}</p>}
      {editorSchema && <div id="preset-parameters" className="presets-schema" role="tabpanel" aria-labelledby={search ? undefined : `preset-tab-${tab}`} aria-label={search ? text('预设参数搜索结果', 'Preset parameter search results') : undefined}><SchemaForm key={revealVersion} readOnly={busy || !!inactiveReason} schema={editorSchema} value={draft.config} onChange={config=>{setDraft({...draft,config});setErrors([]);}} compact showAdvanced={advanced || !!search} groupFilter={search ? undefined : CONFIG_TAB_GROUPS[tab]} search={search} onClearSearch={clearSearch} family={family} families={families.data} errors={errors}/></div>}
      </div>
      </div>
    </div>}
    {(pending || blocker.state === 'blocked') && <Dialog title={text('保存预设修改？', 'Save preset changes?')} onClose={cancelLeave} closeDisabled={busy}><p>{text('当前预设尚未保存。可以先保存，或放弃这些修改。', 'This preset has unsaved changes. Save them or discard the draft.')}</p>{error && <p role="alert" className="studio-error">{error}</p>}<div className="presets-confirm-actions"><button className="studio-secondary" disabled={busy} onClick={cancelLeave}>{text('继续编辑', 'Keep editing')}</button><button className="studio-secondary" disabled={busy} onClick={proceed}>{text('放弃修改', 'Discard changes')}</button><button className="studio-primary" disabled={busy} onClick={()=>{void save().then(ok=>{if(ok)proceed();});}}>{text('保存并继续', 'Save and continue')}</button></div></Dialog>}
    {deleting && <Dialog title={text('删除预设', 'Delete preset')} onClose={()=>setDeleting(false)} closeDisabled={busy}><p>{text('删除后不会改变任何已有项目配置。确定删除', 'Existing project configurations will remain unchanged. Delete')} “{draft?.name}”?</p>{error&&<p role="alert" className="studio-error">{error}</p>}<div className="presets-confirm-actions"><button className="studio-secondary" disabled={busy} onClick={()=>setDeleting(false)}>{text('取消', 'Cancel')}</button><button className="studio-primary" disabled={busy} onClick={()=>void remove()}>{text('确认删除', 'Confirm deletion')}</button></div></Dialog>}
  </section>;
}
