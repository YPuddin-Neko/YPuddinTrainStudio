import React from 'react';
import { useBlocker, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Copy, Plus, Save, Search, Trash2, X, ChevronRight, ChevronDown, Upload, Download } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Preset } from '../../api/types';
import { useFamilies } from '../../api/hooks/useFamilies';
import Dialog from '../../components/Dialog';
import StudioSelect from '../../components/StudioSelect';
import { SchemaForm, type ValidationError } from '../../schema/SchemaForm/SchemaForm';
import { OPAQUE_CONFIG_ISSUE, presentConfigIssues, type ConfigTab, type ConfigIssue } from '../../utils/configPresentation';
import { mergeConfig } from '../../utils/config';
import { formatApiError } from '../../utils/errors';
import { presetEditorSchema, presetPayload, presetFamily } from '../../utils/presetEditor';
import { PRESET_MODEL_FIELDS, reusableTrainingPreset } from '../../utils/trainingPresets';
import { inactiveTrainingReason, trainingFamilyOptions } from '../../utils/trainingFamilies';
import { useWorkspaceText } from '../../utils/workspaceText';
import { useWorkspaceHeight } from '../../components/projects/useWorkspaceHeight';
import './presets.css';
import '../../styles/parameter-workspace.css';
import ParameterModeToggle from '../../components/ParameterModeToggle';
import ParameterSections from '../../components/ParameterSections';
import { workflowSchema } from '../../utils/parameterWorkflow';
import { LoadingNote } from '../../components/Loading';
import PageLocation from '../../components/PageLocation';
import PresetImportDialog, { type ImportedPreset } from './PresetImportDialog';

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
  const editorSchema = React.useMemo(() => schema.data ? workflowSchema(presetEditorSchema(schema.data)) : null, [schema.data]);
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
  const [importing, setImporting] = React.useState(false);
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
      if (retired && (copy || newDraft)) { setError(retired); return false; }
      const defaults = retired ? {} : await apiClient.get<Record<string, any>>('/config/defaults', { params: { family }, silent: true });
      let name = preset?.name || '';
      if (copy) {
        const base = `${name}-copy`; name = base;
        for (let i = 2; list.data?.some(item => item.name.toLocaleLowerCase() === name.toLocaleLowerCase()); i += 1) name = `${base}-${i}`;
      }
      const existing = preset && !copy && !newDraft;
      const merged = mergeConfig(defaults, preset?.config || {});
      // Model files come only from the preset itself, never from family defaults.
      const presetModel = (preset?.config?.model || {}) as Record<string, unknown>;
      merged.model = { ...merged.model, ...Object.fromEntries(PRESET_MODEL_FIELDS.map(field => [field, presetModel[field] || null])) };
      const next: Draft = { name, description: preset?.description || '', config: retired ? merged : reusableTrainingPreset(merged), originalName: existing ? preset.name : null, builtin: false };
      next.config.model = { ...next.config.model, family };
      startingConfig.current = JSON.stringify(next.config);
      setDraft(next); setSaved(existing ? JSON.stringify(presetPayload(next)) : JSON.stringify(presetPayload({...next,name:'',description:''})));
      setErrors([]); setSearch(''); setTab('train'); setDescriptionOpen(false);
      return true;
    } catch (failure) { setError(formatApiError(failure)); return false; }
    finally { setBusy(false); }
  };
  const importPreset = async (preset: ImportedPreset) => {
    const retired = inactiveTrainingReason(preset.config, english);
    if (retired) throw new Error(retired);
    const base = preset.name.trim().replace(/[^\p{L}\p{N}_-]+/gu, '-').slice(0, 116) || 'imported-preset';
    let name = /[\p{L}\p{N}]/u.test(base) ? base : 'imported-preset';
    const stem = name;
    for (let i = 2; list.data?.some(item => item.name.toLocaleLowerCase() === name.toLocaleLowerCase()); i += 1) name = `${stem}-${i}`;
    const opened = await begin({ ...preset, name, builtin: false, updated_at: null }, false, presetFamily(preset.config) || 'anima', true);
    if (!opened) throw new Error(text('无法载入预设，请重试。', 'Could not load the preset. Please try again.'));
    return true;
  };
  const exportPreset = () => {
    if (!draft) return;
    const body = presetPayload(draft);
    const filename = body.name.replace(/[^\p{L}\p{N}_-]+/gu, '-').slice(0, 128) || 'preset';
    const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2) + '\n'], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `${filename}.json`;
    document.body.append(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
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
      setNotice(text('预设已保存。', 'Preset saved.'));
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

  const options = userPresets.map(item=>({value:item.name,label:`${item.name} · ${familyName(presetFamily(item.config))}`}));
  return <section className="presets-page parameter-workspace">
    <div className="parameter-workspace-header">
    <PageLocation trail={[{ label: text('参数预设', 'Presets') }]}/><header className="presets-page-heading"><div><h1>{text('参数预设', 'Presets')}</h1></div><span>{text(`${userPresets.length} 个预设`, `${userPresets.length} presets`)}</span></header>
    <div className="presets-toolbar" ref={toolbarRef} role="group" aria-label={text('预设操作', 'Preset actions')}>
      <div className="presets-actions">
        <span role="status" className={`presets-status${draft && dirty ? ' presets-dirty' : ''}`}>{!draft ? notice : dirty ? text('有未保存修改', 'Unsaved changes') : notice || (draft.originalName ? text('已保存', 'Saved') : text('尚未创建', 'Not created yet'))}</span>
        <button type="button" className="ui-btn" disabled={busy || !schema.data || !families.data || list.isPending || list.isError} onClick={() => requestAction(() => void begin())}><Plus size={15}/>{text('新建预设', 'New preset')}</button>
        <button type="button" className="ui-btn" disabled={busy || !schema.data || !families.data || list.isPending || list.isError} onClick={() => requestAction(() => setImporting(true))}><Upload size={15}/>{text('导入', 'Import')}</button>
        <button type="button" className="ui-btn" disabled={busy || !draft} onClick={exportPreset}><Download size={15}/>{text('导出', 'Export')}</button>
        {draft?.originalName && <><button type="button" className="ui-btn" aria-label={text('复制为新预设', 'Duplicate')} disabled={busy || !!inactiveReason} onClick={() => requestAction(() => void begin({name:draft.name,description:draft.description,config:draft.config,builtin:false,updated_at:null},true))}><Copy size={15}/>{text('复制', 'Duplicate')}</button><button type="button" className="ui-btn ui-btn-danger presets-delete" aria-label={text('删除预设', 'Delete preset')} disabled={busy} onClick={()=>setDeleting(true)}><Trash2 size={15}/>{text('删除', 'Delete')}</button></>}
        <button type="button" className="ui-btn ui-btn-primary" disabled={busy || !!inactiveReason || !dirty || !editorSchema} onClick={()=>void save()}><Save size={15}/>{busy ? text('保存中…', 'Saving…') : text('保存预设', 'Save preset')}</button>
      </div>
    </div>
    {/* Which preset, and its identity: the switcher first, then the model, name and description it is saved with. */}
    <div className="presets-identity">
      <div className="presets-switcher"><span>{text('当前预设', 'Current preset')}</span><StudioSelect searchable aria-label={text('选择预设', 'Choose preset')} value={draft?.originalName || ''} disabled={busy || !userPresets.length} placeholder={text('新预设', 'New preset')} options={options} onValueChange={name=>{const item=userPresets.find(row=>row.name===name);if(item && name!==draft?.originalName)requestAction(()=>void begin(item));}}/></div>
      {draft && <>
        <fieldset disabled={busy || !!inactiveReason} className="presets-meta">
          <label className="presets-family">{text('适用模型', 'Model family')}{inactiveReason ? <span>{draft.config.model.family === 'flux' ? 'FLUX.1' : 'FLUX.2 dev'} · {text('已停用', 'Retired')}</span> : <StudioSelect aria-label={text('适用模型', 'Model family')} disabled={busy || !!draft.originalName} value={draft.config.model.family} onValueChange={target=>{const change=()=>{void begin({name:draft.name,description:draft.description,config:{model:{family:target}},builtin:false,updated_at:null},false,target,true);};if(JSON.stringify(draft.config)!==startingConfig.current)requestAction(change);else change();}} options={trainingFamilyOptions(families.data || [], english, draft.config.model.family)}/>}</label>
          <label>{text('预设名称', 'Preset name')}<input ref={nameRef} aria-label={text('预设名称', 'Preset name')} placeholder={text('例如：人物_LoKr', 'For example: character_LoKr')} value={draft.name} readOnly={!!draft.originalName} onChange={event=>setDraft({...draft,name:event.target.value})}/></label>
          <button type="button" className="ui-btn ui-btn-quiet presets-description-toggle" aria-label={text('编辑用途与说明', 'Edit description')} aria-expanded={descriptionOpen} aria-controls="preset-description" onClick={() => {
            setDescriptionOpen(open => !open);
            if (!descriptionOpen) requestAnimationFrame(() => descriptionRef.current?.focus());
          }}>{text('说明', 'Description')}<ChevronDown size={14}/></button>
        </fieldset>
        {descriptionOpen && <label id="preset-description" className="presets-description">{text('用途与说明', 'Description')}<input ref={descriptionRef} aria-label={text('用途与说明', 'Description')} disabled={busy || !!inactiveReason} placeholder={text('可选，记录用途或参数取舍', 'Optional: purpose or parameter choices')} value={draft.description} onChange={event=>setDraft({...draft,description:event.target.value})}/></label>}
      </>}
    </div>
    {error && !deleting && <div role="alert" className="studio-error presets-validation-error"><p>{error}</p>{issues.length > 0 && <div className="presets-error-links">{issues.map((issue, index) => <button type="button" className="ui-btn ui-btn-sm" key={`${issue.path}-${index}`} onClick={() => goToIssue(issue)}>{text('定位', 'Go to')} {issue.label}<ChevronRight size={14}/></button>)}</div>}</div>}
    {inactiveReason && <p role="alert" className="studio-error" data-testid="retired-preset">{inactiveReason}</p>}
    {[list, schema, families].some(query => query.isError) && <div role="alert" className="studio-error"><span>{[list, schema, families].filter(query => query.error).map(query => formatApiError(query.error)).join('\n')}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => { for (const query of [list, schema, families]) if (query.isError) void query.refetch(); }}>{text('重新加载', 'Reload')}</button></div>}
    {!draft && !error && <LoadingNote block className="presets-loading" label={text('正在读取参数…', 'Loading parameters…')}/>}
    </div>
    {draft && <div className="presets-editor">
      <div className="presets-form-toolbar">
        <label className="presets-field-search"><Search size={16}/><input ref={searchRef} aria-label={text('搜索预设参数', 'Search preset parameters')} placeholder={text('搜索所有分区的参数', 'Search all parameter sections')} value={search} onChange={event=>setSearch(event.target.value)}/>{search && <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={text('清空预设参数搜索', 'Clear preset parameter search')} onClick={clearSearch}><X size={15}/></button>}</label>
        <ParameterModeToggle advanced={advanced} onChange={setAdvanced}/>
      </div>
      <div className="parameter-workspace-body">
      <ParameterSections rootRef={parameterScrollRef} tab={tab} onTabChange={tab => {setTab(tab);setSearch('');}} issues={issues} preset hasTrainingMode={!!editorSchema?.properties?.training} fullTraining={draft.config.training?.mode === 'full'} onRevealAdvanced={() => setAdvanced(true)}/>
      <div className="parameter-scroll-region" ref={parameterScrollRef}>
      {search && <p className="presets-search-context">{text('搜索所有分区，包含高级参数', 'Searching every section, including advanced parameters')}</p>}
      {editorSchema && <div id="preset-parameters" className="presets-schema" role="region" aria-label={search ? text('预设参数搜索结果', 'Preset parameter search results') : text('预设参数内容', 'Preset parameter fields')}><SchemaForm key={revealVersion} preset readOnly={busy || !!inactiveReason} schema={editorSchema} value={draft.config} onChange={config=>{setDraft({...draft,config});setErrors([]);}} compact showAdvanced={advanced || !!search} search={search} onClearSearch={clearSearch} family={family} families={families.data} errors={issues.map(issue => ({ loc: issue.path, msg: issue.message === OPAQUE_CONFIG_ISSUE ? '' : issue.message }))}/></div>}
      </div>
      </div>
    </div>}
    {(pending || blocker.state === 'blocked') && <Dialog title={text('保存预设修改？', 'Save preset changes?')} onClose={cancelLeave} closeDisabled={busy}><p>{text('当前预设尚未保存。可以先保存，或放弃这些修改。', 'This preset has unsaved changes. Save them or discard the draft.')}</p>{error && <p role="alert" className="studio-error">{error}</p>}<div className="presets-confirm-actions"><button type="button" className="ui-btn" disabled={busy} onClick={cancelLeave}>{text('继续编辑', 'Keep editing')}</button><button type="button" className="ui-btn ui-btn-danger" disabled={busy} onClick={proceed}>{text('放弃修改', 'Discard changes')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={busy} onClick={()=>{void save().then(ok=>{if(ok)proceed();});}}>{text('保存并继续', 'Save and continue')}</button></div></Dialog>}
    {deleting && <Dialog title={text('删除预设', 'Delete preset')} onClose={()=>setDeleting(false)} closeDisabled={busy}><p>{text('删除后不会改变任何已有项目配置。确定删除', 'Existing project configurations will remain unchanged. Delete')} “{draft?.name}”?</p>{error&&<p role="alert" className="studio-error">{error}</p>}<div className="presets-confirm-actions"><button type="button" className="ui-btn" disabled={busy} onClick={()=>setDeleting(false)}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-primary ui-btn-danger" disabled={busy} onClick={()=>void remove()}>{text('确认删除', 'Confirm deletion')}</button></div></Dialog>}
    {importing && <PresetImportDialog onClose={() => setImporting(false)} onImport={importPreset}/>}
  </section>;
}
