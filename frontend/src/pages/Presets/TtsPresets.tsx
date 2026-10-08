import React from 'react';
import { useBlocker, useSearchParams, type Location } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ChevronDown, Copy, Download, Plus, Save, Search, Trash2, Upload, X } from 'lucide-react';
import { ttsApi, type TtsEngine, type TtsIssue } from '../../api/tts';
import { ttsPresetsApi, type TtsPreset, type TtsPresetDocument } from '../../api/ttsPresets';
import { ApiError } from '../../api/types';
import Dialog from '../../components/Dialog';
import StudioSelect from '../../components/StudioSelect';
import ParameterModeToggle from '../../components/ParameterModeToggle';
import PageLocation from '../../components/PageLocation';
import { LoadingNote } from '../../components/Loading';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import PresetTypeNavigation from './PresetTypeNavigation';
import TtsPresetConfigEditor, { type TtsPresetConfigHandle } from './TtsPresetConfigEditor';
import { comparablePresetParameters, downloadTtsPreset } from '../Tts/ttsPresetParameters';
import './presets.css';
import './tts-presets.css';

interface Draft { id: string | null; revision: number; name: string; description: string; config: Record<string, unknown>; base: TtsPreset | null }
const snapshot = (value: Pick<Draft, 'name' | 'description' | 'config'>) => JSON.stringify({ name: value.name, description: value.description, config: comparablePresetParameters(value.config) });
const fromPreset = (value: TtsPreset): Draft => ({ id: value.id, revision: value.revision, name: value.name, description: value.description, config: structuredClone(value.config), base: value });
function mergeChanges(base: unknown, local: unknown, remote: unknown): unknown {
  if (JSON.stringify(base) === JSON.stringify(local)) return structuredClone(remote);
  if (base && local && remote && typeof base === 'object' && typeof local === 'object' && typeof remote === 'object' && !Array.isArray(local)) {
    const before = base as Record<string, unknown>, mine = local as Record<string, unknown>, theirs = remote as Record<string, unknown>;
    return Object.fromEntries(Object.keys(mine).map(key => [key, mergeChanges(before[key], mine[key], theirs[key])]));
  }
  return structuredClone(local);
}
export default function TtsPresets({ engine }: { engine: TtsEngine }) {
  const text = useWorkspaceText();
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const list = useQuery({ queryKey: ['tts-presets', engine], queryFn: ({ signal }) => ttsPresetsApi.list(engine, signal) });
  const defaults = useQuery({ queryKey: ['tts-preset-defaults', engine], queryFn: ({ signal }) => ttsPresetsApi.defaults(engine, signal) });
  const schema = useQuery({ queryKey: ['tts-training-schema', engine], queryFn: ({ signal }) => ttsApi.trainSchema(signal, engine) });
  const [draft, setDraft] = React.useState<Draft | null>(null), [saved, setSaved] = React.useState('');
  const [busy, setBusy] = React.useState(false), pendingRequest = React.useRef(false), mounted = React.useRef(true);
  const [error, setError] = React.useState(''), [notice, setNotice] = React.useState('');
  const [search, setSearch] = React.useState('');
  const [advanced, setAdvanced] = React.useState(false);
  const [descriptionOpen, setDescriptionOpen] = React.useState(false);
  const descriptionInput = React.useRef<HTMLInputElement>(null);
  const [pending, setPending] = React.useState<(() => void) | null>(null);
  const [deleting, setDeleting] = React.useState(false), [importing, setImporting] = React.useState(false);
  const [importText, setImportText] = React.useState('');
  const [readingFile, setReadingFile] = React.useState(false);
  const [conflict, setConflict] = React.useState<TtsPreset | null>(null);
  const editor = React.useRef<TtsPresetConfigHandle>(null), nameInput = React.useRef<HTMLInputElement>(null);
  const fileInput = React.useRef<HTMLInputElement>(null), fileRead = React.useRef(0);
  const dirty = !!draft && snapshot(draft) !== saved;
  const bypass = React.useRef(false);
  const workspace = (location: Location) => {
    const background = (location.state as { backgroundLocation?: Location } | null)?.backgroundLocation;
    const target = location.pathname.startsWith('/settings') && background ? background : location;
    return `${target.pathname}${target.search}${target.hash}`;
  };
  const blocker = useBlocker(({ currentLocation, nextLocation }) => (dirty || pendingRequest.current) && !bypass.current && workspace(currentLocation) !== workspace(nextLocation));
  React.useEffect(() => { mounted.current = true; return () => { mounted.current = false; fileRead.current += 1; }; }, []);
  React.useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', warn); return () => window.removeEventListener('beforeunload', warn);
  }, [dirty]);
  const open = (item?: TtsPreset) => {
    if (!item && !defaults.data) return;
    const next = item ? fromPreset(item) : { id: null, revision: 0, name: '', description: '', config: structuredClone(defaults.data!), base: null };
    setDraft(next); setSaved(snapshot(next)); setError(''); setNotice(''); setSearch(''); setConflict(null); setDescriptionOpen(false);
  };
  const initialized = React.useRef(false);
  React.useEffect(() => {
    if (initialized.current || !list.data || !defaults.data) return;
    initialized.current = true;
    const id = params.get('id'); const selected = id ? list.data.find(item => item.id === id) : list.data[0];
    open(selected);
    if (id && !selected) setError(text('找不到指定预设，请选择已有预设或新建。', 'The requested preset was not found. Choose another or create one.'));
    // Background list refreshes must not replace the current draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [list.data, defaults.data]);
  const requestAction = (action: () => void) => { if (pendingRequest.current || conflict) return; if (dirty) setPending(() => action); else action(); };
  const navigateType = (type: 'image' | 'tts', model = engine) => {
    if (type === 'tts' && model === engine) return;
    requestAction(() => { bypass.current = true; setParams(type === 'tts' ? { type, engine: model } : {}); });
  };
  const record = (item: TtsPreset) => {
    client.setQueryData<TtsPreset[]>(['tts-presets', engine], previous => [item, ...(previous || []).filter(row => row.id !== item.id)]);
    void client.invalidateQueries({ queryKey: ['tts-presets', engine] });
  };
  const failed = async (failure: unknown) => {
    if (!mounted.current) return;
    setError(formatApiError(failure));
    const issues = failure instanceof ApiError && Array.isArray(failure.details?.issues) ? failure.details.issues as TtsIssue[] : [];
    editor.current?.showIssues(issues);
    if (failure instanceof ApiError && failure.code === 'tts.preset_conflict' && draft?.id) {
      try { const remote = await ttsPresetsApi.get(draft.id); if (mounted.current) { setConflict(remote); setDeleting(false); } }
      catch (readFailure) { if (mounted.current) setError(formatApiError(readFailure)); }
    }
  };
  const save = async (): Promise<boolean> => {
    if (!draft || pendingRequest.current || conflict) return false;
    const config = editor.current?.getConfig(); if (!config) return false;
    if (!draft.name.trim()) { setError(text('请填写预设名称。', 'Enter a preset name.')); nameInput.current?.focus(); return false; }
    pendingRequest.current = true; setBusy(true); setError(''); setNotice('');
    try {
      const body = { name: draft.name.trim(), description: draft.description, config };
      const item = draft.id ? await ttsPresetsApi.update(draft.id, { ...body, expected_revision: draft.revision }) : await ttsPresetsApi.create(body);
      if (!mounted.current) return false;
      record(item); open(item); setNotice(text('预设已保存。', 'Preset saved.')); return true;
    } catch (failure) { await failed(failure); return false; }
    finally { pendingRequest.current = false; if (mounted.current) setBusy(false); }
  };
  const remove = async () => {
    if (!draft?.id || pendingRequest.current || conflict) return;
    pendingRequest.current = true; setBusy(true); setError('');
    try {
      await ttsPresetsApi.remove(draft.id, draft.revision);
      if (!mounted.current) return;
      const remaining = (list.data || []).filter(item => item.id !== draft.id);
      client.setQueryData(['tts-presets', engine], remaining); void client.invalidateQueries({ queryKey: ['tts-presets', engine] });
      setDeleting(false); open(remaining[0]); setNotice(text('预设已删除。', 'Preset deleted.'));
    } catch (failure) { await failed(failure); }
    finally { pendingRequest.current = false; if (mounted.current) setBusy(false); }
  };
  const duplicate = () => {
    if (!draft || pendingRequest.current || conflict) return;
    const stem = `${draft.name || 'preset'}-copy`; let name = stem;
    for (let i = 2; list.data?.some(item => item.name.toLocaleLowerCase() === name.toLocaleLowerCase()); i += 1) name = `${stem}-${i}`;
    setDraft({ ...draft, id: null, revision: 0, name, base: null }); setSaved(''); setNotice(''); setError('');
    requestAnimationFrame(() => nameInput.current?.focus());
  };
  const exportPreset = async () => {
    if (!draft?.id || dirty || pendingRequest.current) return;
    pendingRequest.current = true; setBusy(true); setError('');
    try { const document = await ttsPresetsApi.export(draft.id); if (mounted.current) downloadTtsPreset(document); }
    catch (failure) { await failed(failure); }
    finally { pendingRequest.current = false; if (mounted.current) setBusy(false); }
  };
  const importPreset = async () => {
    if (pendingRequest.current) return;
    let document: TtsPresetDocument;
    try {
      const value = JSON.parse(importText);
      if (value?.format !== 'ypuddin-tts-preset' || value.schema_version !== 1 || !value.config || typeof value.name !== 'string') throw new Error(text('请选择有效的语音参数预设 JSON 文件。', 'Choose a valid speech preset JSON file.'));
      if (value.config.engine !== engine) throw new Error(text('预设模型类型与当前选择不同，请先切换模型类型。', 'The preset model differs from the selected model. Switch the model type first.'));
      document = value as TtsPresetDocument;
    } catch (failure) { setError(formatApiError(failure)); return; }
    pendingRequest.current = true; setBusy(true); setError('');
    try { const item = await ttsPresetsApi.import(document); if (mounted.current) { record(item); open(item); setImporting(false); setNotice(text('预设已导入。', 'Preset imported.')); } }
    catch (failure) { await failed(failure); }
    finally { pendingRequest.current = false; if (mounted.current) setBusy(false); }
  };
  const cancelLeave = () => { setPending(null); if (blocker.state === 'blocked') blocker.reset(); };
  const proceed = () => { if (blocker.state === 'blocked') blocker.proceed(); else { const action = pending; setPending(null); action?.(); } };
  const merge = () => {
    if (!conflict || !draft) return;
    const remote = fromPreset(conflict);
    const next = draft.base ? { ...remote, ...mergeChanges({ name: draft.base.name, description: draft.base.description, config: draft.base.config }, { name: draft.name, description: draft.description, config: comparablePresetParameters(draft.config) }, { name: remote.name, description: remote.description, config: remote.config }) as Pick<Draft, 'name' | 'description' | 'config'> } : { ...remote, name: draft.name, description: draft.description, config: draft.config };
    setDraft(next); setSaved(snapshot(remote)); setConflict(null); setError(''); setNotice(text('已合并我的修改，请检查后保存。', 'Your changes are merged. Review and save them.'));
  };
  const unavailable = [list, defaults, schema].some(query => query.isPending || query.isError);
  return <section className="presets-page parameter-workspace tts-presets-page">
    <div className="parameter-workspace-header">
      <PageLocation trail={[{ label: text('参数预设', 'Presets') }]}/>
      <header className="presets-page-heading"><div className="presets-title-navigation"><h1>{text('参数预设', 'Presets')}</h1><PresetTypeNavigation value="tts" disabled={busy} onChange={navigateType}/></div><span>{text(`${list.data?.length || 0} 个预设`, `${list.data?.length || 0} presets`)}</span></header>
      <div className="presets-toolbar" role="group" aria-label={text('预设操作', 'Preset actions')}><div className="presets-actions">
        <span role="status" className={`presets-status${dirty ? ' presets-dirty' : ''}`}>{notice || (dirty ? text('有未保存修改', 'Unsaved changes') : draft?.id ? text('已保存', 'Saved') : text('尚未创建', 'Not created yet'))}</span>
        <button type="button" className="ui-btn" disabled={busy || unavailable} onClick={() => requestAction(() => open())}><Plus size={15}/>{text('新建预设', 'New preset')}</button>
        <button type="button" className="ui-btn" disabled={busy || unavailable} onClick={() => requestAction(() => { setImportText(''); setError(''); setImporting(true); })}><Upload size={15}/>{text('导入', 'Import')}</button>
        <button type="button" className="ui-btn" disabled={busy || dirty || !draft?.id} title={dirty ? text('保存后导出', 'Save before exporting') : undefined} onClick={() => void exportPreset()}><Download size={15}/>{text('导出', 'Export')}</button>
        <button type="button" className="ui-btn" disabled={busy || !!conflict || !draft} onClick={duplicate}><Copy size={15}/>{text('复制', 'Duplicate')}</button>
        <button type="button" className="ui-btn ui-btn-danger" disabled={busy || !!conflict || !draft?.id} onClick={() => { setDeleting(true); setError(''); }}><Trash2 size={15}/>{text('删除', 'Delete')}</button>
        <button type="button" className="ui-btn ui-btn-primary" disabled={busy || unavailable || !dirty || !!conflict} onClick={() => void save()}><Save size={15}/>{busy ? text('保存中…', 'Saving…') : text('保存预设', 'Save preset')}</button>
      </div></div>
      <div className="presets-identity">
        <div className="presets-switcher"><span>{text('当前预设', 'Current preset')}</span><StudioSelect searchable aria-label={text('选择语音预设', 'Choose speech preset')} value={draft?.id || ''} disabled={busy || !list.data?.length} placeholder={text('新预设', 'New preset')} options={(list.data || []).map(item => ({ value: item.id, label: item.name }))} onValueChange={id => { const item = list.data?.find(row => row.id === id); if (item) requestAction(() => open(item)); }}/></div>
        <fieldset disabled={busy} className="presets-meta"><label className="presets-family">{text('适用模型', 'Model family')}<StudioSelect aria-label={text('预设模型类型', 'Preset model type')} value={engine} onValueChange={value => navigateType('tts', value as TtsEngine)} options={[{ value: 'voxcpm1.5', label: 'VoxCPM 1.5' }, { value: 'gpt-sovits-v5', label: 'GPT-SoVITS v5' }]}/></label>
          {draft && <label>{text('预设名称', 'Preset name')}<input ref={nameInput} aria-label={text('预设名称', 'Preset name')} value={draft.name} onChange={event => { setDraft({ ...draft, name: event.target.value }); setNotice(''); }}/></label>}
          <button type="button" className="ui-btn ui-btn-quiet presets-description-toggle" aria-label={text('编辑用途与说明', 'Edit description')} aria-expanded={descriptionOpen} aria-controls="tts-preset-description" onClick={() => { setDescriptionOpen(open => !open); if (!descriptionOpen) requestAnimationFrame(() => descriptionInput.current?.focus()); }}>{text('说明', 'Description')}<ChevronDown size={14}/></button>
        </fieldset>
        {draft && descriptionOpen && <label id="tts-preset-description" className="presets-description">{text('用途与说明', 'Description')}<input ref={descriptionInput} aria-label={text('用途与说明', 'Description')} disabled={busy} value={draft.description} onChange={event => { setDraft({ ...draft, description: event.target.value }); setNotice(''); }}/></label>}
      </div>
      {error && !deleting && !importing && !conflict && <p role="alert" className="studio-error">{error}</p>}
      {[list, defaults, schema].some(query => query.isError) && <p role="alert" className="studio-error">{[list, defaults, schema].filter(query => query.error).map(query => formatApiError(query.error)).join('\n')} <button type="button" className="ui-link" onClick={() => { for (const query of [list, defaults, schema]) if (query.isError) void query.refetch(); }}>{text('重新读取', 'Reload')}</button></p>}
      {unavailable && ![list, defaults, schema].some(query => query.isError) && <LoadingNote label={text('正在读取预设…', 'Loading presets…')}/>}
    </div>
    {draft && schema.data && <div className="presets-editor">
      <div className="presets-form-toolbar"><label className="presets-field-search"><Search size={16}/><input type="search" aria-label={text('搜索预设参数', 'Search preset parameters')} placeholder={text('搜索参数名称或关键字…', 'Search parameters…')} value={search} onChange={event => setSearch(event.target.value)}/>{search && <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('清空预设参数搜索', 'Clear preset parameter search')} onClick={() => setSearch('')}><X size={15}/></button>}</label><ParameterModeToggle advanced={advanced} onChange={setAdvanced}/></div>
      <TtsPresetConfigEditor identity={draft.id || 'new'} revision={draft.revision} advanced={advanced} onAdvancedChange={setAdvanced} ref={editor} value={draft.config} schema={schema.data} readOnly={busy || !!conflict} search={search} onSearch={setSearch} onChange={config => { setDraft({ ...draft, config }); setNotice(''); }}/>
    </div>}
    {(pending || blocker.state === 'blocked') && !conflict && <Dialog title={text('保存预设修改？', 'Save preset changes?')} closeDisabled={busy} onClose={cancelLeave}><p>{text('当前预设尚未保存。可以先保存，或放弃这些修改。', 'This preset has unsaved changes. Save them or discard the draft.')}</p>{error && <p role="alert" className="studio-error">{error}</p>}<footer className="presets-confirm-actions"><button type="button" className="ui-btn" disabled={busy} onClick={cancelLeave}>{text('继续编辑', 'Keep editing')}</button><button type="button" className="ui-btn" disabled={busy} onClick={proceed}>{text('放弃修改', 'Discard changes')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={busy} onClick={() => void save().then(ok => { if (ok) proceed(); })}>{text('保存并继续', 'Save and continue')}</button></footer></Dialog>}
    {conflict && <Dialog title={text('预设已被更新', 'Preset changed')} onClose={() => setConflict(null)}><p>{text('本地修改仍保留。合并会保留你修改的字段，并载入服务器其他字段的最新值。', 'Your changes are retained. Merging keeps your edited fields and loads the latest server values for other fields.')}</p><footer className="presets-confirm-actions"><button type="button" className="ui-btn" onClick={() => open(conflict)}>{text('使用服务器预设', 'Use server preset')}</button><button type="button" className="ui-btn ui-btn-primary" onClick={merge}>{text('合并我的修改', 'Merge my changes')}</button></footer></Dialog>}
    {deleting && <Dialog title={text('删除预设', 'Delete preset')} closeDisabled={busy} onClose={() => setDeleting(false)}><p>{text(`删除“${draft?.name}”？已有版本参数保持不变。`, `Delete “${draft?.name}”? Existing version parameters stay unchanged.`)}</p>{error && <p role="alert" className="studio-error">{error}</p>}<footer className="presets-confirm-actions"><button type="button" className="ui-btn" disabled={busy} onClick={() => setDeleting(false)}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-danger" disabled={busy} onClick={() => void remove()}>{text('确认删除', 'Confirm deletion')}</button></footer></Dialog>}
    {importing && <Dialog title={text('导入语音预设', 'Import speech preset')} closeDisabled={busy} onClose={() => { fileRead.current += 1; setReadingFile(false); setImporting(false); setError(''); }}><div className="presets-import"><input ref={fileInput} type="file" hidden accept=".json,application/json" onChange={event => { const file = event.target.files?.[0]; if (!file) return; const sequence = ++fileRead.current; setReadingFile(true); setImportText(''); void file.text().then(value => { if (mounted.current && fileRead.current === sequence) { setImportText(value); setReadingFile(false); } }).catch(failure => { if (mounted.current && fileRead.current === sequence) { setError(formatApiError(failure)); setReadingFile(false); } }); }}/><button type="button" className="ui-btn" disabled={busy} onClick={() => fileInput.current?.click()}>{text('选择 JSON 文件', 'Choose JSON file')}</button><label>{text('预设内容', 'Preset content')}<textarea aria-label={text('预设内容', 'Preset content')} disabled={busy} value={importText} onChange={event => { fileRead.current += 1; setReadingFile(false); setImportText(event.target.value); }}/></label>{error && <p role="alert" className="studio-error">{error}</p>}<footer className="presets-confirm-actions"><button type="button" className="ui-btn" disabled={busy} onClick={() => { fileRead.current += 1; setReadingFile(false); setImporting(false); }}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={busy || readingFile || !importText.trim()} onClick={() => void importPreset()}>{text('导入预设', 'Import preset')}</button></footer></div></Dialog>}
  </section>;
}
