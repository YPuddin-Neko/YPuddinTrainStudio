import React from 'react';
import { rawTtsDraft } from './ttsPresetParameters';
import { useBlocker } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Loader2, Save } from 'lucide-react';
import { ttsApi, type TtsConfigResponse, type TtsIssue, type TtsTrainSchema, type TtsScopedConfig } from '../../api/tts';
import { ApiError } from '../../api/types';
import Dialog from '../../components/Dialog';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { draftKey, fieldCopy, fields, fieldSchemas, groups, mergeDraft, parseDraft, readDraft, resolveTrainingSchema, toDraft, type FieldProblem, type TtsDraft, type TtsField } from './ttsVersionFields';
import '../../schema/SchemaForm/config-fields.css';
import TtsParameterForm, { TtsParameterToolbar } from './TtsParameterForm';
import { ttsParameterGroups } from './ttsParameterPresentation';
import GptSovitsVersionEditor from './GptSovitsVersionEditor';
import { isGptSovitsConfigResponse } from './gptSovitsVersionFields';
import type { VoxConfigResponse } from './ttsVersionFields';

export type TtsEditorHandle = { beforeAction: () => Promise<void>; saveConfig: () => Promise<TtsConfigResponse | null>; getConfig: () => TtsScopedConfig | null; getDraftConfig: () => Record<string, unknown>; loadConfig: (config: TtsScopedConfig) => boolean };
export type TtsEditorProps = { initial: TtsConfigResponse; readOnly: boolean; onSaved: (value: TtsConfigResponse) => void; schema?: TtsTrainSchema; onDirtyChange?: (dirty: boolean) => void; toolbarActions?: React.ReactNode; parameterActions?: React.ReactNode };

type Props = Omit<TtsEditorProps, 'initial'> & { initial: VoxConfigResponse };
const VoxVersionEditor = React.forwardRef<TtsEditorHandle, Props>(function VoxVersionEditor({ initial, readOnly, onSaved, schema, onDirtyChange, toolbarActions, parameterActions }, ref) {
  const text = useWorkspaceText(), english = text('zh', 'en') === 'en';
  const client = useQueryClient();
  const formId = React.useId();
  const { project_id: pid, version_id: vid } = initial.scope;
  const key = draftKey(pid, vid);
  const [local] = React.useState(() => readDraft(key, pid, vid));
  const [base, setBase] = React.useState(local?.base || initial);
  const stored = React.useRef(base);
  const resolved = schema ? resolveTrainingSchema(schema) : null;
  const properties = resolved?.properties || fieldSchemas, orderedGroups = resolved?.groups || groups;
  const [draft, setDraft] = React.useState<TtsDraft>(local?.draft || toDraft(initial.config));
  const [conflict, setConflict] = React.useState<VoxConfigResponse | null>(local && local.base.revision !== initial.revision ? initial : null);
  const [saving, setSaving] = React.useState(false);
  const pending = React.useRef(false), mounted = React.useRef(true);
  const [search, setSearch] = React.useState('');
  const [advanced, setAdvanced] = React.useState(false);
  const updateSearch = (value: string) => { setSearch(value); if (value.trim()) setClosed(new Set()); };
  const [closed, setClosed] = React.useState<Set<string>>(new Set());
  const [problems, setProblems] = React.useState<FieldProblem[]>([]);
  const [error, setError] = React.useState('');
  const [notice, setNotice] = React.useState(local ? text('已恢复未保存的草稿。', 'Recovered the unsaved draft.') : '');
  const [storageError, setStorageError] = React.useState(false);
  const form = React.useRef<HTMLFormElement>(null);
  const dirty = JSON.stringify(draft) !== JSON.stringify(toDraft(base.config));
  React.useEffect(() => { onDirtyChange?.(dirty); }, [dirty, onDirtyChange]);
  React.useEffect(() => {
    if (initial.revision <= base.revision || pending.current) return;
    if (dirty) { setConflict(initial); return; }
    stored.current = initial; setBase(initial); setDraft(toDraft(initial.config));
  }, [initial, base.revision, dirty]);
  React.useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  React.useEffect(() => {
    try {
      if (dirty || conflict) sessionStorage.setItem(key, JSON.stringify({ base, draft }));
      else sessionStorage.removeItem(key);
      setStorageError(false);
    } catch { setStorageError(true); }
  }, [key, base, draft, dirty, conflict]);
  React.useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [dirty]);
  const bypassNavigation = React.useRef(false);
  const blocker = useBlocker(({ currentLocation, nextLocation }) => {
    const workspacePath = (location: typeof currentLocation) => {
      const background = (location.state as { backgroundLocation?: { pathname: string } } | null)?.backgroundLocation;
      return (location.pathname === '/settings' || location.pathname.startsWith('/settings/')) && background?.pathname
        ? background.pathname : location.pathname;
    };
    return (dirty || pending.current) && !bypassNavigation.current && workspacePath(currentLocation) !== workspacePath(nextLocation);
  });
  const reveal = (issues: FieldProblem[]) => {
    setSearch(''); setAdvanced(true); setProblems(issues);
    setClosed(previous => new Set([...previous].filter(id => !orderedGroups.find(group => group.id === id)?.fields.some(field => issues.some(issue => issue.field === field)))));
    requestAnimationFrame(() => {
      const element = form.current?.querySelector<HTMLElement>(`[data-field="${issues[0]?.field}"] .config-field-control input:not(:disabled), [data-field="${issues[0]?.field}"] button[role="switch"], [data-field="${issues[0]?.field}"] .config-field-control button:not(:disabled)`);
      element?.focus();
    });
  };
  const save = async (): Promise<boolean> => {
    if (pending.current || readOnly || conflict) return false;
    const parsed = parseDraft(draft, text('请填写符合范围的有效数值。', 'Enter a valid value within the allowed range.'), properties);
    if (parsed.problems.length) { reveal(parsed.problems); return false; }
    if (!dirty) return true;
    pending.current = true; setSaving(true); setError(''); setNotice(''); setProblems([]);
    try {
      const saved = await ttsApi.saveVersionConfig(pid, vid, { expected_revision: base.revision, config: parsed.config });
      if (!mounted.current) return false;
      if (isGptSovitsConfigResponse(saved)) { onSaved(saved); return false; }
      const voxSaved = saved as VoxConfigResponse;
      stored.current = voxSaved; setBase(voxSaved); setDraft(toDraft(voxSaved.config)); onDirtyChange?.(false); onSaved(saved);
      try { sessionStorage.removeItem(key); } catch { /* The server copy is saved. */ }
      return true;
    } catch (failure) {
      if (!mounted.current) return false;
      if (failure instanceof ApiError && failure.code === 'tts.config_conflict') {
        setError(text('服务器参数已更新，草稿仍保留。', 'Server parameters changed; your draft is preserved.'));
        try {
          const remote = await ttsApi.versionConfig(pid, vid);
          if (!mounted.current) return false;
          if (isGptSovitsConfigResponse(remote)) onSaved(remote);
          else setConflict(remote as VoxConfigResponse);
        }
        catch (readFailure) { if (mounted.current) setError(formatApiError(readFailure)); }
      } else {
        setError(formatApiError(failure));
        const issues = failure instanceof ApiError && Array.isArray(failure.details?.issues) ? failure.details.issues as TtsIssue[] : [];
        const mapped = issues.map(issue => ({ field: String(issue.loc[0] === 'config' ? issue.loc[1] : issue.loc[0] || ''), message: issue.message }));
        if (mapped.length) reveal(mapped);
        void client.invalidateQueries({ queryKey: ['project-versions', pid] });
      }
      return false;
    } finally { pending.current = false; if (mounted.current) setSaving(false); }
  };
  const getConfig = () => {
    const parsed = parseDraft(draft, text('请填写符合范围的有效数值。', 'Enter a valid value within the allowed range.'), properties);
    if (parsed.problems.length) { reveal(parsed.problems); return null; }
    return structuredClone(parsed.config);
  };
  React.useImperativeHandle(ref, () => ({ beforeAction: async () => {
    if (dirty && !await save()) throw new Error(text('请先处理未保存的参数。', 'Resolve the unsaved parameters first.'));
  }, saveConfig: async () => await save() ? stored.current : null, getConfig, getDraftConfig: () => rawTtsDraft(draft),
  loadConfig: config => {
    if (pending.current || readOnly || conflict || config.engine !== 'voxcpm1.5') return false;
    setDraft(toDraft(structuredClone(config))); setProblems([]); setError(''); setNotice('');
    return true;
  } }));
  const change = (field: TtsField, value: TtsDraft[TtsField]) => {
    if (readOnly || saving) return;
    setDraft(previous => ({ ...previous, [field]: value })); setNotice('');
    setProblems(previous => previous.filter(problem => problem.field !== field));
  };
  const presentationGroups = ttsParameterGroups({ draft, properties, problems, orderedGroups, readOnly, saving, english, text, change });
  const differences = conflict ? fields.filter(field => JSON.stringify(draft[field]) !== JSON.stringify(toDraft(conflict.config)[field])) : [];
  const displayValue = (field: TtsField, value: unknown) => {
    if (value == null || value === '') return field === 'valid_interval' ? text('跟随保存间隔', 'Follow save interval') : '—';
    if (typeof value === 'boolean') return value ? text('开启', 'On') : text('关闭', 'Off');
    if (Array.isArray(value)) return value.join(', ') || '—';
    if (field === 'learning_rate' && Number.isFinite(Number(value))) return Number(value).toExponential();
    return String(value);
  };
  const acceptRemote = (merge: boolean) => {
    if (!conflict) return;
    setDraft(merge ? mergeDraft(base.config, draft, conflict.config) : toDraft(conflict.config));
    stored.current = conflict; setBase(conflict); onSaved(conflict); setConflict(null); setProblems([]); setError('');
    setNotice(merge ? text('已合并本地修改，请检查后保存。', 'Local changes merged. Review them before saving.') : text('已载入服务器参数。', 'Server parameters loaded.'));
  };
  return <>
    <form id={formId} ref={form} className="tts-version-editor" onSubmit={event => { event.preventDefault(); void save(); }} aria-busy={saving}>
      <TtsParameterToolbar model={'VoxCPM 1.5 · LoRA'} search={search} onSearch={updateSearch} advanced={advanced} onAdvancedChange={setAdvanced} parameterActions={parameterActions} status={saving ? text('正在保存…', 'Saving…') : dirty ? text('有未保存修改', 'Unsaved changes') : text('参数已保存', 'Parameters saved')} actions={<><button type="submit" form={formId} className={`ui-btn${toolbarActions ? '' : ' ui-btn-primary'}`} disabled={readOnly || saving || !dirty || !!conflict}>{saving ? <Loader2 size={14} className="animate-spin"/> : <Save size={14}/>} {text('保存参数', 'Save parameters')}</button>{toolbarActions}</>}/>
      {readOnly && <p className="workspace-message" role="status">{text('此版本当前只读。', 'This version is currently read-only.')}</p>}
      {error && <p className="workspace-message error" role="alert">{error}</p>}
      {notice && <p className="tts-editor-notice" role="status">{notice}</p>}
      {storageError && dirty && <p role="alert" className="workspace-message error">{text('浏览器无法保留草稿，请在离开前保存。', 'This browser cannot retain the draft. Save before leaving.')}</p>}
      <TtsParameterForm id="tts" groups={presentationGroups} advanced={advanced} onAdvancedChange={setAdvanced} search={search} onSearch={updateSearch} closed={closed} onClosedChange={setClosed}/>
    </form>
    {conflict && <Dialog title={text('服务器参数已更新', 'Server parameters changed')} onClose={() => { setConflict(null); setError(text('草稿仍保留；再次保存前会重新检查版本。', 'Draft preserved. Saving will check the revision again.')); }}>
      <p>{text('本地草稿仍保留。合并会保留你修改过的字段，并采用服务器上其他字段的最新值；合并后需重新保存。', 'Your draft is preserved. Merging keeps fields you edited and uses the latest server values for other fields. Save after reviewing the merge.')}</p>
      <div className="tts-conflict-table"><table><thead><tr><th>{text('参数', 'Parameter')}</th><th>{text('服务器', 'Server')}</th><th>{text('本地草稿', 'Local draft')}</th></tr></thead><tbody>{differences.map(field => <tr key={field}><th>{fieldCopy(field, english).label}</th><td>{displayValue(field, conflict.config[field])}</td><td>{displayValue(field, draft[field])}</td></tr>)}</tbody></table></div>
      <div className="tts-dialog-actions"><button className="ui-btn" onClick={() => acceptRemote(false)}>{text('使用服务器参数', 'Use server parameters')}</button><button className="ui-btn ui-btn-primary" onClick={() => acceptRemote(true)}>{text('合并我的修改', 'Merge my changes')}</button></div>
    </Dialog>}
    {blocker.state === 'blocked' && !conflict && <Dialog title={text('有未保存的参数', 'Unsaved parameters')} onClose={() => blocker.reset()} closeDisabled={saving}><p>{text('是否保存此版本的修改？', 'Save changes to this version?')}</p><div className="tts-dialog-actions"><button className="ui-btn" disabled={saving} onClick={() => blocker.reset()}>{text('继续编辑', 'Keep editing')}</button><button className="ui-btn" disabled={saving} onClick={() => { try { sessionStorage.removeItem(key); } catch { /* Navigation is explicitly confirmed. */ } bypassNavigation.current = true; blocker.proceed(); }}>{text('放弃修改并离开', 'Discard and leave')}</button><button className="ui-btn ui-btn-primary" disabled={saving || readOnly} onClick={() => void save().then(ok => { if (ok) { bypassNavigation.current = true; blocker.proceed(); } })}>{text('保存并离开', 'Save and leave')}</button></div></Dialog>}
  </>;
});
const TtsVersionEditor = React.forwardRef<TtsEditorHandle, TtsEditorProps>(function TtsVersionEditor(props, ref) {
  return isGptSovitsConfigResponse(props.initial)
    ? <GptSovitsVersionEditor {...props} initial={props.initial} ref={ref}/>
    : <VoxVersionEditor {...props} initial={props.initial as VoxConfigResponse} ref={ref}/>;
});
export default TtsVersionEditor;
