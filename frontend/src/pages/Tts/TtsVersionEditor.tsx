import React from 'react';
import { useBlocker } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { ChevronDown, Loader2, Save } from 'lucide-react';
import { ttsApi, type TtsConfigResponse, type TtsIssue, type TtsTrainSchema } from '../../api/tts';
import { ApiError } from '../../api/types';
import ConfigHelp from '../../components/ConfigHelp';
import Dialog from '../../components/Dialog';
import Switch from '../../components/Switch';
import { PathInput } from '../../components/PathBrowser';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { draftKey, fieldCopy, fields, fieldSchemas, groups, mergeDraft, parseDraft, readDraft, resolveTrainingSchema, targetOwners, toDraft, type FieldProblem, type TtsDraft, type TtsField } from './ttsVersionFields';
import '../../schema/SchemaForm/config-fields.css';
import GptSovitsVersionEditor from './GptSovitsVersionEditor';
import { isGptSovitsConfigResponse } from './gptSovitsVersionFields';
import type { VoxConfigResponse } from './ttsVersionFields';

export type TtsEditorHandle = { beforeAction: () => Promise<void>; saveConfig: () => Promise<TtsConfigResponse | null> };
export type TtsEditorProps = { initial: TtsConfigResponse; readOnly: boolean; onSaved: (value: TtsConfigResponse) => void; schema?: TtsTrainSchema; onDirtyChange?: (dirty: boolean) => void; toolbarActions?: React.ReactNode };

type Props = Omit<TtsEditorProps, 'initial'> & { initial: VoxConfigResponse };
const VoxVersionEditor = React.forwardRef<TtsEditorHandle, Props>(function VoxVersionEditor({ initial, readOnly, onSaved, schema, onDirtyChange, toolbarActions }, ref) {
  const text = useWorkspaceText(), english = text('zh', 'en') === 'en';
  const client = useQueryClient();
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
    setProblems(issues);
    setClosed(previous => new Set([...previous].filter(id => !orderedGroups.find(group => group.id === id)?.fields.some(field => issues.some(issue => issue.field === field)))));
    requestAnimationFrame(() => {
      const element = form.current?.querySelector<HTMLElement>(`[data-field="${issues[0]?.field}"] input, [data-field="${issues[0]?.field}"] button`);
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
  React.useImperativeHandle(ref, () => ({ beforeAction: async () => {
    if (dirty && !await save()) throw new Error(text('请先处理未保存的参数。', 'Resolve the unsaved parameters first.'));
  }, saveConfig: async () => await save() ? stored.current : null }));
  const change = (field: TtsField, value: TtsDraft[TtsField]) => {
    if (readOnly || saving) return;
    setDraft(previous => ({ ...previous, [field]: value })); setNotice('');
    setProblems(previous => previous.filter(problem => problem.field !== field));
  };
  const bounds = (field: TtsField) => {
    const prop = properties[field], rule = prop.anyOf?.find(item => item.type !== 'null') || prop;
    return [rule.type === 'integer' ? text('整数', 'Integer') : rule.type === 'number' ? text('数值', 'Number') : '',
      rule.minimum != null ? `≥ ${rule.minimum}` : rule.exclusiveMinimum != null ? `> ${rule.exclusiveMinimum}` : '',
      rule.maximum != null ? `≤ ${rule.maximum}` : rule.exclusiveMaximum != null ? `< ${rule.exclusiveMaximum}` : ''].filter(Boolean).join(' · ');
  };
  const renderField = (field: TtsField): React.ReactNode => {
    const property = properties[field], copy = fieldCopy(field, english), value = draft[field];
    const id = `tts-version-${field}`, problem = problems.filter(issue => issue.field === field).map(issue => issue.message).join(' ');
    const heading = <div className="config-field-heading"><label htmlFor={property.type === 'string' ? undefined : id}>{copy.label}</label><code tabIndex={0} className="tts-field-key">{field}</code><ConfigHelp label={`${copy.label} · ${text('说明', 'Help')}`}>{[copy.help, bounds(field), property.type === 'array' ? text('只选列表中的层名；启用该组件时至少选择一层，不支持正则表达式。', 'Choose listed layer names, at least one for an enabled component. Regular expressions are not supported.') : ''].filter(Boolean).join('\n\n')}</ConfigHelp></div>;
    if (property.type === 'boolean') {
      const target = Object.keys(targetOwners).find(name => targetOwners[name as TtsField] === field) as TtsField;
      return <section key={field} data-field={field} className="tts-lora-component"><div className="tts-lora-heading">{heading}<Switch id={id} aria-label={copy.label} aria-controls={`${id}-targets`} checked={value === true} disabled={readOnly || saving} onCheckedChange={checked => change(field, checked)}/></div>{problem && <p role="alert" className="config-field-error">{problem}</p>}<div id={`${id}-targets`}>{renderField(target)}</div></section>;
    }
    return <div key={field} data-field={field} className={`config-field${property.type === 'string' || property.type === 'array' ? ' config-field-wide' : ''}${problem ? ' config-field-invalid' : ''}`}>
      {heading}<fieldset className="config-field-control" disabled={readOnly || saving || !!targetOwners[field] && draft[targetOwners[field]!] !== true}>
        {property.type === 'string' ? <PathInput ariaLabel={copy.label} value={String(value)} onChange={next => change(field, next)} directoryOnly={field !== 'python_path'}/>
          : property.type === 'array' ? <div id={id} role="group" aria-label={copy.label} className="tts-target-options">{property.items?.enum?.map(target => <label key={target}><input type="checkbox" checked={Array.isArray(value) && value.includes(target)} onChange={event => change(field, event.target.checked ? [...(Array.isArray(value) ? value : []), target] : (value as string[]).filter(item => item !== target))}/><span>{target}</span></label>)}</div>
            : <input id={id} type="text" inputMode="decimal" autoComplete="off" spellCheck={false} value={String(value)} aria-invalid={!!problem} aria-describedby={`${id}-hint${problem ? ` ${id}-error` : ''}`} placeholder={field === 'valid_interval' ? text('跟随保存间隔', 'Follow save interval') : undefined} onChange={event => change(field, event.target.value)} onBlur={() => { if (field === 'learning_rate' && String(value).trim() && Number.isFinite(Number(value))) change(field, Number(value).toExponential()); }}/>}</fieldset>
      <div className="config-field-footer"><p id={`${id}-hint`} className="config-field-hint">{copy.hint}</p>{problem && <p id={`${id}-error`} role="alert" className="config-field-error">{problem}</p>}</div>
    </div>;
  };
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
    <form ref={form} className="tts-version-editor" onSubmit={event => { event.preventDefault(); void save(); }} aria-busy={saving}>
      <div className="tts-editor-toolbar"><span>VoxCPM 1.5 · LoRA</span><span role="status">{saving ? text('正在保存…', 'Saving…') : dirty ? text('有未保存修改', 'Unsaved changes') : text('参数已保存', 'Parameters saved')}</span><button className={`ui-btn${toolbarActions ? '' : ' ui-btn-primary'}`} disabled={readOnly || saving || !dirty || !!conflict}>{saving ? <Loader2 size={14} className="animate-spin"/> : <Save size={14}/>} {text('保存参数', 'Save parameters')}</button>{toolbarActions}</div>
      {readOnly && <p className="workspace-message" role="status">{text('此版本当前只读。', 'This version is currently read-only.')}</p>}
      {error && <p className="workspace-message error" role="alert">{error}</p>}
      {notice && <p className="tts-editor-notice" role="status">{notice}</p>}
      {storageError && dirty && <p role="alert" className="workspace-message error">{text('浏览器无法保留草稿，请在离开前保存。', 'This browser cannot retain the draft. Save before leaving.')}</p>}
      <div className="compact-schema tts-config-form" data-testid="schema-form">{orderedGroups.map(group => <section className="config-group" key={group.id}><button type="button" className="config-group-title" aria-expanded={!closed.has(group.id)} aria-controls={`tts-group-${group.id}`} onClick={() => setClosed(previous => { const next = new Set(previous); if (next.has(group.id)) next.delete(group.id); else next.add(group.id); return next; })}><span>{english ? group.en : group.zh}</span><ChevronDown className={closed.has(group.id) ? 'tts-collapsed' : ''}/></button><div id={`tts-group-${group.id}`} className="tts-disclosure" data-open={!closed.has(group.id)} aria-hidden={closed.has(group.id)} {...(closed.has(group.id) ? { inert: '' } : {})}><div><div className="config-fields"><div className="config-field-section">{group.fields.filter(field => !targetOwners[field]).map(renderField)}</div></div></div></div></section>)}</div>
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
