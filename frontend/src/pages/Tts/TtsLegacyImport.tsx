import React from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Download, Loader2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { ttsApi, type TtsConfig, type TtsConfigResponse, type TtsSourcesResponse, type TtsVersionConfig } from '../../api/tts';
import Dialog from '../../components/Dialog';
import StudioSelect from '../../components/StudioSelect';
import { formatApiError } from '../../utils/errors';
import { projectUrl } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import { fieldCopy, parseDraft, toDraft } from './ttsVersionFields';
import './tts-import.css';

const parameterFields = ['python_path', 'trainer_path', 'model_path', 'batch_size', 'grad_accum_steps', 'num_workers', 'num_iters', 'save_interval', 'learning_rate', 'warmup_steps', 'lora_rank', 'lora_alpha'] as const;
type LegacyConfig = { [K in keyof TtsConfig]: TtsConfig[K] extends number ? number | string : TtsConfig[K] };
type VoxResponse = Omit<TtsConfigResponse, 'config'> & { config: TtsVersionConfig };
type Step = 'config' | 'train' | 'validation';
type Plan = { source: string; config: LegacyConfig; steps: Step[]; done: Step[] };
const steps: Step[] = ['config', 'train', 'validation'];
const storageKey = (pid: string, vid: string) => `tts-legacy-import:v1:${pid}:${vid}`;
function readPlan(key: string): Plan | null {
  try {
    const plan = JSON.parse(sessionStorage.getItem(key) || 'null');
    return plan && typeof plan.source === 'string' && validLegacy(plan.config) && Array.isArray(plan.steps) && plan.steps.length && plan.steps.every((step: Step) => steps.includes(step)) && Array.isArray(plan.done) && plan.done.every((step: Step) => plan.steps.includes(step)) ? plan : null;
  } catch { return null; }
}
function validLegacy(config: unknown): config is LegacyConfig {
  if (!config || typeof config !== 'object') return false;
  const value = config as LegacyConfig;
  return value.engine === 'voxcpm1.5' && ([...parameterFields, 'train_manifest', 'val_manifest'] as const).every(key => ['python_path', 'trainer_path', 'model_path', 'train_manifest', 'val_manifest'].includes(key) ? typeof value[key] === 'string' : ['string', 'number'].includes(typeof value[key]));
}
function browserDraft(base: TtsConfig): LegacyConfig | null {
  try {
    const saved = JSON.parse(sessionStorage.getItem('tts-training-draft:v1') || 'null');
    if (!saved?.draft || !([...parameterFields, 'train_manifest', 'val_manifest'] as const).every(key => typeof saved.draft[key] === 'string')) return null;
    return Object.fromEntries(Object.entries(base).map(([key, value]) => [key, key === 'engine' ? value : saved.draft[key]])) as unknown as LegacyConfig;
  } catch { return null; }
}
function mergeParameters(current: TtsVersionConfig, legacy: LegacyConfig): TtsVersionConfig {
  return { ...current, ...Object.fromEntries(parameterFields.map(field => [field, typeof current[field] === 'number' ? String(legacy[field]).trim() ? Number(legacy[field]) : Number.NaN : legacy[field]])) };
}

export default function TtsLegacyImport({ projectId: pid, versionId: vid, readOnly, ensureSaved, onImported }: {
  projectId: string; versionId: string; readOnly: boolean;
  ensureSaved: () => Promise<TtsConfigResponse | null>; onImported: (value: TtsConfigResponse) => void;
}) {
  const text = useWorkspaceText(), client = useQueryClient();
  const key = storageKey(pid, vid);
  const [open, setOpen] = React.useState(false), [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(''), [needsRefresh, setNeedsRefresh] = React.useState(false);
  const [global, setGlobal] = React.useState<TtsConfig | null>(null), [local, setLocal] = React.useState<LegacyConfig | null>(null);
  const [source, setSource] = React.useState('global'), [selected, setSelected] = React.useState<Step[]>([]);
  const [current, setCurrent] = React.useState<VoxResponse | null>(null), [sources, setSources] = React.useState<TtsSourcesResponse | null>(null);
  const [plan, setPlan] = React.useState<Plan | null>(null);
  const locked = React.useRef(false), alive = React.useRef(true), latestReadOnly = React.useRef(readOnly);
  latestReadOnly.current = readOnly;
  React.useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const owns = (scope: { project_id: string; version_id: string }) => scope.project_id === pid && scope.version_id === vid;
  const requireVox = (response: TtsConfigResponse): VoxResponse => {
    if (response.config.engine !== 'voxcpm1.5') throw new Error(text('旧草稿只能导入 VoxCPM 1.5 版本。', 'Legacy drafts can only be imported into VoxCPM 1.5 versions.'));
    return { ...response, config: response.config };
  };
  const savePlan = (value: Plan) => {
    sessionStorage.setItem(key, JSON.stringify(value));
    if (alive.current) setPlan(value);
  };
  const loadCurrent = async () => {
    const [config, data] = await Promise.all([ttsApi.versionConfig(pid, vid).then(requireVox), ttsApi.sources(pid, vid)]);
    if (!owns(config.scope) || !owns(data.scope) || config.data_revision !== data.data_revision) throw new Error(text('版本数据已变化，请重新读取。', 'Version data changed. Reload it.'));
    if (alive.current) { setCurrent(config); setSources(data); }
    client.setQueryData(['tts-sources', pid, vid], data);
    return { config, data };
  };
  const begin = async () => {
    if (locked.current || readOnly) return;
    locked.current = true; setBusy(true); setError('');
    try {
      if (!await ensureSaved() || latestReadOnly.current || !alive.current) return;
      setOpen(true); setSelected([]); setNeedsRefresh(false); setCurrent(null); setSources(null); setGlobal(null); setLocal(null);
      const [legacy] = await Promise.all([ttsApi.config(), loadCurrent()]);
      if (!validLegacy(legacy)) throw new Error(text('旧配置格式无法识别。', 'The legacy configuration format is not recognized.'));
      if (alive.current) { setGlobal(legacy as TtsConfig); setLocal(browserDraft(legacy as TtsConfig)); setSource('global'); setPlan(readPlan(key)); }
    } catch (caught) { if (alive.current) setError(formatApiError(caught)); }
    finally { locked.current = false; if (alive.current) setBusy(false); }
  };
  const refresh = async () => {
    if (locked.current) return;
    locked.current = true; setBusy(true); setError('');
    try { await loadCurrent(); if (alive.current) setNeedsRefresh(false); }
    catch (caught) { if (alive.current) setError(formatApiError(caught)); }
    finally { locked.current = false; if (alive.current) setBusy(false); }
  };
  const legacy = plan?.config || (source === 'browser' ? local : global);
  const complete = !!plan && plan.steps.every(step => plan.done.includes(step));
  const chosen = plan?.steps || selected;
  const apply = async () => {
    if (locked.current || latestReadOnly.current || needsRefresh || !legacy || !current || !sources || !chosen.length) return;
    locked.current = true; setBusy(true); setError('');
    let next = plan || { source, config: legacy, steps: selected, done: [] };
    let config: VoxResponse, data: TtsSourcesResponse;
    try {
      const [freshConfig, freshSources] = await Promise.all([ttsApi.versionConfig(pid, vid).then(requireVox), ttsApi.sources(pid, vid)]);
      if (!owns(freshConfig.scope) || !owns(freshSources.scope) || freshConfig.data_revision !== freshSources.data_revision
        || next.steps.includes('config') && !next.done.includes('config') && freshConfig.revision !== current.revision
        || next.steps.some(step => step !== 'config' && !next.done.includes(step)) && freshSources.data_revision !== sources.data_revision) {
        throw new Error(text('目标版本已变化，请重新读取后核对导入内容。', 'The destination version changed. Reload and review the selected items.'));
      }
      config = freshConfig; data = freshSources;
      savePlan(next);
      for (const step of next.steps) {
        if (next.done.includes(step)) continue;
        if (!alive.current || latestReadOnly.current) throw new Error(text('版本当前不可编辑，已完成的步骤已保留。', 'The version cannot be edited now. Completed steps are preserved.'));
        if (step === 'config') {
          const merged = mergeParameters(config.config, next.config);
          const parsed = parseDraft(toDraft(merged), text('请检查旧参数的数值范围。', 'Check the ranges of the legacy parameters.'));
          if (parsed.problems.length) throw new Error(parsed.problems.map(item => `${item.field}: ${item.message}`).join('\n'));
          if (parameterFields.some(field => config.config[field] !== merged[field])) config = requireVox(await ttsApi.saveVersionConfig(pid, vid, { expected_revision: config.revision, config: merged }));
          if (!owns(config.scope)) throw new Error(text('返回的配置不属于当前版本。', 'The returned configuration belongs to another version.'));
          onImported(config);
        } else {
          const path = step === 'train' ? next.config.train_manifest.trim() : next.config.val_manifest.trim();
          if (data.items.find(item => item.split === step)?.path !== path) data = await ttsApi.putSource(pid, vid, step, { path, expected_data_revision: data.data_revision });
          if (!owns(data.scope)) throw new Error(text('返回的数据不属于当前版本。', 'The returned data belongs to another version.'));
          client.setQueryData(['tts-sources', pid, vid], data);
        }
        next = { ...next, done: [...next.done, step] }; savePlan(next);
      }
      const fresh = await loadCurrent(); onImported(fresh.config);
      void client.invalidateQueries({ queryKey: ['project-versions', pid] });
    } catch (caught) { if (alive.current) { setError(formatApiError(caught)); setNeedsRefresh(true); } }
    finally { locked.current = false; if (alive.current) setBusy(false); }
  };
  const label = (step: Step) => step === 'config' ? text('训练参数与环境路径', 'Training parameters and environment paths') : step === 'train' ? text('训练清单', 'Training manifest') : text('验证清单', 'Validation manifest');
  const format = (value: unknown, field?: string) => typeof value === 'number' && field === 'learning_rate' ? value.toExponential() : value === '' ? text('空', 'Empty') : String(value ?? '—');
  const close = () => { if (!locked.current) { setOpen(false); setError(''); } };
  return <>
    <button type="button" className="ui-btn" onClick={() => void begin()} disabled={readOnly || busy}><Download size={14}/>{text('导入旧草稿', 'Import legacy draft')}</button>
    {!open && error && <span role="alert">{error}</span>}
    {open && <Dialog wide title={text('导入旧语音草稿', 'Import legacy speech draft')} onClose={close} closeDisabled={busy}>
      <div className="tts-legacy-import">
        <p>{text('将选中内容写入当前版本。旧草稿保留，旧任务仍在任务队列中。', 'Write selected items into this version. The legacy draft is preserved; old jobs remain in the queue.')}</p>
        <p className="tts-legacy-target">{text('目标版本', 'Target version')}：{pid} / {vid}</p>
        {busy && <p role="status"><Loader2 size={14} className="animate-spin"/>{text('正在处理…', 'Working…')}</p>}
        {error && <p role="alert" className="workspace-message error">{error}</p>}
        {global && current && sources && <>
          {local && !plan && <StudioSelect aria-label={text('旧草稿来源', 'Legacy draft source')} value={source} onValueChange={setSource} disabled={busy} options={[{value:'global',label:text('服务端旧配置', 'Legacy server configuration')},{value:'browser',label:text('此浏览器未保存草稿', 'Unsaved draft in this browser')}]}/>}
          {plan && !complete && <p role="status">{text('已保留本次导入进度；继续时只处理未完成的步骤。', 'Import progress is preserved. Continuing processes only unfinished steps.')}</p>}
          {legacy && steps.map(step => {
            const done = plan?.done.includes(step), path = step === 'train' ? legacy.train_manifest : legacy.val_manifest;
            const beforePath = sources.items.find(item => item.split === step)?.path;
            return <section className="tts-legacy-step" key={step}>
              <label><input type="checkbox" aria-label={label(step)} checked={chosen.includes(step)} disabled={busy || !!plan || step !== 'config' && !path.trim()} onChange={event => setSelected(previous => event.target.checked ? steps.filter(item => item === step || previous.includes(item)) : previous.filter(item => item !== step))}/><strong>{label(step)}</strong>{done && <span role="status">{text('已完成', 'Completed')}</span>}</label>
              {step === 'config' ? <div className="tts-conflict-table" role="region" tabIndex={0} aria-label={text('旧参数与当前版本对比', 'Legacy and current parameters')}><table><thead><tr><th>{text('参数', 'Parameter')}</th><th>{text('当前版本', 'Current version')}</th><th>{text('旧草稿', 'Legacy draft')}</th></tr></thead><tbody>{parameterFields.map(field => <tr key={field}><th>{fieldCopy(field, text('zh','en') === 'en').label}</th><td>{format(current.config[field], field)}</td><td>{format(legacy[field], field)}</td></tr>)}</tbody></table></div> : <><dl><dt>{text('当前登记', 'Current registration')}</dt><dd>{beforePath || text('未登记', 'Not registered')}</dd><dt>{text('旧草稿清单', 'Legacy manifest')}</dt><dd>{path || text('无', 'None')}</dd></dl>{path && beforePath && beforePath !== path && <p>{text('选中后替换此项登记，原文件保留。', 'Selecting this replaces the registration and preserves the original files.')}</p>}</>}
            </section>;
          })}
          {complete && <p role="status">{text('所选内容已导入。新登记的清单需要检查后才能训练。', 'Selected items imported. Check newly registered manifests before training.')} <Link className="ui-link" to={projectUrl(pid, vid, 'data')}>{text('前往训练数据', 'Open training data')}</Link></p>}
          <footer className="tts-dialog-actions">
            <button type="button" className="ui-btn" disabled={busy} onClick={close}>{complete ? text('完成', 'Done') : text('关闭', 'Close')}</button>
            {plan && !busy && <button type="button" className="ui-btn" onClick={() => { try { sessionStorage.removeItem(key); setPlan(null); setSelected([]); setError(''); setNeedsRefresh(true); } catch (caught) { setError(formatApiError(caught)); } }}>{text('重新选择内容', 'Select different items')}</button>}
            {!complete && (needsRefresh ? <button type="button" className="ui-btn ui-btn-primary" disabled={busy} onClick={() => void refresh()}>{text('读取最新状态', 'Reload current state')}</button> : <button type="button" className="ui-btn ui-btn-primary" disabled={busy || readOnly || !chosen.length} onClick={() => void apply()}>{plan ? text('继续导入未完成项', 'Continue unfinished steps') : text('确认导入所选内容', 'Import selected items')}</button>)}
          </footer>
        </>}
        {(!global || !current || !sources) && !busy && <button type="button" className="ui-btn" onClick={() => void begin()}>{text('重新读取', 'Reload')}</button>}
      </div>
    </Dialog>}
  </>;
}
