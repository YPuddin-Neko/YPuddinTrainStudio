import React from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { useTtsEnvironments, useTtsEnvironmentOperation, useTtsEnvironmentActions } from '../api/hooks/useTtsEnvironments';
import { ttsEnvironmentActive, type TtsEnvironmentOperation, type TtsEnvironmentSnapshot } from '../api/ttsEnvironments';
import type { TtsEngine } from '../api/tts';
import { formatApiError } from '../utils/errors';
import { useWorkspaceText } from '../utils/workspaceText';
import InstallationOperation, { InstallationLog, InstallationProgress } from './InstallationOperation';
import StudioSelect from './StudioSelect';
import './tts-environment.css';

type Props = { engine?: TtsEngine; disabled?: boolean; pythonPath?: string; trainerPath?: string; onUseAutomatic?: () => void; compact?: boolean };
const engines: TtsEngine[] = ['voxcpm1.5', 'gpt-sovits-v5'];
const engineName = (engine: TtsEngine) => engine === 'voxcpm1.5' ? 'VoxCPM 1.5' : 'GPT-SoVITS v5';

export default function TtsEnvironmentPanel({ engine, compact = false, ...props }: Props) {
  const text = useWorkspaceText(), query = useTtsEnvironments();
  return <div className={`tts-environments${compact ? ' tts-environments-compact' : ''}`}>
    {!query.data && !query.error && <p role="status" className="tts-environment-note"><Loader2 size={14} className="animate-spin"/>{text('正在读取语音环境…', 'Loading speech environments…')}</p>}
    {query.error && <p role="alert" className="tts-environment-error">{text('无法读取语音环境。', 'Could not load speech environments.')} {formatApiError(query.error)}<button type="button" className="ui-link" disabled={query.isFetching} onClick={() => void query.refetch()}>{text('重试', 'Retry')}</button></p>}
    {(engine ? [engine] : engines).map(value => <EngineEnvironment key={value} {...props} engine={value} snapshot={query.data} unavailable={!query.data || !!query.error} compact={compact}/>) }
    {query.data && !query.error && <button type="button" className="ui-link tts-environment-refresh" disabled={query.isFetching} onClick={() => void query.refetch()}>{text('刷新状态', 'Refresh status')}</button>}
  </div>;
}

function EngineEnvironment({ engine, snapshot, unavailable, disabled = false, pythonPath, trainerPath, onUseAutomatic, compact }: Omit<Props, 'engine'> & { engine: TtsEngine; snapshot?: TtsEnvironmentSnapshot; unavailable: boolean }) {
  const text = useWorkspaceText(), actions = useTtsEnvironmentActions();
  const defaultId = snapshot?.defaults[engine];
  const environment = snapshot?.environments.find(item => item.id === defaultId && item.engine === engine);
  const ready = environment?.state === 'ready';
  const available = snapshot?.environments.filter(item => item.engine === engine && item.state === 'ready') || [];
  const operation = snapshot?.operations.filter(item => item.engine === engine).sort((a, b) => b.created_at - a.created_at)[0];
  const active = snapshot?.operations.some(item => item.engine === engine && ttsEnvironmentActive(item));
  const locked = disabled || unavailable || actions.pending;
  const explicit = !!pythonPath?.trim() || !!trainerPath?.trim();
  const mixed = explicit && !(pythonPath?.trim() && trainerPath?.trim());
  const candidateChecks = snapshot?.candidates.flatMap(candidate => (candidate.checks || []).filter(check => check.engine === engine)) || [];
  const checked = candidateChecks.some(check => check.state !== 'unchecked');
  const compatible = candidateChecks.some(check => check.state === 'ready');
  const hasConflict = candidateChecks.some(check => check.state === 'incompatible');
  const status = !snapshot ? text('状态未知', 'Status unknown') : ready ? text('环境已准备', 'Environment prepared')
    : environment?.state === 'changed' ? text('环境已变化', 'Environment changed')
      : environment ? text('环境不可用', 'Environment unavailable') : text('尚未准备', 'Not prepared');
  const sourceName = (kind: string) => kind === 'deployment' ? text('部署环境', 'Deployment environment') : text('受管理环境', 'Managed environment');
  const run = (kind: 'check' | 'prepare') => { actions.reset(); actions.run({ kind, engine }); };
  return <section className="tts-environment" aria-label={engineName(engine)}>
    <div className="tts-environment-row">
      <div className="tts-environment-info">{!compact && <strong>{engineName(engine)}</strong>}<span role="status">{explicit ? mixed ? text('外部路径与自动环境', 'External paths and automatic environment') : text('使用外部路径', 'Using external paths') : status}</span>
        {compact && <small>{explicit ? mixed ? text('已填路径优先；留空项使用默认环境。', 'Entered paths take priority; blank paths use the default environment.') : text('使用下方高级参数中填写的环境路径。', 'Uses the environment paths entered in advanced parameters below.') : text('自动使用此模型类型的默认环境。', 'Automatically uses the default environment for this model type.')}</small>}
        {ready && <small>{sourceName(environment.kind)}{environment.check.python_version ? ` · Python ${environment.check.python_version}` : ''}{explicit ? ` · ${text('自动环境已准备', 'Automatic environment prepared')}` : ''}</small>}
        {!ready && checked && <small>{compatible ? text('现有环境可复用，准备后即可选用。', 'An existing environment can be reused after preparation.') : hasConflict ? text('依赖不兼容，准备时将创建受管理环境。', 'Dependencies conflict. Preparation will create a managed environment.') : text('准备时将补齐源码与所需依赖。', 'Preparation will set up the trainer source and required dependencies.')}</small>}
      </div>
      <div className="tts-environment-actions">
        {explicit && onUseAutomatic && <button type="button" className="ui-btn ui-btn-sm" disabled={disabled} onClick={onUseAutomatic}>{text('使用自动环境', 'Use automatic environment')}</button>}
        <button type="button" className="ui-btn ui-btn-sm" disabled={locked || active} onClick={() => run('check')}><RefreshCw size={13}/>{text('检查环境', 'Check environment')}</button>
        {!ready && <button type="button" className="ui-btn ui-btn-sm" disabled={locked || active} onClick={() => run('prepare')}>{text('准备环境', 'Prepare environment')}</button>}
      </div>
    </div>
    {available.length > 0 && (!ready || available.length > 1) && <label className="tts-environment-default"><span>{text('默认环境', 'Default environment')}</span><StudioSelect aria-label={`${engineName(engine)} ${text('默认环境', 'default environment')}`} value={defaultId || ''} placeholder={text('选择已准备的环境', 'Choose a prepared environment')} options={available.map(item => ({ value: item.id, label: `${sourceName(item.kind)}${item.check.python_version ? ` · Python ${item.check.python_version}` : ''} · ${item.id.slice(-8)}` }))} disabled={locked || active} onValueChange={environmentId => { actions.reset(); actions.run({ kind: 'default', engine, environmentId }); }}/></label>}
    {actions.error && <p role="alert" className="tts-environment-error">{formatApiError(actions.error)}</p>}
    {!!environment?.issues?.length && <ul className="tts-environment-issues">{environment.issues.map((issue, index) => <li key={`${issue.code}:${index}`}>{issue.message}</li>)}</ul>}
    {environment && <details className="tts-environment-details"><summary>{text('查看环境', 'Environment details')}</summary><dl><div><dt>Python</dt><dd>{environment.python_path}</dd></div><div><dt>{text('训练器源码', 'Trainer source')}</dt><dd>{environment.trainer_path}</dd></div><div><dt>{text('源码版本', 'Source revision')}</dt><dd>{environment.upstream_revision}</dd></div></dl></details>}
    {operation && <EnvironmentOperation key={operation.id} operation={operation} disabled={locked} onCancel={() => { actions.reset(); actions.run({ kind: 'cancel', id: operation.id }); }} onRetry={() => { actions.reset(); actions.run({ kind: 'retry', id: operation.id }); }}/>}
  </section>;
}

function EnvironmentOperation({ operation: summary, disabled, onCancel, onRetry }: { operation: TtsEnvironmentOperation; disabled: boolean; onCancel: () => void; onRetry: () => void }) {
  const text = useWorkspaceText();
  const [expanded, setExpanded] = React.useState(ttsEnvironmentActive(summary) || summary.status === 'failed');
  const detail = useTtsEnvironmentOperation(summary.id, expanded);
  const operation = detail.data && detail.data.updated_at >= summary.updated_at ? detail.data : summary;
  const active = ttsEnvironmentActive(operation);
  const phases: Record<string, string> = { queued: text('等待执行', 'Queued'), discovering: text('查找现有环境', 'Discovering environments'), checking: text('检查依赖与源码', 'Checking dependencies and source'), source: text('准备训练器源码', 'Preparing trainer source'), dependencies: text('准备依赖', 'Preparing dependencies'), verifying: text('检查安装结果', 'Verifying installation'), ready: text('环境已准备', 'Environment prepared'), checked: text('检查完成', 'Check completed'), cancelling: text('正在取消…', 'Cancelling…'), cancelled: text('已取消', 'Cancelled'), failed: text('操作失败', 'Operation failed'), interrupted: text('操作已中断', 'Operation interrupted') };
  const status = operation.status === 'completed' ? operation.action === 'check' ? text('检查完成', 'Check completed') : text('准备完成', 'Preparation completed') : operation.status === 'failed' ? text('操作失败', 'Operation failed') : operation.status === 'cancelled' ? text('已取消', 'Cancelled') : ['ready', 'checked', 'cancelled', 'failed', 'interrupted'].includes(operation.phase) ? text('正在结束操作…', 'Finishing operation…') : phases[operation.phase] || text('正在处理…', 'Working…');
  const title = operation.action === 'check' ? text('环境检查', 'Environment check') : text('环境准备', 'Environment preparation');
  return <InstallationOperation title={title} status={status} busy={active} failed={operation.status === 'failed'} expanded={expanded} onToggle={() => setExpanded(value => !value)}>
    {active && <InstallationProgress label={status} percent={operation.progress == null ? undefined : operation.progress * 100}/>}
    {!!operation.issues?.length && <ul className="tts-environment-issues">{operation.issues.map((issue, index) => <li key={`${issue.code}:${index}`}>{issue.message}</li>)}</ul>}
    {detail.error && <p role="alert" className="tts-environment-error">{text('无法读取操作详情。', 'Could not load operation details.')}<button type="button" className="ui-link" disabled={detail.isFetching} onClick={() => void detail.refetch()}>{text('重试', 'Retry')}</button></p>}
    {!!operation.logs?.length && <InstallationLog label={`${engineName(operation.engine)} ${title}`} logs={operation.logs}/>}
    {operation.logs_truncated && <p className="tts-environment-note">{text(`显示最近 ${operation.logs?.length || 0} 行日志。`, `Showing the latest ${operation.logs?.length || 0} log lines.`)}</p>}
    {active && operation.cancellable && <button type="button" className="ui-btn ui-btn-sm" disabled={disabled || operation.phase === 'cancelling'} onClick={onCancel}>{text('取消', 'Cancel')}</button>}
    {['failed', 'cancelled'].includes(operation.status) && <button type="button" className="ui-btn ui-btn-sm" disabled={disabled} onClick={onRetry}>{text('重试', 'Retry')}</button>}
  </InstallationOperation>;
}
