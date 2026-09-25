import React from 'react';
import { createPortal } from 'react-dom';
import { apiClient } from '../api/client';
import { useWorkspaceText } from '../utils/workspaceText';
import { formatApiError } from '../utils/errors';
import StudioSelect from './StudioSelect';
import ServiceControls from './ServiceControls';
import InstallationOperation, { InstallationLog, InstallationProgress } from './InstallationOperation';

type Operation = { plan?: { name: string; from_version?: string | null; version?: string | null; minimum_free_bytes?: number; not_copied?: string[] }[]; id: string; build_id: string; status: string; phase: string; logs: string[]; error: string | null; environment_id: string | null; dismissed_at: number | null };
type Snapshot = { builds: { id: string; label: string; supported: boolean; reason: string | null; recommended: boolean; backend: string }[]; operations: Operation[]; current_python: string; selected_environment: string | null; disk_free_bytes: number; minimum_free_bytes?: number; optional_extensions?: string[] };
const active = (op: Operation) => ['planning', 'installing', 'verifying'].includes(op.status);

export default function TorchEnvironmentPanel({ disabled = false, operationsTarget, onOperationsVisible, showAttentionExtensions = true }: {
  disabled?: boolean; operationsTarget?: HTMLElement | null; onOperationsVisible?: (visible: boolean) => void; showAttentionExtensions?: boolean;
}) {
  const text = useWorkspaceText();
  const [state, setState] = React.useState<Snapshot | null>(null);
  const [choice, setChoice] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');
  const sessionOperations = React.useRef(new Set<string>());
  const refresh = React.useCallback(async () => {
    const data = await apiClient.get<Snapshot>('/environment/torch', { silent: true });
    for (const op of data.operations) if (active(op) || op.status === 'ready') sessionOperations.current.add(op.id);
    setState(data); setError('');
    const installed = data.operations.filter(op => op.status === 'completed' && op.environment_id);
    const current = installed.find(op => op.environment_id === data.selected_environment);
    const availableBuild = (id?: string) => data.builds.some(build => build.id === id && build.supported) ? id : undefined;
    setChoice(old => old || availableBuild(current?.build_id) || availableBuild(installed[0]?.build_id)
      || data.builds.find(b => b.supported && b.recommended)?.id || data.builds.find(b => b.supported)?.id || '');
  }, []);
  React.useEffect(() => { void refresh().catch(e => setError(formatApiError(e))); }, [refresh]);
  React.useEffect(() => {
    if (!state?.operations.some(active)) return;
    const timer = window.setInterval(() => void refresh().catch(e => setError(formatApiError(e))), 1500);
    return () => window.clearInterval(timer);
  }, [state, refresh]);
  const act = async (url: string, body = {}) => {
    setBusy(true); setError('');
    try {
      const op = await apiClient.post<Operation>(url, body, { silent: true });
      if (op?.id) sessionOperations.current.add(op.id);
      await refresh();
    } catch (e) { setError(formatApiError(e)); }
    finally { setBusy(false); }
  };
  const locked = disabled || busy || !!state?.operations.some(active);
  const phases: Record<string, [string, string]> = {
    planning: ['检查安装条件', 'Checking installation requirements'],
    creating_environment: ['创建独立环境', 'Creating an isolated environment'],
    installing_pytorch: ['下载并安装 PyTorch', 'Downloading and installing PyTorch'],
    installing_dependencies: ['安装训练依赖', 'Installing training dependencies'],
    verifying: ['检查新环境是否可用', 'Checking the new environment'],
    ready_to_restart: ['安装完成，重启后可使用', 'Installed; restart to use'],
    failed: ['安装失败', 'Installation failed'], cancelled: ['已取消', 'Cancelled'],
    review: ['可以开始安装', 'Ready to install'],
  };
  const operations = state?.operations.filter(op => !op.dismissed_at && (
    active(op) || op.status === 'ready' || sessionOperations.current.has(op.id)
  )) || [];
  // Finished logs belong to this visit. Installed environments remain selectable
  // without reviving every previous installation as an unfinished task.
  const installedChoices = state?.operations.filter(op => op.build_id === choice && op.status === 'completed' && op.environment_id) || [];
  const installedChoice = installedChoices.find(op => op.environment_id === state?.selected_environment) || installedChoices[0];
  const installedChoiceActive = !!installedChoice && installedChoice.environment_id === state?.selected_environment;
  const selectedBackend = state?.builds.find(build => build.id === choice)?.backend;
  const minimumSpaceGiB = selectedBackend?.startsWith('cu') ? 12 : 8;
  const showInstalledChoice = !!installedChoice && !operations.some(op => op.id === installedChoice.id);
  const buildLabel = (build: Snapshot['builds'][number]) => {
    const installed = state?.operations.filter(op => op.build_id === build.id && op.status === 'completed' && op.environment_id) || [];
    const status = installed.some(op => op.environment_id === state?.selected_environment) ? text(' · 当前使用', ' · Active')
      : installed.length ? text(' · 已安装', ' · Installed') : '';
    return build.label + status + (build.reason?.startsWith('requires_driver') ? text(` · 需 NVIDIA ${build.reason.split('_').at(-1)}+ 驱动`, ` · needs NVIDIA ${build.reason.split('_').at(-1)}+ driver`) : '');
  };
  React.useEffect(() => { onOperationsVisible?.(operations.length > 0); }, [onOperationsVisible, operations.length]);
  React.useEffect(() => () => onOperationsVisible?.(false), [onOperationsVisible]);
  const operationCards = operations.map(op => <InstallationOperation key={op.id}
    title={state?.builds.find(build => build.id === op.build_id)?.label || `PyTorch ${op.build_id}`}
    status={op.status === 'completed' && !!op.environment_id && state?.selected_environment === op.environment_id ? text('已启用', 'Active') : phases[op.phase] ? text(...phases[op.phase]) : phases[op.status] ? text(...phases[op.status]) : text('正在处理', 'Processing')}
    busy={active(op)} failed={op.status === 'failed'}>
    {active(op) && <InstallationProgress label={text('PyTorch 安装进度', 'PyTorch installation progress')}/>}
    {op.status === 'ready' && op.plan && <dl className="space-y-1 text-xs">{op.plan.filter(item => ['torch', 'torchvision'].includes(item.name)).map(item => <div key={item.name} className="flex flex-wrap gap-2"><dt>{item.name === 'torch' ? 'PyTorch' : 'TorchVision'}</dt><dd>{item.from_version || text('未安装', 'Not installed')} → {item.version}</dd></div>)}</dl>}
    {op.error && <p role="alert" className="whitespace-pre-wrap break-words text-xs text-red-600 dark:text-red-300">{op.error}</p>}
    {op.logs.length > 0 && <InstallationLog label={text('PyTorch 安装日志', 'PyTorch installation log')} logs={op.logs}/>}
    {op.status === 'ready' && <div className="flex flex-wrap gap-2"><button type="button" className="ui-btn ui-btn-primary" disabled={locked} onClick={() => void act(`/environment/torch/operations/${op.id}/apply`)}>{text('安装到独立环境', 'Install in an isolated environment')}</button><button type="button" className="ui-btn" disabled={busy} onClick={() => void act(`/environment/torch/operations/${op.id}/cancel`)}>{text('取消安装', 'Cancel installation')}</button></div>}
    {op.status === 'completed' && op.environment_id && state?.selected_environment !== op.environment_id && <ServiceControls environmentId={op.environment_id} onRestarted={() => void refresh()}/>}
    {active(op) && <div className="flex flex-wrap gap-2"><button type="button" className="ui-btn" disabled={busy} onClick={() => void act(`/environment/torch/operations/${op.id}/cancel`)}>{text('取消安装', 'Cancel installation')}</button></div>}
  </InstallationOperation>);
  return <>
    <section id="environment-torch" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><h2>{text('PyTorch 版本', 'PyTorch version')}</h2></div>
      <p className="settings-note">{showAttentionExtensions ? text('所选版本会安装到独立环境，检查通过后重启使用。原环境保留；xFormers、FlashAttention 等扩展需为新版本重新安装。', 'Install the selected version in an isolated environment, check it, then restart to use it. The original environment is retained; compiled extensions need matching installations.') : text('所选版本会安装到独立环境，检查通过后重启使用，原环境保留。', 'Install the selected version in an isolated environment, check it, then restart to use it. The original environment is retained.')}</p>
      <div className="settings-field"><label>{text('选择版本与计算后端', 'Version and compute backend')}</label><div className="settings-field-control"><StudioSelect aria-label={text('选择 PyTorch 版本', 'Choose PyTorch version')} disabled={locked || !state} value={choice} onValueChange={setChoice} options={(state?.builds || []).filter(b => b.supported || b.reason?.startsWith('requires_driver')).map(b => ({ value: b.id, label: buildLabel(b), disabled: !b.supported }))}/></div></div>
      {showInstalledChoice && !installedChoiceActive && <div className="space-y-2" data-testid="installed-torch-environment">
        <p className="settings-note">{text('此版本已安装，可直接切换，无需重新下载。', 'This version is installed. Switch to it without downloading again.')}</p>
        <ServiceControls key={installedChoice.environment_id} environmentId={installedChoice.environment_id!} disabled={locked} onRestarted={() => void refresh()}/>
      </div>}
      {state && <p className="settings-note">{text(`可用空间 ${(state.disk_free_bytes / 1024 ** 3).toFixed(1)} GiB；${selectedBackend ? `所选环境至少预留 ${minimumSpaceGiB} GiB。` : '选择版本后检查所需空间。'}`, `Available space: ${(state.disk_free_bytes / 1024 ** 3).toFixed(1)} GiB. ${selectedBackend ? `Reserve at least ${minimumSpaceGiB} GiB for the selected environment.` : 'Choose a build to check required space.'}`)}</p>}
      {disabled && <p className="settings-note">{text('当前任务完成后可安装运行环境。', 'Finish the current task before installing an environment.')}</p>}
      <div className="flex flex-wrap gap-2"><button type="button" className="ui-btn ui-btn-primary" aria-label={text('检查 PyTorch 安装条件', 'Check PyTorch installation requirements')} disabled={locked || !choice} onClick={() => void act('/environment/torch/operations', { build_id: choice })}>{text('检查安装条件', 'Check installation requirements')}</button><button type="button" className="ui-btn" disabled={busy} onClick={() => void refresh().catch(e => setError(formatApiError(e)))}>{text('刷新状态', 'Refresh status')}</button></div>
      {error && <p role="alert" className="settings-alert">{error}</p>}
    </section>
    {operationsTarget ? createPortal(operationCards, operationsTarget) : operationsTarget === undefined && operations.length > 0 ? <section className="settings-section space-y-3"><h2>{text('安装日志', 'Installation log')}</h2>{operationCards}</section> : null}
  </>;
}
