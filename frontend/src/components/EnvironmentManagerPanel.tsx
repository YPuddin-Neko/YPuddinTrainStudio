import TorchEnvironmentPanel from './TorchEnvironmentPanel';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { Check, ChevronDown, ChevronRight, ExternalLink, Loader2, RefreshCw, Upload, X } from 'lucide-react';
import { apiClient } from '../api/client';
import { formatApiError } from '../utils/errors';
import { SettingsSections } from '../pages/Settings/SettingsSections';

interface PackageStatus {
  name: string; version: string | null; backend: string | null; docs_url: string;
  supported: boolean; reason: string; importable: boolean; kernel_tested: boolean;
  available: boolean; wheel_required: boolean; error: string | null;
}
interface EnvironmentStatus {
  runtime: {
    environment_profile?: string;
    python: string; python_executable: string; platform: string; machine: string; torch: string;
    cuda_runtime: string | null; cuda_available: boolean; mps_available: boolean;
    gpu_capability: number[] | null; virtual_environment: boolean;
    cuda_device_count?: number; distributed_available?: boolean; nccl_available?: boolean;
    cuda_applicable?: boolean; nccl_applicable?: boolean; multi_gpu_training?: boolean; training_device_policy?: string;
    gpus: { name: string; device?: string | null; cuda_available?: boolean; memory_scope?: string; mem_total_mb?: number; telemetry_source?: string; telemetry_note?: string; driver_version?: string }[];
  };
  packages: PackageStatus[]; attention_default: string; restart_required: boolean;
  maintenance: boolean; running_jobs: boolean; probe_deferred: boolean;
}
interface PlanEntry { name: string; from_version: string | null; version: string | null; sha256?: string }
interface Operation {
  id: string; package: string; action: string; status: string; created_at: number;
  dismissed_at?: number | null; plan: PlanEntry[]; logs: string[]; error: string | null; restart_required: boolean;
}
interface Wheel { wheel_id: string; package: string; filename: string; version: string; sha256: string }
const button = 'inline-flex items-center justify-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-600 dark:hover:bg-slate-800';
const input = 'rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-900';
const busyStatus = (op: Operation) => ['planning', 'installing', 'verifying'].includes(op.status);
const managedPackages = new Set(['xformers', 'flash-attn']);

export function EnvironmentManagerPanel({ focusPackage }: { focusPackage?: string } = {}) {
  const { i18n, t } = useTranslation();
  const en = i18n.resolvedLanguage?.startsWith('en');
  const copy = (zh: string, english: string) => en ? english : zh;
  const [status, setStatus] = React.useState<EnvironmentStatus | null>(null);
  const [operations, setOperations] = React.useState<Operation[]>([]);
  const operationsRef = React.useRef<Operation[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');
  const errorRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => { if (error) errorRef.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' }); }, [error]);
  const [selected, setSelected] = React.useState<string | null>(null);
  const [version, setVersion] = React.useState('');
  const [wheel, setWheel] = React.useState<Wheel | null>(null);
  const [expanded, setExpanded] = React.useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = React.useState(false);
  const [createdOperations, setCreatedOperations] = React.useState<Set<string>>(new Set());
  const observedOperations = React.useRef(new Set<string>());
  const visibleOperations = operations.filter((op, index) => !op.dismissed_at && (
    busyStatus(op) || op.status === 'ready' && managedPackages.has(op.package)
    || index === 0 && op.status === 'failed'
    || createdOperations.has(op.id) && op.status === 'completed'
  ));
  React.useEffect(() => {
    const current = operations.find((op, index) => !op.dismissed_at && !observedOperations.current.has(`${op.id}:${op.status}`) && (
      busyStatus(op) || op.status === 'ready' && managedPackages.has(op.package) || index === 0 && op.status === 'failed'
    ));
    if (current) { observedOperations.current.add(`${current.id}:${current.status}`); setExpanded(current.id); }
  }, [operations]);
  const historyOperations = operations.filter(op => !visibleOperations.some(current => current.id === op.id));
  const shownOperations = historyOpen ? operations : visibleOperations;
  const [uploading, setUploading] = React.useState(false);
  const focusAvailable = !!focusPackage && managedPackages.has(focusPackage) && !!status?.packages.some(pkg => pkg.name === focusPackage);
  React.useEffect(() => {
    if (!focusAvailable || !focusPackage) return;
    setSelected(focusPackage);
    document.getElementById(`environment-package-${focusPackage}`)?.scrollIntoView?.({ block: 'start' });
  }, [focusAvailable, focusPackage]);

  const refresh = React.useCallback(async (probe = false) => {
    setLoading(true);
    try {
      const [state, tasks] = await Promise.all([
        apiClient.get<EnvironmentStatus>('/environment', { params: { refresh: probe }, silent: true }),
        apiClient.get<Operation[]>('/environment/operations', { silent: true }),
      ]);
      setStatus(state); setOperations(tasks); operationsRef.current = tasks; setError('');
    } catch (err) { setError(formatApiError(err)); }
    finally { setLoading(false); }
  }, []);
  React.useEffect(() => { void refresh(); }, [refresh]);
  React.useEffect(() => {
    const timer = window.setInterval(() => {
      void apiClient.get<Operation[]>('/environment/operations', { silent: true }).then(next => {
        const finished = operationsRef.current.some(p => busyStatus(p) && !next.some(n => n.id === p.id && busyStatus(n)));
        operationsRef.current = next; setOperations(next);
        if (finished) void refresh();
      }).catch(err => setError(formatApiError(err)));
    }, 1500);
    // A job can start or finish in another tab. Refresh admission state as well as
    // operation logs; the server remains authoritative at both Plan and Apply.
    const stateTimer = window.setInterval(() => {
      void apiClient.get<EnvironmentStatus>('/environment', { silent: true }).then(setStatus).catch(err => setError(formatApiError(err)));
    }, 15000);
    return () => { window.clearInterval(timer); window.clearInterval(stateTimer); };
  }, [refresh]);

  const execute = async (fn: () => Promise<unknown>) => {
    setBusy(true); setError('');
    try { await fn(); await refresh(); }
    catch (err) { setError(formatApiError(err)); }
    finally { setBusy(false); }
  };
  const plan = async (name: string, action: string) => {
    await execute(async () => {
      const op = await apiClient.post<Operation>('/environment/operations', { package: name, action, ...(action === 'install' && version.trim() ? { version: version.trim() } : {}), ...(action !== 'uninstall' && wheel ? { wheel_id: wheel.wheel_id } : {}) }, { silent: true });
      setCreatedOperations(previous => new Set(previous).add(op.id)); setExpanded(op.id); setSelected(null); setVersion(''); setWheel(null);
    });
  };
  const upload = async (file: File, name: string) => {
    setUploading(true); setError(''); setWheel(null);
    try {
      const form = new FormData(); form.append('file', file);
      const uploaded = await apiClient.post<Wheel>('/environment/wheels', form, { silent: true });
      if (uploaded.package !== name) throw new Error(copy(`此 wheel 属于 ${uploaded.package}，请选择 ${name} 的文件。`, `This wheel contains ${uploaded.package}; select a ${name} wheel.`));
      setWheel(uploaded); setVersion(uploaded.version);
    } catch (err) { setError(formatApiError(err)); }
    finally { setUploading(false); }
  };
  const statusLabel = (name: string) => ({ planning: copy('检查兼容性', 'Checking compatibility'), ready: copy('等待确认', 'Review required'), installing: copy('下载并安装', 'Downloading and applying'), verifying: copy('验证环境', 'Verifying environment'), completed: copy('已完成', 'Completed'), failed: copy('失败', 'Failed'), cancelled: copy('已取消', 'Cancelled') }[name] || name);
  const packageLabel = (name: string) => ({ xformers: 'xFormers', 'flash-attn': 'FlashAttention 2' }[name] || name);
  const purpose = (name: string) => ({
    xformers: copy('训练与采样加速', 'Training and sampling acceleration'),
    'flash-attn': copy('FP16 / BF16 训练与采样加速', 'FP16 / BF16 training and sampling acceleration'),
  }[name] || '');
  const reason = (pkg: PackageStatus) => {
    if (pkg.reason === 'requires_cuda') return copy('需要可用的 NVIDIA CUDA', 'Requires working NVIDIA CUDA');
    if (pkg.reason === 'requires_ampere') return copy('需要 Ampere 或更新的显卡', 'Requires Ampere or newer GPU');
    if (pkg.reason === 'requires_nvidia') return copy('Apple GPU 不适用', 'Not applicable to Apple GPU');
    if (!pkg.version) return copy('未启用 · 可选', 'Not enabled · optional');
    if (status?.probe_deferred) return copy('任务运行中，检测已延后', 'Probe deferred while a job runs');
    if (pkg.available) return pkg.backend ? copy('已通过 CUDA 内核检测', 'CUDA kernel probe passed') : copy('可用', 'Available');
    return copy('检测失败，展开查看', 'Probe failed; expand for details');
  };
  const textUnavailable = () => copy(' · 驱动可见，当前 PyTorch 不可用',' · visible to driver, unavailable to current PyTorch');
  const locked = busy || uploading || operations.some(busyStatus) || !!status?.running_jobs;
  const profile = status?.runtime.environment_profile || 'legacy';
  const profileLabel = ({ 'windows-cuda': 'Windows CUDA', 'linux-cuda': 'Linux CUDA', 'macos-mps': 'macOS MPS', 'windows-cpu': 'Windows CPU', 'linux-cpu': 'Linux CPU', 'macos-cpu': 'macOS CPU', legacy: copy('旧版环境', 'Legacy environment') } as Record<string, string>)[profile] || copy('未知环境', 'Unknown environment');
  const cpuProfile = profile.endsWith('-cpu');
  const computeBackend = cpuProfile ? 'CPU' : status?.runtime.cuda_available
    ? `CUDA ${status.runtime.cuda_runtime || ''}`.trim()
    : status?.runtime.mps_available ? 'Apple MPS' : 'CPU';
  const facts = status && [
    [copy('部署环境', 'Deployment environment'), profileLabel],
    ['Python', status.runtime.python],
    ['PyTorch', status.runtime.torch],
    ...(status.runtime.mps_available ? [] : [[copy('CUDA 版本', 'CUDA version'), status.runtime.cuda_runtime || copy('CPU 版本', 'CPU build')], [copy('NVIDIA 显卡计算', 'NVIDIA GPU compute'), status.runtime.cuda_available ? copy('可用', 'Available') : copy('未启用', 'Not enabled')]]),
    [copy('计算后端', 'Compute backend'), computeBackend],
    [copy('设备', 'Device'), cpuProfile ? 'CPU' : status.runtime.gpus.map(g => g.name).join(' / ') || (status.runtime.mps_available ? 'Apple GPU' : 'CPU')],
  ];

  return <div data-testid="environment-manager"><SettingsSections sections={[
    { id: 'environment-runtime', label: copy('当前环境', 'Current runtime') },
    { id: 'environment-torch', label: copy('PyTorch 版本', 'PyTorch version') },
    { id: 'environment-attention', label: copy('注意力加速', 'Attention acceleration') },
    ...(operations.length ? [{ id: 'environment-installation', label: copy('安装状态', 'Installation status') }] : []),
  ]}>
    <section id="environment-runtime" data-settings-section tabIndex={-1} className="settings-section">
    <div className="settings-section-heading">
      <div><h2 className="text-base font-semibold">{copy('环境与计算后端', 'Runtime and compute backends')}</h2></div>
      <button className={button} disabled={loading || busy} onClick={() => void refresh(true)}><RefreshCw size={14} className={loading ? 'animate-spin' : ''} />{copy('重新检测', 'Refresh probes')}</button>
    </div>
    {error && <div ref={errorRef} role="alert" className="whitespace-pre-wrap rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-300">{error}</div>}
    {loading && !status && <p role="status" className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400"><Loader2 size={16} className="animate-spin" />{copy('检测当前环境与已安装扩展…', 'Checking runtime and installed extensions…')}</p>}
    {status && <>
      <dl className="settings-facts">{facts?.map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-xs text-slate-500 dark:text-slate-400">{label}</dt><dd className="mt-1 break-words text-sm font-medium">{value}</dd></div>)}</dl>
      {status.runtime.gpus.length>0&&<ul className="settings-note" aria-label={copy('已检测设备','Detected devices')}>{status.runtime.gpus.map((gpu,index)=><li key={`${gpu.device||index}:${gpu.name}`}><strong>{gpu.device||`GPU ${index+1}`}</strong> · {gpu.name}{gpu.mem_total_mb!=null?` · ${(gpu.mem_total_mb/1024).toFixed(1)} GiB ${gpu.memory_scope==='unified_system'?copy('统一内存','unified memory'):copy('设备内存','device memory')}`:''}{gpu.cuda_available===false?textUnavailable():''}</li>)}</ul>}
      <details className="settings-inline-details"><summary>{copy('解释器与显卡诊断', 'Interpreter & GPU diagnostics')}</summary><p className="font-mono break-all">{status.runtime.python_executable}</p><p>{status.runtime.platform} · {status.runtime.machine}</p><p>{copy('这里显示当前服务实际加载的解释器。', 'This is the interpreter loaded by the running service.')}</p>
        <div className="settings-note" data-testid="environment-training-devices"><p>{copy('每个训练任务使用一张显卡；单任务多卡训练尚未接入。', 'Each training task uses one GPU; multi-GPU training for a single task is not implemented.')}</p>{!status.runtime.mps_available && <p>{copy(`PyTorch 可用 CUDA 显卡：${status.runtime.cuda_device_count ?? 0} 张`, `CUDA GPUs available to PyTorch: ${status.runtime.cuda_device_count ?? 0}`)}</p>}<p>{copy('NCCL 是 NVIDIA 多卡通信组件。', 'NCCL handles communication between NVIDIA GPUs.')} {status.runtime.mps_available || status.runtime.platform === 'Windows' ? copy('当前平台不适用，不影响单卡训练。', 'It does not apply to this platform and is not needed for single-GPU training.') : status.runtime.nccl_available ? copy('当前版本已包含。', 'Included in this build.') : copy('当前版本未包含；单卡训练不需要它。', 'Not included in this build; single-GPU training does not need it.')}</p></div>
        {status.runtime.gpus.some(g => g.telemetry_source) && <p>{status.runtime.gpus.map(g => `${g.name}: ${g.telemetry_source || '—'}${g.telemetry_note ? ` (${t(`hardware.${g.telemetry_note}`)})` : ''}`).join(' / ')}</p>}
      </details>
      {!status.runtime.cuda_available && !status.runtime.mps_available && <p className="settings-note">{status.runtime.cuda_runtime ? copy(`当前 PyTorch 含 CUDA ${status.runtime.cuda_runtime}，但无法使用 CUDA。请检查显卡驱动后重启。`, `PyTorch includes CUDA ${status.runtime.cuda_runtime}, but CUDA is unavailable. Check the GPU driver and restart.`) : copy('当前使用 CPU 计算。若需使用 NVIDIA 显卡，请通过启动器配置 CUDA 版 PyTorch。', 'Currently using CPU compute. To use an NVIDIA GPU, configure CUDA-enabled PyTorch through the launcher.')}</p>}
      {(status.running_jobs || status.restart_required) && <p role="status" className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-950/30 dark:text-amber-200">{status.running_jobs ? copy('训练或缓存任务正在运行。等待任务完成或停止、工作进程退出后，才能修改环境。', 'Training or caching is running. Wait for the task to finish or stop and its worker to exit before changing dependencies.') : copy('环境已发生变更，请停止并重新启动 Studio。重启前队列不会启动新任务；如安装失败，可先在这里修复或卸载。', 'The environment changed. Stop and restart Studio before new queued jobs can start. Failed packages can be repaired or removed here first.')}</p>}
    </>}
    </section>
    {status && <>
      <TorchEnvironmentPanel disabled={status.running_jobs || busy || operations.some(busyStatus)}/>
      <section id="environment-attention" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{copy('注意力加速', 'Attention acceleration')}</h2><p className="settings-note">{copy('默认 SDPA 即可训练；可选扩展用于 CUDA 加速。', 'SDPA is ready for training. Optional extensions provide CUDA acceleration.')}</p></div></div>
      <div className="settings-dependencies">{status.packages.filter(pkg => managedPackages.has(pkg.name)).map(pkg => <div key={pkg.name} className="settings-dependency">
        <div id={`environment-package-${pkg.name}`} className="settings-dependency-row" data-testid={`environment-package-${pkg.name}`}>
          <div><button type="button" disabled={uploading || busy} className="settings-dependency-name disabled:opacity-50" aria-expanded={selected === pkg.name} aria-controls={`environment-details-${pkg.name}`} onClick={() => { setSelected(selected === pkg.name ? null : pkg.name); setVersion(''); setWheel(null); }}>{selected === pkg.name ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{packageLabel(pkg.name)}</button><p className="settings-dependency-purpose">{purpose(pkg.name)}</p></div>
          <span className="settings-dependency-version break-all font-mono text-xs">{pkg.version || '—'}</span>
          <span className={`settings-dependency-state text-xs ${pkg.available ? 'text-emerald-700 dark:text-emerald-400' : pkg.error && pkg.supported ? 'text-amber-700 dark:text-amber-300' : 'text-slate-500 dark:text-slate-400'}`}>{reason(pkg)}</span>
          <div className="settings-dependency-actions flex flex-wrap gap-1.5 justify-end"><button className={button} disabled={locked || !pkg.supported} onClick={() => { setSelected(pkg.name); setVersion(''); setWheel(null); }}>{pkg.version ? copy('管理', 'Manage') : copy('安装', 'Install')}</button><a className={button} href={pkg.docs_url} target="_blank" rel="noreferrer" aria-label={`${pkg.name} ${copy('文档', 'documentation')}`}><ExternalLink size={12} /></a></div>
        </div>
        {selected === pkg.name && <div id={`environment-details-${pkg.name}`} className="settings-dependency-detail space-y-3">
          {pkg.error && <p className="whitespace-pre-wrap break-words text-xs text-red-600 dark:text-red-300">{pkg.error}</p>}
          <>
            <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">{pkg.wheel_required ? copy('此平台需要预编译 wheel，请上传匹配当前 Python、Torch 和 CUDA 的文件。', 'This platform needs a prebuilt wheel matching the current Python, Torch and CUDA.') : copy('自动匹配兼容版本，确认变更计划后安装。', 'Find a compatible version and review the changes before installing.')}</p>
            <div className="flex flex-wrap items-center gap-2">
              <button className={`${button} border-blue-600 bg-blue-600 text-white hover:bg-blue-700`} disabled={locked || !pkg.supported || pkg.wheel_required && !wheel} onClick={() => void plan(pkg.name, 'install')}>{copy('检查安装计划', 'Review install plan')}</button>
              {pkg.version && <><button className={button} disabled={locked || !pkg.supported || pkg.wheel_required && !wheel} onClick={() => void plan(pkg.name, 'repair')}>{copy('修复当前版本', 'Repair current version')}</button><button className={button} disabled={locked} onClick={() => void plan(pkg.name, 'uninstall')}>{copy('卸载', 'Uninstall')}</button></>}
            </div>
            <details className="settings-inline-details" open={pkg.wheel_required}><summary>{copy('手动版本与 wheel', 'Manual version and wheel')}</summary>
            <div className="flex flex-wrap items-center gap-2"><label className="flex items-center gap-2 text-xs">{copy('版本', 'Version')}<input className={`${input} w-40`} aria-label={`${pkg.name} ${copy('版本', 'version')}`} placeholder={copy('自动匹配兼容版本', 'Compatible version')} value={version} onChange={event => setVersion(event.target.value)} disabled={locked || !!wheel} /></label>
              <label className={`${button} cursor-pointer ${locked ? 'pointer-events-none opacity-40' : ''}`}><Upload size={13} />{uploading ? copy('上传并校验…', 'Uploading and checking…') : copy('上传 wheel', 'Upload wheel')}<input type="file" accept=".whl" className="sr-only" aria-label={`${pkg.name} wheel`} disabled={locked} onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file, pkg.name); event.target.value = ''; }} /></label>
            </div></details>
            {wheel && <p className="flex items-center gap-2 break-all text-xs text-emerald-700 dark:text-emerald-400"><Check size={13} />{wheel.filename}<button className="text-slate-500 dark:text-slate-400" aria-label={copy('清除 wheel', 'Clear wheel')} onClick={() => { setWheel(null); setVersion(''); }}><X size={13} /></button></p>}
          </>
        </div>}
      </div>)}</div></section>
    </>}
    {operations.length > 0 && <section id="environment-installation" data-settings-section tabIndex={-1} className="settings-section space-y-3" data-testid={shownOperations.length ? "environment-operations" : "environment-history"}><div className="settings-section-heading"><h2>{copy('安装记录', 'Installation records')}</h2>{historyOperations.length > 0 && <button type="button" className={button} aria-expanded={historyOpen} onClick={() => { setHistoryOpen(!historyOpen); setExpanded(null); }}>{historyOpen ? copy('收起历史', 'Hide history') : copy(`历史记录（${historyOperations.length}）`, `History (${historyOperations.length})`)}</button>}</div>{shownOperations.map(op => <div key={op.id} className="rounded-lg border border-slate-200 dark:border-slate-700">
      <button aria-expanded={expanded === op.id} className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left text-xs" onClick={() => setExpanded(expanded === op.id ? null : op.id)}>{expanded === op.id ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{busyStatus(op) && <Loader2 size={13} className="animate-spin" />}<span className="font-medium">{packageLabel(op.package)}</span><time className="text-slate-500 dark:text-slate-400" dateTime={new Date(op.created_at * 1000).toISOString()}>{new Date(op.created_at * 1000).toLocaleString(en ? 'en' : 'zh-CN')}</time><span>{op.action === 'uninstall' ? copy('卸载', 'Uninstall') : op.action === 'repair' ? copy('修复', 'Repair') : copy('安装', 'Install')}</span><span className={`ml-auto ${op.status === 'failed' ? 'text-red-600 dark:text-red-300' : 'text-slate-500 dark:text-slate-400'}`}>{statusLabel(op.status)}</span></button>
      {expanded === op.id && <div className="space-y-2 border-t border-slate-100 px-3 py-3 dark:border-slate-700">
        {op.plan.length > 0 && <div className="space-y-1 text-xs">{op.plan.map(item => <p key={item.name} className="break-words"><span className="font-mono">{item.name}</span> · {item.from_version || copy('未安装', 'not installed')} → <strong>{item.version || copy('移除', 'remove')}</strong></p>)}</div>}
        {op.error && <p role="alert" className="whitespace-pre-wrap break-words text-xs text-red-600 dark:text-red-300">{op.error}</p>}
        {op.status === 'ready' && <div className="flex flex-wrap items-center gap-2"><button className={`${button} border-blue-600 bg-blue-600 text-white hover:bg-blue-700`} disabled={locked} onClick={() => void execute(() => apiClient.post(`/environment/operations/${op.id}/apply`, {}, { silent: true }))}>{copy('确认并执行此计划', 'Apply this reviewed plan')}</button><span className="text-xs text-slate-500 dark:text-slate-400">{copy('执行后需要重启 Studio。', 'Restart Studio after applying.')}</span></div>}
        {['planning', 'ready'].includes(op.status) && <button className={button} disabled={busy} onClick={() => void execute(() => apiClient.post(`/environment/operations/${op.id}/cancel`, {}, { silent: true }))}>{copy('取消计划', 'Cancel plan')}</button>}
        {['installing', 'verifying'].includes(op.status) && <p className="text-xs text-slate-500 dark:text-slate-400">{copy('正在执行已确认的计划。为避免留下半安装状态，此阶段不能中断。', 'Applying the reviewed plan. This stage cannot be interrupted because it may leave a partial installation.')}</p>}
        {op.logs.length > 0 && <pre aria-label={copy('安装日志', 'Installer logs')} className="max-h-56 overflow-auto whitespace-pre-wrap break-all rounded-md bg-slate-950 p-3 font-mono text-[11px] leading-5 text-slate-200">{op.logs.join('\n')}</pre>}
      </div>}
      {!op.dismissed_at && ['completed', 'failed', 'cancelled'].includes(op.status) && <div className="px-3 pb-2"><button type="button" className="text-xs text-slate-500 dark:text-slate-400 underline" disabled={busy} onClick={() => void execute(() => apiClient.post(`/environment/operations/${op.id}/dismiss`, {}, { silent: true }))}>{copy('关闭结果', 'Dismiss result')}</button></div>}
    </div>)}</section>}
  </SettingsSections></div>;
}
