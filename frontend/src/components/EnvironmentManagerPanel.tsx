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
    python: string; python_executable: string; platform: string; machine: string; torch: string;
    cuda_runtime: string | null; cuda_available: boolean; mps_available: boolean;
    gpu_capability: number[] | null; virtual_environment: boolean;
    gpus: { name: string; telemetry_source?: string; telemetry_note?: string; driver_version?: string }[];
  };
  packages: PackageStatus[]; attention_default: string; restart_required: boolean;
  maintenance: boolean; running_jobs: boolean; probe_deferred: boolean;
}
interface PlanEntry { name: string; from_version: string | null; version: string | null; sha256?: string }
interface Operation {
  id: string; package: string; action: string; status: string; created_at: number;
  plan: PlanEntry[]; logs: string[]; error: string | null; restart_required: boolean;
}
interface Wheel { wheel_id: string; package: string; filename: string; version: string; sha256: string }
const button = 'inline-flex items-center justify-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-600 dark:hover:bg-slate-800';
const input = 'rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-900';
const busyStatus = (op: Operation) => ['planning', 'installing', 'verifying'].includes(op.status);
const backendNames: Record<string, string> = { auto: 'Auto · PyTorch SDPA', sdpa: 'PyTorch SDPA', xformers: 'xFormers', flash_attn: 'FlashAttention 2', sage: 'SageAttention' };

export function EnvironmentManagerPanel() {
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
  const observedFailures = React.useRef(new Set<string>());
  React.useEffect(() => {
    const failure = operations.find(op => op.status === 'failed' && !observedFailures.current.has(op.id));
    if (failure) { observedFailures.current.add(failure.id); setExpanded(failure.id); }
  }, [operations]);
  const [uploading, setUploading] = React.useState(false);

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
      setExpanded(op.id); setSelected(null); setVersion(''); setWheel(null);
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
  const purpose = (name: string) => ({
    torch: copy('训练基础运行时，版本受保护', 'Protected training runtime'),
    xformers: copy('训练与采样；不支持的输入回退 SDPA', 'Training and sampling; unsupported inputs use SDPA'),
    'flash-attn': copy('训练与采样；FP16/BF16 CUDA 内核', 'Training and sampling; FP16/BF16 CUDA kernels'),
    sageattention: copy('仅加速采样；训练反向传播使用 SDPA', 'Sampling only; training gradients use SDPA'),
    'nvidia-ml-py': copy('NVIDIA 显卡功率、温度与利用率采集', 'NVIDIA power, temperature and utilization telemetry'),
    tensorboard: copy('训练曲线与本地 TensorBoard 日志', 'Training curves and local TensorBoard logs'),
    wandb: copy('可选云端训练日志，启用任务时需自行登录', 'Optional cloud logs; sign in when enabling a run'),
    schedulefree: copy('Schedule-free 优化器', 'Schedule-free optimizers'),
  }[name] || '');
  const reason = (pkg: PackageStatus) => {
    if (pkg.reason === 'protected_runtime') return copy('保留当前 Torch / CUDA', 'Keep current Torch / CUDA');
    if (pkg.reason === 'requires_cuda') return copy('需要可用的 NVIDIA CUDA', 'Requires working NVIDIA CUDA');
    if (pkg.reason === 'requires_ampere') return copy('需要 Ampere 或更新的显卡', 'Requires Ampere or newer GPU');
    if (pkg.reason === 'requires_nvidia') return copy('Apple GPU 不适用', 'Not applicable to Apple GPU');
    if (!pkg.version) return copy('未安装', 'Not installed');
    if (status?.probe_deferred) return copy('任务运行中，检测已延后', 'Probe deferred while a job runs');
    if (pkg.available) return pkg.backend ? copy('已通过 CUDA 内核检测', 'CUDA kernel probe passed') : copy('可用', 'Available');
    return copy('检测失败，展开查看', 'Probe failed; expand for details');
  };
  const locked = busy || uploading || operations.some(busyStatus) || !!status?.running_jobs;
  const facts = status && [
    ['Python', `${status.runtime.python} · ${status.runtime.platform} ${status.runtime.machine}`],
    ['PyTorch', status.runtime.torch],
    [copy('Torch CUDA 运行时', 'Torch CUDA runtime'), status.runtime.cuda_runtime ? `${status.runtime.cuda_runtime} · ${status.runtime.cuda_available ? copy('可用', 'available') : copy('不可用', 'unavailable')}` : copy('当前构建无 CUDA', 'No CUDA in this build')],
    [copy('计算设备', 'Compute device'), status.runtime.gpus.map(g => g.name).join(' / ') || (status.runtime.mps_available ? 'Apple MPS' : 'CPU')],
  ];

  return <div data-testid="environment-manager"><SettingsSections sections={[
    { id: 'environment-runtime', label: copy('当前运行环境', 'Current runtime') },
    { id: 'environment-attention', label: copy('注意力默认值', 'Attention default') },
    { id: 'environment-packages', label: copy('计算与扩展', 'Compute & extensions') },
    { id: 'environment-history', label: copy('操作记录', 'Operations') },
  ]}>
    <section id="environment-runtime" data-settings-section tabIndex={-1} className="settings-section">
    <div className="settings-section-heading">
      <div><h2 className="text-base font-semibold">{copy('环境与计算后端', 'Runtime and compute backends')}</h2></div>
      <button className={button} disabled={loading || busy} onClick={() => void refresh(true)}><RefreshCw size={14} className={loading ? 'animate-spin' : ''} />{copy('重新检测', 'Refresh probes')}</button>
    </div>
    {error && <div ref={errorRef} role="alert" className="whitespace-pre-wrap rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-300">{error}</div>}
    {loading && !status && <p role="status" className="flex items-center gap-2 text-sm text-slate-500"><Loader2 size={16} className="animate-spin" />{copy('检测当前环境与已安装扩展…', 'Checking runtime and installed extensions…')}</p>}
    {status && <>
      <dl className="settings-facts">{facts?.map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-xs text-slate-500">{label}</dt><dd className="mt-1 break-words text-sm font-medium">{value}</dd></div>)}</dl>
      <details className="settings-inline-details"><summary>{copy('解释器与显卡诊断', 'Interpreter & GPU diagnostics')}</summary><p className="font-mono">{status.runtime.python_executable}</p>
        {status.runtime.gpus.some(g => g.telemetry_source) && <p>{status.runtime.gpus.map(g => `${g.name}: ${g.telemetry_source || '—'}${g.telemetry_note ? ` (${t(`hardware.${g.telemetry_note}`)})` : ''}`).join(' / ')}</p>}
      </details>
      {(status.running_jobs || status.restart_required) && <p role="status" className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-950/30 dark:text-amber-200">{status.running_jobs ? copy('训练或缓存任务正在运行。暂停并等待工作进程退出后，才能修改环境。', 'A training or cache job is running. Pause it and wait for its worker to exit before changing dependencies.') : copy('环境已发生变更，请停止并重新启动 Studio。重启前队列不会启动新任务；如安装失败，可先在这里修复或卸载。', 'The environment changed. Stop and restart Studio before new queued jobs can start. Failed packages can be repaired or removed here first.')}</p>}
    </>}
    </section>
    {status && <>
      <section id="environment-attention" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-field"><label htmlFor="environment-attention-select">{copy('新任务默认注意力', 'Default attention for new jobs')}</label><div className="settings-field-control">
          <select id="environment-attention-select" aria-label={copy('新任务默认注意力', 'Default attention for new jobs')} className={input} value={status.attention_default} disabled={busy} onChange={event => void execute(() => apiClient.put('/environment/settings', { attention_default: event.target.value }, { silent: true }))}>{['auto', 'sdpa', 'xformers', 'flash_attn', 'sage'].map(backend => <option key={backend} value={backend} disabled={!['auto', 'sdpa'].includes(backend) && !status.packages.some(p => p.backend === backend && p.available)}>{backendNames[backend]}</option>)}</select>
          <p className="settings-note">{copy('用于新配置。Auto 使用 PyTorch SDPA。', 'For new configurations. Auto uses PyTorch SDPA.')}</p>
        </div></div>
      </section>
      <section id="environment-packages" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><div><h2>{copy('计算与扩展', 'Compute & extensions')}</h2><p className="settings-note">{copy('展开条目管理版本，安装前先检查变更计划。', 'Expand an extension to manage it. Review the package plan before applying.')}</p></div></div>
      <div className="settings-dependencies">{status.packages.map(pkg => <div key={pkg.name} className="settings-dependency">
        <div className="settings-dependency-row" data-testid={`environment-package-${pkg.name}`}>
          <div><button type="button" disabled={uploading || busy} className="settings-dependency-name disabled:opacity-50" aria-expanded={selected === pkg.name} aria-controls={`environment-details-${pkg.name}`} onClick={() => { setSelected(selected === pkg.name ? null : pkg.name); setVersion(''); setWheel(null); }}>{selected === pkg.name ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{pkg.name}</button><p className="settings-dependency-purpose">{purpose(pkg.name)}</p></div>
          <span className="settings-dependency-version break-all font-mono text-xs">{pkg.version || '—'}</span>
          <span className={`settings-dependency-state text-xs ${pkg.available ? 'text-emerald-600 dark:text-emerald-400' : pkg.error && pkg.supported ? 'text-amber-600' : 'text-slate-500'}`}>{reason(pkg)}</span>
          <div className="settings-dependency-actions flex flex-wrap gap-1.5 justify-end">{pkg.name !== 'torch' && <><button className={button} disabled={locked || !pkg.supported} onClick={() => { setSelected(pkg.name); setVersion(''); setWheel(null); }}>{pkg.version ? copy('管理', 'Manage') : copy('安装', 'Install')}</button></>}<a className={button} href={pkg.docs_url} target="_blank" rel="noreferrer" aria-label={`${pkg.name} ${copy('文档', 'documentation')}`}><ExternalLink size={12} /></a></div>
        </div>
        {selected === pkg.name && <div id={`environment-details-${pkg.name}`} className="settings-dependency-detail space-y-3">
          {pkg.error && <p className="whitespace-pre-wrap break-words text-xs text-red-600 dark:text-red-300">{pkg.error}</p>}
          {pkg.name === 'torch' ? <p className="text-xs leading-relaxed text-slate-500">{copy('PyTorch 是服务和训练共同使用的基础依赖，运行时不卸载、不替换。若当前是 CPU 版或 CUDA 不可用，请停止 Studio 后，按官方安装说明修复该解释器，再重新启动；这里会重新显示实际版本与 CUDA 可用性。', 'PyTorch is shared by the service and training and is not replaced while running. If this is a CPU build or CUDA is unavailable, stop Studio, repair this interpreter using the official installation instructions, then restart and recheck.')}</p> : <>
            <p className="text-xs leading-relaxed text-slate-500">{pkg.wheel_required ? copy('此平台需要预编译 wheel。请先从扩展发布者取得与上方 Python、Torch、CUDA 一致的文件，再上传校验。没有匹配文件时，请使用 SDPA；不会自动尝试源码编译。', 'This platform needs a prebuilt wheel. Obtain a wheel matching the Python, Torch and CUDA shown above from the extension publisher, then upload it for validation. Use SDPA if no matching wheel exists; source builds are never attempted.') : copy('默认只查找兼容的二进制 wheel，也可填写精确版本或上传本地 wheel。安装计划会列出所需依赖；如果需要替换基础运行时，计划会失败并说明冲突。', 'Searches compatible binary wheels only. Optionally choose an exact version or upload a local wheel. The plan lists required dependencies and fails if it would replace the protected runtime.')}</p>
            <div className="flex flex-wrap items-center gap-2"><label className="flex items-center gap-2 text-xs">{copy('版本', 'Version')}<input className={`${input} w-40`} aria-label={`${pkg.name} ${copy('版本', 'version')}`} placeholder={copy('自动匹配兼容版本', 'Compatible version')} value={version} onChange={event => setVersion(event.target.value)} disabled={locked || !!wheel} /></label>
              <label className={`${button} cursor-pointer ${locked ? 'pointer-events-none opacity-40' : ''}`}><Upload size={13} />{uploading ? copy('上传并校验…', 'Uploading and checking…') : copy('上传 wheel', 'Upload wheel')}<input type="file" accept=".whl" className="sr-only" aria-label={`${pkg.name} wheel`} disabled={locked} onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file, pkg.name); event.target.value = ''; }} /></label>
              <button className={`${button} border-blue-600 bg-blue-600 text-white hover:bg-blue-700`} disabled={locked || !pkg.supported || pkg.wheel_required && !wheel} onClick={() => void plan(pkg.name, 'install')}>{copy('检查安装计划', 'Review install plan')}</button>
              {pkg.version && <><button className={button} disabled={locked || !pkg.supported || pkg.wheel_required && !wheel} onClick={() => void plan(pkg.name, 'repair')}>{copy('修复当前版本', 'Repair current version')}</button><button className={button} disabled={locked} onClick={() => void plan(pkg.name, 'uninstall')}>{copy('卸载', 'Uninstall')}</button></>}
            </div>
            {wheel && <p className="flex items-center gap-2 break-all text-xs text-emerald-600"><Check size={13} />{wheel.filename}<button className="text-slate-500" aria-label={copy('清除 wheel', 'Clear wheel')} onClick={() => { setWheel(null); setVersion(''); }}><X size={13} /></button></p>}
          </>}
        </div>}
      </div>)}</div></section>
    </>}
    <section id="environment-history" data-settings-section tabIndex={-1} className="settings-section space-y-3" data-testid="environment-operations"><h2>{copy('环境操作记录', 'Environment operations')}</h2>{operations.length === 0 && <p className="settings-note">{copy('暂无环境变更。', 'No environment changes yet.')}</p>}{operations.slice(0, 12).map(op => <div key={op.id} className="rounded-lg border border-slate-200 dark:border-slate-700">
      <button aria-expanded={expanded === op.id} className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left text-xs" onClick={() => setExpanded(expanded === op.id ? null : op.id)}>{expanded === op.id ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{busyStatus(op) && <Loader2 size={13} className="animate-spin" />}<span className="font-medium">{op.package}</span><span>{op.action === 'uninstall' ? copy('卸载', 'Uninstall') : op.action === 'repair' ? copy('修复', 'Repair') : copy('安装', 'Install')}</span><span className={`ml-auto ${op.status === 'failed' ? 'text-red-600' : 'text-slate-500'}`}>{statusLabel(op.status)}</span></button>
      {expanded === op.id && <div className="space-y-2 border-t border-slate-100 px-3 py-3 dark:border-slate-700">
        {op.plan.length > 0 && <div className="space-y-1 text-xs">{op.plan.map(item => <p key={item.name} className="break-words"><span className="font-mono">{item.name}</span> · {item.from_version || copy('未安装', 'not installed')} → <strong>{item.version || copy('移除', 'remove')}</strong></p>)}</div>}
        {op.error && <p role="alert" className="whitespace-pre-wrap break-words text-xs text-red-600 dark:text-red-300">{op.error}</p>}
        {op.status === 'ready' && <div className="flex flex-wrap items-center gap-2"><button className={`${button} border-blue-600 bg-blue-600 text-white hover:bg-blue-700`} disabled={locked} onClick={() => void execute(() => apiClient.post(`/environment/operations/${op.id}/apply`, {}, { silent: true }))}>{copy('确认并执行此计划', 'Apply this reviewed plan')}</button><span className="text-xs text-slate-500">{copy('执行后需要重启 Studio。', 'Restart Studio after applying.')}</span></div>}
        {['planning', 'ready'].includes(op.status) && <button className={button} disabled={busy} onClick={() => void execute(() => apiClient.post(`/environment/operations/${op.id}/cancel`, {}, { silent: true }))}>{copy('取消计划', 'Cancel plan')}</button>}
        {['installing', 'verifying'].includes(op.status) && <p className="text-xs text-slate-500">{copy('正在执行已确认的计划。为避免留下半安装状态，此阶段不能中断。', 'Applying the reviewed plan. This stage cannot be interrupted because it may leave a partial installation.')}</p>}
        {op.logs.length > 0 && <pre aria-label={copy('安装日志', 'Installer logs')} className="max-h-56 overflow-auto whitespace-pre-wrap break-all rounded-md bg-slate-950 p-3 font-mono text-[11px] leading-5 text-slate-200">{op.logs.join('\n')}</pre>}
      </div>}
    </div>)}</section>
  </SettingsSections></div>;
}
