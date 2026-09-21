import TorchEnvironmentPanel from './TorchEnvironmentPanel';
import InstallationOperation, { InstallationLog, InstallationProgress } from './InstallationOperation';
import DtkWheelPicker, { type DtkWheel } from './DtkWheelPicker';
import DtkRuntimePanel from './DtkRuntimePanel';
import WindowsAttentionWheelPicker, { type WindowsAttentionWheel } from './WindowsAttentionWheelPicker';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { Check, ChevronDown, ChevronRight, Download, ExternalLink, Loader2, RefreshCw, Upload, X } from 'lucide-react';
import { apiClient } from '../api/client';
import { formatApiError } from '../utils/errors';
import { formatBytes, formatEta } from '../utils/format';
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
    hip_runtime?: string | null; compute_backend?: string;
    gpu_capability: number[] | null; virtual_environment: boolean;
    cuda_device_count?: number; distributed_available?: boolean; nccl_available?: boolean;
    cuda_applicable?: boolean; nccl_applicable?: boolean; multi_gpu_training?: boolean; training_device_policy?: string;
    gloo_available?: boolean; multi_gpu_backend?: 'nccl' | 'gloo' | null; multi_gpu_probe_required?: boolean;
    gpus: { name: string; device?: string | null; cuda_available?: boolean; memory_scope?: string; mem_total_mb?: number; telemetry_source?: string; telemetry_note?: string; driver_version?: string }[];
  };
  packages: PackageStatus[]; attention_default: string; restart_required: boolean;
  maintenance: boolean; running_jobs: boolean; probe_deferred: boolean;
  sdpa?: { status: 'passed' | 'failed' | 'not_tested'; reason: string | null; error: string | null; detail: string | null; checked_at: number | null; device: string | null; device_name: string | null } | null;
}
interface PlanEntry { name: string; from_version: string | null; version: string | null; sha256?: string }
interface Operation {
  id: string; package: string; action: string; status: string; created_at: number;
  dismissed_at?: number | null; plan: PlanEntry[]; logs: string[]; error: string | null; restart_required: boolean;
  phase?: string; downloaded_bytes?: number; total_bytes?: number | null; bytes_per_second?: number | null; eta_seconds?: number | null;
}
interface Wheel { wheel_id: string; package: string; filename: string; version: string; sha256: string }
const button = 'inline-flex items-center justify-center gap-1.5 rounded-md border border-slate-200 px-2.5 py-1.5 text-xs hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-600 dark:hover:bg-slate-800';
const input = 'rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-900';
const busyStatus = (op: Operation) => ['planning', 'installing', 'verifying'].includes(op.status);
const managedPackages = new Set(['xformers', 'flash-attn']);

function DownloadProgress({ operation, copy }: {operation: Operation; copy: (zh: string, en: string) => string}) {
  const downloading = operation.status === 'planning' && operation.phase === 'download';
  const downloaded = Math.max(0, operation.downloaded_bytes || 0);
  const total = operation.total_bytes && operation.total_bytes > 0 ? operation.total_bytes : null;
  if (!downloading && !downloaded) return null;
  const percent = total ? Math.min(100, downloaded / total * 100) : undefined;
  return <div className="space-y-2" data-testid="environment-download-progress">
    <div className="flex flex-wrap justify-between gap-2 text-xs"><strong>{downloading ? copy('下载适配包', 'Downloading build') : copy('已下载', 'Downloaded')}</strong><span className="tabular-nums">{formatBytes(downloaded)}{total ? ` / ${formatBytes(total)}` : ''}</span></div>
    <InstallationProgress label={copy('适配包下载进度', 'Build download progress')} percent={percent ?? (downloading ? undefined : 0)}/>
    {downloading && <p className="flex flex-wrap gap-4 text-xs text-slate-500 dark:text-slate-400 tabular-nums"><span>{operation.bytes_per_second && operation.bytes_per_second > 0 ? `${formatBytes(operation.bytes_per_second)}/s` : copy('正在计算速度…', 'Measuring speed…')}</span><span>{copy('预计剩余', 'Remaining')} {formatEta(operation.eta_seconds)}</span></p>}
  </div>;
}

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
  const wheelInputRef = React.useRef<HTMLInputElement>(null);
  React.useEffect(() => { if (error) errorRef.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' }); }, [error]);
  const [selected, setSelected] = React.useState<string | null>(null);
  const [version, setVersion] = React.useState('');
  const [wheel, setWheel] = React.useState<Wheel | null>(null);
  const [vendorWheel, setVendorWheel] = React.useState<DtkWheel | WindowsAttentionWheel | null>(null);
  const [expanded, setExpanded] = React.useState<string | null>(null);
  const [torchOperationsTarget, setTorchOperationsTarget] = React.useState<HTMLDivElement | null>(null);
  const [torchOperationsVisible, setTorchOperationsVisible] = React.useState(false);
  const [createdOperations, setCreatedOperations] = React.useState<Set<string>>(new Set());
  const observedOperations = React.useRef(new Set<string>());
  const visibleOperations = operations.filter(op => !op.dismissed_at && (
    busyStatus(op) || op.status === 'ready' && managedPackages.has(op.package)
    || createdOperations.has(op.id) && ['completed', 'failed', 'cancelled'].includes(op.status)
  ));
  React.useEffect(() => {
    setCreatedOperations(previous => {
      const activeIds = operations.filter(op => busyStatus(op) || op.status === 'ready').map(op => op.id);
      return activeIds.some(id => !previous.has(id)) ? new Set([...previous, ...activeIds]) : previous;
    });
    const current = operations.find(op => !op.dismissed_at && !observedOperations.current.has(`${op.id}:${op.status}`) && (
      busyStatus(op) || op.status === 'ready' && managedPackages.has(op.package) || createdOperations.has(op.id) && op.status === 'failed'
    ));
    if (current) { observedOperations.current.add(`${current.id}:${current.status}`); setExpanded(current.id); }
  }, [operations, createdOperations]);
  const [uploading, setUploading] = React.useState(false);
  const focusAvailable = !!focusPackage && managedPackages.has(focusPackage) && !!status?.packages.some(pkg => pkg.name === focusPackage);
  React.useEffect(() => {
    if (!focusAvailable || !focusPackage) return;
    setSelected(focusPackage);
    setVersion(''); setWheel(null); setVendorWheel(null);
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
      const op = await apiClient.post<Operation>('/environment/operations', { package: name, action, ...(action === 'install' && version.trim() ? { version: version.trim() } : {}), ...(action !== 'uninstall' && wheel ? { wheel_id: wheel.wheel_id } : {}), ...(action !== 'uninstall' && vendorWheel ? {vendor_wheel_id: vendorWheel.id} : {}) }, { silent: true });
      setCreatedOperations(previous => new Set(previous).add(op.id)); setExpanded(op.id); setSelected(null); setVersion(''); setWheel(null); setVendorWheel(null);
    });
  };
  const upload = async (file: File, name: string) => {
    setUploading(true); setError(''); setWheel(null); setVendorWheel(null);
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
    if (hipBackend && ['requires_cuda', 'requires_dtk_wheel', 'vendor_wheel_required'].includes(pkg.reason)) return copy('需要匹配的 DTK 适配包', 'A compatible DTK build is required');
    if (pkg.reason === 'requires_cuda') return copy('需要可用的 NVIDIA CUDA', 'Requires working NVIDIA CUDA');
    if (pkg.reason === 'requires_ampere') return copy('需要 Ampere 或更新的显卡', 'Requires Ampere or newer GPU');
    if (pkg.reason === 'requires_nvidia') return copy('Apple GPU 不适用', 'Not applicable to Apple GPU');
    if (!pkg.version) return copy('未启用 · 可选', 'Not enabled · optional');
    if (status?.probe_deferred) return copy('任务运行中，检测已延后', 'Probe deferred while a job runs');
    if (pkg.available) return pkg.backend ? hipBackend ? copy('已通过 DTK / HIP 内核检测', 'DTK / HIP kernel probe passed') : copy('已通过 CUDA 内核检测', 'CUDA kernel probe passed') : copy('可用', 'Available');
    return copy('检测失败，展开查看', 'Probe failed; expand for details');
  };
  const textUnavailable = () => copy(' · 驱动可见，当前 PyTorch 不可用',' · visible to driver, unavailable to current PyTorch');
  const locked = busy || uploading || operations.some(busyStatus) || !!status?.running_jobs;
  const profile = status?.runtime.environment_profile || 'legacy';
  const profileLabel = ({ 'windows-cuda': 'Windows CUDA', 'linux-cuda': 'Linux CUDA', 'linux-dtk': 'Linux DTK', 'macos-mps': 'macOS MPS', 'windows-cpu': 'Windows CPU', 'linux-cpu': 'Linux CPU', 'macos-cpu': 'macOS CPU', legacy: copy('旧版环境', 'Legacy environment') } as Record<string, string>)[profile] || copy('未知环境', 'Unknown environment');
  const cpuProfile = profile.endsWith('-cpu');
  const hipBackend = status?.runtime.compute_backend === 'hip' || !!status?.runtime.hip_runtime || profile === 'linux-dtk';
  const wheelUpload = (packageName: string) => <>
    <button type="button" className={button} disabled={locked} onClick={() => wheelInputRef.current?.click()}><Upload size={13}/>{uploading ? copy('上传并校验…', 'Uploading and checking…') : copy('上传 wheel', 'Upload wheel')}</button>
    <input ref={wheelInputRef} type="file" accept=".whl" hidden aria-label={`${packageName} wheel`} disabled={locked} onChange={event => {const file = event.target.files?.[0]; if (file) void upload(file, packageName); event.target.value = '';}}/>
  </>;
  const computeBackend = cpuProfile ? 'CPU' : status?.runtime.cuda_available
    ? hipBackend ? 'DTK / HIP' : `CUDA ${status.runtime.cuda_runtime || ''}`.trim()
    : status?.runtime.mps_available ? 'Apple MPS' : 'CPU';
  const facts = status && [
    [copy('部署环境', 'Deployment environment'), profileLabel],
    ['Python', status.runtime.python],
    ['PyTorch', status.runtime.torch],
    ...(status.runtime.mps_available ? [] : [[hipBackend ? copy('HIP 运行时', 'HIP runtime') : copy('CUDA 版本', 'CUDA version'), (hipBackend ? status.runtime.hip_runtime : status.runtime.cuda_runtime) || copy('未检测到', 'Not detected')], [hipBackend ? copy('海光显卡计算', 'Hygon GPU compute') : copy('NVIDIA 显卡计算', 'NVIDIA GPU compute'), !cpuProfile && status.runtime.cuda_available ? copy('可用', 'Available') : copy('未启用', 'Not enabled')]]),
    [copy('计算后端', 'Compute backend'), computeBackend],
    [copy('设备', 'Device'), cpuProfile ? 'CPU' : status.runtime.gpus.map(g => g.name).join(' / ') || (status.runtime.mps_available ? 'Apple GPU' : 'CPU')],
  ];

  return <div data-testid="environment-manager"><SettingsSections sections={[
    { id: 'environment-runtime', label: copy('当前环境', 'Current runtime') },
    { id: 'environment-torch', label: profile === 'linux-dtk' ? copy('DTK 安装指南', 'DTK installation guide') : copy('PyTorch 版本', 'PyTorch version') },
    { id: 'environment-attention', label: copy('注意力加速', 'Attention acceleration') },
    ...(visibleOperations.length || torchOperationsVisible ? [{ id: 'environment-installation', label: copy('安装日志', 'Installation log') }] : []),
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
        <div className="settings-note" data-testid="environment-training-devices">
          <p>{!cpuProfile && status.runtime.multi_gpu_probe_required
            ? copy('Windows 数据并行（DDP）会在每次启动时检查所选显卡的 Gloo 通信，通过后才加载训练模型；检查失败会停止启动并显示原因。每张卡保留完整模型，当前不支持 Windows 原生显存分片（FSDP）。', 'Windows DDP checks Gloo communication on the selected GPUs before loading the training model. A failed check stops startup and explains why. Each GPU holds the full model; native Windows FSDP is not supported.')
            : !cpuProfile && status.runtime.multi_gpu_training
              ? copy('可在训练参数中选择显卡数量和多卡方式。数据并行（DDP）分担训练数据，每张卡保留完整模型；主模型全参训练可选择显存分片（FSDP），分担参数、梯度和优化器状态。', 'Choose the GPU count and distributed strategy in training parameters. Data parallelism (DDP) splits training data and keeps the full model on each GPU. Full-backbone training can use FSDP to shard parameters, gradients and optimizer state.')
              : copy('当前环境使用单设备训练。多卡训练需要至少两张可用显卡；Linux 使用 NCCL 兼容后端，Windows 数据并行需要通过 Gloo 通信检查。', 'This environment uses a single training device. Multi-GPU training needs at least two available GPUs: an NCCL-compatible backend on Linux, or a successful Gloo communication check for Windows DDP.')}</p>
          {!status.runtime.mps_available && <p>{copy(`PyTorch 可用显卡：${cpuProfile ? 0 : status.runtime.cuda_device_count ?? 0} 张`, `GPUs available to PyTorch: ${cpuProfile ? 0 : status.runtime.cuda_device_count ?? 0}`)}</p>}
          <p>{status.runtime.multi_gpu_backend === 'gloo' ? copy('GPU 集体通信（Gloo）：当前版本已包含，实际可用性会在任务启动时检查。', 'GPU collective communication (Gloo): included in this build; checked on the selected GPUs at job startup.') : <>{copy('GPU 集体通信（NCCL 兼容接口）：', 'GPU collective communication (NCCL-compatible API): ')} {status.runtime.mps_available || status.runtime.platform === 'Windows' || cpuProfile ? copy('当前平台不适用，不影响单设备训练。', 'Not applicable on this platform; single-device training is unaffected.') : status.runtime.nccl_available ? copy('当前版本已包含。', 'Included in this build.') : copy('当前版本未包含；单卡训练不需要它。', 'Not included in this build; single-GPU training does not need it.')}</>}</p>
        </div>
        {status.runtime.gpus.some(g => g.telemetry_source) && <p>{status.runtime.gpus.map(g => `${g.name}: ${g.telemetry_source || '—'}${g.telemetry_note ? ` (${t(`hardware.${g.telemetry_note}`)})` : ''}`).join(' / ')}</p>}
      </details>
      {!status.runtime.cuda_available && !status.runtime.mps_available && <p className="settings-note">{hipBackend ? copy('当前 DTK / HIP 无法访问显卡，请检查厂商运行时、驱动及库路径后重启。', 'DTK / HIP cannot access the GPUs. Check the vendor runtime, driver and library paths, then restart.') : status.runtime.cuda_runtime ? copy(`当前 PyTorch 含 CUDA ${status.runtime.cuda_runtime}，但无法使用 CUDA。请检查显卡驱动后重启。`, `PyTorch includes CUDA ${status.runtime.cuda_runtime}, but CUDA is unavailable. Check the GPU driver and restart.`) : copy('当前使用 CPU 计算。若需使用 NVIDIA 显卡，请通过启动器配置 CUDA 版 PyTorch。', 'Currently using CPU compute. To use an NVIDIA GPU, configure CUDA-enabled PyTorch through the launcher.')}</p>}
      {(status.running_jobs || status.restart_required) && <p role="status" className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-950/30 dark:text-amber-200">{status.running_jobs ? copy('训练或缓存任务正在运行。等待任务完成或停止、工作进程退出后，才能修改环境。', 'Training or caching is running. Wait for the task to finish or stop and its worker to exit before changing dependencies.') : copy('环境已发生变更，请停止并重新启动 Studio。重启前队列不会启动新任务；如安装失败，可先在这里修复或卸载。', 'The environment changed. Stop and restart Studio before new queued jobs can start. Failed packages can be repaired or removed here first.')}</p>}
    </>}
    </section>
    {status && <>
      {profile === 'linux-dtk' ? <DtkRuntimePanel/> : <TorchEnvironmentPanel disabled={status.running_jobs || busy || operations.some(busyStatus)} operationsTarget={torchOperationsTarget} onOperationsVisible={setTorchOperationsVisible}/>}
      <section id="environment-attention" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{copy('注意力加速', 'Attention acceleration')}</h2><p className="settings-note">{hipBackend ? copy('使用当前 DTK 适配版 PyTorch 的 SDPA；扩展包需要与厂商运行时匹配。', 'Use SDPA from the DTK-compatible PyTorch build. Extension packages must match the vendor runtime.') : copy('默认 SDPA 即可训练；可选扩展用于 CUDA 加速。', 'SDPA is ready for training. Optional extensions provide CUDA acceleration.')}</p></div></div>
      {hipBackend && <div className="settings-sdpa-status" data-testid="environment-sdpa">
        <div className="flex flex-wrap items-center justify-between gap-2"><strong className="text-sm">PyTorch SDPA</strong><span className={`text-xs ${status.sdpa?.status === 'passed' ? 'text-emerald-700 dark:text-emerald-400' : status.sdpa?.status === 'failed' ? 'text-amber-700 dark:text-amber-300' : 'text-slate-500 dark:text-slate-400'}`}>{status.sdpa?.status === 'passed' ? copy('已通过前向与反向检测', 'Forward and backward checks passed') : status.sdpa?.status === 'failed' ? copy('当前计算路径不可用', 'Current compute path unavailable') : status.probe_deferred ? copy('任务运行中，检测已延后', 'Probe deferred while a job runs') : copy('尚未检测', 'Not tested yet')}</span></div>
        {status.sdpa?.status === 'passed' && <p className="settings-note">{copy('已检查当前设备的 FP16 / BF16 小规模运算；具体模型仍以训练结果为准。', 'Checked small FP16 / BF16 operations on the current device; individual models still need training validation.')}{status.sdpa.device && ` · ${status.sdpa.device}${status.sdpa.device_name ? ` · ${status.sdpa.device_name}` : ''}`}</p>}
        {status.sdpa?.reason === 'hip_sdpa_flash_library_missing' ? <div className="mt-2 space-y-2"><p className="settings-note">{copy('当前厂商 PyTorch 的这条 SDPA 路径需要 FlashAttention 动态库。请安装匹配当前 DTK / PyTorch 的官方 FlashAttention 包，重启后重新检测。', 'This vendor PyTorch SDPA path needs a FlashAttention library. Install the official build matching DTK / PyTorch, restart, and check again.')}</p><button type="button" className={button} disabled={locked} onClick={() => { setSelected('flash-attn'); setVersion(''); setWheel(null); setVendorWheel(null); document.getElementById('environment-package-flash-attn')?.scrollIntoView?.({ block: 'nearest' }); }}>{copy('查看匹配的 FlashAttention 包', 'View matching FlashAttention builds')}</button></div> : status.sdpa?.error && <p role="alert" className="settings-note whitespace-pre-wrap break-words">{status.sdpa.error}</p>}
        {status.sdpa?.error && status.sdpa.reason === 'hip_sdpa_flash_library_missing' && <details className="settings-inline-details"><summary>{copy('查看检测详情', 'Probe details')}</summary><pre className="whitespace-pre-wrap break-words text-xs">{status.sdpa.detail || status.sdpa.error}</pre></details>}
      </div>}
      <div className="settings-dependencies">{status.packages.filter(pkg => managedPackages.has(pkg.name)).map(pkg => <div key={pkg.name} className="settings-dependency">
        <div id={`environment-package-${pkg.name}`} className="settings-dependency-row" data-testid={`environment-package-${pkg.name}`}>
          <div className="settings-dependency-info"><button type="button" disabled={uploading || busy} className="settings-dependency-name disabled:opacity-50" aria-expanded={selected === pkg.name} aria-controls={`environment-details-${pkg.name}`} onClick={() => { setSelected(selected === pkg.name ? null : pkg.name); setVersion(''); setWheel(null); setVendorWheel(null); }}>{selected === pkg.name ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{packageLabel(pkg.name)}</button><p className="settings-dependency-purpose">{purpose(pkg.name)}</p></div>
          <span className="settings-dependency-version break-all font-mono text-xs">{pkg.version || copy('未安装', 'Not installed')}</span>
          <span className={`settings-dependency-state text-xs ${pkg.available ? 'text-emerald-700 dark:text-emerald-400' : pkg.error && pkg.supported ? 'text-amber-700 dark:text-amber-300' : 'text-slate-500 dark:text-slate-400'}`}>{reason(pkg)}</span>
          <div className="settings-dependency-actions flex flex-wrap gap-1.5 justify-end"><button className={button} disabled={locked || !pkg.supported && !hipBackend} onClick={() => { setSelected(pkg.name); setVersion(''); setWheel(null); setVendorWheel(null); }}>{pkg.version ? copy('管理', 'Manage') : copy('安装', 'Install')}</button><a className={button} href={pkg.docs_url} target="_blank" rel="noreferrer" aria-label={`${pkg.name} ${copy('文档', 'documentation')}`}><ExternalLink size={12} /></a></div>
        </div>
        {selected === pkg.name && <div id={`environment-details-${pkg.name}`} className="settings-dependency-detail space-y-3">
          {pkg.error && <p className="whitespace-pre-wrap break-words text-xs text-red-600 dark:text-red-300">{pkg.error}</p>}
          <>
            {hipBackend && <DtkWheelPicker packageName={pkg.name} selected={vendorWheel && 'dtk' in vendorWheel ? vendorWheel : null} disabled={locked} onSelect={next => {setVendorWheel(next); setWheel(null); setVersion('');}}/>}
            {!hipBackend && status.runtime.platform === 'Linux' && pkg.name === 'flash-attn' && <WindowsAttentionWheelPicker selected={vendorWheel && 'cuda' in vendorWheel ? vendorWheel : null} disabled={locked} onSelect={next => {setVendorWheel(next); setWheel(null); setVersion('');}}/>}
            {!hipBackend && status.runtime.platform === 'Windows' && pkg.name === 'flash-attn' && <WindowsAttentionWheelPicker selected={vendorWheel && 'cuda' in vendorWheel ? vendorWheel : null} disabled={locked} onSelect={next => {setVendorWheel(next); setWheel(null); setVersion('');}}/>}
            <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">{hipBackend || status.runtime.platform === 'Windows' && pkg.name === 'flash-attn' ? copy('可自动下载；也可手动下载后上传此 wheel，再检查安装条件。', 'Download automatically, or download the wheel manually and upload it to review the install plan.') : pkg.wheel_required ? copy('此平台需要预编译 wheel，请上传匹配当前 Python、Torch 和 CUDA 的文件。', 'This platform needs a prebuilt wheel matching the current Python, Torch and CUDA.') : copy('检查版本和依赖要求后，会列出要安装的内容；确认后才开始安装。', 'Check version and dependency requirements, review the packages, then confirm installation.')}</p>
            <div className="flex flex-wrap items-center gap-2">
              <button className={`${button} border-blue-600 bg-blue-600 text-white hover:bg-blue-700`} disabled={locked || (hipBackend ? !wheel && !vendorWheel : !pkg.supported || pkg.wheel_required && !wheel && !vendorWheel)} onClick={() => void plan(pkg.name, 'install')}>{vendorWheel ? copy('下载并检查安装包', 'Download and check the package') : copy('检查安装条件', 'Check installation requirements')}</button>
              {vendorWheel && <a className={button} href={vendorWheel.url} target="_blank" rel="noreferrer"><Download size={13}/>{copy('手动下载此 wheel', 'Download this wheel manually')}</a>}
              {hipBackend && wheelUpload(pkg.name)}
              {pkg.version && <><button className={button} disabled={locked || (hipBackend ? !wheel && !vendorWheel : !pkg.supported || pkg.wheel_required && !wheel && !vendorWheel)} onClick={() => void plan(pkg.name, 'repair')}>{copy('重装当前版本', 'Reinstall current version')}</button><button className={button} disabled={locked} onClick={() => void plan(pkg.name, 'uninstall')}>{copy('卸载', 'Uninstall')}</button></>}
            </div>
            {!hipBackend && <details className="settings-inline-details" open={pkg.wheel_required}><summary>{pkg.wheel_required ? copy('手动上传 wheel', 'Upload wheel manually') : copy('手动版本与 wheel', 'Manual version and wheel')}</summary>
            <div className="flex flex-wrap items-center gap-2">{!pkg.wheel_required && <label className="flex items-center gap-2 text-xs">{copy('版本', 'Version')}<input className={`${input} w-40`} aria-label={`${pkg.name} ${copy('版本', 'version')}`} placeholder={copy('自动匹配兼容版本', 'Compatible version')} value={version} onChange={event => setVersion(event.target.value)} disabled={locked || !!wheel}/></label>}
              {wheelUpload(pkg.name)}
            </div></details>}
            {pkg.wheel_required && !wheel && !vendorWheel && <p className="settings-note">{copy('先选择兼容构建或上传 wheel，即可检查安装条件。', 'Choose a compatible build or upload a wheel to review the install plan.')}</p>}
            {wheel && <p className="flex items-center gap-2 break-all text-xs text-emerald-700 dark:text-emerald-400"><Check size={13} />{wheel.filename}<button className="text-slate-500 dark:text-slate-400" aria-label={copy('清除 wheel', 'Clear wheel')} onClick={() => { setWheel(null); setVendorWheel(null); setVersion(''); }}><X size={13} /></button></p>}
          </>
        </div>}
      </div>)}</div></section>
    </>}
    <section id="environment-installation" data-settings-section tabIndex={-1} hidden={!visibleOperations.length && !torchOperationsVisible} className="settings-section space-y-3" data-testid={visibleOperations.length || torchOperationsVisible ? 'environment-operations' : undefined}>
      {(visibleOperations.length > 0 || torchOperationsVisible) && <div className="settings-section-heading"><h2>{copy('安装日志', 'Installation log')}</h2></div>}
      <div ref={setTorchOperationsTarget} className="space-y-3"/>
      {visibleOperations.map(op => <InstallationOperation key={op.id} title={packageLabel(op.package)}
        action={op.action === 'uninstall' ? copy('卸载', 'Uninstall') : op.action === 'repair' ? copy('重装', 'Reinstall') : copy('安装', 'Install')}
        status={op.status === 'installing' ? op.action === 'uninstall' ? copy('正在卸载', 'Uninstalling') : op.action === 'repair' ? copy('正在重装', 'Reinstalling') : copy('正在安装', 'Installing') : statusLabel(op.status)}
        busy={busyStatus(op)} failed={op.status === 'failed'} expanded={expanded === op.id} onToggle={() => setExpanded(expanded === op.id ? null : op.id)}>
        <DownloadProgress operation={op} copy={copy}/>
        {op.plan.length > 0 && <div className="space-y-1 text-xs">{op.plan.map(item => <p key={item.name} className="break-words"><span className="font-mono">{item.name}</span> · {item.from_version || copy('未安装', 'not installed')} → <strong>{item.version || copy('移除', 'remove')}</strong></p>)}</div>}
        {op.error && <p role="alert" className="whitespace-pre-wrap break-words text-xs text-red-600 dark:text-red-300">{op.error}</p>}
        {op.status === 'ready' && <div className="flex flex-wrap items-center gap-2"><button className={`${button} border-blue-600 bg-blue-600 text-white hover:bg-blue-700`} disabled={locked} onClick={() => void execute(() => apiClient.post(`/environment/operations/${op.id}/apply`, {}, { silent: true }))}>{op.action === 'uninstall' ? copy('确认卸载', 'Confirm uninstall') : op.action === 'repair' ? copy('确认重装', 'Confirm reinstall') : copy('确认安装', 'Confirm install')}</button><span className="text-xs text-slate-500 dark:text-slate-400">{copy('执行后需要重启 Studio。', 'Restart Studio after applying.')}</span></div>}
        {['planning', 'ready'].includes(op.status) && <button className={button} disabled={busy} onClick={() => void execute(() => apiClient.post(`/environment/operations/${op.id}/cancel`, {}, { silent: true }))}>{op.action === 'uninstall' ? copy('取消卸载', 'Cancel uninstall') : op.phase === 'download' ? copy('取消下载', 'Cancel download') : copy('取消安装', 'Cancel installation')}</button>}
        {['installing', 'verifying'].includes(op.status) && <p className="text-xs text-slate-500 dark:text-slate-400">{copy('正在修改已安装的软件包。为避免留下不完整的环境，此阶段不能中断。', 'Installed packages are being changed. This stage cannot be interrupted because it could leave an incomplete environment.')}</p>}
        {op.logs.length > 0 && <InstallationLog label={copy('安装日志', 'Installation log')} logs={op.logs}/>}
      </InstallationOperation>)}
    </section>
  </SettingsSections></div>;
}
