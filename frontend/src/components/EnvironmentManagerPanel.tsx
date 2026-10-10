import { formatGpuMemory } from '../utils/gpuMemory';
import TorchEnvironmentPanel from './TorchEnvironmentPanel';
import TtsEnvironmentPanel from './TtsEnvironmentPanel';
import { SlidingIndicator } from './motion';
import InstallationOperation, { InstallationLog, InstallationProgress } from './InstallationOperation';
import DtkWheelPicker, { type DtkWheel } from './DtkWheelPicker';
import DtkRuntimePanel from './DtkRuntimePanel';
import { LoadingNote } from './Loading';
import ConfigHelp from './ConfigHelp';
import WindowsAttentionWheelPicker, { type WindowsAttentionWheel } from './WindowsAttentionWheelPicker';
import React from 'react';
import { useQueryClient, type QueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Check, ChevronDown, ChevronRight, Download, ExternalLink, Loader2, RefreshCw, Upload, X } from 'lucide-react';
import { apiClient } from '../api/client';
import { useEnvironmentRead } from './useEnvironmentRead';
import { IDLE_POLL_MS, ONLINE_STALE_MS, ResourceActivityContext, invalidateResource, useLocalResourceSync } from '../api/resourcePolicy';
import { formatApiError } from '../utils/errors';
import { formatBytes, formatEta } from '../utils/format';
import { SettingsSections } from '../pages/Settings/SettingsSections';
import { RestartRequiredContext } from './restartRequiredContext';

interface PackageStatus {
  name: string; version: string | null; backend: string | null; docs_url: string;
  supported: boolean; reason: string; importable: boolean; kernel_tested: boolean;
  available: boolean; wheel_required: boolean; error: string | null;
}
export interface EnvironmentStatus {
  runtime: {
    environment_profile?: string;
    python: string; python_executable: string; platform: string; machine: string; torch: string;
    cuda_runtime: string | null; cuda_available: boolean; mps_available: boolean;
    hip_runtime?: string | null; compute_backend?: string;
    gpu_capability: number[] | null; virtual_environment: boolean;
    cuda_device_count?: number; distributed_available?: boolean; nccl_available?: boolean;
    cuda_applicable?: boolean; nccl_applicable?: boolean; multi_gpu_training?: boolean; training_device_policy?: string;
    gloo_available?: boolean; multi_gpu_backend?: 'nccl' | 'gloo' | null; multi_gpu_probe_required?: boolean;
    gpus: { name: string; kind?: string; device?: string | null; cuda_available?: boolean; memory_scope?: string; mem_total_mb?: number; telemetry_source?: string; telemetry_note?: string; driver_version?: string }[];
  };
  packages: PackageStatus[]; attention_default: string; restart_required: boolean;
  maintenance: boolean; running_jobs: boolean; probe_deferred: boolean;
  probed_at?: number | null;
  sdpa?: { status: 'passed' | 'failed' | 'not_tested'; reason: string | null; error: string | null; detail: string | null; checked_at: number | null; device: string | null; device_name: string | null } | null;
}
interface PlanEntry { name: string; from_version: string | null; version: string | null; sha256?: string }
interface LoraEnvironment {
  checked_at: number;
  local: { builtin?: boolean; version?: string | null; commit?: string | null };
  upstream: { version?: string | null; commit?: string | null; head?: string | null; head_date?: string | null; error?: string | null };
}
interface Operation {
  id: string; package: string; action: string; status: string; created_at: number;
  dismissed_at?: number | null; plan: PlanEntry[]; logs: string[]; error: string | null; restart_required: boolean;
  phase?: string; downloaded_bytes?: number; total_bytes?: number | null; bytes_per_second?: number | null; eta_seconds?: number | null;
}
interface Wheel { wheel_id: string; package: string; filename: string; version: string; sha256: string }
interface LatestVersions { packages: Record<string, { version: string | null; source: string; index?: string | null; error: string | null }> }
const button = 'ui-btn ui-btn-sm';
const input = 'rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm dark:border-slate-600 dark:bg-slate-900';
const busyStatus = (op: Operation) => ['planning', 'installing', 'verifying'].includes(op.status);
const cudaAttentionPackages = new Set(['xformers', 'flash-attn']);
const managedPackages = new Set([...cudaAttentionPackages, 'mtlattn', 'bitsandbytes', 'onnxruntime', 'onnxruntime-gpu', 'triton-windows']);
const metalFlashVersion = '0.4.1';
const runtimeIdentities = new WeakMap<QueryClient, string>();
const runtimeIdentity = (value: EnvironmentStatus) => JSON.stringify([
  value.runtime.environment_profile, value.runtime.python, value.runtime.torch, value.runtime.platform,
  value.runtime.machine, value.runtime.cuda_runtime, value.runtime.hip_runtime, value.runtime.cuda_available,
  value.runtime.gpu_capability, value.packages.map(pkg => [pkg.name, pkg.version]),
]);

function runtimeTarget(runtime: EnvironmentStatus['runtime']): 'cpu' | 'mps' | 'cuda' | 'hip' {
  const profile = runtime.environment_profile || 'legacy';
  if (profile.endsWith('-cpu')) return 'cpu';
  if (profile === 'macos-mps') return 'mps';
  if (profile === 'linux-dtk') return 'hip';
  if (profile.endsWith('-cuda')) return 'cuda';
  // Legacy environments have no launcher-owned target; inspect the loaded runtime.
  if (runtime.compute_backend === 'hip' || runtime.hip_runtime) return 'hip';
  if (runtime.compute_backend === 'mps') return 'mps';
  if (runtime.cuda_runtime || runtime.cuda_available || runtime.compute_backend === 'cuda') return 'cuda';
  return runtime.mps_available ? 'mps' : 'cpu';
}

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

type EnvironmentView = 'image' | 'speech';
const viewStorageKey = 'studio.environment-view';

export function EnvironmentManagerPanel({ focusPackage, mode, initialStatus, onStatusChange, initialView, onViewChange }: { focusPackage?: string; mode?: 'onboarding-attention'; initialStatus?: EnvironmentStatus; onStatusChange?: (status: EnvironmentStatus) => void; initialView?: EnvironmentView; onViewChange?: (view: EnvironmentView) => void } = {}) {
  const queryClient = useQueryClient();
  const onboardingAttention = mode === 'onboarding-attention';
  const [view, setView] = React.useState<EnvironmentView>(() => {
    if (focusPackage) return 'image';
    if (initialView) return initialView;
    try { return localStorage.getItem(viewStorageKey) === 'speech' ? 'speech' : 'image'; } catch { return 'image'; }
  });
  const imageVisible = onboardingAttention || view === 'image';
  const parentActive = React.useContext(ResourceActivityContext);
  const viewTabs = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => { if (initialView) setView(initialView); }, [initialView]);
  const pendingFocus = React.useRef<string | undefined>(focusPackage);
  React.useEffect(() => {
    pendingFocus.current = focusPackage;
    if (focusPackage) setView('image');
  }, [focusPackage]);
  React.useEffect(() => {
    if (!onboardingAttention) {
      try { localStorage.setItem(viewStorageKey, view); } catch { /* The current view remains usable without browser storage. */ }
    }
  }, [view, onboardingAttention]);
  const selectView = (next: EnvironmentView) => {
    setView(next);
    onViewChange?.(next);
    const scroller = viewTabs.current?.closest('.settings-scroll');
    if (scroller) scroller.scrollTop = 0;
  };
  const { i18n } = useTranslation();
  const en = i18n.resolvedLanguage?.startsWith('en');
  const copy = (zh: string, english: string) => en ? english : zh;
  const terminalRuntimeRefreshes = React.useRef(new Set<string>());
  const runtimeRead = useEnvironmentRead<EnvironmentStatus>('/environment', { interval: IDLE_POLL_MS, refreshParam: true, probe: true, initialData: initialStatus, onSuccess: next => {
    onStatusChange?.(next);
    const previous = runtimeIdentities.get(queryClient), identity = runtimeIdentity(next);
    runtimeIdentities.set(queryClient, identity);
    if (previous && previous !== identity) {
      void invalidateResource(queryClient, ['environment', '/environment/latest']);
      void invalidateResource(queryClient, ['environment', '/environment/lora']);
      void invalidateResource(queryClient, ['environment', '/environment/windows/wheels']);
      void invalidateResource(queryClient, ['environment', '/environment/dtk/wheels']);
      void invalidateResource(queryClient, ['vision-models']);
    }
  } });
  const status = runtimeRead.data, loading = runtimeRead.loading, probing = runtimeRead.probing;
  const operationsRead = useEnvironmentRead<Operation[]>('/environment/operations', { enabled: imageVisible, interval: rows => rows?.some(busyStatus) ? 1500 : IDLE_POLL_MS, onSuccess: (next, previous) => {
    const finished = previous?.filter(op => busyStatus(op) && !next.some(current => current.id === op.id && busyStatus(current))) || [];
    const needsRuntime = finished.some(op => !terminalRuntimeRefreshes.current.has(op.id));
    for (const op of finished) terminalRuntimeRefreshes.current.delete(op.id);
    if (needsRuntime) void runtimeRead.read();
  } });
  const operations = React.useMemo(() => operationsRead.data || [], [operationsRead.data]);
  React.useEffect(() => {
    if (!imageVisible) void queryClient.invalidateQueries({ queryKey: ['environment', '/environment/operations'], exact: true, refetchType: 'none' });
  }, [imageVisible, queryClient]);
  useLocalResourceSync([['environment', '/environment'], ['environment', '/environment/operations']], event => !!event.id?.startsWith('environment-') || !!event.id?.startsWith('torch-') && ['completed', 'failed', 'cancelled'].includes(event.state ?? ''), event => {
    const operationId = event.id?.startsWith('environment-') ? event.id.slice('environment-'.length) : undefined;
    if (operationId && ['completed', 'failed', 'cancelled'].includes(event.state ?? '') && operations.some(op => op.id === operationId && busyStatus(op)))
      terminalRuntimeRefreshes.current.add(operationId);
  });
  const latestRead = useEnvironmentRead<LatestVersions>('/environment/latest', { enabled: !onboardingAttention && imageVisible, staleTime: ONLINE_STALE_MS, refetchOnVisible: false, refreshParam: true, reconcile: (next, previous) => (
    { ...next, packages: Object.fromEntries(Object.entries(next.packages).map(([name, release]) => [name,
      release.error && previous?.packages[name] ? { ...previous.packages[name], error: release.error } : release,
    ])) }
  ) });
  const latest = latestRead.data;
  const latestError = latestRead.error, refreshLatest = latestRead.reload;
  const loraRead = useEnvironmentRead<LoraEnvironment>('/environment/lora', { enabled: !onboardingAttention && imageVisible, staleTime: ONLINE_STALE_MS, refetchOnVisible: false, refreshParam: true,
    reconcile: (next, previous) => ({ ...next, upstream: next.upstream.error && previous ? { ...previous.upstream, error: next.upstream.error } : next.upstream }),
  });
  const lora = loraRead.data;
  const loraError = loraRead.error, refreshLora = loraRead.reload;
  const reportRestart = React.useContext(RestartRequiredContext);
  const restartRequired = status?.restart_required;
  React.useEffect(() => { if (restartRequired !== undefined) reportRestart(restartRequired); }, [restartRequired, reportRestart]);
  const [busy, setBusy] = React.useState(false);
  const writeInFlight = React.useRef(false);
  const [error, setError] = React.useState('');
  const errorRef = React.useRef<HTMLDivElement>(null);
  const wheelInputRef = React.useRef<HTMLInputElement>(null);
  React.useEffect(() => { if (error && imageVisible) errorRef.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' }); }, [error, imageVisible]);
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
  const profile = status?.runtime.environment_profile || 'legacy';
  const target = status ? runtimeTarget(status.runtime) : 'cpu';
  const cpuProfile = target === 'cpu';
  const hipBackend = target === 'hip';
  const showAttentionExtensions = target === 'cuda' || hipBackend;
  // The attention extensions this machine can use: Metal FlashAttention on Apple GPUs, the CUDA builds on
  // NVIDIA and DTK, none on CPU.
  const visiblePackages = status?.packages.filter(pkg => target === 'mps' ? pkg.name === 'mtlattn' : showAttentionExtensions && cudaAttentionPackages.has(pkg.name)) || [];
  // The 8-bit optimizers run only on CUDA / DTK GPUs; training refuses them elsewhere.
  const optimizerPackages = status?.packages.filter(pkg => showAttentionExtensions && pkg.name === 'bitsandbytes') || [];
  // Triton: installed with PyTorch on Linux and DTK, the matching triton-windows build on Windows, a note elsewhere.
  const tritonPackages = status?.packages.filter(pkg => pkg.name === (target === 'cuda' && status.runtime.platform === 'Windows' ? 'triton-windows' : 'triton')) || [];
  // Tagging and head masks: the GPU build on NVIDIA; elsewhere the CPU build. A CPU build left on an
  // NVIDIA machine stays listed so it can be removed before installing the GPU one.
  const visionPackages = status?.packages.filter(pkg => target === 'cuda' ? pkg.name === 'onnxruntime-gpu' || (pkg.name === 'onnxruntime' && !!pkg.version) : pkg.name === 'onnxruntime') || [];
  // Tagging pages link to "onnxruntime"; on NVIDIA without a CPU build installed the entry to open is the GPU build.
  const focusTarget = focusPackage === 'onnxruntime' && !visionPackages.some(pkg => pkg.name === 'onnxruntime') && visionPackages.some(pkg => pkg.name === 'onnxruntime-gpu') ? 'onnxruntime-gpu' : focusPackage;
  const focusAvailable = !!focusTarget && [...visiblePackages, ...optimizerPackages, ...visionPackages].some(pkg => pkg.name === focusTarget);
  React.useEffect(() => {
    if (!imageVisible || !focusAvailable || !focusTarget || pendingFocus.current !== focusPackage || !focusPackage) return;
    pendingFocus.current = undefined;
    setSelected(focusTarget);
    setVersion(''); setWheel(null); setVendorWheel(null);
    document.getElementById(`environment-package-${focusTarget}`)?.scrollIntoView?.({ block: 'start' });
  }, [focusAvailable, focusTarget, focusPackage, imageVisible]);

  const refreshRuntime = runtimeRead.reload, refreshOperations = operationsRead.reload;
  const refresh = React.useCallback(async (probe = false) => {
    await Promise.all([refreshRuntime(probe), refreshOperations()]);
  }, [refreshRuntime, refreshOperations]);

  const execute = async (fn: () => Promise<Operation>) => {
    if (writeInFlight.current) return;
    writeInFlight.current = true; setBusy(true); setError('');
    try {
      const op = await fn();
      operationsRead.updateData(previous => previous?.some(item => item.id === op.id)
        ? previous.map(item => item.id === op.id ? op : item) : [op, ...(previous || [])]);
      if (op.restart_required) runtimeRead.updateData(previous => previous ? { ...previous, restart_required: true, maintenance: true } : previous);
      void refresh();
    } catch (err) { setError(formatApiError(err)); }
    finally { writeInFlight.current = false; setBusy(false); }
  };
  const plan = async (name: string, action: string) => {
    await execute(async () => {
      const op = await apiClient.post<Operation>('/environment/operations', { package: name, action, ...(name === 'mtlattn' && action !== 'uninstall' ? { version: metalFlashVersion } : action === 'install' && version.trim() ? { version: version.trim() } : {}), ...(name !== 'mtlattn' && action !== 'uninstall' && wheel ? { wheel_id: wheel.wheel_id } : {}), ...(name !== 'mtlattn' && action !== 'uninstall' && vendorWheel ? {vendor_wheel_id: vendorWheel.id} : {}) }, { silent: true });
      setCreatedOperations(previous => new Set(previous).add(op.id)); setExpanded(op.id); setSelected(null); setVersion(''); setWheel(null); setVendorWheel(null);
      return op;
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
  const packageLabel = (name: string) => ({ xformers: 'xFormers', 'flash-attn': 'FlashAttention 2', mtlattn: 'Metal FlashAttention', onnxruntime: 'ONNX Runtime', 'onnxruntime-gpu': 'ONNX Runtime GPU', triton: 'Triton', 'triton-windows': 'Triton' }[name] || name);
  const purpose = (name: string) => ({
    xformers: copy('降低注意力计算的显存占用', 'Reduce attention memory usage'),
    'flash-attn': copy('加速 FP16 / BF16 注意力计算', 'Accelerate FP16 / BF16 attention'),
    mtlattn: copy('Apple GPU 训练与采样加速（可选）', 'Optional Apple GPU training and sampling acceleration'),
    bitsandbytes: copy('8-bit 优化器：AdamW 8-bit、Lion 8-bit', '8-bit optimizers: AdamW 8-bit, Lion 8-bit'),
    onnxruntime: copy('自动打标与自动遮罩', 'Automatic tagging and masks'),
    'onnxruntime-gpu': copy('自动打标与自动遮罩，使用 NVIDIA 显卡', 'Automatic tagging and masks on NVIDIA GPUs'),
    triton: copy('编译模型和部分扩展使用的 GPU 内核编译器', 'GPU kernel compiler for model compiling and some extensions'),
    'triton-windows': copy('编译模型和部分扩展使用的 GPU 内核编译器', 'GPU kernel compiler for model compiling and some extensions'),
  }[name] || '');
  const checkHint = (name: string) => name === 'bitsandbytes'
    ? copy('点击“运行检查”，检查 8-bit 优化器能否在当前显卡上运行。', 'Click “Run checks” to check whether the 8-bit optimizers run on the current GPU.')
    : copy('点击“运行检查”，检查此扩展能否在当前显卡上执行前向与反向计算。', 'Click “Run checks” to check whether this extension can run forward and backward computations on the current GPU.');
  const tritonRow = (name: string) => name === 'triton' || name === 'triton-windows';
  const tritonHelp = () => [
    copy('开启「编译模型」时必须安装 Triton；xFormers 有它时可多用一部分加速内核。不编译模型时，不装也能正常训练。', 'Required when compiling the model; with it, xFormers can also use some extra kernels. Without compiling, training works without Triton.'),
    target === 'cuda' && status?.runtime.platform === 'Windows' ? copy('Windows 上安装与当前 PyTorch 对应的 triton-windows 版本。没有对应版本时，可以上传与当前 Python 和 PyTorch 匹配的 triton-windows 安装包（.whl）。', 'On Windows, the triton-windows version matching the current PyTorch is installed. If none matches, upload a triton-windows wheel (.whl) built for the current Python and PyTorch.')
      : target === 'cuda' ? copy('Linux 的 Triton 随 PyTorch 安装，版本由 PyTorch 决定，这里不单独安装。', 'On Linux, Triton ships with PyTorch, which decides its version; it is not installed separately here.')
        : hipBackend ? copy('海光环境使用随厂商 PyTorch 安装的配套 Triton，海光 FlashAttention 也需要它。', 'DTK uses the matching vendor Triton installed with the vendor PyTorch; the DTK FlashAttention build needs it too.')
          : copy('Apple 芯片和 CPU 不使用 Triton，也不会编译模型。', 'Apple chips and CPUs do not use Triton and do not compile the model.'),
  ].join('\n\n');
  const reason = (pkg: PackageStatus) => {
    const tritonState = tritonRow(pkg.name) ? ({
      triton_unavailable: target === 'mps' ? copy('Apple 芯片不使用 Triton', 'Not used on Apple chips') : copy('CPU 环境不使用 Triton', 'Not used on CPU'),
      triton_missing: hipBackend ? copy('未安装 · 应随厂商 PyTorch 安装', 'Not installed · ships with the vendor PyTorch') : copy('未安装 · 应随 PyTorch 安装', 'Not installed · ships with PyTorch'),
      triton_version_mismatch: copy('版本与 PyTorch 不配套', 'Does not match PyTorch'),
      triton_no_build: copy('当前 PyTorch 没有配套版本', 'No build matches this PyTorch'),
      triton_unsupported_gpu: copy('当前显卡不支持 Triton', 'This GPU does not support Triton'),
      triton_requires_ampere: copy('需要 RTX 30 系或更新的显卡', 'Requires an RTX 30-series or newer GPU'),
      triton_requires_cuda128: copy('需要 CUDA 12.8 及以上的 PyTorch', 'Requires a PyTorch build with CUDA 12.8 or newer'),
    } as Record<string, string>)[pkg.reason] : undefined;
    if (tritonState) return tritonState;
    if (probing && pkg.version && (pkg.supported || ['bundled_with_torch', 'vendor_build'].includes(pkg.reason))) return copy('检查中…', 'Checking…');
    if (pkg.name === 'mtlattn') {
      const requirements: Record<string, [string, string]> = {
        metal_requires_apple_silicon: ['需要 Apple Silicon Mac', 'Requires an Apple Silicon Mac'],
        metal_requires_mps_profile: ['需要 macOS MPS 环境', 'Requires a macOS MPS environment'],
        metal_requires_macos_15: ['需要 macOS 15 或更新版本', 'Requires macOS 15 or later'],
        metal_requires_python_311_312: ['需要 Python 3.11 或 3.12', 'Requires Python 3.11 or 3.12'],
        metal_requires_torch_2_13: ['需要 PyTorch 2.13.x', 'Requires PyTorch 2.13.x'],
        metal_requires_mps: ['当前 Apple GPU 不可用', 'Apple GPU is currently unavailable'],
      };
      if (requirements[pkg.reason]) return copy(...requirements[pkg.reason]);
      if (!pkg.version) return copy('未安装 · 可选', 'Not installed · optional');
      if (status?.probe_deferred) return copy('任务运行中，检测已延后', 'Probe deferred while a job runs');
      if (!status?.probed_at && !pkg.available && !pkg.error) return copy('已安装 · 待验证运行', 'Installed · runtime check pending');
      return pkg.available ? copy('检测通过', 'Check passed') : copy('检测失败，展开查看', 'Check failed; expand for details');
    }
    if (hipBackend && ['requires_cuda', 'requires_dtk_wheel', 'vendor_wheel_required'].includes(pkg.reason)) return copy('需要匹配的 DTK 适配包', 'A compatible DTK build is required');
    if (pkg.reason === 'requires_cuda') return copy('需要可用的 NVIDIA CUDA', 'Requires working NVIDIA CUDA');
    if (pkg.reason === 'requires_ampere') return copy('需要 Ampere 或更新的显卡', 'Requires Ampere or newer GPU');
    if (pkg.reason === 'requires_nvidia') return copy('需要 NVIDIA 显卡', 'Requires an NVIDIA GPU');
    if (!pkg.version) return copy('未安装 · 可选', 'Not installed · optional');
    if (status?.probe_deferred) return copy('任务运行中，检测已延后', 'Probe deferred while a job runs');
    if (pkg.available) return pkg.kernel_tested ? hipBackend ? copy('已通过 DTK / HIP 内核检测', 'DTK / HIP kernel probe passed') : copy('已通过 CUDA 内核检测', 'CUDA kernel probe passed') : copy('可用', 'Available');
    if (!status?.probed_at && !pkg.error) return copy('已安装 · 待验证运行', 'Installed · runtime check pending');
    return pkg.name === 'triton' ? copy('检测失败', 'Probe failed') : copy('检测失败，展开查看', 'Probe failed; expand for details');
  };
  const onlineLookup = (value: string | null | undefined, failed: boolean, resource: typeof latestRead | typeof loraRead, name: string) => <span className="inline-flex flex-wrap items-baseline gap-x-2" aria-live="polite">
    {value && <span>{value}</span>}
    {resource.loading || failed ? <span className="inline-flex items-baseline gap-2 whitespace-nowrap">
      <span className={resource.loading ? 'text-slate-500 dark:text-slate-400' : 'text-red-600 dark:text-red-300'}>{resource.loading ? copy('查询中…', 'Checking…') : copy('查询失败', 'Lookup failed')}</span>
      {failed && <button type="button" className="ui-link" disabled={resource.loading} aria-label={copy(`重试 ${name} 在线版本`, `Retry ${name} online version`)} onClick={() => void resource.reload(true)}>{copy('重试', 'Retry')}</button>}
    </span> : !value && copy('未找到匹配版本', 'No matching release found')}
  </span>;
  const onlineVersion = (pkg: PackageStatus) => {
    const release = latest?.packages[pkg.name];
    if (tritonRow(pkg.name) && !release?.version) return '—';
    return onlineLookup(release?.version, !!(release?.error || latestError), latestRead, packageLabel(pkg.name));
  };
  const sourceHost = (url: string) => { try { return new URL(url).host; } catch { return url; } };
  const textUnavailable = () => copy(' · 驱动可见，当前 PyTorch 不可用',' · visible to driver, unavailable to current PyTorch');
  const locked = busy || uploading || !status || !!runtimeRead.error || !operationsRead.data || !!operationsRead.error || operations.some(busyStatus) || !!status?.running_jobs;
  const profileLabel = ({ 'windows-cuda': 'Windows CUDA', 'linux-cuda': 'Linux CUDA', 'linux-dtk': 'Linux DTK', 'macos-mps': 'macOS MPS', 'windows-cpu': 'Windows CPU', 'linux-cpu': 'Linux CPU', 'macos-cpu': 'macOS CPU', legacy: copy('旧版环境', 'Legacy environment') } as Record<string, string>)[profile] || copy('未知环境', 'Unknown environment');
  const wheelUpload = (packageName: string) => <>
    <button type="button" className={button} disabled={locked} onClick={() => wheelInputRef.current?.click()}><Upload size={13}/>{uploading ? copy('上传并校验…', 'Uploading and checking…') : copy('上传 wheel', 'Upload wheel')}</button>
    <input ref={wheelInputRef} type="file" accept=".whl" hidden aria-label={`${packageName} wheel`} disabled={locked} onChange={event => {const file = event.target.files?.[0]; if (file) void upload(file, packageName); event.target.value = '';}}/>
  </>;
  const packageItem = (pkg: PackageStatus) => <div key={pkg.name} className="settings-dependency">
    <div id={`environment-package-${pkg.name}`} className="settings-dependency-row" data-testid={`environment-package-${pkg.name}`}>
      <div className="settings-dependency-info">{pkg.name === 'triton'
        ? <span className="settings-dependency-name settings-dependency-name-static">{packageLabel(pkg.name)}</span>
        : <button type="button" disabled={uploading || busy} className="settings-dependency-name disabled:opacity-50" aria-expanded={selected === pkg.name} aria-controls={`environment-details-${pkg.name}`} onClick={() => { setSelected(selected === pkg.name ? null : pkg.name); setVersion(''); setWheel(null); setVendorWheel(null); }}>{selected === pkg.name ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{packageLabel(pkg.name)}</button>
      }<p className="settings-dependency-purpose">{purpose(pkg.name)}{tritonRow(pkg.name) && <ConfigHelp label={copy('Triton 说明', 'Triton help')}>{tritonHelp()}</ConfigHelp>}</p></div>
      {(!onboardingAttention || pkg.version) && <dl className="settings-dependency-version text-xs">
        <div><dt>{onboardingAttention ? copy('版本：', 'Version:') : copy('本地服务端版本：', 'Local server version:')}</dt><dd>{pkg.version || copy('未安装', 'Not installed')}</dd></div>
        {!onboardingAttention && <div title={latest?.packages[pkg.name]?.index ? copy(`来自 ${sourceHost(latest.packages[pkg.name].index!)}`, `From ${sourceHost(latest.packages[pkg.name].index!)}`) : undefined}><dt>{pkg.name === 'mtlattn' ? copy('兼容版本：', 'Compatible version:') : tritonRow(pkg.name) ? copy('配套版本：', 'Matching version:') : copy('云端版本：', 'Online version:')}</dt><dd>{onlineVersion(pkg)}</dd></div>}
      </dl>}
      <span title={pkg.version && pkg.supported && !status?.probed_at && !pkg.available && !pkg.error ? checkHint(pkg.name) : undefined} className={`settings-dependency-state text-xs ${pkg.available ? 'text-emerald-700 dark:text-emerald-400' : (pkg.error && pkg.supported) || ['triton_missing', 'triton_version_mismatch'].includes(pkg.reason) ? 'text-amber-700 dark:text-amber-300' : 'text-slate-500 dark:text-slate-400'}`}>{reason(pkg)}</span>
      <div className="settings-dependency-actions flex flex-wrap gap-1.5 justify-end">{(!tritonRow(pkg.name) || pkg.name === 'triton-windows' && pkg.supported) && <button className={button} disabled={locked || !pkg.supported && !hipBackend} onClick={() => { setSelected(pkg.name); setVersion(''); setWheel(null); setVendorWheel(null); }}>{copy('管理', 'Manage')}</button>}<a className={`${button} ui-btn-icon`} href={pkg.docs_url} target="_blank" rel="noreferrer" aria-label={`${pkg.name} ${copy('文档', 'documentation')}`}><ExternalLink size={12} /></a></div>
    </div>
    {(pkg.name === 'triton' ? !!pkg.error : selected === pkg.name) && <div id={`environment-details-${pkg.name}`} className="settings-dependency-detail space-y-3">
      {pkg.error && <p className="whitespace-pre-wrap break-words text-xs text-red-600 dark:text-red-300">{pkg.error}</p>}
      {pkg.name === 'mtlattn' ? <>
        <p className="settings-note">{copy('使用预编译安装包，无需本机编译。', 'Uses a prebuilt package; no local compilation needed.')}</p>
        {pkg.reason === 'metal_requires_torch_2_13' && <div className="space-y-2"><p className="settings-note">{copy('先在上方“PyTorch 版本”中安装并切换到 2.13.x，再安装此扩展。', 'First install and switch to PyTorch 2.13.x using the version selector above, then install this extension.')}</p><button type="button" className={button} onClick={() => { const section = document.getElementById('environment-torch'); section?.focus(); section?.scrollIntoView?.({ block: 'start' }); }}>{copy('查看 PyTorch 版本', 'View PyTorch versions')}</button></div>}
        <div className="flex flex-wrap gap-2">
          <button type="button" className={`${button} ui-btn-primary`} disabled={locked || !pkg.supported} onClick={() => void plan(pkg.name, 'install')}>{copy('检查安装条件', 'Check installation requirements')}</button>
          {pkg.version && <><button type="button" className={button} disabled={locked || !pkg.supported} onClick={() => void plan(pkg.name, pkg.version === metalFlashVersion ? 'repair' : 'install')}>{copy('重新安装兼容版本', 'Reinstall the compatible version')}</button><button type="button" className={`${button} ui-btn-danger`} disabled={locked} onClick={() => void plan(pkg.name, 'uninstall')}>{copy('卸载', 'Uninstall')}</button></>}
        </div>
        <details className="settings-inline-details"><summary>{copy('安装要求', 'Installation requirements')}</summary><p className="settings-note">mtlattn {metalFlashVersion} · Apple Silicon · macOS 15+ · Python 3.11 / 3.12 · PyTorch 2.13.x</p><p className="settings-note">{copy('安装后重启并重新检测；通过检测后，可在训练参数中选择 Metal FlashAttention。', 'Restart and run the check after installation. Once it passes, choose Metal FlashAttention in training settings.')}</p></details>
      </> : pkg.name === 'triton' ? null : pkg.name === 'triton-windows' ? <>
        <div className="flex flex-wrap items-center gap-2">
          {pkg.supported && <><button type="button" className={`${button} ui-btn-primary`} disabled={locked} onClick={() => void plan(pkg.name, 'install')}>{copy('检查安装条件', 'Check installation requirements')}</button>{wheelUpload(pkg.name)}</>}
          {pkg.version && <>{pkg.reason === 'supported' && <button type="button" className={button} disabled={locked} onClick={() => void plan(pkg.name, 'repair')}>{copy('重装当前版本', 'Reinstall current version')}</button>}<button type="button" className={`${button} ui-btn-danger`} disabled={locked} onClick={() => void plan(pkg.name, 'uninstall')}>{copy('卸载', 'Uninstall')}</button></>}
        </div>
        {wheel && <p className="flex items-center gap-2 break-all text-xs text-emerald-700 dark:text-emerald-400"><Check size={13} />{wheel.filename}<button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={copy('清除 wheel', 'Clear wheel')} onClick={() => { setWheel(null); setVersion(''); }}><X size={13} /></button></p>}
      </> : <>
        {hipBackend && <DtkWheelPicker packageName={pkg.name} selected={vendorWheel && 'dtk' in vendorWheel ? vendorWheel : null} disabled={locked} onSelect={next => {setVendorWheel(next); setWheel(null); setVersion('');}}/>}
        {!hipBackend && status?.runtime.platform === 'Linux' && pkg.name === 'flash-attn' && <WindowsAttentionWheelPicker selected={vendorWheel && 'cuda' in vendorWheel ? vendorWheel : null} disabled={locked} onSelect={next => {setVendorWheel(next); setWheel(null); setVersion('');}}/>}
        {!hipBackend && status?.runtime.platform === 'Windows' && pkg.name === 'flash-attn' && <WindowsAttentionWheelPicker selected={vendorWheel && 'cuda' in vendorWheel ? vendorWheel : null} disabled={locked} onSelect={next => {setVendorWheel(next); setWheel(null); setVersion('');}}/>}
        {pkg.wheel_required && !hipBackend && <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">{copy('上传的 wheel 须匹配当前 Python、PyTorch 和 CUDA 版本。', 'Uploaded wheels must match the current Python, PyTorch and CUDA versions.')}</p>}
        <div className="flex flex-wrap items-center gap-2">
          <button className={`${button} ui-btn-primary`} disabled={locked || (hipBackend ? !wheel && !vendorWheel : !pkg.supported || pkg.wheel_required && !wheel && !vendorWheel)} onClick={() => void plan(pkg.name, 'install')}>{vendorWheel ? copy('下载并检查安装包', 'Download and check the package') : copy('检查安装条件', 'Check installation requirements')}</button>
          {vendorWheel && <a className={button} href={vendorWheel.url} target="_blank" rel="noreferrer"><Download size={13}/>{copy('手动下载此 wheel', 'Download this wheel manually')}</a>}
          {hipBackend && wheelUpload(pkg.name)}
          {pkg.version && <><button className={button} disabled={locked || (hipBackend ? !wheel && !vendorWheel : !pkg.supported || pkg.wheel_required && !wheel && !vendorWheel)} onClick={() => void plan(pkg.name, 'repair')}>{copy('重装当前版本', 'Reinstall current version')}</button><button type="button" className={`${button} ui-btn-danger`} disabled={locked} onClick={() => void plan(pkg.name, 'uninstall')}>{copy('卸载', 'Uninstall')}</button></>}
        </div>
        {!hipBackend && (pkg.wheel_required ? <div className="flex flex-wrap items-center gap-2" aria-label={copy('上传匹配 wheel', 'Upload matching wheel')}>
          {wheelUpload(pkg.name)}
        </div> : <details className="settings-inline-details"><summary>{copy('手动版本与 wheel', 'Manual version and wheel')}</summary>
          <div className="flex flex-wrap items-center gap-2"><label className="flex items-center gap-2 text-xs">{copy('版本', 'Version')}<input className={`${input} w-40`} aria-label={`${pkg.name} ${copy('版本', 'version')}`} placeholder={copy('自动匹配兼容版本', 'Compatible version')} value={version} onChange={event => setVersion(event.target.value)} disabled={locked || !!wheel}/></label>
            {wheelUpload(pkg.name)}
          </div></details>)}
        {pkg.wheel_required && !wheel && !vendorWheel && <p className="settings-note">{copy('先选择兼容构建或上传 wheel，即可检查安装条件。', 'Choose a compatible build or upload a wheel to review the install plan.')}</p>}
        {wheel && <p className="flex items-center gap-2 break-all text-xs text-emerald-700 dark:text-emerald-400"><Check size={13} />{wheel.filename}<button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={copy('清除 wheel', 'Clear wheel')} onClick={() => { setWheel(null); setVendorWheel(null); setVersion(''); }}><X size={13} /></button></p>}
      </>}
    </div>}
  </div>;
  const computeBackend = cpuProfile ? 'CPU' : target === 'mps' ? status?.runtime.mps_available ? 'Apple MPS' : 'CPU' : status?.runtime.cuda_available
    ? hipBackend ? 'DTK / HIP' : `CUDA ${status.runtime.cuda_runtime || ''}`.trim()
    : status?.runtime.mps_available ? 'Apple MPS' : 'CPU';
  const facts = status && [
    [copy('部署环境', 'Deployment environment'), profileLabel],
    ['Python', status.runtime.python],
    ['PyTorch', status.runtime.torch],
    ...(!showAttentionExtensions ? [] : [[hipBackend ? copy('HIP 运行时', 'HIP runtime') : copy('CUDA 版本', 'CUDA version'), (hipBackend ? status.runtime.hip_runtime : status.runtime.cuda_runtime) || copy('未检测到', 'Not detected')], [hipBackend ? copy('海光显卡计算', 'Hygon GPU compute') : copy('NVIDIA 显卡计算', 'NVIDIA GPU compute'), status.runtime.cuda_available ? copy('可用', 'Available') : copy('未启用', 'Not enabled')]]),
    [copy('计算后端', 'Compute backend'), computeBackend],
    ...(showAttentionExtensions && (status.runtime.cuda_device_count ?? 0) > 1 ? [[copy('多卡通信', 'Multi-GPU communication'),
      status.runtime.multi_gpu_backend === 'gloo' ? copy('Gloo · 启动任务时检测', 'Gloo · checked at job start') : status.runtime.platform === 'Linux' && status.runtime.nccl_available ? 'NCCL' : copy('不可用', 'Unavailable')]] : []),
  ];
  const detectedDevices = status && status.runtime.gpus.length > 0 && <ul className="space-y-1" aria-label={copy('已检测设备', 'Detected devices')}>{status.runtime.gpus.map((gpu, index) => <li key={`${gpu.device || index}:${gpu.name}`}><strong>{gpu.name}</strong><span className="settings-note"> · {gpu.device || `GPU ${index + 1}`}{gpu.mem_total_mb != null ? ` · ${formatGpuMemory(gpu.mem_total_mb)} ${gpu.memory_scope === 'unified_system' ? copy('统一内存', 'unified memory') : copy('设备内存', 'device memory')}` : ''}{gpu.cuda_available === false ? textUnavailable() : ''}</span></li>)}</ul>;

  const readFailure = (label: string, resource: { error: string; loading: boolean; reload: (refresh?: boolean) => Promise<void> }) => resource.error && <p role="alert" className="settings-alert">
    <span>{label}：{resource.error}</span>{' '}<button type="button" className="ui-link" disabled={resource.loading} onClick={() => void resource.reload()}>{copy('重新读取', 'Reload')}</button>
  </p>;

  const installation = <section id="environment-installation" data-settings-section tabIndex={-1} hidden={!visibleOperations.length && !torchOperationsVisible && !operationsRead.error} className="settings-section space-y-3" data-testid={visibleOperations.length || torchOperationsVisible ? 'environment-operations' : undefined}>
      {(visibleOperations.length > 0 || torchOperationsVisible || !!operationsRead.error) && <div className="settings-section-heading"><h2>{copy('安装日志', 'Installation log')}</h2></div>}
      {readFailure(copy('安装记录', 'Installation records'), operationsRead)}
      <div ref={setTorchOperationsTarget} className="space-y-3"/>
      {visibleOperations.map(op => <InstallationOperation key={op.id} title={packageLabel(op.package)}
        action={op.action === 'uninstall' ? copy('卸载', 'Uninstall') : op.action === 'repair' ? copy('重装', 'Reinstall') : copy('安装', 'Install')}
        status={op.status === 'installing' ? op.action === 'uninstall' ? copy('正在卸载', 'Uninstalling') : op.action === 'repair' ? copy('正在重装', 'Reinstalling') : copy('正在安装', 'Installing') : statusLabel(op.status)}
        busy={busyStatus(op)} failed={op.status === 'failed'} expanded={expanded === op.id} onToggle={() => setExpanded(expanded === op.id ? null : op.id)}>
        <DownloadProgress operation={op} copy={copy}/>
        {op.plan.length > 0 && <div className="space-y-1 text-xs">{op.plan.map(item => <p key={item.name} className="break-words"><span className="font-mono">{item.name}</span> · {item.from_version || copy('未安装', 'not installed')} → <strong>{item.version || copy('移除', 'remove')}</strong></p>)}</div>}
        {op.error && <p role="alert" className="whitespace-pre-wrap break-words text-xs text-red-600 dark:text-red-300">{op.error}</p>}
        {op.status === 'ready' && <div className="flex flex-wrap items-center gap-2"><button className={`${button} ui-btn-primary`} disabled={locked} onClick={() => void execute(() => apiClient.post<Operation>(`/environment/operations/${op.id}/apply`, {}, { silent: true }))}>{op.action === 'uninstall' ? copy('确认卸载', 'Confirm uninstall') : op.action === 'repair' ? copy('确认重装', 'Confirm reinstall') : copy('确认安装', 'Confirm install')}</button><span className="text-xs text-slate-500 dark:text-slate-400">{copy('执行后需要重启 Studio。', 'Restart Studio after applying.')}</span></div>}
        {['planning', 'ready'].includes(op.status) && <button className={button} disabled={busy} onClick={() => void execute(() => apiClient.post<Operation>(`/environment/operations/${op.id}/cancel`, {}, { silent: true }))}>{op.action === 'uninstall' ? copy('取消卸载', 'Cancel uninstall') : op.phase === 'download' ? copy('取消下载', 'Cancel download') : copy('取消安装', 'Cancel installation')}</button>}
        {['installing', 'verifying'].includes(op.status) && <p className="text-xs text-slate-500 dark:text-slate-400">{copy('安装中，请勿关闭服务。', 'Installation in progress. Keep the service running.')}</p>}
        {op.logs.length > 0 && <InstallationLog label={copy('安装日志', 'Installation log')} logs={op.logs}/>}
      </InstallationOperation>)}
    </section>;

  if (onboardingAttention) return <div className="environment-onboarding" data-testid="environment-manager">
    <section id="environment-attention" className="settings-section">
      <div className="settings-section-heading"><h2>{copy('注意力加速', 'Attention acceleration')}</h2><button type="button" className={button} disabled={loading || locked || !!status?.maintenance} onClick={() => void refresh(true)}><RefreshCw size={14} className={probing ? 'animate-spin' : ''}/>{probing ? copy('检查中…', 'Checking…') : copy('重新检测', 'Refresh probes')}</button></div>
      <p className="settings-note" role={restartRequired ? 'status' : undefined}>{status && cpuProfile ? copy('CPU 使用 PyTorch 内置 SDPA，无需安装注意力扩展。', 'CPU uses built-in PyTorch SDPA; no attention extension is needed.') : copy('安装后需重启训练器生效。', 'Restart the trainer after installation for changes to take effect.')}</p>
      {error && <div ref={errorRef} role="alert" className="whitespace-pre-wrap rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-300">{error}</div>}
      {readFailure(copy('当前环境', 'Current runtime'), runtimeRead)}
      {loading && !status && <p role="status" className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400"><Loader2 size={16} className="animate-spin" />{copy('检测当前环境与已安装扩展…', 'Checking runtime and installed extensions…')}</p>}
      {status?.running_jobs && <p role="status" className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-950/30 dark:text-amber-200">{copy('任务运行中，完成或停止后可修改环境。', 'Finish or stop running tasks before changing the environment.')}</p>}
      <div className="settings-dependencies">{visiblePackages.map(packageItem)}</div>
    </section>
    {installation}
  </div>;

  const viewOptions = [{ id: 'image' as const, label: copy('图像训练', 'Image training') }, { id: 'speech' as const, label: copy('语音训练', 'Speech training') }];
  return <div data-testid="environment-manager">
    <SettingsSections sections={[
      { id: 'environment-runtime', label: copy('当前环境', 'Current runtime') },
      ...(imageVisible ? [
        { id: 'environment-torch', label: profile === 'linux-dtk' ? copy('DTK 安装指南', 'DTK installation guide') : copy('PyTorch 版本', 'PyTorch version') },
        { id: 'environment-attention', label: copy('注意力加速', 'Attention acceleration') },
        { id: 'environment-lora', label: copy('LoRA 环境', 'LoRA environment') },
        { id: 'environment-vision', label: copy('打标与遮罩', 'Tagging and masks') },
        ...(visibleOperations.length || torchOperationsVisible ? [{ id: 'environment-installation', label: copy('安装日志', 'Installation log') }] : []),
      ] : [{ id: 'environment-tts', label: copy('语音训练环境', 'Speech training environments') }]),
    ]}>
    <section id="environment-runtime" data-settings-section tabIndex={-1} className="settings-section">
    <div className="settings-section-heading environment-runtime-heading">
      <div className="environment-runtime-title">
        <h2 className="text-base font-semibold">{copy('环境与计算后端', 'Runtime and compute backends')}</h2>
        <div ref={viewTabs} className="environment-view-switch">
          <div className="ui-segmented" role="tablist" aria-label={copy('训练环境视图', 'Training environment views')}>
            {viewOptions.map((option, index) => <button key={option.id} id={`environment-tab-${option.id}`} type="button" role="tab" aria-selected={view === option.id} aria-controls={`environment-view-${option.id}`} tabIndex={view === option.id ? 0 : -1} onClick={() => selectView(option.id)} onKeyDown={event => {
              if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
              event.preventDefault();
              const next = event.key === 'Home' ? 0 : event.key === 'End' ? viewOptions.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + viewOptions.length) % viewOptions.length;
              selectView(viewOptions[next].id);
              document.getElementById(`environment-tab-${viewOptions[next].id}`)?.focus();
            }}>{option.label}</button>)}
            <SlidingIndicator className="ui-segmented-thumb"/>
          </div>
        </div>
      </div>
      <button className={button} disabled={loading || busy} onClick={() => { void refreshRuntime(true); if (imageVisible) { void refreshOperations(); void refreshLatest(true); void refreshLora(true); } }}><RefreshCw size={14} className={loading ? 'animate-spin' : ''} />{copy('重新检测', 'Refresh probes')}</button>
    </div>
    {imageVisible && error && <div ref={errorRef} role="alert" className="whitespace-pre-wrap rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-300">{error}</div>}
    {readFailure(copy('当前环境', 'Current runtime'), runtimeRead)}
    {loading && !status && <p role="status" className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400"><Loader2 size={16} className="animate-spin" />{copy('检测当前环境与已安装扩展…', 'Checking runtime and installed extensions…')}</p>}
    {status && <>
      <dl className="settings-facts">{facts?.map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-xs text-slate-500 dark:text-slate-400">{label}</dt><dd className="mt-1 break-words text-sm font-medium">{value}</dd></div>)}<div className="min-w-0"><dt className="text-xs text-slate-500 dark:text-slate-400">{copy('设备', 'Device')}</dt><dd className="mt-1 break-words text-sm font-medium">{cpuProfile ? 'CPU' : detectedDevices || (status.runtime.mps_available ? 'Apple GPU' : 'CPU')}</dd></div></dl>
      {target === 'mps' && !status.runtime.mps_available && <p className="settings-note">{copy('当前 PyTorch 无法使用 Apple MPS，请检查 macOS 与 PyTorch 环境。', 'Apple MPS is unavailable in the current PyTorch environment. Check macOS and PyTorch compatibility.')}</p>}
      {showAttentionExtensions && !status.runtime.cuda_available && <p className="settings-note">{hipBackend ? copy('当前 DTK / HIP 无法访问显卡，请检查厂商运行时、驱动及库路径后重启。', 'DTK / HIP cannot access the GPUs. Check the vendor runtime, driver and library paths, then restart.') : status.runtime.cuda_runtime ? copy(`当前 PyTorch 含 CUDA ${status.runtime.cuda_runtime}，但无法使用 CUDA。请检查显卡驱动后重启。`, `PyTorch includes CUDA ${status.runtime.cuda_runtime}, but CUDA is unavailable. Check the GPU driver and restart.`) : copy('当前 PyTorch 未提供 CUDA 运行时，请检查此 CUDA 环境的 PyTorch 安装。', 'This PyTorch build has no CUDA runtime. Check the PyTorch installation in this CUDA environment.')}</p>}
      {status.running_jobs && <p role="status" className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-950/30 dark:text-amber-200">{copy('任务运行中，完成或停止后可修改环境。', 'Finish or stop running tasks before changing the environment.')}</p>}
    </>}
    </section>
    <ResourceActivityContext.Provider value={parentActive && imageVisible}>
    <div id="environment-view-image" role="tabpanel" aria-labelledby="environment-tab-image" hidden={!imageVisible} className="environment-view-content">
    {status && <>
      {profile === 'linux-dtk' ? <DtkRuntimePanel/> : <TorchEnvironmentPanel showAttentionExtensions={showAttentionExtensions} disabled={status.running_jobs || busy || operations.some(busyStatus)} statusUnavailable={!operationsRead.data || !!operationsRead.error || !!runtimeRead.error} operationsTarget={torchOperationsTarget} onOperationsVisible={setTorchOperationsVisible}/>}
      <section id="environment-attention" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{copy('注意力加速', 'Attention acceleration')}</h2>{target !== 'mps' && <p className="settings-note">{hipBackend ? copy('内置 PyTorch SDPA；FlashAttention、xFormers 需使用匹配 DTK / PyTorch 的厂商构建。', 'PyTorch SDPA is built in. FlashAttention and xFormers require vendor builds matching DTK / PyTorch.') : showAttentionExtensions ? copy('内置 PyTorch SDPA；可选扩展用于 CUDA 加速。', 'PyTorch SDPA is built in. Optional extensions provide CUDA acceleration.') : copy('CPU 使用 PyTorch 内置 SDPA，无需安装额外注意力扩展。', 'CPU uses built-in PyTorch SDPA; no additional attention extension is needed.')}</p>}</div>{!cpuProfile && <button type="button" className={button} disabled={loading || locked || status.maintenance} title={status.running_jobs ? copy('任务结束或暂停后可运行检查', 'Run checks after the task finishes or pauses') : undefined} onClick={() => void refresh(true)}><RefreshCw size={14} className={probing ? 'animate-spin' : ''}/>{probing ? copy('检查中…', 'Checking…') : copy('运行检查', 'Run checks')}</button>}</div>
      {(hipBackend || target === 'mps') && <div className="settings-sdpa-status" data-testid="environment-sdpa">
        <div className="flex flex-wrap items-center justify-between gap-2"><strong className="text-sm">{target === 'mps' ? copy('Apple GPU 内置加速', 'Built-in Apple GPU acceleration') : 'PyTorch SDPA'}</strong><span className={`text-xs ${status.sdpa?.status === 'passed' ? 'text-emerald-700 dark:text-emerald-400' : status.sdpa?.status === 'failed' ? 'text-amber-700 dark:text-amber-300' : 'text-slate-500 dark:text-slate-400'}`}>{status.sdpa?.status === 'passed' ? copy('检测通过', 'Check passed') : status.sdpa?.status === 'failed' ? target === 'mps' ? copy('检测失败', 'Check failed') : copy('当前计算路径不可用', 'Current compute path unavailable') : status.probe_deferred ? copy('任务运行中，检测已延后', 'Probe deferred while a job runs') : copy('尚未检测', 'Not tested yet')}</span></div>
        {target === 'mps' && <p className="settings-note">{copy('使用 PyTorch 自带的 SDPA，无需额外安装。', 'Uses SDPA included with PyTorch. No extra installation is needed.')}</p>}
        {hipBackend && status.sdpa?.reason === 'hip_sdpa_flash_library_missing' ? <div className="mt-2 space-y-2"><p className="settings-note">{copy('当前厂商 PyTorch 的这条 SDPA 路径需要 FlashAttention 动态库。请安装匹配当前 DTK / PyTorch 的官方 FlashAttention 包，重启后重新检测。', 'This vendor PyTorch SDPA path needs a FlashAttention library. Install the official build matching DTK / PyTorch, restart, and check again.')}</p><button type="button" className={button} disabled={locked} onClick={() => { setSelected('flash-attn'); setVersion(''); setWheel(null); setVendorWheel(null); document.getElementById('environment-package-flash-attn')?.scrollIntoView?.({ block: 'nearest' }); }}>{copy('查看匹配的 FlashAttention 包', 'View matching FlashAttention builds')}</button></div> : status.sdpa?.error && <p role="alert" className="settings-note whitespace-pre-wrap break-words">{status.sdpa.error}</p>}
        {hipBackend && status.sdpa?.error && status.sdpa.reason === 'hip_sdpa_flash_library_missing' && <details className="settings-inline-details"><summary>{copy('查看检测详情', 'Probe details')}</summary><pre className="whitespace-pre-wrap break-words text-xs">{status.sdpa.detail || status.sdpa.error}</pre></details>}
      </div>}
      <div className="settings-dependencies">{visiblePackages.map(packageItem)}{tritonPackages.map(packageItem)}</div></section>
    </>}
    <section id="environment-lora" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><h2>{copy('LoRA 环境', 'LoRA environment')}</h2>{status && optimizerPackages.length > 0 && <button type="button" className={button} disabled={loading || locked || status.maintenance} title={status.running_jobs ? copy('任务结束或暂停后可运行检查', 'Run checks after the task finishes or pauses') : undefined} onClick={() => void refresh(true)}><RefreshCw size={14} className={probing ? 'animate-spin' : ''}/>{probing ? copy('检查中…', 'Checking…') : copy('运行检查', 'Run checks')}</button>}</div>
      <div className="settings-dependencies"><div className="settings-dependency">
        <div className="settings-dependency-row settings-dependency-row-reference" data-testid="environment-lycoris">
          <div className="settings-dependency-info"><span className="settings-dependency-name settings-dependency-name-static">LyCORIS</span></div>
          <dl className="settings-dependency-version text-xs">
            <div><dt>{copy('本地服务端版本：', 'Local server version:')}</dt><dd>{lora?.local.version
              ? copy(`内置实现 · 格式参考 ${[lora.local.version, lora.local.commit].filter(Boolean).join(' · ')}`, `Built in · format reference ${[lora.local.version, lora.local.commit].filter(Boolean).join(' · ')}`) : copy('内置实现', 'Built in')}</dd></div>
            <div><dt>{copy('云端版本：', 'Online version:')}</dt><dd>{onlineLookup(lora?.upstream.version ? [lora.upstream.version, lora.upstream.commit].filter(Boolean).join(' · ') : null, !!(loraError || lora?.upstream.error), loraRead, 'LyCORIS')}</dd></div>
            {!!lora?.upstream.head && <div><dt>{copy('主分支：', 'Main branch:')}</dt><dd>{[lora.upstream.head, lora.upstream.head_date].filter(Boolean).join(' · ')}</dd></div>}
          </dl>
          <div className="settings-dependency-actions flex justify-end"><a className={`${button} ui-btn-icon`} href="https://github.com/KohakuBlueleaf/LyCORIS" target="_blank" rel="noreferrer" aria-label={`LyCORIS ${copy('代码仓库', 'repository')}`}><ExternalLink size={12} /></a></div>
        </div>
      </div>{optimizerPackages.map(packageItem)}</div>
    </section>
    <section id="environment-vision" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><h2>{copy('打标与遮罩', 'Tagging and masks')}</h2></div>
      {status ? <div className="settings-dependencies">{visionPackages.map(packageItem)}</div> : !runtimeRead.error && <LoadingNote label={copy('检测扩展包…', 'Checking packages…')}/>}
    </section>
    {installation}
    </div>
    </ResourceActivityContext.Provider>
    <ResourceActivityContext.Provider value={parentActive && !imageVisible}>
      <div id="environment-view-speech" role="tabpanel" aria-labelledby="environment-tab-speech" hidden={imageVisible} className="environment-view-content">
    <section id="environment-tts" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><h2>{copy('语音训练环境', 'Speech training environments')}</h2></div>
      <TtsEnvironmentPanel/>
    </section>
      </div>
    </ResourceActivityContext.Provider>
  </SettingsSections></div>;
}
