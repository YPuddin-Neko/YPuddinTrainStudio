import React from 'react';
import { ExternalLink, Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '../api/client';
import { formatApiError } from '../utils/errors';
import { formatBytes } from '../utils/format';
import { useWorkspaceText } from '../utils/workspaceText';
import StudioSelect from './StudioSelect';

export interface WindowsAttentionWheel {
  id: string; package: string; version: string; filename: string; url: string; source_url: string;
  provider: string; size_bytes: number; sha256: string; torch: string; cuda: string;
  python_tag: string; platform_tag: string; validation: string; compatible: boolean; reason: string | null;
  release?: string;
}
export interface WindowsAttentionCatalog {
  source_url: string; release: string; provider: string; origin: 'live' | 'cached' | 'bundled';
  checked_at: number | null; error: string | null; reason: string | null;
  runtime: Record<string, string | null>; wheels: WindowsAttentionWheel[];
  release_count?: number; limited?: boolean; unverified_assets?: number;
}
const source = 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases';

export default function WindowsAttentionWheelPicker({ selected, onSelect, disabled }: {
  selected: WindowsAttentionWheel | null; onSelect: (wheel: WindowsAttentionWheel | null) => void; disabled: boolean;
}) {
  const text = useWorkspaceText();
  const [catalog, setCatalog] = React.useState<WindowsAttentionCatalog | null>(null);
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(true);
  const [revision, setRevision] = React.useState(0);
  React.useEffect(() => {
    let active = true;
    setLoading(true); setError('');
    void apiClient.get<WindowsAttentionCatalog>(`/environment/windows/wheels${revision ? '?refresh=true' : ''}`, {silent: true}).then(value => {
      if (active) setCatalog(value);
    }).catch(err => { if (active) setError(formatApiError(err)); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [revision]);
  const compatible = catalog?.wheels.filter(wheel => wheel.compatible) || [];
  const reasons: Record<string, [string, string]> = {
    requires_windows_cuda: ['需要 NVIDIA CUDA 环境', 'Requires NVIDIA CUDA'],
    requires_windows_x86_64: ['需要 Windows 或 Linux x86_64 系统', 'Requires Windows or Linux x86_64'],
    cuda_runtime_unavailable: ['当前 PyTorch 无法使用 CUDA', 'CUDA is unavailable in the current PyTorch'],
    python_abi_mismatch: ['Python 版本不匹配', 'Python version does not match'],
    torch_version_mismatch: ['PyTorch 版本不匹配', 'PyTorch version does not match'],
    cuda_version_mismatch: ['CUDA 构建版本不匹配', 'CUDA build does not match'],
    runtime_version_unrecognized: ['无法识别当前运行时版本', 'Runtime version is not recognized'],
  };
  return <div className="space-y-3" data-testid="windows-attention-wheels">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <strong className="text-sm">{text('社区预编译版本', 'Community builds')}</strong>
      <div className="flex flex-wrap items-center gap-2">
        <a className="studio-link inline-flex items-center gap-1 text-xs" href={catalog?.source_url || source} target="_blank" rel="noreferrer">{text('维护者发布页', 'Publisher release')}<ExternalLink size={12}/></a>
        <button type="button" className="settings-input inline-flex items-center gap-1.5" disabled={disabled || loading} onClick={() => { onSelect(null); setRevision(value => value + 1); }}><RefreshCw size={13}/>{text('刷新版本', 'Refresh builds')}</button>
      </div>
    </div>
    <p className="settings-note">{text('安装包由社区维护者 mjun0812 提供。', 'Packages provided by community maintainer mjun0812.')}</p>
    {catalog && <p className="settings-note break-words">Python {catalog.runtime.python} · PyTorch {catalog.runtime.torch} · CUDA {catalog.runtime.cuda_runtime || '—'} · {catalog.runtime.machine}{catalog.origin !== 'live' && ` · ${text(catalog.origin === 'bundled' ? `使用内置版本列表 ${catalog.release}` : '使用上次获取的版本列表', catalog.origin === 'bundled' ? `Bundled catalog ${catalog.release}` : 'Cached catalog')}`}</p>}
    {loading ? <p role="status" className="settings-note flex items-center gap-2"><Loader2 size={14} className="animate-spin"/>{text('正在查询兼容版本…', 'Looking up compatible builds…')}</p> : error ? <p role="alert" className="settings-note break-words">{error} {text('仍可从发布页下载后手动上传。', 'You can still download from the release page and upload manually.')}</p> : <>
      {catalog?.error && <p role="alert" className="settings-note break-words whitespace-pre-wrap">{catalog.error}</p>}
      {compatible.length ? <StudioSelect aria-label={text('FlashAttention 版本', 'FlashAttention build')} value={selected?.id || ''} disabled={disabled}
        placeholder={text('选择兼容版本', 'Choose a compatible build')}
        options={compatible.map(wheel => ({value:wheel.id,label:`${wheel.version} · ${wheel.python_tag}${wheel.release ? ` · ${wheel.release}` : ''} · ${formatBytes(wheel.size_bytes)}`}))}
        onValueChange={id => onSelect(compatible.find(wheel => wheel.id === id) || null)}/> : <p role="status" className="settings-note">{text('已查询的版本列表中没有适合当前环境的安装包。可上传其他兼容的 .whl 文件，再检查安装条件。', 'The queried catalog has no package matching the current environment. Upload another compatible .whl file to check its installation requirements.')}</p>}
      {selected && <div className="space-y-2"><p className="settings-note break-all">{selected.filename}</p><details className="settings-inline-details"><summary>{text('包信息与校验值', 'Build information and checksum')}</summary><dl className="settings-facts text-xs"><div><dt>Python / Torch / CUDA</dt><dd>{selected.python_tag} / {selected.torch} / {selected.cuda}</dd></div><div><dt>SHA256</dt><dd className="break-all font-mono">{selected.sha256}</dd></div></dl></details></div>}
      {!!catalog?.wheels.some(wheel => !wheel.compatible && wheel.reason !== 'platform_mismatch') && <details className="settings-inline-details"><summary>{text('其他构建与不匹配原因', 'Other builds and incompatibilities')}</summary><ul className="max-h-48 overflow-auto space-y-2 text-xs">{catalog.wheels.filter(wheel => !wheel.compatible && wheel.reason !== 'platform_mismatch').map(wheel => <li className="break-words" key={wheel.id}>{wheel.version} · {wheel.python_tag}<p className="settings-note">{wheel.reason && reasons[wheel.reason] ? text(...reasons[wheel.reason]) : wheel.reason}</p></li>)}</ul></details>}
    </>}
  </div>;
}
