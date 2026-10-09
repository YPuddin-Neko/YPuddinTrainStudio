import { ExternalLink, Loader2, RefreshCw } from 'lucide-react';
import { useEnvironmentRead } from './useEnvironmentRead';
import { formatBytes } from '../utils/format';
import { useWorkspaceText } from '../utils/workspaceText';
import StudioSelect from './StudioSelect';

export interface DtkWheel {
  id: string; package: string; version: string; filename: string; url: string;
  size_bytes: number; sha256: string; dtk: string; torch: string;
  python_tag: string; platform_tag: string; compatible: boolean; reason: string | null;
  binary?: boolean; validation?: string; requires_packages?: string[]; declared_torch?: string | null;
}
export interface DtkCatalog {
  source_url: string;
  runtime: {environment_profile: string; torch: string; python: string; dtk: string | null; machine: string;
    distribution?: string | null; distribution_version?: string | null; kernel_release?: string | null; glibc_version?: string | null;
    dtk_root?: string | null; installed_dtk?: string | null; hip_runtime?: string | null};
  wheels: DtkWheel[];
  reason: string | null;
  guidance?: {
    toolkit_source_url: string; driver_source_url: string; compatibility_source_url: string;
    driver_version: string | null; driver_verification: string; current_stack_reason?: string;
    recommendation: null | {dtk: string; toolkit_url: string; toolkit_checksum_url: string; python_tag: string;
      minimum_driver: string; status: string; reason: string; wheels: {package: string; version: string; url: string}[]};
  };
}

export default function DtkWheelPicker({ packageName, selected, onSelect, disabled }: {
  packageName: string; selected: DtkWheel | null; onSelect: (wheel: DtkWheel | null) => void; disabled: boolean;
}) {
  const text = useWorkspaceText();
  const { data: catalog, error, loading, reload } = useEnvironmentRead<DtkCatalog>('/environment/dtk/wheels');
  const wheels = catalog?.wheels.filter(item => item.package === packageName) || [];
  const compatible = wheels.filter(item => item.compatible);
  const reason = (value: string | null) => {
    const messages: Record<string, [string, string]> = {
      dtk_profile_required: ['请使用独立的 Linux DTK 启动入口', 'Use the dedicated Linux DTK launcher'],
      requires_linux_x86_64: ['需要 Linux x86_64 环境', 'Requires Linux x86_64'],
      hip_runtime_unavailable: ['当前 HIP 运行时无法访问显卡', 'HIP cannot access the GPUs'],
      torch_version_mismatch: ['与当前 PyTorch 版本不匹配', 'Does not match the current PyTorch version'],
      dtk_version_mismatch: ['与当前 DTK 版本不匹配', 'Does not match the current DTK version'],
      python_abi_mismatch: ['与当前 Python 版本不匹配', 'Does not match the current Python version'],
      integrity_verification_pending: ['尚未完成官方包完整性核验', 'Official build integrity verification is pending'],
    };
    if (value?.startsWith('requires_package:')) return `${text('需要先安装', 'Install first:')} ${value.slice('requires_package:'.length)}`;
    if (value?.startsWith('requires_torch_runtime_api')) return text(`当前 PyTorch 缺少此包需要的功能，要求 ${value.slice('requires_torch_runtime_api'.length)}。`, `This build needs PyTorch runtime APIs from ${value.slice('requires_torch_runtime_api'.length)}.`);
    return value && messages[value] ? text(...messages[value]) : value || text('不匹配当前运行环境', 'Incompatible with this runtime');
  };
  return <div className="space-y-3" data-testid={`dtk-wheels-${packageName}`}>
    <div className="flex flex-wrap items-center justify-between gap-2">
      <strong className="text-sm">{text('DTK 官方安装包', 'Official DTK packages')}</strong>
      <div className="flex items-center gap-2">
        {catalog?.source_url && <a className="ui-link" href={catalog.source_url} target="_blank" rel="noreferrer">{text('官方目录', 'Official catalog')}<ExternalLink size={12}/></a>}
        <button type="button" className="ui-btn ui-btn-sm" disabled={disabled || loading} onClick={() => { onSelect(null); void reload(); }}><RefreshCw size={13}/>{text('重新检查兼容版本', 'Check compatible versions again')}</button>
      </div>
    </div>
    {catalog && <p className="settings-note">DTK {catalog.runtime.dtk || '—'} · PyTorch {catalog.runtime.torch} · Python {catalog.runtime.python} · {catalog.runtime.machine}</p>}
    {loading ? <p role="status" className="settings-note flex items-center gap-2"><Loader2 size={14} className="animate-spin"/>{text('正在匹配当前环境的官方包…', 'Matching official builds to this runtime…')}</p> : error ? <p role="alert" className="break-words text-xs text-amber-700 dark:text-amber-300">{error}</p> : <>
      {compatible.length ? <StudioSelect aria-label={text('DTK 适配版本', 'DTK build')} value={selected?.id || ''} disabled={disabled}
        placeholder={text('选择适配版本', 'Choose a compatible build')}
        options={compatible.map(item => ({value:item.id,label:`${item.version} · ${item.python_tag} · ${formatBytes(item.size_bytes)}`}))}
        onValueChange={id => onSelect(compatible.find(item => item.id === id) || null)}/> : <p role="status" className="settings-note">{catalog?.reason === 'dtk_profile_required' ? text('请使用独立的 Linux DTK 启动入口后匹配安装包。', 'Use the dedicated Linux DTK launcher to match builds.') : text('官方目录中暂未找到与当前环境匹配的版本。可以上传已取得的厂商适配 wheel。', 'No matching build was found in the official catalog. You can upload a vendor-compatible wheel obtained separately.')}</p>}
      {selected && <div className="space-y-2">
        <p className="settings-note break-all">{selected.filename}</p>
        <details className="settings-inline-details"><summary>{text('包信息与运行要求', 'Build information and requirements')}</summary>
          <dl className="settings-facts text-xs">
          <div><dt>{text('发布目录 DTK 标签', 'Catalog DTK label')}</dt><dd>{selected.dtk.split(' ')[0]}</dd></div>
          <div><dt>{text('PyTorch 要求', 'PyTorch requirement')}</dt><dd>{selected.torch}</dd></div>
          {selected.declared_torch && selected.declared_torch !== selected.torch && <div><dt>{text('安装包声明的依赖', 'Declared dependencies')}</dt><dd>{selected.declared_torch}</dd></div>}
          <div><dt>{text('Python 与平台', 'Python and platform')}</dt><dd>{selected.python_tag} · {selected.platform_tag}</dd></div>
          {!!selected.requires_packages?.length && <div><dt>{text('需要已安装', 'Required packages')}</dt><dd>{selected.requires_packages.join(', ')}</dd></div>}
        </dl></details>
      </div>}
      {wheels.some(item => !item.compatible) && <details className="settings-inline-details"><summary>{text('查看其他版本与不匹配原因', 'Other versions and incompatibilities')}</summary><ul className="max-h-48 space-y-2 overflow-auto text-xs">{wheels.filter(item => !item.compatible).map(item => <li key={item.id} className="break-words"><strong>{item.version}</strong> · DTK {item.dtk.split(' ')[0]} · Torch {item.torch} · {item.python_tag}<p className="settings-note">{reason(item.reason)}</p></li>)}</ul></details>}
    </>}
  </div>;
}
