import React from 'react';
import { Download, ExternalLink, Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '../api/client';
import { formatApiError } from '../utils/errors';
import { useWorkspaceText } from '../utils/workspaceText';
import type { DtkCatalog } from './DtkWheelPicker';

const toolkitSource = 'https://download.sourcefind.cn:65024/1/main';
const driverSource = 'https://download.sourcefind.cn:65024/6/main';
const linkButton = 'settings-input inline-flex items-center justify-center gap-1.5';

export default function DtkRuntimePanel() {
  const text = useWorkspaceText();
  const [catalog, setCatalog] = React.useState<DtkCatalog | null>(null);
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(true);
  const [revision, setRevision] = React.useState(0);
  React.useEffect(() => {
    let active = true;
    setLoading(true); setError(''); setCatalog(null);
    void apiClient.get<DtkCatalog>('/environment/dtk/wheels', {silent: true}).then(value => {
      if (active) setCatalog(value);
    }).catch(err => { if (active) setError(formatApiError(err)); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [revision]);
  const runtime = catalog?.runtime;
  const guidance = catalog?.guidance;
  const recommendation = guidance?.recommendation;
  return <section id="environment-torch" data-settings-section tabIndex={-1} className="settings-section" data-testid="dtk-runtime-guidance">
    <div className="settings-section-heading"><h2>{text('DTK 安装指南', 'DTK installation guide')}</h2>
      <button type="button" className={linkButton} disabled={loading} onClick={() => setRevision(value => value + 1)}><RefreshCw size={13}/>{text('刷新推荐', 'Refresh recommendations')}</button>
    </div>
    {loading && !catalog && <p role="status" className="settings-note flex items-center gap-2"><Loader2 size={14} className="animate-spin motion-reduce:animate-none"/>{text('正在查找适合本机的安装包…', 'Finding packages for this system…')}</p>}
    {error && <p role="alert" className="settings-note break-words">{error}</p>}
    {runtime && <div className="settings-field"><span className="settings-field-label">{text('当前系统', 'Current system')}</span><div className="settings-field-control">
      <p className="break-words text-sm">{[runtime.distribution || runtime.distribution_version, runtime.machine].filter(Boolean).join(' · ')}</p>
      <p className="settings-note">DTK {runtime.installed_dtk || runtime.dtk || '—'} · Python {runtime.python} · PyTorch {runtime.torch}</p>
    </div></div>}
    {guidance?.current_stack_reason === 'torch24_transformers5_diffusers040_conflict' && <p className="settings-note">{text('当前 PyTorch 2.4 不满足新版 Diffusers 和 Transformers 的要求，升级时需使用配套组合。', 'PyTorch 2.4 does not meet the requirements of newer Diffusers and Transformers. Upgrade these packages as a matching set.')}</p>}
    <div className="settings-field"><span className="settings-field-label">{text('当前驱动', 'Current driver')}</span><div className="settings-field-control">
      <p className="break-words text-sm">{guidance?.driver_version || text('未检测到版本', 'Version not detected')}</p>
      {recommendation && <p className="settings-note">{text(`DTK ${recommendation.dtk} 要求驱动 ${recommendation.minimum_driver}。`, `DTK ${recommendation.dtk} requires driver ${recommendation.minimum_driver}.`)}</p>}
    </div></div>
    {recommendation ? <div className="space-y-3" data-testid="dtk-runtime-recommendation">
      <div className="settings-field"><span className="settings-field-label">{text('可选的 DTK 版本', 'Available DTK version')}</span><div className="settings-field-control">
        <p className="text-sm font-medium">DTK {recommendation.dtk} · Python {runtime?.python.split('.').slice(0,2).join('.') || recommendation.python_tag}</p>
      </div></div>
      <div className="flex flex-wrap gap-2">
        <a className={linkButton} href={recommendation.toolkit_url} target="_blank" rel="noreferrer"><Download size={13}/>{text(`下载 DTK ${recommendation.dtk}`, `Download DTK ${recommendation.dtk}`)}</a>
        <a className={linkButton} href={recommendation.toolkit_checksum_url} target="_blank" rel="noreferrer">{text('下载校验文件', 'Download checksum file')}<ExternalLink size={12}/></a>
      </div>
      <dl>{recommendation.wheels.map(wheel => <div key={wheel.package} className="settings-field">
        <dt className="settings-field-label">{{torch:'PyTorch',torchvision:'TorchVision',triton:'Triton','flash-attn':'FlashAttention'}[wheel.package] || wheel.package}</dt>
        <dd className="settings-field-control flex flex-wrap items-center justify-between gap-2"><span className="min-w-0 break-all text-sm">{wheel.version}</span><a className={linkButton} href={wheel.url} target="_blank" rel="noreferrer" aria-label={text(`手动下载 ${wheel.package} ${wheel.version}`, `Download ${wheel.package} ${wheel.version} manually`)}><Download size={13}/>{text('下载', 'Download')}</a></dd>
      </div>)}</dl>
    </div> : !loading && !error && <p className="settings-note">{text('暂未找到适合本机的配套版本。请前往官方目录，按系统版本选择安装包。', 'No matching package set was found. Choose packages for your OS in the official catalog.')}</p>}
    <div className="mt-3 flex flex-wrap gap-2">
      <a className={linkButton} href={guidance?.toolkit_source_url || toolkitSource} target="_blank" rel="noreferrer">{text('DTK 版本目录', 'DTK versions')}<ExternalLink size={12}/></a>
      <a className={linkButton} href={guidance?.driver_source_url || driverSource} target="_blank" rel="noreferrer">{text('驱动下载目录', 'Driver downloads')}<ExternalLink size={12}/></a>
      {guidance?.compatibility_source_url && <a className={linkButton} href={guidance.compatibility_source_url} target="_blank" rel="noreferrer">{text('驱动配套表', 'Driver compatibility')}<ExternalLink size={12}/></a>}
    </div>
    <p className="settings-note mt-3">{text('DTK 和驱动需在服务器上安装；注意力扩展可在下方安装。', 'Install DTK and drivers on the server; install attention extensions below.')}</p>
  </section>;
}
