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
    <div className="settings-section-heading"><h2>{text('DTK 安装与更新', 'DTK installation and updates')}</h2>
      <button type="button" className={linkButton} disabled={loading} onClick={() => setRevision(value => value + 1)}><RefreshCw size={13}/>{text('重新匹配', 'Match again')}</button>
    </div>
    {loading && !catalog && <p role="status" className="settings-note flex items-center gap-2"><Loader2 size={14} className="animate-spin motion-reduce:animate-none"/>{text('正在匹配本机安装包…', 'Matching packages for this system…')}</p>}
    {error && <p role="alert" className="settings-note break-words">{error}</p>}
    {runtime && <div className="settings-field"><span className="settings-field-label">{text('匹配依据', 'Matched against')}</span><div className="settings-field-control">
      <p className="break-words text-sm">{[runtime.distribution || runtime.distribution_version, runtime.machine].filter(Boolean).join(' · ')}</p>
      <p className="settings-note">DTK {runtime.installed_dtk || runtime.dtk || '—'} · Python {runtime.python} · PyTorch {runtime.torch}</p>
    </div></div>}
    {guidance?.current_stack_reason === 'torch24_transformers5_diffusers040_conflict' && <p className="settings-note">{text('当前 PyTorch 2.4 不满足新版 Diffusers 和 Transformers 的要求，升级时需使用配套组合。', 'PyTorch 2.4 does not meet the requirements of newer Diffusers and Transformers. Upgrade these packages as a matching set.')}</p>}
    <div className="settings-field"><span className="settings-field-label">{text('当前驱动', 'Current driver')}</span><div className="settings-field-control">
      <p className="break-words text-sm">{guidance?.driver_version || text('未检测到版本', 'Version not detected')}</p>
      {recommendation && <p className="settings-note">{text(`候选 DTK ${recommendation.dtk} 要求驱动 ${recommendation.minimum_driver}；需人工核对。`, `Candidate DTK ${recommendation.dtk} requires driver ${recommendation.minimum_driver}; verify it manually.`)}</p>}
    </div></div>
    {recommendation ? <div className="space-y-3" data-testid="dtk-runtime-recommendation">
      <div className="settings-field"><span className="settings-field-label">{text('建议验证的版本', 'Suggested version to validate')}</span><div className="settings-field-control">
        <p className="text-sm font-medium">DTK {recommendation.dtk} · Python {runtime?.python.split('.').slice(0,2).join('.') || recommendation.python_tag}</p>
        <p className="settings-note">{text('按本机系统和 Python 匹配。请在独立环境中按整套版本准备，再核对驱动并验证训练。', 'Matched to this system and Python. Prepare the complete version set in a separate environment, then verify the driver and training.')}</p>
      </div></div>
      <div className="flex flex-wrap gap-2">
        <a className={linkButton} href={recommendation.toolkit_url} target="_blank" rel="noreferrer"><Download size={13}/>{text(`下载 DTK ${recommendation.dtk}`, `Download DTK ${recommendation.dtk}`)}</a>
        <a className={linkButton} href={recommendation.toolkit_checksum_url} target="_blank" rel="noreferrer">{text('校验文件', 'Checksum file')}<ExternalLink size={12}/></a>
      </div>
      <dl>{recommendation.wheels.map(wheel => <div key={wheel.package} className="settings-field">
        <dt className="settings-field-label">{{torch:'PyTorch',torchvision:'TorchVision',triton:'Triton','flash-attn':'FlashAttention'}[wheel.package] || wheel.package}</dt>
        <dd className="settings-field-control flex flex-wrap items-center justify-between gap-2"><span className="min-w-0 break-all text-sm">{wheel.version}</span><a className={linkButton} href={wheel.url} target="_blank" rel="noreferrer" aria-label={text(`手动下载 ${wheel.package} ${wheel.version}`, `Download ${wheel.package} ${wheel.version} manually`)}><Download size={13}/>{text('下载', 'Download')}</a></dd>
      </div>)}</dl>
    </div> : !loading && !error && <p className="settings-note">{text('暂无已核对的完整组合，请按当前系统版本在官方目录选择安装包。', 'No reviewed combination is available for this system. Choose packages for your OS in the official catalog.')}</p>}
    <div className="mt-3 flex flex-wrap gap-2">
      <a className={linkButton} href={guidance?.toolkit_source_url || toolkitSource} target="_blank" rel="noreferrer">{text('DTK 版本目录', 'DTK versions')}<ExternalLink size={12}/></a>
      <a className={linkButton} href={guidance?.driver_source_url || driverSource} target="_blank" rel="noreferrer">{text('驱动下载目录', 'Driver downloads')}<ExternalLink size={12}/></a>
      {guidance?.compatibility_source_url && <a className={linkButton} href={guidance.compatibility_source_url} target="_blank" rel="noreferrer">{text('驱动配套表', 'Driver compatibility')}<ExternalLink size={12}/></a>}
    </div>
    <p className="settings-note mt-3">{text('DTK 和驱动需在训练服务器手动安装。下方支持上传 xFormers 与 FlashAttention 的 wheel；PyTorch 等运行时包请按配套组合准备。', 'Install DTK and drivers manually on the training server. Upload xFormers and FlashAttention wheels below; prepare PyTorch and other runtime packages as a matching set.')}</p>
  </section>;
}
