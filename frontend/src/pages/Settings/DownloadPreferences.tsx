import React from 'react';
import { RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { components } from '../../api/generated';
import ConfigHelp from '../../components/ConfigHelp';
import StudioSelect, { type StudioSelectOption } from '../../components/StudioSelect';
import Switch from '../../components/Switch';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import './download-preferences.css';

export type DownloadSources = components['schemas']['SettingsDownloads'];
type SourceProbe = components['schemas']['DownloadSourceProbe'];
type SourcesProbe = components['schemas']['DownloadSourcesProbe'];

interface DownloadSourceProps {
  value: DownloadSources;
  onChange: (value: DownloadSources) => void;
  disabled?: boolean;
}

const pypiUrls = {
  ustc: 'https://mirrors.ustc.edu.cn/pypi/simple',
  tuna: 'https://pypi.tuna.tsinghua.edu.cn/simple',
  aliyun: 'https://mirrors.aliyun.com/pypi/simple',
  official: 'https://pypi.org/simple',
};

export function DownloadSourceFields({ value, onChange, disabled = false }: DownloadSourceProps) {
  const text = useWorkspaceText();
  const id = React.useId();
  const [platform, setPlatform] = React.useState('');
  const [hip, setHip] = React.useState(false);
  const [probe, setProbe] = React.useState<SourcesProbe | null>(null);
  const [probing, setProbing] = React.useState(false);
  const [error, setError] = React.useState('');
  const probeRequest = React.useRef<AbortController | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();
    void apiClient.get<{ platform?: string; hip?: string | null }>('/system/info', { signal: controller.signal, silent: true })
      .then(info => { if (!controller.signal.aborted) { setPlatform(info.platform || ''); setHip(!!info.hip); } }).catch(() => {});
    return () => { controller.abort(); probeRequest.current?.abort(); };
  }, []);

  const checkSources = async () => {
    if (disabled || probeRequest.current) return;
    const controller = new AbortController();
    probeRequest.current = controller;
    setProbing(true);
    setProbe(null);
    setError('');
    try {
      const result = await apiClient.post<SourcesProbe>('/settings/downloads/probe', undefined, { signal: controller.signal, silent: true });
      if (!controller.signal.aborted) setProbe(result);
    } catch (cause) {
      if (!controller.signal.aborted) setError(formatApiError(cause));
    } finally {
      if (!controller.signal.aborted) setProbing(false);
      if (probeRequest.current === controller) probeRequest.current = null;
    }
  };

  const pypiOptions: StudioSelectOption[] = [
    { value: 'auto', label: text('自动选择', 'Automatic') },
    { value: 'ustc', label: text('中国科学技术大学', 'USTC') },
    { value: 'tuna', label: text('清华大学', 'Tsinghua University') },
    { value: 'aliyun', label: text('阿里云', 'Aliyun') },
    { value: 'official', label: text('PyPI 官方', 'Official PyPI') },
  ];
  const pytorchOptions: StudioSelectOption[] = [
    { value: 'auto', label: text('自动选择', 'Automatic') },
    ...(value.pytorch === 'mirror' ? [{ value: 'mirror', label: text('国内镜像（上海交大 → 阿里云）', 'Mirrors (SJTU → Aliyun)') }] : []),
    { value: 'aliyun', label: text('阿里云', 'Aliyun') },
    { value: 'sjtu', label: text('上海交通大学', 'Shanghai Jiao Tong University') },
    { value: 'official', label: text('PyTorch 官方', 'Official PyTorch') },
  ];
  const latency = (source: SourceProbe) => !source.available ? text('无法连接', 'Unreachable')
    : source.latency_ms !== null && Number.isFinite(source.latency_ms) ? `${Math.round(source.latency_ms)} ms` : text('可连接', 'Reachable');
  const measuredOptions = (options: StudioSelectOption[], results?: SourceProbe[]) => options.map(option => {
    const source = results?.find(result => result.id === option.value);
    return source ? { ...option, displayLabel: option.label, label: `${option.label} · ${latency(source)}` } : option;
  });
  const resultNote = (selected: string, options: StudioSelectOption[], results?: SourceProbe[]) => {
    if (!results) return null;
    const source = selected === 'auto'
      ? results.filter(result => result.available && result.latency_ms !== null && Number.isFinite(result.latency_ms)).sort((a, b) => a.latency_ms! - b.latency_ms!)[0]
      : results.find(result => result.id === selected);
    if (source) return <p className="settings-note download-source-result" data-unavailable={!source.available || undefined}>
      {selected === 'auto' ? `${text('最低延迟：', 'Lowest latency: ')}${options.find(option => option.value === source.id)?.label || source.name} · ` : text('响应延迟：', 'Response latency: ')}{latency(source)}
    </p>;
    return selected === 'auto' && !results.some(result => result.available)
      ? <p className="settings-note download-source-result" data-unavailable>{text('没有可连接的下载源。', 'No download sources are reachable.')}</p> : null;
  };
  // Apple silicon installs PyTorch from the Python package source; DTK uses the vendor's PyTorch.
  const mac = /macOS|Darwin/i.test(platform);
  const pytorchSource = !mac && !hip;

  return <div className="download-source-fields">
    <div className="download-source-probe">
      <p className="settings-note">{text('检测服务器到各下载源的响应延迟。', 'Check response latency from the server to each source.')}</p>
      <button type="button" className="ui-btn ui-btn-secondary" disabled={disabled || probing} onClick={() => void checkSources()}>
        <RefreshCw size={14} className={probing ? 'animate-spin' : undefined} aria-hidden="true" />
        {probing ? text('正在检测…', 'Checking…') : text('检测延迟', 'Check latency')}
      </button>
    </div>
    {error && <p role="alert" className="download-source-error">{error}</p>}
    <div className="settings-field">
      <span className="settings-field-label download-source-label"><label htmlFor={`${id}-pypi`}>{text('Python 依赖包', 'Python packages')}</label>
        <ConfigHelp label={text('Python 依赖包来源说明', 'Python package source help')}>{text(
          '训练器依赖、运行环境页的扩展和网页更新的依赖从这里下载。\n自动选择会检测各来源的响应速度，从最快的开始尝试；指定来源时先使用它。开启“下载失败时”自动换源后，失败会继续尝试其他镜像和 PyPI 官方。中科大、清华和阿里云是国内镜像。'
          + (mac ? '\nApple 芯片的 PyTorch 也从这里安装。' : ''),
          'Trainer dependencies, extensions from the runtime page and the dependencies of web updates are downloaded from here.\nAutomatic tests the response of each source and tries the fastest first; a chosen source is tried first. With fallback on, failures move on to the other mirrors and official PyPI. USTC, Tsinghua and Aliyun are mirrors in mainland China.'
          + (mac ? '\nPyTorch for Apple silicon is installed from here as well.' : ''),
        )}</ConfigHelp></span>
      <div className="settings-field-control">
        <StudioSelect id={`${id}-pypi`} disabled={disabled} value={value.pypi} onValueChange={pypi => onChange({ ...value, pypi: pypi as DownloadSources['pypi'] })} options={measuredOptions(pypiOptions, probe?.pypi)} />
        <p className="settings-note">{text('安装训练器依赖和扩展时使用。', 'Used for trainer dependencies and extensions.')}</p>
        {value.pypi !== 'auto' && <p className="settings-note break-all font-mono">{pypiUrls[value.pypi]}</p>}
        <div aria-live="polite">{resultNote(value.pypi, pypiOptions, probe?.pypi)}</div>
      </div>
    </div>
    {pytorchSource && <div className="settings-field">
      <span className="settings-field-label download-source-label"><label htmlFor={`${id}-pytorch`}>PyTorch</label>
        <ConfigHelp label={text('PyTorch 来源说明', 'PyTorch source help')}>{text(
          'CUDA 和 CPU 版 PyTorch 来自单独的仓库，用于在运行环境页准备其他 PyTorch 版本和安装 xFormers。\n上海交通大学和阿里云是国内镜像，官方为 download.pytorch.org。自动选择会检测响应速度；开启自动换源后，失败会继续尝试其他来源。',
          'CUDA and CPU builds of PyTorch come from separate repositories. They are used to prepare other PyTorch versions on the runtime page and to install xFormers.\nShanghai Jiao Tong University and Aliyun are mirrors in mainland China; the official source is download.pytorch.org. Automatic tests their response; with fallback on, failures move on to the other sources.',
        )}</ConfigHelp></span>
      <div className="settings-field-control">
        <StudioSelect id={`${id}-pytorch`} disabled={disabled} value={value.pytorch} onValueChange={pytorch => onChange({ ...value, pytorch: pytorch as DownloadSources['pytorch'] })} options={measuredOptions(pytorchOptions, probe?.pytorch)} />
        <p className="settings-note">{text('安装 PyTorch 和 xFormers 时使用。', 'Used for PyTorch and xFormers.')}</p>
        <div aria-live="polite">{resultNote(value.pytorch, pytorchOptions, probe?.pytorch)}</div>
      </div>
    </div>}
    <div className="settings-field">
      <span className="settings-field-label">{text('下载失败时', 'On download failure')}</span>
      <div className="settings-field-control"><Switch checked={value.fallback} disabled={disabled} onCheckedChange={fallback => onChange({ ...value, fallback })}>{text('自动尝试其他镜像和官方源', 'Try other mirrors and the official source')}</Switch></div>
    </div>
  </div>;
}

export default function DownloadPreferences(props: DownloadSourceProps) {
  const text = useWorkspaceText();
  return <section id="preferences-downloads" data-settings-section tabIndex={-1} className="settings-section">
    <div className="settings-section-heading"><div><h2>{text('软件下载源', 'Package sources')}</h2><p className="settings-note">{text('保存后用于新的软件安装。', 'Applies to new installations after saving.')}</p></div></div>
    <DownloadSourceFields {...props} />
  </section>;
}
