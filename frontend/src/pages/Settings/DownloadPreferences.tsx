import React from 'react';
import { RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { components } from '../../api/generated';
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
  const [probe, setProbe] = React.useState<SourcesProbe | null>(null);
  const [probing, setProbing] = React.useState(false);
  const [error, setError] = React.useState('');
  const probeRequest = React.useRef<AbortController | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();
    void apiClient.get<{ platform?: string }>('/system/info', { signal: controller.signal, silent: true })
      .then(info => { if (!controller.signal.aborted) setPlatform(info.platform || ''); }).catch(() => {});
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
  const mac = /macOS|Darwin/i.test(platform);

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
      <label htmlFor={`${id}-pypi`}>{text('Python 依赖包', 'Python packages')}</label>
      <div className="settings-field-control">
        <StudioSelect id={`${id}-pypi`} disabled={disabled} value={value.pypi} onValueChange={pypi => onChange({ ...value, pypi: pypi as DownloadSources['pypi'] })} options={measuredOptions(pypiOptions, probe?.pypi)} />
        {value.pypi !== 'auto' && <p className="settings-note break-all font-mono">{pypiUrls[value.pypi]}</p>}
        <div aria-live="polite">{resultNote(value.pypi, pypiOptions, probe?.pypi)}</div>
      </div>
    </div>
    {!mac && <div className="settings-field">
      <label htmlFor={`${id}-pytorch`}>PyTorch</label>
      <div className="settings-field-control">
        <StudioSelect id={`${id}-pytorch`} disabled={disabled} value={value.pytorch} onValueChange={pytorch => onChange({ ...value, pytorch: pytorch as DownloadSources['pytorch'] })} options={measuredOptions(pytorchOptions, probe?.pytorch)} />
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
