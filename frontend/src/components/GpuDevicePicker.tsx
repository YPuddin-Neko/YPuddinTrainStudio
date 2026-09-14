import React from 'react';
import { Cpu, ChevronDown, RefreshCw } from 'lucide-react';
import type { QueueDevice } from '../api/types';
import { useQueueDevices } from '../api/hooks/useQueueDevices';
import { gpuDeviceLabel } from '../utils/gpuDevices';
import { useWorkspaceText } from '../utils/workspaceText';
import StudioSelect from './StudioSelect';
import '../styles/gpu-device-picker.css';

export default function GpuDevicePicker({ value, onChange, count = 1, disabled = false, compact = false, onValidityChange }: {
  value: string[]; onChange: (devices: string[]) => void; count?: number; disabled?: boolean; compact?: boolean;
  onValidityChange?: (valid: boolean) => void;
}) {
  const text = useWorkspaceText();
  const { snapshot, error, loading, refresh } = useQueueDevices();
  const popover = React.useRef<HTMLDetailsElement>(null);
  const devices = snapshot?.devices || [];
  const wrongCount = value.length > 0 && value.length !== count;
  const missing = snapshot !== null && value.some(id => !devices.some(device => device.device === id && device.status !== 'unavailable'));
  const valid = !wrongCount && !missing;
  React.useEffect(() => { onValidityChange?.(valid); }, [valid, onValidityChange]);
  React.useEffect(() => {
    if (!compact) return;
    const close = (event: MouseEvent | KeyboardEvent) => {
      if (!popover.current?.open) return;
      if (event instanceof KeyboardEvent) { if (event.key === 'Escape') popover.current.open = false; return; }
      const target = event.target as Node;
      const ownedMenu = Array.from(popover.current.querySelectorAll('[aria-controls]')).some(trigger => document.getElementById(trigger.getAttribute('aria-controls') || '')?.contains(target));
      if (!popover.current.contains(target) && !ownedMenu) popover.current.open = false;
    };
    document.addEventListener('mousedown', close); document.addEventListener('keydown', close);
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', close); };
  }, [compact]);
  const statusLabel = (device: QueueDevice) => device.job_id
    ? text(`占用中 · ${device.job_name || device.job_id}`, `Busy · ${device.job_name || device.job_id}`)
    : device.status === 'unavailable' ? text('不可用', 'Unavailable') : text('空闲', 'Free');
  const describe = (device: QueueDevice) => `${gpuDeviceLabel(device.device)} · ${device.name} · ${statusLabel(device)}`;
  const body = <div className="gpu-picker-body">
    <div className="gpu-picker-heading"><strong>{text('运行显卡', 'Run on GPU')}</strong><button type="button" disabled={disabled || loading} onClick={refresh} aria-label={text('刷新显卡状态', 'Refresh GPU status')}><RefreshCw size={13} className={loading ? 'animate-spin' : ''}/></button></div>
    {count === 1 ? <StudioSelect aria-label={text('运行显卡', 'Run on GPU')} value={value[0] || ''} disabled={disabled} onValueChange={id => onChange(id ? [id] : [])} options={[
      { value: '', label: text('自动选择空闲显卡', 'Automatically choose a free GPU') },
      ...devices.map(device => ({ value: device.device, label: describe(device), displayLabel: gpuDeviceLabel(device.device), disabled: device.status === 'unavailable' })),
      ...value.filter(id => !devices.some(device => device.device === id)).map(id => ({ value: id, label: `${gpuDeviceLabel(id)} · ${text('暂不可用', 'Unavailable')}`, disabled: true })),
    ]}/> : <>
      <label className="gpu-picker-auto"><input type="checkbox" checked={value.length === 0} disabled={disabled} onChange={event => onChange(event.target.checked ? [] : devices.filter(device => device.status !== 'unavailable').slice(0, count).map(device => device.device))}/>{text(`自动选择 ${count} 张空闲显卡`, `Automatically choose ${count} free GPUs`)}</label>
      <div className="gpu-picker-cards" role="group" aria-label={text('选择训练显卡', 'Choose training GPUs')}>{devices.map(device => <label key={device.device}><input type="checkbox" disabled={disabled || device.status === 'unavailable' || !value.includes(device.device) && value.length >= count} checked={value.includes(device.device)} onChange={event => onChange(event.target.checked ? [...value, device.device] : value.filter(id => id !== device.device))}/><span><strong>{gpuDeviceLabel(device.device)} · {device.name}</strong><small>{statusLabel(device)}{device.mem_free_mb != null && ` · ${text('可用', 'Free')} ${(device.mem_free_mb / 1024).toFixed(1)} GiB`}</small></span></label>)}</div>
    </>}
    {loading && !snapshot && <p>{text('读取显卡状态…', 'Reading GPU status…')}</p>}
    {error && <p role="status">{text('显卡状态暂时无法读取；自动选择仍由队列处理。', 'GPU status is unavailable; automatic selection is handled by the queue.')} <span className="gpu-picker-error-detail">{error}</span></p>}
    {snapshot && !devices.length && <p>{text('未检测到可调度显卡，将使用当前环境的默认设备。', 'No schedulable GPU was detected; the current environment will use its default device.')}</p>}
    {!valid && <p role="alert" className="gpu-picker-warning">{missing ? text('所选显卡已不可用，请重新选择。', 'A selected GPU is unavailable. Choose again.') : text(`当前训练需要选择 ${count} 张显卡，也可以改为自动选择。`, `Select ${count} GPUs for this run, or use automatic selection.`)}</p>}
    {value.some(id => devices.find(device => device.device === id)?.job_id) && <p>{text('所选显卡正在使用，任务会等待它空闲后启动。', 'A selected GPU is busy. This job will wait until it is free.')}</p>}
    {snapshot?.max_concurrent === 1 && devices.length > 1 && <p>{text('队列目前限制为同时运行 1 个任务；需要并行时，在队列调度设置中改为自动。', 'The queue currently allows one job at a time. Use automatic concurrency in queue settings to run on separate GPUs in parallel.')}</p>}
  </div>;
  if (!compact) return <section className="gpu-picker" aria-label={text('显卡选择', 'GPU selection')}>{body}</section>;
  return <details className="gpu-picker gpu-picker-compact" ref={popover}><summary aria-label={text('选择运行显卡', 'Choose run GPUs')}><Cpu size={14}/><span>{value.length ? value.map(gpuDeviceLabel).join(', ') : text('自动选卡', 'Auto GPU')}{wrongCount ? ' !' : ''}</span><ChevronDown size={12}/></summary>{body}</details>;
}
