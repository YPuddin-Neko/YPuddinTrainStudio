import React from 'react';
import { createPortal } from 'react-dom';
import { Cpu, ChevronDown, RefreshCw } from 'lucide-react';
import type { QueueDevice } from '../api/types';
import { useQueueDevices } from '../api/hooks/useQueueDevices';
import { gpuDeviceLabel, gpuSelectionValid } from '../utils/gpuDevices';
import { useWorkspaceText } from '../utils/workspaceText';
import StudioSelect from './StudioSelect';
import '../styles/gpu-device-picker.css';

interface GpuDevicePickerProps {
  value: string[];
  onChange: (devices: string[]) => void;
  count?: number;
  disabled?: boolean;
  compact?: boolean;
  onValidityChange?: (valid: boolean) => void;
  /** Training selects concrete devices for the independently configured count. */
  training?: boolean;
  id?: string;
  label?: string;
  deviceState?: ReturnType<typeof useQueueDevices>;
}

export default function GpuDevicePicker(props: GpuDevicePickerProps) {
  return props.deviceState ? <DevicePicker {...props} deviceState={props.deviceState}/> : <LiveDevicePicker {...props}/>;
}
function LiveDevicePicker(props: GpuDevicePickerProps) {
  const deviceState = useQueueDevices();
  return <DevicePicker {...props} deviceState={deviceState}/>;
}
function DevicePicker({ value, onChange, count = 1, disabled = false, compact = false, onValidityChange,
  training = false, id, label, deviceState }: GpuDevicePickerProps & {deviceState: ReturnType<typeof useQueueDevices>}) {
  const text = useWorkspaceText();
  const { snapshot, error, loading, refresh } = deviceState;
  const [open, setOpen] = React.useState(false);
  const [position, setPosition] = React.useState<React.CSSProperties>({});
  const trigger = React.useRef<HTMLButtonElement>(null);
  const menu = React.useRef<HTMLDivElement>(null);
  const generatedId = React.useId();
  const menuId = `training-gpus-${generatedId}`;
  const visible = open && !disabled;
  React.useLayoutEffect(() => {
    if (!training || !visible) return;
    const place = () => {
      const rect = trigger.current?.getBoundingClientRect();
      if (!rect) return;
      if (rect.bottom < 0 || rect.top > window.innerHeight) { setOpen(false); return; }
      const anchorTop = Math.max(12, Math.min(rect.top, window.innerHeight - 12));
      const anchorBottom = Math.max(12, Math.min(rect.bottom, window.innerHeight - 12));
      const below = window.innerHeight - anchorBottom - 17;
      const above = anchorTop - 17;
      const upwards = below < 300 && above > below;
      const width = Math.min(Math.max(rect.width, 340), window.innerWidth - 24);
      setPosition({position:'fixed', width, left:Math.max(12, Math.min(rect.left, window.innerWidth - width - 12)),
        ...(upwards ? {bottom:window.innerHeight - anchorTop + 5} : {top:anchorBottom + 5}),
        maxHeight:Math.max(0, Math.min(520, upwards ? above : below))});
    };
    const outside = (event: MouseEvent) => {
      const target = event.target as Node;
      const ownedSelect = Array.from(menu.current?.querySelectorAll('[aria-controls]') || []).some(node => document.getElementById(node.getAttribute('aria-controls') || '')?.contains(target));
      if (!trigger.current?.contains(target) && !menu.current?.contains(target) && !ownedSelect) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') { setOpen(false); trigger.current?.focus({preventScroll:true}); } };
    menu.current?.querySelector<HTMLInputElement>('input:not(:disabled)')?.focus({preventScroll:true});
    place();
    window.addEventListener('resize', place); window.addEventListener('scroll', place, true);
    document.addEventListener('mousedown', outside); document.addEventListener('keydown', escape);
    return () => {
      window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true);
      document.removeEventListener('mousedown', outside); document.removeEventListener('keydown', escape);
    };
  }, [training, visible]);
  const popover = React.useRef<HTMLDetailsElement>(null);
  const devices = snapshot?.devices || [];
  const wrongCount = value.length > 0 && value.length !== count;
  const missing = snapshot !== null && value.some(id => !devices.some(device => device.device === id && device.status !== 'unavailable'));
  const valid = training ? gpuSelectionValid(value, count, snapshot) : !wrongCount && !missing;
  const invalidMessage = missing ? text('所选显卡已不可用，请重新选择。', 'A selected GPU is unavailable. Choose again.')
     : !Number.isInteger(count) || count < 1 || count > 64 ? text('显卡数量须为 1–64 的整数。', 'GPU count must be an integer from 1 to 64.')
    : value.length && !snapshot ? text('正在确认所选显卡是否可用。', 'Checking selected GPU availability.')
    : value.length && value.length < count ? text(`已选 ${value.length} 张，还需选择 ${count - value.length} 张；也可改为自动。`, `${value.length} selected; choose ${count - value.length} more, or use automatic selection.`)
    : value.length ? text(`当前训练需要选择 ${count} 张显卡，也可以改为自动选择。`, `Select ${count} GPUs for this run, or use automatic selection.`)
    : text(`当前可用设备不足 ${count} 张，请调整显卡数量。`, `Fewer than ${count} GPUs are available. Adjust the device count.`);
  React.useEffect(() => { onValidityChange?.(valid); }, [valid, onValidityChange]);
  React.useEffect(() => {
    if (!compact || training) return;
    const close = (event: MouseEvent | KeyboardEvent) => {
      if (!popover.current?.open) return;
      if (event instanceof KeyboardEvent) { if (event.key === 'Escape') popover.current.open = false; return; }
      const target = event.target as Node;
      const ownedMenu = Array.from(popover.current.querySelectorAll('[aria-controls]')).some(trigger => document.getElementById(trigger.getAttribute('aria-controls') || '')?.contains(target));
      if (!popover.current.contains(target) && !ownedMenu) popover.current.open = false;
    };
    document.addEventListener('mousedown', close); document.addEventListener('keydown', close);
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', close); };
  }, [compact, training]);
  const statusLabel = (device: QueueDevice) => device.job_id
    ? text(`占用中 · ${device.job_name || device.job_id}`, `Busy · ${device.job_name || device.job_id}`)
    : device.status === 'unavailable' ? text('不可用', 'Unavailable') : text('空闲', 'Free');
  const describe = (device: QueueDevice) => `${gpuDeviceLabel(device.device)} · ${device.name} · ${statusLabel(device)}`;
  const body = <div className="gpu-picker-body">
    <div className="gpu-picker-heading"><strong>{text('运行显卡', 'Run on GPU')}</strong><button type="button" disabled={disabled || loading} onClick={refresh} aria-label={text('刷新显卡状态', 'Refresh GPU status')}><RefreshCw size={13} className={loading ? 'animate-spin' : ''}/></button></div>
    {training ? <>
      <label className="gpu-picker-auto"><input type={count === 1 ? 'radio' : 'checkbox'} name={`${menuId}-device`} checked={value.length === 0} disabled={disabled}
        onChange={event => onChange(event.target.checked ? [] : devices.filter(device => device.status !== 'unavailable').slice(0, count).map(device => device.device))}/>{text(`自动选择 ${count} 张空闲显卡`, `Automatically choose ${count} free GPU${count === 1 ? '' : 's'}`)}</label>
      <p>{count === 1 ? text('单选；选择另一张显卡会替换当前选择。', 'Select one GPU; choosing another replaces the current selection.')
        : text(`最多选择 ${count} 张；显卡数量在训练参数中修改。`, `Select up to ${count} GPUs. Change the count in training parameters.`)}</p>
      <div className="gpu-picker-cards" role="group" aria-label={text('选择训练显卡', 'Choose training GPUs')}>{devices.map(device => <label key={device.device}>
        <input type={count === 1 ? 'radio' : 'checkbox'} name={`${menuId}-device`} disabled={disabled || !value.includes(device.device) && (device.status === 'unavailable' || count > 1 && value.length >= count)} checked={value.includes(device.device)}
          onChange={event => onChange(event.target.checked ? count === 1 ? [device.device] : [...value, device.device] : value.filter(selected => selected !== device.device))}/>
        <span><strong>{gpuDeviceLabel(device.device)} · {device.name}</strong><small>{statusLabel(device)}{device.mem_free_mb != null && ` · ${text('可用', 'Free')} ${(device.mem_free_mb / 1024).toFixed(1)} GiB`}</small></span>
      </label>)}{value.filter(selected => !devices.some(device => device.device === selected)).map(selected => <label key={selected}>
        <input type="checkbox" checked disabled={disabled} onChange={() => onChange(value.filter(id => id !== selected))}/><span>{gpuDeviceLabel(selected)} · {text('已不可用', 'Unavailable')}</span>
      </label>)}</div>
    </> : count === 1 ? <StudioSelect aria-label={text('运行显卡', 'Run on GPU')} value={value[0] || ''} disabled={disabled} onValueChange={id => onChange(id ? [id] : [])} options={[
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
    {!training && !valid && <p role="alert" className="gpu-picker-warning">{invalidMessage}</p>}
    {value.some(id => devices.find(device => device.device === id)?.job_id) && <p>{text('所选显卡正在使用，任务会等待它空闲后启动。', 'A selected GPU is busy. This job will wait until it is free.')}</p>}
    {snapshot?.max_concurrent === 1 && devices.length > 1 && <p>{text('队列目前限制为同时运行 1 个任务；需要并行时，在队列调度设置中改为自动。', 'The queue currently allows one job at a time. Use automatic concurrency in queue settings to run on separate GPUs in parallel.')}</p>}
  </div>;
  if (training) return <div className="gpu-picker gpu-picker-training">
    <button ref={trigger} type="button" id={id} className="gpu-picker-trigger" aria-label={label || text('训练显卡', 'Training GPUs')}
      aria-expanded={visible} aria-controls={visible ? menuId : undefined} aria-haspopup="dialog" aria-invalid={!valid} disabled={disabled} onClick={() => setOpen(!open)}>
      <Cpu size={14}/><span>{value.length ? value.map(selected => {
        const device = devices.find(device => device.device === selected);
        return `${gpuDeviceLabel(selected)}${device ? ` · ${device.name}` : ''}`;
      }).join('、') : text('自动', 'Auto')}</span><ChevronDown size={14}/>
    </button>
    {!valid && <p role="alert" className="gpu-picker-warning">{invalidMessage}</p>}
    {visible && createPortal(<div id={menuId} ref={menu} role="dialog" aria-label={text('训练显卡选择', 'Training GPU selection')} className="gpu-picker gpu-picker-menu" style={position}>{body}</div>, document.body)}
  </div>;
  if (!compact) return <section className="gpu-picker" aria-label={text('显卡选择', 'GPU selection')}>{body}</section>;
  return <details className="gpu-picker gpu-picker-compact" ref={popover}><summary aria-label={text('选择运行显卡', 'Choose run GPUs')}><Cpu size={14}/><span>{value.length ? value.map(gpuDeviceLabel).join(', ') : text('自动选卡', 'Auto GPU')}{wrongCount ? ' !' : ''}</span><ChevronDown size={12}/></summary>{body}</details>;
}
