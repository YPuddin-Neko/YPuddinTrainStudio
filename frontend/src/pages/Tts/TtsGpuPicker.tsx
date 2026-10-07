import React from 'react';
import { useQueueDevices } from '../../api/hooks/useQueueDevices';
import GpuDevicePicker from '../../components/GpuDevicePicker';
import { useWorkspaceText } from '../../utils/workspaceText';
import './tts.css';

export default function TtsGpuPicker({ value, onChange, disabled, onValidityChange, training = false }: {
  value: string[]; onChange: (value: string[]) => void; disabled?: boolean; onValidityChange: (valid: boolean) => void; training?: boolean;
}) {
  const text = useWorkspaceText();
  const state = useQueueDevices();
  const devices = state.snapshot?.devices.filter(device => device.device.startsWith('cuda:')) || [];
  const unavailable = !!state.snapshot && !devices.some(device => device.status !== 'unavailable');
  React.useEffect(() => { if (unavailable) onValidityChange(false); }, [unavailable, onValidityChange]);
  if (unavailable) return <div className="tts-device-unavailable"><span>{text('未检测到可用的 CUDA 显卡', 'No available CUDA GPU detected')}</span><button type="button" className="ui-link" disabled={state.loading || disabled} onClick={state.refresh}>{text('刷新', 'Refresh')}</button></div>;
  return <GpuDevicePicker value={value} onChange={onChange} count={1} training={training} compact={!training} disabled={disabled} onValidityChange={onValidityChange}
    deviceState={{ ...state, snapshot: state.snapshot ? { ...state.snapshot, devices } : null }}/>
}
