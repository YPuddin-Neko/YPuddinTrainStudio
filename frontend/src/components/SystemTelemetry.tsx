import React from 'react';
import { Cpu, HardDrive, MemoryStick, createLucideIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { SystemStats } from '../api/types';
import { useWorkspaceText } from '../utils/workspaceText';

// Official Lucide GPU geometry, backported for the installed 0.359 package.
// https://github.com/lucide-icons/lucide/blob/main/icons/gpu.svg
// ISC, Copyright (c) 2026 Lucide Icons and Contributors.
// The full notice ships in public/licenses/lucide-gpu.txt.
const Gpu = createLucideIcon('Gpu', [
  ['path', { d: 'M2 17h18a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H2', key: 'board' }],
  ['path', { d: 'M2 21V3', key: 'bracket' }],
  ['path', { d: 'M7 17v3a1 1 0 0 0 1 1h5a1 1 0 0 0 1-1v-3', key: 'connector' }],
  ['circle', { cx: '16', cy: '11', r: '2', key: 'fan-right' }],
  ['circle', { cx: '8', cy: '11', r: '2', key: 'fan-left' }],
]);

const known = (value: number | null | undefined): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const percent = (value: number | null | undefined) => known(value) ? `${Math.round(value)}%` : '—';
const ratio = (used: number | null | undefined, total: number | null | undefined) => known(used) && known(total) && total > 0 ? used / total * 100 : null;
const reading = (value: number | null | undefined, unit: string) => known(value) ? `${Math.round(value)} ${unit}` : '—';
const capacity = (used: number | null | undefined, total: number | null | undefined, divisor = 1) => {
  const unit = known(total) && total / divisor >= 1024 ? 'TiB' : 'GiB';
  const scale = divisor * (unit === 'TiB' ? 1024 : 1);
  const format = (value: number | null | undefined) => known(value) ? (value / scale).toFixed(1) : '—';
  return `${format(used)} / ${format(total)} ${unit}`;
};

export default function SystemTelemetry({ stats }: { stats: SystemStats | null }) {
  const text = useWorkspaceText();
  const { t } = useTranslation();
  const [selectedGpu, setSelectedGpu] = React.useState(0);
  const gpus = stats?.gpus || [];
  const gpu = gpus.find(item => item.index === selectedGpu) || gpus[0];
  const unified = gpu?.kind === 'mps';
  const ram = stats?.ram;
  const disk = stats?.disks?.[0];
  const memoryLabel = unified ? text('统一内存', 'Unified') : text('显存', 'VRAM');
  const gpuDescription = [gpu?.name || text('未检测到显卡', 'No GPU detected'), gpu?.telemetry_note ? t(`hardware.${gpu.telemetry_note}`) : '', unified ? text('此处为全系统统一内存使用量，并非训练进程分配量。', 'This is system unified memory use, not a training process allocation.') : ''].filter(Boolean).join(' · ');
  const gpuMemory = capacity(gpu?.mem_used_mb, gpu?.mem_total_mb, 1024);

  return <div className="system-telemetry" role="group" aria-label={text('系统硬件状态，可横向滚动', 'System hardware status, horizontally scrollable')} tabIndex={0}>
    <div className="telemetry-strip">
      <div className="telemetry-group telemetry-cpu" role="group" aria-label="CPU" data-testid="telemetry-cpu">
        <Cpu size={16} aria-hidden="true" /><div className="telemetry-value"><span className="telemetry-label">CPU</span><strong aria-label={text('CPU 占用率', 'CPU utilization')}>{percent(stats?.cpu_pct)}</strong></div>
      </div>
      <div className="telemetry-group telemetry-memory" role="group" aria-label={text('内存', 'Memory')} title={text('系统内存使用量 / 总容量', 'System memory used / total')} data-testid="telemetry-memory">
        <MemoryStick size={16} aria-hidden="true" /><div className="telemetry-value"><span className="telemetry-heading"><span className="telemetry-label">{text('内存', 'Memory')}</span><strong aria-label={text('内存占用率', 'Memory utilization')}>{percent(ratio(ram?.used_mb, ram?.total_mb))}</strong></span><span className="telemetry-capacity">{capacity(ram?.used_mb, ram?.total_mb, 1024)}</span></div>
      </div>
      <div className="telemetry-group telemetry-gpu" role="group" aria-label="GPU" title={gpuDescription} data-testid="telemetry-gpu">
        <Gpu size={18} aria-hidden="true" />
        <div className="telemetry-device">{gpus.length > 1 ? <select aria-label={text('选择监控显卡', 'Choose monitored GPU')} value={gpu?.index} onChange={event => setSelectedGpu(Number(event.target.value))}>{gpus.map(item => <option key={item.index} value={item.index}>GPU {item.index} · {item.name}</option>)}</select> : <span className="telemetry-label">GPU{gpu ? '' : ' —'}</span>}<span className="telemetry-device-kind">{unified ? 'MPS' : gpu?.kind === 'cuda' ? 'CUDA' : ''}</span></div>
        <div className="telemetry-gpu-readings">
          <div><span className="telemetry-label">{text('占用', 'Load')}</span><strong aria-label={text('GPU 占用率', 'GPU utilization')} data-testid="topbar-gpu-util">{percent(gpu?.util_pct)}</strong></div>
          <div title={`${memoryLabel} · ${gpuMemory}`}><span className="telemetry-label">{memoryLabel}</span><strong aria-label={unified ? text('系统统一内存占用率', 'System unified memory utilization') : text('显存占用率', 'VRAM utilization')} data-testid="topbar-gpu-memory">{percent(ratio(gpu?.mem_used_mb, gpu?.mem_total_mb))}</strong></div>
          <div><span className="telemetry-label">{text('功率', 'Power')}</span><strong aria-label={text('GPU 功率', 'GPU power')} data-testid="topbar-gpu-power">{reading(gpu?.power_w, 'W')}</strong></div>
          <div><span className="telemetry-label">{text('温度', 'Temp')}</span><strong aria-label={text('GPU 温度', 'GPU temperature')} data-testid="topbar-gpu-temperature">{reading(gpu?.temp_c, '°C')}</strong></div>
        </div>
      </div>
      <div className="telemetry-group telemetry-disk" role="group" aria-label={text('硬盘', 'Disk')} title={disk ? `${text('项目数据所在磁盘', 'Project data disk')} · ${disk.path}` : text('磁盘信息不可用', 'Disk information unavailable')} data-testid="telemetry-disk">
        <HardDrive size={16} aria-hidden="true" /><div className="telemetry-value"><span className="telemetry-heading"><span className="telemetry-label">{text('硬盘', 'Disk')}</span><strong aria-label={text('硬盘占用率', 'Disk utilization')}>{percent(ratio(disk?.used_gb, disk?.total_gb))}</strong></span><span className="telemetry-capacity">{capacity(disk?.used_gb, disk?.total_gb)}</span></div>
      </div>
    </div>
  </div>;
}
