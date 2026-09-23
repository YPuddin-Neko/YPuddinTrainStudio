import React from 'react';
import StudioSelect from './StudioSelect';
import { Cpu, HardDrive, MemoryStick, createLucideIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { GpuStats, SystemStats } from '../api/types';
import { formatGpuPower, formatGpuTemperature, gpuPowerDescription, gpuTemperatureDescription } from '../utils/gpuTelemetry';
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
const ratio = (used: number | null | undefined, total: number | null | undefined) => known(used) && known(total) && total > 0 ? used / total * 100 : null;
const average = (values: Array<number | null | undefined>) => values.length && values.every(known) ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const total = (values: Array<number | null | undefined>) => values.length && values.every(known) ? values.reduce((sum, value) => sum + value, 0) : null;
const capacity = (used: number | null | undefined, total: number | null | undefined, divisor = 1) => {
  const unit = known(total) && total / divisor >= 1024 ? 'TiB' : 'GiB';
  const scale = divisor * (unit === 'TiB' ? 1024 : 1);
  const format = (value: number | null | undefined) => known(value) ? (value / scale).toFixed(1) : '—';
  return `${format(used)} / ${format(total)} ${unit}`;
};

function Reading({ value, unit = '%', label, testId, display }: { value: number | null | undefined; unit?: string; label: string; testId?: string; display?: string }) {
  const available = known(value);
  return <strong className="telemetry-reading" aria-label={label} data-testid={testId}>
    <span className="telemetry-number">{available ? display ?? Math.round(value) : '—'}</span>
    {available && unit !== '%' ? ' ' : ''}<span className="telemetry-unit">{available ? unit : ''}</span>
  </strong>;
}

export default function SystemTelemetry({ stats }: { stats: SystemStats | null }) {
  const text = useWorkspaceText();
  const { t } = useTranslation();
  const [selectedGpu, setSelectedGpu] = React.useState('all');
  const gpus = stats?.gpus || [];
  const selected = gpus.find(item => String(item.index) === selectedGpu);
  const aggregate = gpus.length > 1 && !selected;
  const gpu: GpuStats | undefined = aggregate ? {
    index: -1, kind: gpus[0].kind, name: text('多卡平均', 'GPU average'),
    util_pct: average(gpus.map(item => item.util_pct)),
    power_w: average(gpus.map(item => item.power_w)),
    temp_c: average(gpus.map(item => item.temp_c)),
    mem_used_mb: total(gpus.map(item => item.mem_used_mb)),
    mem_total_mb: total(gpus.map(item => item.mem_total_mb)),
  } : selected || gpus[0];
  const unified = gpu?.kind === 'mps';
  const deviceKind: Record<string, string> = { mps: 'MPS', cuda: 'CUDA', dtk: 'DTK', rocm: 'ROCm' };
  const ram = stats?.ram;
  const disk = stats?.disks?.[0];
  const memoryLabel = unified ? text('统一内存', 'Unified') : text('显存', 'VRAM');
  const gpuNote = gpu?.telemetry_note === 'hip_driver_metrics_unavailable'
    ? text('暂时无法可靠获取这张卡的占用率、功率和温度。', 'Load, power and temperature readings are not yet reliably mapped to this device.')
    : gpu?.telemetry_note && !(unified && gpu.telemetry_note === 'mps_power_unavailable') ? t(`hardware.${gpu.telemetry_note}`) : '';
  const averageDescription = (values: Array<number | null | undefined>, label: string, description: string) => {
    const count = values.filter(known).length;
    return count === gpus.length
      ? description
      : text(`仅收到 ${count}/${gpus.length} 张显卡的${label}，暂不显示平均值。`, `${label} is available for only ${count}/${gpus.length} GPUs; no average is shown.`);
  };
  const utilizationDescription = aggregate ? averageDescription(gpus.map(item => item.util_pct), text('占用率', 'utilization'), text('每张显卡占用率的平均值。', 'Average GPU utilization.')) : [unified ? t('hardware.systemGpuScope') : '', unified && gpu?.telemetry_source === 'ioreg' ? t('hardware.appleGpuDriverSource') : '', !known(gpu?.util_pct) ? t('hardware.gpuUtilizationMissing') : ''].filter(Boolean).join(' ');
  const powerDescription = aggregate ? averageDescription(gpus.map(item => item.power_w), text('功率', 'power'), text('平均每张显卡的功率。', 'Average power per GPU.')) : gpuPowerDescription(gpu, t);
  const temperatureDescription = aggregate ? averageDescription(gpus.map(item => item.temp_c), text('温度', 'temperature'), text('每张显卡当前温度的平均值。', 'Average current temperature of the GPUs.')) : gpuTemperatureDescription(gpu, t);
  const gpuDescription = aggregate ? undefined : [gpu?.name || text('未检测到显卡', 'No GPU detected'), gpuNote, utilizationDescription, powerDescription, temperatureDescription, unified ? t('hardware.unifiedMemoryScope') : ''].filter(Boolean).join(' · ');
  const gpuMemory = capacity(gpu?.mem_used_mb, gpu?.mem_total_mb, 1024);
  const memoryRatios = gpus.map(item => ratio(item.mem_used_mb, item.mem_total_mb));
  const memoryUsage = aggregate ? average(memoryRatios) : ratio(gpu?.mem_used_mb, gpu?.mem_total_mb);
  const memoryDescription = aggregate ? `${averageDescription(memoryRatios, text('显存占用率', 'VRAM utilization'), text('先计算每张显卡已用显存的百分比，再取平均值。', 'Average of each GPU’s used-memory percentage.'))}\n${text('各卡已用显存 / 总容量合计', 'Combined memory used / total capacity')}：${gpuMemory}` : `${memoryLabel} · ${gpuMemory}`;
  const deviceDescription = `${deviceKind[gpu?.kind ?? ''] ?? ''}${aggregate ? ` · ${text(`${gpus.length} 卡`, `${gpus.length} GPUs`)}` : ''}`;
  const deviceHelp = aggregate ? text(`当前显示 ${gpus.length} 张显卡的平均状态。\n点击可切换到单张显卡。`, `Showing averages across ${gpus.length} GPUs.\nClick to view an individual GPU.`) : undefined;

  return <div className="system-telemetry" role="group" aria-label={text('系统硬件状态，可横向滚动', 'System hardware status, horizontally scrollable')} tabIndex={0}>
    <div className="telemetry-strip">
      <div className="telemetry-group telemetry-cpu" role="group" aria-label="CPU" data-testid="telemetry-cpu">
        <Cpu size={16} aria-hidden="true" /><div className="telemetry-value"><span className="telemetry-label">CPU</span><Reading value={stats?.cpu_pct} label={text('CPU 占用率', 'CPU utilization')}/></div>
      </div>
      <div className="telemetry-group telemetry-memory" role="group" aria-label={text('内存', 'Memory')} title={`${text('系统内存使用量 / 总容量', 'System memory used / total')} · ${capacity(ram?.used_mb, ram?.total_mb, 1024)}`} data-testid="telemetry-memory">
        <MemoryStick size={16} aria-hidden="true" /><div className="telemetry-value"><span className="telemetry-label">{text('内存', 'Memory')}</span><Reading value={ratio(ram?.used_mb, ram?.total_mb)} label={text('内存占用率', 'Memory utilization')}/></div>
      </div>
      <div className="telemetry-group telemetry-gpu" role="group" aria-label="GPU" title={gpuDescription} data-testid="telemetry-gpu">
        <Gpu size={18} aria-hidden="true" />
        <div className="telemetry-device" title={deviceHelp}>{gpus.length > 1 ? <StudioSelect className="telemetry-gpu-select" aria-label={text('选择监控显卡', 'Choose monitored GPU')} value={aggregate ? 'all' : String(gpu?.index)} onValueChange={setSelectedGpu} triggerDescription={deviceDescription} options={[{value:'all', label:text(`多卡平均 · ${gpus.length} 张显卡`, `GPU average · ${gpus.length} GPUs`), displayLabel:text('多卡平均', 'GPU average')}, ...gpus.map(item => ({value:String(item.index), label:`GPU ${item.index} · ${item.name}`, displayLabel:`GPU ${item.index}`}))]}/> : <><span className="telemetry-label">{gpu ? `GPU ${gpu.index}` : 'GPU —'}</span><span className="telemetry-device-kind">{deviceDescription}</span></>}</div>
        <div className="telemetry-gpu-readings">
          <div title={utilizationDescription || undefined}><span className="telemetry-label">{text('占用', 'Load')}</span><Reading value={gpu?.util_pct} label={unified ? t('hardware.systemGpuUtilization') : text('GPU 占用率', 'GPU utilization')} testId="topbar-gpu-util"/></div>
          <div title={memoryDescription}><span className="telemetry-label">{memoryLabel}</span><Reading value={memoryUsage} label={unified ? text('系统统一内存占用率', 'System unified memory utilization') : text('显存占用率', 'VRAM utilization')} testId="topbar-gpu-memory"/></div>
          <div title={powerDescription}><span className="telemetry-label">{unified ? t('hardware.estimatedPowerShort') : text('功率', 'Power')}</span><Reading value={gpu?.power_w} display={formatGpuPower(gpu)} unit="W" label={unified ? t('hardware.estimatedGpuPower') : text('GPU 功率', 'GPU power')} testId="topbar-gpu-power"/></div>
          <div title={temperatureDescription}><span className="telemetry-label">{unified ? t('hardware.meanTemperatureShort') : text('温度', 'Temp')}</span><Reading value={gpu?.temp_c} display={formatGpuTemperature(gpu)} unit="°C" label={unified ? t('hardware.meanGpuTemperature') : text('GPU 温度', 'GPU temperature')} testId="topbar-gpu-temperature"/></div>
        </div>
      </div>
      <div className="telemetry-group telemetry-disk" role="group" aria-label={text('硬盘', 'Disk')} title={disk ? `${text('项目数据所在磁盘', 'Project data disk')} · ${disk.path} · ${capacity(disk.used_gb, disk.total_gb)}` : text('磁盘信息不可用', 'Disk information unavailable')} data-testid="telemetry-disk">
        <HardDrive size={16} aria-hidden="true" /><div className="telemetry-value"><span className="telemetry-label">{text('硬盘', 'Disk')}</span><Reading value={ratio(disk?.used_gb, disk?.total_gb)} label={text('硬盘占用率', 'Disk utilization')}/></div>
      </div>
    </div>
  </div>;
}
