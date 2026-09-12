import { Cpu, Zap } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import type { GpuStats } from '../api/types';
import { formatBytesMB } from '../utils/format';

const known = (value: number | null | undefined): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function GpuCard({ gpu }: { gpu: GpuStats }) {
  const { t } = useTranslation();
  const apple = gpu.kind === 'mps';
  const missingReadings = [!known(gpu.util_pct) && t('hardware.gpuUtilizationMissing'), !known(gpu.power_w) && t('hardware.gpuPowerMissing'), !known(gpu.temp_c) && t('hardware.gpuTemperatureMissing')].filter(Boolean).join(' ');
  const showNote = gpu.telemetry_note && !(apple && gpu.telemetry_note === 'mps_power_unavailable');
  const usedPercent = gpu.mem_used_mb != null && gpu.mem_total_mb ? Math.min(100, gpu.mem_used_mb / gpu.mem_total_mb * 100) : 0;
  return (
    <section className="p-5 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800" aria-label={gpu.name}>
      <div className="flex items-center justify-between gap-4">
        <div className="flex min-w-0 items-center gap-2 font-semibold text-sm">
          <Cpu className="w-4 h-4 text-blue-500 shrink-0" /><span className="truncate">{gpu.name}</span>
        </div>
        <span className="text-[11px] text-slate-500 font-mono shrink-0">{apple ? 'MPS' : `GPU ${gpu.index}`} · {gpu.telemetry_source || '—'}</span>
      </div>
      <div className="grid grid-cols-3 gap-3 mt-5">
        <div><div className="text-xs text-slate-500 flex items-center gap-1"><Zap className="w-3.5 h-3.5" />{t('hardware.power')}</div>
          <div className="mt-1 text-2xl font-semibold tabular-nums" data-testid="gpu-power">{known(gpu.power_w) ? <>{Math.round(gpu.power_w)}<span className="ml-1 text-sm font-normal text-slate-500">W</span></> : <span className="text-sm text-slate-500">{t('hardware.unavailable')}</span>}</div>
          {known(gpu.power_limit_w) && <div className="text-[11px] text-slate-500">{t('hardware.powerLimit', { value: Math.round(gpu.power_limit_w) })}</div>}
        </div>
        <div><div className="text-xs text-slate-500">{t(apple ? 'hardware.systemGpuUtilization' : 'hardware.utilization')}</div><div className="mt-1 text-2xl font-semibold tabular-nums">{known(gpu.util_pct) ? `${Math.round(gpu.util_pct)}%` : '—'}</div></div>
        <div><div className="text-xs text-slate-500">{t('hardware.temperature')}</div><div className="mt-1 text-2xl font-semibold tabular-nums">{known(gpu.temp_c) ? `${Math.round(gpu.temp_c)}°` : '—'}</div></div>
      </div>
      <div className="mt-4 flex justify-between gap-2 text-xs text-slate-500" title={apple ? t('hardware.unifiedMemoryScope') : undefined}><span>{t(apple ? 'dashboard.unifiedMemory' : 'dashboard.vram')}</span><span className="font-mono">{formatBytesMB(gpu.mem_used_mb)} / {formatBytesMB(gpu.mem_total_mb)}</span></div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800"><div className="h-full rounded-full bg-blue-500 transition-[width]" style={{ width: `${usedPercent}%` }} /></div>
      {apple && <p className="mt-3 text-xs leading-relaxed text-slate-500">{t('hardware.systemGpuScope')}{gpu.telemetry_source === 'ioreg' && <> {t('hardware.appleGpuDriverSource')}</>}</p>}
      {apple && missingReadings && <p className="mt-1 text-xs leading-relaxed text-slate-500">{missingReadings}</p>}
      {showNote && <p className="mt-3 text-xs leading-relaxed text-slate-500">{t(`hardware.${gpu.telemetry_note}`)}{!apple && <Link to="/settings#environment" className="ml-1 text-blue-600 hover:underline">{t('hardware.checkEnvironment')}</Link>}</p>}
    </section>
  );
}
