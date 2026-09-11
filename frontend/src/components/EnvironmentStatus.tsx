import React from 'react';
import { Cpu, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../api/client';
import type { SystemInfo, SystemStats } from '../api/types';

export function EnvironmentStatus() {
  const { t, i18n } = useTranslation();
  const text = (zh: string, en: string) => i18n.resolvedLanguage?.startsWith('en') ? en : zh;
  const [info, setInfo] = React.useState<SystemInfo | null>(null);
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const refresh = React.useCallback(async () => {
    setLoading(true); setError('');
    try {
      const [info, stats] = await Promise.all([apiClient.get<SystemInfo>('/system/info', {silent: true}), apiClient.get<SystemStats>('/system/stats', {silent: true})]);
      setInfo(info); setStats(stats);
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { setLoading(false); }
  }, []);
  React.useEffect(() => { void refresh(); }, [refresh]);
  return <section id="environment" className="p-6 space-y-4 rounded-xl bg-white border border-slate-200 dark:bg-slate-800 dark:border-slate-700">
    <div className="flex items-center justify-between"><h3 className="flex gap-2 items-center font-semibold"><Cpu className="w-4 h-4 text-slate-400" />{text('训练环境与显卡采集', 'Training runtime and GPU telemetry')}</h3><button type="button" disabled={loading} onClick={() => void refresh()} className="inline-flex gap-1.5 items-center text-sm text-blue-600 disabled:opacity-50"><RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />{t('common.refresh', '刷新')}</button></div>
    {error && <p role="alert" className="text-red-600 text-sm">{error}</p>}
    {info && <dl className="grid grid-cols-2 gap-3 text-sm">{[
      ['Studio', info.ypuddin || '—'], ['PyTorch', info.packages?.torch || '—'],
      ['CUDA', info.cuda ? `${info.cuda} · ${info.cuda_available ? text('可用', 'available') : text('当前不可用', 'unavailable')}` : text('当前 PyTorch 不包含 CUDA', 'This PyTorch build has no CUDA')],
      ['NVML', info.packages?.['nvidia-ml-py'] || (stats?.gpus.some(gpu => gpu.kind === 'mps') ? text('不适用于 Apple GPU', 'Not applicable to Apple GPU') : text('未安装，可用时回退至 nvidia-smi', 'Not installed; falls back to nvidia-smi when available'))],
    ].map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-xs text-slate-500">{label}</dt><dd className="mt-1 break-words font-mono">{value}</dd></div>)}</dl>}
    {stats?.gpus.map(gpu => <div key={gpu.index} className="p-3 rounded-lg bg-slate-50 dark:bg-slate-900 text-xs space-y-1.5"><p className="font-medium">{gpu.name} · {gpu.telemetry_source || '—'}</p><p>{t('hardware.power')}：{gpu.power_w != null ? `${gpu.power_w.toFixed(1)} W` : t('hardware.unavailable')}</p>{gpu.telemetry_note && <p className="leading-relaxed text-slate-500">{t(`hardware.${gpu.telemetry_note}`)}</p>}</div>)}
    {stats && !stats.gpus.length && <p className="text-sm text-slate-500">{t('dashboard.noGpuDesc')}</p>}
  </section>;
}
