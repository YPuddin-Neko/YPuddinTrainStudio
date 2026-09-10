import React from 'react';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { SystemStats, Job, JobListResponse, Artifact } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatBytesGB, formatBytesMB, formatEta } from '../../utils/format';
import { Activity, Layers, Box, Cpu, HardDrive, Zap, AlertTriangle } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const { t } = useTranslation();
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [artifacts, setArtifacts] = React.useState<Artifact[]>([]);

  const fetchJobs = React.useCallback(() => {
    apiClient.get<JobListResponse | Job[]>('/jobs').then((res) => {
      if (Array.isArray(res)) setJobs(res);
      else if (res && Array.isArray(res.items)) setJobs(res.items);
    }).catch(console.error);
  }, []);

  React.useEffect(() => {
    apiClient.get<SystemStats>('/system/stats').then(setStats).catch(console.error);
    fetchJobs();
    apiClient.get<Artifact[]>('/artifacts').then((res) => setArtifacts(Array.isArray(res) ? res : [])).catch(console.error);
  }, [fetchJobs]);

  useEventStream<SystemStats>(EVENT_TYPES.SYSTEM_STATS, (newStats) => {
    setStats(newStats);
  });

  useEventStream(EVENT_TYPES.JOB_STATE, () => {
    fetchJobs();
  });

  const runningJob = jobs.find((j) => j.status === 'running');
  const queuedCount = jobs.filter((j) => j.status === 'queued' || j.status === 'scheduled').length;
  const completedJobs = jobs.filter((j) => j.status === 'completed');
  const gpus = stats?.gpus || [];

  return (
    <div className="space-y-6" data-testid="dashboard-page">
      {/* 1. 顶部系统状态条 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {gpus.map((gpu) => (
          <div key={gpu.index} className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            <div className="flex justify-between items-center text-xs text-slate-400">
              <span className="flex items-center space-x-1">
                <Zap className="w-3.5 h-3.5 text-amber-500" />
                <span>{t('dashboard.gpu')} {gpu.index}{gpu.name ? ` · ${gpu.name}` : ''}</span>
              </span>
              <span className="font-mono">{gpu.temp_c != null ? `${gpu.temp_c}°C` : ''}</span>
            </div>
            <div className="text-2xl font-bold font-mono mt-1">{gpu.util_pct ?? 0}%</div>
            <div className="text-xs text-slate-400 mt-1 font-mono">
              {t('dashboard.vram')}: {formatBytesMB(gpu.mem_used_mb)} / {formatBytesMB(gpu.mem_total_mb)}
            </div>
          </div>
        ))}

        {gpus.length === 0 && (
          <div className="p-4 bg-amber-50 dark:bg-amber-950/30 rounded-xl border border-amber-200 dark:border-amber-800" data-testid="no-gpu-card">
            <div className="flex items-center space-x-1 text-xs text-amber-600 dark:text-amber-400">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>{t('dashboard.noGpuTitle')}</span>
            </div>
            <p className="text-xs text-amber-600/80 dark:text-amber-400/80 mt-1.5">{t('dashboard.noGpuDesc')}</p>
          </div>
        )}

        <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="flex justify-between items-center text-xs text-slate-400">
            <span className="flex items-center space-x-1">
              <Cpu className="w-3.5 h-3.5 text-blue-500" />
              <span>{t('dashboard.cpu')}</span>
            </span>
          </div>
          <div className="text-2xl font-bold font-mono mt-1">{stats?.cpu_pct != null ? `${Math.round(stats.cpu_pct)}%` : '--'}</div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="flex justify-between items-center text-xs text-slate-400">
            <span className="flex items-center space-x-1">
              <Activity className="w-3.5 h-3.5 text-green-500" />
              <span>{t('dashboard.ram')}</span>
            </span>
          </div>
          <div className="text-2xl font-bold font-mono mt-1">
            {stats?.ram ? formatBytesMB(stats.ram.used_mb) : '--'}
          </div>
          <div className="text-xs text-slate-400 mt-1 font-mono">
            / {stats?.ram ? formatBytesMB(stats.ram.total_mb) : '--'}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="flex justify-between items-center text-xs text-slate-400">
            <span className="flex items-center space-x-1">
              <HardDrive className="w-3.5 h-3.5 text-purple-500" />
              <span>{t('dashboard.disk')}</span>
            </span>
          </div>
          <div className="text-2xl font-bold font-mono mt-1">
            {stats?.disks?.[0] ? formatBytesGB(stats.disks[0].used_gb) : '--'}
          </div>
          <div className="text-xs text-slate-400 mt-1 font-mono">
            / {stats?.disks?.[0] ? formatBytesGB(stats.disks[0].total_gb) : '--'}
          </div>
        </div>
      </div>

      {/* 2. 正在运行的任务卡片 */}
      {runningJob ? (
        <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border border-blue-200 dark:border-blue-900 shadow-sm space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <span className="text-xs font-semibold px-2 py-0.5 bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 rounded uppercase">
                {t('dashboard.activeJob')}
              </span>
              <h3 className="text-xl font-bold mt-2">
                <Link to={`/jobs/${runningJob.id}`} className="hover:underline">
                  {runningJob.name}
                </Link>
              </h3>
            </div>
            <Link
              to={`/jobs/${runningJob.id}`}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium"
            >
              {t('dashboard.monitorJob')}
            </Link>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
            <div>
              <div className="text-xs text-slate-400">{t('dashboard.phaseStep')}</div>
              <div className="font-semibold text-sm capitalize font-mono">
                {runningJob.progress.phase || '--'} ({runningJob.progress.step ?? 0}/{runningJob.progress.total_steps ?? 0})
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400">{t('dashboard.lossEma')}</div>
              <div className="font-semibold text-sm font-mono">
                {runningJob.latest.loss?.toFixed(4) ?? '--'} / {runningJob.latest.loss_ema?.toFixed(4) ?? '--'}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400">{t('dashboard.speed')}</div>
              <div className="font-semibold text-sm font-mono">
                {runningJob.progress.it_s != null ? `${Number(runningJob.progress.it_s).toFixed(2)} it/s` : '--'}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400">{t('dashboard.eta')}</div>
              <div className="font-semibold text-sm font-mono">
                {formatEta(runningJob.progress.eta_s)}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 flex items-center justify-center min-h-[120px] text-slate-500">
          {t('dashboard.noActiveJob')}
        </div>
      )}

      {/* 3. 队列摘要与最近产物 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-bold flex items-center space-x-2">
              <Layers className="w-5 h-5 text-blue-500" />
              <span>{t('dashboard.queueSummary')}</span>
            </h3>
            <Link to="/queue" className="text-xs text-blue-500 hover:underline">{t('dashboard.viewAll')}</Link>
          </div>
          <div className="flex space-x-4">
            <div className="flex-1 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
              <div className="text-xs text-slate-400">{t('dashboard.queuedScheduled')}</div>
              <div className="text-xl font-bold font-mono mt-1">{queuedCount}</div>
            </div>
            <div className="flex-1 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
              <div className="text-xs text-slate-400">{t('dashboard.completedRecent')}</div>
              <div className="text-xl font-bold font-mono mt-1">{completedJobs.length}</div>
            </div>
          </div>
        </div>

        <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-bold flex items-center space-x-2">
              <Box className="w-5 h-5 text-indigo-500" />
              <span>{t('dashboard.recentArtifacts')}</span>
            </h3>
            <Link to="/artifacts" className="text-xs text-blue-500 hover:underline">{t('dashboard.viewAll')}</Link>
          </div>
          <div className="divide-y divide-slate-100 dark:divide-slate-700 text-xs">
            {artifacts.length === 0 && <p className="text-slate-400 py-2">{t('common.empty')}</p>}
            {artifacts.slice(0, 3).map((a) => (
              <div key={a.id} className="py-2 flex justify-between items-center">
                <span className="font-mono truncate mr-3">{a.name}</span>
                <span className="text-slate-400 whitespace-nowrap">{a.algo} (rank {a.rank})</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
