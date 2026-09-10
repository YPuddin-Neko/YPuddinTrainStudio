import React from 'react';
import { apiClient } from '../../api/client';
import { SystemStats, Job, JobListResponse, Artifact } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { Activity, Layers, Box, Cpu, HardDrive, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [artifacts, setArtifacts] = React.useState<Artifact[]>([]);

  React.useEffect(() => {
    apiClient.get<SystemStats>('/system/stats').then(setStats).catch(console.error);
    apiClient.get<JobListResponse | Job[]>('/jobs').then((res) => {
      if (Array.isArray(res)) {
        setJobs(res);
      } else if (res && Array.isArray(res.items)) {
        setJobs(res.items);
      }
    }).catch(console.error);
    apiClient.get<Artifact[]>('/artifacts').then((res) => setArtifacts(Array.isArray(res) ? res : [])).catch(console.error);
  }, []);

  useEventStream<SystemStats>(EVENT_TYPES.SYSTEM_STATS, (newStats) => {
    setStats(newStats);
  });

  useEventStream(EVENT_TYPES.JOB_STATE, () => {
    apiClient.get<JobListResponse | Job[]>('/jobs').then((res) => {
      if (Array.isArray(res)) {
        setJobs(res);
      } else if (res && Array.isArray(res.items)) {
        setJobs(res.items);
      }
    }).catch(console.error);
  });

  const runningJob = jobs.find((j) => j.status === 'running');
  const queuedCount = jobs.filter((j) => j.status === 'queued' || j.status === 'scheduled').length;
  const completedJobs = jobs.filter((j) => j.status === 'completed');

  return (
    <div className="space-y-6" data-testid="dashboard-page">
      {/* 1. 顶部系统状态条 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {stats?.gpus?.map((gpu) => (
          <div key={gpu.index} className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            <div className="flex justify-between items-center text-xs text-slate-400">
              <span className="flex items-center space-x-1">
                <Zap className="w-3.5 h-3.5 text-amber-500" />
                <span>GPU {gpu.index} ({gpu.name || 'NVIDIA'})</span>
              </span>
              <span>{gpu.temp_c}°C</span>
            </div>
            <div className="text-xl font-bold mt-1">{gpu.util_pct}%</div>
            <div className="text-xs text-slate-400 mt-1">
              VRAM: {Math.round(gpu.mem_used_mb / 1024)} / {Math.round(gpu.mem_total_mb / 1024)} GB
            </div>
          </div>
        ))}

        <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="flex justify-between items-center text-xs text-slate-400">
            <span className="flex items-center space-x-1">
              <Cpu className="w-3.5 h-3.5 text-blue-500" />
              <span>CPU Usage</span>
            </span>
          </div>
          <div className="text-xl font-bold mt-1">{stats?.cpu_pct ?? 0}%</div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="flex justify-between items-center text-xs text-slate-400">
            <span className="flex items-center space-x-1">
              <Activity className="w-3.5 h-3.5 text-green-500" />
              <span>RAM</span>
            </span>
          </div>
          <div className="text-xl font-bold mt-1">
            {stats ? `${Math.round(stats.ram.used_mb / 1024)} GB` : '--'}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Total: {stats ? `${Math.round(stats.ram.total_mb / 1024)} GB` : '--'}
          </div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="flex justify-between items-center text-xs text-slate-400">
            <span className="flex items-center space-x-1">
              <HardDrive className="w-3.5 h-3.5 text-purple-500" />
              <span>Disk</span>
            </span>
          </div>
          <div className="text-xl font-bold mt-1">
            {stats?.disks?.[0] ? `${stats.disks[0].used_gb} / ${stats.disks[0].total_gb} GB` : '--'}
          </div>
        </div>
      </div>

      {/* 2. 正在运行的任务卡片 */}
      {runningJob ? (
        <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border border-blue-200 dark:border-blue-900 shadow-sm space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <span className="text-xs font-semibold px-2 py-0.5 bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 rounded uppercase">
                Active Training Job
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
              Monitor Job
            </Link>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
            <div>
              <div className="text-xs text-slate-400">Phase / Step</div>
              <div className="font-semibold text-sm capitalize">
                {runningJob.progress.phase} ({runningJob.progress.step}/{runningJob.progress.total_steps})
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Loss / EMA</div>
              <div className="font-semibold text-sm">
                {runningJob.latest.loss.toFixed(4)} / {runningJob.latest.loss_ema.toFixed(4)}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Speed</div>
              <div className="font-semibold text-sm">{runningJob.progress.it_s} it/s</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">ETA</div>
              <div className="font-semibold text-sm">
                {runningJob.progress.eta_s ? `${Math.round(runningJob.progress.eta_s / 60)}m` : '--'}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 flex items-center justify-center min-h-[140px] text-slate-500">
          No training jobs running currently.
        </div>
      )}

      {/* 3. 队列摘要与最近产物 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 队列摘要 */}
        <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-bold flex items-center space-x-2">
              <Layers className="w-5 h-5 text-blue-500" />
              <span>Queue Summary</span>
            </h3>
            <Link to="/queue" className="text-xs text-blue-500 hover:underline">View All</Link>
          </div>
          <div className="flex space-x-4">
            <div className="flex-1 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
              <div className="text-xs text-slate-400">Queued / Scheduled</div>
              <div className="text-xl font-bold mt-1">{queuedCount}</div>
            </div>
            <div className="flex-1 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
              <div className="text-xs text-slate-400">Completed (Recent)</div>
              <div className="text-xl font-bold mt-1">{completedJobs.length}</div>
            </div>
          </div>
        </div>

        {/* 最近产物 */}
        <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-bold flex items-center space-x-2">
              <Box className="w-5 h-5 text-indigo-500" />
              <span>Recent Artifacts</span>
            </h3>
            <Link to="/artifacts" className="text-xs text-blue-500 hover:underline">View All</Link>
          </div>
          <div className="divide-y divide-slate-100 dark:divide-slate-700 text-xs">
            {artifacts.slice(0, 3).map((a) => (
              <div key={a.id} className="py-2 flex justify-between items-center">
                <span className="font-mono">{a.name}</span>
                <span className="text-slate-400">{a.algo} (rank {a.rank})</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
