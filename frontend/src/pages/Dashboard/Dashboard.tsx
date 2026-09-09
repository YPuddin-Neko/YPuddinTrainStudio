import React from 'react';
import { apiClient } from '../../api/client';
import { SystemStats } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { Activity } from 'lucide-react';

export default function Dashboard() {
  const [stats, setStats] = React.useState<SystemStats | null>(null);

  React.useEffect(() => {
    apiClient.get<SystemStats>('/system/stats')
      .then(setStats)
      .catch(console.error);
  }, []);

  useEventStream<SystemStats>(EVENT_TYPES.SYSTEM_STATS, (newStats) => {
    setStats(newStats);
  });

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Dashboard</h2>
      
      {/* Stats Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="text-sm text-slate-500">CPU Usage</div>
          <div className="text-2xl font-semibold mt-1">{stats?.cpu_pct ?? 0}%</div>
        </div>

        <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="text-sm text-slate-500">RAM Usage</div>
          <div className="text-2xl font-semibold mt-1">
            {stats ? `${Math.round(stats.ram.used_mb / 1024)} / ${Math.round(stats.ram.total_mb / 1024)} GB` : '--'}
          </div>
        </div>

        {stats?.gpus?.map((gpu) => (
          <div key={gpu.index} className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            <div className="text-sm text-slate-500">GPU {gpu.index} ({gpu.name || 'NVIDIA'})</div>
            <div className="text-2xl font-semibold mt-1">{gpu.util_pct}%</div>
            <div className="text-xs text-slate-400 mt-1">
              VRAM: {Math.round(gpu.mem_used_mb / 1024)} / {Math.round(gpu.mem_total_mb / 1024)} GB
            </div>
          </div>
        ))}
      </div>

      <div className="p-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 flex flex-col items-center justify-center min-h-[300px]">
        <Activity className="w-12 h-12 text-slate-400 mb-2" />
        <p className="text-slate-500">No active training jobs.</p>
      </div>
    </div>
  );
}
