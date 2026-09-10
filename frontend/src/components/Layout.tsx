import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Activity, Folder, Layers, Box, Settings as SettingsIcon, Moon, Sun, Globe, Cpu, Zap, HardDrive, PlayCircle } from 'lucide-react';
import { apiClient } from '../api/client';
import { SystemStats, Job, JobListResponse } from '../api/types';
import { useEventStream } from '../events/useEventStream';
import { EVENT_TYPES } from '../events/eventTypes';
import { formatBytesGB, formatBytesMB } from '../utils/format';

const NavItem = ({ to, icon: Icon, label, active }: any) => (
  <Link
    to={to}
    className={`flex items-center space-x-3 px-4 py-2.5 rounded-lg transition-colors text-sm ${
      active
        ? 'bg-blue-50 text-blue-600 dark:bg-slate-800 dark:text-blue-400 font-medium'
        : 'text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800'
    }`}
  >
    <Icon className="w-[18px] h-[18px]" />
    <span>{label}</span>
  </Link>
);

export default function Layout() {
  const { t, i18n } = useTranslation();
  const location = useLocation();
  const [isDark, setIsDark] = React.useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [runningJobs, setRunningJobs] = React.useState<Job[]>([]);

  React.useEffect(() => {
    if (isDark) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [isDark]);

  const fetchJobs = React.useCallback(() => {
    apiClient.get<JobListResponse | Job[]>('/jobs').then((res) => {
      const items = Array.isArray(res) ? res : res.items || [];
      setRunningJobs(items.filter((j) => j.status === 'running'));
    }).catch(() => {});
  }, []);

  React.useEffect(() => {
    apiClient.get<SystemStats>('/system/stats').then(setStats).catch(() => {});
    fetchJobs();
  }, [fetchJobs]);

  useEventStream<SystemStats>(EVENT_TYPES.SYSTEM_STATS, setStats);
  useEventStream(EVENT_TYPES.JOB_STATE, fetchJobs);

  const toggleTheme = () => setIsDark(!isDark);
  const toggleLang = () => {
    const newLang = i18n.language === 'zh-CN' ? 'en' : 'zh-CN';
    i18n.changeLanguage(newLang);
    localStorage.setItem('i18nextLng', newLang);
  };

  const navItems = [
    { to: '/', icon: Activity, label: t('nav.dashboard') },
    { to: '/projects', icon: Folder, label: t('nav.projects') },
    { to: '/queue', icon: Layers, label: t('nav.queue') },
    { to: '/artifacts', icon: Box, label: t('nav.artifacts') },
    { to: '/models', icon: HardDrive, label: t('nav.models') },
    { to: '/settings', icon: SettingsIcon, label: t('nav.settings') },
  ];

  const gpu = stats?.gpus?.[0];
  const runningJob = runningJobs[0];

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col">
        <div className="h-14 flex items-center px-5 border-b border-slate-200 dark:border-slate-800">
          <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
            YPuddin
          </h1>
          <span className="ml-2 text-[10px] font-mono text-slate-400 mt-1">Train Studio</span>
        </div>
        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          {navItems.map((item) => (
            <NavItem
              key={item.to}
              to={item.to}
              icon={item.icon}
              label={item.label}
              active={location.pathname === item.to || (item.to !== '/' && location.pathname.startsWith(item.to))}
            />
          ))}
        </nav>
        <div className="p-3 border-t border-slate-200 dark:border-slate-800 flex space-x-2">
          <button onClick={toggleTheme} className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400" title={t('settings.theme')}>
            {isDark ? <Sun className="w-[18px] h-[18px]" /> : <Moon className="w-[18px] h-[18px]" />}
          </button>
          <button onClick={toggleLang} className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400 flex items-center" title={t('settings.language')}>
            <Globe className="w-[18px] h-[18px] mr-1" />
            <span className="text-xs font-medium">{i18n.language === 'zh-CN' ? 'EN' : '中文'}</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden relative">
        {/* Topbar：实时系统状态 + 训练中胶囊 */}
        <header className="h-14 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-6 z-10 shrink-0">
          <div className="flex items-center space-x-3 min-w-0">
            {runningJob && (
              <Link
                to={`/jobs/${runningJob.id}`}
                className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-green-50 dark:bg-green-950/40 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-xs font-medium truncate"
                data-testid="topbar-running-job"
              >
                <PlayCircle className="w-3.5 h-3.5 animate-pulse" />
                <span className="truncate max-w-[180px]">{runningJob.name}</span>
                {runningJob.progress?.step != null && runningJob.progress?.total_steps != null && (
                  <span className="font-mono">{runningJob.progress.step}/{runningJob.progress.total_steps}</span>
                )}
              </Link>
            )}
          </div>
          <div className="flex items-center space-x-4 text-xs font-mono text-slate-500 dark:text-slate-400">
            {gpu ? (
              <span className="flex items-center space-x-1.5" title={t('dashboard.gpu')}>
                <Zap className="w-3.5 h-3.5 text-amber-500" />
                <span>{gpu.util_pct ?? 0}% · {formatBytesMB(gpu.mem_used_mb)}/{formatBytesMB(gpu.mem_total_mb)}</span>
                {gpu.temp_c != null && <span className="text-slate-400">{gpu.temp_c}°C</span>}
              </span>
            ) : (
              <span className="text-amber-500/80">{t('topbar.noGpu')}</span>
            )}
            <span className="flex items-center space-x-1.5" title={t('dashboard.cpu')}>
              <Cpu className="w-3.5 h-3.5 text-blue-500" />
              <span>{stats?.cpu_pct != null ? `${Math.round(stats.cpu_pct)}%` : '--'}</span>
            </span>
            <span className="flex items-center space-x-1.5" title={t('dashboard.ram')}>
              <Activity className="w-3.5 h-3.5 text-green-500" />
              <span>{stats?.ram ? formatBytesMB(stats.ram.used_mb) : '--'}</span>
            </span>
            <span className="flex items-center space-x-1.5" title={t('dashboard.disk')}>
              <HardDrive className="w-3.5 h-3.5 text-purple-500" />
              <span>{stats?.disks?.[0] ? `${formatBytesGB(stats.disks[0].used_gb)}/${formatBytesGB(stats.disks[0].total_gb)}` : '--'}</span>
            </span>
          </div>
        </header>

        <div className="flex-1 overflow-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
