import { ApiErrorNotice } from './ApiErrorNotice';
import { ACTIVE_JOB_STATUSES, mergeJobEvent } from '../utils/jobs';
import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Activity, Folder, Layers, Settings as SettingsIcon, Moon, Sun, Globe, Cpu, Zap, HardDrive, PlayCircle, Menu, X, Plus } from 'lucide-react';
import { apiClient } from '../api/client';
import { SystemStats, SystemInfo, Job, JobListResponse, Settings, isAppleSilicon } from '../api/types';
import { useEventStream, useEventStreamStatus } from '../events/useEventStream';
import { EVENT_TYPES } from '../events/eventTypes';
import { formatBytesGB, formatBytesMB } from '../utils/format';
import { useWorkspaceText } from '../utils/workspaceText';

const NavItem = ({ to, icon: Icon, label, active }: any) => (
  <Link
    to={to}
    aria-current={active ? 'page' : undefined}
    className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors text-sm ${
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
  const text = useWorkspaceText();
  const location = useLocation();
  const [theme, setTheme] = React.useState<Settings['ui']['theme']>('system');
  const [systemDark, setSystemDark] = React.useState(window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false);
  const [menuOpen, setMenuOpen] = React.useState(false);
  const [selectedGpu, setSelectedGpu] = React.useState(0);
  const contentRef = React.useRef<HTMLDivElement>(null);
  const isDark = theme === 'dark' || (theme === 'system' && systemDark);
  const connectionStatus = useEventStreamStatus();
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [sysInfo, setSysInfo] = React.useState<SystemInfo | null>(null);
  const [runningJobs, setRunningJobs] = React.useState<Job[]>([]);

  React.useEffect(() => {
    if (isDark) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [isDark]);
  React.useEffect(() => { document.documentElement.lang = i18n.resolvedLanguage || 'zh-CN'; }, [i18n.resolvedLanguage]);
  React.useEffect(() => {
    const panel = contentRef.current;
    if (!panel) return;
    panel.scrollTop = 0;
    if (!location.hash) return;
    let targetId: string;
    try { targetId = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const reveal = () => {
      const target = document.getElementById(targetId);
      if (!target || !panel.contains(target)) return false;
      target.scrollIntoView({block: 'start'});
      return true;
    };
    if (reveal()) return;
    const observer = new MutationObserver(() => { if (reveal()) observer.disconnect(); });
    observer.observe(panel, {childList: true, subtree: true});
    const timer = setTimeout(() => observer.disconnect(), 5000);
    return () => { observer.disconnect(); clearTimeout(timer); };
  }, [location.pathname, location.search, location.hash]);

  React.useEffect(() => {
    const apply = (settings: Settings) => {
      setTheme(settings.ui.theme);
      void i18n.changeLanguage(settings.ui.language);
    };
    apiClient.get<Settings>('/settings').then(apply).catch(() => {});
    const changed = (event: Event) => apply((event as CustomEvent<Settings>).detail);
    window.addEventListener('studio.settings.changed', changed);
    const media = window.matchMedia?.('(prefers-color-scheme: dark)');
    const update = (event: MediaQueryListEvent) => setSystemDark(event.matches);
    media?.addEventListener?.('change', update);
    return () => { window.removeEventListener('studio.settings.changed', changed); media?.removeEventListener?.('change', update); };
  }, [i18n]);

  const fetchJobs = React.useCallback(() => {
    apiClient.get<JobListResponse | Job[]>('/jobs', { params: { status: ACTIVE_JOB_STATUSES } }).then((res) => {
      const items = Array.isArray(res) ? res : res.items || [];
      setRunningJobs(items.filter((j) => ACTIVE_JOB_STATUSES.split(',').includes(j.status)));
    }).catch(() => {});
  }, []);

  React.useEffect(() => {
    apiClient.get<SystemStats>('/system/stats').then(setStats).catch(() => {});
    apiClient.get<SystemInfo>('/system/info').then(setSysInfo).catch(() => {});
    fetchJobs();
  }, [fetchJobs]);

  useEventStream<SystemStats>(EVENT_TYPES.SYSTEM_STATS, setStats);
  useEventStream(EVENT_TYPES.JOB_STATE, fetchJobs);
  useEventStream(EVENT_TYPES.JOB_STEP, (data: any) => setRunningJobs((jobs) => jobs.map((job) => mergeJobEvent(job, data))));
  useEventStream(EVENT_TYPES.JOB_PHASE, (data: any) => setRunningJobs((jobs) => jobs.map((job) => mergeJobEvent(job, data))));

  const toggleTheme = () => {
    const next = isDark ? 'light' : 'dark';
    setTheme(next);
    void apiClient.put<Settings>('/settings', { ui: { theme: next } }).then(settings => window.dispatchEvent(new CustomEvent('studio.settings.changed', { detail: settings }))).catch(() => {});
  };
  const toggleLang = () => {
    const newLang = i18n.language === 'zh-CN' ? 'en' : 'zh-CN';
    i18n.changeLanguage(newLang);
    localStorage.setItem('i18nextLng', newLang);
    void apiClient.put<Settings>('/settings', { ui: { language: newLang } }).catch(() => {});
  };

  const navItems = [
    { to: '/', icon: Activity, label: t('nav.dashboard') },
    { to: '/projects', icon: Folder, label: t('nav.projects') },
    { to: '/queue', icon: Layers, label: t('nav.queue') },
    { to: '/settings', icon: SettingsIcon, label: t('nav.settings') },
  ];

  const gpu = stats?.gpus?.find(item => item.index === selectedGpu) || stats?.gpus?.[0];
  const runningJob = runningJobs[0];

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      {menuOpen && <button className="fixed inset-0 z-20 bg-black/40 md:hidden" aria-label={t('hardware.closeMenu')} onClick={() => setMenuOpen(false)} />}
      <aside className={`w-[184px] flex-shrink-0 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex-col fixed inset-y-0 left-0 z-30 md:static ${menuOpen ? 'flex' : 'hidden md:flex'}`}>
        <div className="h-12 flex shrink-0 items-center px-3 border-b border-slate-200 dark:border-slate-800">
          <h1 className="text-base font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
            YPuddin
          </h1>
          <span className="ml-2 hidden md:inline text-[9px] font-mono text-slate-400 mt-1">Train Studio</span>
          <button onClick={() => setMenuOpen(false)} className="ml-auto md:hidden p-2" aria-label={t('hardware.closeMenu')}><X className="w-4 h-4" /></button>
        </div>
        <div className="px-3 pt-3"><Link to="/projects" onClick={() => setMenuOpen(false)} className="flex items-center justify-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-700 px-3 py-2.5 text-sm font-medium text-white"><Plus className="w-4 h-4" />{t('hardware.startTraining')}</Link></div>
        <nav aria-label={text('主导航', 'Main navigation')} className="flex-1 overflow-y-auto p-2 space-y-1" onClick={() => setMenuOpen(false)}>
          {navItems.map((item) => (
            <NavItem
              key={item.to}
              to={item.to}
              icon={item.icon}
              label={item.label}
              active={location.pathname === item.to || (item.to !== '/' && location.pathname.startsWith(item.to))}
            />
          ))}
          {location.pathname.startsWith('/settings') && <div className="ml-5 border-l border-slate-200 pl-2 dark:border-slate-700">
            {[['environment', text('环境设置', 'Environment')], ['preferences', text('存储与界面', 'Preferences')]].map(([path, label]) => <Link key={path} to={`/settings/${path}`} aria-current={location.pathname.endsWith(path) ? 'page' : undefined} className={`block rounded-md px-3 py-2 text-xs ${location.pathname.endsWith(path) ? 'font-medium text-blue-600 dark:text-blue-300' : 'text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-800'}`}>{label}</Link>)}
          </div>}
        </nav>
        {sysInfo?.ypuddin && <div className="px-3 py-2 text-[11px] text-slate-400" title={t('hardware.serverVersion')}>v{sysInfo.ypuddin}</div>}
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
      <main className="flex-1 min-w-0 flex flex-col overflow-hidden relative">
        {/* Topbar：实时系统状态 + 训练中胶囊 */}
        <header className="h-12 min-h-12 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 flex gap-2 items-center justify-between px-3 md:px-4 z-10 shrink-0">
          <button className="shrink-0 p-1.5 md:hidden" aria-label={t('hardware.openMenu')} onClick={() => setMenuOpen(true)}><Menu className="w-5 h-5" /></button>
          <div className="flex min-w-0 flex-1 items-center gap-2">
            {runningJob && (
              <Link
                to={`/jobs/${runningJob.id}`}
                title={runningJob.name}
                className="flex min-w-0 items-center space-x-1.5 px-2 py-1.5 rounded-full bg-green-50 dark:bg-green-950/40 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-xs font-medium truncate"
                data-testid="topbar-running-job"
              >
                <PlayCircle className="w-3.5 h-3.5 animate-pulse" />
                <span className="hidden md:inline truncate max-w-[100px] xl:max-w-[180px]">{runningJob.name}</span><span className="md:hidden">{text('训练', 'Training')}</span>
                {runningJob.progress?.step != null && runningJob.progress?.total_steps != null && (
                  <span className="hidden lg:inline font-mono">{runningJob.progress.step}/{runningJob.progress.total_steps}</span>
                )}
              </Link>
            )}
          </div>
          <div className="flex min-w-0 shrink-0 items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
            <span role="status" data-testid="event-connection" className={connectionStatus === 'connected' ? 'text-green-600' : 'text-amber-600'}>
              <span className="hidden sm:inline">{t(`connection.${connectionStatus}`)}</span><span className="sm:hidden" aria-label={t(`connection.${connectionStatus}`)}><span aria-hidden="true" className={`inline-block h-2 w-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-amber-500'}`} /></span>
            </span>
            {(stats?.gpus?.length || 0) > 1 && <select aria-label={t('dashboard.gpu')} value={gpu?.index} onChange={event => setSelectedGpu(Number(event.target.value))} className="bg-transparent max-w-20 sm:max-w-40 rounded border border-slate-200 dark:border-slate-700 p-1">{stats?.gpus.map(item => <option key={item.index} value={item.index}>{item.name}</option>)}</select>}
            {gpu ? (
              <Link to="/" className="flex items-center gap-2 hover:text-blue-600" title={`${gpu.name}${gpu.telemetry_note ? ' · ' + t(`hardware.${gpu.telemetry_note}`) : ''}`}>
                <Zap className="w-3.5 h-3.5 text-amber-500" />
                <span>
                  <span className="font-semibold text-slate-700 dark:text-slate-200" data-testid="topbar-gpu-power">{t('hardware.power')} {gpu.power_w != null ? `${Math.round(gpu.power_w)} W` : t('hardware.unavailable')}</span>
                  <span className="hidden lg:inline ml-2 font-mono">{gpu.util_pct != null ? `${gpu.util_pct}% · ` : ''}{formatBytesMB(gpu.mem_used_mb)}/{formatBytesMB(gpu.mem_total_mb)}</span>
                </span>
                {gpu.temp_c != null && <span className="hidden sm:inline text-slate-400">{gpu.temp_c}°C</span>}
              </Link>
            ) : isAppleSilicon(sysInfo) ? (
              <span className="text-blue-500/80" data-testid="topbar-apple-gpu">
                {t('topbar.appleGpuWip', 'Apple GPU · MPS（适配中）')}
              </span>
            ) : (
              <span className="text-amber-500/80">{t('topbar.noGpu')}</span>
            )}
            <span className="hidden xl:flex items-center space-x-1.5" title={t('dashboard.cpu')}>
              <Cpu className="w-3.5 h-3.5 text-blue-500" />
              <span>{stats?.cpu_pct != null ? `${Math.round(stats.cpu_pct)}%` : '--'}</span>
            </span>
            <span className="hidden xl:flex items-center space-x-1.5" title={t('dashboard.ram')}>
              <Activity className="w-3.5 h-3.5 text-green-500" />
              <span>{stats?.ram ? formatBytesMB(stats.ram.used_mb) : '--'}</span>
            </span>
            <span className="hidden 2xl:flex items-center space-x-1.5" title={t('dashboard.disk')}>
              <HardDrive className="w-3.5 h-3.5 text-purple-500" />
              <span>{stats?.disks?.[0] ? `${formatBytesGB(stats.disks[0].used_gb)}/${formatBytesGB(stats.disks[0].total_gb)}` : '--'}</span>
            </span>
          </div>
        </header>
        <ApiErrorNotice />

        <div ref={contentRef} className="flex-1 overflow-auto p-3 md:p-4">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
