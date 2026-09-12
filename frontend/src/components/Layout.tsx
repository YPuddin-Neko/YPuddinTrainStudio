import { ApiErrorNotice } from './ApiErrorNotice';
import { ACTIVE_JOB_STATUSES, mergeJobEvent } from '../utils/jobs';
import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Activity, Folder, Layers, SlidersHorizontal, Settings as SettingsIcon, Moon, Sun, Monitor, Globe, PlayCircle, Menu, X, Plus, WifiOff, Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '../api/client';
import { SystemStats, Job, JobListResponse, Settings } from '../api/types';
import { useEventStream, useEventStreamStatus } from '../events/useEventStream';
import { EVENT_TYPES } from '../events/eventTypes';
import { formatApiError } from '../utils/errors';
import { useWorkspaceText } from '../utils/workspaceText';
import SystemTelemetry from './SystemTelemetry';
import StudioSelect from './StudioSelect';
import { ProjectSidebarContext } from './projects/ProjectSidebarContext';
import '../styles/project-sidebar.css';

const NavItem = ({ to, icon: Icon, label, active, state }: any) => (
  <Link
    to={to}
    state={state}
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
  const [projectSidebarTarget, setProjectSidebarTarget] = React.useState<HTMLDivElement | null>(null);
  const closeNavigation = React.useCallback(() => setMenuOpen(false), []);
  const projectSidebar = React.useMemo(() => ({ target: projectSidebarTarget, closeNavigation }), [projectSidebarTarget, closeNavigation]);
  const contentRef = React.useRef<HTMLDivElement>(null);
  const isDark = theme === 'dark' || (theme === 'system' && systemDark);
  const connectionStatus = useEventStreamStatus();
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [telemetryError, setTelemetryError] = React.useState('');
  const statsVersionRef = React.useRef(0);
  const previousConnectionRef = React.useRef(connectionStatus);
  const [savingUi, setSavingUi] = React.useState(false);
  const [runningJobs, setRunningJobs] = React.useState<Job[]>([]);
  React.useEffect(() => { setMenuOpen(false); }, [location.pathname, location.search]);

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

  const receiveStats = React.useCallback((next: SystemStats) => {
    statsVersionRef.current += 1;
    setStats(next);
    setTelemetryError('');
  }, []);
  const refreshTelemetry = React.useCallback(async () => {
    const version = statsVersionRef.current;
    try {
      const next = await apiClient.get<SystemStats>('/system/stats', { silent: true });
      // A late cold-start response must not overwrite a newer live reading.
      if (version === statsVersionRef.current) receiveStats(next);
    } catch (err) {
      if (version === statsVersionRef.current) setTelemetryError(formatApiError(err));
    }
  }, [receiveStats]);

  React.useEffect(() => {
    void refreshTelemetry();
    fetchJobs();
  }, [fetchJobs, refreshTelemetry]);
  React.useEffect(() => {
    if (connectionStatus === 'connected' && previousConnectionRef.current !== 'connected') void refreshTelemetry();
    previousConnectionRef.current = connectionStatus;
  }, [connectionStatus, refreshTelemetry]);

  useEventStream<SystemStats>(EVENT_TYPES.SYSTEM_STATS, receiveStats);
  useEventStream(EVENT_TYPES.JOB_STATE, fetchJobs);
  useEventStream(EVENT_TYPES.JOB_STEP, (data: any) => setRunningJobs((jobs) => jobs.map((job) => mergeJobEvent(job, data))));
  useEventStream(EVENT_TYPES.JOB_PHASE, (data: any) => setRunningJobs((jobs) => jobs.map((job) => mergeJobEvent(job, data))));

  const changeUi = async (ui: Partial<Settings['ui']>) => {
    setSavingUi(true);
    try {
      const settings = await apiClient.put<Settings>('/settings', { ui });
      window.dispatchEvent(new CustomEvent('studio.settings.changed', { detail: settings }));
    } catch { /* The shared API notice reports failed saves; keep the saved selection. */ }
    finally { setSavingUi(false); }
  };

  const navItems = [
    { to: '/', icon: Activity, label: t('nav.dashboard') },
    { to: '/projects', icon: Folder, label: t('nav.projects') },
    { to: '/queue', icon: Layers, label: t('nav.queue') },
    { to: '/presets', icon: SlidersHorizontal, label: text('参数预设', 'Training presets') },
    { to: '/settings', icon: SettingsIcon, label: t('nav.settings') },
  ];

  const runningJob = runningJobs[0];

  return (
    <ProjectSidebarContext.Provider value={projectSidebar}><div className="app-shell flex overflow-hidden">
      {/* Sidebar */}
      {menuOpen && <button className="app-sidebar-backdrop fixed inset-0 bg-black/40 md:hidden" aria-label={t('hardware.closeMenu')} onClick={() => setMenuOpen(false)} />}
      <aside className={`app-sidebar w-[184px] flex-shrink-0 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex-col fixed inset-y-0 left-0 md:static ${menuOpen ? 'flex' : 'hidden md:flex'}`}>
        <div className="sidebar-brand-row">
          <Link to="/projects" className="sidebar-brand" onClick={() => setMenuOpen(false)} aria-label="YPuddin Train Studio">
            <svg className="sidebar-brand-mark" width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
              <rect width="28" height="28" rx="8" fill="currentColor"/>
              <path d="M8 8.5L14 14L20 8.5M14 14V21" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="8" cy="8.5" r="2" fill="white"/><circle cx="20" cy="8.5" r="2" fill="white"/>
            </svg>
            <span className="sidebar-brand-text"><strong>YPuddin</strong><span>Train Studio</span></span>
          </Link>
          <button onClick={() => setMenuOpen(false)} className="sidebar-menu-close md:hidden" aria-label={t('hardware.closeMenu')}><X className="w-4 h-4" /></button>
        </div>
        <div className="px-3 pt-3"><Link to="/projects" onClick={() => setMenuOpen(false)} className="flex items-center justify-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-700 px-3 py-2.5 text-sm font-medium text-white"><Plus className="w-4 h-4" />{t('hardware.startTraining')}</Link></div>
        <nav aria-label={text('主导航', 'Main navigation')} className="flex-1 overflow-y-auto p-2 space-y-1" onClick={event => { if (event.target instanceof Element && event.target.closest('a[href]')) setMenuOpen(false); }}>
          {navItems.map((item) => (
            <React.Fragment key={item.to}>
            <NavItem
              to={item.to}
              icon={item.icon}
              label={item.label}
              active={location.pathname === item.to || (item.to !== '/' && location.pathname.startsWith(item.to))}
              state={item.to === '/settings' ? (location.pathname.startsWith('/settings') ? location.state?.backgroundLocation ? { backgroundLocation: location.state.backgroundLocation } : undefined : { backgroundLocation: location }) : undefined}
            />
            {item.to === '/projects' && <div ref={setProjectSidebarTarget} className="project-sidebar-slot" data-testid="project-sidebar-slot"/>}
            </React.Fragment>
          ))}
        </nav>
        <div className="sidebar-preferences" aria-label={text('界面偏好', 'Interface preferences')}>
          <StudioSelect className="sidebar-preference" aria-label={t('settings.theme')} value={theme} disabled={savingUi}
            icon={theme === 'system' ? <Monitor size={14}/> : theme === 'dark' ? <Moon size={14}/> : <Sun size={14}/>}
            options={[{value:'system',label:text('自动','Auto')},{value:'light',label:text('浅色','Light')},{value:'dark',label:text('深色','Dark')}]}
            onValueChange={value => void changeUi({theme:value as Settings['ui']['theme']})}/>
          <StudioSelect className="sidebar-preference" aria-label={t('settings.language')} value={i18n.resolvedLanguage === 'en' ? 'en' : 'zh-CN'} disabled={savingUi}
            icon={<Globe size={14}/>} options={[{value:'zh-CN',label:'中文'},{value:'en',label:'EN'}]}
            onValueChange={value => void changeUi({language:value as Settings['ui']['language']})}/>
        </div>
      </aside>

      {/* Main Content */}
      <main className="app-main flex-1 min-w-0 flex flex-col overflow-hidden relative">
        {/* Topbar：实时系统状态 + 训练中胶囊 */}
        <header className="app-topbar bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800" data-testid="app-topbar">
          <button className="topbar-menu p-1.5" aria-label={t('hardware.openMenu')} onClick={() => setMenuOpen(true)}><Menu className="w-5 h-5" /></button>
          {runningJob && <div className="topbar-job-slot">
              <Link
                to={`/jobs/${runningJob.id}`}
                title={runningJob.name}
                className="topbar-running-job bg-green-50 dark:bg-green-950/40 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300"
                data-testid="topbar-running-job"
              >
                <PlayCircle className="w-3.5 h-3.5 animate-pulse" />
                <span className="topbar-job-name">{runningJob.name}</span><span className="sr-only">{text('训练', 'Training')}</span>
                {runningJob.progress?.step != null && runningJob.progress?.total_steps != null && (
                  <span className="topbar-job-progress">{runningJob.progress.step}/{runningJob.progress.total_steps}</span>
                )}
              </Link>
          </div>}
          {(telemetryError || connectionStatus !== 'connected') && <div className="topbar-feedback-slot">
            {telemetryError ? <button className="text-amber-600" onClick={() => void refreshTelemetry()} title={telemetryError} aria-label={text('硬件状态读取失败，点击重试', 'Hardware status failed; retry')}><RefreshCw size={15}/><span className="sr-only" role="alert">{telemetryError}</span></button> : connectionStatus !== 'connected' && <span role="status" data-testid="event-connection" title={t(`connection.${connectionStatus}`)} className="text-amber-600">{connectionStatus === 'disconnected' ? <WifiOff size={15}/> : <Loader2 size={15} className="animate-spin"/>}<span className="sr-only">{t(`connection.${connectionStatus}`)}</span></span>}
          </div>}
          <SystemTelemetry stats={stats} />
        </header>
        <ApiErrorNotice />

        <div ref={contentRef} className="app-page-viewport" data-testid="app-page-viewport">
          <div className="app-page-frame" data-testid="app-page-frame"><React.Suspense fallback={<div className="space-y-4" role="status" data-testid="app-page-loading"><span className="sr-only">{t('common.loading')}</span><div className="h-6 w-48 rounded bg-slate-200 dark:bg-slate-800"/><div className="h-40 rounded-lg bg-slate-100 dark:bg-slate-900"/></div>}><Outlet /></React.Suspense></div>
        </div>
      </main>
    </div></ProjectSidebarContext.Provider>
  );
}
