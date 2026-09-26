import { ApiErrorNotice } from './ApiErrorNotice';
import { ACTIVE_JOB_STATUSES, mergeJobEvent } from '../utils/jobs';
import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Activity, Folder, Layers, Grid2X2, SlidersHorizontal, Settings as SettingsIcon, PlayCircle, Menu, X, Plus, WifiOff, Loader2, RefreshCw, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { apiClient } from '../api/client';
import { SystemStats, Job, JobListResponse, Settings } from '../api/types';
import { useEventStream, useEventStreamStatus } from '../events/useEventStream';
import { EVENT_TYPES } from '../events/eventTypes';
import { formatApiError } from '../utils/errors';
import { useWorkspaceText } from '../utils/workspaceText';
import SystemTelemetry from './SystemTelemetry';
import PersistentProjectSidebar from './projects/PersistentProjectSidebar';
import { ProjectSidebarContext, type ProjectSidebarSelection } from './projects/ProjectSidebarContext';
import { TopbarContext } from './topbarContext';
import '../styles/project-sidebar.css';
import '../styles/motion.css';
import BrandMark from './BrandMark';
import { useEnterAnimation } from '../utils/motion';
import { LoadingNote } from './Loading';

const NavItem = ({ to, icon: Icon, label, active, state }: any) => (
  <Link
    to={to}
    state={state}
    aria-label={label}
    title={label}
    aria-current={active ? 'page' : undefined}
    className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors text-sm ${
      active
        ? 'bg-blue-50 text-blue-600 dark:bg-slate-800 dark:text-blue-400 font-medium'
        : 'text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800'
    }`}
  >
    <Icon className="w-[18px] h-[18px] shrink-0" />
    <span>{label}</span>
  </Link>
);

export default function Layout({ navigationKey }: { navigationKey?: string }) {
  const { t, i18n } = useTranslation();
  const text = useWorkspaceText();
  const location = useLocation();
  const [theme, setTheme] = React.useState<Settings['ui']['theme']>('system');
  const [systemDark, setSystemDark] = React.useState(window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false);
  const [menuOpen, setMenuOpen] = React.useState(false);
  const [locationSlot, setLocationSlot] = React.useState<HTMLDivElement | null>(null);
  const menuTrigger = React.useRef<HTMLButtonElement>(null);
  const dismissMenu = () => {
    setMenuOpen(false);
    menuTrigger.current?.focus({ preventScroll: true });
  };
  const [collapsed,setCollapsed]=React.useState(() => {try{return localStorage.getItem('studio.sidebar.collapsed')==='true';}catch{return false;}});
  const toggleSidebar=()=>setCollapsed(previous=>{const next=!previous;try{localStorage.setItem('studio.sidebar.collapsed',String(next));}catch{/* Optional browser preference. */}return next;});
  const [projectSidebarTarget, setProjectSidebarTarget] = React.useState<HTMLDivElement | null>(null);
  const closeNavigation = React.useCallback(() => setMenuOpen(false), []);
  const [projectSelection,setProjectSelection] = React.useState<ProjectSidebarSelection | null>(null);
  const projectAction = React.useRef<{owner:symbol;beforeAction?:()=>Promise<void>} | null>(null);
  const registerProject = React.useCallback((selection:ProjectSidebarSelection,beforeAction?:()=>Promise<void>)=>{
    const owner=Symbol('project-page');
    projectAction.current={owner,beforeAction};
    setProjectSelection(previous=>JSON.stringify(previous)===JSON.stringify(selection) ? previous : selection);
    return ()=>{if(projectAction.current?.owner===owner)projectAction.current=null;};
  },[]);
  const beforeProjectAction = React.useCallback(async()=>{await projectAction.current?.beforeAction?.();},[]);
  const projectSidebar = React.useMemo(() => ({ target: projectSidebarTarget, closeNavigation, register:registerProject }), [projectSidebarTarget, closeNavigation,registerProject]);
  const contentRef = React.useRef<HTMLDivElement>(null);
  // A new page or project step fades in; query changes inside a page (tabs, filters) do not.
  const pageFrame = useEnterAnimation<HTMLDivElement>(`${location.pathname}|${new URLSearchParams(location.search).get('step') || ''}`, { distance: 0, duration: 160 });
  const isDark = theme === 'dark' || (theme === 'system' && systemDark);
  const connectionStatus = useEventStreamStatus();
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [telemetryError, setTelemetryError] = React.useState('');
  const statsVersionRef = React.useRef(0);
  const previousConnectionRef = React.useRef(connectionStatus);
  const [runningJobs, setRunningJobs] = React.useState<Job[]>([]);
  // Settings keeps this route's location frozen to preserve the workspace draft.
  // The outer router supplies its real navigation key for transient controls only.
  const currentNavigationKey = navigationKey ?? location.key;
  React.useLayoutEffect(() => { setMenuOpen(false); }, [currentNavigationKey]);

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
  useEventStream(EVENT_TYPES.JOB_XYZ_PROGRESS, (data: any) => setRunningJobs((jobs) => jobs.map((job) => mergeJobEvent(job, data))));

  const navItems = [
    { to: '/', icon: Activity, label: t('nav.dashboard') },
    { to: '/projects', icon: Folder, label: t('nav.projects') },
    { to: '/queue', icon: Layers, label: t('nav.queue') },
    { to: '/sampling', icon: Grid2X2, label: text('模型测试', 'Model testing') },
  ];

  const runningJob = runningJobs[0];

  return (
    <ProjectSidebarContext.Provider value={projectSidebar}><TopbarContext.Provider value={locationSlot}><div className={`app-shell flex overflow-hidden${collapsed ? ' sidebar-collapsed' : ''}`}>
      {/* Sidebar */}
      {menuOpen && <button className="app-sidebar-backdrop fixed inset-0 bg-black/40 md:hidden" aria-label={t('hardware.closeMenu')} onClick={dismissMenu} />}
      <aside id="app-sidebar" className={`app-sidebar w-[184px] flex-shrink-0 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex-col fixed inset-y-0 left-0 md:static ${menuOpen ? 'flex' : 'hidden md:flex'}`}>
        <div className="sidebar-brand-row">
          <Link to="/" className="sidebar-brand" onClick={() => setMenuOpen(false)} aria-label="YPuddin Train Studio">
            <BrandMark/>
            <span className="sidebar-brand-text"><strong>YPuddin</strong><span>Train Studio</span></span>
          </Link>
          <button type="button" onClick={dismissMenu} className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon sidebar-menu-close md:hidden" aria-label={t('hardware.closeMenu')}><X className="w-4 h-4" /></button>
        </div>
        <div className="sidebar-start px-3 pt-3"><Link to="/projects" aria-label={t('hardware.startTraining')} title={t('hardware.startTraining')} onClick={() => setMenuOpen(false)} className="ui-btn ui-btn-primary ui-btn-lg ui-btn-block"><Plus className="w-4 h-4" /><span>{t('hardware.startTraining')}</span></Link></div>
        <nav aria-label={text('主导航', 'Main navigation')} className="flex-1 min-h-0 overflow-y-auto p-2 space-y-1" onClick={event => { if (event.target instanceof Element && event.target.closest('a[href]')) setMenuOpen(false); }}>
          {navItems.map((item) => (
            <React.Fragment key={item.to}>
            <NavItem
              to={item.to}
              icon={item.icon}
              label={item.label}
              active={location.pathname === item.to || (item.to !== '/' && location.pathname.startsWith(item.to))}
            />
            {item.to === '/projects' && <div ref={setProjectSidebarTarget} className="project-sidebar-slot" data-testid="project-sidebar-slot">{projectSelection && <PersistentProjectSidebar key={projectSelection.project.id} selection={projectSelection} beforeAction={beforeProjectAction}/>}</div>}
            </React.Fragment>
          ))}
        </nav>
        <div className="sidebar-footer">
          <NavItem to="/presets" icon={SlidersHorizontal} label={text('参数预设', 'Training presets')}
            active={location.pathname.startsWith('/presets')}/>
          <NavItem to="/settings" icon={SettingsIcon} label={t('nav.settings')}
            active={location.pathname.startsWith('/settings')}
            state={location.pathname.startsWith('/settings') ? location.state?.backgroundLocation ? { backgroundLocation: location.state.backgroundLocation } : undefined : { backgroundLocation: location }}/>
          <button type="button" className="sidebar-collapse-control" aria-label={collapsed?text('展开侧边栏','Expand sidebar'):text('收起侧边栏','Collapse sidebar')} title={collapsed?text('展开侧边栏','Expand sidebar'):text('收起侧边栏','Collapse sidebar')} aria-expanded={!collapsed} aria-controls="app-sidebar" onClick={toggleSidebar}>
            {collapsed?<PanelLeftOpen size={18}/>:<PanelLeftClose size={18}/>}
            <span>{collapsed?text('展开侧边栏','Expand sidebar'):text('收起侧边栏','Collapse sidebar')}</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="app-main flex-1 min-w-0 flex flex-col overflow-hidden relative">
        {/* Top bar: the page's location, then running work and system status. */}
        <header className="app-topbar bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800" data-testid="app-topbar">
          <button ref={menuTrigger} type="button" className="ui-btn ui-btn-quiet ui-btn-icon topbar-menu" aria-label={t('hardware.openMenu')} aria-expanded={menuOpen} aria-controls="app-sidebar" onClick={() => setMenuOpen(true)}><Menu className="w-5 h-5" /></button>
          <div ref={setLocationSlot} className="topbar-location" data-testid="topbar-location"/>
          <div className="topbar-status">
            {runningJob && <div className="topbar-job-slot">
                <Link
                  to={`/jobs/${runningJob.id}`}
                  title={runningJob.name}
                  className="topbar-running-job bg-green-50 dark:bg-green-950/40 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300"
                  data-testid="topbar-running-job"
                >
                  <PlayCircle className="w-3.5 h-3.5 animate-pulse" />
                  <span className="topbar-job-name">{runningJob.name}</span><span className="sr-only">{runningJob.type === 'xyz' ? text('模型测试', 'Model testing') : runningJob.type === 'cache' ? text('缓存', 'Cache') : text('训练', 'Training')}</span>
                  {runningJob.type === 'xyz' && runningJob.progress?.total != null && <span className="topbar-job-progress">{runningJob.progress.done ?? 0}/{runningJob.progress.total}</span>}
                  {runningJob.type !== 'xyz' && runningJob.progress?.step != null && runningJob.progress?.total_steps != null && (
                    <span className="topbar-job-progress">{runningJob.progress.step}/{runningJob.progress.total_steps}</span>
                  )}
                </Link>
            </div>}
            {(telemetryError || connectionStatus !== 'connected') && <div className="topbar-feedback-slot">
              {telemetryError ? <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon topbar-warning" onClick={() => void refreshTelemetry()} title={telemetryError} aria-label={text('硬件状态读取失败，点击重试', 'Hardware status failed; retry')}><RefreshCw size={15}/><span className="sr-only" role="alert">{telemetryError}</span></button> : connectionStatus !== 'connected' && <span role="status" data-testid="event-connection" title={t(`connection.${connectionStatus}`)} className="topbar-warning">{connectionStatus === 'disconnected' ? <WifiOff size={15}/> : <Loader2 size={15} className="animate-spin"/>}<span className="sr-only">{t(`connection.${connectionStatus}`)}</span></span>}
            </div>}
            <SystemTelemetry stats={stats} />
          </div>
        </header>
        <ApiErrorNotice />

        <div ref={contentRef} className="app-page-viewport" data-testid="app-page-viewport">
          <div ref={pageFrame} className="app-page-frame" data-testid="app-page-frame"><React.Suspense fallback={<div data-testid="app-page-loading"><LoadingNote block label={t('common.loading')}/></div>}><Outlet /></React.Suspense></div>
        </div>
      </main>
    </div></TopbarContext.Provider></ProjectSidebarContext.Provider>
  );
}
