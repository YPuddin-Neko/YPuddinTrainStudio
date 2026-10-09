import React from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Cpu, HardDrive, FolderCog, Palette, KeyRound, Download, Tags, Settings as SettingsIcon, CircleAlert, RefreshCw } from 'lucide-react';
import { apiClient, READ_TIMEOUT_MS } from '../../api/client';
import { RestartRequiredContext } from '../../components/restartRequiredContext';
import { useWorkspaceText } from '../../utils/workspaceText';
import { SlidingIndicator } from '../../components/motion';
import { useEnterAnimation } from '../../utils/motion';
import '../../styles/settings.css';
import PageLocation from '../../components/PageLocation';
import OverflowStrip from '../../components/OverflowStrip';

export default function Settings() {
  const text = useWorkspaceText();
  const location = useLocation();
  const navigate = useNavigate();
  const params = new URLSearchParams(location.search);
  const selected = location.pathname.endsWith('/updates') ? 'updates' : location.pathname.endsWith('/page') || location.pathname.endsWith('/charts') ? 'page' : location.pathname.endsWith('/preferences') ? params.get('section') === 'downloads' ? 'downloads' : params.get('section') === 'interface' ? 'interface' : 'storage' : params.get('tab') === 'credentials' ? 'credentials' : params.get('tab') === 'models' ? 'models' : params.get('tab') === 'tagging' ? 'tagging' : 'runtime';
  const scroll = React.useRef<HTMLDivElement>(null);
  const panel = useEnterAnimation<HTMLDivElement>(selected, { skipFirst: true });
  React.useEffect(() => { if (scroll.current) scroll.current.scrollTop = 0; }, [selected]);
  const [restartRequired, setRestartRequired] = React.useState(false);
  const reported = React.useRef(false);
  const reportRestart = React.useCallback((required: boolean) => { reported.current = true; setRestartRequired(required); }, []);
  React.useEffect(() => {
    const controller = new AbortController();
    // A page that reads the flag itself reports a newer value than this first look.
    apiClient.get<{ restart_required?: boolean }>('/service/runtime', { silent: true, signal: controller.signal, timeout: READ_TIMEOUT_MS })
      .then(runtime => { if (!controller.signal.aborted && !reported.current) setRestartRequired(!!runtime.restart_required); })
      .catch(() => {});
    return () => controller.abort();
  }, []);
  const tabs = [
    { id: 'runtime', label: text('运行环境', 'Runtime'), Icon: Cpu },
    { id: 'models', label: text('模型权重', 'Model weights'), Icon: HardDrive },
    { id: 'tagging', label: text('打标', 'Tagging'), Icon: Tags },
    { id: 'credentials', label: text('访问密钥', 'Access keys'), Icon: KeyRound },
    { id: 'downloads', label: text('软件下载源', 'Package sources'), Icon: Download },
    { id: 'storage', label: text('存储路径', 'Storage'), Icon: FolderCog },
    { id: 'page', label: text('页面设置', 'Pages'), Icon: Palette },
    { id: 'interface', label: text('系统设置', 'System'), Icon: SettingsIcon },
    { id: 'updates', label: text('训练器更新', 'Trainer updates'), Icon: RefreshCw },
  ];
  const select = (id: string) => {
    if (id === 'page' || id === 'updates') { navigate(`/settings/${id}`, { state: location.state, replace: true }); return; }
    const next = new URLSearchParams(location.search);
    const preferences = id === 'storage' || id === 'interface' || id === 'downloads';
    next.delete(preferences ? 'tab' : 'section');
    next.set(preferences ? 'section' : 'tab', id);
    navigate(`/settings/${preferences ? 'preferences' : 'environment'}?${next}`, { state: location.state, replace: true });
  };
  return <div className="settings-workspace" data-testid="settings-shell">
    {/* Only as a page of its own; the settings drawer leaves the top bar to the page underneath. */}
    <PageLocation trail={[{ label: text('设置', 'Settings') }, ...tabs.filter(tab => tab.id === selected).map(tab => ({ label: tab.label }))]}/>
    <header className="settings-heading">
      <div className="settings-title-row">
        <h1>{text('设置', 'Settings')}</h1>
        {restartRequired && <p role="status" className="settings-restart-notice" data-testid="settings-restart-notice"><CircleAlert size={15} aria-hidden="true"/><span>{text('一些环境设置发生了变化，需要重启服务后才能生效，队列将在重启后继续。', 'Some environment settings changed and take effect after the service restarts. The queue continues after the restart.')}</span></p>}
      </div>
      <OverflowStrip className="settings-tabs ui-tabs" label={text('设置分区', 'Settings sections')} activeKey={selected}>
        {tabs.map(({ id, label, Icon }, index) => <button key={id} id={`settings-tab-${id}`} type="button" role="tab" aria-selected={selected === id} aria-controls="settings-content" tabIndex={selected === id ? 0 : -1} onClick={() => select(id)} onKeyDown={event => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
          select(tabs[next].id); document.getElementById(`settings-tab-${tabs[next].id}`)?.focus();
        }}><Icon size={15} /><span>{label}</span></button>)}
        <SlidingIndicator className="ui-tabs-indicator"/>
      </OverflowStrip>
    </header>
    <div ref={scroll} id="settings-content" className="settings-scroll" role="tabpanel" aria-labelledby={`settings-tab-${selected}`}><div ref={panel}><RestartRequiredContext.Provider value={reportRestart}><Outlet /></RestartRequiredContext.Provider></div></div>
  </div>;
}
