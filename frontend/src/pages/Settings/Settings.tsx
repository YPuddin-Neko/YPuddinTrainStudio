import React from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Cpu, HardDrive, FolderCog, Palette } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';
import '../../styles/settings.css';

export default function Settings() {
  const text = useWorkspaceText();
  const location = useLocation();
  const navigate = useNavigate();
  const params = new URLSearchParams(location.search);
  const selected = location.pathname.endsWith('/preferences') ? params.get('section') === 'interface' ? 'interface' : 'storage' : params.get('tab') === 'models' ? 'models' : 'runtime';
  const scroll = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => { if (scroll.current) scroll.current.scrollTop = 0; }, [selected]);
  const tabs = [
    { id: 'runtime', label: text('运行环境', 'Runtime'), Icon: Cpu },
    { id: 'models', label: text('模型权重', 'Model weights'), Icon: HardDrive },
    { id: 'storage', label: text('存储路径', 'Storage'), Icon: FolderCog },
    { id: 'interface', label: text('界面与服务', 'Appearance & service'), Icon: Palette },
  ];
  const select = (id: string) => {
    const next = new URLSearchParams(location.search);
    const preferences = id === 'storage' || id === 'interface';
    next.delete(preferences ? 'tab' : 'section');
    next.set(preferences ? 'section' : 'tab', id);
    navigate(`/settings/${preferences ? 'preferences' : 'environment'}?${next}`, { state: location.state, replace: true });
  };
  return <div className="settings-workspace" data-testid="settings-shell">
    <header className="settings-heading">
      <h1>{text('设置', 'Settings')}</h1>
      <div className="settings-tabs" role="tablist" aria-label={text('设置分区', 'Settings sections')}>
        {tabs.map(({ id, label, Icon }, index) => <button key={id} id={`settings-tab-${id}`} type="button" role="tab" aria-selected={selected === id} aria-controls="settings-content" tabIndex={selected === id ? 0 : -1} onClick={() => select(id)} onKeyDown={event => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
          select(tabs[next].id); document.getElementById(`settings-tab-${tabs[next].id}`)?.focus();
        }}><Icon size={15} /><span>{label}</span></button>)}
      </div>
    </header>
    <div ref={scroll} id="settings-content" className="settings-scroll" role="tabpanel" aria-labelledby={`settings-tab-${selected}`}><Outlet /></div>
  </div>;
}
