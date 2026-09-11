import { NavLink, Outlet } from 'react-router-dom';
import { Settings2, Server, SlidersHorizontal } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';

export default function Settings() {
  const text = useWorkspaceText();
  return <div className="min-w-0 space-y-4" data-testid="settings-shell">
    <header className="flex flex-wrap items-center justify-between gap-3">
      <div><h1 className="flex items-center gap-2 text-xl font-semibold"><Settings2 className="h-5 w-5 text-slate-500" />{text('设置', 'Settings')}</h1><p className="mt-1 text-xs text-slate-500">{text('准备训练环境，管理权重与产物，调整本机偏好。', 'Prepare the runtime, manage weights and outputs, and adjust local preferences.')}</p></div>
      <nav aria-label={text('设置分区', 'Settings sections')} className="flex flex-wrap gap-1 rounded-lg border border-slate-200 bg-white p-1 dark:border-slate-700 dark:bg-slate-900">
        {[{ path: 'environment', label: text('环境设置', 'Environment'), Icon: Server }, { path: 'preferences', label: text('存储与界面', 'Preferences'), Icon: SlidersHorizontal }].map(({ path, label, Icon }) => <NavLink key={path} to={`/settings/${path}`} className={({ isActive }) => `flex items-center gap-2 rounded-md px-3 py-2 text-sm ${isActive ? 'bg-blue-50 font-medium text-blue-700 dark:bg-blue-950/50 dark:text-blue-300' : 'text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-800'}`}><Icon className="h-4 w-4" />{label}</NavLink>)}
      </nav>
    </header>
    <Outlet />
  </div>;
}
