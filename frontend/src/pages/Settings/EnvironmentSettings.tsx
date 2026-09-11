import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { Cpu, HardDrive, PackageOpen } from 'lucide-react';
import { EnvironmentManagerPanel } from '../../components/EnvironmentManagerPanel';
import { useWorkspaceText } from '../../utils/workspaceText';
const Models = React.lazy(() => import('../Models/Models'));
const Artifacts = React.lazy(() => import('../Artifacts/Artifacts'));
const tabs = ['runtime', 'models', 'artifacts'] as const;

export default function EnvironmentSettings() {
  const text = useWorkspaceText();
  const [params, setParams] = useSearchParams();
  const current = tabs.find(tab => tab === params.get('tab')) || 'runtime';
  const labels = { runtime: text('运行环境', 'Runtime'), models: text('模型权重', 'Model weights'), artifacts: text('训练产物', 'Training outputs') };
  const icons = { runtime: Cpu, models: HardDrive, artifacts: PackageOpen };
  const selectTab = (tab: typeof tabs[number]) => { const next = new URLSearchParams(params); next.set('tab', tab); setParams(next); };
  return <section className="min-w-0 space-y-4" data-testid="environment-settings">
    <div role="tablist" aria-label={text('环境设置分类', 'Environment categories')} className="flex gap-1 overflow-x-auto border-b border-slate-200 dark:border-slate-700">
      {tabs.map(tab => { const Icon = icons[tab]; return <button key={tab} id={`environment-tab-${tab}`} role="tab" type="button" aria-selected={current === tab} aria-controls={`environment-panel-${tab}`} tabIndex={current === tab ? 0 : -1} onClick={() => selectTab(tab)} onKeyDown={(event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (tabs.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
        selectTab(tabs[index]); document.getElementById(`environment-tab-${tabs[index]}`)?.focus();
      }} className={`flex shrink-0 items-center gap-2 border-b-2 px-4 py-2.5 text-sm ${current === tab ? 'border-blue-600 font-medium text-blue-700 dark:text-blue-300' : 'border-transparent text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-800'}`}><Icon className="h-4 w-4" />{labels[tab]}</button>; })}
    </div>
    <div id={`environment-panel-${current}`} role="tabpanel" aria-labelledby={`environment-tab-${current}`} className="min-w-0">
      <React.Suspense fallback={<p role="status" className="py-6 text-sm text-slate-500">{text('正在加载…', 'Loading…')}</p>}>
        {current === 'runtime' ? <EnvironmentManagerPanel /> : current === 'models' ? <Models embedded /> : <Artifacts embedded />}
      </React.Suspense>
    </div>
  </section>;
}
