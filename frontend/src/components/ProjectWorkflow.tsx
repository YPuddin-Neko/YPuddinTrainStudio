import { Link } from 'react-router-dom';
import { ArrowRight, Database, Box, SlidersHorizontal, Activity } from 'lucide-react';
import { useWorkspaceText } from '../utils/workspaceText';

export type WorkspaceStep = 'data' | 'models' | 'train' | 'results';

export function ProjectWorkflow({ projectId, active }: { projectId: string; active: WorkspaceStep }) {
  const text = useWorkspaceText();
  const steps = [
    { key: 'data', label: text('训练数据', 'Training data'), detail: text('上传图片与标签', 'Images and captions'), icon: Database, url: `/projects/${projectId}?step=data` },
    { key: 'models', label: text('模型准备', 'Model setup'), detail: text('选择底模与组件', 'Base model and components'), icon: Box, url: `/projects/${projectId}?step=models` },
    { key: 'train', label: text('参数与启动', 'Configure and train'), detail: text('设置参数，检查并启动', 'Configure, validate and start'), icon: SlidersHorizontal, url: `/projects/${projectId}/train` },
    { key: 'results', label: text('任务与结果', 'Jobs and results'), detail: text('监控训练，下载模型', 'Monitor and download'), icon: Activity, url: `/projects/${projectId}?step=results` },
  ];
  return <nav aria-label={text('项目训练步骤', 'Project training steps')} className="grid grid-cols-2 xl:grid-cols-4 gap-2">
    {steps.map((step, index) => <Link key={step.key} to={step.url} aria-current={active === step.key ? 'step' : undefined}
      className={`flex items-center gap-3 rounded-lg border px-3 py-3 transition-colors ${active === step.key ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-blue-400'}`}>
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-100 dark:bg-slate-700 text-xs font-semibold">{index + 1}</span>
      <div className="min-w-0 flex-1"><span className="text-sm font-semibold">{step.label}</span><span className="hidden sm:block mt-0.5 text-xs text-slate-500 dark:text-slate-400">{step.detail}</span></div>
      <step.icon className="hidden sm:block h-4 w-4 shrink-0 opacity-60" />
    </Link>)}
  </nav>;
}

export function NextStepLink({ to, children }: { to: string; children: React.ReactNode }) {
  return <Link to={to} className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700">{children}<ArrowRight className="h-4 w-4" /></Link>;
}
