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
  return <nav aria-label={text('项目训练步骤', 'Project training steps')} className="project-workflow">
    {steps.map((step, index) => <Link key={step.key} to={step.url} aria-current={active === step.key ? 'step' : undefined} title={step.detail}>
      <span className="step-number">{index + 1}</span><step.icon size={14}/><span>{step.label}</span>
    </Link>)}
  </nav>;
}

export function NextStepLink({ to, children }: { to: string; children: React.ReactNode }) {
  return <Link to={to} className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700">{children}<ArrowRight className="h-4 w-4" /></Link>;
}
