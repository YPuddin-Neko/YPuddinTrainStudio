import { Link } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Database, SlidersHorizontal, Activity, LayoutDashboard } from 'lucide-react';
import { projectUrl } from '../utils/projectVersions';
import { useWorkspaceText } from '../utils/workspaceText';
import '../styles/project-workspace.css';

export type WorkspaceStep = 'overview' | 'data' | 'train' | 'results';

export function ProjectWorkflow({ projectId, versionId, active, sidebar = false }: { projectId: string; versionId?: string | null; active?: WorkspaceStep; sidebar?: boolean }) {
  const text = useWorkspaceText();
  const steps = [
    { key: 'data', label: text('训练数据', 'Training data'), detail: text('上传图片与标签', 'Images and captions'), icon: Database, url: projectUrl(projectId, versionId, 'data') },
    { key: 'train', label: text('训练参数', 'Training parameters'), detail: text('设置参数，检查并启动', 'Configure, validate and start'), icon: SlidersHorizontal, url: projectUrl(projectId, versionId, 'train') },
    { key: 'results', label: text('训练结果', 'Training results'), detail: text('本版本的产物、采样与训练记录', 'Outputs, samples and training history in this version'), icon: Activity, url: projectUrl(projectId, versionId, 'results') },
  ];
  const currentIndex = steps.findIndex(step => step.key === active);
  return <div className={`project-stage-navigation${sidebar ? ' sidebar-project-stages' : ''}`}><nav aria-label={text('项目训练步骤', 'Project training steps')} className="project-workflow">
    <Link to={projectUrl(projectId, versionId, 'overview')} aria-label={text('项目概览', 'Project overview')} aria-current={active === 'overview' ? 'page' : undefined}><LayoutDashboard size={14}/><span>{text('项目概览', 'Overview')}</span></Link>
    {steps.map((step, index) => <Link key={step.key} to={step.url} aria-label={`${index + 1} ${step.label}`} aria-current={active === step.key ? 'step' : undefined} title={`${step.label} · ${step.detail}`}>
      <span className="step-number">{index + 1}</span><step.icon size={14}/><span>{step.label}</span>
    </Link>)}
  </nav>{!sidebar && <div className="project-stage-actions" aria-label={text('前后训练阶段', 'Previous and next training stages')}>
    {currentIndex > 0 && <Link to={steps[currentIndex-1].url} title={steps[currentIndex-1].label} aria-label={text(`上一阶段：${steps[currentIndex-1].label}`, `Previous stage: ${steps[currentIndex-1].label}`)}><ArrowLeft size={14}/></Link>}
    {currentIndex < steps.length-1 && <Link to={steps[currentIndex+1].url} title={steps[currentIndex+1].label} aria-label={text(`下一阶段：${steps[currentIndex+1].label}`, `Next stage: ${steps[currentIndex+1].label}`)}><span>{steps[currentIndex+1].label}</span><ArrowRight size={14}/></Link>}
  </div>}</div>;
}

export function NextStepLink({ to, children }: { to: string; children: React.ReactNode }) {
  return <Link to={to} className="ui-btn ui-btn-primary ui-btn-lg">{children}<ArrowRight className="h-4 w-4" /></Link>;
}
