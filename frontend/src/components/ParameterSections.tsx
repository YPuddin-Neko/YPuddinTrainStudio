import React from 'react';
import { Check, ChevronRight } from 'lucide-react';
import StudioSelect from './StudioSelect';
import { useWorkspaceText } from '../utils/workspaceText';
import { CONFIG_TAB_GROUPS, type ConfigTab, type ConfigIssue } from '../utils/configPresentation';

const flow: Array<{ group: string; label: [string, string] }> = [
  { group: 'model', label: ['模型选择', 'Model'] },
  { group: 'training', label: ['训练方式', 'Training mode'] },
  { group: 'dataset', label: ['数据与分桶', 'Data and buckets'] },
  { group: 'caption', label: ['标签处理', 'Captions'] },
  { group: 'loop', label: ['训练时长', 'Duration'] },
  { group: 'adapter', label: ['适配器', 'Adapter'] },
  { group: 'optimizer', label: ['优化器', 'Optimizer'] },
  { group: 'scheduler', label: ['学习率调度', 'LR schedule'] },
  { group: 'memory', label: ['显存与计算', 'Memory and compute'] },
  { group: 'objective', label: ['噪声与损失', 'Noise and loss'] },
  { group: 'sampling', label: ['采样预览', 'Sample previews'] },
  { group: 'validation', label: ['验证', 'Validation'] },
  { group: 'checkpoint', label: ['保存与恢复', 'Save and resume'] },
  { group: 'logging', label: ['训练记录', 'Logging'] },
];

export default function ParameterSections({ rootRef, tab, onTabChange, issues = [], preset = false, hasTrainingMode = false, fullTraining = false, onRevealAdvanced }: {
  rootRef: React.RefObject<HTMLDivElement>; tab: ConfigTab; onTabChange: (tab: ConfigTab) => void;
  issues?: ConfigIssue[]; preset?: boolean; hasTrainingMode?: boolean; fullTraining?: boolean; onRevealAdvanced: () => void;
}) {
  const text = useWorkspaceText();
  const items = flow.filter(item => (item.group !== 'training' || hasTrainingMode) && (item.group !== 'adapter' || !fullTraining)).map(item => item.group === 'model' && preset ? {...item, label:['模型加载精度','Model loading precision'] as [string,string]} : item);
  const [active, setActive] = React.useState('');
  const pending = React.useRef<string | null>(null);
  const jumped = React.useRef<{ group: string; scrollTop: number } | null>(null);
  const revealed = React.useRef(false);
  const jump = (group: string) => {
    const targetTab = (Object.keys(CONFIG_TAB_GROUPS) as ConfigTab[]).find(key => CONFIG_TAB_GROUPS[key].includes(group)) || 'train';
    pending.current = group;
    jumped.current = null;
    revealed.current = false;
    onTabChange(targetTab);
    setActive(group);
    requestAnimationFrame(() => reveal());
  };
  const reveal = () => {
    const root = rootRef.current;
    if (!root || !pending.current) return;
    const section = Array.from(root.querySelectorAll<HTMLElement>('[data-group]')).find(node => node.dataset.group === pending.current);
    if (!section) {
      // A deliberate jump may target a section whose controls are all advanced.
      if (!revealed.current) { revealed.current = true; onRevealAdvanced(); }
      return;
    }
    const heading = section.querySelector<HTMLButtonElement>('.config-group-title');
    if (heading?.getAttribute('aria-expanded') === 'false') heading.click();
    root.scrollTop += section.getBoundingClientRect().top - root.getBoundingClientRect().top - 16;
    jumped.current = { group: pending.current, scrollTop: root.scrollTop };
    heading?.focus({ preventScroll: true });
    pending.current = null;
  };
  const revealRef = React.useRef(reveal);
  React.useLayoutEffect(() => { revealRef.current = reveal; });
  React.useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const update = () => {
      if (pending.current) { revealRef.current(); return; }
      const sections = Array.from(root.querySelectorAll<HTMLElement>('[data-group]'));
      // A short final section cannot reach the top. Keep the explicit selection
      // until the scroll position changes, including queued programmatic events.
      if (jumped.current && Math.abs(root.scrollTop - jumped.current.scrollTop) < 1 && sections.some(node => node.dataset.group === jumped.current?.group)) {
        setActive(jumped.current.group);
        return;
      }
      jumped.current = null;
      const top = root.getBoundingClientRect().top + 48;
      const atBottom = root.scrollTop > 0 && root.scrollTop + root.clientHeight >= root.scrollHeight - 1;
      const selected = atBottom ? sections.at(-1) : sections.filter(node => node.getBoundingClientRect().top <= top).at(-1) || sections[0];
      if (selected?.dataset.group) setActive(selected.dataset.group);
    };
    root.addEventListener('scroll', update, { passive: true });
    const observer = new MutationObserver(update);
    observer.observe(root, { childList: true, subtree: true });
    update();
    return () => { observer.disconnect(); root.removeEventListener('scroll', update); };
  }, [rootRef, tab]);
  return <nav className="parameter-sections" aria-label={text('参数配置流程', 'Parameter workflow')}>
    <div className="parameter-sections-title">{text('配置流程', 'Configuration')}</div>
    <div className="parameter-sections-mobile"><StudioSelect aria-label={text('跳转到参数分组', 'Jump to parameter group')} value={items.some(item => item.group === active) ? active : ''} onValueChange={jump} options={items.map(item => ({ value: item.group, label: text(...item.label) }))} /></div>
    <ol>{items.map((item, index) => {
      const problems = issues.filter(issue => (issue.path === 'model.attention' ? 'memory' : issue.path === 'dataset.batch_size' ? 'loop' : issue.path.startsWith('dataset.caption.') ? 'caption' : issue.path.split('.')[0]) === item.group);
      return <li key={item.group}><button type="button" aria-current={active === item.group ? 'step' : undefined} onClick={() => jump(item.group)}>
        <span className="parameter-step-number">{String(index + 1).padStart(2, '0')}</span><span>{text(...item.label)}</span>
        {problems.length ? <span className="parameter-step-issues" aria-label={text(`${problems.length} 项待配置`, `${problems.length} incomplete`)}>{problems.length}</span> : active === item.group ? <ChevronRight size={14}/> : <Check size={13} className="parameter-step-placeholder"/>}
      </button></li>;
    })}</ol>
  </nav>;
}
