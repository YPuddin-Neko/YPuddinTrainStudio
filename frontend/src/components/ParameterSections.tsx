import React from 'react';
import { Check, CheckCircle2, ChevronRight } from 'lucide-react';
import StudioSelect from './StudioSelect';
import { useWorkspaceText } from '../utils/workspaceText';
import { PARAMETER_FLOW as flow } from '../utils/parameterWorkflow';
import { CONFIG_TAB_GROUPS, type ConfigTab, type ConfigIssue } from '../utils/configPresentation';

export default function ParameterSections({ rootRef, tab, group, onTabChange, issues = [], checked = false, planChecked = false, preset = false, hasTrainingMode = false, fullTraining = false, onRevealAdvanced }: {
  rootRef: React.RefObject<HTMLDivElement>; tab: ConfigTab; group?: string; onTabChange: (tab: ConfigTab, group?: string) => void;
  issues?: ConfigIssue[]; checked?: boolean; planChecked?: boolean; preset?: boolean; hasTrainingMode?: boolean; fullTraining?: boolean; onRevealAdvanced: () => void;
}) {
  const text = useWorkspaceText();
  const items = flow.filter(item => item.group !== 'adapter' || !fullTraining);
  const issueGroup = (issue: ConfigIssue) => issue.path.startsWith('training.') ? (issue.path.endsWith('_lr') ? 'optimizer' : 'model') : issue.path === 'model.attention' ? 'memory' : issue.path === 'dataset.batch_size' ? 'loop' : issue.path.startsWith('dataset.caption.') ? 'caption' : issue.path.split('.')[0];
  const hasGlobalIssue = issues.some(issue => !flow.some(item => item.group === issueGroup(issue)));
  const completed = (group: string) => checked && !hasGlobalIssue && !issues.some(issue => issueGroup(issue) === group)
    // A schema failure returns before model paths and cross-group constraints are checked.
    && (planChecked || !['model', 'training', 'dataset', 'adapter', 'memory', 'validation'].includes(group));
  const status = (group: string) => {
    const count = issues.filter(issue => issueGroup(issue) === group).length;
    return count ? text(`${count} 项待配置`, `${count} incomplete`) : completed(group) ? text('检查通过', 'Checked') : text('待检查', 'Not checked');
  };
  const [active, setActive] = React.useState('');
  const pending = React.useRef<string | null>(null);
  const pendingFocus = React.useRef(false);
  const jumped = React.useRef<{ group: string; scrollTop: number } | null>(null);
  const revealed = React.useRef(false);
  const jump = (group: string, moveFocus = false) => {
    const targetTab = (Object.keys(CONFIG_TAB_GROUPS) as ConfigTab[]).find(key => CONFIG_TAB_GROUPS[key].includes(group)) || 'train';
    pending.current = group;
    pendingFocus.current = moveFocus;
    jumped.current = null;
    revealed.current = false;
    onTabChange(targetTab, group);
    setActive(group);
    requestAnimationFrame(() => reveal());
  };
  const reveal = () => {
    const root = rootRef.current;
    if (!root || !pending.current) return;
    const section = Array.from(root.querySelectorAll<HTMLElement>('[data-group]')).find(node => node.dataset.group === pending.current);
    if (!section) {
      if (!root.querySelector('[data-testid="schema-form"], [data-group]')) return;
      // A deliberate jump may target a section whose controls are all advanced.
      if (!revealed.current) { revealed.current = true; onRevealAdvanced(); }
      return;
    }
    const heading = section.querySelector<HTMLButtonElement>('.config-group-title');
    if (heading?.getAttribute('aria-expanded') === 'false') heading.click();
    root.scrollTop += section.getBoundingClientRect().top - root.getBoundingClientRect().top - 16;
    jumped.current = { group: pending.current, scrollTop: root.scrollTop };
    setActive(pending.current);
    if (pendingFocus.current) heading?.focus({ preventScroll: true });
    pendingFocus.current = false;
    pending.current = null;
  };
  const revealRef = React.useRef(reveal);
  React.useLayoutEffect(() => { revealRef.current = reveal; });
  const defaultGroup = tab === 'model' ? 'model' : tab === 'data' ? 'dataset' : tab === 'advanced' ? 'objective' : hasTrainingMode ? 'model' : 'loop';
  const requestedGroup = group === 'training' ? 'model' : group;
  const locationGroup = items.some(item => item.group === requestedGroup) ? requestedGroup! : defaultGroup;
  const previousLocation = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (previousLocation.current === locationGroup) return;
    previousLocation.current = locationGroup;
    // Explicit workflow clicks already hold the exact target, including groups
    // that shared a legacy tab. External links and browser history locate it here.
    if (!pending.current) {
      pending.current = locationGroup;
      pendingFocus.current = false;
      jumped.current = null;
      revealed.current = false;
    }
    requestAnimationFrame(() => revealRef.current());
  }, [locationGroup]);
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
    observer.observe(root, { childList: true, subtree: true, attributes: true, attributeFilter: ['hidden', 'aria-expanded'] });
    update();
    return () => { observer.disconnect(); root.removeEventListener('scroll', update); };
  }, [rootRef, tab]);
  return <nav className="parameter-sections" aria-label={text('参数配置流程', 'Parameter workflow')}>
    <div className="parameter-sections-title">{text('配置流程', 'Configuration')}</div>
    <div className="parameter-sections-mobile"><StudioSelect aria-label={text('跳转到参数分组', 'Jump to parameter group')} value={items.some(item => item.group === active) ? active : ''} onValueChange={jump} options={items.map(item => ({ value: item.group, label: `${text(...item.label)} · ${status(item.group)}` }))} /></div>
    <ol>{items.map((item, index) => {
      const problems = issues.filter(issue => issueGroup(issue) === item.group);
      const passed = completed(item.group);
      const statusId = `${preset ? 'preset' : 'training'}-step-${item.group}-status`;
      return <li key={item.group}><button type="button" aria-describedby={statusId} aria-controls={preset ? 'preset-parameters' : 'training-parameters'} aria-current={active === item.group ? 'step' : undefined} onClick={event => jump(item.group, event.detail === 0)}>
        <span className="parameter-step-number">{String(index + 1).padStart(2, '0')}</span><span>{text(...item.label)}</span>
        {problems.length ? <span className="parameter-step-issues" aria-hidden="true">{problems.length}</span> : passed ? <CheckCircle2 size={16} className="parameter-step-complete" aria-hidden="true"/> : active === item.group ? <ChevronRight size={14}/> : <Check size={13} className="parameter-step-placeholder"/>}
      </button><span className="sr-only" id={statusId}>{status(item.group)}</span></li>;
    })}</ol>
  </nav>;
}
