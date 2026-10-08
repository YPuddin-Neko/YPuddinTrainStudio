import React from 'react';
import { createPortal } from 'react-dom';
import { TtsParameterHeadingContext } from './TtsParameterHeadingContext';
import { ChevronDown, ChevronRight, Search, X } from 'lucide-react';
import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';
import '../../styles/parameter-workspace.css';

export type TtsParameterField = { id: string; search: string; node: React.ReactNode; toggle?: boolean };
export type TtsParameterSection = { label?: string; fields: TtsParameterField[] };
export type TtsParameterGroup = { id: string; label: string; inactive?: boolean; notice?: React.ReactNode; sections: TtsParameterSection[] };

export function TtsParameterToolbar({ model, status, search, onSearch, actions, parameterActions, headingTarget }: {
  model: string; status: React.ReactNode; search: string; onSearch: (value: string) => void; actions: React.ReactNode; parameterActions?: React.ReactNode; headingTarget?: HTMLElement | null;
}) {
  const text = useWorkspaceText();
  const contextTarget = React.useContext(TtsParameterHeadingContext);
  const target = headingTarget ?? contextTarget;
  return <div className="tts-parameter-header">
    <>{target ? createPortal(<div className="tts-heading-actions"><span role="status" className="draft-indicator">{status}</span><div className="tts-editor-actions">{actions}</div></div>, target) : <div className="tts-editor-toolbar"><span>{model}</span><span role="status">{status}</span><div className="tts-editor-actions">{actions}</div></div>}</>
    <div className="training-toolbar tts-parameter-toolbar" role="group" aria-label={text('训练参数工具栏', 'Training parameter controls')}>
      <label className="config-search"><Search size={16}/><input type="search" aria-label={text('搜索训练参数', 'Search training parameters')} placeholder={text('搜索参数名称或关键字…', 'Search parameters…')} value={search} onKeyDown={event => { if (event.key === 'Enter') event.preventDefault(); }} onChange={event => onSearch(event.target.value)}/>{search && <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={text('清空搜索', 'Clear search')} onClick={() => onSearch('')}><X size={15}/></button>}</label>
      {parameterActions && <div className="toolbar-actions">{parameterActions}</div>}
    </div>
  </div>;
}

export default function TtsParameterForm({ id, groups, search, onSearch, closed, onClosedChange }: {
  id: string; groups: TtsParameterGroup[]; search: string; onSearch: (value: string) => void;
  closed: Set<string>; onClosedChange: React.Dispatch<React.SetStateAction<Set<string>>>;
}) {
  const text = useWorkspaceText();
  const root = React.useRef<HTMLDivElement>(null);
  const [active, setActive] = React.useState(groups[0]?.id || '');
  const jumped = React.useRef<{ group: string; scrollTop: number } | null>(null);
  const terms = search.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const visible = groups.map(group => ({ ...group, sections: group.sections.map(section => ({ ...section,
    fields: section.fields.filter(field => terms.every(term => `${group.label} ${section.label || ''} ${field.search}`.toLocaleLowerCase().includes(term))),
  })).filter(section => section.fields.length) })).filter(group => group.sections.length);
  const jump = (group: string, focus: boolean) => {
    onSearch('');
    onClosedChange(previous => new Set([...previous].filter(value => value !== group)));
    setActive(group);
    requestAnimationFrame(() => {
      const container = root.current;
      const section = container?.querySelector<HTMLElement>(`[data-group="${group}"]`);
      if (!container || !section) return;
      container.scrollTop += section.getBoundingClientRect().top - container.getBoundingClientRect().top - 16;
      jumped.current = { group, scrollTop: container.scrollTop };
      if (focus) section.querySelector<HTMLButtonElement>('.config-group-title')?.focus({ preventScroll: true });
    });
  };
  React.useEffect(() => {
    const container = root.current;
    if (!container) return;
    const update = () => {
      const sections = [...container.querySelectorAll<HTMLElement>('[data-group]')];
      if (jumped.current && Math.abs(container.scrollTop - jumped.current.scrollTop) < 1 && sections.some(section => section.dataset.group === jumped.current?.group)) return;
      jumped.current = null;
      const top = container.getBoundingClientRect().top + 48;
      const bottom = container.scrollTop > 0 && container.scrollTop + container.clientHeight >= container.scrollHeight - 1;
      const section = bottom ? sections.at(-1) : sections.filter(node => node.getBoundingClientRect().top <= top).at(-1) || sections[0];
      if (section?.dataset.group) setActive(section.dataset.group);
    };
    container.addEventListener('scroll', update, { passive: true });
    return () => container.removeEventListener('scroll', update);
  }, []);
  return <div className="parameter-workspace-body tts-parameter-body">
    <nav className="parameter-sections" aria-label={text('参数配置流程', 'Parameter workflow')}>
      <div className="parameter-sections-title">{text('配置流程', 'Configuration')}</div>
      <div className="parameter-sections-mobile"><StudioSelect aria-label={text('跳转到参数分组', 'Jump to parameter group')} value={active} onValueChange={group => jump(group, false)} options={groups.map(group => ({ value: group.id, label: group.label }))}/></div>
      <ol>{groups.map((group, index) => <li key={group.id}><button type="button" aria-label={text(`跳转到${group.label}`, `Jump to ${group.label}`)} aria-controls={`${id}-parameters`} aria-current={active === group.id ? 'step' : undefined} onClick={event => jump(group.id, event.detail === 0)}><span className="parameter-step-number">{String(index + 1).padStart(2, '0')}</span><span>{group.label}</span>{active === group.id && <ChevronRight size={14}/>}</button></li>)}</ol>
    </nav>
    <div ref={root} id={`${id}-parameters`} className="parameter-scroll-region tts-parameter-scroll" role="region" aria-label={search ? text('训练参数搜索结果', 'Training parameter search results') : text('训练参数内容', 'Training parameter fields')}>
      {!visible.length && <p className="workspace-message">{text('没有匹配的参数。', 'No matching parameters.')} <button type="button" className="ui-link" onClick={() => onSearch('')}>{text('清空搜索', 'Clear search')}</button></p>}
      <div className="compact-schema tts-config-form" data-testid="schema-form">{visible.map(group => <section className="config-group" key={group.id} data-group={group.id} data-stage={group.id} data-stage-active={!group.inactive}>
        <button type="button" className="config-group-title" aria-label={`${group.label}${group.inactive ? ` · ${text('本次不执行', 'Not run this time')}` : ''}`} aria-expanded={!closed.has(group.id)} aria-controls={`${id}-group-${group.id}`} onClick={() => onClosedChange(previous => { const next = new Set(previous); if (next.has(group.id)) next.delete(group.id); else next.add(group.id); return next; })}><span>{group.label}{group.inactive && <span className="config-field-label-note"> · {text('本次不执行', 'Not run this time')}</span>}</span><ChevronDown className={closed.has(group.id) ? 'tts-collapsed' : ''}/></button>
        <div id={`${id}-group-${group.id}`} className="tts-disclosure" data-open={!closed.has(group.id)} aria-hidden={closed.has(group.id)} {...(closed.has(group.id) ? { inert: '' } : {})}><div><div className="config-fields">
          {group.notice}
          {group.sections.map((section, index) => <div key={section.label || index} className="config-field-section" data-only-toggles={section.fields.every(field => field.toggle)}>{section.label && <h3>{section.label}</h3>}{section.fields.map(field => <React.Fragment key={field.id}>{field.node}</React.Fragment>)}</div>)}
        </div></div></div>
      </section>)}</div>
    </div>
  </div>;
}
