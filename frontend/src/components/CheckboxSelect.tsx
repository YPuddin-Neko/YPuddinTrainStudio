import React from 'react';
import { createPortal } from 'react-dom';
import { Check, ChevronDown, ChevronLeft, ChevronRight } from 'lucide-react';
import { useWorkspaceText } from '../utils/workspaceText';
import type { StudioSelectOption } from './StudioSelect';
import './studio-select-search.css';

export type CheckboxSelectOption = StudioSelectOption & { group?: string };
export interface CheckboxSelectGroup { value: string; label: string }

interface Props {
  values: string[];
  options: CheckboxSelectOption[];
  onValuesChange: (values: string[]) => void;
  /**
   * With more than one group the list opens on the groups; picking one shows its options, so a long
   * list of options per group stays one level down. Searching looks through every group at once.
   */
  groups?: CheckboxSelectGroup[];
  disabled?: boolean;
  /** Further options cannot be ticked once this many are selected. */
  max?: number;
  searchable?: boolean;
  placeholder?: string;
  className?: string;
  'aria-label'?: string;
  'aria-describedby'?: string;
}

/**
 * A multi-select dropdown in the look of StudioSelect: the list stays open while options are ticked,
 * Space or Enter toggles the active one, and the trigger names the choice or counts it.
 */
export default function CheckboxSelect({ values, options, onValuesChange, groups = [], disabled, max, searchable = false, placeholder, className = '', ...aria }: Props) {
  const text = useWorkspaceText();
  const generatedId = React.useId();
  const triggerId = `checkbox-select-${generatedId}`;
  const listId = `${triggerId}-options`;
  const trigger = React.useRef<HTMLButtonElement>(null);
  const list = React.useRef<HTMLDivElement>(null);
  const filterInput = React.useRef<HTMLInputElement>(null);
  const [open, setOpen] = React.useState(false);
  const [active, setActive] = React.useState(-1);
  const [activeGroup, setActiveGroup] = React.useState(0);
  const [group, setGroup] = React.useState<string | null>(null);
  const [filter, setFilter] = React.useState('');
  const [position, setPosition] = React.useState<React.CSSProperties>({});
  const chosen = new Set(values);
  const unavailable = disabled || options.length === 0;
  const visible = open && !unavailable;
  const full = max != null && values.length >= max;
  const grouped = groups.length > 1;
  const needle = filter.trim().toLocaleLowerCase();
  // The groups themselves are listed until one is opened, unless a search looks through all of them.
  const atGroups = visible && grouped && group == null && !needle;
  const groupLabel = (value?: string) => groups.find(item => item.value === value)?.label || '';
  const blocked = (option: CheckboxSelectOption) => !!option.disabled || (full && !chosen.has(option.value));
  const matches = options.map((option, index) => ({ option, index })).filter(({ option }) =>
    (!grouped || needle || option.group === group) && (!searchable || !needle || `${option.label} ${grouped ? groupLabel(option.group) : ''}`.toLocaleLowerCase().includes(needle)));
  const picked = options.filter(option => chosen.has(option.value));
  const summary = picked.length === 0 ? (placeholder || text('请选择', 'Choose'))
    : picked.length === 1 ? picked[0].displayLabel ?? picked[0].label
      : text(`已选 ${picked.length} 个：${picked.map(option => option.displayLabel ?? option.label).join('、')}`, `${picked.length} selected: ${picked.map(option => option.displayLabel ?? option.label).join(', ')}`);

  const toggle = (index: number) => {
    const option = options[index];
    if (!option || blocked(option)) return;
    // Keep the options' order, not the order of clicks.
    const next = chosen.has(option.value) ? values.filter(value => value !== option.value) : options.filter(item => chosen.has(item.value) || item.value === option.value).map(item => item.value);
    onValuesChange(next);
  };
  const firstEnabled = (group: string | null) => options.findIndex(option => (!grouped || option.group === group) && !blocked(option));
  const openList = () => { setFilter(''); setGroup(null); setActiveGroup(0); setActive(grouped ? -1 : firstEnabled(null)); setOpen(true); };
  const enter = (value: string) => { setGroup(value); setActive(firstEnabled(value)); };
  const leave = () => { setActiveGroup(Math.max(0, groups.findIndex(item => item.value === group))); setGroup(null); setActive(-1); };

  React.useLayoutEffect(() => {
    if (!visible) return;
    const place = () => {
      const rect = trigger.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(Math.max(rect.width, 200), window.innerWidth - 16);
      const below = window.innerHeight - rect.bottom - 12, above = rect.top - 12;
      const upwards = below < Math.min(options.length * 34 + 48, 280) && above > below;
      setPosition({ position: 'fixed', width, left: Math.max(8, Math.min(rect.left, window.innerWidth - width - 8)),
        ...(upwards ? { bottom: window.innerHeight - rect.top + 5 } : { top: rect.bottom + 5 }), maxHeight: Math.max(80, Math.min(320, upwards ? above : below)) });
    };
    const outside = (event: PointerEvent) => {
      if (!trigger.current?.contains(event.target as Node) && !list.current?.contains(event.target as Node)) setOpen(false);
    };
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    document.addEventListener('pointerdown', outside);
    return () => { window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true); document.removeEventListener('pointerdown', outside); };
  }, [visible, options.length]);
  const activeId = !visible ? undefined : atGroups ? `${listId}-group-${activeGroup}` : active >= 0 ? `${listId}-${active}` : undefined;
  React.useEffect(() => { if (activeId) document.getElementById(activeId)?.scrollIntoView?.({ block: 'nearest' }); }, [activeId]);
  React.useEffect(() => { if (visible && searchable) filterInput.current?.focus(); }, [visible, searchable]);

  const move = (direction: 1 | -1, edge = false) => {
    if (atGroups) {
      setActiveGroup(current => edge ? (direction > 0 ? 0 : groups.length - 1) : (current + direction + groups.length) % groups.length);
      return;
    }
    const enabled = matches.filter(({ option }) => !blocked(option)).map(({ index }) => index);
    if (!enabled.length) { setActive(-1); return; }
    const current = enabled.indexOf(active);
    setActive(edge ? enabled[direction > 0 ? 0 : enabled.length - 1] : enabled[(current + direction + enabled.length) % enabled.length] ?? enabled[0]);
  };
  const keyDown = (event: React.KeyboardEvent) => {
    const typing = event.target instanceof HTMLInputElement && !!filter;
    if (event.key === 'Tab') { setOpen(false); return; }
    if (event.key === 'Escape') { if (visible) { event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus(); } return; }
    if (visible && grouped && group != null && !needle && (event.key === 'ArrowLeft' || (event.key === 'Backspace' && !typing))) { event.preventDefault(); leave(); return; }
    if (atGroups && event.key === 'ArrowRight') { event.preventDefault(); enter(groups[activeGroup].value); return; }
    if (event.key === 'Enter' || (event.key === ' ' && !(event.target instanceof HTMLInputElement))) {
      event.preventDefault();
      if (!visible) openList(); else if (atGroups) enter(groups[activeGroup].value); else toggle(active);
      return;
    }
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      if (!visible) { openList(); return; }
      if (event.key === 'Home' || event.key === 'End') move(event.key === 'Home' ? 1 : -1, true); else move(event.key === 'ArrowDown' ? 1 : -1);
    }
  };

  // Inside a group, select-all and clear act on that group alone.
  const scope = grouped && group != null && !needle ? options.filter(option => option.group === group) : options;
  const selectable = (atGroups ? options : matches.map(({ option }) => option)).filter(option => !option.disabled).map(option => option.value);
  const inScope = new Set(scope.map(option => option.value));
  return <>
    <button {...aria} id={triggerId} ref={trigger} type="button" role="combobox" disabled={unavailable} title={picked.length > 1 ? summary : undefined}
      className={`studio-select checkbox-select ${className}`} aria-haspopup="listbox" aria-expanded={visible} aria-controls={visible ? listId : undefined}
      aria-activedescendant={activeId}
      onClick={() => visible ? setOpen(false) : openList()} onKeyDown={keyDown} onBlur={event => { if (!list.current?.contains(event.relatedTarget)) setOpen(false); }}>
      <span className="studio-select-value">{summary}</span><ChevronDown size={13} className="studio-select-chevron" aria-hidden="true"/>
    </button>
    {visible && createPortal(<div ref={list} className="studio-select-menu checkbox-select-menu" style={position}
      onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget) && event.relatedTarget !== trigger.current) setOpen(false); }}
      onPointerDown={event => { if (!(event.target instanceof HTMLInputElement)) event.preventDefault(); }}>
      {searchable && <input ref={filterInput} type="search" role="searchbox" aria-label={`${aria['aria-label'] || text('选项', 'Options')} · ${text('搜索', 'Search')}`} placeholder={grouped ? text('搜索名称或任务…', 'Search names or runs…') : text('搜索名称…', 'Search names…')}
        value={filter} aria-controls={listId} aria-activedescendant={activeId} onKeyDown={keyDown}
        onChange={event => { setFilter(event.target.value); setActive(-1); }}/>}
      {grouped && group != null && !needle && <button type="button" className="checkbox-select-back" onClick={leave}><ChevronLeft size={13} aria-hidden="true"/><span>{text('全部任务', 'All runs')}</span><strong>{groupLabel(group)}</strong></button>}
      <div className="checkbox-select-actions">
        <span>{max != null ? text(`已选 ${values.length} / 最多 ${max}`, `${values.length} / ${max} selected`) : text(`已选 ${values.length}`, `${values.length} selected`)}</span>
        <button type="button" className="ui-link" disabled={!selectable.length || selectable.every(value => chosen.has(value))}
          onClick={() => onValuesChange(options.filter(option => chosen.has(option.value) || selectable.includes(option.value)).map(option => option.value).slice(0, max ?? Infinity))}>{max != null && selectable.length > max ? text(`选前 ${max} 个`, `First ${max}`) : text('全选', 'All')}</button>
        <button type="button" className="ui-link" disabled={scope === options ? !values.length : !values.some(value => inScope.has(value))} onClick={() => onValuesChange(scope === options ? [] : values.filter(value => !inScope.has(value)))}>{text('清空', 'Clear')}</button>
      </div>
      {atGroups ? <div id={listId} role="listbox" className="studio-select-results" aria-label={`${aria['aria-label'] || text('选项', 'Options')} · ${text('训练任务', 'Runs')}`}>
        {groups.map((item, index) => {
          const members = options.filter(option => option.group === item.value);
          const ticked = members.filter(option => chosen.has(option.value)).length;
          return <div key={item.value} id={`${listId}-group-${index}`} role="option" aria-selected={false} data-active={index === activeGroup || undefined}
            className="studio-select-option checkbox-select-group" onPointerMove={() => setActiveGroup(index)} onClick={() => enter(item.value)}>
            <span className="checkbox-select-group-name">{item.label}</span>
            <span className="checkbox-select-group-count">{ticked ? text(`已选 ${ticked} / ${members.length}`, `${ticked} / ${members.length}`) : text(`${members.length} 个`, `${members.length}`)}</span>
            <ChevronRight size={14} aria-hidden="true"/>
          </div>;
        })}
      </div> : <div id={listId} role="listbox" aria-multiselectable="true" className="studio-select-results" aria-label={aria['aria-label']}>
        {matches.map(({ option, index }) => <div key={option.value} id={`${listId}-${index}`} role="option" aria-selected={chosen.has(option.value)} aria-disabled={blocked(option) || undefined}
          data-active={index === active || undefined} className="studio-select-option checkbox-select-option" onPointerMove={() => !blocked(option) && setActive(index)} onClick={() => toggle(index)}>
          <span className="checkbox-select-box" aria-hidden="true">{chosen.has(option.value) && <Check size={12}/>}</span><span>{option.label}</span>
          {grouped && needle && <small className="checkbox-select-option-group">{groupLabel(option.group)}</small>}
        </div>)}
      </div>}
      {!atGroups && !matches.length && <p role="status" className="studio-select-empty">{text('没有匹配的选项', 'No matching options')}</p>}
    </div>, document.body)}
  </>;
}
