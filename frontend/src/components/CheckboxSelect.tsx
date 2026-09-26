import React from 'react';
import { createPortal } from 'react-dom';
import { Check, ChevronDown } from 'lucide-react';
import { useWorkspaceText } from '../utils/workspaceText';
import type { StudioSelectOption } from './StudioSelect';
import './studio-select-search.css';

interface Props {
  values: string[];
  options: StudioSelectOption[];
  onValuesChange: (values: string[]) => void;
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
export default function CheckboxSelect({ values, options, onValuesChange, disabled, max, searchable = false, placeholder, className = '', ...aria }: Props) {
  const text = useWorkspaceText();
  const generatedId = React.useId();
  const triggerId = `checkbox-select-${generatedId}`;
  const listId = `${triggerId}-options`;
  const trigger = React.useRef<HTMLButtonElement>(null);
  const list = React.useRef<HTMLDivElement>(null);
  const filterInput = React.useRef<HTMLInputElement>(null);
  const [open, setOpen] = React.useState(false);
  const [active, setActive] = React.useState(-1);
  const [filter, setFilter] = React.useState('');
  const [position, setPosition] = React.useState<React.CSSProperties>({});
  const chosen = new Set(values);
  const unavailable = disabled || options.length === 0;
  const visible = open && !unavailable;
  const full = max != null && values.length >= max;
  const blocked = (option: StudioSelectOption) => !!option.disabled || (full && !chosen.has(option.value));
  const matches = options.map((option, index) => ({ option, index })).filter(({ option }) => !searchable || option.label.toLocaleLowerCase().includes(filter.trim().toLocaleLowerCase()));
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
  const openList = () => { setFilter(''); setActive(matches.find(({ option }) => !blocked(option))?.index ?? -1); setOpen(true); };

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
  React.useEffect(() => { if (visible) document.getElementById(`${listId}-${active}`)?.scrollIntoView?.({ block: 'nearest' }); }, [active, visible, listId]);
  React.useEffect(() => { if (visible && searchable) filterInput.current?.focus(); }, [visible, searchable]);

  const move = (direction: 1 | -1, edge = false) => {
    const enabled = matches.filter(({ option }) => !blocked(option)).map(({ index }) => index);
    if (!enabled.length) { setActive(-1); return; }
    const current = enabled.indexOf(active);
    setActive(edge ? enabled[direction > 0 ? 0 : enabled.length - 1] : enabled[(current + direction + enabled.length) % enabled.length] ?? enabled[0]);
  };
  const keyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Tab') { setOpen(false); return; }
    if (event.key === 'Escape') { if (visible) { event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus(); } return; }
    if (event.key === 'Enter' || (event.key === ' ' && !(event.target instanceof HTMLInputElement))) {
      event.preventDefault();
      if (visible) toggle(active); else openList();
      return;
    }
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      if (!visible) { openList(); return; }
      if (event.key === 'Home' || event.key === 'End') move(event.key === 'Home' ? 1 : -1, true); else move(event.key === 'ArrowDown' ? 1 : -1);
    }
  };

  const selectable = matches.filter(({ option }) => !option.disabled).map(({ option }) => option.value);
  return <>
    <button {...aria} id={triggerId} ref={trigger} type="button" role="combobox" disabled={unavailable} title={picked.length > 1 ? summary : undefined}
      className={`studio-select checkbox-select ${className}`} aria-haspopup="listbox" aria-expanded={visible} aria-controls={visible ? listId : undefined}
      aria-activedescendant={visible && active >= 0 ? `${listId}-${active}` : undefined}
      onClick={() => visible ? setOpen(false) : openList()} onKeyDown={keyDown} onBlur={event => { if (!list.current?.contains(event.relatedTarget)) setOpen(false); }}>
      <span className="studio-select-value">{summary}</span><ChevronDown size={13} className="studio-select-chevron" aria-hidden="true"/>
    </button>
    {visible && createPortal(<div ref={list} className="studio-select-menu checkbox-select-menu" style={position}
      onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget) && event.relatedTarget !== trigger.current) setOpen(false); }}
      onPointerDown={event => { if (!(event.target instanceof HTMLInputElement)) event.preventDefault(); }}>
      {searchable && <input ref={filterInput} type="search" role="searchbox" aria-label={`${aria['aria-label'] || text('选项', 'Options')} · ${text('搜索', 'Search')}`} placeholder={text('搜索名称…', 'Search names…')}
        value={filter} aria-controls={listId} aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined} onKeyDown={keyDown}
        onChange={event => { setFilter(event.target.value); setActive(-1); }}/>}
      <div className="checkbox-select-actions">
        <span>{max != null ? text(`已选 ${values.length} / 最多 ${max}`, `${values.length} / ${max} selected`) : text(`已选 ${values.length}`, `${values.length} selected`)}</span>
        <button type="button" className="ui-link" disabled={!selectable.length || selectable.every(value => chosen.has(value))}
          onClick={() => onValuesChange(options.filter(option => chosen.has(option.value) || selectable.includes(option.value)).map(option => option.value).slice(0, max ?? Infinity))}>{max != null && selectable.length > max ? text(`选前 ${max} 个`, `First ${max}`) : text('全选', 'All')}</button>
        <button type="button" className="ui-link" disabled={!values.length} onClick={() => onValuesChange([])}>{text('清空', 'Clear')}</button>
      </div>
      <div id={listId} role="listbox" aria-multiselectable="true" className="studio-select-results" aria-label={aria['aria-label']}>
        {matches.map(({ option, index }) => <div key={option.value} id={`${listId}-${index}`} role="option" aria-selected={chosen.has(option.value)} aria-disabled={blocked(option) || undefined}
          data-active={index === active || undefined} className="studio-select-option checkbox-select-option" onPointerMove={() => !blocked(option) && setActive(index)} onClick={() => toggle(index)}>
          <span className="checkbox-select-box" aria-hidden="true">{chosen.has(option.value) && <Check size={12}/>}</span><span>{option.label}</span>
        </div>)}
      </div>
      {!matches.length && <p role="status" className="studio-select-empty">{text('没有匹配的选项', 'No matching options')}</p>}
    </div>, document.body)}
  </>;
}
