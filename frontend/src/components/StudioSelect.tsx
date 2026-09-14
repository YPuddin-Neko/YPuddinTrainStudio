import React from 'react';
import { createPortal } from 'react-dom';
import { Check, ChevronDown } from 'lucide-react';
import { useWorkspaceText } from '../utils/workspaceText';
import './studio-select-search.css';

export interface StudioSelectOption { value: string; label: string; displayLabel?: string; disabled?: boolean; }
interface Props {
  id?: string;
  value: string;
  options: StudioSelectOption[];
  onValueChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
  icon?: React.ReactNode;
  triggerDescription?: string;
  searchable?: boolean;
  'aria-invalid'?: React.AriaAttributes['aria-invalid'];
  'aria-label'?: string;
  'aria-labelledby'?: string;
  'aria-describedby'?: string;
  'data-testid'?: string;
}

/** A select-only combobox; focus stays on the trigger while its list is open. */
export default function StudioSelect({ id, value, options, onValueChange, disabled, className = '', icon, triggerDescription, searchable = false, ...aria }: Props) {
  const text = useWorkspaceText();
  const generatedId = React.useId();
  const triggerId = id || `studio-select-${generatedId}`;
  const listId = `${triggerId}-options`;
  const trigger = React.useRef<HTMLButtonElement>(null);
  const list = React.useRef<HTMLDivElement>(null);
  const filterInput = React.useRef<HTMLInputElement>(null);
  const [filter, setFilter] = React.useState('');
  const search = React.useRef({ text: '', time: 0 });
  const [open, setOpen] = React.useState(false);
  const [active, setActive] = React.useState(-1);
  const [position, setPosition] = React.useState<React.CSSProperties>({});
  const selected = options.findIndex(option => option.value === value);
  const visible = open && !disabled;
  const matches = options.map((option, index) => ({ option, index })).filter(({ option }) => !searchable || option.label.toLocaleLowerCase().includes(filter.trim().toLocaleLowerCase()));
  const openList = (last = false, edge = false) => {
    const fallback = last ? options.reduce((index, option, current) => option.disabled ? index : current, -1) : options.findIndex(option => !option.disabled);
    search.current = { text: '', time: 0 };
    setFilter('');
    setActive(!edge && selected >= 0 && !options[selected].disabled ? selected : fallback);
    setOpen(true);
  };
  const choose = (index: number) => {
    if (trigger.current?.matches(':disabled')) { setOpen(false); return; }
    const option = options[index];
    if (!option || option.disabled || !matches.some(match => match.index === index)) return;
    setOpen(false);
    onValueChange(option.value);
    trigger.current?.focus();
  };

  React.useLayoutEffect(() => {
    if (!visible) return;
    const place = () => {
      const rect = trigger.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(Math.max(rect.width, 160), window.innerWidth - 16);
      const below = window.innerHeight - rect.bottom - 12;
      const above = rect.top - 12;
      const upwards = below < Math.min(options.length * 34 + 10, 240) && above > below;
      setPosition({ position: 'fixed', width, left: Math.max(8, Math.min(rect.left, window.innerWidth - width - 8)),
        ...(upwards ? { bottom: window.innerHeight - rect.top + 5 } : { top: rect.bottom + 5 }),
        maxHeight: Math.max(40, Math.min(280, upwards ? above : below)) });
    };
    const outside = (event: PointerEvent) => {
      if (!trigger.current?.contains(event.target as Node) && !list.current?.contains(event.target as Node)) setOpen(false);
    };
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    document.addEventListener('pointerdown', outside);
    return () => {
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
      document.removeEventListener('pointerdown', outside);
    };
  }, [visible, options.length]);
  React.useEffect(() => {
    if (visible) document.getElementById(`${listId}-${active}`)?.scrollIntoView?.({ block: 'nearest' });
  }, [active, visible, listId]);
  React.useEffect(() => { if (visible && searchable) filterInput.current?.focus(); }, [visible, searchable]);

  const keyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Tab') { setOpen(false); return; }
    if (event.key === 'Escape') { if (visible) { event.preventDefault(); event.stopPropagation(); setOpen(false); } return; }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (visible) choose(active); else openList();
      return;
    }
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      if (!visible) { openList(event.key === 'ArrowUp' || event.key === 'End', event.key === 'Home' || event.key === 'End'); return; }
      const direction = event.key === 'ArrowUp' || event.key === 'End' ? -1 : 1;
      let index = event.key === 'Home' ? -1 : event.key === 'End' ? options.length : active;
      for (let count = 0; count < options.length; count += 1) {
        index = (index + direction + options.length) % options.length;
        if (!options[index].disabled) { setActive(index); break; }
      }
      return;
    }
    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      if (searchable && !visible) {
        const next = event.key;
        search.current = { text: '', time: 0 };
        setFilter(next);
        setActive(options.findIndex(option => !option.disabled && option.label.toLocaleLowerCase().includes(next.toLocaleLowerCase())));
        setOpen(true);
        return;
      }
      const now = event.timeStamp;
      search.current = { text: `${now - search.current.time > 700 ? '' : search.current.text}${event.key.toLocaleLowerCase()}`, time: now };
      const index = options.findIndex(option => !option.disabled && option.label.toLocaleLowerCase().startsWith(search.current.text));
      if (!visible) setOpen(true);
      if (index >= 0) setActive(index);
    }
  };

  const filterKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus(); return; }
    if (event.key === 'Tab') { setOpen(false); trigger.current?.focus(); return; }
    if (event.key === 'Enter') { event.preventDefault(); if (active >= 0) choose(active); return; }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const enabled = matches.filter(({option}) => !option.disabled).map(({index}) => index);
    if (!enabled.length) { setActive(-1); return; }
    const current = enabled.indexOf(active);
    setActive(event.key === 'Home' ? enabled[0] : event.key === 'End' ? enabled[enabled.length-1] : enabled[(current + (event.key === 'ArrowUp' ? enabled.length-1 : 1)) % enabled.length]);
  };
  const optionNodes = matches.map(({ option, index }) => <div key={option.value} id={`${listId}-${index}`} role="option" aria-selected={option.value === value}
    aria-disabled={option.disabled || undefined} data-active={index === active || undefined} className="studio-select-option"
    onPointerMove={() => !option.disabled && setActive(index)} onClick={() => choose(index)}>
    <span>{option.label}</span>{option.value === value && <Check size={14} aria-hidden="true"/>}
  </div>);

  return <>
    <button {...aria} id={triggerId} ref={trigger} type="button" role="combobox" disabled={disabled}
      className={`studio-select ${className}`} aria-haspopup="listbox" aria-expanded={visible}
      aria-controls={visible ? listId : undefined} aria-activedescendant={visible && active >= 0 ? `${listId}-${active}` : undefined}
      onClick={() => visible ? setOpen(false) : openList()} onKeyDown={keyDown} onBlur={event => { if (!list.current?.contains(event.relatedTarget)) setOpen(false); }}>
      {icon}<span className="studio-select-value">{triggerDescription ? <><span className="studio-select-title">{options[selected]?.displayLabel ?? options[selected]?.label ?? value}</span><span className="studio-select-description">{triggerDescription}</span></> : options[selected]?.displayLabel ?? options[selected]?.label ?? value}</span><ChevronDown size={13} className="studio-select-chevron" aria-hidden="true"/>
    </button>
    {visible && createPortal(<div ref={list} id={searchable ? undefined : listId} role={searchable ? undefined : 'listbox'} className={`studio-select-menu${searchable ? ' studio-select-menu-searchable' : ''}`} style={position}
      aria-label={searchable ? undefined : aria['aria-label']} aria-labelledby={searchable ? undefined : aria['aria-labelledby'] || (!aria['aria-label'] ? triggerId : undefined)}
      onBlur={event => {if (!event.currentTarget.contains(event.relatedTarget) && event.relatedTarget !== trigger.current) setOpen(false);}}
      onPointerDown={event => {if (!(event.target instanceof HTMLInputElement)) event.preventDefault();}}>
      {searchable ? <><input ref={filterInput} type="search" role="searchbox" aria-label={`${aria['aria-label'] || text('选项', 'Options')} · ${text('搜索', 'Search')}`} placeholder={text('搜索版本或名称…', 'Search names…')} value={filter} aria-controls={listId} aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined} onKeyDown={filterKeyDown} onChange={event => {const next=event.target.value;setFilter(next);setActive(options.findIndex(option => !option.disabled && option.label.toLocaleLowerCase().includes(next.trim().toLocaleLowerCase())));}}/>
        <div id={listId} role="listbox" className="studio-select-results" aria-label={aria['aria-label']} aria-labelledby={aria['aria-labelledby'] || (!aria['aria-label'] ? triggerId : undefined)}>{optionNodes}</div>
        {!matches.length && <p role="status" className="studio-select-empty">{text('没有匹配的选项', 'No matching options')}</p>}
      </> : optionNodes}
    </div>, document.body)}
  </>;
}
