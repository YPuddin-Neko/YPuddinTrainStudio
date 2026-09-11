import React from 'react';
import { createPortal } from 'react-dom';
import { Check, ChevronDown } from 'lucide-react';

export interface StudioSelectOption { value: string; label: string; disabled?: boolean; }
interface Props {
  id?: string;
  value: string;
  options: StudioSelectOption[];
  onValueChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
  icon?: React.ReactNode;
  'aria-invalid'?: React.AriaAttributes['aria-invalid'];
  'aria-label'?: string;
  'aria-labelledby'?: string;
  'aria-describedby'?: string;
  'data-testid'?: string;
}

/** A select-only combobox; focus stays on the trigger while its list is open. */
export default function StudioSelect({ id, value, options, onValueChange, disabled, className = '', icon, ...aria }: Props) {
  const generatedId = React.useId();
  const triggerId = id || `studio-select-${generatedId}`;
  const listId = `${triggerId}-options`;
  const trigger = React.useRef<HTMLButtonElement>(null);
  const list = React.useRef<HTMLDivElement>(null);
  const search = React.useRef({ text: '', time: 0 });
  const [open, setOpen] = React.useState(false);
  const [active, setActive] = React.useState(-1);
  const [position, setPosition] = React.useState<React.CSSProperties>({});
  const selected = options.findIndex(option => option.value === value);
  const visible = open && !disabled;
  const openList = (last = false, edge = false) => {
    const fallback = last ? options.reduce((index, option, current) => option.disabled ? index : current, -1) : options.findIndex(option => !option.disabled);
    search.current = { text: '', time: 0 };
    setActive(!edge && selected >= 0 && !options[selected].disabled ? selected : fallback);
    setOpen(true);
  };
  const choose = (index: number) => {
    if (trigger.current?.matches(':disabled')) { setOpen(false); return; }
    const option = options[index];
    if (!option || option.disabled) return;
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
      const now = Date.now();
      search.current = { text: `${now - search.current.time > 700 ? '' : search.current.text}${event.key.toLocaleLowerCase()}`, time: now };
      const index = options.findIndex(option => !option.disabled && option.label.toLocaleLowerCase().startsWith(search.current.text));
      if (!visible) setOpen(true);
      if (index >= 0) setActive(index);
    }
  };

  return <>
    <button {...aria} id={triggerId} ref={trigger} type="button" role="combobox" disabled={disabled}
      className={`studio-select ${className}`} aria-haspopup="listbox" aria-expanded={visible}
      aria-controls={visible ? listId : undefined} aria-activedescendant={visible && active >= 0 ? `${listId}-${active}` : undefined}
      onClick={() => visible ? setOpen(false) : openList()} onKeyDown={keyDown} onBlur={() => setOpen(false)}>
      {icon}<span className="studio-select-value">{options[selected]?.label ?? value}</span><ChevronDown size={13} className="studio-select-chevron" aria-hidden="true"/>
    </button>
    {visible && createPortal(<div ref={list} id={listId} role="listbox" className="studio-select-menu" style={position}
      aria-label={aria['aria-label']} aria-labelledby={aria['aria-labelledby'] || (!aria['aria-label'] ? triggerId : undefined)}
      onPointerDown={event => event.preventDefault()}>
      {options.map((option, index) => <div key={option.value} id={`${listId}-${index}`} role="option" aria-selected={option.value === value}
        aria-disabled={option.disabled || undefined} data-active={index === active || undefined} className="studio-select-option"
        onPointerMove={() => !option.disabled && setActive(index)} onClick={() => choose(index)}>
        <span>{option.label}</span>{option.value === value && <Check size={14} aria-hidden="true"/>}
      </div>)}
    </div>, document.body)}
  </>;
}
