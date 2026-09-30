import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { prefersReducedMotion } from '../utils/motion';
import { useWorkspaceText } from '../utils/workspaceText';
import './overflow-strip.css';

type OverflowStripProps = {
  children: React.ReactNode;
  className?: string;
  containerClassName?: string;
  label: string;
  activeKey?: string | number;
  role?: React.AriaRole;
};

const ACTIVE = '[aria-selected="true"],[aria-current]:not([aria-current="false"]),[aria-pressed="true"]';

export default function OverflowStrip({ children, className = '', containerClassName = '', label, activeKey, role = 'tablist' }: OverflowStripProps) {
  const text = useWorkspaceText();
  const container = React.useRef<HTMLDivElement>(null);
  const strip = React.useRef<HTMLDivElement>(null);
  const id = React.useId();
  const [state, setState] = React.useState({ paged: false, before: false, after: false });

  const measure = React.useCallback(() => {
    const element = strip.current;
    const room = container.current?.clientWidth;
    if (!element || !room) return;
    // Compare with the full row so arrows disappear again when the container grows.
    const paged = element.scrollWidth > room + 1;
    const next = { paged, before: paged && element.scrollLeft > 1, after: paged && element.scrollLeft + element.clientWidth < element.scrollWidth - 1 };
    setState(current => current.paged === next.paged && current.before === next.before && current.after === next.after ? current : next);
  }, []);

  const reveal = React.useCallback((item?: HTMLElement) => {
    const element = strip.current;
    const target = item ?? element?.querySelector<HTMLElement>(ACTIVE);
    if (!element || !target || !element.clientWidth) return;
    const view = element.getBoundingClientRect();
    const bounds = target.getBoundingClientRect();
    if (bounds.left < view.left - 1 || bounds.width > view.width) element.scrollTo({ left: element.scrollLeft + bounds.left - view.left, behavior: 'auto' });
    else if (bounds.right > view.right + 1) element.scrollTo({ left: element.scrollLeft + bounds.right - view.right, behavior: 'auto' });
    measure();
  }, [measure]);

  React.useLayoutEffect(() => {
    const element = strip.current;
    if (!element) return;
    const resize = () => { measure(); reveal(); };
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize);
    const observe = () => {
      observer?.disconnect();
      if (container.current) observer?.observe(container.current);
      observer?.observe(element);
      for (const child of element.children) observer?.observe(child);
    };
    observe();
    resize();
    element.addEventListener('scroll', measure, { passive: true });
    const mutation = new MutationObserver(() => { observe(); resize(); });
    mutation.observe(element, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['aria-selected', 'aria-current', 'aria-pressed'] });
    return () => { observer?.disconnect(); mutation.disconnect(); element.removeEventListener('scroll', measure); };
  }, [measure, reveal]);

  React.useLayoutEffect(() => { reveal(); }, [activeKey, state.paged, reveal]);

  const page = (direction: -1 | 1) => {
    const element = strip.current;
    if (!element) return;
    element.scrollBy({ left: direction * element.clientWidth * 0.8, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  };

  const arrow = (direction: -1 | 1) => <button type="button" className="ui-btn ui-btn-sm ui-btn-icon overflow-strip-page"
    aria-label={direction < 0 ? text(`向左翻动${label}`, `Scroll ${label} left`) : text(`向右翻动${label}`, `Scroll ${label} right`)}
    aria-controls={id} disabled={direction < 0 ? !state.before : !state.after} onClick={() => page(direction)}>
    {direction < 0 ? <ChevronLeft size={15} aria-hidden="true"/> : <ChevronRight size={15} aria-hidden="true"/>}
  </button>;

  return <div ref={container} className={`overflow-strip ${containerClassName}`}>
    {state.paged && arrow(-1)}
    <div ref={strip} id={id} role={role} aria-label={label} className={`overflow-strip-track ${className}`} onFocusCapture={event => reveal(event.target as HTMLElement)}>{children}</div>
    {state.paged && arrow(1)}
  </div>;
}
