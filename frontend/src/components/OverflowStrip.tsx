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
  /** Page so that a whole item starts each view, instead of moving most of a screenful. */
  snap?: boolean;
  /** Arrow names for rows where "Scroll <label> left" reads poorly. */
  pageLabels?: { previous: string; next: string };
};

const ACTIVE = '[aria-selected="true"],[aria-current]:not([aria-current="false"]),[aria-pressed="true"]';

/** The scroll position that starts the next or previous view at a whole item. */
function itemTarget(element: HTMLElement, direction: -1 | 1): number {
  const items = (Array.from(element.children) as HTMLElement[]).filter(item => item.getAttribute('aria-hidden') !== 'true');
  const view = element.getBoundingClientRect();
  const start = (item: HTMLElement) => item.getBoundingClientRect().left - view.left + element.scrollLeft;
  if (direction > 0) {
    // The first item not wholly shown becomes the first one.
    const next = items.find(item => item.getBoundingClientRect().right > view.right + 1);
    return next ? start(next) : element.scrollWidth;
  }
  // The items before the first one shown fill the view up to where it began.
  const first = Math.max(0, items.findIndex(item => item.getBoundingClientRect().left >= view.left - 1));
  let index = first;
  while (index > 0 && start(items[first]) - start(items[index - 1]) <= element.clientWidth + 1) index -= 1;
  return items[index] ? start(items[index]) : 0;
}

/**
 * One line of tabs, choices or cards. When it needs more width than it has, arrows beside it page sideways, the
 * active item stays in view, and `data-before` / `data-after` mark the edges that hide more.
 */
export default function OverflowStrip({ children, className = '', containerClassName = '', label, activeKey, role = 'tablist', snap = false, pageLabels }: OverflowStripProps) {
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
    let viewportWidth = element.clientWidth;
    let containerWidth = container.current?.clientWidth;
    const resize = () => {
      const nextViewportWidth = element.clientWidth;
      const nextContainerWidth = container.current?.clientWidth;
      const viewportChanged = nextViewportWidth !== viewportWidth || nextContainerWidth !== containerWidth;
      viewportWidth = nextViewportWidth;
      containerWidth = nextContainerWidth;
      measure();
      if (viewportChanged) reveal();
    };
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
    const mutation = new MutationObserver(records => {
      if (records.some(record => record.type === 'childList' && record.target === element)) observe();
      measure();
      if (records.some(record => record.type === 'attributes')) reveal();
    });
    mutation.observe(element, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['aria-selected', 'aria-current', 'aria-pressed'] });
    return () => { observer?.disconnect(); mutation.disconnect(); element.removeEventListener('scroll', measure); };
  }, [measure, reveal]);

  React.useLayoutEffect(() => { reveal(); }, [activeKey, state.paged, reveal]);

  const page = (direction: -1 | 1) => {
    const element = strip.current;
    if (!element) return;
    const behavior = prefersReducedMotion() ? 'auto' : 'smooth';
    if (snap) element.scrollTo({ left: itemTarget(element, direction), behavior });
    else element.scrollBy({ left: direction * element.clientWidth * 0.8, behavior });
  };

  // Focus inside an item brings the whole item into view when paging by items.
  const focusItem = (target: HTMLElement) => {
    const element = strip.current;
    let item = target;
    if (snap && element) while (item !== element && item.parentElement && item.parentElement !== element) item = item.parentElement;
    reveal(item);
  };

  // An arrow at its edge stays focusable and only reports that it is unavailable, so keyboard focus stays on it.
  const arrow = (direction: -1 | 1) => {
    const blocked = direction < 0 ? !state.before : !state.after;
    return <button type="button" className="ui-btn ui-btn-sm ui-btn-icon overflow-strip-page"
      aria-label={direction < 0 ? pageLabels?.previous ?? text(`向左翻动${label}`, `Scroll ${label} left`) : pageLabels?.next ?? text(`向右翻动${label}`, `Scroll ${label} right`)}
      aria-controls={id} aria-disabled={blocked || undefined} onClick={() => { if (!blocked) page(direction); }}>
      {direction < 0 ? <ChevronLeft size={15} aria-hidden="true"/> : <ChevronRight size={15} aria-hidden="true"/>}
    </button>;
  };

  return <div ref={container} className={`overflow-strip ${containerClassName}`} data-before={state.before || undefined} data-after={state.after || undefined}>
    {state.paged && arrow(-1)}
    <div ref={strip} id={id} role={role} aria-label={label} className={`overflow-strip-track ${className}`} onFocusCapture={event => focusItem(event.target as HTMLElement)}>{children}</div>
    {state.paged && arrow(1)}
  </div>;
}
