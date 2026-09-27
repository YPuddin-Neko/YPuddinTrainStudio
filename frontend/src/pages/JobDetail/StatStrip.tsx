import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { prefersReducedMotion } from '../../utils/motion';
import { useWorkspaceText } from '../../utils/workspaceText';

/**
 * Summary cards on one line. When they need more width than the page has, arrows beside the line move it one
 * screenful at a time, starting at a whole card.
 */
export default function StatStrip({ label, children }: { label: string; children: React.ReactNode }) {
  const text = useWorkspaceText();
  const strip = React.useRef<HTMLDivElement>(null);
  const [state, setState] = React.useState({ paged: false, before: false, after: false });

  const measure = React.useCallback(() => {
    const element = strip.current;
    const room = element?.parentElement?.clientWidth;
    if (!element || !room) return;
    // While paged, the cards sit at their narrowest, so scrollWidth is the width they need; compare it with the whole
    // row, arrows included, or the arrows would stay after the page widens.
    const paged = element.scrollWidth > room + 1;
    const hidden = element.scrollWidth - element.clientWidth;
    const next = { paged, before: paged && element.scrollLeft > 1, after: paged && element.scrollLeft < hidden - 1 };
    setState(current => current.paged === next.paged && current.before === next.before && current.after === next.after ? current : next);
  }, []);

  React.useLayoutEffect(() => {
    const element = strip.current;
    if (!element) return;
    measure();
    element.addEventListener('scroll', measure, { passive: true });
    // Card widths follow their values, so a longer reading can make the line overflow.
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    observer?.observe(element);
    Array.from(element.children).forEach(card => observer?.observe(card));
    return () => { element.removeEventListener('scroll', measure); observer?.disconnect(); };
  }, [measure]);

  const page = (direction: 1 | -1) => {
    const element = strip.current;
    if (!element) return;
    const cards = Array.from(element.children) as HTMLElement[];
    const view = element.getBoundingClientRect();
    const start = (card: HTMLElement) => card.getBoundingClientRect().left - view.left + element.scrollLeft;
    const target = () => {
      if (direction > 0) {
        // The first card not wholly shown becomes the first one.
        const next = cards.find(card => card.getBoundingClientRect().right > view.right + 1);
        return next ? start(next) : element.scrollWidth;
      }
      // The cards before the first one shown fill the view up to where it began.
      const first = Math.max(0, cards.findIndex(card => card.getBoundingClientRect().left >= view.left - 1));
      let index = first;
      while (index > 0 && start(cards[first]) - start(cards[index - 1]) <= element.clientWidth + 1) index -= 1;
      return cards[index] ? start(cards[index]) : 0;
    };
    element.scrollTo({ left: target(), behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  };

  const arrow = (direction: 1 | -1) => <button type="button" className="ui-btn ui-btn-sm ui-btn-icon job-stat-page"
    aria-label={direction > 0 ? text('下一组指标', 'Next metrics') : text('上一组指标', 'Previous metrics')}
    disabled={direction > 0 ? !state.after : !state.before} onClick={() => page(direction)}>
    {direction > 0 ? <ChevronRight size={15}/> : <ChevronLeft size={15}/>}
  </button>;

  return <div className={`job-stat-strip${state.before ? ' has-before' : ''}${state.after ? ' has-after' : ''}`}>
    {state.paged && arrow(-1)}
    <div ref={strip} className="job-stat-grid" role="group" aria-label={label}>{children}</div>
    {state.paged && arrow(1)}
  </div>;
}
