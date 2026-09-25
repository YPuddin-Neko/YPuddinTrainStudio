import React from 'react';
import { EASE_OUT, prefersReducedMotion } from '../utils/motion';

const ACTIVE = '[aria-selected="true"],[aria-current]:not([aria-current="false"]),[aria-pressed="true"]';

/**
 * Tab underline or segmented thumb that follows the selected sibling. It reads the
 * selection from the siblings' aria state, so the owner only renders it last.
 */
export function SlidingIndicator({ className }: { className: string }) {
  const ref = React.useRef<HTMLSpanElement>(null);
  React.useLayoutEffect(() => {
    const indicator = ref.current;
    const container = indicator?.parentElement;
    if (!indicator || !container) return;
    let frame = 0;
    const place = () => {
      const active = [...container.children].find((child): child is HTMLElement => child !== indicator && child instanceof HTMLElement && child.matches(ACTIVE));
      if (!active || !active.offsetWidth) { indicator.removeAttribute('data-ready'); return; }
      const jump = !indicator.hasAttribute('data-ready') || prefersReducedMotion();
      if (jump) indicator.style.transition = 'none';
      indicator.style.width = `${active.offsetWidth}px`;
      indicator.style.transform = `translateX(${active.offsetLeft}px)`;
      if (jump) { void indicator.offsetWidth; indicator.style.transition = ''; }
      indicator.setAttribute('data-ready', '');
    };
    const schedule = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(place); };
    place();
    const resize = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(schedule);
    resize?.observe(container);
    for (const child of container.children) if (child !== indicator) resize?.observe(child);
    const mutation = new MutationObserver(place);
    mutation.observe(container, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-selected', 'aria-current', 'aria-pressed'] });
    return () => { cancelAnimationFrame(frame); resize?.disconnect(); mutation.disconnect(); };
  }, []);
  return <span ref={ref} className={className} aria-hidden="true"/>;
}

/** A number that briefly lifts when it changes, so the result of an action is noticed. */
export function AnimatedCount({ value, className }: { value: React.ReactNode; className?: string }) {
  const ref = React.useRef<HTMLSpanElement>(null);
  const previous = React.useRef(value);
  React.useLayoutEffect(() => {
    if (previous.current === value) return;
    previous.current = value;
    const element = ref.current;
    if (!element || typeof element.animate !== 'function' || prefersReducedMotion()) return;
    const accent = getComputedStyle(element).getPropertyValue('--studio-accent').trim() || 'currentColor';
    const animation = element.animate([{ transform: 'none' }, { transform: 'translateY(-2px) scale(1.12)', color: accent }, { transform: 'none' }], { duration: 360, easing: EASE_OUT });
    return () => animation.cancel();
  }, [value]);
  return <span ref={ref} className={className} style={{ display: 'inline-block' }}>{value}</span>;
}
