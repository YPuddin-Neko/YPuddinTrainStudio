import { useEffect, useRef } from 'react';

const END_PAUSE_MS = 1200;
const PIXELS_PER_SECOND = 20;

export default function ConfigFieldKey({ children, title = children }: { children: string; title?: string }) {
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let disposed = false, frame = 0, timer = 0, limit = 0, direction = 1;
    let visible = false, hovered = element.matches(':hover'), focused = document.activeElement === element;
    let dragging = false, selected = false, windowActive = document.hasFocus();

    const stop = () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(timer);
      frame = timer = 0;
    };
    const canMove = () => !disposed && limit > 1 && visible && windowActive && !document.hidden
      && !reducedMotion.matches && !hovered && !focused && !dragging && !selected;
    const start = () => {
      if (!canMove() || frame || timer) return;
      timer = window.setTimeout(() => {
        timer = 0;
        if (!canMove()) return;
        const from = element.scrollLeft;
        if (from >= limit - 1) direction = -1;
        else if (from <= 1) direction = 1;
        const to = direction > 0 ? limit : 0;
        const duration = Math.max(1200, Math.abs(to - from) / PIXELS_PER_SECOND * 1000);
        let started: number | undefined;
        const tick = (now: number) => {
          frame = 0;
          if (!canMove()) return;
          started ??= now;
          const progress = Math.min(1, (now - started) / duration);
          element.scrollLeft = from + (to - from) * (1 - Math.cos(Math.PI * progress)) / 2;
          if (progress < 1) frame = window.requestAnimationFrame(tick);
          else { direction *= -1; start(); }
        };
        frame = window.requestAnimationFrame(tick);
      }, END_PAUSE_MS);
    };
    const resume = () => { stop(); start(); };
    const measure = () => {
      limit = Math.max(0, element.scrollWidth - element.clientWidth);
      resume();
    };
    const selectionChanged = () => {
      const selection = document.getSelection();
      const next = !!selection && !selection.isCollapsed && Array.from(
        { length: selection.rangeCount }, (_, index) => selection.getRangeAt(index),
      ).some(range => range.intersectsNode(element));
      if (next !== selected) { selected = next; resume(); }
    };
    const enter = () => { hovered = true; stop(); };
    const leave = () => { hovered = false; resume(); };
    const focus = () => { focused = true; stop(); };
    const blur = () => { focused = false; resume(); };
    const down = () => { dragging = true; stop(); };
    const up = () => {
      if (!dragging) return;
      dragging = false;
      selectionChanged();
      resume();
    };
    const windowFocus = () => { windowActive = true; resume(); };
    const windowBlur = () => { windowActive = false; dragging = false; stop(); };
    const keydown = (event: KeyboardEvent) => {
      if (selected || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const target = { ArrowLeft: element.scrollLeft - 32, ArrowRight: element.scrollLeft + 32, Home: 0, End: limit }[event.key];
      if (target === undefined || limit <= 1) return;
      event.preventDefault();
      element.scrollLeft = Math.max(0, Math.min(limit, target));
    };

    // Keep one real text node so native drag selection and copying see the complete key once.
    const resize = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    resize?.observe(element);
    const intersection = typeof IntersectionObserver === 'undefined' ? null : new IntersectionObserver(entries => {
      visible = entries[0]?.isIntersecting === true;
      resume();
    });
    if (intersection) intersection.observe(element);
    else visible = true;
    element.addEventListener('pointerenter', enter);
    element.addEventListener('pointerleave', leave);
    element.addEventListener('focus', focus);
    element.addEventListener('blur', blur);
    element.addEventListener('pointerdown', down);
    element.addEventListener('keydown', keydown);
    document.addEventListener('pointerup', up);
    document.addEventListener('pointercancel', up);
    document.addEventListener('selectionchange', selectionChanged);
    document.addEventListener('visibilitychange', resume);
    window.addEventListener('focus', windowFocus);
    window.addEventListener('blur', windowBlur);
    reducedMotion.addEventListener('change', resume);
    void document.fonts?.ready.then(() => { if (!disposed) measure(); });
    measure();
    selectionChanged();
    return () => {
      disposed = true;
      stop();
      resize?.disconnect();
      intersection?.disconnect();
      element.removeEventListener('pointerenter', enter);
      element.removeEventListener('pointerleave', leave);
      element.removeEventListener('focus', focus);
      element.removeEventListener('blur', blur);
      element.removeEventListener('pointerdown', down);
      element.removeEventListener('keydown', keydown);
      document.removeEventListener('pointerup', up);
      document.removeEventListener('pointercancel', up);
      document.removeEventListener('selectionchange', selectionChanged);
      document.removeEventListener('visibilitychange', resume);
      window.removeEventListener('focus', windowFocus);
      window.removeEventListener('blur', windowBlur);
      reducedMotion.removeEventListener('change', resume);
    };
  }, [children]);

  return <code ref={ref} className="config-field-key" tabIndex={0} title={title}>{children}</code>;
}
