import React from 'react';
import { EASE_OUT, prefersReducedMotion } from '../../utils/motion';
import type { MetricChartSetting } from '../../utils/metricCharts';

type Point = { x: number; y: number };
type Gesture = {
  id: string; pointer: number; handle: HTMLButtonElement; capture: HTMLOListElement; start: Point; point: Point;
  charts: MetricChartSetting[]; order: string[]; rect: DOMRect; active: boolean; settling: boolean;
};
const inside = (point: Point, rect: DOMRect) => point.x >= rect.left && point.x <= rect.right && point.y >= rect.top && point.y <= rect.bottom;
const moveTo = (order: string[], id: string, index: number) => {
  const next = order.filter(key => key !== id); next.splice(index, 0, id); return next;
};
/** A chart picked up from the keyboard: the order it started from and where it is now. */
type Hold = { id: string; handle: HTMLButtonElement; charts: MetricChartSetting[]; order: string[] };
/** What keyboard sorting reports for screen readers; positions count from 1. */
export type SortAnnouncement = { kind: 'pick' | 'move' | 'drop' | 'cancel'; id: string; position: number; total: number };
const STEPS = new Map([['ArrowUp', -1], ['ArrowLeft', -1], ['ArrowDown', 1], ['ArrowRight', 1]]);

/**
 * Keep the card mounted in its grid slot while its body follows the pointer. From the keyboard, Space or Enter on the
 * handle picks a chart up, arrow keys (Home / End) move it, Space or Enter drops it and Escape puts it back.
 */
export function useMetricChartDrag(charts: MetricChartSetting[], onChange: (charts: MetricChartSetting[]) => void, announce?: (message: SortAnnouncement) => void) {
  const list = React.useRef<HTMLOListElement>(null);
  const gesture = React.useRef<Gesture | null>(null);
  const previous = React.useRef(new Map<string, DOMRect>());
  const animations = React.useRef<Animation[]>([]);
  const frame = React.useRef(0);
  const settleTimer = React.useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const latest = React.useRef({ charts, onChange, announce });
  React.useLayoutEffect(() => { latest.current = { charts, onChange, announce }; }, [charts, onChange, announce]);
  const [order, setOrder] = React.useState<string[] | null>(null);
  const [drag, setDrag] = React.useState<{ id: string; rect: DOMRect; settling: boolean } | null>(null);
  const hold = React.useRef<Hold | null>(null);
  const [held, setHeld] = React.useState<string | null>(null);
  const items = React.useCallback(() => Array.from(list.current?.querySelectorAll<HTMLElement>(':scope > [data-chart-id]') ?? []), []);
  const record = React.useCallback(() => { previous.current = new Map(items().map(item => [item.dataset.chartId!, item.getBoundingClientRect()])); }, [items]);
  const body = React.useCallback((id: string) => items().find(item => item.dataset.chartId === id)?.querySelector<HTMLElement>('.metric-chart-body'), [items]);
  const scrollParents = React.useCallback(() => {
    const parents: HTMLElement[] = [];
    for (let parent = list.current?.parentElement; parent; parent = parent.parentElement) {
      if (/(auto|scroll)/.test(getComputedStyle(parent).overflowY) && parent.scrollHeight > parent.clientHeight) parents.push(parent);
    }
    const root = document.scrollingElement as HTMLElement | null;
    if (root && !parents.includes(root)) parents.push(root);
    return parents;
  }, []);
  const validPoint = React.useCallback((point: Point) => {
    if (!list.current || !inside(point, list.current.getBoundingClientRect()) || point.x < 0 || point.y < 0 || point.x > window.innerWidth || point.y > window.innerHeight) return false;
    for (const parent of scrollParents()) if (parent !== document.scrollingElement && !inside(point, parent.getBoundingClientRect())) return false;
    const hit = document.elementFromPoint?.(point.x, point.y);
    return !hit || list.current.contains(hit);
  }, [scrollParents]);
  const place = React.useCallback(() => {
    const current = gesture.current;
    if (!current?.active || current.settling) return;
    const element = body(current.id);
    element?.style.setProperty('--metric-drag-x', `${current.point.x - current.start.x}px`);
    element?.style.setProperty('--metric-drag-y', `${current.point.y - current.start.y}px`);
    // Container queries and drawer transforms can establish a fixed containing block.
    if (element) {
      const rect = element.getBoundingClientRect();
      element.style.left = `${parseFloat(element.style.left) + current.rect.left + current.point.x - current.start.x - rect.left}px`;
      element.style.top = `${parseFloat(element.style.top) + current.rect.top + current.point.y - current.start.y - rect.top}px`;
    }

    if (!validPoint(current.point)) return;
    const targets = items();
    let target = targets.findIndex(item => inside(current.point, item.getBoundingClientRect()));
    if (target < 0) {
      let nearest = Infinity;
      targets.forEach((item, index) => {
        const rect = item.getBoundingClientRect();
        const distance = Math.hypot(current.point.x - (rect.left + rect.width / 2), current.point.y - (rect.top + rect.height / 2));
        if (distance < nearest) { nearest = distance; target = index; }
      });
    }
    if (target < 0 || target === current.order.indexOf(current.id)) return;
    record(); current.order = moveTo(current.order, current.id, target); setOrder(current.order);
  }, [body, items, record, validPoint]);
  const autoScroll = React.useCallback(function tick() {
    const current = gesture.current;
    if (!current?.active || current.settling) return;
    for (const parent of scrollParents()) {
      const root = parent === document.scrollingElement;
      const rect = root ? { left: 0, right: window.innerWidth, top: 0, bottom: window.innerHeight } : parent.getBoundingClientRect();
      if (current.point.x < rect.left || current.point.x > rect.right || current.point.y < rect.top || current.point.y > rect.bottom) continue;
      const distance = current.point.y < rect.top + 44 ? current.point.y - rect.top - 44 : current.point.y > rect.bottom - 44 ? current.point.y - rect.bottom + 44 : 0;
      if (!distance) continue;
      const before = parent.scrollTop;
      parent.scrollTop += Math.sign(distance) * Math.min(18, Math.ceil(Math.abs(distance) / 3));
      if (parent.scrollTop !== before) { place(); break; }
    }
    frame.current = requestAnimationFrame(tick);
  }, [place, scrollParents]);
  const release = React.useCallback((current: Gesture) => {
    try { if (current.capture.hasPointerCapture?.(current.pointer)) current.capture.releasePointerCapture(current.pointer); } catch { /* The browser may already have cancelled this pointer. */ }
  }, []);
  const reset = React.useCallback(() => {
    const current = gesture.current;
    gesture.current = null;
    if (current) { release(current); body(current.id)?.style.removeProperty('--metric-drag-x'); body(current.id)?.style.removeProperty('--metric-drag-y'); }
    cancelAnimationFrame(frame.current); clearTimeout(settleTimer.current);
    setOrder(null); setDrag(null);
  }, [body, release]);
  /** Drops a held chart where it is, or puts every chart back in its original place. */
  const endHold = React.useCallback((keep: boolean) => {
    const current = hold.current;
    if (!current) return;
    hold.current = null;
    record();
    const moved = current.order.some((key, index) => key !== current.charts[index]?.id);
    if (keep && moved) latest.current.onChange(current.order.map(key => current.charts.find(chart => chart.id === key)!));
    const final = keep ? current.order : current.charts.map(chart => chart.id);
    latest.current.announce?.({ kind: keep ? 'drop' : 'cancel', id: current.id, position: final.indexOf(current.id) + 1, total: final.length });
    setOrder(null); setHeld(null);
  }, [record]);
  const finish = React.useCallback((cancel: boolean) => {
    const current = gesture.current;
    if (!current || current.settling) return;
    if (!current.active) { reset(); return; }
    current.settling = true; cancelAnimationFrame(frame.current);
    const accept = !cancel && validPoint(current.point);
    record();
    if (!accept) current.order = current.charts.map(chart => chart.id);
    setOrder(current.order); setDrag({ id: current.id, rect: current.rect, settling: true });
  }, [record, reset, validPoint]);

  React.useLayoutEffect(() => {
    for (const animation of animations.current) animation.cancel();
    animations.current = [];
    if (!prefersReducedMotion()) for (const item of items()) {
      const key = item.dataset.chartId!;
      const from = previous.current.get(key), to = item.getBoundingClientRect();
      const element = item.querySelector<HTMLElement>('.metric-chart-body');
      if (key === gesture.current?.id || !from || !element?.animate || (from.left === to.left && from.top === to.top)) continue;
      animations.current.push(element.animate([{ transform: `translate(${from.left - to.left}px, ${from.top - to.top}px)` }, { transform: 'none' }], { duration: 220, easing: EASE_OUT }));
    }
    previous.current.clear();
    // Reordering moves the held card's element, which takes focus away from its handle.
    const holding = hold.current;
    if (holding && document.activeElement !== holding.handle) holding.handle.focus();
    const current = gesture.current;
    if (!current?.active) return;
    const element = body(current.id);
    if (!current.settling) { place(); return; }
    const target = items().find(item => item.dataset.chartId === current.id)?.getBoundingClientRect();
    const complete = () => {
      if (gesture.current !== current) return;
      if (current.order.some((key, index) => key !== current.charts[index]?.id)) {
        latest.current.onChange(current.order.map(key => current.charts.find(chart => chart.id === key)!));
      }
      reset(); current.handle.focus({ preventScroll: true });
    };
    if (!element?.animate || !target || prefersReducedMotion()) { complete(); return; }
    const animation = element.animate([
      { transform: `translate(${current.point.x - current.start.x}px, ${current.point.y - current.start.y}px)` },
      { transform: `translate(${target.left - current.rect.left}px, ${target.top - current.rect.top}px)` },
    ], { duration: 220, easing: EASE_OUT, fill: 'forwards' });
    animations.current.push(animation);
    animation.onfinish = complete;
    settleTimer.current = setTimeout(complete, 260);
  }, [order, drag, charts, body, items, place, reset]); // Pointer coordinates update the mounted body directly between grid changes.

  React.useEffect(() => {
    const move = (event: PointerEvent) => {
      const current = gesture.current;
      if (!current || current.pointer !== event.pointerId || current.settling) return;
      current.point = { x: event.clientX, y: event.clientY };
      if (!current.active && Math.hypot(current.point.x - current.start.x, current.point.y - current.start.y) < 6) return;
      event.preventDefault();
      if (!current.active) {
        current.active = true; setOrder(current.order); setDrag({ id: current.id, rect: current.rect, settling: false });
        frame.current = requestAnimationFrame(autoScroll);
      } else place();
    };
    const up = (event: PointerEvent) => {
      if (gesture.current?.pointer !== event.pointerId) return;
      gesture.current.point = { x: event.clientX, y: event.clientY }; place(); finish(false);
    };
    const cancel = (event: PointerEvent) => { if (gesture.current?.pointer === event.pointerId) finish(true); };
    // Escape is caught before a surrounding dialog or drawer closes on it.
    const key = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || !(gesture.current || hold.current)) return;
      event.preventDefault(); event.stopPropagation();
      if (hold.current) endHold(false); else finish(true);
    };
    const blur = () => finish(true);
    // A held chart goes back when the pointer or focus turns to anything else.
    const away = (event: Event) => { if (hold.current && event.target !== hold.current.handle) endHold(false); };
    window.addEventListener('pointermove', move, { passive: false }); window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', cancel); window.addEventListener('blur', blur); window.addEventListener('keydown', key, true);
    window.addEventListener('pointerdown', away, true); document.addEventListener('focusin', away);
    return () => {
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', cancel); window.removeEventListener('blur', blur); window.removeEventListener('keydown', key, true);
      window.removeEventListener('pointerdown', away, true); document.removeEventListener('focusin', away);
      cancelAnimationFrame(frame.current); clearTimeout(settleTimer.current);
      for (const animation of animations.current) animation.cancel();
      if (gesture.current) { const current = gesture.current; gesture.current = null; release(current); }
    };
  }, [autoScroll, endHold, finish, place, release]);

  return {
    list, drag, held, active: drag !== null || held !== null,
    charts: order ? order.map(key => charts.find(chart => chart.id === key)!).filter(Boolean) : charts,
    keyDown: (event: React.KeyboardEvent<HTMLButtonElement>, id: string) => {
      const current = hold.current;
      const toggle = event.key === ' ' || event.key === 'Enter';
      if (!current) {
        if (!toggle || gesture.current || charts.length < 2 || event.currentTarget.disabled) return;
        event.preventDefault();
        hold.current = { id, handle: event.currentTarget, charts, order: charts.map(chart => chart.id) };
        setOrder(hold.current.order); setHeld(id);
        latest.current.announce?.({ kind: 'pick', id, position: hold.current.order.indexOf(id) + 1, total: charts.length });
        return;
      }
      if (current.id !== id) return;
      if (toggle) { event.preventDefault(); endHold(true); return; }
      const index = current.order.indexOf(id);
      const target = event.key === 'Home' ? 0 : event.key === 'End' ? current.order.length - 1 : STEPS.has(event.key) ? index + STEPS.get(event.key)! : null;
      if (target === null) return;
      event.preventDefault();
      if (target < 0 || target >= current.order.length || target === index) return;
      record(); current.order = moveTo(current.order, id, target); setOrder(current.order);
      latest.current.announce?.({ kind: 'move', id, position: target + 1, total: current.order.length });
    },
    begin: (event: React.PointerEvent<HTMLButtonElement>, id: string) => {
      if (event.button !== 0 || event.isPrimary === false || gesture.current || hold.current || charts.length < 2 || event.currentTarget.disabled || event.currentTarget.closest('fieldset:disabled')) return;
      const item = event.currentTarget.closest<HTMLElement>('[data-chart-id]');
      if (!item || !list.current) return;
      event.preventDefault(); event.currentTarget.focus({ preventScroll: true });
      list.current.setPointerCapture?.(event.pointerId);
      const point = { x: event.clientX, y: event.clientY };
      gesture.current = { id, pointer: event.pointerId, handle: event.currentTarget, capture: list.current, start: point, point, rect: item.getBoundingClientRect(), charts, order: charts.map(chart => chart.id), active: false, settling: false };
    },
    lostCapture: () => { if (gesture.current && !gesture.current.settling) finish(true); },
    measure: record,
  };
}
