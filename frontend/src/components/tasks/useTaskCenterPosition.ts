import React from 'react';

const SIZE = 44;
const MARGIN = 12;
const GAP = 8;
const STORAGE_KEY = 'studio.task-center.position';
type Point = { x: number; y: number };

function viewport() {
  const visible = window.visualViewport;
  return { x: visible?.offsetLeft ?? 0, y: visible?.offsetTop ?? 0,
    width: visible?.width ?? window.innerWidth, height: visible?.height ?? window.innerHeight };
}
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(value, Math.max(min, max)));
function bounded(point: Point, view = viewport()): Point {
  return { x: clamp(point.x, view.x + MARGIN, view.x + view.width - SIZE - MARGIN),
    y: clamp(point.y, view.y + MARGIN, view.y + view.height - SIZE - MARGIN) };
}
function initialPosition() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) return bounded(saved);
  } catch { /* Optional browser preference. */ }
  const view = viewport();
  return bounded({ x: view.x + view.width - SIZE - MARGIN, y: view.y + 60 }, view);
}
function remember(point: Point) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(point)); } catch { /* Optional browser preference. */ }
}

export function useTaskCenterPosition(panel: React.RefObject<HTMLDivElement>, open: boolean) {
  const [position, setPosition] = React.useState(initialPosition);
  const current = React.useRef(position);
  const [view, setView] = React.useState(viewport);
  const [dragging, setDragging] = React.useState(false);
  const [panelHeight, setPanelHeight] = React.useState(0);
  const suppressClick = React.useRef(false);
  const gesture = React.useRef<{
    id: number; x: number; y: number; origin: Point; moved: boolean; target: HTMLButtonElement;
  } | null>(null);

  React.useEffect(() => {
    const finish = () => {
      const active = gesture.current;
      if (!active) return;
      gesture.current = null;
      if (active.moved) { suppressClick.current = true; remember(current.current); }
      setDragging(false);
      try { active.target.releasePointerCapture?.(active.id); } catch { /* The browser may have released it. */ }
    };
    const move = (event: PointerEvent) => {
      const active = gesture.current;
      if (!active || active.id !== event.pointerId) return;
      const dx = event.clientX - active.x, dy = event.clientY - active.y;
      if (!active.moved && Math.hypot(dx, dy) < 6) return;
      active.moved = true;
      setDragging(true);
      current.current = bounded({ x: active.origin.x + dx, y: active.origin.y + dy });
      setPosition(current.current);
    };
    const end = (event: PointerEvent) => { if (gesture.current?.id === event.pointerId) finish(); };
    const resize = () => {
      finish();
      const nextView = viewport();
      setView(nextView);
      current.current = bounded(current.current, nextView);
      setPosition(current.current);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', end);
    window.addEventListener('pointercancel', end);
    window.addEventListener('lostpointercapture', end);
    window.addEventListener('blur', finish);
    window.addEventListener('resize', resize);
    window.visualViewport?.addEventListener('resize', resize);
    window.visualViewport?.addEventListener('scroll', resize);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', end);
      window.removeEventListener('pointercancel', end);
      window.removeEventListener('lostpointercapture', end);
      window.removeEventListener('blur', finish);
      window.removeEventListener('resize', resize);
      window.visualViewport?.removeEventListener('resize', resize);
      window.visualViewport?.removeEventListener('scroll', resize);
      const active = gesture.current;
      gesture.current = null;
      if (active?.moved) remember(current.current);
      try { active?.target.releasePointerCapture?.(active.id); } catch { /* Capture already ended. */ }
    };
  }, []);

  React.useLayoutEffect(() => {
    if (!open || !panel.current) return;
    const element = panel.current;
    const measure = () => setPanelHeight(element.getBoundingClientRect().height);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [open, panel]);

  const below = position.y + SIZE / 2 < view.y + view.height / 2;
  const panelWidth = Math.min(380, Math.max(0, view.width - MARGIN * 2));
  const maxHeight = Math.max(0, Math.min(560, below
    ? view.y + view.height - MARGIN - position.y - SIZE - GAP
    : position.y - view.y - MARGIN - GAP));
  const panelStyle: React.CSSProperties = {
    left: clamp(position.x + SIZE - panelWidth, view.x + MARGIN, view.x + view.width - panelWidth - MARGIN),
    top: below ? position.y + SIZE + GAP : position.y - GAP - Math.min(panelHeight, maxHeight),
    width: panelWidth, maxHeight,
  };
  return {
    dragging, ballStyle: { left: position.x, top: position.y }, panelStyle,
    onPointerDown: (event: React.PointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0 || event.isPrimary === false || gesture.current) return;
      suppressClick.current = false;
      gesture.current = { id: event.pointerId, x: event.clientX, y: event.clientY,
        origin: current.current, moved: false, target: event.currentTarget };
      try { event.currentTarget.setPointerCapture?.(event.pointerId); } catch { /* Window listeners still finish the drag. */ }
    },
    onClickCapture: (event: React.MouseEvent<HTMLButtonElement>) => {
      if (suppressClick.current && event.detail !== 0) {
        suppressClick.current = false;
        event.preventDefault();
        event.stopPropagation();
      }
    },
  };
}
