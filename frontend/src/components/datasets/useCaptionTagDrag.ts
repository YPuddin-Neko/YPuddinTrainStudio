import { useCallback, useEffect, useRef, useState, type PointerEvent } from 'react';

interface Position { field: string; index: number; }
interface Source extends Position { value: string; }
interface DragState { source: Source; target: Position | null; }
interface Gesture {
  source: Source; x: number; y: number; startX: number; startY: number;
  id: number; element: HTMLElement; active: boolean; target: Position | null;
}
export interface CaptionTagDragBinding {
  field: string;
  dragging: number | null;
  insertionIndex: number | null;
  onPointerDown: (event: PointerEvent<HTMLElement>, index: number, value: string) => void;
}

export function useCaptionTagDrag({ scope, revision, disabled, onDrop }: {
  scope: string; revision: unknown; disabled: boolean;
  onDrop: (source: Source, target: Position) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const gesture = useRef<Gesture | null>(null);
  const frame = useRef<number | null>(null);
  const [state, setState] = useState<DragState | null>(null);
  const release = useCallback(() => {
    const current = gesture.current;
    gesture.current = null;
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
    if (current?.element.hasPointerCapture(current.id)) current.element.releasePointerCapture(current.id);
  }, []);
  const cancel = useCallback(() => { release(); setState(null); }, [release]);

  useEffect(() => {
    cancel();
    const keydown = (event: KeyboardEvent) => { if (event.key === 'Escape') cancel(); };
    window.addEventListener('keydown', keydown);
    window.addEventListener('blur', cancel);
    window.addEventListener('pointerup', cancel);
    window.addEventListener('pointercancel', cancel);
    return () => {
      release();
      window.removeEventListener('keydown', keydown);
      window.removeEventListener('blur', cancel);
      window.removeEventListener('pointerup', cancel);
      window.removeEventListener('pointercancel', cancel);
    };
  }, [scope, revision, disabled, cancel, release]);

  const locate = (current: Gesture) => {
    const root = ref.current;
    const element = document.elementFromPoint(current.x, current.y);
    const field = element?.closest<HTMLElement>('[data-caption-tag-field]');
    let target: Position | null = null;
    if (root && field && root.contains(field) && !field.closest('fieldset:disabled')) {
      const chips = [...field.querySelectorAll<HTMLElement>('[data-caption-tag-index]')];
      let index = chips.length;
      for (const chip of chips) {
        const rect = chip.getBoundingClientRect();
        if (current.y < rect.top || current.y <= rect.bottom && current.x < rect.left + rect.width / 2) {
          index = Number(chip.dataset.captionTagIndex); break;
        }
      }
      target = { field: field.dataset.captionTagField!, index };
    }
    current.target = target;
    setState(previous => previous?.source === current.source && previous.target?.field === target?.field && previous.target?.index === target?.index
      ? previous : { source: current.source, target });
  };

  const scroll = () => {
    const current = gesture.current, root = ref.current;
    if (!current?.active || !root) return;
    const rect = root.getBoundingClientRect();
    if (current.x >= rect.left && current.x <= rect.right && current.y >= rect.top && current.y <= rect.bottom) {
      const edge = Math.min(36, rect.height / 4);
      const delta = current.y < rect.top + edge ? -8 : current.y > rect.bottom - edge ? 8 : 0;
      if (delta) { root.scrollTop += delta; locate(current); }
    }
    frame.current = requestAnimationFrame(scroll);
  };
  const move = (event: PointerEvent) => {
    const current = gesture.current;
    if (!current || event.pointerId !== current.id) return;
    current.x = event.clientX; current.y = event.clientY;
    if (!current.active && Math.abs(current.x - current.startX) + Math.abs(current.y - current.startY) > 5) {
      current.active = true;
      current.element.setPointerCapture(current.id);
      frame.current = requestAnimationFrame(scroll);
    }
    if (current.active) { event.preventDefault(); locate(current); }
  };
  const end = (event: PointerEvent) => {
    const current = gesture.current;
    if (!current || event.pointerId !== current.id) return;
    if (current.active) {
      current.x = event.clientX; current.y = event.clientY;
      locate(current);
    }
    cancel();
    if (current.active && current.target && !disabled) onDrop(current.source, current.target);
  };
  const bindField = (field: string): CaptionTagDragBinding => ({
    field,
    dragging: state?.source.field === field ? state.source.index : null,
    insertionIndex: state?.target?.field === field ? state.target.index : null,
    onPointerDown: (event, index, value) => {
      if (disabled || gesture.current || event.button !== 0 || event.detail > 1 || (event.target as HTMLElement).closest('button, input, textarea, fieldset:disabled')) return;
      // 移动超过阈值才捕获指针：按下就捕获会让 WebView2 的双击编辑失效
      gesture.current = { source: { field, index, value }, id: event.pointerId, element: event.currentTarget,
        x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, active: false, target: null };
    },
  });
  return { bindField, editorProps: { ref, onPointerMove: move, onPointerUp: end, onPointerCancel: cancel,
    onLostPointerCapture: cancel, onScroll: () => { if (gesture.current?.active) locate(gesture.current); } } };
}
