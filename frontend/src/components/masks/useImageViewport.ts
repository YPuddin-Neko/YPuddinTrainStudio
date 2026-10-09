import React from 'react';

interface Options {
  viewport: React.RefObject<HTMLDivElement>;
  canvas: React.RefObject<HTMLCanvasElement>;
  preview: React.RefObject<HTMLDivElement>;
  zoom: number;
  setZoom: React.Dispatch<React.SetStateAction<number>>;
  enabled: boolean;
  panTool: boolean;
  isDrawing: () => boolean;
  finishStroke: () => void;
}

export function useImageViewport({ viewport, canvas, preview, zoom, setZoom, enabled, panTool, isDrawing, finishStroke }: Options) {
  const [panning, setPanning] = React.useState(false);
  const zoomValue = React.useRef(zoom);
  const drag = React.useRef<{ id: number; buttons: number; x: number; y: number; left: number; top: number } | null>(null);
  const anchor = React.useRef<{ x: number; y: number; clientX: number; clientY: number } | null>(null);
  const endPan = React.useCallback(() => {
    const active = drag.current;
    drag.current = null;
    if (active && viewport.current?.hasPointerCapture(active.id)) viewport.current.releasePointerCapture(active.id);
    setPanning(false);
  }, [viewport]);

  React.useEffect(() => {
    const host = viewport.current;
    if (!host || !enabled) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      if (isDrawing() || drag.current || !canvas.current || !event.deltaY) return;
      const rect = canvas.current.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? host.clientHeight : 1);
      const next = Math.max(.25, Math.min(8, zoomValue.current * Math.exp(-Math.max(-240, Math.min(240, delta)) * .002)));
      if (next === zoomValue.current) return;
      anchor.current = { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height, clientX: event.clientX, clientY: event.clientY };
      preview.current?.style.setProperty('visibility', 'hidden');
      zoomValue.current = next;
      setZoom(next);
    };
    host.addEventListener('wheel', wheel, { passive: false });
    return () => host.removeEventListener('wheel', wheel);
  }, [viewport, canvas, preview, setZoom, enabled, isDrawing]);

  React.useLayoutEffect(() => {
    zoomValue.current = zoom;
    const at = anchor.current, host = viewport.current, element = canvas.current;
    anchor.current = null;
    if (!at || !host || !element) return;
    const rect = element.getBoundingClientRect();
    host.scrollTo({ left:host.scrollLeft + rect.left + at.x * rect.width - at.clientX, top:host.scrollTop + rect.top + at.y * rect.height - at.clientY, behavior:'instant' });
  }, [zoom, viewport, canvas]);

  React.useEffect(() => {
    window.addEventListener('blur', endPan);
    return () => window.removeEventListener('blur', endPan);
  }, [endPan]);
  React.useEffect(() => { if (!enabled) endPan(); }, [enabled, endPan]);

  const onPointerDownCapture = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!enabled || (event.button !== 1 && !(panTool && event.button === 0))) return;
    event.preventDefault(); event.stopPropagation();
    if (drag.current) return;
    finishStroke();
    const host = event.currentTarget;
    drag.current = { id: event.pointerId, buttons: event.button === 1 ? 4 : 1, x: event.clientX, y: event.clientY, left: host.scrollLeft, top: host.scrollTop };
    host.setPointerCapture(event.pointerId);
    preview.current?.style.setProperty('visibility', 'hidden');
    setPanning(true);
  };
  const onPointerMoveCapture = (event: React.PointerEvent<HTMLDivElement>) => {
    const active = drag.current;
    if (!active || event.pointerId !== active.id) return;
    event.preventDefault(); event.stopPropagation();
    if (!(event.buttons & active.buttons)) { endPan(); return; }
    event.currentTarget.scrollLeft = active.left - (event.clientX - active.x);
    event.currentTarget.scrollTop = active.top - (event.clientY - active.y);
  };
  const onPointerUpCapture = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!drag.current || event.pointerId !== drag.current.id) return;
    event.preventDefault(); event.stopPropagation();
    if (!(event.buttons & drag.current.buttons)) endPan();
  };
  const onPointerCancel = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerId === drag.current?.id) endPan();
  };
  return { panning, viewportEvents: {
    onPointerDownCapture, onPointerMoveCapture, onPointerUpCapture,
    onPointerCancel, onLostPointerCapture: onPointerCancel,
    onAuxClick: (event: React.MouseEvent<HTMLDivElement>) => { if (event.button === 1) event.preventDefault(); },
  } };
}
