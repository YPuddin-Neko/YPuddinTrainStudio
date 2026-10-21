import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { Loader2 } from 'lucide-react';
import { prefersReducedMotion } from '../../utils/motion';

type Props = { anchor: RefObject<HTMLElement>; loading: boolean; label: string };

export default function DatasetLoadingPill({ anchor, loading, label }: Props) {
  const [host, setHost] = useState<HTMLElement | null>(null);
  const [phase, setPhase] = useState<'hidden' | 'visible' | 'exiting'>('hidden');
  const [bottom, setBottom] = useState(20);
  const shownAt = useRef(0);
  const pill = useRef<HTMLDivElement>(null);
  useEffect(() => { setHost(anchor.current?.closest<HTMLElement>('.app-main') ?? null); }, [anchor]);

  useEffect(() => {
    let timer: number | undefined;
    if (loading) {
      if (phase === 'hidden') timer = window.setTimeout(() => {
        shownAt.current = performance.now();
        setPhase('visible');
      }, 180);
      else if (phase === 'exiting') {
        shownAt.current = performance.now();
        setPhase('visible');
      }
    } else if (phase === 'visible') {
      timer = window.setTimeout(() => setPhase('exiting'), Math.max(0, 450 - (performance.now() - shownAt.current)));
    } else if (phase === 'exiting') {
      timer = window.setTimeout(() => setPhase('hidden'), prefersReducedMotion() ? 0 : 220);
    }
    return () => window.clearTimeout(timer);
  }, [loading, phase]);

  const present = phase !== 'hidden';
  useLayoutEffect(() => {
    if (!host || !present || !pill.current) return;
    // Task-center surfaces can be dragged over the bottom of this workspace.
    const measure = () => {
      const bounds = host.getBoundingClientRect();
      const width = pill.current?.offsetWidth ?? 0;
      const left = bounds.left + (bounds.width - width) / 2;
      let inset = 20;
      const obstacles = Array.from(host.querySelectorAll<HTMLElement>('.task-center-ball, .task-center-panel'))
        .map(element => element.getBoundingClientRect()).sort((a, b) => b.top - a.top);
      for (const box of obstacles) {
        if (box.width && box.height && box.right > left && box.left < left + width
          && box.bottom > bounds.bottom - inset - 34 && box.top < bounds.bottom - inset) {
          inset = Math.max(inset, bounds.bottom - box.top + 12);
        }
      }
      setBottom(Math.min(inset, Math.max(20, bounds.height - 54)));
    };
    const resize = new ResizeObserver(measure);
    const observe = () => {
      resize.disconnect();
      resize.observe(host);
      if (pill.current) resize.observe(pill.current);
      host.querySelectorAll('.task-center-ball, .task-center-panel').forEach(element => resize.observe(element));
      measure();
    };
    const mutation = new MutationObserver(observe);
    const taskCenter = host.querySelector('.task-center');
    if (taskCenter) mutation.observe(taskCenter, { subtree: true, childList: true, attributes: true, attributeFilter: ['style'] });
    observe();
    return () => { resize.disconnect(); mutation.disconnect(); };
  }, [host, present]);

  if (!host || !present) return null;
  return createPortal(<div className="dataset-loading-overlay" style={{ bottom }}>
    <div ref={pill} className="dataset-loading-pill" data-phase={phase} role="status">
      <Loader2 size={14} className="animate-spin" aria-hidden="true"/><span>{label}</span>
    </div>
  </div>, host);
}
