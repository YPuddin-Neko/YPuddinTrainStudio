import { useLayoutEffect, useRef } from 'react';

/** Keep stacked workspace toolbars aligned when content, language or viewport changes. */
export function useWorkspaceHeight(variable: string) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const node = ref.current;
    const parent = node?.parentElement;
    if (!node || !parent) return;
    const update = () => { const height = node.getBoundingClientRect().height; if (height) parent.style.setProperty(variable, `${height}px`); };
    update();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update);
    observer?.observe(node);
    window.addEventListener('resize', update);
    return () => { observer?.disconnect(); window.removeEventListener('resize', update); parent.style.removeProperty(variable); };
  }, [variable]);
  return ref;
}
