import React from 'react';

export const fullRowPageSize = (target: number, columns: number) => Math.ceil(target / Math.max(1, columns)) * Math.max(1, columns);

/**
 * Keep a page's image count in whole rows at the grid's actual CSS column count. `ready` turns true once the grid
 * has been looked at, so the first request can already use its page size.
 */
export function useGridPageSize(target: number) {
  const [node, setNode] = React.useState<HTMLElement | null>(null);
  const [columns, setColumns] = React.useState(1);
  const [ready, setReady] = React.useState(false);
  React.useLayoutEffect(() => {
    if (!node) return;
    const measure = () => {
      const tracks = window.getComputedStyle(node).gridTemplateColumns.trim();
      if (tracks && tracks !== 'none' && !tracks.includes('repeat(')) setColumns(Math.max(1, tracks.split(/\s+/).length));
      // A grid that cannot be measured yet (hidden, or without layout) keeps one column rather than blocking requests.
      setReady(true);
    };
    measure();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    observer?.observe(node);
    window.addEventListener('resize', measure);
    return () => { observer?.disconnect(); window.removeEventListener('resize', measure); };
  }, [node]);
  return { gridRef: setNode, columns, pageSize: fullRowPageSize(target, columns), ready };
}
