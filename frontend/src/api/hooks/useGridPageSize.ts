import React from 'react';

export const fullRowPageSize = (target: number, columns: number) => Math.ceil(target / Math.max(1, columns)) * Math.max(1, columns);

/** Keep a page's image count in whole rows at the grid's actual CSS column count. */
export function useGridPageSize(target: number) {
  const [node, setNode] = React.useState<HTMLElement | null>(null);
  const [columns, setColumns] = React.useState(1);
  React.useLayoutEffect(() => {
    if (!node) return;
    const measure = () => {
      const tracks = window.getComputedStyle(node).gridTemplateColumns.trim();
      if (!tracks || tracks === 'none' || tracks.includes('repeat(')) return;
      setColumns(Math.max(1, tracks.split(/\s+/).length));
    };
    measure();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    observer?.observe(node);
    window.addEventListener('resize', measure);
    return () => { observer?.disconnect(); window.removeEventListener('resize', measure); };
  }, [node]);
  return { gridRef: setNode, columns, pageSize: fullRowPageSize(target, columns) };
}
