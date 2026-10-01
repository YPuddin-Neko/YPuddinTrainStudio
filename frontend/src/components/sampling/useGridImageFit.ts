import React from 'react';

/** Keep a whole image visible below the comparison's column labels. Additional rows can scroll. */
export function useGridImageFit(layoutKey: string) {
  const viewport = React.useRef<HTMLDivElement>(null);
  React.useLayoutEffect(() => {
    const box = viewport.current;
    const table = box?.querySelector('table');
    const workspace = box?.closest('.xyz-workspace');
    if (!box || !table || !workspace) return;
    const measure = () => {
      const padding = getComputedStyle(box);
      const tableStyle = getComputedStyle(table);
      const cell = table.querySelector('td');
      const cellStyle = cell ? getComputedStyle(cell) : null;
      const number = (value: string | undefined) => Number.parseFloat(value || '') || 0;
      // Stacked layouts follow the page; desktop panels have a bounded flex viewport.
      const height = workspace.getBoundingClientRect().width > 850 ? box.clientHeight : window.innerHeight * 0.75;
      if (height <= 0) return;
      const labels = (table.caption?.getBoundingClientRect().height || 0) + (table.tHead?.getBoundingClientRect().height || 0);
      const spacing = number(tableStyle.borderSpacing.split(' ').at(-1)) * 2;
      const available = height - labels - spacing - number(padding.paddingTop) - number(padding.paddingBottom)
        - number(cellStyle?.paddingTop) - number(cellStyle?.paddingBottom);
      box.style.setProperty('--xyz-cell-max-height', `${Math.max(1, Math.floor(available))}px`);
    };
    measure();
    const observer = new ResizeObserver(measure);
    [box, workspace, table.caption, table.tHead].forEach(element => { if (element) observer.observe(element); });
    window.addEventListener('resize', measure);
    return () => { observer.disconnect(); window.removeEventListener('resize', measure); };
  }, [layoutKey]);
  return viewport;
}
