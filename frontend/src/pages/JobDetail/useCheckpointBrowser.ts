import React from 'react';
import type { JobCheckpoint } from '../../api/types';

const PAGE_SIZES = [10, 20, 50];
export function useCheckpointBrowser(items: JobCheckpoint[], storageKey: string, canSelect: (item: JobCheckpoint) => boolean = () => true) {
  const [pageSize, setPageSize] = React.useState(() => {
    try { const saved = Number(localStorage.getItem(storageKey)); return PAGE_SIZES.includes(saved) ? saved : 10; } catch { return 10; }
  });
  const [page, setPage] = React.useState(1);
  const [managing, setManaging] = React.useState(false);
  const [paths, setPaths] = React.useState<Set<string>>(() => new Set());
  const pages = Math.max(1, Math.ceil(items.length / pageSize));
  const currentPage = Math.min(page, pages);
  React.useEffect(() => { setPage(value => Math.min(value, pages)); }, [pages]);
  const visible = items.slice((currentPage - 1) * pageSize, currentPage * pageSize);
  const selected = items.filter(item => paths.has(item.path) && canSelect(item));
  const selectable = visible.filter(canSelect);
  const allSelected = selectable.length > 0 && selectable.every(item => paths.has(item.path));
  const someSelected = selectable.some(item => paths.has(item.path));
  const toggle = (item: JobCheckpoint) => {
    if (!canSelect(item)) return;
    setPaths(previous => { const next = new Set(previous); if (next.has(item.path)) next.delete(item.path); else next.add(item.path); return next; });
  };
  const togglePage = () => setPaths(previous => {
    const next = new Set(previous);
    for (const item of selectable) { if (allSelected) next.delete(item.path); else next.add(item.path); }
    return next;
  });
  const changeSize = (value: string) => {
    const next = Number(value); if (!PAGE_SIZES.includes(next)) return;
    setPageSize(next); setPage(1);
    try { localStorage.setItem(storageKey, value); } catch { /* Storage may be unavailable. */ }
  };
  return { pageSize, page: currentPage, pages, setPage, changeSize, visible, managing, selected, selectable, allSelected, someSelected,
    toggle, togglePage, isSelected: (item: JobCheckpoint) => paths.has(item.path) && canSelect(item),
    toggleManaging: () => { setManaging(value => !value); setPaths(new Set()); },
    resetFilter: () => { setPage(1); setPaths(new Set()); }, clear: () => setPaths(new Set()) };
}
