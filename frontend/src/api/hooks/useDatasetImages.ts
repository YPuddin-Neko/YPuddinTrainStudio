import React from 'react';
import { apiClient } from '../client';
import { DatasetImage, DatasetImagesPage } from '../types';
import type { ImageSort } from '../../components/datasets/ImageSortSelect';
import { fullRowPageSize } from './useGridPageSize';

/**
 * 数据集图片分页 + 搜索 + 多选 hook。
 * - page/pageSize/q 变化时重新请求第一页或对应页
 * - 支持追加加载（loadMore）与全量刷新（refresh）
 * - 本地更新某张图 caption（caption 编辑保存后）
 */
export function useDatasetImages(datasetId: string | undefined, targetPageSize = 60, membership: 'all' | 'training' | 'unused' = 'all') {
  const [columns, setColumns] = React.useState(1);
  const pageSize = fullRowPageSize(targetPageSize, columns);
  const [items, setItems] = React.useState<DatasetImage[]>([]);
  const [total, setTotal] = React.useState(0);
  const [page, setPage] = React.useState(1);
  const [q, setQ] = React.useState('');
  const [sort, setSort] = React.useState<ImageSort>('filename');
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const request = React.useRef<AbortController | null>(null);
  const [selected, setSelected] = React.useState<Set<string>>(new Set());
  const loaded = React.useRef({ items, total });
  const previousQuery = React.useRef<{ scope: string; pageSize: number } | null>(null);
  const replacementPending = React.useRef(false);
  React.useEffect(() => { loaded.current = { items, total }; }, [items, total]);

  const fetchPage = React.useCallback(
    async (p: number, query: string, append: boolean) => {
      if (!datasetId) return;
      if (!append) replacementPending.current = true;
      request.current?.abort();
      const controller = new AbortController(); request.current = controller;
      setLoading(true);
      setError(null);
      try {
        const resp = await apiClient.get<DatasetImagesPage>(`/datasets/${datasetId}/images`, {
          signal: controller.signal,
          params: { page: p, page_size: pageSize, q: query || undefined, membership, sort },
        });
        if (controller.signal.aborted) return;
        if (!append) replacementPending.current = false;
        setTotal(resp.total);
        setPage(resp.page);
        setItems(prev => {
          if (!append) return resp.items;
          const existing = new Set(prev.map(item => item.rel_path));
          return [...prev, ...resp.items.filter(item => !existing.has(item.rel_path))];
        });
      } catch (e: any) {
        if (!controller.signal.aborted) setError(e?.message || 'failed to load images');
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    },
    [datasetId, pageSize, membership, sort]
  );

  // Resizing fills the last row without clearing images already in view.
  React.useEffect(() => {
    const scope = JSON.stringify([datasetId, q, membership, sort]);
    const previous = previousQuery.current;
    previousQuery.current = { scope, pageSize };
    const count = loaded.current.items.length;
    const resizing = previous?.scope === scope && previous.pageSize !== pageSize;
    if (resizing && count && !replacementPending.current) {
      const lastPage = Math.ceil(count / pageSize);
      // Only a partly loaded last row needs the rest of its page; a fully loaded list stays as it is.
      if (count % pageSize && count < loaded.current.total) void fetchPage(lastPage, q, true);
      else { setPage(lastPage); setLoading(false); }
    } else {
      if (!resizing) setItems([]);
      setPage(1);
      void fetchPage(1, q, false);
    }
    return () => request.current?.abort();
  }, [datasetId, q, membership, pageSize, sort, fetchPage]);

  React.useEffect(() => { setSelected(new Set()); }, [datasetId, q, membership, sort]);

  const loadMore = React.useCallback(() => {
    if (loading) return;
    if (replacementPending.current) {
      void fetchPage(1, q, false);
      return;
    }
    if (items.length >= total) return;
    fetchPage(Math.floor(items.length / pageSize) + 1, q, true);
  }, [loading, items.length, total, pageSize, q, fetchPage]);

  const refresh = React.useCallback(() => {
    fetchPage(1, q, false);
  }, [fetchPage, q]);

  const toggleSelect = React.useCallback((relPath: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(relPath)) next.delete(relPath);
      else next.add(relPath);
      return next;
    });
  }, []);

  const selectRange = React.useCallback((paths: string[]) => {
    setSelected(previous => new Set([...previous, ...paths]));
  }, []);

  const clearSelection = React.useCallback(() => setSelected(new Set()), []);

  const selectAll = React.useCallback(() => {
    setSelected(new Set(items.map((i) => i.rel_path)));
  }, [items]);

  const updateCaption = React.useCallback((hash: string, caption: string, relPath: string) => {
    setItems((prev) => prev.map((i) => (i.hash === hash && i.rel_path === relPath ? { ...i, caption_tags: caption, caption: i.caption_description ? `${caption}${caption ? ". " : ""}${i.caption_description}` : caption } : i)));
  }, []);

  return {
    items,
    total,
    page,
    pageSize,
    setColumns,
    q,
    setQ,
    sort,
    setSort,
    loading,
    error,
    selected,
    toggleSelect,
    selectRange,
    clearSelection,
    selectAll,
    loadMore,
    refresh,
    updateCaption,
    hasMore: items.length < total,
  };
}
