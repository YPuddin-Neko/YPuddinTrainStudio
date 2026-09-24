import React from 'react';
import { apiClient } from '../client';
import { DatasetImage, DatasetImagesPage } from '../types';

/**
 * 数据集图片分页 + 搜索 + 多选 hook。
 * - page/pageSize/q 变化时重新请求第一页或对应页
 * - 支持追加加载（loadMore）与全量刷新（refresh）
 * - 本地更新某张图 caption（caption 编辑保存后）
 */
export function useDatasetImages(datasetId: string | undefined, pageSize = 60, membership: 'all' | 'training' | 'unused' = 'all') {
  const [items, setItems] = React.useState<DatasetImage[]>([]);
  const [total, setTotal] = React.useState(0);
  const [page, setPage] = React.useState(1);
  const [q, setQ] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const request = React.useRef<AbortController | null>(null);
  const [selected, setSelected] = React.useState<Set<string>>(new Set());

  const fetchPage = React.useCallback(
    async (p: number, query: string, append: boolean) => {
      if (!datasetId) return;
      request.current?.abort();
      const controller = new AbortController(); request.current = controller;
      setLoading(true);
      setError(null);
      try {
        const resp = await apiClient.get<DatasetImagesPage>(`/datasets/${datasetId}/images`, {
          signal: controller.signal,
          params: { page: p, page_size: pageSize, q: query || undefined, membership },
        });
        if (controller.signal.aborted) return;
        setTotal(resp.total);
        setPage(resp.page);
        setItems((prev) => (append ? [...prev, ...resp.items] : resp.items));
      } catch (e: any) {
        if (!controller.signal.aborted) setError(e?.message || 'failed to load images');
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    },
    [datasetId, pageSize, membership]
  );

  // datasetId 或 q 变化：重置并加载第一页
  React.useEffect(() => {
    setItems([]);
    setPage(1);
    setSelected(new Set());
    fetchPage(1, q, false);
    return () => request.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, q, membership]);

  const loadMore = React.useCallback(() => {
    if (loading) return;
    if (items.length >= total) return;
    fetchPage(page + 1, q, true);
  }, [loading, items.length, total, page, q, fetchPage]);

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
    q,
    setQ,
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
