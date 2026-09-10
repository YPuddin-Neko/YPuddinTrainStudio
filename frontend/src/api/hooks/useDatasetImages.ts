import React from 'react';
import { apiClient } from '../client';
import { DatasetImage, DatasetImagesPage } from '../types';

/**
 * 数据集图片分页 + 搜索 + 多选 hook。
 * - page/pageSize/q 变化时重新请求第一页或对应页
 * - 支持追加加载（loadMore）与全量刷新（refresh）
 * - 本地更新某张图 caption（caption 编辑保存后）
 */
export function useDatasetImages(datasetId: string | undefined, pageSize = 60) {
  const [items, setItems] = React.useState<DatasetImage[]>([]);
  const [total, setTotal] = React.useState(0);
  const [page, setPage] = React.useState(1);
  const [q, setQ] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [selected, setSelected] = React.useState<Set<string>>(new Set());

  const fetchPage = React.useCallback(
    async (p: number, query: string, append: boolean) => {
      if (!datasetId) return;
      setLoading(true);
      setError(null);
      try {
        const resp = await apiClient.get<DatasetImagesPage>(`/datasets/${datasetId}/images`, {
          params: { page: p, page_size: pageSize, q: query || undefined },
        });
        setTotal(resp.total);
        setPage(resp.page);
        setItems((prev) => (append ? [...prev, ...resp.items] : resp.items));
      } catch (e: any) {
        setError(e?.message || 'failed to load images');
      } finally {
        setLoading(false);
      }
    },
    [datasetId, pageSize]
  );

  // datasetId 或 q 变化：重置并加载第一页
  React.useEffect(() => {
    setItems([]);
    setPage(1);
    setSelected(new Set());
    fetchPage(1, q, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, q]);

  const loadMore = React.useCallback(() => {
    if (loading) return;
    if (items.length >= total) return;
    fetchPage(page + 1, q, true);
  }, [loading, items.length, total, page, q, fetchPage]);

  const refresh = React.useCallback(() => {
    setItems([]);
    fetchPage(1, q, false);
  }, [fetchPage, q]);

  const toggleSelect = React.useCallback((hash: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(hash)) next.delete(hash);
      else next.add(hash);
      return next;
    });
  }, []);

  const clearSelection = React.useCallback(() => setSelected(new Set()), []);

  const selectAll = React.useCallback(() => {
    setSelected(new Set(items.map((i) => i.hash)));
  }, [items]);

  const updateCaption = React.useCallback((hash: string, caption: string) => {
    setItems((prev) => prev.map((i) => (i.hash === hash ? { ...i, caption } : i)));
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
    clearSelection,
    selectAll,
    loadMore,
    refresh,
    updateCaption,
    hasMore: items.length < total,
  };
}
