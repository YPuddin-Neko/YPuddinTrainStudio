import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { handlers } from '../mocks/handlers';
import { useDatasetImages } from '../../../frontend/src/api/hooks/useDatasetImages';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('useDatasetImages (A2: 分页 / 过滤 / 多选)', () => {
  it('loads first page with total from MSW mock', async () => {
    const { result } = renderHook(() => useDatasetImages('ds_01', 60));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.total).toBe(240);
    expect(result.current.items.length).toBe(60);
    expect(result.current.hasMore).toBe(true);
  });

  it('loadMore appends next page', async () => {
    const { result } = renderHook(() => useDatasetImages('ds_01', 60));
    await waitFor(() => expect(result.current.items.length).toBe(60));

    await act(async () => {
      result.current.loadMore();
    });
    await waitFor(() => expect(result.current.items.length).toBe(120));
    expect(result.current.page).toBe(2);
  });

  it('q filter narrows results and resets pagination', async () => {
    const { result } = renderHook(() => useDatasetImages('ds_01', 60));
    await waitFor(() => expect(result.current.total).toBe(240));

    await act(async () => {
      result.current.setQ('sword');
    });
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.total).toBeLessThan(240);
    expect(result.current.total).toBeGreaterThan(0);
    expect(
      result.current.items.every((i) => i.caption.toLowerCase().includes('sword'))
    ).toBe(true);
  });

  it('selection toggle / all / clear', async () => {
    const { result } = renderHook(() => useDatasetImages('ds_01', 60));
    await waitFor(() => expect(result.current.items.length).toBe(60));

    act(() => result.current.toggleSelect('hash_00001'));
    expect(result.current.selected.has('hash_00001')).toBe(true);

    act(() => result.current.toggleSelect('hash_00001'));
    expect(result.current.selected.has('hash_00001')).toBe(false);

    act(() => result.current.selectAll());
    expect(result.current.selected.size).toBe(60);

    act(() => result.current.clearSelection());
    expect(result.current.selected.size).toBe(0);
  });

  it('updateCaption patches local item', async () => {
    const { result } = renderHook(() => useDatasetImages('ds_01', 60));
    await waitFor(() => expect(result.current.items.length).toBe(60));

    const first = result.current.items[0];
    act(() => result.current.updateCaption(first.hash, 'new_caption, edited', first.rel_path));
    expect(result.current.items[0].caption).toBe('new_caption, edited');
  });

  it('updates only the saved path when identical images have different captions', async () => {
    server.use(http.get('/api/datasets/duplicates/images', () => HttpResponse.json({
      total: 2, page: 1, page_size: 60,
      items: [
        { hash: 'same-content', rel_path: 'first/photo.png', caption: 'first original', caption_tags: 'first original', caption_format: 'txt' },
        { hash: 'same-content', rel_path: 'second/photo.png', caption: 'second original', caption_tags: 'second original', caption_format: 'txt' },
      ],
    })));
    const { result } = renderHook(() => useDatasetImages('duplicates'));
    await waitFor(() => expect(result.current.items).toHaveLength(2));
    const untouched = result.current.items[1];
    act(() => result.current.updateCaption('same-content', 'edited first', 'first/photo.png'));
    expect(result.current.items[0].caption_tags).toBe('edited first');
    expect(result.current.items[1]).toBe(untouched);
    expect(result.current.items[1].caption_tags).toBe('second original');
    act(() => result.current.updateCaption('same-content', 'wrong path', 'missing/photo.png'));
    expect(result.current.items.map(item => item.caption_tags)).toEqual(['edited first', 'second original']);
  });
});
