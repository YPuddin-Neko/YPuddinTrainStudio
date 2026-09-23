import { act, render, renderHook, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../../../frontend/src/api/client';
import { ApiError } from '../../../frontend/src/api/types';
import { useDatasetImportProgress, type DatasetImportProgressSnapshot } from '../../../frontend/src/utils/useDatasetImportProgress';
import DatasetImportProgress from '../../../frontend/src/components/datasets/DatasetImportProgress';
import i18n from '../../../frontend/src/i18n';

function snapshot(id: string, changes: Partial<DatasetImportProgressSnapshot> = {}): DatasetImportProgressSnapshot {
  return { id, phase: 'receiving', bytes_done: 512, bytes_total: 1024, files_done: 0, files_total: null, elapsed_seconds: 1,
    phase_elapsed_seconds: 1, bytes_per_second: 512, eta_seconds: 1, error: null, ...changes };
}
const idFrom = (url: string) => url.split('/').pop()!;
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); vi.useFakeTimers(); });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('import progress polling and phase display', () => {
  it.each([false, true])('uses random bytes when randomUUID is unavailable or throws (%s)', async throws => {
    const randomBytes = globalThis.crypto.getRandomValues.bind(globalThis.crypto);
    const getRandomValues = vi.fn(randomBytes);
    vi.stubGlobal('crypto', { getRandomValues, ...(throws ? { randomUUID: () => { throw new Error('Unavailable'); } } : {}) });
    const get = vi.spyOn(apiClient, 'get').mockImplementation(async url => snapshot(idFrom(url)));
    const { result } = renderHook(() => useDatasetImportProgress('p_import'));
    let id: string | undefined;
    await act(async () => { id = result.current.start('upload'); });
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(getRandomValues).toHaveBeenCalledOnce();
    expect(get.mock.calls[0][0]).toContain(`/import-progress/${id}`);
    expect(result.current.operation).toMatchObject({ id, state: 'active', unavailable: false });
  });

  it.each([undefined, { getRandomValues: () => { throw new Error('Unavailable'); } }])('keeps local waiting and completion when random APIs are unavailable (%s)', async crypto => {
    vi.stubGlobal('crypto', crypto);
    const get = vi.spyOn(apiClient, 'get');
    const { result } = renderHook(() => useDatasetImportProgress('p_import'));
    let id: string | undefined = 'not-started';
    await act(async () => { id = result.current.start('path'); await vi.advanceTimersByTimeAsync(1000); });
    expect(id).toBeUndefined();
    expect(get).not.toHaveBeenCalled();
    expect(result.current.operation).toMatchObject({ mode: 'path', state: 'active', unavailable: true, snapshot: null, elapsed: 1 });
    render(<DatasetImportProgress operation={result.current.operation!}/>);
    expect(screen.getByText('暂时无法读取进度，正在等待导入结果')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
    await act(async () => { result.current.finish('completed'); });
    expect(result.current.operation?.state).toBe('completed');
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(result.current.operation?.elapsed).toBe(1);
  });

  it('tracks each phase independently and keeps actual POST completion authoritative', async () => {
    const get = vi.spyOn(apiClient, 'get').mockImplementation(async url => snapshot(idFrom(url), get.mock.calls.length > 1
      ? { phase: 'copying', bytes_done: 25, bytes_total: 100, elapsed_seconds: 2 } : {}));
    const hook = renderHook(() => useDatasetImportProgress('p_import'));
    await act(async () => { hook.result.current.start('upload'); });
    const view = render(<DatasetImportProgress operation={hook.result.current.operation!}/>);
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '50');
    await act(async () => { await vi.advanceTimersByTimeAsync(500); });
    view.rerender(<DatasetImportProgress operation={hook.result.current.operation!}/>);
    expect(screen.getByRole('progressbar', { name: '导入进度 · 同步到当前版本' })).toHaveAttribute('aria-valuenow', '25');
    await act(async () => { hook.result.current.finish('completed'); });
    view.rerender(<DatasetImportProgress operation={hook.result.current.operation!}/>);
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '100');
    expect(screen.getByText('导入与登记已完成')).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(get).toHaveBeenCalledTimes(2);
  });

  it('does not fabricate a percentage, speed or ETA when totals are unknown', () => {
    render(<DatasetImportProgress operation={{ id: 'unknown', mode: 'path', state: 'active', elapsed: 3, unavailable: false,
      snapshot: snapshot('unknown', { phase: 'validating', bytes_done: 0, bytes_total: null, files_done: 4, files_total: null, bytes_per_second: null, eta_seconds: null }) }}/>);
    expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
    expect(screen.getByText('已处理 4 个文件')).toBeInTheDocument();
    expect(screen.queryByText(/MiB\/s/)).not.toBeInTheDocument();
    expect(screen.getByText('3s')).toBeInTheDocument();
  });

  it('recovers from the initial GET 404 racing the POST without overlapping polls', async () => {
    const get = vi.spyOn(apiClient, 'get').mockRejectedValueOnce(new ApiError(404, { code: 'progress.not_found', message: 'not created' }))
      .mockImplementation(async url => snapshot(idFrom(url)));
    const { result } = renderHook(() => useDatasetImportProgress('p_import'));
    await act(async () => { result.current.start('upload'); });
    expect(result.current.operation?.snapshot).toBeNull();
    await act(async () => { await vi.advanceTimersByTimeAsync(499); });
    expect(get).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(get).toHaveBeenCalledTimes(2);
    expect(result.current.operation?.snapshot?.phase).toBe('receiving');
  });

  it.each([[404, 8], [500, 3]])('stops bounded failures on HTTP %s while allowing the main import to succeed', async (status, attempts) => {
    const get = vi.spyOn(apiClient, 'get').mockRejectedValue(new ApiError(status, { code: 'unavailable', message: 'not available' }));
    const { result } = renderHook(() => useDatasetImportProgress('p_import'));
    await act(async () => { result.current.start('upload'); await vi.advanceTimersByTimeAsync(10000); });
    expect(get).toHaveBeenCalledTimes(attempts);
    expect(result.current.operation).toMatchObject({ state: 'active', unavailable: true });
    await act(async () => { result.current.finish('completed'); });
    expect(result.current.operation?.state).toBe('completed');
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(get).toHaveBeenCalledTimes(attempts);
  });

  it('stops at a terminal snapshot without publishing success before the import response', async () => {
    const get = vi.spyOn(apiClient, 'get').mockImplementation(async url => snapshot(idFrom(url), { phase: 'completed', bytes_done: 0, bytes_total: null }));
    const { result } = renderHook(() => useDatasetImportProgress('p_import'));
    await act(async () => { result.current.start('upload'); await vi.advanceTimersByTimeAsync(2000); });
    expect(get).toHaveBeenCalledOnce();
    expect(result.current.operation?.state).toBe('active');
    render(<DatasetImportProgress operation={result.current.operation!}/>);
    expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
    expect(screen.getByText('正在返回导入结果')).toBeInTheDocument();
  });

  it('aborts a stalled poll on timeout and waits for settlement before issuing another', async () => {
    const signals: AbortSignal[] = [];
    const get = vi.spyOn(apiClient, 'get').mockImplementation((_url, options) => new Promise((_resolve, reject) => {
      signals.push(options!.signal as AbortSignal);
      options!.signal!.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    }));
    const { result } = renderHook(() => useDatasetImportProgress('p_import'));
    await act(async () => { result.current.start('upload'); await vi.advanceTimersByTimeAsync(2999); });
    expect(get).toHaveBeenCalledOnce();
    expect(signals[0].aborted).toBe(false);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(signals[0].aborted).toBe(true);
    expect(get).toHaveBeenCalledOnce();
    await act(async () => { await vi.advanceTimersByTimeAsync(500); });
    expect(get).toHaveBeenCalledTimes(2);
  });

  it('aborts on unmount and ignores an older request resolving during a new operation', async () => {
    let resolveOld: (value: unknown) => void = () => {};
    let oldId = '';
    let oldSignal: AbortSignal | undefined;
    const get = vi.spyOn(apiClient, 'get').mockImplementationOnce((url, options) => new Promise(resolve => {
      oldId = idFrom(url); oldSignal = options?.signal as AbortSignal; resolveOld = resolve;
    })).mockImplementation(async url => snapshot(idFrom(url), { phase: 'copying' }));
    const { result, unmount } = renderHook(() => useDatasetImportProgress('p_import'));
    await act(async () => { result.current.start('upload'); });
    await act(async () => { result.current.start('path'); });
    const newId = result.current.operation!.id;
    expect(oldSignal?.aborted).toBe(true);
    await act(async () => { resolveOld(snapshot(oldId, { phase: 'failed' })); });
    expect(result.current.operation).toMatchObject({ id: newId, mode: 'path', snapshot: { phase: 'copying' } });
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
    expect(get).toHaveBeenCalledTimes(2);
  });
});
