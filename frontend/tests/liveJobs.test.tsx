import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { handlers } from '../src/mocks/handlers';
import { mockJobs } from '../src/mocks/mockStore';
import '../src/i18n';

const subscriptions = vi.hoisted(() => new Map<string, Set<(data: any) => void>>());
vi.mock('../src/events/useEventStream', async () => {
  const { useEffect, useRef } = await import('react');
  return { useEventStream: (type: string, callback: (data: any) => void) => {
    const ref = useRef(callback); ref.current = callback;
    useEffect(() => {
      const listener = (data: any) => ref.current(data);
      if (!subscriptions.has(type)) subscriptions.set(type, new Set());
      subscriptions.get(type)!.add(listener);
      return () => { subscriptions.get(type)?.delete(listener); };
    }, [type]);
  } };
});
vi.mock('../src/components/EChart', () => ({ EChart: ({ option }: { option: unknown }) => <output data-testid="chart-option">{JSON.stringify(option)}</output> }));
import JobDetail from '../src/pages/JobDetail/JobDetail';
import Queue from '../src/pages/Queue/Queue';

const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => server.close());
const emit = (type: string, data: any) => act(() => subscriptions.get(type)?.forEach((callback) => callback(data)));
const chart = () => JSON.parse(screen.getAllByTestId('chart-option')[0].textContent!);
const perfChart = () => JSON.parse(screen.getAllByTestId('chart-option').at(-1)!.textContent!);

function showJob(vramMetric?: string) {
  server.use(
    http.get('/api/jobs/job_01', () => HttpResponse.json({ ...mockJobs[0], progress: { step: 2, total_steps: 100, steps_per_epoch: 10, phase: 'training', vram_peak_mb: 2048, vram_metric: vramMetric } })),
    http.get('/api/jobs/job_01/metrics', () => HttpResponse.json({ steps: [1, 2], loss: [1, 0], loss_ema: [99, 99], lr: { default: [0.01, 0.005] }, grad_norm: [1, 1], vram_mb: [10, 20], vram_metric: vramMetric, it_s: [2, 2], validation: [] })),
  );
  render(<MemoryRouter initialEntries={['/jobs/job_01']}><Routes>
    <Route path="/jobs/:id" element={<JobDetail />} />
    <Route path="/resumed" element={<div>Resume created</div>} />
  </Routes></MemoryRouter>);
}

describe('training monitor interactions and events', () => {
  it('recomputes EMA from raw loss and converts actual x values to epochs', async () => {
    showJob();
    await waitFor(() => expect(screen.getAllByTestId('chart-option').length).toBe(2));
    expect(chart().series[1].data).toEqual([[1, 1], [2, 0.9]]);
    fireEvent.change(screen.getByRole('slider'), { target: { value: '0.5' } });
    expect(chart().series[1].data).toEqual([[1, 1], [2, 0.5]]);
    fireEvent.click(screen.getByRole('button', { name: '轮' }));
    expect(chart().series[0].data).toEqual([[0.1, 1], [0.2, 0]]);
  });

  it('updates header progress, LR series and phase from SSE, ignoring replayed steps', async () => {
    showJob();
    await screen.findByText('2 / 100');
    await waitFor(() => expect(screen.getAllByTestId('chart-option').length).toBe(2));
    emit('job.step', { job_id: 'job_01', step: 3, epoch: 0, loss: 0.2, loss_ema: 0.3, lr: { default: 0.001, new_group: 0.002 }, it_s: 4, eta_s: 12, vram_mb: 50 });
    expect(screen.getByText('3 / 100')).toBeInTheDocument();
    expect(screen.getByText('4.00 it/s')).toBeInTheDocument();
    expect(chart().series.find((item: any) => item.name === 'lr:default').data).toEqual([[1, 0.01], [2, 0.005], [3, 0.001]]);
    expect(chart().series.find((item: any) => item.name === 'lr:new_group').data).toEqual([[1, null], [2, null], [3, 0.002]]);
    emit('job.step', { job_id: 'job_01', step: 2, loss: 100 });
    expect(screen.getByText('3 / 100')).toBeInTheDocument();
    expect(chart().series[0].data).toHaveLength(3);
    emit('job.phase', { job_id: 'job_01', phase: 'prepared', total_steps: 200, steps_per_epoch: 20 });
    expect(screen.getByText('3 / 200')).toBeInTheDocument();
    emit('job.phase', { job_id: 'job_01', phase: 'caching_text' });
    expect(screen.getByText('缓存').parentElement).toHaveClass('bg-blue-100');
  });

  it('shows MPS current allocation from the API and updates a decreasing value from SSE', async () => {
    showJob('current_allocated');
    await screen.findByText('2 / 100');
    await waitFor(() => expect(screen.getAllByTestId('chart-option').length).toBe(2));
    expect(screen.getByText('当前分配').parentElement).toHaveTextContent('2.0 GB');
    expect(screen.queryByText('显存峰值')).not.toBeInTheDocument();
    expect(screen.getByText('吞吐与当前训练分配量')).toBeInTheDocument();
    expect(perfChart().series[1].name).toBe('当前训练分配量 (GB)');
    emit('job.step', { job_id: 'job_01', step: 3, vram_mb: 1024, vram_metric: 'current_allocated' });
    expect(screen.getByText('当前分配').parentElement).toHaveTextContent('1.0 GB');
    expect(perfChart().series[1].data.at(-1)).toEqual([3, 1]);
    emit('job.step', { job_id: 'job_01', step: 2, vram_mb: 4096, vram_metric: 'peak_allocated' });
    expect(screen.getByText('当前分配').parentElement).toHaveTextContent('1.0 GB');
    expect(perfChart().series[1].name).toBe('当前训练分配量 (GB)');
  });

  it('retains legacy peak behavior and switches chart labels when CUDA metric metadata arrives', async () => {
    showJob();
    await screen.findByText('2 / 100');
    await waitFor(() => expect(screen.getAllByTestId('chart-option').length).toBe(2));
    expect(perfChart().series[1].name).toBe('VRAM (GB)');
    emit('job.step', { job_id: 'job_01', step: 3, vram_mb: 1024 });
    expect(screen.getByText('显存峰值').parentElement).toHaveTextContent('2.0 GB');
    emit('job.step', { job_id: 'job_01', step: 4, vram_mb: 4096, vram_metric: 'peak_allocated' });
    expect(screen.getByText('显存峰值').parentElement).toHaveTextContent('4.0 GB');
    expect(perfChart().series[1].name).toBe('显存峰值 (GB)');
    expect(perfChart().yAxis[1].name).toBe('显存峰值 (GB)');
    emit('job.step', { job_id: 'job_01', step: 5, vram_mb: null, vram_metric: 'peak_allocated' });
    expect(perfChart().series[1].data.at(-1)).toEqual([5, null]);
  });

  it('refreshes checkpoints, exposes the real weight link and submits full-state continuation', async () => {
    let checkpoints: any[] = [];
    let body: any;
    server.use(
      http.get('/api/jobs/job_01/checkpoints', () => HttpResponse.json(checkpoints)),
      http.get('/api/jobs/job_01/config', () => HttpResponse.json({ model: { family: 'toy' }, dataset: { sources: [{ path: '/images' }] }, checkpoint: { output_dir: '/old' } })),
      http.post('/api/jobs', async ({ request }) => { body = await request.json(); return HttpResponse.json({ id: 'resumed-job' }); }),
    );
    showJob();
    await screen.findByText('2 / 100');
    checkpoints = [
      { step: 10, kind: 'weights', path: '/weights.safetensors', artifact_id: 'art_test', created_at: 1, size: 12, ema: true },
      { step: 10, kind: 'full', path: '/state-10', created_at: 1, size: 22 },
    ];
    emit('job.checkpoint', { job_id: 'job_01', step: 10 });
    fireEvent.click(screen.getByRole('button', { name: /检查点/ }));
    expect(await screen.findByRole('link', { name: '下载' })).toHaveAttribute('href', 'http://localhost:3000/api/artifacts/art_test/download');
    expect(screen.getByText('仅权重 (EMA)')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '从此状态继续训练' }));
    await waitFor(() => expect(body?.config?.checkpoint?.resume).toBe('/state-10'));
    expect(body.config.dataset.sources).toEqual([{ path: '/images' }]);
    expect(body.project_id).toBe('proj_01');
  });
});

describe('queue pagination', () => {
  it('loads subsequent pages and applies server-side status filters', async () => {
    const requests: URL[] = [];
    server.use(http.get('/api/jobs', ({ request }) => {
      const url = new URL(request.url); requests.push(url);
      const page = Number(url.searchParams.get('page'));
      return HttpResponse.json({ items: [{ ...mockJobs[0], id: `page-${page}`, name: `Page ${page}` }], total: 51, page, page_size: 50 });
    }));
    render(<MemoryRouter><Queue /></MemoryRouter>);
    await screen.findByText('Page 1');
    fireEvent.click(screen.getByRole('button', { name: '下一页' }));
    await screen.findByText('Page 2');
    expect(requests[requests.length - 1].searchParams.get('page')).toBe('2');
    fireEvent.change(screen.getByRole('combobox', { name: '任务状态筛选' }), { target: { value: 'failed' } });
    await screen.findByText('Page 1');
    expect(requests[requests.length - 1].searchParams.get('status')).toBe('failed');
  });
});
