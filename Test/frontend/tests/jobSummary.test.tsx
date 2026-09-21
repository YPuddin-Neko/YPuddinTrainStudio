import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import type { Job, JobMetrics } from '../../../frontend/src/api/types';
import { mockJobs } from '../mocks/mockStore';
import '../../../frontend/src/i18n';

const subscriptions = vi.hoisted(() => new Map<string, Set<(event: Record<string, unknown>) => void>>());
vi.mock('../../../frontend/src/events/useEventStream', async () => {
  const { useEffect, useRef } = await import('react');
  return { useEventStream: (type: string, callback: (event: Record<string, unknown>) => void) => {
    const current = useRef(callback); current.current = callback;
    useEffect(() => {
      const listener = (event: Record<string, unknown>) => current.current(event);
      if (!subscriptions.has(type)) subscriptions.set(type, new Set());
      subscriptions.get(type)!.add(listener);
      return () => { subscriptions.get(type)?.delete(listener); };
    }, [type]);
  } };
});
vi.mock('../../../frontend/src/components/EChart', () => ({ EChart: () => <div data-testid="training-chart" /> }));
import JobDetail from '../../../frontend/src/pages/JobDetail/JobDetail';

const server = setupServer();
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => { server.resetHandlers(); vi.useRealTimers(); vi.restoreAllMocks(); });
afterAll(() => server.close());

const metrics: JobMetrics = { steps: [1, 2, 3, 4], loss: [1, 0, null, 0.5], loss_ema: [99, 99, 99, 99],
  lr: { w1: [0.01, 0.005, 0.003, 0.001], w2: [0.02, 0.01, 0.006, 0.002] },
  grad_norm: [1, 1, 1, 1], vram_mb: [100, 100, 100, 100], it_s: [2, 2, 2, 2], validation: [] };

function job(overrides: Partial<Job> = {}): Job {
  return { ...mockJobs[0], project_id: null, version_id: null, name: 'Summary run',
    started_at: 1000, finished_at: null, ...overrides,
    progress: { step: 4, total_steps: 20, steps_per_epoch: 10, epoch: 0, phase: 'training', it_s: 2, eta_s: 8, ...overrides.progress },
    latest: { loss: 0.25, loss_ema: 99, loss_mean: 0.5, loss_count: 4, loss_mean_scope: 'run', lr: { w1: 0.001, w2: 0.002 }, ...overrides.latest },
  };
}

async function show(value = job(), history = metrics) {
  server.use(
    http.get('/api/jobs/job_01', () => HttpResponse.json(value)),
    http.get('/api/jobs/job_01/metrics', () => HttpResponse.json(history)),
    http.get('/api/jobs/job_01/samples', () => HttpResponse.json([])),
    http.get('/api/jobs/job_01/checkpoints', () => HttpResponse.json([])),
    http.get('/api/jobs/job_01/config', () => HttpResponse.json({ model: { family: 'toy' }, loop: { epochs: 2 } })),
    http.get('/api/jobs/job_01/log', () => HttpResponse.json({ lines: [], next_offset: 0, has_more: false })),
  );
  render(<MemoryRouter initialEntries={['/jobs/job_01']}><Routes><Route path="/jobs/:id" element={<JobDetail />} /></Routes></MemoryRouter>);
  await screen.findByRole('heading', { name: value.name });
  await screen.findByTestId('training-chart');
}

function card(label: string) {
  const summary = screen.getByLabelText('训练核心指标');
  return within(within(summary).getByText(label, { exact: true }).closest('.job-stat') as HTMLElement);
}
function value(label: string) {
  return card(label).getByText((_, element) => element?.className === 'job-stat-value').textContent;
}
const emit = (event: Record<string, unknown>) => act(() => subscriptions.get('job.step')?.forEach(listener => listener(event)));

it('shows seven actual training summaries and names each optimizer parameter-group learning rate', async () => {
  await show();
  const summary = screen.getByLabelText('训练核心指标');
  expect(summary.children).toHaveLength(7);
  expect(value('步数')).toBe('4 / 20');
  expect(value('轮次')).toBe('0.40 / 2');
  expect(value('Loss')).toBe('0.2500');
  expect(value('平均 Loss')).toBe('0.5000');
  expect(card('学习率').getByText('w1')).toBeInTheDocument();
  expect(card('学习率').getByText('w2')).toBeInTheDocument();
  expect(value('学习率')).toBe('w11.00e-3w22.00e-3');
  expect(value('速度')).toBe('2.00 it/s');
  expect(value('预计剩余')).toBe('8s');
  fireEvent.click(screen.getByRole('button', { name: '平均 Loss · 说明' }));
  expect(screen.getByRole('tooltip')).toHaveTextContent('所有已完成训练步的损失平均值');
  expect(screen.getByLabelText('运行信息')).toHaveTextContent('Summary run · 参数快照');
  expect(screen.getByLabelText('运行信息')).toHaveTextContent('job_01');
});

it('updates mean scope and all optimizer groups from SSE, preserving real zero values and rejecting stale or foreign steps', async () => {
  await show();
  emit({ job_id: 'job_01', step: 5, epoch: 0, loss: 0, loss_mean: 0, loss_count: 1,
    loss_mean_scope: 'since_resume', lr: { w1: 0, w2: 0.004 }, it_s: 0, eta_s: 0 });
  expect(value('步数')).toBe('5 / 20');
  expect(value('轮次')).toBe('0.50 / 2');
  expect(value('Loss')).toBe('0.0000');
  expect(value('平均 Loss')).toBe('0.0000');
  expect(value('学习率')).toBe('w10.00e+0w24.00e-3');
  expect(value('速度')).toBe('0.00 it/s');
  expect(value('预计剩余')).toBe('0s');
  fireEvent.click(screen.getByRole('button', { name: '平均 Loss · 说明' }));
  expect(screen.getByRole('tooltip')).toHaveTextContent('从此次恢复训练起');
  emit({ job_id: 'job_01', step: 3, loss_mean: 100, loss_count: 300, loss_mean_scope: 'run' });
  emit({ job_id: 'other', step: 6, loss: 999, loss_mean: 999 });
  expect(value('平均 Loss')).toBe('0.0000');
  expect(screen.getByRole('tooltip')).toHaveTextContent('从此次恢复训练起');
  emit({ job_id: 'job_01', step: 6, loss: 0.4, loss_mean: 0.2, loss_count: 2, loss_mean_scope: 'since_resume' });
  expect(value('平均 Loss')).toBe('0.2000');
});

it('labels a legacy recorded-step mean clearly and never substitutes stored display EMA', async () => {
  await show(job({ latest: { loss: 0, loss_ema: 999, loss_mean: null, loss_count: null, loss_mean_scope: null, lr: {} } }));
  expect(value('平均 Loss')).toBe('0.5000'); // (1 + 0 + 0.5) / 3; the null entry is absent, not zero.
  expect(value('Loss')).toBe('0.0000');
  expect(value('学习率')).toBe('—');
  fireEvent.click(screen.getByRole('button', { name: '平均 Loss · 说明' }));
  expect(screen.getByRole('tooltip')).toHaveTextContent('旧任务没有完整累计值');
  expect(screen.getByRole('tooltip')).toHaveTextContent('已有日志中训练步的平均值');
});

it('keeps missing current loss and empty historical mean unknown instead of inventing zero', async () => {
  await show(job({ latest: { loss: null, loss_mean: null, loss_count: null, lr: {} } }), { ...metrics, loss: [null, null, null, null] });
  expect(value('Loss')).toBe('—');
  expect(value('平均 Loss')).toBe('—');
});

it.each(['completed', 'paused'] as const)('freezes elapsed time at the recorded %s endpoint', async status => {
  await show(job({ status, started_at: 1000, finished_at: 1125 }));
  const elapsed = () => within(screen.getByLabelText('运行信息')).getByText('训练时长').parentElement;
  expect(elapsed()).toHaveTextContent('2m 5s');
  vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] });
  await act(async () => { vi.advanceTimersByTime(120000); });
  expect(elapsed()).toHaveTextContent('2m 5s');
  if (status === 'completed') expect(value('预计剩余')).toBe('0s');
});

it('advances running duration from wall time while retaining the recorded start timestamp', async () => {
  vi.spyOn(Date, 'now').mockReturnValue(1100000);
  await show(job({ status: 'running', started_at: 1000 }));
  const elapsed = () => within(screen.getByLabelText('运行信息')).getByText('训练时长').parentElement;
  expect(elapsed()).toHaveTextContent('1m 40s');
  // Advance one interval without globally faking MSW's response scheduling.
  vi.mocked(Date.now).mockReturnValue(1102000);
  await waitFor(() => expect(elapsed()).toHaveTextContent('1m 42s'), { timeout: 1500 });
});
