import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { handlers } from '../src/mocks/handlers';
import { mockJobs } from '../src/mocks/mockStore';
import { mergeJobEvent } from '../src/utils/jobs';
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
const lrChart = () => JSON.parse(screen.getAllByTestId('chart-option')[1].textContent!);
const perfChart = () => JSON.parse(screen.getAllByTestId('chart-option').at(-1)!.textContent!);

function showJob(vramMetric?: string, versionId?: string) {
  server.use(
    http.get('/api/projects/proj_01/versions', () => HttpResponse.json([])),
    http.get('/api/jobs/job_01', () => HttpResponse.json({ ...mockJobs[0], ...(versionId ? { version_id: versionId } : {}), progress: { step: 2, total_steps: 100, steps_per_epoch: 10, phase: 'training', vram_peak_mb: 2048, vram_metric: vramMetric } })),
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
    await waitFor(() => expect(screen.getAllByTestId('chart-option').length).toBe(1));
    expect(chart().series.map((series: any) => series.name)).toEqual(['原始损失', '显示 EMA']);
    expect(screen.getByText(/不改变训练参数或权重 EMA/)).toBeInTheDocument();
    expect(chart().series[1].data).toEqual([[1, 1], [2, 0.9]]);
    fireEvent.change(screen.getByRole('slider'), { target: { value: '0.5' } });
    expect(chart().series[1].data).toEqual([[1, 1], [2, 0.5]]);
    fireEvent.click(screen.getByRole('button', { name: '轮' }));
    expect(chart().series[0].data).toEqual([[0.1, 1], [0.2, 0]]);
  });

  it('uses each sample exact-step training loss, including live zero, without substituting nearby curve values', async () => {
    const sample = (step: number, loss: number | null) => ({step,loss,prompt_index:0,prompt:`loss sample ${step}`,seed:7,url:`/api/jobs/job_01/files?path=loss-${step}.png&kind=sample`,width:64,height:64,created_at:step+1});
    server.use(http.get('/api/jobs/job_01/samples', () => HttpResponse.json([sample(0,null),sample(2,null),sample(3,0.123456789)])));
    showJob(); fireEvent.click(await screen.findByRole('tab', {name:'采样图 (3)'}));
    const card = (step:number) => within(screen.getByRole('img',{name:`loss sample ${step}`}).closest('a')!.parentElement!);
    expect(card(0).getByText('初始采样 · 未训练')).toBeInTheDocument();
    expect(card(2).getByText('未记录')).toBeInTheDocument();
    expect(card(3).getByText('0.12346')).toBeInTheDocument();
    emit('job.sample', {...sample(4,0),job_id:'job_01'});
    expect(card(4).getByText('0')).toBeInTheDocument();
    expect(card(4).getByTitle(/不是这张采样图的质量评分/)).toBeInTheDocument();
  });

  it('updates header progress, LR series and phase from SSE, ignoring replayed steps', async () => {
    showJob();
    await screen.findByText('2 / 100');
    await waitFor(() => expect(screen.getAllByTestId('chart-option').length).toBe(1));
    fireEvent.click(screen.getByRole('button', { name: '学习率、梯度与性能诊断' }));
    emit('job.step', { job_id: 'job_01', step: 3, epoch: 0, loss: 0.2, loss_ema: 0.3, lr: { default: 0.001, new_group: 0.002, w1:0.0001, w2:0.0002 }, it_s: 4, eta_s: 12, vram_mb: 50 });
    expect(screen.getByText('3 / 100')).toBeInTheDocument();
    expect(screen.getByText('4.00 it/s')).toBeInTheDocument();
    expect(lrChart().series.find((item: any) => item.name === '学习率 · default').data).toEqual([[1, 0.01], [2, 0.005], [3, 0.001]]);
    expect(lrChart().series.find((item: any) => item.name === '学习率 · new_group').data).toEqual([[1, null], [2, null], [3, 0.002]]);
    expect(lrChart().series.find((item: any) => item.name === '学习率 · w1').data.at(-1)).toEqual([3,0.0001]);
    expect(lrChart().series.find((item: any) => item.name === '学习率 · w2').data.at(-1)).toEqual([3,0.0002]);
    expect(screen.getByText(/LoKr 的 w1 \/ w2 是两组矩阵参数/)).toBeInTheDocument();
    expect(chart().series).toHaveLength(2);
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
    await waitFor(() => expect(screen.getAllByTestId('chart-option').length).toBe(1));
    fireEvent.click(screen.getByRole('button', { name: '学习率、梯度与性能诊断' }));
    expect(perfChart().series[1].data).toEqual([[1,10/1024],[2,20/1024]]);
    expect(screen.queryByText('显存峰值')).not.toBeInTheDocument();
    expect(screen.getByText('训练速度与当前已分配内存')).toBeInTheDocument();
    expect(perfChart().series[1].name).toBe('训练当前已分配内存 (GB)');
    emit('job.step', { job_id: 'job_01', step: 3, vram_mb: 1024, vram_metric: 'current_allocated' });
    expect(perfChart().series[1].data.at(-1)).toEqual([3, 1]);
    emit('job.step', { job_id: 'job_01', step: 2, vram_mb: 4096, vram_metric: 'peak_allocated' });
    expect(perfChart().series[1].data.at(-1)).toEqual([3, 1]);
    expect(perfChart().series[1].data).toHaveLength(3);
    expect(perfChart().series[1].name).toBe('训练当前已分配内存 (GB)');
  });

  it('retains legacy peak behavior and switches chart labels when CUDA metric metadata arrives', async () => {
    showJob();
    await screen.findByText('2 / 100');
    await waitFor(() => expect(screen.getAllByTestId('chart-option').length).toBe(1));
    fireEvent.click(screen.getByRole('button', { name: '学习率、梯度与性能诊断' }));
    expect(perfChart().series[1].name).toBe('显存 (GB)');
    emit('job.step', { job_id: 'job_01', step: 3, vram_mb: 1024 });
    expect(perfChart().series[1].data.at(-1)).toEqual([3,1]);
    const previous={...mockJobs[0],progress:{step:2,vram_peak_mb:2048}};
    const updated=mergeJobEvent(previous,{job_id:previous.id,step:3,vram_mb:1024});
    expect(updated.progress?.vram_peak_mb).toBe(2048);
    emit('job.step', { job_id: 'job_01', step: 4, vram_mb: 4096, vram_metric: 'peak_allocated' });
    expect(perfChart().series[1].data.at(-1)).toEqual([4,4]);
    expect(mergeJobEvent(updated,{job_id:updated.id,step:4,vram_mb:4096,vram_metric:'peak_allocated'}).progress?.vram_peak_mb).toBe(4096);
    expect(perfChart().series[1].name).toBe('显存峰值 (GB)');
    expect(perfChart().yAxis[1].name).toBe('GB');
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
    showJob(undefined, 'v_original');
    await screen.findByText('2 / 100');
    expect(screen.getByTitle('v_original').closest('a')).toHaveAttribute('href', '/projects/proj_01/v/v_original?step=results');
    checkpoints = [
      { step: 10, kind: 'weights', path: '/weights.safetensors', artifact_id: 'art_test', created_at: 1, size: 12, ema: true },
      { step: 10, kind: 'full', path: '/state-10', created_at: 1, size: 22 },
    ];
    emit('job.checkpoint', { job_id: 'job_01', step: 10 });
    fireEvent.click(screen.getByRole('tab', { name: /检查点/ }));
    expect(await screen.findByRole('link', { name: '下载' })).toHaveAttribute('href', 'http://localhost:3000/api/artifacts/art_test/download');
    const weights=screen.getByText('weights.safetensors').closest('article')!;
    expect(within(weights).getByText('仅权重 · EMA')).toBeInTheDocument();
    expect(within(weights).getByText('步 10')).toBeInTheDocument();
    expect(within(weights).getByText('轮 1')).toBeInTheDocument();
    fireEvent.click(within(weights).getByText('文件详情',{selector:'summary'}));
    expect(within(weights).getByText('/weights.safetensors')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '从此状态继续训练' }));
    await waitFor(() => expect(body?.config?.checkpoint?.resume).toBe('/state-10'));
    expect(body.config.dataset.sources).toEqual([{ path: '/images' }]);
    expect(body.project_id).toBe('proj_01');
    expect(body.version_id).toBe('v_original');
  });

  it('merges delayed initial samples with live files and recovers complete history after the job finishes', async () => {
    const sample = (file: string, step: number) => ({ step, prompt_index: 0, prompt: file, seed: 7, url: `/api/jobs/job_01/files?path=${file}.png&kind=sample`, width: 64, height: 64, created_at: step + 1 });
    const initial = sample('initial', 0), step5 = sample('step5', 5), epoch1 = sample('epoch1', 5), step10 = sample('step10', 10), epoch2 = sample('epoch2', 10);
    let requests = 0; let releaseInitial: () => void = () => {};
    server.use(http.get('/api/jobs/job_01/samples', async () => {
      requests += 1;
      if (requests === 1) { await new Promise<void>(resolve => { releaseInitial = resolve; }); return HttpResponse.json([initial]); }
      return HttpResponse.json([initial, step5, epoch1, step10, epoch2]);
    }));
    showJob(); await screen.findByText('2 / 100');
    await waitFor(() => expect(requests).toBe(1));
    emit('job.sample', { ...step5, job_id: 'job_01' });
    emit('job.sample', { ...epoch1, job_id: 'job_01' });
    emit('job.sample', { ...epoch1, job_id: 'job_01' });
    expect(screen.getByRole('tab', { name: '采样图 (2)' })).toBeInTheDocument();
    await act(async () => releaseInitial());
    await screen.findByRole('tab', { name: '采样图 (3)' });
    emit('job.state', { job_id: 'other-job', status: 'completed' }); expect(requests).toBe(1);
    emit('job.state', { job_id: 'job_01', status: 'completed' });
    const samples = await screen.findByRole('tab', { name: '采样图 (5)' });
    expect(requests).toBe(2); fireEvent.click(samples);
    expect(screen.getAllByRole('img')).toHaveLength(5);
    expect(screen.getByRole('img', { name: 'initial' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'epoch2' })).toBeInTheDocument();
  });
});

describe('queue pagination', () => {
  it('tracks XYZ image progress and offers only supported actions', async () => {
    server.use(http.get('/api/jobs', () => HttpResponse.json({ items: [{ ...mockJobs[0], id: 'xyz-run', type: 'xyz', name: 'XYZ grid', status: 'running', progress: { done: 1, total: 8 } }], total: 1, page: 1, page_size: 50 })));
    render(<MemoryRouter><Queue /></MemoryRouter>);
    const row = await screen.findByTestId('job-row-xyz-run');
    expect(within(row).getByText('1 / 8 张')).toBeInTheDocument();
    expect(within(row).queryByRole('button', { name: '暂停' })).not.toBeInTheDocument();
    expect(within(row).queryByRole('button', { name: '保存检查点' })).not.toBeInTheDocument();
    expect(within(row).getByRole('button', { name: '取消' })).toBeEnabled();
    emit('job.xyz_progress', { job_id: 'xyz-run', done: 3, total: 8, sample_step: 2, sample_steps: 8 });
    expect(within(row).getByText('3 / 8 张')).toBeInTheDocument();
    emit('job.xyz_progress', { job_id: 'xyz-run', done: 1, total: 8 });
    expect(within(row).getByText('3 / 8 张')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('combobox', { name: '任务类型' }));
    expect(screen.getByRole('option', { name: '模型测试' })).toBeInTheDocument();
  });
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
    fireEvent.click(screen.getByRole('tab', { name: /训练历史/ }));
    fireEvent.click(screen.getByRole('combobox', { name: '状态筛选' })); fireEvent.click(screen.getByRole('option', { name: '失败' }));
    await screen.findByText('Page 1');
    expect(requests[requests.length - 1].searchParams.get('status')).toBe('failed');
  });
});
