import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll, vi } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { handlers } from '../src/mocks/handlers';
import { mockJobs } from '../src/mocks/mockStore';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import '../src/i18n';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Mock 轻量 EChart 封装，避免 jsdom 下 canvas / ResizeObserver 问题
vi.mock('../src/components/EChart', () => ({
  EChart: ({ option }: { option: { dataZoom: { type: string }[] } }) => <div data-testid="echarts-mock" data-zoom-types={option.dataZoom.map(zoom => zoom.type).join(',')}>Chart</div>,
}));

import JobDetail from '../src/pages/JobDetail/JobDetail';
import Queue from '../src/pages/Queue/Queue';

describe('JobDetail Page (B1, B2, B3, B4)', () => {
  it('does not report an unknown phase as running after the job has failed', async () => {
    server.use(http.get('/api/jobs/job_01', () => HttpResponse.json({ ...mockJobs[0], status: 'failed', progress: { step: 3, total_steps: 10 }, error: 'model unavailable' })));
    render(<MemoryRouter initialEntries={['/jobs/job_01']}><Routes><Route path="/jobs/:id" element={<JobDetail/>}/></Routes></MemoryRouter>);
    await screen.findByRole('heading',{name:/chara-v1/i});
    expect(screen.queryByText('进行中')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '重新训练' })).toBeInTheDocument();
  });

  it('renders header stats, metrics charts, and timeline without crash', async () => {
    render(
      <MemoryRouter initialEntries={['/jobs/job_01']}>
        <Routes>
          <Route path="/jobs/:id" element={<JobDetail />} />
        </Routes>
      </MemoryRouter>
    );

    // 等待头部信息加载
    await waitFor(() => {
      expect(screen.getByTestId('job-detail-page')).toBeInTheDocument();
      expect(screen.getByRole('heading',{name:/chara-v1/i})).toBeInTheDocument();
    });

    // 检查阶段时间线（i18n 中文文案）
    expect(screen.getByText('准备')).toBeInTheDocument();
    expect(screen.getByText('训练')).toBeInTheDocument();
    
    // 默认只呈现主损失，诊断图按需加载，避免隐藏容器初始化。
    const charts = screen.getAllByTestId('echarts-mock');
    expect(charts).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: '学习率、梯度与性能诊断' }));
    expect(screen.getAllByTestId('echarts-mock').length).toBeGreaterThanOrEqual(4);
    for (const chart of screen.getAllByTestId('echarts-mock')) expect(chart).toHaveAttribute('data-zoom-types', 'slider');
    expect(screen.getByText('拖动图下方滑块缩放或调整查看范围。')).toBeInTheDocument();
    expect(screen.queryByText(/Ctrl.*滚轮/)).not.toBeInTheDocument();
  });

  it('shows an archived version name after a separate lookup without delaying the job or Chinese sample tab', async () => {
    let releaseVersions: () => void = () => {}; let requested: URL | undefined;
    server.use(
      http.get('/api/jobs/job_01', () => HttpResponse.json({ ...mockJobs[0], version_id: 'v_original' })),
      http.get('/api/projects/proj_01/versions', async ({ request }) => {
        requested = new URL(request.url);
        await new Promise<void>(resolve => { releaseVersions = resolve; });
        return HttpResponse.json([{ id: 'v_active', project_id: 'proj_01', name: 'Current experiment' }, { id: 'v_original', project_id: 'proj_01', name: '服装遮罩实验', archived: true }]);
      }),
      http.get('/api/jobs/job_01/samples', () => HttpResponse.json([0, 5, 10].map(step => ({ step, prompt_index: 0, prompt: `sample ${step}`, seed: 7, url: `/api/jobs/job_01/files?path=${step}.png&kind=sample`, width: 64, height: 64, created_at: 1 })))),
    );
    render(<MemoryRouter initialEntries={['/jobs/job_01']}><Routes><Route path="/jobs/:id" element={<JobDetail/>}/></Routes></MemoryRouter>);
    await screen.findByRole('heading',{name:/chara-v1/i});
    const samples = await screen.findByRole('tab', { name: '采样图 (3)' });
    await waitFor(() => expect(requested?.searchParams.get('include_archived')).toBe('true'));
    expect(screen.queryByText(/v_original/)).not.toBeInTheDocument();
    expect(screen.getByTitle('v_original').closest('a')).toHaveAttribute('href', '/projects/proj_01/v/v_original?step=results');
    await act(async () => releaseVersions());
    expect(await screen.findByRole('link', { name: /版本 服装遮罩实验/ })).toBeInTheDocument();
    expect(screen.getByTitle('v_original')).toHaveTextContent('服装遮罩实验');
    fireEvent.click(samples);
    expect(within(screen.getByTestId('samples-gallery')).getAllByRole('img')).toHaveLength(3);
  });

  it.each(['missing', 'unavailable'])('keeps legacy job monitoring usable when version metadata is $0', async outcome => {
    let lookedUp = false;
    server.use(
      http.get('/api/jobs/job_01', () => HttpResponse.json({ ...mockJobs[0], version_id: 'v_legacy' })),
      http.get('/api/projects/proj_01/versions', () => { lookedUp = true; return outcome === 'missing' ? HttpResponse.json([]) : HttpResponse.json({ error: { message: 'Version metadata unavailable' } }, { status: 500 }); }),
    );
    render(<MemoryRouter initialEntries={['/jobs/job_01']}><Routes><Route path="/jobs/:id" element={<JobDetail/>}/></Routes></MemoryRouter>);
    await screen.findByRole('heading',{name:/chara-v1/i});
    await waitFor(() => expect(lookedUp).toBe(true));
    expect(screen.getByTitle('v_legacy')).toHaveTextContent('所属版本');
    expect(screen.queryByText('Version metadata unavailable')).not.toBeInTheDocument();
    expect(screen.getAllByTestId('echarts-mock')).toHaveLength(1);
  });
});

it('keeps actions present in logs and reads history with byte cursors', async () => {
  const requests: URL[] = [];
  server.use(http.get('/api/jobs/job_01/log', ({ request }) => {
    const url = new URL(request.url); requests.push(url);
    if (url.searchParams.get('tail') === 'true') return HttpResponse.json({ lines: [{ level: 'info', msg: 'live last line', ts: null }], next_offset: 900, has_more: false });
    const second = url.searchParams.get('offset') === '223';
    return HttpResponse.json({ lines: [{ level: second ? 'warn' : 'info', msg: second ? 'history page two' : 'history page one', ts: null }], next_offset: second ? 900 : 223, has_more: !second });
  }));
  render(<MemoryRouter initialEntries={['/jobs/job_01?tab=logs']}><Routes><Route path="/jobs/:id" element={<JobDetail/>}/></Routes></MemoryRouter>);
  await screen.findByText('live last line');
  expect(screen.getByRole('tab', { name: '日志' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByRole('button', { name: '暂停' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: '保存检查点' })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /全局训练队列/ })).toHaveAttribute('href', '/queue');
  fireEvent.click(screen.getByRole('combobox', { name: '日志模式' })); fireEvent.click(screen.getByRole('option', { name: '完整历史 · 分页读取' }));
  await screen.findByText('history page one');
  expect(screen.queryByText('live last line')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '下一页日志' }));
  await screen.findByText('history page two');
  expect(requests.at(-1)?.searchParams.get('offset')).toBe('223');
  expect(screen.getByRole('button', { name: '下一页日志' })).toBeDisabled();
  fireEvent.click(screen.getByRole('combobox', { name: '日志级别' })); fireEvent.click(screen.getByRole('option', { name: 'WARN' }));
  expect(screen.getByText('history page two')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '上一页日志' }));
  await waitFor(() => expect(requests.at(-1)?.searchParams.get('offset')).toBe('0'));
});

it('paginates a large sample history and preserves full-image links', async () => {
  server.use(http.get('/api/jobs/job_01/samples', () => HttpResponse.json(Array.from({ length: 60 }, (_, step) => ({ step, prompt_index: 0, prompt: `sample ${step}`, seed: 7, url: `/api/jobs/job_01/files?path=${step}.png&kind=sample`, width: 64, height: 64, created_at: step })))));
  render(<MemoryRouter initialEntries={['/jobs/job_01?tab=samples']}><Routes><Route path="/jobs/:id" element={<JobDetail/>}/></Routes></MemoryRouter>);
  await screen.findByRole('img', { name: 'sample 59' });
  expect(within(screen.getByTestId('samples-gallery')).getAllByRole('img')).toHaveLength(24);
  fireEvent.click(screen.getByRole('button', { name: '下一页' }));
  expect(screen.getByRole('img', { name: 'sample 35' })).toBeInTheDocument();
  expect(screen.queryByRole('img', { name: 'sample 59' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('combobox', { name: '采样步数' })); fireEvent.click(screen.getByRole('option', { name: '步数 5' }));
  expect(within(screen.getByTestId('samples-gallery')).getAllByRole('img')).toHaveLength(1);
  expect(screen.getByRole('link', { name: '打开完整采样图: sample 5' })).toHaveAttribute('href', expect.stringContaining('/jobs/job_01/files?path=5.png'));
});

it('returns to the exact filtered queue page after navigating through detail tabs', async () => {
  const queueUrl = '/queue?view=history&project_id=proj_01&type=train&status=completed&q=chara&page=2&size=20';
  const requests: URL[] = [];
  server.use(http.get('/api/jobs', ({ request }) => {
    const url = new URL(request.url); requests.push(url);
    return HttpResponse.json({ items: [{ ...mockJobs[0], status: 'completed' }], total: 25, page: Number(url.searchParams.get('page') || 1), page_size: Number(url.searchParams.get('page_size') || 20) });
  }));
  function CurrentLocation() { const location = useLocation(); return <output data-testid="current-location">{location.pathname}{location.search}</output>; }
  render(<MemoryRouter initialEntries={[queueUrl]}><Routes><Route path="/queue" element={<Queue/>}/><Route path="/jobs/:id" element={<JobDetail/>}/></Routes><CurrentLocation/></MemoryRouter>);
  const row = await screen.findByTestId('job-row-job_01');
  fireEvent.click(within(row).getByRole('link', { name: /chara-v1/i }));
  await screen.findByTestId('job-detail-page');
  fireEvent.click(screen.getByRole('tab', { name: '日志' }));
  expect(screen.getByTestId('current-location')).toHaveTextContent('tab=logs');
  fireEvent.keyDown(screen.getByRole('tab', { name: '日志' }), { key: 'ArrowRight' });
  expect(screen.getByRole('tab', { name: '配置快照' })).toHaveAttribute('aria-selected', 'true');
  fireEvent.click(screen.getByRole('link', { name: /全局训练队列/ }));
  await screen.findByTestId('job-row-job_01');
  expect(screen.getByTestId('current-location')).toHaveTextContent(queueUrl);
  expect(requests.some(url => url.searchParams.get('group') === 'history' && url.searchParams.get('project_id') === 'proj_01' && url.searchParams.get('type') === 'train' && url.searchParams.get('status') === 'completed' && url.searchParams.get('q') === 'chara' && url.searchParams.get('page') === '2')).toBe(true);
});


it.each(['logs', 'config'])('keeps XYZ %s free of training metrics when navigating between tabs', async (tab) => {
  server.use(http.get('/api/jobs/job_01', () => HttpResponse.json({...mockJobs[0], id: 'job_01', name: 'XYZ 对比任务', type: 'xyz', status: 'completed'})));
  const {container} = render(<MemoryRouter initialEntries={[`/jobs/job_01?tab=${tab}`]}><Routes><Route path="/jobs/:id" element={<JobDetail/>}/></Routes></MemoryRouter>);
  await screen.findByRole('heading', {name: 'XYZ 对比任务'});
  expect(container.querySelector('.job-monitor-summary')).not.toBeInTheDocument();
  expect(screen.getByText('运行时长')).toBeInTheDocument();
  expect(screen.getByText('任务配置')).toBeInTheDocument();
  expect(screen.queryByLabelText('训练核心指标')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', {name: tab === 'logs' ? '配置快照' : '日志'}));
  expect(container.querySelector('.job-monitor-summary')).not.toBeInTheDocument();
  expect(screen.queryByText('训练时长')).not.toBeInTheDocument();
});
