import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll, vi } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { handlers } from '../src/mocks/handlers';
import { mockJobs } from '../src/mocks/mockStore';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import '../src/i18n';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Mock 轻量 EChart 封装，避免 jsdom 下 canvas / ResizeObserver 问题
vi.mock('../src/components/EChart', () => ({
  EChart: () => <div data-testid="echarts-mock">Chart</div>,
}));

import JobDetail from '../src/pages/JobDetail/JobDetail';

describe('JobDetail Page (B1, B2, B3, B4)', () => {
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
      expect(screen.getByText(/chara-v1/i)).toBeInTheDocument();
    });

    // 检查阶段时间线（i18n 中文文案）
    expect(screen.getByText('准备')).toBeInTheDocument();
    expect(screen.getByText('训练')).toBeInTheDocument();
    
    // 检查 mock 图表是否正常注入（loss + validation + throughput 三张图）
    const charts = screen.getAllByTestId('echarts-mock');
    expect(charts.length).toBeGreaterThanOrEqual(3);
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
    await screen.findByText(/chara-v1/i);
    const samples = await screen.findByRole('button', { name: '采样图 (3)' });
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
    await screen.findByText(/chara-v1/i);
    await waitFor(() => expect(lookedUp).toBe(true));
    expect(screen.getByTitle('v_legacy')).toHaveTextContent('所属版本');
    expect(screen.queryByText('Version metadata unavailable')).not.toBeInTheDocument();
    expect(screen.getAllByTestId('echarts-mock').length).toBeGreaterThanOrEqual(3);
  });
});
