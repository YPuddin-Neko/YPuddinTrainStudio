import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll, vi } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from '../src/mocks/handlers';
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
});
