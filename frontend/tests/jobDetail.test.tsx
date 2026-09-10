import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll, vi } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from '../src/mocks/handlers';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Mock ReactECharts component to avoid canvas rendering issues in jsdom
vi.mock('echarts-for-react', () => ({
  default: () => <div data-testid="echarts-mock">Chart</div>,
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

    // 检查阶段时间线
    expect(screen.getByText('preparing')).toBeInTheDocument();
    expect(screen.getByText('training')).toBeInTheDocument();
    
    // 检查 mock 图表是否正常注入（loss + validation + throughput 三张图）
    const charts = screen.getAllByTestId('echarts-mock');
    expect(charts.length).toBeGreaterThanOrEqual(3);
  });
});
