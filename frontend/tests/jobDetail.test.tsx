import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from '../src/mocks/handlers';
import JobDetail from '../src/pages/JobDetail/JobDetail';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

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
      expect(screen.getByText('Job Monitor')).toBeDefined();
    });

    // 检查阶段时间线
    expect(screen.getByText('preparing')).toBeInTheDocument();
    expect(screen.getByText('training')).toBeInTheDocument();
  });
});
