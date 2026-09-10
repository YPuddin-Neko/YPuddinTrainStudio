import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from '../src/mocks/handlers';
import Dashboard from '../src/pages/Dashboard/Dashboard';
import { MemoryRouter } from 'react-router-dom';
import '../src/i18n';
import i18n from '../src/i18n';

const server = setupServer(...handlers);

beforeAll(async () => {
  server.listen();
  await i18n.changeLanguage('zh-CN'); // 页面文案已 i18n 化，固定默认中文再按文本断言
});
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Dashboard Page (C2)', () => {
  it('renders system stats bar and active job card', async () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      expect(screen.getByText('队列摘要')).toBeInTheDocument();
      expect(screen.getByText('最近产物')).toBeInTheDocument();
    });
  });
});
