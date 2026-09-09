import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from '../src/mocks/handlers';
import Queue from '../src/pages/Queue/Queue';
import { MemoryRouter } from 'react-router-dom';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Queue Page (C1, C3)', () => {
  it('renders jobs table and handles queue settings', async () => {
    render(
      <MemoryRouter>
        <Queue />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByTestId('jobs-table')).toBeInTheDocument();
      expect(screen.getByText('Job Queue')).toBeInTheDocument();
    });

    // 检查暂停调度按钮
    const heldBtn = screen.getByText(/Scheduling Held|Pause Scheduling/i);
    fireEvent.click(heldBtn);

    // 触发 POST pause 动作
    const pauseBtn = await screen.findByTitle('Pause');
    fireEvent.click(pauseBtn);
  });
});
