import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from '../src/mocks/handlers';
import Dashboard from '../src/pages/Dashboard/Dashboard';
import { MemoryRouter } from 'react-router-dom';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
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
      expect(screen.getByText('Queue Summary')).toBeInTheDocument();
      expect(screen.getByText('Recent Artifacts')).toBeInTheDocument();
    });
  });
});
