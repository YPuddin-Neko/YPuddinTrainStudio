import { fireEvent, render, screen } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import ProjectDetail from '../../../frontend/src/pages/ProjectDetail/ProjectDetail';
import '../../../frontend/src/i18n';

vi.mock('../../../frontend/src/events/useEventStream', () => ({ useEventStream: () => {} }));
const server = setupServer(
  http.get('/api/projects/p_redirect', () => HttpResponse.json({ id: 'p_redirect', name: 'Redirect project', active_version_id: 'v2' })),
  http.get('/api/projects/p_redirect/versions', () => HttpResponse.json([])),
);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => server.resetHandlers());

function Destination() {
  const location = useLocation();
  const navigate = useNavigate();
  return <>
    <output data-testid="destination">{location.pathname}{location.search}{location.hash}</output>
    <output data-testid="navigation-state">{JSON.stringify(location.state)}</output>
    <button onClick={() => navigate(-1)}>Back</button>
  </>;
}

it.each([
  { label: 'the results jobs tab', search: '?step=results&result_tab=jobs&q=red%20hair', hash: '', expected: '?step=results&result_tab=jobs&q=red%20hair' },
  { label: 'the data stage and dataset anchor', search: '?step=data&data_step=import', hash: '#version-datasets', expected: '?step=data&data_step=import' },
  { label: 'the overview default with other query parameters', search: '?retained=1', hash: '#version-datasets', expected: '?retained=1&step=overview' },
  { label: 'the overview default without query parameters', search: '', hash: '', expected: '?step=overview' },
])('preserves $label and navigation state when resolving the active version', async ({ search, hash, expected }) => {
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter initialEntries={['/previous', { pathname: '/projects/p_redirect', search, hash, state: { origin: 'kept', selected: ['a', 'b'] } }]} initialIndex={1}>
      <Routes>
        <Route path="/previous" element={<p>Previous page</p>}/>
        <Route path="/projects/:id" element={<ProjectDetail/>}/>
        <Route path="/projects/:id/v/:versionId" element={<Destination/>}/>
      </Routes>
    </MemoryRouter>
  </QueryClientProvider>);
  expect(await screen.findByTestId('destination')).toHaveTextContent(`/projects/p_redirect/v/v2${expected}${hash}`);
  expect(screen.getByTestId('navigation-state')).toHaveTextContent('{"origin":"kept","selected":["a","b"]}');
  fireEvent.click(screen.getByRole('button', { name: 'Back' }));
  expect(await screen.findByText('Previous page')).toBeInTheDocument();
});
