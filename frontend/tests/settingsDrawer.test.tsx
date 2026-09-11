import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, useLocation } from 'react-router-dom';
import AppRoutes from '../src/router';
import { apiClient } from '../src/api/client';
import i18n from '../src/i18n';

vi.mock('../src/components/Layout', async () => {
  const { Suspense } = await import('react'); const { Outlet } = await import('react-router-dom');
  return { default: () => <Suspense fallback={<p>Loading workspace</p>}><Outlet/></Suspense> };
});
vi.mock('../src/pages/ProjectDetail/ProjectDetail', async () => {
  const { useState } = await import('react'); const { Link, useLocation } = await import('react-router-dom');
  return { default: function WorkspaceFixture() {
    const [draft, setDraft] = useState(''); const location = useLocation();
    return <main data-testid="background-project"><input aria-label="Workspace draft" value={draft} onChange={event => setDraft(event.target.value)}/><Link to="/settings/environment?tab=runtime" state={{ backgroundLocation: location }}>Open settings</Link></main>;
  } };
});
vi.mock('../src/components/EnvironmentManagerPanel', () => ({ EnvironmentManagerPanel: () => <p>Runtime manager</p> }));
vi.mock('../src/pages/Models/Models', () => ({ default: () => <p>Model manager</p> }));

beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  vi.spyOn(apiClient, 'get').mockImplementation(async endpoint => {
    if (endpoint === '/settings') return { paths: { data_root: 'D:/studio', cache_dir: 'D:/cache', output_dir: 'D:/runs', models_dir: 'D:/models' }, server: { host: '127.0.0.1', port: 8765 }, ui: { language: 'zh-CN', theme: 'light' } } as any;
    throw new Error(`Unexpected GET ${endpoint}`);
  });
});
afterEach(() => vi.restoreAllMocks());
function Location() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}{location.hash} {JSON.stringify(location.state)}</output>; }

describe('settings drawer workspace context', () => {
  it.each(['close button', 'Escape'])('keeps the version workspace mounted through settings tabs and returns to it using $0', async close => {
    render(<MemoryRouter initialEntries={[{ pathname: '/projects/p1/v/v2', search: '?step=models', hash: '#weights', state: { origin: 'saved-context' } }]}><AppRoutes/><Location/></MemoryRouter>);
    const draft = await screen.findByRole('textbox', { name: 'Workspace draft' });
    fireEvent.change(draft, { target: { value: 'Unsubmitted version notes' } });
    fireEvent.click(screen.getByRole('link', { name: 'Open settings' }));
    const drawer = await screen.findByRole('dialog', { name: '系统设置' });
    await within(drawer).findByText('Runtime manager');
    expect(draft).toBeInTheDocument(); expect(draft).toHaveValue('Unsubmitted version notes');
    expect(screen.getByTestId('background-project').closest('.route-surface')).toHaveAttribute('aria-hidden', 'true');
    expect(screen.getByTestId('background-project').closest('.route-surface')).toHaveAttribute('inert');
    fireEvent.click(within(drawer).getByRole('tab', { name: '存储路径' }));
    await within(drawer).findByRole('textbox', { name: i18n.t('settings.dataRoot') });
    expect(screen.getByTestId('location')).toHaveTextContent('backgroundLocation');
    if (close === 'Escape') fireEvent.keyDown(drawer, { key: 'Escape' });
    else fireEvent.click(within(drawer).getByRole('button', { name: '关闭设置，返回工作区' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Workspace draft' })).toBe(draft);
    expect(draft).toHaveValue('Unsubmitted version notes');
    expect(screen.getByTestId('location')).toHaveTextContent('/projects/p1/v/v2?step=models#weights');
    expect(screen.getByTestId('location')).toHaveTextContent('saved-context');
    expect(screen.getByTestId('background-project').closest('.route-surface')).not.toHaveAttribute('inert');
  });
});
