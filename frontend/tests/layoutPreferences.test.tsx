import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { createMemoryRouter, MemoryRouter, Route, RouterProvider, Routes, useLocation } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import Layout from '../src/components/Layout';
import AppRoutes from '../src/router';
import type { Settings } from '../src/api/types';
import i18n from '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {}, useEventStreamStatus: () => 'connected' }));
vi.mock('../src/pages/ProjectDetail/ProjectDetail', async () => {
  const { useState } = await import('react');
  return { default: function DraftWorkspace() {
    const [draft, setDraft] = useState('');
    return <input aria-label="Workspace draft" value={draft} onChange={event => setDraft(event.target.value)}/>;
  } };
});
vi.mock('../src/pages/Settings/EnvironmentSettings', () => ({ default: () => <p>Runtime settings content</p> }));
const server = setupServer();
const initialSettings = (): Settings => ({ paths: { data_root: 'D:/studio', cache_dir: 'D:/cache', models_dir: 'D:/models', output_dir: 'D:/runs', output_mode: 'project' }, server: { host: '127.0.0.1', port: 8765 }, ui: { theme: 'system', language: 'zh-CN' } });
let settings: Settings;
let writes: unknown[];
let mediaChanged: ((event: MediaQueryListEvent) => void) | undefined;
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
beforeEach(async () => {
  localStorage.removeItem('studio.sidebar.collapsed');
  await i18n.changeLanguage('zh-CN');
  settings = initialSettings(); writes = []; mediaChanged = undefined;
  vi.stubGlobal('matchMedia', () => ({ matches: false, media: '(prefers-color-scheme: dark)', addEventListener: (_: string, callback: typeof mediaChanged) => { mediaChanged = callback; }, removeEventListener: () => {} }));
  server.use(
    http.get('/api/settings', () => HttpResponse.json(settings)),
    http.get('/api/system/stats', () => HttpResponse.json({ cpu_pct: 0, ram: { used_mb: 1000, total_mb: 8000 }, disks: [], gpus: [] })),
    http.get('/api/jobs', () => HttpResponse.json({ items: [] })),
    http.get('/api/system/info', () => HttpResponse.json({ ypuddin: '99.88.77-server' })),
    http.put('/api/settings', async ({ request }) => {
      const body = await request.json() as { ui: Partial<Settings['ui']> };
      writes.push(body); settings = { ...settings, ui: { ...settings.ui, ...body.ui } };
      return HttpResponse.json(settings);
    }),
  );
});
afterEach(() => { cleanup(); server.resetHandlers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); document.documentElement.classList.remove('dark'); localStorage.removeItem('studio.sidebar.collapsed'); });
function Page() { const location = useLocation(); return <output data-testid="route">{location.pathname}{location.search}</output>; }
function show() {
  render(<MemoryRouter initialEntries={['/projects/p1/train?version=v1']}><Routes><Route element={<Layout />}><Route path="*" element={<Page />} /></Route></Routes></MemoryRouter>);
}
const sidebar = () => within(screen.getByRole('complementary'));

describe('sidebar footer navigation', () => {
  it('persists sidebar collapse without writing trainer settings and restores accessible navigation after remount',async()=>{
    const user=userEvent.setup();
    show();await waitFor(()=>expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
    const collapse=sidebar().getByRole('button',{name:'收起侧边栏'});
    const footer=collapse.closest('.sidebar-footer');
    expect(footer).toBeInTheDocument();
    expect(footer?.lastElementChild).toBe(collapse);
    expect(footer).toContainElement(sidebar().getByRole('link',{name:'系统设置'}));
    expect(collapse).toHaveTextContent('收起侧边栏');
    expect(sidebar().queryByRole('button',{name:'界面偏好'})).not.toBeInTheDocument();
    expect(sidebar().getAllByRole('link',{name:'系统设置'})).toHaveLength(1);
    expect(collapse.closest('.sidebar-brand-row')).toBeNull();
    expect(collapse).toHaveAttribute('aria-controls','app-sidebar');
    expect(screen.getByTestId('app-topbar')).not.toContainElement(collapse);
    collapse.focus();await user.keyboard('{Enter}');
    expect(screen.getByRole('button',{name:'展开侧边栏'})).toHaveAttribute('aria-expanded','false');
    expect(sidebar().getByRole('button',{name:'展开侧边栏'})).toBe(collapse);
    expect(collapse.closest('.sidebar-footer')).toBe(footer);
    expect(footer?.lastElementChild).toBe(collapse);
    expect(collapse).toHaveFocus();
    expect(screen.getByTestId('app-topbar').closest('.app-shell')).toHaveClass('sidebar-collapsed');
    expect(localStorage.getItem('studio.sidebar.collapsed')).toBe('true');expect(writes).toEqual([]);
    cleanup();show();const expand=sidebar().getByRole('button',{name:'展开侧边栏'});
    expect(expand.closest('.sidebar-footer')).toBeInTheDocument();
    expect(screen.getByTestId('app-topbar')).not.toContainElement(expand);
    expect(screen.getByRole('link',{name:'项目'})).toHaveAttribute('href','/projects');
    expand.focus();await user.keyboard(' ');expect(localStorage.getItem('studio.sidebar.collapsed')).toBe('false');
    expect(sidebar().getByRole('button',{name:'收起侧边栏'})).toHaveFocus();
    expect(screen.getByTestId('app-topbar').closest('.app-shell')).not.toHaveClass('sidebar-collapsed');expect(writes).toEqual([]);
  });
  it('keeps settings accessible after collapsing the sidebar without changing that preference',async()=>{
    const user=userEvent.setup();show();
    await waitFor(()=>expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
    await user.click(sidebar().getByRole('button',{name:'收起侧边栏'}));
    const settingsLink=sidebar().getByRole('link',{name:'系统设置'});
    expect(settingsLink.closest('.sidebar-footer')).toBeInTheDocument();
    settingsLink.focus();await user.keyboard('{Enter}');
    expect(screen.getByTestId('route')).toHaveTextContent('/settings');
    expect(sidebar().getByRole('button',{name:'展开侧边栏'})).toHaveAttribute('aria-expanded','false');
    expect(localStorage.getItem('studio.sidebar.collapsed')).toBe('true');
    expect(writes).toEqual([]);
  });
  it('keeps the mobile menu operable with a remembered desktop collapse and returns focus on dismissal',async()=>{
    const user=userEvent.setup();localStorage.setItem('studio.sidebar.collapsed','true');show();
    await waitFor(()=>expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
    const opener=within(screen.getByTestId('app-topbar')).getByRole('button',{name:i18n.t('hardware.openMenu')});
    const panel=screen.getByRole('complementary');
    expect(opener).toHaveAttribute('aria-controls',panel.id);
    expect(opener).toHaveAttribute('aria-expanded','false');
    opener.focus();await user.keyboard('{Enter}');
    expect(opener).toHaveAttribute('aria-expanded','true');expect(panel).not.toHaveClass('hidden');
    const close=within(panel).getByRole('button',{name:i18n.t('hardware.closeMenu')});close.focus();await user.keyboard('{Enter}');
    expect(panel).toHaveClass('hidden');expect(opener).toHaveAttribute('aria-expanded','false');expect(opener).toHaveFocus();
    await user.click(opener);await user.click(document.querySelector('.app-sidebar-backdrop')!);
    expect(opener).toHaveAttribute('aria-expanded','false');expect(opener).toHaveFocus();
    expect(localStorage.getItem('studio.sidebar.collapsed')).toBe('true');expect(writes).toEqual([]);
  });
  it('follows system theme changes in auto mode without saving or navigating', async () => {
    show();
    await waitFor(() => expect(mediaChanged).toBeTypeOf('function'));
    act(() => mediaChanged?.({ matches: true } as MediaQueryListEvent));
    expect(document.documentElement).toHaveClass('dark');
    act(() => mediaChanged?.({ matches: false } as MediaQueryListEvent));
    expect(document.documentElement).not.toHaveClass('dark');
    expect(writes).toEqual([]);
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });
});

it('opens footer settings over the workspace and returns focus, location and the unsaved draft on close',async()=>{
  const user=userEvent.setup();
  const router=createMemoryRouter([{path:'*',element:<AppRoutes/>}],{initialEntries:[{
    pathname:'/projects/p1/v/v2',search:'?step=data',hash:'#images',state:{origin:'workspace-context'},
  }]});
  render(<RouterProvider router={router}/>);
  const draft=await screen.findByRole('textbox',{name:'Workspace draft'});
  await user.type(draft,'unsaved version notes');
  await waitFor(()=>expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
  const settingsLink=screen.getByRole('link',{name:'系统设置'});
  act(()=>settingsLink.focus());
  await user.keyboard('{Enter}');
  const drawer=await screen.findByRole('dialog',{name:'系统设置'});
  await within(drawer).findByText('Runtime settings content');
  expect(router.state.location.pathname+router.state.location.search).toBe('/settings/environment?tab=runtime');
  expect(router.state.location.state.backgroundLocation.pathname).toBe('/projects/p1/v/v2');
  const close=within(drawer).getByRole('button',{name:'关闭设置，返回工作区'});
  expect(close).toHaveFocus();
  expect(draft).toHaveValue('unsaved version notes');
  expect(draft.closest('.route-surface')).toHaveAttribute('inert');
  await user.click(close);
  await waitFor(()=>expect(screen.queryByRole('dialog',{name:'系统设置'})).not.toBeInTheDocument());
  expect(router.state.location.pathname+router.state.location.search+router.state.location.hash).toBe('/projects/p1/v/v2?step=data#images');
  expect(router.state.location.state).toEqual({origin:'workspace-context'});
  expect(screen.getByRole('textbox',{name:'Workspace draft'})).toBe(draft);
  expect(draft).toHaveValue('unsaved version notes');
  expect(settingsLink).toHaveFocus();
  expect(document.querySelector('.sidebar-preferences-popover')).toBeNull();
  expect(writes).toEqual([]);
});
