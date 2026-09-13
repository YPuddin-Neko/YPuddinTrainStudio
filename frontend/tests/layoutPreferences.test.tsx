import { act, fireEvent, cleanup, render, screen, waitFor, within } from '@testing-library/react';
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
function show(openPreferences=true) {
  render(<MemoryRouter initialEntries={['/projects/p1/train?version=v1']}><Routes><Route element={<Layout />}><Route path="*" element={<Page />} /></Route></Routes></MemoryRouter>);
  if(openPreferences) fireEvent.click(screen.getByRole('button',{name:'界面偏好'}));
}
const sidebar = () => within(screen.getByRole('complementary'));

describe('sidebar appearance preferences', () => {
  it('persists sidebar collapse without writing trainer settings and restores accessible navigation after remount',async()=>{
    const user=userEvent.setup();
    show(false);await waitFor(()=>expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
    const collapse=sidebar().getByRole('button',{name:'收起侧边栏'});
    expect(collapse.closest('.sidebar-brand-row')).toBeInTheDocument();
    expect(collapse).toHaveAttribute('aria-controls','app-sidebar');
    expect(screen.getByTestId('app-topbar')).not.toContainElement(collapse);
    collapse.focus();await user.keyboard('{Enter}');
    expect(screen.getByRole('button',{name:'展开侧边栏'})).toHaveAttribute('aria-expanded','false');
    expect(sidebar().getByRole('button',{name:'展开侧边栏'})).toBe(collapse);
    expect(collapse).toHaveFocus();
    expect(screen.getByTestId('app-topbar').closest('.app-shell')).toHaveClass('sidebar-collapsed');
    expect(localStorage.getItem('studio.sidebar.collapsed')).toBe('true');expect(writes).toEqual([]);
    cleanup();show(false);const expand=sidebar().getByRole('button',{name:'展开侧边栏'});
    expect(screen.getByTestId('app-topbar')).not.toContainElement(expand);
    expect(screen.getByRole('link',{name:'项目'})).toHaveAttribute('href','/projects');
    expand.focus();await user.keyboard(' ');expect(localStorage.getItem('studio.sidebar.collapsed')).toBe('false');
    expect(sidebar().getByRole('button',{name:'收起侧边栏'})).toHaveFocus();
    expect(screen.getByTestId('app-topbar').closest('.app-shell')).not.toHaveClass('sidebar-collapsed');expect(writes).toEqual([]);
  });
  it('keeps the mobile menu operable with a remembered desktop collapse and returns focus on dismissal',async()=>{
    const user=userEvent.setup();localStorage.setItem('studio.sidebar.collapsed','true');show(false);
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
  it('opens compact preferences on demand and keeps its select interaction separate from outside and Escape dismissal',async()=>{
    const user=userEvent.setup();show(false);await waitFor(()=>expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
    const trigger=screen.getByRole('button',{name:'界面偏好'});expect(trigger).toHaveAttribute('aria-expanded','false');expect(screen.queryByRole('combobox',{name:'主题'})).not.toBeInTheDocument();
    await user.click(trigger);const theme=screen.getByRole('combobox',{name:'主题'});await user.click(theme);
    await user.click(screen.getByRole('option',{name:'深色'}));await waitFor(()=>expect(theme).toHaveTextContent('深色'));
    expect(trigger).toHaveAttribute('aria-expanded','true');await user.click(theme);await user.keyboard('{Escape}');
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();expect(theme).toBeInTheDocument();expect(trigger).toHaveAttribute('aria-expanded','true');
    await user.keyboard('{Escape}');expect(trigger).toHaveAttribute('aria-expanded','false');
    await user.click(trigger);fireEvent.pointerDown(screen.getByTestId('app-page-frame'));expect(screen.queryByRole('combobox',{name:'主题'})).not.toBeInTheDocument();
    expect(writes).toEqual([{ui:{theme:'dark'}}]);
  });
  it('offers system/light/dark plus the current language in the footer without a sidebar version or empty feedback slot', async () => {
    const user = userEvent.setup();
    show();
    await waitFor(() => expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
    const footer = screen.getByRole('complementary').querySelector('.sidebar-preferences');
    const theme = sidebar().getByRole('combobox', { name: '主题' });
    expect(footer).toContainElement(theme);
    expect(footer).toContainElement(sidebar().getByRole('combobox', { name: '语言' }));
    expect(theme).toHaveTextContent('自动');
    expect(sidebar().getByRole('combobox', { name: '语言' })).toHaveTextContent('中文');
    await user.click(theme);
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['自动', '浅色', '深色']);
    expect(sidebar().queryByText(/99\.88\.77|服务版本|Server version/)).not.toBeInTheDocument();
    expect(screen.getByTestId('app-topbar').querySelector('.topbar-feedback-slot')).toBeNull();
    expect(screen.getByTestId('app-topbar').querySelector('.topbar-job-slot')).toBeNull();
    expect(writes).toEqual([]);
  });

  it('keeps the previous selection during saving then applies the server response without navigation', async () => {
    const user = userEvent.setup();
    let finish = () => {};
    const waitForSave = new Promise<void>(resolve => { finish = resolve; });
    server.use(http.put('/api/settings', async ({ request }) => {
      const body = await request.json() as { ui: Partial<Settings['ui']> }; writes.push(body);
      await waitForSave;
      settings = { ...settings, ui: { ...settings.ui, ...body.ui } };
      return HttpResponse.json(settings);
    }));
    show();
    const theme = await sidebar().findByRole('combobox', { name: '主题' });
    await user.click(theme);
    await user.click(screen.getByRole('option', { name: '深色' }));
    await waitFor(() => expect(writes).toEqual([{ ui: { theme: 'dark' } }]));
    expect(theme).toHaveTextContent('自动');
    expect(theme).toBeDisabled();
    expect(sidebar().getByRole('combobox', { name: '语言' })).toBeDisabled();
    expect(document.documentElement).not.toHaveClass('dark');
    await act(async () => finish());
    await waitFor(() => expect(theme).toHaveTextContent('深色'));
    expect(theme).toBeEnabled();
    expect(document.documentElement).toHaveClass('dark');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });

  it('keeps the saved theme and reports a failed save instead of a false success', async () => {
    settings.ui.theme = 'light';
    server.use(http.put('/api/settings', () => HttpResponse.json({ error: { code: 'settings.write', message: 'Unable to save preferences' } }, { status: 503 })));
    const user = userEvent.setup(); show();
    const theme = await sidebar().findByRole('combobox', { name: '主题' });
    await waitFor(() => expect(theme).toHaveTextContent('浅色'));
    await user.click(theme); await user.click(screen.getByRole('option', { name: '深色' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Unable to save preferences');
    await waitFor(() => expect(theme).toBeEnabled());
    expect(theme).toHaveTextContent('浅色');
    expect(document.documentElement).not.toHaveClass('dark');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });

  it('persists the selected language, updates its label and keeps the same route', async () => {
    const user = userEvent.setup(); show();
    await user.click(await sidebar().findByRole('combobox', { name: '语言' }));
    await user.click(screen.getByRole('option', { name: 'EN' }));
    await waitFor(() => expect(i18n.resolvedLanguage).toBe('en'));
    expect(writes).toEqual([{ ui: { language: 'en' } }]);
    expect(sidebar().getByRole('combobox', { name: 'Language' })).toHaveTextContent('EN');
    expect(document.documentElement).toHaveAttribute('lang', 'en');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });

  it('preserves the current language after a failed language save', async () => {
    server.use(http.put('/api/settings', () => HttpResponse.json({ error: { code: 'settings.write', message: 'Language save failed' } }, { status: 503 })));
    const user = userEvent.setup(); show();
    await user.click(await sidebar().findByRole('combobox', { name: '语言' }));
    await user.click(screen.getByRole('option', { name: 'EN' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Language save failed');
    expect(i18n.resolvedLanguage).toBe('zh-CN');
    expect(sidebar().getByRole('combobox', { name: '语言' })).toHaveTextContent('中文');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });

  it('follows system theme changes in auto mode without saving or navigating', async () => {
    show();
    await waitFor(() => expect(mediaChanged).toBeTypeOf('function'));
    act(() => mediaChanged?.({ matches: true } as MediaQueryListEvent));
    expect(document.documentElement).toHaveClass('dark');
    expect(sidebar().getByRole('combobox', { name: '主题' })).toHaveTextContent('自动');
    act(() => mediaChanged?.({ matches: false } as MediaQueryListEvent));
    expect(document.documentElement).not.toHaveClass('dark');
    expect(writes).toEqual([]);
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });
});

it('returns keyboard focus to the preferences trigger after Escape closes the panel',async()=>{
  const user=userEvent.setup();show(false);
  await waitFor(()=>expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
  const trigger=screen.getByRole('button',{name:'界面偏好'});
  await user.click(trigger);
  const theme=screen.getByRole('combobox',{name:'主题'});
  await user.click(theme);
  await user.keyboard('{Escape}');
  expect(theme).toHaveFocus();
  expect(trigger).toHaveAttribute('aria-expanded','true');
  await user.keyboard('{Escape}');
  expect(screen.queryByRole('combobox',{name:'主题'})).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
  expect(writes).toEqual([]);
});

it('closes preferences on keyboard route navigation without moving focus back to the sidebar trigger',async()=>{
  const user=userEvent.setup();show(false);
  await waitFor(()=>expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
  const trigger=screen.getByRole('button',{name:'界面偏好'});
  await user.click(trigger);
  const settingsLink=screen.getByRole('link',{name:'系统设置'});
  act(()=>settingsLink.focus());
  await user.keyboard('{Enter}'); // No pointerdown: route changes must also dismiss the fixed popup.
  expect(screen.getByTestId('route')).toHaveTextContent('/settings');
  expect(screen.queryByRole('combobox',{name:'主题'})).not.toBeInTheDocument();
  expect(trigger).toHaveAttribute('aria-expanded','false');
  expect(trigger).not.toHaveFocus();
  expect(writes).toEqual([]);
});

it('dismisses preferences when the real App router opens settings over a frozen background workspace',async()=>{
  const user=userEvent.setup();
  const router=createMemoryRouter([{path:'*',element:<AppRoutes/>}],{initialEntries:[{
    pathname:'/projects/p1/v/v2',search:'?step=data',hash:'#images',state:{origin:'workspace-context'},
  }]});
  render(<RouterProvider router={router}/>);
  const draft=await screen.findByRole('textbox',{name:'Workspace draft'});
  await user.type(draft,'unsaved version notes');
  await waitFor(()=>expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
  const trigger=screen.getByRole('button',{name:'界面偏好'});
  await user.click(trigger);
  expect(screen.getByRole('combobox',{name:'主题'})).toBeInTheDocument();
  const settingsLink=screen.getByRole('link',{name:'系统设置'});
  act(()=>settingsLink.focus());
  await user.keyboard('{Enter}');
  const drawer=await screen.findByRole('dialog',{name:'系统设置'});
  await within(drawer).findByText('Runtime settings content');
  expect(router.state.location.pathname+router.state.location.search).toBe('/settings/environment?tab=runtime');
  expect(router.state.location.state.backgroundLocation.pathname).toBe('/projects/p1/v/v2');
  // Role queries hide inert background content, so inspect the DOM and trigger too.
  expect(document.querySelector('.sidebar-preferences-popover')).toBeNull();
  expect(trigger).toHaveAttribute('aria-expanded','false');
  expect(trigger).not.toHaveFocus();
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
