import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useLocation, useSearchParams } from 'react-router-dom';
import Settings from '../../../frontend/src/pages/Settings/Settings';
import EnvironmentSettings from '../../../frontend/src/pages/Settings/EnvironmentSettings';
import Preferences from '../../../frontend/src/pages/Settings/Preferences';
import { apiClient } from '../../../frontend/src/api/client';
import i18n from '../../../frontend/src/i18n';

vi.mock('../../../frontend/src/components/EnvironmentManagerPanel', () => ({ EnvironmentManagerPanel: () => <div data-testid="runtime-panel">Runtime manager</div> }));
vi.mock('../../../frontend/src/pages/Models/Models', () => ({ default: function EmbeddedModels({ embedded }: { embedded?: boolean }) { const [params] = useSearchParams(); return <div data-testid="embedded-models">{embedded ? 'Embedded' : 'Standalone'} model weights: {params.get('family')}</div>; } }));
const config = { paths: { data_root: 'D:/studio', models_dir: 'D:/models', cache_dir: 'D:/cache', output_dir: 'D:/runs' }, server: { host: '127.0.0.1', port: 8765 }, ui: { language: 'zh-CN', theme: 'light' } };
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  vi.spyOn(apiClient, 'get').mockImplementation(async endpoint => {
    if (endpoint === '/credentials') return {huggingface:{configured:false},modelscope:{configured:false},danbooru:{configured:false},gelbooru:{configured:false}} as any;
    if (endpoint === '/settings') return config as any;
    if (endpoint === '/system/info') return { ypuddin: 'test' } as any;
    throw new Error(`Unexpected GET ${endpoint}`);
  });
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });
function Location() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search} {JSON.stringify(location.state)}</output>; }
function show(path: string) {
  render(<MemoryRouter initialEntries={[{ pathname: path.split('?')[0], search: path.includes('?') ? `?${path.split('?')[1]}` : '', state: { backgroundLocation: { pathname: '/projects/p1' } } }]}>
    <Routes><Route path="settings" element={<Settings />}><Route path="environment" element={<EnvironmentSettings />} /><Route path="preferences" element={<Preferences />} /></Route></Routes><Location />
  </MemoryRouter>);
}

describe('settings drawer content navigation', () => {
  it('uses one set of purpose tabs, preserving model queries and the background on keyboard navigation', async () => {
    show('/settings/environment?family=krea2&project=p1&tab=models');
    expect(await screen.findByTestId('embedded-models')).toHaveTextContent('Embedded model weights: krea2');
    expect(screen.getAllByRole('tablist')).toHaveLength(1);
    expect(screen.getByRole('tab', { name: '模型权重' })).toHaveAttribute('aria-selected', 'true');
    fireEvent.keyDown(screen.getByRole('tab', { name: '模型权重' }), { key: 'ArrowLeft' });
    expect(await screen.findByTestId('runtime-panel')).toBeInTheDocument();
    expect(screen.getByTestId('location')).toHaveTextContent('family=krea2&project=p1&tab=runtime');
    expect(screen.getByTestId('location')).toHaveTextContent('backgroundLocation');
    expect(screen.getByRole('tab', { name: '运行环境' })).toHaveFocus();
    expect(screen.queryByRole('tab', { name: '训练产物' })).not.toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
  it('separates storage from appearance and service while keeping chapter navigation', async () => {
    show('/settings/environment?tab=runtime');
    await screen.findByTestId('runtime-panel');
    fireEvent.click(screen.getByRole('tab', { name: '存储路径' }));
    const page = await screen.findByTestId('settings-page');
    expect(screen.getByTestId('location')).toHaveTextContent('/settings/preferences?section=storage');
    expect(within(page).getByRole('textbox', { name: i18n.t('settings.dataRoot') })).toHaveAttribute('readonly');
    expect(screen.queryByTestId('settings-theme')).not.toBeInTheDocument();
    expect(screen.queryByTestId('runtime-panel')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: '界面与服务' }));
    expect(screen.getByTestId('settings-theme')).toHaveTextContent(i18n.t('settings.themeLight'));
    expect(screen.queryByRole('textbox', { name: i18n.t('settings.dataRoot') })).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: i18n.t('settings.host') })).toHaveValue('127.0.0.1');
    const index = screen.getByRole('navigation', { name: '当前页章节' });
    fireEvent.click(within(index).getByRole('button', { name: i18n.t('settings.server') }));
    expect(document.getElementById('preferences-service')).toHaveFocus();
    expect(within(index).getByRole('button', { name: i18n.t('settings.server') })).toHaveAttribute('aria-current', 'location');
  });
  it('opens the independent access-keys page with all four providers and preserves drawer context',async()=>{
    show('/settings/environment?tab=runtime');
    fireEvent.click(screen.getByRole('tab',{name:'访问密钥'}));
    expect(await screen.findByTestId('access-keys-settings')).toBeInTheDocument();
    expect(screen.getByRole('tab',{name:'访问密钥'})).toHaveAttribute('aria-selected','true');
    expect(screen.getByTestId('location')).toHaveTextContent('/settings/environment?tab=credentials');
    expect(screen.getByTestId('location')).toHaveTextContent('backgroundLocation');
    expect(screen.getByLabelText('Danbooru 用户名')).toHaveValue('');
    expect(screen.getByLabelText('Gelbooru 用户 ID')).toHaveValue('');
    expect(screen.queryByTestId('embedded-models')).not.toBeInTheDocument();
    expect(screen.queryByTestId('runtime-panel')).not.toBeInTheDocument();
  });

});
