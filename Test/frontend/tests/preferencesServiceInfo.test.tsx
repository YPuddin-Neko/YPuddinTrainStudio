import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import Preferences from '../../../frontend/src/pages/Settings/Preferences';
import { apiClient } from '../../../frontend/src/api/client';
import type { Settings } from '../../../frontend/src/api/types';
import i18n from '../../../frontend/src/i18n';

const settings: Settings = {
  paths: { bootstrap_env_dir: '', data_root: 'D:/studio', models_dir: 'D:/models', cache_dir: 'D:/cache', output_dir: 'D:/runs', output_mode: 'project' },
  server: { open_browser: true, host: '127.0.0.1', port: 8765 },
  ui: { language: 'zh-CN', theme: 'light' },
};
const systemInfo = vi.fn();

beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  systemInfo.mockReset().mockResolvedValue({ ypuddin: '0.5.1-test' });
  vi.spyOn(apiClient, 'get').mockImplementation(async endpoint => {
    if (endpoint === '/settings') return settings as never;
    if (endpoint === '/system/info') return systemInfo();
    throw new Error(`Unexpected GET ${endpoint}`);
  });
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

function show(section = 'interface') {
  return render(<MemoryRouter initialEntries={[`/settings/preferences?section=${section}`]}><Preferences /></MemoryRouter>);
}

describe('service version in appearance and service settings', () => {
  it('shows the running server version from the API beside the connection information', async () => {
    show();
    const row = await screen.findByTestId('settings-service-version');
    expect(await within(row).findByText('0.5.1-test')).toBeInTheDocument();
    expect(within(row).getByText('服务版本')).toBeInTheDocument();
    expect(document.getElementById('preferences-service')).toContainElement(row);
    expect(screen.getByText(window.location.origin)).toBeInTheDocument();
    expect(apiClient.get).toHaveBeenCalledWith('/system/info', expect.objectContaining({ signal: expect.any(AbortSignal), silent: true }));
  });

  it('does not fetch or show service information in storage settings', async () => {
    show('storage');
    await screen.findByTestId('settings-page');
    expect(screen.queryByTestId('settings-service-version')).not.toBeInTheDocument();
    expect(systemInfo).not.toHaveBeenCalled();
  });

  it('persists an explicit base environment directory and allows returning to the default', async () => {
    const put = vi.spyOn(apiClient, 'put').mockImplementation(async (_path, value) => value as never);
    show('storage');
    const input = await screen.findByRole('textbox', { name: '基础环境目录' });
    fireEvent.change(input, { target: { value: 'E:/trainer-environments' } });
    fireEvent.click(screen.getByTestId('settings-save-btn'));
    await waitFor(() => expect(put).toHaveBeenCalledWith('/settings', expect.objectContaining({ paths: expect.objectContaining({ bootstrap_env_dir: 'E:/trainer-environments' }) })));
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: '' } });
    fireEvent.click(screen.getByTestId('settings-save-btn'));
    await waitFor(() => expect(put).toHaveBeenLastCalledWith('/settings', expect.objectContaining({ paths: expect.objectContaining({ bootstrap_env_dir: '' }) })));
  });

  it('shows an unavailable value instead of inventing a version when the API omits it', async () => {
    systemInfo.mockResolvedValue({ python: '3.11.9' });
    show();
    const row = await screen.findByTestId('settings-service-version');
    expect(await within(row).findByText(i18n.t('hardware.unavailable'))).toBeInTheDocument();
    expect(within(row).queryByRole('status')).not.toBeInTheDocument();
  });

  it('can retry a version lookup failure while settings remain editable', async () => {
    systemInfo.mockRejectedValueOnce(new Error('Service info temporarily unavailable'));
    show();
    const row = await screen.findByTestId('settings-service-version');
    expect(await within(row).findByRole('alert')).toHaveTextContent('Service info temporarily unavailable');
    fireEvent.click(screen.getByTestId('settings-theme'));
    fireEvent.click(screen.getByRole('option', { name: i18n.t('settings.themeDark') }));
    expect(screen.getByTestId('settings-save-btn')).toBeEnabled();
    fireEvent.click(within(row).getByRole('button', { name: i18n.t('common.retry') }));
    expect(await within(row).findByText('0.5.1-test')).toBeInTheDocument();
    await waitFor(() => expect(systemInfo).toHaveBeenCalledTimes(2));
    expect(screen.getByTestId('settings-theme')).toHaveTextContent(i18n.t('settings.themeDark'));
    expect(within(row).queryByRole('alert')).not.toBeInTheDocument();
  });
});
