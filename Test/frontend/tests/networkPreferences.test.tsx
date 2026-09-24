import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import Preferences from '../../../frontend/src/pages/Settings/Preferences';
import { apiClient } from '../../../frontend/src/api/client';
import i18n from '../../../frontend/src/i18n';

vi.mock('../../../frontend/src/components/ServiceControls', () => ({ default: () => null }));
vi.mock('../../../frontend/src/pages/Settings/ServiceInfo', () => ({ ServiceInfo: () => null }));
const network = { proxy_mode: 'custom', proxy_url: 'http://localhost:7890', proxy_username: 'account', proxy_password_configured: true };
const settings = { paths: {}, server: { open_browser: true, host: '127.0.0.1', port: 8765 }, ui: { language: 'zh-CN', theme: 'light' }, network };
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  vi.spyOn(apiClient, 'get').mockResolvedValue(structuredClone(settings));
  vi.spyOn(apiClient, 'put').mockResolvedValue(structuredClone(settings));
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });
function show() { render(<MemoryRouter initialEntries={['/settings/preferences?section=interface']}><Preferences /></MemoryRouter>); }

it('keeps the existing private proxy password when saving unrelated settings', async () => {
  show();
  const password = await screen.findByLabelText('代理密码（可选）');
  expect(password).toHaveValue('');
  expect(password).toHaveAttribute('type', 'password');
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  await waitFor(() => expect(apiClient.put).toHaveBeenCalled());
  const body = vi.mocked(apiClient.put).mock.calls[0][1] as { network: Record<string, unknown> };
  expect(body.network).not.toHaveProperty('proxy_password');
});

it('sends a changed password once and clears its editor after a successful save', async () => {
  show();
  const password = await screen.findByLabelText('代理密码（可选）');
  fireEvent.change(password, { target: { value: 'fixture-secret' } });
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  await waitFor(() => expect(apiClient.put).toHaveBeenCalledWith('/settings', expect.objectContaining({ network: expect.objectContaining({ proxy_password: 'fixture-secret' }) })));
  await waitFor(() => expect(password).toHaveValue(''));
  expect(screen.queryByText('fixture-secret')).not.toBeInTheDocument();
});

it('clears a stored password only through the explicit clear action', async () => {
  show();
  fireEvent.click(await screen.findByRole('button', { name: '清除已保存密码' }));
  expect(screen.getByRole('button', { name: '保存时清除密码' })).toBeInTheDocument();
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  await waitFor(() => expect(apiClient.put).toHaveBeenCalledWith('/settings', expect.objectContaining({ network: expect.objectContaining({ proxy_password: '' }) })));
});

it('does not ask for a service restart after saving proxy settings', async () => {
  show();
  await screen.findByLabelText('代理密码（可选）');
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  await screen.findByText('已保存');
  expect(screen.queryByText('已保存，重启服务后生效')).not.toBeInTheDocument();
});

it('keeps the restart requirement when the listening address changes', async () => {
  vi.mocked(apiClient.put).mockImplementation(async (_path, body) => body as any);
  show();
  const host = await screen.findByLabelText('监听地址');
  fireEvent.change(host, {target: {value: '0.0.0.0'}});
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  await screen.findByText('已保存，重启服务后生效。');
});
