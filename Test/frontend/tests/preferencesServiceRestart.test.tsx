import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import Preferences from '../../../frontend/src/pages/Settings/Preferences';
import { apiClient } from '../../../frontend/src/api/client';
import type { Settings } from '../../../frontend/src/api/types';
import i18n from '../../../frontend/src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it('restarts at the saved port through one action without submitting later unsaved edits', async () => {
  let persisted: Settings = {
    paths: { bootstrap_env_dir: '', data_root: 'D:/studio', models_dir: 'D:/models', cache_dir: 'D:/cache', output_dir: 'D:/runs', output_mode: 'project' },
    server: { host: '127.0.0.1', port: 8765 },
    ui: { language: 'zh-CN', theme: 'light' },
  };
  vi.spyOn(apiClient, 'get').mockImplementation(async endpoint => {
    if (endpoint === '/settings') return structuredClone(persisted) as never;
    if (endpoint === '/system/info') return { ypuddin: '0.5.9-test' } as never;
    if (endpoint === '/service/runtime') return {
      worker_id: 11, managed: true, can_restart: true, reason: null,
      current_host: '127.0.0.1', current_port: 8765,
      saved_host: persisted.server.host, saved_port: persisted.server.port,
      current_python: 'project/venv/python', can_restore_original: false, selected_environment: null,
    } as never;
    throw new Error(`Unexpected GET ${endpoint}`);
  });
  const put = vi.spyOn(apiClient, 'put').mockImplementation(async (_endpoint, body) => {
    persisted = structuredClone(body as Settings);
    return structuredClone(persisted) as never;
  });
  const post = vi.spyOn(apiClient, 'post').mockImplementation(async (_endpoint, body) => {
    const address = (body as { apply_saved_address?: boolean }).apply_saved_address
      ? persisted.server : { host: '127.0.0.1', port: 8765 };
    return { ...address, address_changed: address.port !== 8765, reconnect_url: `http://${address.host}:${address.port}/` } as never;
  });

  render(<MemoryRouter initialEntries={['/settings/preferences?section=interface']}><Preferences /></MemoryRouter>);
  const port = await screen.findByRole('spinbutton', { name: i18n.t('settings.port') });
  const restart = screen.getByRole('button', { name: '重启服务' });
  await waitFor(() => expect(restart).toBeEnabled());
  expect(screen.getAllByRole('button', { name: /重启/ })).toEqual([restart]);
  fireEvent.change(port, { target: { value: '9000' } });
  expect(put).not.toHaveBeenCalled();
  expect(post).not.toHaveBeenCalled();
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  await waitFor(() => expect(put).toHaveBeenCalledWith('/settings', expect.objectContaining({ server: { host: '127.0.0.1', port: 9000 } })));
  await waitFor(() => expect(restart).toBeEnabled());
  fireEvent.change(port, { target: { value: '9001' } });
  expect(port).toHaveValue(9001);
  expect(screen.getAllByRole('button', { name: /重启/ })).toEqual([restart]);
  fireEvent.click(restart);

  expect(await screen.findByRole('link', { name: '打开新的服务地址' })).toHaveAttribute('href', 'http://127.0.0.1:9000/');
  expect(post).toHaveBeenCalledOnce();
  expect(post).toHaveBeenCalledWith('/service/restart', { apply_saved_address: true }, { silent: true, signal: expect.any(AbortSignal) });
  expect(put).toHaveBeenCalledOnce();
  expect(persisted.server.port).toBe(9000);
  expect(port).toHaveValue(9001);
});
