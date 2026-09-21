import { fireEvent, render, screen } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { apiClient } from '../../../frontend/src/api/client';
import { ApiError } from '../../../frontend/src/api/types';
import { ApiErrorNotice } from '../../../frontend/src/components/ApiErrorNotice';
import { formatApiError } from '../../../frontend/src/utils/errors';
import i18n from '../../../frontend/src/i18n';

const server = setupServer();
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('API validation error presentation', () => {
  it('turns transport failures into a localized service recovery instruction', async () => {
    await i18n.changeLanguage('zh-CN');
    expect(formatApiError(new TypeError('Failed to fetch'))).toBe('暂时无法连接训练服务，请确认服务已启动后重试。');
    await i18n.changeLanguage('en');
    expect(formatApiError(new TypeError('Load failed'))).toContain('Check that it is running');
    await i18n.changeLanguage('zh-CN');
  });
  it('formats string and array field locations and retains plain errors', () => {
    const error = new ApiError(400, { code: 'config.invalid', message: 'invalid config', details: { errors: [
      { loc: ['sampling', 'prompts', 0, 'steps'], msg: 'Input should be greater than 0' },
      { loc: 'checkpoint.save_full_state', msg: 'Extra inputs are not permitted' },
    ] } });
    expect(formatApiError(error)).toBe('invalid config\nsampling.prompts.0.steps: Input should be greater than 0\ncheckpoint.save_full_state: Extra inputs are not permitted');
    expect(formatApiError(new Error('Network unavailable'))).toBe('Network unavailable');
  });

  it('displays field details through the shared API notice and lets the user dismiss them', async () => {
    server.use(http.post('/api/presets', () => HttpResponse.json({ error: {
      code: 'config.invalid', message: 'invalid config',
      details: { errors: [{ loc: ['loop', 'epochs'], msg: 'Input should be greater than 0' }] },
    } }, { status: 400 })));
    render(<><ApiErrorNotice /><button onClick={() => { void apiClient.post('/presets', {}).catch(() => {}); }}>Save</button></>);
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('loop.epochs: Input should be greater than 0');
    fireEvent.click(screen.getByRole('button', { name: '关闭' }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
