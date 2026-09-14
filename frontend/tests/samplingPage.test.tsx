import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import Sampling from '../src/pages/Sampling/Sampling';
import { apiClient } from '../src/api/client';
import { samplingUrl } from '../src/utils/samplingRoutes';
import i18n from '../src/i18n';
const source = { id: 'live', name: '正在训练的 Krea2', type: 'train', status: 'running', project_id: 'p1', version_id: 'v1', project_name: '服装', version_name: 'v1' };
const options = { family: 'krea2', defaults: { prompt: 'a pudding', negative: '', width: 512, height: 512, steps: 20, cfg: 4, seed: 42, sampler: 'euler', scheduler: 'uniform', shift: null, guidance: null, adapter_scale: 1, checkpoint_id: null, sampling_model_id: null }, checkpoints: [], sampling_models: [], axes: [{ key: 'steps', label: 'Steps' }, { key: 'seed', label: 'Seed' }], limits: { max_cells: 64, max_axis_values: 12 } };
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  vi.spyOn(apiClient, 'get').mockImplementation(async (url) => {
    if (url === '/projects') return [{ id: 'p1', name: '服装' }] as never;
    if (url === '/projects/p1') return { archived: false } as never;
    if (url === '/projects/p1/versions/v1') return { archived: false, status: 'ready' } as never;
    if (url === '/jobs') return { items: [source], total: 1 } as never;
    if (url === '/jobs/live') return source as never;
    if (url === '/jobs/live/xyz/options') return options as never;
    if (url === '/jobs/live/xyz') return [] as never;
    if (url === '/queue/devices') return { devices: [{ device: 'cuda:0', name: 'BW', job_id: 'live', job_name: source.name, status: 'running' }, { device: 'cuda:1', name: 'BW', job_id: null, status: 'free' }], max_concurrent: null } as never;
    throw Error(`Unexpected ${url}`);
  });
  vi.spyOn(apiClient, 'post').mockResolvedValue({ id: 'xy1', status: 'queued', created_at: 1, done: 0, total: 3, can_cancel: true, request: { ...options.defaults, x: { key: 'steps', values: [15, 20, 25] }, gpu_devices: ['cuda:1'] }, manifest: { cells: [], grids: [] } });
});
afterEach(() => vi.restoreAllMocks());
function Location() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output>; }
function view(url = '/sampling') { render(<MemoryRouter initialEntries={[url]}><Sampling/><Location/></MemoryRouter>); }
it('opens a running training source on the independent page and schedules XY on GPU 1', async () => {
  view('/sampling?source_job_id=live&project_id=p1&version_id=v1');
  await screen.findByText('对比设置');
  expect(screen.getByRole('heading', { name: 'XY 对比' })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: '查看训练任务' })).toHaveAttribute('href', '/jobs/live?tab=metrics');
  fireEvent.click(screen.getByRole('combobox', { name: '运行显卡' }));
  fireEvent.click(screen.getByRole('option', { name: /GPU 1.*空闲/ }));
  fireEvent.click(screen.getByRole('button', { name: '生成对比图' }));
  await waitFor(() => expect(apiClient.post).toHaveBeenCalledWith('/jobs/live/xyz', expect.objectContaining({ gpu_devices: ['cuda:1'] }), { silent: true }));
  expect(screen.getByTestId('location')).toHaveTextContent('/sampling');
});
it('requires a source selection and keeps it in the page URL', async () => {
  view(); await screen.findByRole('combobox', { name: '来源训练任务' });
  expect(screen.queryByText('对比设置')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('combobox', { name: '来源训练任务' }));
  fireEvent.click(screen.getByRole('option', { name: /正在训练的 Krea2/ }));
  await screen.findByText('对比设置');
  expect(screen.getByTestId('location')).toHaveTextContent('source_job_id=live');
});
it('preserves exact source, result and project context in deep links', () => {
  expect(samplingUrl('run/one', 'xy?two', 'p1', 'v1')).toBe('/sampling?project_id=p1&version_id=v1&source_job_id=run%2Fone&task_id=xy%3Ftwo');
});
it('shows and clears a version filter without losing the selected source', async () => {
  view('/sampling?project_id=p1&version_id=v1&source_job_id=live');
  await screen.findByText('对比设置');
  expect(screen.getByText(/当前版本/)).toHaveTextContent('v1');
  fireEvent.click(screen.getByRole('button', { name: '清除版本筛选' }));
  await waitFor(() => expect(screen.getByTestId('location')).not.toHaveTextContent('version_id'));
  expect(screen.getByTestId('location')).toHaveTextContent('source_job_id=live');
  expect(screen.getByText('对比设置')).toBeInTheDocument();
});
it('reports and retries a failed project list while keeping the selected source usable', async () => {
  const original = vi.mocked(apiClient.get).getMockImplementation()!;
  let projectAttempts = 0;
  vi.mocked(apiClient.get).mockImplementation(async (url, config) => {
    if (url === '/projects' && projectAttempts++ === 0) throw new Error('连接暂时不可用');
    return original(url, config);
  });
  view('/sampling?source_job_id=live');
  await screen.findByText('对比设置');
  expect(screen.getByRole('alert')).toHaveTextContent('项目列表读取失败');
  expect(screen.getByRole('button', { name: '生成对比图' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: '重试项目列表' }));
  await waitFor(() => expect(screen.queryByRole('button', { name: '重试项目列表' })).not.toBeInTheDocument());
  expect(projectAttempts).toBe(2);
  expect(screen.getByText('对比设置')).toBeInTheDocument();
});

it('keeps an archived source readable but prevents generating new comparisons', async () => {
  const original = vi.mocked(apiClient.get).getMockImplementation()!;
  vi.mocked(apiClient.get).mockImplementation(async (url, config) => url === '/projects/p1/versions/v1' ? { archived: true, status: 'ready' } as never : original(url, config));
  view('/sampling?source_job_id=live');
  await screen.findByText('对比设置');
  await screen.findByText('来源项目或版本已归档；可以查看对比记录，恢复归档后才能生成。');
  expect(screen.getByRole('button', { name: '生成对比图' })).toBeDisabled();
  expect(apiClient.post).not.toHaveBeenCalled();
});
