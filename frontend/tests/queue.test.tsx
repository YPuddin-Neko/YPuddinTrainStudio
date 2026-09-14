import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import { it, expect, beforeEach, afterEach, vi } from 'vitest';
import { MemoryRouter, useLocation } from 'react-router-dom';
import Queue from '../src/pages/Queue/Queue';
import { apiClient } from '../src/api/client';
import { mockJobs } from '../src/mocks/mockStore';
import i18n from '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {} }));
let jobs: any[];
let settings: { held: boolean; max_concurrent: number; memory_admission: boolean };
const make = (id: string, status: string, extra = {}) => ({ ...mockJobs[0], id, name: `Run ${id}`, status, project_id: 'p1', project_name: '衣装实验', version_id: 'v_original', version_name: '蓝色衣装', version_number: 2, type: 'train', ...extra });
function Location() { const location = useLocation(); return <output data-testid="location">{location.search}</output>; }
const view = (url = '/queue') => <MemoryRouter initialEntries={[url]}><Queue/><Location/></MemoryRouter>;
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  jobs = [make('live', 'running'), make('waiting', 'queued'), ...Array.from({ length: 24 }, (_, i) => make(`old${i}`, 'completed'))];
  settings = { held: false, max_concurrent: 1, memory_admission: true };
  vi.spyOn(apiClient, 'get').mockImplementation(async (url, options) => {
    if (url === '/projects') return [{ id: 'p1', name: '衣装实验' }] as any;
    if (url === '/queue/settings') return settings as any;
    if (url === '/jobs') {
      const params = options?.params || {};
      const groups: Record<string, string[]> = { active: ['running', 'paused', 'pausing', 'cancelling'], waiting: ['queued', 'scheduled'], history: ['completed', 'failed', 'cancelled'] };
      const rows = jobs.filter(job => (!params.group || groups[String(params.group)]?.includes(job.status)) && (!params.status || job.status === params.status) && (!params.project_id || job.project_id === params.project_id) && (!params.type || job.type === params.type) && (!params.q || job.name.includes(String(params.q))));
      const page = Number(params.page || 1), size = Number(params.page_size || 20);
      return { items: rows.slice((page - 1) * size, page * size), total: rows.length, page, page_size: size } as any;
    }
    throw Error(`Unexpected GET ${url}`);
  });
  vi.spyOn(apiClient, 'post').mockImplementation(async url => { const id = url.split('/')[2]; const index = jobs.findIndex(job => job.id === id); jobs[index] = { ...jobs[index], status: 'paused' }; return jobs[index] as any; });
  vi.spyOn(apiClient, 'put').mockImplementation(async (_url, patch) => { settings = { ...settings, ...(patch as any) }; return settings as any; });
  vi.spyOn(apiClient, 'patch').mockResolvedValue({} as any);
});
afterEach(() => vi.restoreAllMocks());

it('separates running jobs from history, links their exact version and pauses through the API', async () => {
  jobs[0] = { ...jobs[0], progress: { phase: 'loading' } };
  render(view());
  const row = await screen.findByTestId('job-row-live');
  expect(row).toHaveTextContent('加载权重');
  expect(screen.queryByTestId('job-row-waiting')).not.toBeInTheDocument();
  expect(within(row).getByRole('link', { name: /衣装实验/ })).toHaveAttribute('href', '/projects/p1/v/v_original?step=results');
  expect(row).toHaveTextContent('v2 · 蓝色衣装');
  fireEvent.click(within(row).getByRole('button', { name: '暂停' }));
  await waitFor(() => expect(apiClient.post).toHaveBeenCalledWith('/jobs/live/pause', {}, { silent: true }));
  await within(row).findByRole('button', { name: '继续' });
  expect(document.querySelector('select')).toBeNull();
});

it('persists history filters in the URL and asks the server for the second filtered page', async () => {
  render(view('/queue?view=history'));
  await screen.findByTestId('job-row-old0');
  fireEvent.click(screen.getByRole('combobox', { name: '项目筛选' })); fireEvent.click(screen.getByRole('option', { name: '衣装实验' }));
  fireEvent.click(screen.getByRole('combobox', { name: '任务类型' })); fireEvent.click(screen.getByRole('option', { name: '训练' }));
  fireEvent.change(screen.getByRole('textbox', { name: '搜索任务' }), { target: { value: 'Run old' } });
  await waitFor(() => expect(screen.getByRole('button', { name: '下一页' })).toBeEnabled());
  fireEvent.click(within(screen.getByTestId('queue-controls')).getByRole('button', { name: '下一页' }));
  await screen.findByTestId('job-row-old20');
  expect(screen.queryByTestId('job-row-old0')).not.toBeInTheDocument();
  expect(apiClient.get).toHaveBeenCalledWith('/jobs', expect.objectContaining({ params: expect.objectContaining({ group: 'history', type: 'train', project_id: 'p1', q: 'Run old', page: 2, page_size: 20 }) }));
  expect(screen.getByTestId('location')).toHaveTextContent('page=2');
  expect(screen.queryByRole('spinbutton', { name: /优先级/ })).not.toBeInTheDocument();
});

it('edits only a waiting job priority and explains that holding scheduling does not pause running jobs', async () => {
  jobs[1] = { ...jobs[1], progress: { phase: 'waiting_for_device', wait_reason: 'waiting for a free accelerator with enough memory' } };
  render(view('/queue?view=waiting'));
  const input = await screen.findByRole('spinbutton', { name: '优先级: Run waiting' });
  expect(screen.getByTestId('job-row-waiting')).toHaveTextContent('等待空闲且显存充足的显卡');
  fireEvent.change(input, { target: { value: '42' } }); fireEvent.blur(input);
  await waitFor(() => expect(apiClient.patch).toHaveBeenCalledWith('/jobs/waiting', { priority: 42 }, { silent: true }));
  fireEvent.click(screen.getByTestId('toggle-held'));
  expect(await screen.findByText(/调度已暂停：正在运行的任务会继续/)).toBeInTheDocument();
  expect(screen.getByTestId('toggle-held')).toHaveTextContent('继续调度');
});

it('shows an actionable error when a task action is rejected and leaves its state unchanged', async () => {
  vi.mocked(apiClient.post).mockRejectedValue(new Error('Version is archived'));
  render(view()); const row = await screen.findByTestId('job-row-live');
  fireEvent.click(within(row).getByRole('button', { name: '暂停' }));
  expect(await within(row).findByRole('alert')).toHaveTextContent('Version is archived');
  expect(within(row).getByRole('button', { name: '暂停' })).toBeEnabled();
  expect(row).toHaveTextContent('运行中');
});

it('switches queue views with the keyboard and gives the selected tab a matching panel', async () => {
  render(view()); await screen.findByTestId('job-row-live');
  const current = screen.getByRole('tab', { name: /运行与暂停/ }); current.focus();
  fireEvent.keyDown(current, { key: 'ArrowRight' });
  await screen.findByTestId('job-row-waiting');
  expect(screen.getByRole('tab', { name: /等待调度/ })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', 'queue-tab-waiting');
  expect(screen.getByTestId('location')).toHaveTextContent('view=waiting');
});

it('does not repeat an unchanged version number as its display name', async () => {
  jobs = [make('live', 'running', { version_name: 'v1', version_number: 1 })];
  render(view());
  const row = await screen.findByTestId('job-row-live');
  const context = within(row).getByRole('link', { name: /衣装实验/ });
  expect(context).toHaveTextContent('v1');
  expect(context).not.toHaveTextContent('v1 · v1');
  expect(within(row).getByRole('button', { name: '暂停' })).toBeEnabled();
  expect(within(row).getByRole('button', { name: '保存检查点' })).toBeEnabled();
});
