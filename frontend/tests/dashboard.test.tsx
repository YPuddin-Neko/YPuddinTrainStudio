import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { it, expect, beforeEach, afterEach, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import Dashboard from '../src/pages/Dashboard/Dashboard';
import { apiClient } from '../src/api/client';
import { mockJobs } from '../src/mocks/mockStore';
import i18n from '../src/i18n';
vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {} }));
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  vi.spyOn(apiClient, 'get').mockImplementation(async (url, options) => {
    if (url === '/projects') return [{ id: 'hero', name: '服装实验', active_version_id: 'v_blue', version_count: 2, updated_at: 10 }] as any;
    if (url === '/jobs') return { items: options?.params?.group === 'active' ? [{ ...mockJobs[0], id: 'live', name: 'Current training', project_id: 'hero', version_id: 'v_blue' }] : [], total: options?.params?.group === 'waiting' ? 8 : 1 } as any;
    throw Error(`Unexpected GET ${url}`);
  });
});
afterEach(() => vi.restoreAllMocks());
it('guides users back to the exact project version and the focused queue without another queue table', async () => {
  render(<MemoryRouter><Dashboard/></MemoryRouter>);
  await screen.findByRole('link', { name: 'Current training' });
  expect(screen.getByRole('link', { name: /准备训练数据/ })).toHaveAttribute('href', '/projects/hero/v/v_blue?step=data');
  expect(screen.getByRole('link', { name: /检查参数并启动/ })).toHaveAttribute('href', '/projects/hero/v/v_blue/train');
  expect(screen.getByRole('link', { name: /检查训练结果/ })).toHaveAttribute('href', '/projects/hero/v/v_blue?step=results');
  expect(screen.getByRole('link', { name: /8.*等待调度/ })).toHaveAttribute('href', '/queue?view=waiting');
  expect(screen.queryByRole('table')).not.toBeInTheDocument();
  expect(vi.mocked(apiClient.get).mock.calls.every(([url]) => url === '/jobs' || url === '/projects')).toBe(true);
});
it('offers project creation when no projects or active jobs exist', async () => {
  vi.mocked(apiClient.get).mockImplementation(async url => (url === '/projects' ? [] : { items: [], total: 0 }) as any);
  render(<MemoryRouter><Dashboard/></MemoryRouter>);
  await waitFor(() => expect(screen.getByText('现在没有运行或暂停的任务')).toBeInTheDocument());
  expect(screen.getByRole('link', { name: '创建项目与版本' })).toHaveAttribute('href', '/projects');
});

it('keeps projects and running jobs when one statistic fails, shows an unknown count and clears the error after retry', async () => {
  const get = vi.mocked(apiClient.get).getMockImplementation()!;
  let failed = true;
  vi.mocked(apiClient.get).mockImplementation((url, options) => url === '/jobs' && options?.params?.group === 'history' && failed ? Promise.reject(new Error('History temporarily unavailable')) : get(url, options));
  render(<MemoryRouter><Dashboard/></MemoryRouter>);
  await screen.findByRole('link', { name: 'Current training' });
  expect(screen.getByRole('link', { name: /准备训练数据/ })).toHaveAttribute('href', '/projects/hero/v/v_blue?step=data');
  const history = within(screen.getByLabelText('任务总览')).getByRole('link', { name: /历史记录/ });
  expect(history).toHaveTextContent('—');
  expect(screen.getByRole('alert')).toHaveTextContent('历史记录: History temporarily unavailable');
  expect(screen.getByRole('link', { name: /8.*等待调度/ })).toBeInTheDocument();
  failed = false;
  fireEvent.click(screen.getByRole('button', { name: '重试' }));
  await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
  expect(history).toHaveTextContent('1');
});

it.each(['projects', 'active'])('does not report unavailable %s as an empty collection', async unavailable => {
  const get = vi.mocked(apiClient.get).getMockImplementation()!;
  vi.mocked(apiClient.get).mockImplementation((url, options) => (unavailable === 'projects' ? url === '/projects' : options?.params?.group === 'active') ? Promise.reject(new Error('Temporarily unavailable')) : get(url, options));
  render(<MemoryRouter><Dashboard/></MemoryRouter>);
  await screen.findByRole('alert');
  if (unavailable === 'projects') {
    expect(screen.getByText('项目列表暂时无法读取')).toBeInTheDocument();
    expect(screen.queryByText('从第一个项目开始')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Current training' })).toBeInTheDocument();
  } else {
    expect(screen.getByText('当前任务暂时无法读取')).toBeInTheDocument();
    expect(screen.queryByText('现在没有运行或暂停的任务')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: /准备训练数据/ })).toHaveAttribute('href', '/projects/hero/v/v_blue?step=data');
  }
});
