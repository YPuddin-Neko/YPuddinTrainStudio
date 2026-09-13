import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import ProjectOverview, { type OverviewDataset, type ProjectOverviewProps } from '../src/pages/ProjectDetail/ProjectOverview';
import DatasetPipelinePanel from '../src/components/datasets/DatasetPipelinePanel';
import { apiClient } from '../src/api/client';
import i18n from '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {} }));
const dataset = (id: string, images: number, captioned: number, masks: number, is_reg = false): OverviewDataset => ({
  source: { id, path: `/private/training/${id}`, is_reg } as OverviewDataset['source'],
  stats: { images, captioned, masks }, index_status: 'ready',
});
const props: ProjectOverviewProps = {
  project: { id: 'p1', name: 'Project', note: '项目描述', version_count: 3 } as ProjectOverviewProps['project'],
  versionId: 'v2', version: { id: 'v2', name: '角色实验第二版', note: '本版调整了训练图片', updated_at: 1700000000 } as ProjectOverviewProps['version'],
  config: { model: { family: 'krea2', dit_path: '/private/models/krea-test.safetensors' }, adapter: { algo: 'lokr', rank: 16 }, dataset: { batch_size: 3 }, optimizer: { lr: 0.0001 }, loop: { epochs: 8, max_steps: 1200 } },
  datasets: [dataset('train', 12, 10, 5), dataset('reg', 4, 4, 0, true)],
};
const defaults = { model: { family: 'anima', dit_path: '' }, adapter: { algo: 'lora', rank: 11 }, dataset: { batch_size: 5 }, optimizer: { lr: 0.006 }, loop: { epochs: 23, max_steps: null } };
const emptyJobs = { items: [], total: 0, page: 1, page_size: 3 };
function show(patch: Partial<ProjectOverviewProps> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const element = (next: Partial<ProjectOverviewProps>) => <QueryClientProvider client={client}><MemoryRouter><ProjectOverview {...props} {...next}/></MemoryRouter></QueryClientProvider>;
  const result = render(element(patch));
  return { ...result, update: (next: Partial<ProjectOverviewProps>) => result.rerender(element(next)) };
}
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  vi.spyOn(apiClient, 'get').mockImplementation(async url => url === '/config/defaults' ? defaults : emptyJobs as any);
});
afterEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); });

it('presents actual scoped data and saved parameters without exposing storage paths', async () => {
  show();
  await screen.findByText('这个版本还没有训练记录');
  expect(screen.getByRole('heading', { name: '角色实验第二版' })).toBeInTheDocument();
  expect(screen.getByText('本版调整了训练图片')).toBeInTheDocument();
  const metrics = screen.getByLabelText('当前版本数据统计');
  for (const [label, value] of [['图片总数', '16'], ['标签文件', '14'], ['遮罩文件', '5'], ['数据来源', '2']]) {
    expect(within(metrics).getByRole('link', { name: new RegExp(label) })).toHaveTextContent(value);
  }
  expect(screen.getByText('2 张图片没有标签文件')).toBeInTheDocument();
  expect(screen.getByText('krea-test.safetensors')).toBeInTheDocument();
  expect(screen.getByText('批量大小').nextElementSibling).toHaveTextContent('3');
  expect(screen.getByText('训练轮数').nextElementSibling).toHaveTextContent('8');
  expect(screen.getByTestId('project-overview').textContent).not.toContain('/private/');
  expect(apiClient.get).toHaveBeenCalledWith('/jobs', { params: { project_id: 'p1', version_id: 'v2', type: 'train', page: 1, page_size: 3 }, silent: true });
  expect(screen.getByRole('link', { name: '查看版本结果' })).toHaveAttribute('href', '/projects/p1/v/v2?step=results');
  expect(screen.getByRole('link', { name: '检查训练参数' })).toHaveAttribute('href', '/projects/p1/v/v2/train');
  expect(within(metrics).getByRole('link', { name: /标签文件/ })).toHaveAttribute('href', '/projects/p1/v/v2?step=data&data_step=captions');
  expect(within(metrics).getByRole('link', { name: /遮罩文件/ })).toHaveAttribute('href', '/projects/p1/v/v2?step=data&data_step=paint');
});

it('uses server defaults for sparse saved configs while preserving explicit overrides and null limits', async () => {
  const page = show({ config: { adapter: { rank: 7 }, loop: { max_steps: 400 } } });
  await screen.findByText('23');
  expect(screen.getByText('Rank').nextElementSibling).toHaveTextContent('7');
  expect(screen.getByText('批量大小').nextElementSibling).toHaveTextContent('5');
  expect(screen.getByText('学习率').nextElementSibling).toHaveTextContent('0.006');
  expect(screen.getByText('最大步数').nextElementSibling).toHaveTextContent('400');
  page.update({ versionId: 'v3', config: { loop: { epochs: null } } });
  expect(screen.getByText('Rank').nextElementSibling).toHaveTextContent('11');
  expect(screen.getByText('训练轮数').nextElementSibling).toHaveTextContent('未设置');
  expect(vi.mocked(apiClient.get).mock.calls.filter(([url]) => url === '/config/defaults')).toHaveLength(1);
});

it('does not call unknown defaults unset after a failed defaults request and can recover', async () => {
  let failed = true;
  vi.mocked(apiClient.get).mockImplementation(async url => {
    if (url !== '/config/defaults') return emptyJobs as any;
    if (failed) throw new Error('Defaults unavailable');
    return defaults as any;
  });
  show({ config: {} });
  expect(await screen.findByRole('alert')).toHaveTextContent('默认参数读取失败');
  expect(screen.getByText('训练轮数').nextElementSibling).toHaveTextContent('—');
  failed = false;
  fireEvent.click(screen.getByRole('button', { name: '重新读取默认参数' }));
  await screen.findByText('23');
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it('opens the dataset library from Sources even after a different pipeline stage was remembered', async () => {
  sessionStorage.setItem('studio.pipeline.stage.p1.v2', 'prepare');
  vi.mocked(apiClient.get).mockImplementation(async url => url === '/config/defaults' ? defaults : url.endsWith('/pipeline') ? { operations: [], busy: false, inspection: null } : emptyJobs as any);
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter initialEntries={['/']}><Routes>
    <Route path="/" element={<ProjectOverview {...props}/>}/>
    <Route path="/projects/p1/v/v2" element={<DatasetPipelinePanel projectId="p1" versionId="v2" readOnly importPanel={null} datasetList={<p>Version dataset library</p>} onChanged={() => {}}/>}/>
  </Routes></MemoryRouter></QueryClientProvider>);
  fireEvent.click(screen.getByRole('link', { name: /数据来源/ }));
  expect(await screen.findByText('Version dataset library')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: '导入' })).toHaveAttribute('aria-current', 'step');
  expect(screen.queryByText('缓存里保存什么？')).not.toBeInTheDocument();
});

it('does not represent missing or unfinished indexes as zero images or complete preparation', async () => {
  show({ datasets: [{ ...dataset('pending', 10, 10, 10), index_status: 'indexing' }, { source: dataset('missing', 1, 1, 1).source }] });
  await screen.findByText('这个版本还没有训练记录');
  const metrics = screen.getByLabelText('当前版本数据统计');
  expect(within(metrics).getByRole('link', { name: /图片总数/ })).toHaveTextContent('—');
  expect(within(metrics).getAllByText('待索引')).toHaveLength(3);
  expect(screen.queryByText('所有图片都有标签文件')).not.toBeInTheDocument();
  expect(screen.getByText('正在索引')).toBeInTheDocument();
});

it('offers model configuration only after the version has indexed training images', async () => {
  show({ config: { model: { family: 'anima' } } });
  await screen.findByText('这个版本还没有训练记录');
  expect(screen.getByRole('link', { name: '配置训练模型' })).toHaveAttribute('href', '/projects/p1/v/v2/train?tab=model');
  expect(screen.queryByText('已配置')).not.toBeInTheDocument();
});

it('keeps failed job loading distinct from a version with no runs and supports retry', async () => {
  let failed = true;
  vi.mocked(apiClient.get).mockImplementation(async url => {
    if (url === '/config/defaults') return defaults as any;
    if (failed) { failed = false; throw new Error('History unavailable'); }
    return { items: [{ id: 'job-old', name: '第二版训练', status: 'failed', created_at: 1700000000, progress: { step: 27, total_steps: 400 } }], total: 1 } as any;
  });
  show();
  expect(await screen.findByRole('alert')).toHaveTextContent('History unavailable');
  expect(screen.queryByText('这个版本还没有训练记录')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '重试' }));
  const job = await screen.findByRole('link', { name: /第二版训练/ });
  expect(job).toHaveAttribute('href', '/jobs/job-old');
  expect(job).toHaveTextContent('27 / 400');
  expect(job).toHaveTextContent('查看原因');
});
