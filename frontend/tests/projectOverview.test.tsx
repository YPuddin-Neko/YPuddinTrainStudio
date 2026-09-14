import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import ProjectOverview, { type OverviewDataset, type ProjectOverviewProps } from '../src/pages/ProjectDetail/ProjectOverview';
import DatasetPipelinePanel from '../src/components/datasets/DatasetPipelinePanel';
import { apiClient } from '../src/api/client';
import i18n from '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {} }));
const dataset = (id: string, images: number, captioned: number, masks: number, is_reg = false): OverviewDataset => ({
  source: { id, project_id: 'p1', version_id: 'v2', path: `/private/training/${id}`, is_reg } as OverviewDataset['source'],
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
function mockResponse(url: string, options?: any): any {
  if (url === '/config/defaults') return defaults;
  if (url === '/jobs') return emptyJobs;
  if (url === '/artifacts') return [];
  if (url.endsWith('/pipeline')) return { operations: [], busy: false, inspection: null };
  const id = url.match(/\/datasets\/(.+)\/overview$/)?.[1];
  if (!id) throw new Error(`Unexpected API request: ${url}`);
  const reg = id === 'reg';
  const scoped = !!options?.params?.folder;
  const count = scoped ? 3 : reg ? 4 : 12;
  const name = reg ? 'regularization' : scoped ? 'nested' : 'train';
  const item = { hash: `${id}-hash`, rel_path: `${scoped ? 'concept/nested' : 'concept'}/${name}.png`, width: 1024, height: 768, caption: `${name}, smile`, has_mask: false, caption_status: 'captioned' };
  return { dataset_id: id, folders: [{ path: 'concept', count }, { path: 'concept/nested', count: 3 }],
    stats: { images: count, captioned: count, masks: 0, resolutions: [{ w: 1024, h: 768, count }], ar_hist: [{ ar: '1.3', count }] },
    caption_stats: { images: count, captioned: count, missing: 0, invalid: 0, formats: { txt: count }, unique_tags: 2, tags: [{ tag: name, count }, { tag: 'smile', count: count - 1 }] },
    images: { items: options?.params?.q === 'missing' ? [] : [item], total: options?.params?.q === 'missing' ? 0 : count, page: 1, page_size: 16 } };
}
function show(patch: Partial<ProjectOverviewProps> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const element = (next: Partial<ProjectOverviewProps>) => <QueryClientProvider client={client}><MemoryRouter><ProjectOverview {...props} {...next}/></MemoryRouter></QueryClientProvider>;
  const result = render(element(patch));
  return { ...result, update: (next: Partial<ProjectOverviewProps>) => result.rerender(element(next)) };
}
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  vi.spyOn(apiClient, 'get').mockImplementation(async (url, options) => mockResponse(url, options));
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
    if (url !== '/config/defaults') return mockResponse(url);
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
  vi.mocked(apiClient.get).mockImplementation(async (url, options) => mockResponse(url, options));
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
    if (url !== '/jobs') return mockResponse(url);
    if (failed) { failed = false; throw new Error('History unavailable'); }
    return { items: [{ id: 'job-old', name: '第二版训练', status: 'failed', created_at: 1700000000, progress: { step: 27, total_steps: 400 } }], total: 1 } as any;
  });
  show();
  await waitFor(() => expect(screen.getAllByRole('alert').some(el => el.textContent?.includes('History unavailable'))).toBe(true));
  expect(screen.queryByText('这个版本还没有训练记录')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '重试' }));
  const job = await screen.findByRole('link', { name: /第二版训练.*查看原因/ });
  expect(job).toHaveAttribute('href', '/jobs/job-old');
  expect(job).toHaveTextContent('27 / 400');
  expect(job).toHaveTextContent('查看原因');
});


it('switches dataset roles without leaking training captions into regularization and links to the actual reg step', async () => {
  show();
  await screen.findByRole('button', { name: '预览图片：concept/train.png' });
  fireEvent.click(screen.getByRole('button', { name: /正则集 4/ }));
  await screen.findByRole('button', { name: '预览图片：concept/regularization.png' });
  expect(screen.queryByRole('button', { name: '预览图片：concept/train.png' })).not.toBeInTheDocument();
  expect(screen.getByRole('link', { name: '查看全部' })).toHaveAttribute('href', '/projects/p1/v/v2?step=data&data_step=reg#version-datasets');
  expect(apiClient.get).toHaveBeenCalledWith('/datasets/reg/overview', expect.objectContaining({ params: expect.objectContaining({ project_id: 'p1', version_id: 'v2' }) }));
});

it('selects a source and nested folder with matching full-scope statistics, while text search filters previews only', async () => {
  show();
  await screen.findByRole('button', { name: '预览图片：concept/train.png' });
  fireEvent.click(screen.getByRole('combobox', { name: '概览数据集' }));
  fireEvent.click(screen.getByRole('option', { name: 'train · 12 张' }));
  fireEvent.click(await screen.findByRole('combobox', { name: '概览子目录' }));
  fireEvent.click(screen.getByRole('option', { name: 'concept/nested · 3' }));
  await screen.findByRole('button', { name: '预览图片：concept/nested/nested.png' });
  expect(screen.getByText('3 / 3 张已标注 · 2 种标签')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: '查看全部' })).toHaveAttribute('href', '/datasets/train?project=p1&version=v2');
  fireEvent.change(screen.getByRole('textbox', { name: '筛选概览图片' }), { target: { value: 'missing' } });
  await screen.findByText('没有匹配的图片。');
  expect(screen.getByText('3 / 3 张已标注 · 2 种标签')).toBeInTheDocument();
  expect(apiClient.get).toHaveBeenCalledWith('/datasets/train/overview', expect.objectContaining({ params: expect.objectContaining({ folder: 'concept/nested', q: 'missing' }) }));
});

it('previews inside a dismissible dialog and preserves project/version context when opening the source', async () => {
  show();
  const image = await screen.findByRole('button', { name: '预览图片：concept/train.png' });
  image.focus(); fireEvent.click(image);
  const dialog = screen.getByRole('dialog', { name: '图片预览' });
  expect(within(dialog).getByRole('link', { name: '打开所属数据集' })).toHaveAttribute('href', '/datasets/train?project=p1&version=v2');
  expect(within(dialog).getByText('train, smile')).toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole('button', { name: '关闭' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  expect(image).toHaveFocus();
});

it('excludes explicitly foreign and legacy sources from a version overview and never fetches their data', async () => {
  const other = dataset('foreign', 500, 500, 0);
  other.source.version_id = 'other-version';
  const legacy = dataset('legacy', 100, 100, 0); legacy.source.version_id = null;
  show({ datasets: [props.datasets[0], other, legacy] });
  await screen.findByRole('button', { name: '预览图片：concept/train.png' });
  expect(within(screen.getByLabelText('当前版本数据统计')).getByRole('link', { name: /图片总数/ })).toHaveTextContent('12');
  expect(vi.mocked(apiClient.get).mock.calls.some(([url]) => url.includes('foreign') || url.includes('legacy'))).toBe(false);
});

it('shows active metrics and downloadable complete model outputs scoped to the version', async () => {
  vi.mocked(apiClient.get).mockImplementation(async (url, options: any) => {
    if (url === '/jobs' && options.params.group === 'active') return { ...emptyJobs, items: [{ id: 'running', project_id: 'p1', version_id: 'v2', name: '训练中', status: 'running', progress: { step: 12, total_steps: 100, epoch: 2, it_s: 1.2, eta_s: 70 }, latest: { loss: 0.1234, loss_mean: 0.2, loss_mean_scope: 'since_resume' } }] } as any;
    if (url === '/artifacts') return [{ id: 'model1', project_id: 'p1', version_id: 'v2', name: 'full-model', kind: 'model', size: 1024, step: 10, created_at: 1700000000 }, { id: 'leak', project_id: 'p1', version_id: 'v1', name: 'wrong-version', kind: 'model' }] as any;
    return mockResponse(url, options);
  });
  show();
  await screen.findByText('0.1234');
  expect(screen.getByText('恢复后平均损失')).toBeInTheDocument();
  expect(screen.getByText('1m 10s')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: '下载 full-model' })).toHaveAttribute('href', 'http://localhost:3000/api/artifacts/model1/download');
  expect(screen.getByText(/完整模型 · ZIP/)).toBeInTheDocument();
  expect(screen.queryByText('wrong-version')).not.toBeInTheDocument();
});

it('keeps malformed distribution responses distinct from empty datasets and provides recovery', async () => {
  vi.mocked(apiClient.get).mockImplementation(async (url, options) => url.endsWith('/overview') ? { dataset_id: 'train', caption_stats: {} } as any : mockResponse(url, options));
  show();
  expect(await screen.findByText('数据概览响应格式不完整，请重新读取。')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: '重新读取数据分布' })).toBeInTheDocument();
  expect(screen.queryByText('没有匹配的图片。')).not.toBeInTheDocument();
});
