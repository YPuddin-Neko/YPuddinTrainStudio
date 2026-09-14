import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, useLocation } from 'react-router-dom';
import VersionResults from '../src/components/projects/VersionResults';
import Artifacts from '../src/pages/Artifacts/Artifacts';
import { apiClient } from '../src/api/client';
import { mockJobs } from '../src/mocks/mockStore';
import type { JobSample } from '../src/api/types';
import { ApiError } from '../src/api/types';
import i18n from '../src/i18n';

const listeners = vi.hoisted(() => new Map<string, Set<(data: any) => void>>());
vi.mock('../src/events/useEventStream', async () => {
  const { useEffect, useRef } = await import('react');
  return { useEventStream: (type: string, callback: (data: any) => void) => {
    const ref = useRef(callback); ref.current = callback;
    useEffect(() => { const current = (data: any) => ref.current(data); if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type)!.add(current); return () => { listeners.get(type)?.delete(current); }; }, [type]);
  } };
});
const emit = (type: string, data: any) => act(() => listeners.get(type)?.forEach(listener => listener(data)));
const job = (id: string, version = 'v1') => ({ ...mockJobs[0], id, name: `Run ${id}`, project_id: 'p1', version_id: version, type: 'train', status: 'running', created_at: 1700000000, progress: { step: 1, total_steps: 10 }, latest: { loss: 0.5 } });
const sample = (id: string, step = 10): JobSample => ({ step, loss: null, prompt_index: 0, prompt: `${id} prompt`, seed: 7, width: 640, height: 480, created_at: 1700000010, url: `/api/jobs/${id}/files?path=sample-${step}.png&kind=sample` });
const artifact = (id: string, version = 'v1', jobId = 'j1') => ({ id, project_id: 'p1', version_id: version, job_id: jobId, name: `${id}.safetensors`, path: `D:/outputs/${id}.safetensors`, kind: 'weights', created_at: 1700000010, step: 10, size: 512, metadata: { version } });
let jobs: ReturnType<typeof job>[];
let artifacts: ReturnType<typeof artifact>[];
let samples: Record<string, JobSample[]>;
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN'); jobs = [job('j1'), job('j2')]; artifacts = [artifact('a1'), artifact('foreign', 'v2')]; samples = { j1: [sample('j1')], j2: [sample('j2', 20)] };
  vi.spyOn(apiClient, 'get').mockImplementation(async (endpoint, options) => {
    if (endpoint === '/jobs') return { items: jobs.filter(item => !options?.params?.version_id || item.version_id === options.params.version_id), total: jobs.length, page: options?.params?.page || 1, page_size: 50, type: 'train' } as any;
    if (endpoint === '/artifacts') return artifacts as any;
    const match = endpoint.match(/^\/jobs\/([^/]+)\/samples$/);
    if (match) return (samples[match[1]] || []) as any;
    throw new Error(`Unexpected GET ${endpoint}`);
  });
  vi.spyOn(apiClient, 'post').mockImplementation(async endpoint => { if (endpoint === '/artifacts/a1/convert') { const output = artifact('converted'); artifacts.push(output); return output as any; } throw new Error(`Unexpected POST ${endpoint}`); });
});
afterEach(() => { vi.restoreAllMocks(); });
const view = (versionId = 'v1') => <MemoryRouter initialEntries={['/projects/p1?result_tab=jobs']}><VersionResults projectId="p1" versionId={versionId}/></MemoryRouter>;
function Location() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output>; }

describe('version result workspace', () => {
  it('requests the current version with pagination and merges only its visible jobs from events', async () => {
    render(view());
    const row = await screen.findByTestId('result-job-j1');
    expect(apiClient.get).toHaveBeenCalledWith('/jobs', expect.objectContaining({ params: { project_id: 'p1', version_id: 'v1', page: 1, page_size: 50, type: 'train' } }));
    expect(vi.mocked(apiClient.get).mock.calls.some(([url]) => url.endsWith('/samples'))).toBe(false);
    expect(vi.mocked(apiClient.get).mock.calls.some(([url]) => url === '/artifacts')).toBe(false);
    emit('job.step', { job_id: 'j1', step: 8, loss: 0.125 });
    expect(row).toHaveTextContent('8 / 10'); expect(row).toHaveTextContent('0.1250');
    emit('job.step', { job_id: 'foreign', step: 999 });
    expect(screen.queryByText(/999/)).not.toBeInTheDocument();
    jobs[0].status = 'completed'; emit('job.state', { job_id: 'j1', status: 'completed' });
    expect(row).toHaveTextContent(i18n.t('queue.status.completed'));
  });

  it('loads samples only for the selected job, refreshes that job on sample events, and opens a real image', async () => {
    render(view()); await screen.findByText('Run j1');
    fireEvent.click(within(screen.getByTestId('result-job-j1')).getByRole('button', { name: '查看采样图' }));
    const image = await screen.findByRole('img', { name: 'j1 prompt' });
    expect(image).toHaveAttribute('src', 'http://localhost:3000/api/jobs/j1/files?path=sample-10.png&kind=sample');
    expect(vi.mocked(apiClient.get).mock.calls.filter(([url]) => url.endsWith('/samples')).map(([url]) => url)).toEqual(['/jobs/j1/samples']);
    fireEvent.click(screen.getByRole('combobox', { name: '采样所属任务' })); fireEvent.click(screen.getByRole('option', { name: /Run j2/ }));
    await screen.findByRole('img', { name: 'j2 prompt' });
    expect(screen.queryByRole('img', { name: 'j1 prompt' })).not.toBeInTheDocument();
    const before = vi.mocked(apiClient.get).mock.calls.length;
    emit('job.sample', { job_id: 'j1' }); expect(apiClient.get).toHaveBeenCalledTimes(before);
    samples.j2.push(sample('j2', 30)); emit('job.sample', { job_id: 'j2' });
    const preview = await screen.findByRole('button', { name: '查看采样图：第 30 步，提示词 1' });
    fireEvent.click(preview);
    const dialog = screen.getByRole('dialog', { name: '采样图预览' });
    expect(dialog).toHaveTextContent('Run j2'); expect(dialog).toHaveTextContent('640 × 480');
    expect(within(dialog).getByRole('link', { name: '下载' })).toHaveAttribute('href', expect.stringContaining('/jobs/j2/files?path=sample-30.png'));
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument(); expect(preview).toHaveFocus();
  });

  it('keeps the existing gallery visible while a new sample event is refreshing it', async () => {
    render(view()); await screen.findByText('Run j1');
    fireEvent.click(screen.getByRole('tab', { name: '采样图' }));
    const image = await screen.findByRole('img', { name: 'j1 prompt' });
    const original = vi.mocked(apiClient.get).getMockImplementation()!;
    let finish: (value: JobSample[]) => void = () => {};
    vi.mocked(apiClient.get).mockImplementation((url, options) => url === '/jobs/j1/samples' ? new Promise(resolve => { finish = resolve as (value: JobSample[]) => void; }) : original(url, options));
    emit('job.sample', { job_id: 'j1' });
    expect(image).toBeInTheDocument();
    expect(screen.getByRole('button', { name: i18n.t('common.refresh') })).toBeDisabled();
    await act(async () => finish([sample('j1'), sample('j1', 20)]));
    expect(screen.getAllByRole('img', { name: 'j1 prompt' })).toHaveLength(2);
  });

  it('ignores a late sample response after switching jobs and resets all selections on version change', async () => {
    const original = vi.mocked(apiClient.get).getMockImplementation()!;
    let finish: (value: JobSample[]) => void = () => {};
    vi.mocked(apiClient.get).mockImplementation((url, options) => url === '/jobs/j1/samples' ? new Promise(resolve => { finish = resolve as (value: JobSample[]) => void; }) : original(url, options));
    const { rerender } = render(view()); await screen.findByText('Run j1');
    fireEvent.click(screen.getByRole('tab', { name: '采样图' }));
    await waitFor(() => expect(apiClient.get).toHaveBeenCalledWith('/jobs/j1/samples', expect.anything()));
    fireEvent.click(screen.getByRole('combobox', { name: '采样所属任务' })); fireEvent.click(screen.getByRole('option', { name: /Run j2/ }));
    await screen.findByRole('img', { name: 'j2 prompt' });
    await act(async () => finish([sample('j1')]));
    expect(screen.queryByRole('img', { name: 'j1 prompt' })).not.toBeInTheDocument();
    jobs = [job('new-version-job', 'v2')]; rerender(view('v2'));
    fireEvent.click(screen.getByRole('tab', { name: '训练记录' }));
    await screen.findByText('Run new-version-job');
    expect(screen.getByRole('tab', { name: '训练记录' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(apiClient.get).toHaveBeenLastCalledWith('/jobs', expect.objectContaining({ params: { project_id: 'p1', version_id: 'v2', page: 1, page_size: 50, type: 'train' } }));
  });

  it('lets the sample selector move through paginated jobs rather than fetching every job', async () => {
    const original = vi.mocked(apiClient.get).getMockImplementation()!;
    vi.mocked(apiClient.get).mockImplementation(async (url, options) => url === '/jobs' ? { items: [job(options?.params?.page === 2 ? 'j2' : 'j1')], total: 51, page: options?.params?.page, page_size: 50, type: 'train' } as any : original(url, options));
    render(view()); await screen.findByText('Run j1');
    fireEvent.click(screen.getByRole('tab', { name: '采样图' }));
    await screen.findByRole('img', { name: 'j1 prompt' });
    fireEvent.click(within(screen.getByRole('combobox', { name: '采样所属任务' }).closest<HTMLElement>('.results-sample-toolbar')!).getByRole('button', { name: '下一页任务' }));
    await screen.findByRole('img', { name: 'j2 prompt' });
    expect(apiClient.get).toHaveBeenCalledWith('/jobs', expect.objectContaining({ params: { project_id: 'p1', version_id: 'v1', page: 2, page_size: 50, type: 'train' } }));
    expect(screen.getByRole('combobox', { name: '采样所属任务' })).toHaveTextContent('Run j2');
  });

  it('pages a large sample gallery from its top controls and hides unnecessary single-page task navigation', async () => {
    samples.j1 = Array.from({ length: 49 }, (_, step) => ({ ...sample('j1', step), prompt: `sample ${step}` }));
    render(view()); await screen.findByText('Run j1');
    fireEvent.click(screen.getByRole('tab', { name: '采样图' }));
    await screen.findByRole('img', { name: 'sample 48' });
    const controls = screen.getByRole('combobox', { name: '采样所属任务' }).closest<HTMLElement>('.results-selection-controls')!;
    expect(screen.queryByRole('button', { name: '下一页任务' })).not.toBeInTheDocument();
    expect(screen.getAllByRole('img')).toHaveLength(24);
    fireEvent.click(within(controls).getByRole('button', { name: '下一页采样图' }));
    expect(screen.getByRole('img', { name: 'sample 24' })).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: 'sample 48' })).not.toBeInTheDocument();
    fireEvent.click(within(controls).getByRole('button', { name: '下一页采样图' }));
    expect(screen.getAllByRole('img')).toHaveLength(1);
    expect(screen.getByRole('link', { name: '下载第 0 步采样图' })).toHaveAttribute('href', expect.stringContaining('/jobs/j1/files?path=sample-0.png'));
    fireEvent.click(screen.getByRole('combobox', { name: '采样所属任务' })); fireEvent.click(screen.getByRole('option', { name: /Run j2/ }));
    await screen.findByRole('img', { name: 'j2 prompt' });
    expect(screen.queryByRole('button', { name: '下一页采样图' })).not.toBeInTheDocument();
  });

  it('recovers from job and selected sample request failures', async () => {
    const original = vi.mocked(apiClient.get).getMockImplementation()!;
    vi.mocked(apiClient.get).mockRejectedValueOnce(new Error('Jobs offline'));
    render(view()); expect(await screen.findByRole('alert')).toHaveTextContent('Jobs offline');
    fireEvent.click(screen.getByRole('button', { name: '重试' })); await screen.findByText('Run j1');
    vi.mocked(apiClient.get).mockImplementation((url, options) => url === '/jobs/j1/samples' ? Promise.reject(new Error('Samples offline')) : original(url, options));
    fireEvent.click(screen.getByRole('tab', { name: '采样图' })); expect(await screen.findByRole('alert')).toHaveTextContent('Samples offline');
    vi.mocked(apiClient.get).mockImplementation(original);
    fireEvent.click(screen.getByRole('button', { name: '重试' })); await screen.findByRole('img', { name: 'j1 prompt' });
  });
});

describe('embedded output scope', () => {
  it('keeps embedded search, sorting, counts and refresh together without a duplicate title', async () => {
    artifacts = [{ ...artifact('recent'), step: 10, created_at: 1700000020 }, { ...artifact('advanced'), step: 30 }];
    render(<MemoryRouter><Artifacts embedded projectId="p1" versionId="v1"/></MemoryRouter>);
    await screen.findByTestId('artifact-row-recent');
    expect(screen.queryByRole('heading')).not.toBeInTheDocument();
    const controls = screen.getByRole('textbox', { name: '搜索训练产物' }).closest<HTMLElement>('.artifact-controls')!;
    expect(controls).toHaveTextContent('2 个产物');
    fireEvent.click(within(controls).getByRole('combobox', { name: '产物排序' }));
    fireEvent.click(screen.getByRole('option', { name: '训练步数降序' }));
    expect(screen.getAllByTestId(/^artifact-row-/)[0]).toHaveAttribute('data-testid', 'artifact-row-advanced');
    fireEvent.change(within(controls).getByRole('textbox', { name: '搜索训练产物' }), { target: { value: 'advanced' } });
    expect(controls).toHaveTextContent('1 个产物');
    fireEvent.click(within(controls).getByRole('button', { name: '刷新产物' }));
    await screen.findByTestId('artifact-row-advanced');
    expect(within(controls).getByRole('textbox', { name: '搜索训练产物' })).toHaveValue('advanced');
    expect(within(controls).getByRole('combobox', { name: '产物排序' })).toHaveTextContent('训练步数降序');
    expect(screen.queryByTestId('artifact-row-recent')).not.toBeInTheDocument();
    fireEvent.click(within(controls).getByRole('button', { name: '清除筛选' }));
    expect(screen.getAllByTestId(/^artifact-row-/)).toHaveLength(2);
    expect(apiClient.get).toHaveBeenCalledTimes(2);
  });

  it('pages weights using the top filter controls and keeps download actions on the selected page', async () => {
    artifacts = Array.from({ length: 26 }, (_, index) => artifact(`output-${index}`));
    render(<MemoryRouter><Artifacts embedded projectId="p1" versionId="v1"/></MemoryRouter>);
    await screen.findByTestId('artifact-row-output-0');
    expect(screen.getAllByTestId(/^artifact-row-/)).toHaveLength(25);
    const controls = screen.getByRole('textbox', { name: '搜索训练产物' }).closest<HTMLElement>('.artifact-controls')!;
    fireEvent.click(within(controls).getByRole('button', { name: '下一页产物' }));
    expect(screen.getAllByTestId(/^artifact-row-/)).toHaveLength(1);
    expect(screen.getByRole('link', { name: '下载: output-25.safetensors' })).toHaveAttribute('href', 'http://localhost:3000/api/artifacts/output-25/download');
    fireEvent.change(screen.getByRole('textbox', { name: '搜索训练产物' }), { target: { value: 'output-0.' } });
    expect(screen.getByTestId('artifact-row-output-0')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '下一页产物' })).not.toBeInTheDocument();
  });
  it('locks props against URL filters and keeps scope through search clearing, conversion and download', async () => {
    artifacts.push(artifact('other-job', 'v1', 'j2'));
    const path = '/projects/p1/v/v1?step=results&project_id=p2&version_id=v2&job_id=alien&q=wrong';
    render(<MemoryRouter initialEntries={[path]}><Artifacts embedded projectId="p1" versionId="v1" jobId="j1"/><Location/></MemoryRouter>);
    await screen.findByTestId('artifact-row-a1');
    expect(screen.queryByTestId('artifact-row-foreign')).not.toBeInTheDocument();
    expect(screen.queryByTestId('artifact-row-other-job')).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: '搜索训练产物' })).toHaveValue('');
    expect(apiClient.get).toHaveBeenCalledWith('/artifacts', expect.objectContaining({ params: { project_id: 'p1', version_id: 'v1', job_id: 'j1' } }));
    fireEvent.change(screen.getByRole('textbox', { name: '搜索训练产物' }), { target: { value: 'missing' } });
    expect(screen.queryByTestId('artifact-row-a1')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '清除筛选' }));
    expect(screen.getByTestId('location')).toHaveTextContent(path);
    expect(screen.getByTestId('artifact-row-a1')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '查看元数据: a1.safetensors' }));
    expect(screen.getByRole('dialog')).toHaveTextContent('v1'); fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
    fireEvent.click(screen.getByRole('combobox', { name: '转换格式: a1.safetensors' })); fireEvent.click(screen.getByRole('option', { name: 'kohya' }));
    await screen.findByTestId('artifact-row-converted');
    expect(apiClient.post).toHaveBeenCalledWith('/artifacts/a1/convert', { format: 'kohya' }, { silent: true });
    expect(screen.getByRole('link', { name: '下载: converted.safetensors' })).toHaveAttribute('href', 'http://localhost:3000/api/artifacts/converted/download');
    expect(screen.queryByTestId('artifact-row-foreign')).not.toBeInTheDocument();
  });

  it('ignores an earlier version output request after switching its props', async () => {
    let finish: (value: unknown) => void = () => {};
    vi.mocked(apiClient.get).mockImplementation(async (_url, options) => options?.params?.version_id === 'v1' ? new Promise(resolve => { finish = resolve; }) : [artifact('v2-output', 'v2')] as any);
    const { rerender } = render(<MemoryRouter><Artifacts embedded projectId="p1" versionId="v1"/></MemoryRouter>);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalledOnce());
    rerender(<MemoryRouter><Artifacts embedded projectId="p1" versionId="v2"/></MemoryRouter>);
    await screen.findByTestId('artifact-row-v2-output');
    await act(async () => finish([artifact('old-version')]));
    expect(screen.queryByTestId('artifact-row-old-version')).not.toBeInTheDocument();
    expect(screen.getByTestId('artifact-row-v2-output')).toBeInTheDocument();
  });
});

it('opens weights by default and scopes weights to a chosen training record without queue controls', async () => {
  render(<MemoryRouter><VersionResults projectId="p1" versionId="v1"/><Location/></MemoryRouter>);
  await screen.findByTestId('artifact-row-a1');
  expect(screen.getByRole('tab', { name: '模型权重' })).toHaveAttribute('aria-selected', 'true');
  const toolbar = screen.getByRole('tablist', { name: '版本训练结果' }).closest<HTMLElement>('.results-toolbar')!;
  expect(toolbar).toHaveTextContent('共 2 次训练');
  expect(within(toolbar).getByRole('link', { name: '全局队列' })).toHaveAttribute('href', '/queue?project_id=p1');
  expect(screen.queryByText('此版本的模型权重、采样图与训练记录。')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: '暂停' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: '训练记录' }));
  const row = await screen.findByTestId('result-job-j1');
  fireEvent.click(within(row).getByRole('button', { name: '查看权重' }));
  await waitFor(() => expect(apiClient.get).toHaveBeenCalledWith('/artifacts', expect.objectContaining({ params: { project_id: 'p1', version_id: 'v1', job_id: 'j1' } })));
  expect(screen.getByRole('combobox', { name: '权重所属任务' })).toHaveTextContent('Run j1');
  expect(screen.getByTestId('location')).toHaveTextContent('result_tab=artifacts');
});

it('opens this version training records from the empty weights action', async () => {
  artifacts = [];
  render(<MemoryRouter initialEntries={['/projects/p1/v/v1?step=results']}><VersionResults projectId="p1" versionId="v1"/><Location/></MemoryRouter>);
  const openJobs = await screen.findByRole('link', { name: '查看训练任务' });
  expect(screen.getByRole('tab', { name: '模型权重' })).toHaveAttribute('aria-selected', 'true');
  fireEvent.click(openJobs);
  expect(screen.getByRole('tab', { name: '训练记录' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByTestId('location')).toHaveTextContent('/projects/p1/v/v1?step=results&result_tab=jobs');
  expect(await screen.findByTestId('result-job-j1')).toHaveTextContent('Run j1');
});

it('keeps the chosen weight job and its name when paging the task selector', async () => {
  const original = vi.mocked(apiClient.get).getMockImplementation()!;
  vi.mocked(apiClient.get).mockImplementation(async (url, options) => {
    if (url === '/jobs') return { items: [job(options?.params?.page === 2 ? 'j2' : 'j1')], total: 51, page: options?.params?.page, page_size: 50, type: 'train' } as any;
    if (url === '/jobs/j1') return job('j1') as any;
    return original(url, options);
  });
  render(<MemoryRouter><VersionResults projectId="p1" versionId="v1"/></MemoryRouter>);
  await screen.findByTestId('artifact-row-a1');
  fireEvent.click(screen.getByRole('combobox', { name: '权重所属任务' }));
  fireEvent.click(screen.getByRole('option', { name: 'Run j1' }));
  await screen.findByTestId('artifact-row-a1');
  fireEvent.click(screen.getByRole('button', { name: '下一页任务' }));
  await waitFor(() => expect(apiClient.get).toHaveBeenCalledWith('/jobs/j1', expect.anything()));
  expect(screen.getByRole('combobox', { name: '权重所属任务' })).toHaveTextContent('Run j1');
  expect(screen.getByTestId('artifact-row-a1')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: '查看此任务检查点' })).toHaveAttribute('href', '/jobs/j1?tab=checkpoints');
  fireEvent.click(screen.getByRole('combobox', { name: '权重所属任务' }));
  expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['此版本全部训练', 'Run j1', 'Run j2']);
});

it('retains a paged-out weight selection on a connection failure and clears only a confirmed deleted job', async () => {
  const original = vi.mocked(apiClient.get).getMockImplementation()!;
  let deleted = false;
  vi.mocked(apiClient.get).mockImplementation(async (url, options) => {
    if (url === '/jobs') return { items: [job(options?.params?.page === 2 ? 'j2' : 'j1')], total: 51, page: options?.params?.page, page_size: 50, type: 'train' } as any;
    if (url === '/jobs/j1') throw deleted ? new ApiError(404, { code: 'job.not_found', message: 'Deleted' }) : new Error('Failed to fetch');
    return original(url, options);
  });
  render(<MemoryRouter><VersionResults projectId="p1" versionId="v1"/></MemoryRouter>);
  await screen.findByTestId('artifact-row-a1');
  fireEvent.click(screen.getByRole('combobox', { name: '权重所属任务' }));
  fireEvent.click(screen.getByRole('option', { name: 'Run j1' }));
  fireEvent.click(screen.getByRole('button', { name: '下一页任务' }));
  await waitFor(() => expect(apiClient.get).toHaveBeenCalledWith('/jobs/j1', expect.anything()));
  expect(screen.getByRole('combobox', { name: '权重所属任务' })).toHaveTextContent('Run j1');
  deleted = true; emit('queue.changed', {});
  await waitFor(() => expect(screen.getByRole('combobox', { name: '权重所属任务' })).toHaveTextContent('此版本全部训练'));
  expect(screen.queryByRole('link', { name: '查看此任务检查点' })).not.toBeInTheDocument();
  expect(apiClient.get).toHaveBeenCalledWith('/artifacts', expect.objectContaining({ params: { project_id: 'p1', version_id: 'v1' } }));
});

it('opens XYZ comparison in its independent page with the exact project version', async () => {
  render(<MemoryRouter initialEntries={['/projects/p1/v/v1?step=results']}><VersionResults projectId="p1" versionId="v1"/><Location/></MemoryRouter>);
  fireEvent.click(await screen.findByRole('tab', {name:'XYZ 对比'}));
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/sampling?project_id=p1&version_id=v1'));
  expect(screen.queryByRole('region', {name:'XYZ 对比采样'})).not.toBeInTheDocument();
});
