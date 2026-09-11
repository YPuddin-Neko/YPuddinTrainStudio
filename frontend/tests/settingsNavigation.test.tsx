import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, useLocation, useSearchParams } from 'react-router-dom';
import AppRoutes from '../src/router';
import { apiClient } from '../src/api/client';
import { ApiError, type Artifact } from '../src/api/types';
import i18n from '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {}, useEventStreamStatus: () => 'connected' }));
vi.mock('../src/components/EnvironmentManagerPanel', () => ({ EnvironmentManagerPanel: () => <div data-testid="runtime-panel">Runtime manager</div> }));
vi.mock('../src/pages/Models/Models', () => ({ default: function EmbeddedModels({ embedded }: { embedded?: boolean }) { const [params] = useSearchParams(); return <div data-testid="embedded-models">{embedded ? 'Embedded' : 'Standalone'} model weights: {params.get('family')}</div>; } }));
const settings = { paths: { data_root: 'D:/studio', models_dir: 'D:/models', cache_dir: 'D:/cache', output_dir: 'D:/runs' }, server: { host: '127.0.0.1', port: 8765 }, ui: { language: 'zh-CN', theme: 'light' } };
const original: Artifact = { id: 'a1', name: 'cat.safetensors', job_id: 'j1', project_id: 'p1', size: 1024, created_at: 1, path: 'D:/runs/cat.safetensors', kind: 'weights', step: 100, metadata: { 'ypuddin.family': 'anima' } };
let artifacts: Artifact[];
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN'); artifacts = [{ ...original }];
  vi.spyOn(apiClient, 'get').mockImplementation(async endpoint => {
    if (endpoint === '/settings') return settings as any;
    if (endpoint === '/system/stats') return { gpus: [{ index: 0, name: 'NVIDIA test', kind: 'cuda', power_w: 72, mem_used_mb: 2048, mem_total_mb: 8192 }] } as any;
    if (endpoint === '/system/info') return { ypuddin: '0.3.0' } as any;
    if (endpoint === '/jobs') return { items: [], total: 0, page: 1, page_size: 50 } as any;
    if (endpoint === '/artifacts') return artifacts as any;
    throw new Error(`Unexpected GET ${endpoint}`);
  });
  vi.spyOn(apiClient, 'post').mockImplementation(async (endpoint, body) => {
    if (endpoint === '/artifacts/a1/convert') { const converted = { ...original, id: 'a2', name: `cat-${(body as { format: string }).format}.safetensors` }; artifacts.push(converted); return converted as any; }
    throw new Error(`Unexpected POST ${endpoint}`);
  });
  vi.spyOn(apiClient, 'delete').mockImplementation(async () => { artifacts = artifacts.filter(item => item.id !== 'a1'); return { ok: true } as any; });
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });
function Location() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}{location.hash}</output>; }
function show(path: string) { render(<MemoryRouter initialEntries={[path]}><AppRoutes /><Location /></MemoryRouter>); }

describe('settings information architecture', () => {
  it('moves a bookmarked model link with its query into Settings and supports keyboard tab navigation', async () => {
    show('/models?family=krea2&project=p1');
    expect(await screen.findByTestId('embedded-models')).toHaveTextContent('Embedded model weights: krea2');
    expect(screen.getByTestId('location')).toHaveTextContent('/settings/environment?family=krea2&project=p1&tab=models');
    expect(screen.getByRole('tab', { name: '模型权重' })).toHaveAttribute('aria-selected', 'true');
    const navigation = screen.getByRole('navigation', { name: '主导航' });
    expect(within(navigation).queryByRole('link', { name: i18n.t('nav.models') })).not.toBeInTheDocument();
    expect(within(navigation).queryByRole('link', { name: i18n.t('nav.artifacts') })).not.toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole('tab', { name: '模型权重' }), { key: 'ArrowLeft' });
    expect(await screen.findByTestId('runtime-panel')).toBeInTheDocument();
    expect(screen.getByTestId('location')).toHaveTextContent('family=krea2&project=p1&tab=runtime');
    expect(screen.getByRole('tab', { name: '运行环境' })).toHaveFocus();
    expect(screen.getByTestId('topbar-gpu-power')).toHaveTextContent('72 W');
  });
  it('retains legacy artifact project filters and keeps download, metadata, conversion and removal working', async () => {
    show('/artifacts?project_id=p1&job_id=j1');
    const row = await screen.findByTestId('artifact-row-a1');
    expect(screen.getByTestId('location')).toHaveTextContent('project_id=p1&job_id=j1&tab=artifacts');
    expect(apiClient.get).toHaveBeenCalledWith('/artifacts', expect.objectContaining({ params: { project_id: 'p1' } }));
    expect(within(row).getByRole('link', { name: '下载: cat.safetensors' })).toHaveAttribute('href', 'http://localhost:3000/api/artifacts/a1/download');
    fireEvent.click(within(row).getByRole('button', { name: '查看元数据: cat.safetensors' }));
    expect(await screen.findByRole('dialog')).toHaveTextContent('ypuddin.family');
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    const convert = within(row).getByRole('combobox', { name: '转换格式: cat.safetensors' });
    expect(within(convert).queryByRole('option', { name: 'peft' })).not.toBeInTheDocument();
    fireEvent.change(convert, { target: { value: 'kohya' } });
    expect(await screen.findByTestId('artifact-row-a2')).toHaveTextContent('cat-kohya.safetensors');
    expect(apiClient.post).toHaveBeenCalledWith('/artifacts/a1/convert', { format: 'kohya' }, { silent: true });
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: '移除产物记录: cat.safetensors' }));
    await waitFor(() => expect(screen.queryByTestId('artifact-row-a1')).not.toBeInTheDocument());
    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('磁盘中的权重文件会保留'));
    fireEvent.click(screen.getByRole('button', { name: '清除筛选' }));
    expect(screen.getByTestId('location')).toHaveTextContent('/settings/environment?tab=artifacts');
  });
  it('routes old environment anchors and separates Preferences from environment and default-model controls', async () => {
    show('/settings#environment');
    await screen.findByTestId('runtime-panel');
    fireEvent.click(within(screen.getByRole('navigation', { name: '设置分区' })).getByRole('link', { name: '存储与界面' }));
    const page = await screen.findByTestId('settings-page');
    expect(screen.getByTestId('location')).toHaveTextContent('/settings/preferences');
    expect(within(page).getByRole('textbox', { name: i18n.t('settings.dataRoot') })).toHaveAttribute('readonly');
    expect(screen.queryByTestId('runtime-panel')).not.toBeInTheDocument();
    expect(screen.queryByTestId('embedded-models')).not.toBeInTheDocument();
    expect(screen.getByTestId('settings-theme')).toHaveValue('light');
  });
  it('shows conversion errors without pretending an output was created', async () => {
    vi.mocked(apiClient.post).mockRejectedValue(new ApiError(400, { code: 'convert.bad_format', message: 'Unsupported weights for this conversion' }));
    show('/settings/environment?tab=artifacts');
    const row = await screen.findByTestId('artifact-row-a1');
    fireEvent.change(within(row).getByRole('combobox'), { target: { value: 'comfyui' } });
    expect(await screen.findByRole('alert')).toHaveTextContent('Unsupported weights for this conversion');
    expect(screen.queryByTestId('artifact-row-a2')).not.toBeInTheDocument();
    expect(screen.getByTestId('artifact-row-a1')).toBeInTheDocument();
  });
});
