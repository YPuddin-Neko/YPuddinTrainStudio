import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import XyzSampling from '../src/components/sampling/XyzSampling';
import { apiClient } from '../src/api/client';
import { axisCount, parseAxis, type XyzOptions, type XyzTask } from '../src/components/sampling/xyzTypes';
import i18n from '../src/i18n';

const options: XyzOptions = {
  family: 'krea2', defaults: { prompt: 'a pudding', negative: '', width: 1024, height: 1024, steps: 28, cfg: 4.5, seed: 42, sampler: 'euler', scheduler: 'uniform', shift: 1.15, guidance: null, adapter_scale: 1, checkpoint_id: 'cp2', sampling_model_id: null },
  checkpoints: [{ id: 'cp1', name: 'pudding-epoch1.safetensors', step: 10 }, { id: 'cp2', name: 'pudding-final.safetensors', step: 20 }],
  axes: [{ key: 'checkpoint', label: 'Checkpoint' }, { key: 'steps', label: 'Steps' }, { key: 'cfg', label: 'CFG' }, { key: 'seed', label: 'Seed' }, { key: 'adapter_scale', label: 'LoRA strength' }, { key: 'sampler', label: 'Sampler', values: ['euler', 'heun'] }, { key: 'scheduler', label: 'Scheduler', values: ['uniform', 'simple'] }],
  sampling_models: [{ id: 'turbo', name: 'Krea2 Turbo FP8', variant: 'turbo' }], limits: { max_cells: 64, max_axis_values: 12 },
};
const task: XyzTask = {
  id: 'xy1', job_id: 'jxy', source_job_id: 'run', status: 'completed', phase: 'completed', done: 4, total: 4, error: null, created_at: 1700000000, finished_at: 1700000030, can_cancel: false,
  request: { ...options.defaults, x: { key: 'checkpoint', values: ['cp1', 'cp2'] }, y: null, z: { key: 'seed', values: [42, 43] } },
  manifest: { complete: true, grids: [0, 1].map(z => ({ z, z_value: 42 + z, file: `grid-${z}.png`, url: `/api/xyz/xy1/file?name=grid-${z}.png` })), cells: [0, 1, 2, 3].map(index => ({ index, x: index % 2, y: 0, z: Math.floor(index / 2), x_value: `cp${index % 2 + 1}`, y_value: null, z_value: 42 + Math.floor(index / 2), seed: 42 + Math.floor(index / 2), steps: 28, cfg: 4.5, sampler: 'euler', scheduler: 'uniform', shift: 1.15, adapter_scale: 1, checkpoint_id: `cp${index % 2 + 1}`, file: `cell-${index}.png`, url: `/api/xyz/xy1/file?name=cell-${index}.png` })) },
};
let history: XyzTask[];
let currentOptions: XyzOptions;
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN'); history = []; currentOptions = structuredClone(options);
  vi.spyOn(apiClient, 'get').mockImplementation(async url => {
    if (url === '/jobs/run/xyz/options') return structuredClone(currentOptions) as never;
    if (url === '/jobs/run/xyz') return history as never;
    if (url === '/xyz/xy1') return { ...task, status: 'running', done: 1, can_cancel: true } as never;
    throw new Error(`Unexpected ${url}`);
  });
  vi.spyOn(apiClient, 'post').mockResolvedValue(task);
});
afterEach(() => vi.restoreAllMocks());
const view = (readOnly = false) => render(<MemoryRouter><XyzSampling sourceJobId="run" readOnly={readOnly}/></MemoryRouter>);
function choose(name: string, value: string) {
  fireEvent.click(screen.getByRole('combobox', { name }));
  fireEvent.click(screen.getByRole('option', { name: value }));
}

describe('XYZ sampling workspace', () => {
  it('compares saved checkpoints with source defaults and submits all three axes explicitly', async () => {
    view(); await screen.findByText('对比设置');
    expect(screen.getByRole('combobox', { name: '对比使用的训练权重' })).toHaveTextContent('pudding-final');
    expect(screen.getByRole('combobox', { name: '对比使用的训练权重' })).toBeDisabled();
    choose('Y · 纵向比较', 'LoRA 强度'); choose('Z · 分页比较', '随机种子');
    expect(screen.getByText('2 列 × 3 行 × 3 页，共 18 张')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '生成对比图' }));
    await waitFor(() => expect(apiClient.post).toHaveBeenCalledWith('/jobs/run/xyz', expect.objectContaining({ name: '模型测试', x: { key: 'checkpoint', values: ['cp1', 'cp2'] }, y: { key: 'adapter_scale', values: [0.6, 0.8, 1] }, z: { key: 'seed', values: [42, 43, 44] }, checkpoint_id: 'cp2', width: 1024, seed: 42 }), expect.anything()));
  });
  it('applies Turbo inference defaults when chosen, then restores Raw defaults', async () => {
    view(); await screen.findByText('对比设置'); choose('X · 横向比较', '采样步数');
    choose('采样底模', 'Krea2 Turbo FP8');
    expect(screen.getByLabelText('固定 steps')).toHaveValue(8); expect(screen.getByLabelText('固定 cfg')).toHaveValue(0);
    expect(screen.getByLabelText('X · 横向比较 · 取值')).toHaveValue('4, 8, 12');
    choose('采样底模', '沿用本次训练底模');
    expect(screen.getByLabelText('固定 steps')).toHaveValue(28); expect(screen.getByLabelText('固定 cfg')).toHaveValue(4.5);
  });
  it('rejects invalid or excessive grids locally before a request is sent', async () => {
    view(); await screen.findByText('对比设置'); choose('X · 横向比较', '采样步数');
    fireEvent.change(screen.getByLabelText('X · 横向比较 · 取值'), { target: { value: 'NaN,8' } });
    expect(screen.getByRole('button', { name: '生成对比图' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('X · 横向比较 · 取值'), { target: { value: '1,2,3,4,5,6,7,8,9' } });
    choose('Y · 纵向比较', '随机种子'); choose('Z · 分页比较', 'LoRA 强度');
    expect(screen.getByText('一次最多 64 张，请减少取值')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '生成对比图' })).toBeDisabled(); expect(apiClient.post).not.toHaveBeenCalled();
  });
  it('restores history, switches Z pages and exports the matching page or original cell', async () => {
    history = [task]; view(); await screen.findByText('对比设置');
    expect(screen.getByRole('link', { name: '下载本页网格' })).toHaveAttribute('href', expect.stringContaining('grid-0.png'));
    fireEvent.click(screen.getByRole('button', { name: '随机种子 · 43' }));
    expect(screen.getByRole('link', { name: '下载本页网格' })).toHaveAttribute('href', expect.stringContaining('grid-1.png'));
    fireEvent.click(screen.getByRole('button', { name: '查看第 1 列第 1 行' }));
    const dialog = screen.getByRole('dialog'); expect(within(dialog).getByRole('link', { name: '下载原图' })).toHaveAttribute('href', expect.stringContaining('cell-2.png'));
    fireEvent.keyDown(within(dialog).getByRole('button', { name: '关闭' }), { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    fireEvent.change(screen.getByLabelText('模型测试提示词'), { target: { value: 'changed' } });
    fireEvent.click(screen.getByRole('button', { name: '复用参数' }));
    expect(screen.getByLabelText('模型测试提示词')).toHaveValue('a pudding');
    expect(screen.getByLabelText('Z · 分页比较 · 取值')).toHaveValue('42, 43');
  });
  it('preserves partial results and cancels the actual selected task', async () => {
    history = [{ ...task, status: 'running', done: 1, can_cancel: true, manifest: { ...task.manifest, complete: false, cells: task.manifest.cells.slice(0, 1) } }];
    view(); await screen.findByText('对比设置');
    expect(screen.getByRole('button', { name: '查看第 1 列第 1 行' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: '取消生成' }));
    await waitFor(() => expect(apiClient.post).toHaveBeenCalledWith('/xyz/xy1/cancel', {}, expect.anything()));
  });
  it('keeps archived results viewable while preventing mutation', async () => {
    history = [task]; view(true); await screen.findByText('对比设置');
    expect(screen.getByRole('button', { name: '生成对比图' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '查看第 1 列第 1 行' })).toBeEnabled();
  });
  it('uses native full checkpoints without base-model or LoRA controls, including reused legacy settings', async () => {
    currentOptions.training_mode = 'full';
    currentOptions.defaults.checkpoint_id = null;
    history = [{ ...task, request: { ...task.request, checkpoint_id: null, sampling_model_id: 'turbo', adapter_scale: 0.5, x: { key: 'adapter_scale', values: [0.5, 1] }, y: null, z: null } }];
    view(); await screen.findByText('对比设置');
    fireEvent.click(screen.getByRole('button', { name: '复用参数' }));
    expect(screen.queryByRole('combobox', { name: '采样底模' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('固定 adapter_scale')).not.toBeInTheDocument();
    choose('X · 横向比较', '采样步数');
    fireEvent.click(screen.getByRole('combobox', { name: '对比使用的训练权重' }));
    expect(screen.queryByRole('option', { name: /只看底模/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('option', { name: 'pudding-final.safetensors' }));
    fireEvent.click(screen.getByRole('button', { name: '生成对比图' }));
    await waitFor(() => expect(apiClient.post).toHaveBeenCalledWith('/jobs/run/xyz', expect.objectContaining({ checkpoint_id: 'cp2', sampling_model_id: null, adapter_scale: 1, x: { key: 'steps', values: [8, 16, 24] } }), expect.anything()));
  });
  it('blocks full comparison until a saved checkpoint exists', async () => {
    currentOptions.training_mode = 'full'; currentOptions.checkpoints = []; currentOptions.defaults.checkpoint_id = null;
    view(); await screen.findByText('对比设置');
    expect(screen.getByText('当前训练尚未保存模型检查点，保存后才能生成对比图')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '生成对比图' })).toBeDisabled();
    expect(apiClient.post).not.toHaveBeenCalled();
  });
  it('parses Chinese separators without losing order and calculates page count', () => {
    expect(parseAxis('steps', '8，12\n16')).toEqual({ key: 'steps', values: [8, 12, 16] });
    expect(axisCount([parseAxis('steps', '8,16'), null, parseAxis('seed', '1,2,3')])).toBe(6);
  });
});
