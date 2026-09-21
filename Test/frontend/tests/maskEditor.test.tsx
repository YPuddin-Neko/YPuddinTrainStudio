import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { MaskEditor } from '../../../frontend/src/components/masks/MaskEditor';
import Dataset from '../../../frontend/src/pages/Dataset/Dataset';
import * as maskApi from '../../../frontend/src/components/masks/maskApi';
import { apiClient } from '../../../frontend/src/api/client';
import '../../../frontend/src/i18n';

vi.mock('../../../frontend/src/components/masks/maskApi', async (original) => ({ ...(await original<typeof maskApi>()), loadMask: vi.fn(), saveMask: vi.fn() }));
vi.mock('../../../frontend/src/events/useEventStream', () => ({ useEventStream: () => {} }));
vi.mock('../../../frontend/src/api/hooks/useDatasetImages', () => ({ useDatasetImages: () => ({ items: [{ hash: 'image1', rel_path: 'photo.png', width: 8, height: 8, has_mask: false, caption: 'cat' }], total: 1, q: '', selected: new Set(), loading: false, error: null, refresh: vi.fn(), loadMore: vi.fn(), setQ: vi.fn(), selectAll: vi.fn(), clearSelection: vi.fn(), toggleSelect: vi.fn() }) }));
const initialInfo: maskApi.MaskInfo = { source: 'alpha', width: 8, height: 8, coverage: 128 / 255, has_mask: false, filename: 'photo.mask.png', revision: 'alpha:1:100', resized: false };
const props = () => ({ datasetId: 'data1', imageId: 'image1', relPath: 'photo.png', onClose: vi.fn(), onSaved: vi.fn(), onEnableTraining: vi.fn().mockResolvedValue(undefined) });
beforeEach(() => {
  vi.mocked(maskApi.loadMask).mockResolvedValue({ info: { ...initialInfo }, pixels: new Uint8Array(64).fill(128) });
  vi.mocked(maskApi.saveMask).mockImplementation(async (_did, _h, _path, info) => ({ ...info, source: 'sidecar', has_mask: true, revision: 'sidecar:2:90' }));
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => ({ createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData: vi.fn() }) as any);
  vi.stubGlobal('PointerEvent', MouseEvent); HTMLCanvasElement.prototype.setPointerCapture = vi.fn();
});
afterEach(() => { vi.restoreAllMocks(); vi.clearAllMocks(); vi.unstubAllGlobals(); });

describe('mask editor interactions', () => {
  it('loads alpha, edits real pixels with brush/eraser and undo/redo, then saves original-sized pixels', async () => {
    const callbacks = props(); render(<MaskEditor {...callbacks} />);
    await screen.findByText(/原图 Alpha 通道/);
    fireEvent.click(screen.getByRole('button', { name: '清空 · 全黑' }));
    expect(screen.getByText('当前为全黑，这张图片不会贡献训练损失。')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '反转' }));
    fireEvent.click(screen.getByRole('button', { name: '撤销' }));
    fireEvent.click(screen.getByRole('button', { name: '重做' }));
    const canvas = screen.getByLabelText('遮罩绘制画布');
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, width: 80, height: 80 } as DOMRect);
    fireEvent.change(screen.getByRole('slider', { name: '笔刷直径' }), { target: { value: '1' } });
    fireEvent.pointerDown(canvas, { button: 0, clientX: 35, clientY: 35, pointerId: 1 });
    fireEvent.pointerMove(canvas, { clientX: 45, clientY: 35, pointerId: 1 });
    fireEvent.pointerUp(canvas, { pointerId: 1 });
    fireEvent.click(screen.getByRole('button', { name: '笔刷 · 参与' }));
    fireEvent.pointerDown(canvas, { button: 0, clientX: 35, clientY: 35, pointerId: 2 });
    fireEvent.pointerUp(canvas, { pointerId: 2 });
    fireEvent.change(screen.getByRole('slider', { name: '叠加透明度' }), { target: { value: '0.75' } });
    fireEvent.click(screen.getByRole('checkbox', { name: '仅看黑白遮罩' }));
    fireEvent.click(screen.getByRole('button', { name: '保存遮罩' }));
    await waitFor(() => expect(callbacks.onSaved).toHaveBeenCalledOnce());
    const pixels = vi.mocked(maskApi.saveMask).mock.calls[0][4];
    expect(pixels[3 * 8 + 3]).toBe(255); expect(pixels[3 * 8 + 4]).toBe(0); expect(pixels[0]).toBe(255);
    expect(maskApi.saveMask).toHaveBeenCalledWith('data1', 'image1', 'photo.png', expect.objectContaining({ revision: 'alpha:1:100' }), expect.any(Uint8Array));
    expect(screen.getByText('遮罩已保存，下次训练会读取最新文件。')).toBeInTheDocument();
  });
  it('protects unsaved edits on close and exposes save failures without losing the draft', async () => {
    const callbacks = props(); render(<MaskEditor {...callbacks} />);
    await screen.findByRole('button', { name: '清空 · 全黑' });
    fireEvent.click(screen.getByRole('button', { name: '清空 · 全黑' }));
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    fireEvent.click(screen.getByRole('button', { name: '关闭遮罩编辑器' })); expect(callbacks.onClose).not.toHaveBeenCalled();
    vi.mocked(maskApi.saveMask).mockRejectedValueOnce(new Error('Mask changed; reload before saving'));
    fireEvent.click(screen.getByRole('button', { name: '保存并启用遮罩训练' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Mask changed'); expect(callbacks.onEnableTraining).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: '保存并启用遮罩训练' }));
    await waitFor(() => expect(callbacks.onEnableTraining).toHaveBeenCalledOnce());
    expect(vi.mocked(maskApi.saveMask).mock.calls[1][4].every((value) => value === 0)).toBe(true);
  });
  it('shows load errors, disables saving and supports retry', async () => {
    vi.mocked(maskApi.loadMask).mockRejectedValueOnce(new Error('Image outside allowed dataset'));
    render(<MaskEditor {...props()} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Image outside allowed dataset');
    expect(screen.getByRole('button', { name: '保存遮罩' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: '重新读取遮罩' }));
    expect(await screen.findByRole('button', { name: '反转' })).toBeEnabled();
  });
  it('opens from Dataset and enables masked loss while preserving existing project settings', async () => {
    const config = { model: { family: 'anima', dit: 'D:/models/model.safetensors' }, dataset: { sources: [{ path: 'D:/photos', repeats: 4 }], masked_loss: false, cache_dir: 'D:/cache' }, loop: { epochs: 12 } };
    vi.spyOn(apiClient, 'get').mockImplementation(async (endpoint) => endpoint.endsWith('/config') ? config as any : { source: { id: 'data1', project_id: 'project1', path: 'D:/photos', repeats: 4 }, index_status: 'ready', stats: { images: 1, masks: 0 } } as any);
    const put = vi.spyOn(apiClient, 'put').mockResolvedValue(config);
    render(<MemoryRouter initialEntries={['/datasets/data1']}><Routes><Route path="/datasets/:id" element={<Dataset />} /><Route path="/projects/:id/train" element={<p>Train configuration destination</p>} /></Routes></MemoryRouter>);
    fireEvent.click(await screen.findByRole('button', { name: '编辑遮罩' }));
    await screen.findByRole('button', { name: '反转' });
    fireEvent.click(screen.getByRole('button', { name: '保存并启用遮罩训练' }));
    await screen.findByText('Train configuration destination');
    expect(put).toHaveBeenCalledWith('/projects/project1/config', { ...config, dataset: { ...config.dataset, masked_loss: true } });
    expect(maskApi.saveMask).toHaveBeenCalledOnce();
  });
  it('keeps a managed Windows dataset title compact and opens distribution only when requested', async () => {
    const path = 'D:\\Trainer\\data\\projects\\project1\\datasets\\d_012abc-人物素材\\';
    vi.spyOn(apiClient, 'get').mockResolvedValue({ source: { id: 'data1', project_id: 'project1', path, repeats: 2 }, index_status: 'ready', stats: { images: 1, masks: 0, captioned: 1, resolutions: [{ w: 512, h: 768, count: 1 }] } });
    render(<MemoryRouter initialEntries={['/datasets/data1']}><Routes><Route path="/datasets/:id" element={<Dataset />} /></Routes></MemoryRouter>);
    const title = await screen.findByRole('heading', { name: '人物素材' });
    expect(title).toHaveAttribute('title', path);
    expect(screen.getByText(path).closest('details')).not.toHaveAttribute('open');
    expect(screen.queryByText('512×768')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '分布与分桶' }));
    expect(screen.getByText('512×768')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '分布与分桶' }));
    expect(screen.queryByText('512×768')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '编辑遮罩' })).toBeEnabled();
  });
});
