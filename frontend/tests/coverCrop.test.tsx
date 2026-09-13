import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ProjectCoverCropper from '../src/pages/Projects/ProjectCoverCropper';
import { COVER_ASPECT, cropForView, cropImageStyle, INITIAL_CROP_VIEW, viewForCrop } from '../src/pages/Projects/coverCrop';
import i18n from '../src/i18n';

beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, writable: true, value: vi.fn(() => 'blob:crop-fixture') });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, writable: true, value: vi.fn() });
  vi.stubGlobal('PointerEvent', class extends MouseEvent {
    readonly pointerId: number;
    constructor(type: string, options: PointerEventInit = {}) { super(type, options); this.pointerId = options.pointerId ?? 0; }
  });
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });
function decodedImage(width = 1600, height = 2000) {
  const image = screen.getByAltText('待裁切的项目封面');
  Object.defineProperties(image, { naturalWidth: { configurable: true, value: width }, naturalHeight: { configurable: true, value: height } });
  fireEvent.load(image); return image;
}
const submit = () => fireEvent.click(screen.getByRole('button', { name: '使用此裁切' }));

it('limits zoom to a source pixel and prevents unusably small crops before upload', () => {
  const apply = vi.fn();
  const mounted = render(<ProjectCoverCropper file={new File(['source'], 'tiny.png', { type: 'image/png' })} onApply={apply} onCancel={vi.fn()}/>);
  decodedImage(4, 4);
  const zoom = screen.getByRole('slider', { name: /^缩放(?:\s|$)/ });
  expect(zoom).toHaveAttribute('max', '2.5');
  fireEvent.change(zoom, { target: { value: '4' } }); submit();
  expect(apply.mock.lastCall![0].height * 4).toBeGreaterThanOrEqual(1);
  mounted.unmount(); apply.mockClear();
  render(<ProjectCoverCropper file={new File(['source'], 'pixel.png', { type: 'image/png' })} onApply={apply} onCancel={vi.fn()}/>);
  decodedImage(1, 1);
  expect(screen.getByRole('alert')).toHaveTextContent('图片尺寸太小');
  expect(screen.getByRole('button', { name: '使用此裁切' })).toBeDisabled();
  submit(); expect(apply).not.toHaveBeenCalled();
});

describe('project cover crop geometry', () => {
  it.each([[1600, 2000], [3000, 1000], [1600, 1000]])('keeps a 16:10 crop within a %sx%s source at center and all pan limits', (width, height) => {
    for (const view of [INITIAL_CROP_VIEW, { zoom: 4, cx: -5, cy: 8 }, { zoom: 2, cx: 8, cy: -5 }]) {
      const crop = cropForView(width, height, view);
      expect(crop.x).toBeGreaterThanOrEqual(0); expect(crop.y).toBeGreaterThanOrEqual(0);
      expect(crop.x + crop.width).toBeLessThanOrEqual(1); expect(crop.y + crop.height).toBeLessThanOrEqual(1);
      expect(crop.width * width / (crop.height * height)).toBeCloseTo(16 / 10, 12);
    }
  });

  it('uses the correct portrait crop, constrains zoom, and restores a saved noncentral crop', () => {
    expect(COVER_ASPECT).toBe(1.6);
    expect(cropForView(1600, 2000, INITIAL_CROP_VIEW)).toEqual({ x: 0, y: .25, width: 1, height: .5 });
    expect(cropForView(1600, 2000, { zoom: 0, cx: .5, cy: .5 })).toEqual({ x: 0, y: .25, width: 1, height: .5 });
    expect(cropForView(1600, 2000, { zoom: 9, cx: .5, cy: .5 })).toEqual({ x: .375, y: .4375, width: .25, height: .125 });
    const saved = { x: .35, y: .2, width: .5, height: .25 };
    const restored = cropForView(1600, 2000, viewForCrop(1600, 2000, saved));
    for (const key of ['x', 'y', 'width', 'height'] as const) expect(restored[key]).toBeCloseTo(saved[key], 12);
    expect(cropImageStyle({ x: 0, y: .25, width: 1, height: .5 })).toMatchObject({ width: '100%', height: '200%', left: '0%', top: '-50%' });
  });
});

it('loads a saved crop, moves it with the keyboard, and resets both zoom and framing', () => {
  const apply = vi.fn();
  render(<ProjectCoverCropper file={new File(['source'], 'portrait.png', { type: 'image/png' })} initialCrop={{ x: .25, y: .375, width: .5, height: .25 }} onApply={apply} onCancel={vi.fn()}/>);
  decodedImage();
  const frame = screen.getByRole('group', { name: '封面裁切区域' });
  expect(frame).toHaveFocus(); expect(screen.getByRole('slider', { name: /^缩放(?:\s|$)/ })).toHaveValue('2');
  fireEvent.keyDown(frame, { key: 'ArrowLeft' }); fireEvent.keyDown(frame, { key: 'ArrowUp', shiftKey: true }); submit();
  expect(apply).toHaveBeenLastCalledWith({ x: .26, y: .395, width: .5, height: .25 });
  for (let n = 0; n < 60; n += 1) fireEvent.keyDown(frame, { key: 'ArrowLeft', shiftKey: true });
  submit(); expect(apply.mock.lastCall![0].x).toBe(.5);
  fireEvent.click(screen.getByRole('button', { name: '重置' }));
  expect(screen.getByRole('slider', { name: /^缩放(?:\s|$)/ })).toHaveValue('1');
  submit(); expect(apply).toHaveBeenLastCalledWith({ x: 0, y: .25, width: 1, height: .5 });
});

it('bounds pointer dragging to the original image and ignores another pointer or movement after release', () => {
  const apply = vi.fn();
  const mounted = render(<ProjectCoverCropper file={new File(['source'], 'portrait.png', { type: 'image/png' })} onApply={apply} onCancel={vi.fn()}/>);
  decodedImage(); const frame = screen.getByRole('group', { name: '封面裁切区域' });
  Object.defineProperties(frame, { setPointerCapture: { value: vi.fn() }, releasePointerCapture: { value: vi.fn() } });
  vi.spyOn(frame, 'getBoundingClientRect').mockReturnValue({ x: 0, y: 0, left: 0, top: 0, right: 400, bottom: 250, width: 400, height: 250, toJSON: () => ({}) });
  fireEvent.change(screen.getByRole('slider', { name: /^缩放(?:\s|$)/ }), { target: { value: '2' } });
  fireEvent.pointerDown(frame, { button: 0, pointerId: 10, clientX: 50, clientY: 50 });
  expect(frame.setPointerCapture).toHaveBeenCalledWith(10);
  fireEvent.pointerMove(frame, { pointerId: 11, clientX: 1000, clientY: 1000 }); submit();
  expect(apply).toHaveBeenLastCalledWith({ x: .25, y: .375, width: .5, height: .25 });
  fireEvent.pointerMove(frame, { pointerId: 10, clientX: 1000, clientY: 1000 }); submit();
  expect(apply).toHaveBeenLastCalledWith({ x: 0, y: 0, width: .5, height: .25 });
  fireEvent.pointerUp(frame, { pointerId: 10 }); expect(frame.releasePointerCapture).toHaveBeenCalledWith(10);
  fireEvent.pointerMove(frame, { pointerId: 10, clientX: -1000, clientY: -1000 }); submit();
  expect(apply).toHaveBeenLastCalledWith({ x: 0, y: 0, width: .5, height: .25 });
  fireEvent.pointerDown(frame, { button: 0, pointerId: 12, clientX: 50, clientY: 50 });
  fireEvent.pointerMove(frame, { pointerId: 12, clientX: -1000, clientY: -1000 }); submit();
  expect(apply).toHaveBeenLastCalledWith({ x: .5, y: .75, width: .5, height: .25 });
  fireEvent.pointerCancel(frame, { pointerId: 12 });
  fireEvent.pointerMove(frame, { pointerId: 12, clientX: 1000, clientY: 1000 }); submit();
  expect(apply).toHaveBeenLastCalledWith({ x: .5, y: .75, width: .5, height: .25 });
  mounted.unmount(); expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:crop-fixture');
});
