import { describe, expect, it } from 'vitest';
import { MaskDocument, imagePoint, paintSegment } from '../src/components/masks/maskDocument';
describe('mask pixel editing', () => {
  it('maps zoomed canvas coordinates to original pixels and interpolates an entire stroke', () => {
    expect(imagePoint(150, 100, { left: 50, top: 50, width: 200, height: 100 }, 1000, 500)).toEqual({ x: 500, y: 250 });
    const pixels = new Uint8Array(100).fill(255);
    paintSegment(pixels, 10, 10, { x: 1.5, y: 5.5 }, { x: 8.5, y: 5.5 }, 1, 0);
    expect([...pixels.slice(51, 59)]).toEqual(Array(8).fill(0)); expect(pixels[45]).toBe(255);
    paintSegment(pixels, 10, 10, { x: 4.5, y: 5.5 }, { x: 4.5, y: 5.5 }, 1, 255); expect(pixels[54]).toBe(255);
  });
  it('preserves gray weights, undoes operations and discards redo after a new edit', () => {
    const doc = new MaskDocument(2, 2, new Uint8Array([0, 64, 128, 255]));
    doc.apply({ kind: 'invert' }); expect([...doc.pixels]).toEqual([255, 191, 127, 0]);
    doc.undo(); expect([...doc.pixels]).toEqual([0, 64, 128, 255]); expect(doc.dirty).toBe(false);
    doc.redo(); doc.markSaved(); expect(doc.dirty).toBe(false);
    doc.undo(); doc.apply({ kind: 'fill', value: 0 }); expect(doc.canRedo).toBe(false); expect(doc.dirty).toBe(true);
    expect(doc.coverage).toBe(0); doc.undo(); expect([...doc.pixels]).toEqual([0, 64, 128, 255]);
  });
  it('undoes and replays a live stroke without losing the baseline', () => {
    const doc = new MaskDocument(4, 4, new Uint8Array(16).fill(255));
    const stroke = { kind: 'stroke' as const, points: [{ x: 1.5, y: 1.5 }, { x: 2.5, y: 1.5 }], diameter: 1, value: 0 };
    paintSegment(doc.pixels, 4, 4, stroke.points[0], stroke.points[1], 1, 0);
    doc.apply(stroke, true); expect(doc.pixels[5]).toBe(0); expect(doc.pixels[6]).toBe(0);
    doc.undo(); expect(doc.coverage).toBe(1); doc.redo(); expect(doc.pixels[5]).toBe(0); expect(doc.pixels[6]).toBe(0);
  });
});
