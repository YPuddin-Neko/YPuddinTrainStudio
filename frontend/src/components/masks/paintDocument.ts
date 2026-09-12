import type { MaskPoint } from './maskDocument';

export type PaintStroke = { kind: 'stroke'; points: MaskPoint[]; diameter: number; color: string; erase: boolean };
export type PaintOperation = PaintStroke | { kind: 'clear' };
export function rgb(color: string): [number, number, number] {
  if (!/^#[0-9a-f]{6}$/i.test(color)) throw new Error('Choose an RGB color');
  return [1, 3, 5].map(start => Number.parseInt(color.slice(start, start + 2), 16)) as [number, number, number];
}
/** Edits preserve source alpha; erasing restores the pixels loaded at this edit session. */
export class PaintDocument {
  readonly pixels: Uint8ClampedArray;
  readonly base: Uint8ClampedArray;
  private operations: PaintOperation[] = [];
  private position = 0;
  private savedPosition = 0;
  constructor(readonly width: number, readonly height: number, pixels: Uint8ClampedArray) {
    if (pixels.length !== width * height * 4) throw new Error('Image dimensions do not match its pixels');
    this.base = pixels.slice(); this.pixels = pixels.slice();
  }
  get dirty() { return this.position !== this.savedPosition; }
  get canUndo() { return this.position > 0; }
  get canRedo() { return this.position < this.operations.length; }
  markSaved() { this.savedPosition = this.position; }
  colorAt({ x, y }: MaskPoint) {
    const offset = (Math.min(this.height - 1, Math.floor(y)) * this.width + Math.min(this.width - 1, Math.floor(x))) * 4;
    return `#${[0, 1, 2].map(channel => this.pixels[offset + channel].toString(16).padStart(2, '0')).join('')}`;
  }
  segment(from: MaskPoint, to: MaskPoint, diameter: number, color: string, erase: boolean) {
    const [red, green, blue] = rgb(color); const radius = Math.max(.5, diameter / 2);
    const dx = to.x - from.x, dy = to.y - from.y, length = dx * dx + dy * dy;
    for (let y = Math.max(0, Math.floor(Math.min(from.y, to.y) - radius)); y <= Math.min(this.height - 1, Math.ceil(Math.max(from.y, to.y) + radius)); y++) {
      for (let x = Math.max(0, Math.floor(Math.min(from.x, to.x) - radius)); x <= Math.min(this.width - 1, Math.ceil(Math.max(from.x, to.x) + radius)); x++) {
        const along = length ? Math.max(0, Math.min(1, ((x + .5 - from.x) * dx + (y + .5 - from.y) * dy) / length)) : 0;
        if (Math.hypot(x + .5 - from.x - along * dx, y + .5 - from.y - along * dy) > radius) continue;
        const at = (y * this.width + x) * 4;
        this.pixels[at] = erase ? this.base[at] : red;
        this.pixels[at + 1] = erase ? this.base[at + 1] : green;
        this.pixels[at + 2] = erase ? this.base[at + 2] : blue;
      }
    }
  }
  apply(operation: PaintOperation, alreadyPainted = false) {
    if (this.savedPosition > this.position) this.savedPosition = -1;
    this.operations = this.operations.slice(0, this.position); this.operations.push(operation); this.position++;
    if (!alreadyPainted) this.render(operation);
  }
  undo() { if (this.canUndo) { this.position--; this.pixels.set(this.base); for (let i = 0; i < this.position; i++) this.render(this.operations[i]); } }
  redo() { if (this.canRedo) this.render(this.operations[this.position++]); }
  private render(operation: PaintOperation) {
    if (operation.kind === 'clear') this.pixels.set(this.base);
    else operation.points.forEach((point, index) => this.segment(operation.points[Math.max(0, index - 1)], point, operation.diameter, operation.color, operation.erase));
  }
}
