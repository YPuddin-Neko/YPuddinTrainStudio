export interface MaskPoint { x: number; y: number }
export type MaskOperation = { kind: 'fill'; value: number } | { kind: 'invert' } | { kind: 'stroke'; points: MaskPoint[]; diameter: number; value: number };

/** Coordinates and brush diameter are always source-image pixels, independent of view zoom. */
export function paintSegment(pixels: Uint8Array, width: number, height: number, from: MaskPoint, to: MaskPoint, diameter: number, value: number) {
  const radius = Math.max(0.5, diameter / 2);
  const left = Math.max(0, Math.floor(Math.min(from.x, to.x) - radius));
  const right = Math.min(width - 1, Math.ceil(Math.max(from.x, to.x) + radius));
  const top = Math.max(0, Math.floor(Math.min(from.y, to.y) - radius));
  const bottom = Math.min(height - 1, Math.ceil(Math.max(from.y, to.y) + radius));
  const dx = to.x - from.x; const dy = to.y - from.y; const lengthSquared = dx * dx + dy * dy;
  for (let y = top; y <= bottom; y++) for (let x = left; x <= right; x++) {
    const along = lengthSquared ? Math.max(0, Math.min(1, ((x + 0.5 - from.x) * dx + (y + 0.5 - from.y) * dy) / lengthSquared)) : 0;
    const distance = Math.hypot(x + 0.5 - from.x - along * dx, y + 0.5 - from.y - along * dy);
    if (distance <= radius) pixels[y * width + x] = value;
  }
}

export class MaskDocument {
  readonly width: number;
  readonly height: number;
  readonly pixels: Uint8Array;
  private base: Uint8Array;
  private operations: MaskOperation[] = [];
  private position = 0;
  private savedPosition = 0;
  constructor(width: number, height: number, pixels: Uint8Array) {
    if (width * height !== pixels.length) throw new Error('Mask dimensions do not match its pixels');
    this.width = width; this.height = height; this.base = pixels.slice(); this.pixels = pixels.slice();
  }
  get dirty() { return this.position !== this.savedPosition; }
  get canUndo() { return this.position > 0; }
  get canRedo() { return this.position < this.operations.length; }
  get coverage() { return this.pixels.reduce((sum, value) => sum + value, 0) / (this.pixels.length * 255); }
  markSaved() { this.savedPosition = this.position; }
  apply(operation: MaskOperation, alreadyPainted = false) {
    if (this.savedPosition > this.position) this.savedPosition = -1;
    this.operations = this.operations.slice(0, this.position);
    this.operations.push(operation); this.position++;
    if (!alreadyPainted) this.renderOperation(operation);
  }
  undo() { if (this.canUndo) { this.position--; this.rebuild(); } }
  redo() { if (this.canRedo) { this.renderOperation(this.operations[this.position]); this.position++; } }
  private rebuild() { this.pixels.set(this.base); for (let i = 0; i < this.position; i++) this.renderOperation(this.operations[i]); }
  private renderOperation(operation: MaskOperation) {
    if (operation.kind === 'fill') this.pixels.fill(operation.value);
    else if (operation.kind === 'invert') for (let i = 0; i < this.pixels.length; i++) this.pixels[i] = 255 - this.pixels[i];
    else operation.points.forEach((point, i) => paintSegment(this.pixels, this.width, this.height, operation.points[Math.max(0, i - 1)], point, operation.diameter, operation.value));
  }
}

export function imagePoint(clientX: number, clientY: number, bounds: { left: number; top: number; width: number; height: number }, width: number, height: number): MaskPoint {
  return { x: Math.max(0, Math.min(width, (clientX - bounds.left) * width / bounds.width)), y: Math.max(0, Math.min(height, (clientY - bounds.top) * height / bounds.height)) };
}
