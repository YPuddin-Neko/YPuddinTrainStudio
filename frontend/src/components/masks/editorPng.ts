import { apiUrl } from '../../api/client';
import { ApiError } from '../../api/types';

type Size = { width: number; height: number };
type Layer = 'image' | 'mask';
type Pixels = Uint8ClampedArray | Uint8Array;

// Bound canvas readback and ImageData allocation independently of the edited size.
const BLOCK_PIXELS = 16 * 1024 * 1024;
const rowsPerBlock = (width: number, height: number) => Math.min(height, Math.max(1, Math.floor(BLOCK_PIXELS / width)));

export function readEditorPng(endpoint: string, size: Size, signal: AbortSignal, layer: 'image'): Promise<Uint8ClampedArray>;
export function readEditorPng(endpoint: string, size: Size, signal: AbortSignal, layer: 'mask'): Promise<Uint8Array>;
export async function readEditorPng(endpoint: string, { width, height }: Size, signal: AbortSignal, layer: Layer): Promise<Pixels> {
  const response = await fetch(apiUrl(endpoint), { signal, cache: 'no-store' });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(response.status, payload?.error || { code: layer === 'mask' ? 'mask.load' : 'paint.load', message: `HTTP ${response.status}` });
  }
  const url = URL.createObjectURL(await response.blob());
  const image = new Image();
  try {
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error(`Cannot decode ${layer} PNG`));
      image.src = url;
    });
    signal.throwIfAborted();
    if (image.naturalWidth !== width || image.naturalHeight !== height) throw new Error('Image changed during loading; reload the editor');

    const canvas = document.createElement('canvas');
    const blockHeight = rowsPerBlock(width, height);
    canvas.width = width;
    canvas.height = blockHeight;
    try {
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Canvas is unavailable');
      context.globalCompositeOperation = 'copy';
      const read = (top: number, rows: number) => {
        context.drawImage(image, 0, top, width, rows, 0, 0, width, rows);
        return context.getImageData(0, 0, width, rows).data;
      };
      if (layer === 'image' && blockHeight === height) return read(0, height);

      const pixels = layer === 'image' ? new Uint8ClampedArray(width * height * 4) : new Uint8Array(width * height);
      for (let top = 0; top < height; top += blockHeight) {
        const rows = Math.min(blockHeight, height - top);
        const rgba = read(top, rows);
        if (layer === 'image') pixels.set(rgba, top * width * 4);
        else for (let index = 0; index < width * rows; index++) pixels[top * width + index] = rgba[index * 4];
      }
      return pixels;
    } finally {
      canvas.width = 1;
      canvas.height = 1;
    }
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function encodeEditorPng(width: number, height: number, pixels: Pixels, layer: Layer): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  try {
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas is unavailable');
    const blockHeight = rowsPerBlock(width, height);
    for (let top = 0; top < height; top += blockHeight) {
      const rows = Math.min(blockHeight, height - top);
      const output = context.createImageData(width, rows);
      if (layer === 'image') {
        output.data.set(pixels.subarray(top * width * 4, (top + rows) * width * 4));
      } else {
        for (let index = 0; index < width * rows; index++) {
          const offset = index * 4;
          output.data[offset] = output.data[offset + 1] = output.data[offset + 2] = pixels[top * width + index];
          output.data[offset + 3] = 255;
        }
      }
      context.putImageData(output, 0, top);
    }
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob(blob => {
      if (blob) resolve(blob); else reject(new Error(`Could not encode ${layer} PNG`));
    }, 'image/png'));
  } finally {
    canvas.width = 1;
    canvas.height = 1;
  }
}
