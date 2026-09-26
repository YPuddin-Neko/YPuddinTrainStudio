import type { QueueDevices } from '../api/types';

export function gpuDeviceLabel(device: string) { return device.startsWith('cuda:') ? `GPU ${device.slice(5)}` : device.toUpperCase(); }

/** Explicit device requests must retain their identity, even when a device disappears. */
export function gpuSelectionValid(devices: string[], count: number, snapshot: QueueDevices | null) {
  if (!Number.isInteger(count) || count < 1 || count > 64) return false;
  if (devices.length) return devices.length === count && new Set(devices).size === devices.length
    && snapshot !== null && devices.every(id => snapshot.devices.some(device => device.device === id && device.status !== 'unavailable'));
  // Automatic single-device execution also supports a CPU-only environment.
  return !snapshot || count === 1 || snapshot.devices.filter(device => device.status !== 'unavailable').length >= count;
}
