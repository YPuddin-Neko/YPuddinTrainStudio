export function gpuDeviceLabel(device: string) { return device.startsWith('cuda:') ? `GPU ${device.slice(5)}` : device; }
