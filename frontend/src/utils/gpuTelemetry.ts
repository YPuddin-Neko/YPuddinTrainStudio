import type { TFunction } from 'i18next';
import type { GpuStats } from '../api/types';

export const knownGpuReading = (value: number | null | undefined): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function formatGpuPower(gpu: GpuStats | undefined): string {
  const value = gpu?.power_w;
  if (!knownGpuReading(value)) return '—';
  if (gpu?.kind !== 'mps') return String(Math.round(value));
  if (value > 0 && value < 0.01) return '<0.01';
  return String(Number(value.toFixed(value < 10 ? 2 : 1)));
}

export function formatGpuTemperature(gpu: GpuStats | undefined): string {
  const value = gpu?.temp_c;
  if (!knownGpuReading(value)) return '—';
  return String(gpu?.kind === 'mps' ? Number(value.toFixed(1)) : Math.round(value));
}

export function gpuPowerDescription(gpu: GpuStats | undefined, t: TFunction): string | undefined {
  if (!knownGpuReading(gpu?.power_w)) return t('hardware.gpuPowerMissing');
  if (gpu.power_source === 'hwmon') return t(gpu.kind === 'rocm' ? 'hardware.rocmDevicePowerScope' : 'hardware.driverDevicePowerScope');
  if (gpu?.kind !== 'mps') return undefined;
  const interval = knownGpuReading(gpu.power_sample_seconds) && gpu.power_sample_seconds > 0
    ? t('hardware.gpuPowerSampleWindow', { seconds: Number(gpu.power_sample_seconds.toFixed(2)) })
    : t('hardware.gpuPowerSampleWindowMissing');
  return [t('hardware.appleGpuEstimatedPowerScope'), gpu.power_source === 'ioreport' ? t('hardware.appleGpuPowerSource') : '', interval].filter(Boolean).join(' ');
}

export function gpuTemperatureDescription(gpu: GpuStats | undefined, t: TFunction): string | undefined {
  if (!knownGpuReading(gpu?.temp_c)) return t('hardware.gpuTemperatureMissing');
  if (gpu.temperature_source === 'hwmon-edge') return t('hardware.driverEdgeTemperatureScope');
  if (gpu?.kind !== 'mps') return undefined;
  const maximum = knownGpuReading(gpu.temp_max_c) ? t('hardware.gpuTemperatureMaximum', { value: Number(gpu.temp_max_c.toFixed(1)) }) : t('hardware.gpuTemperatureMaximumMissing');
  const count = knownGpuReading(gpu.temp_sensor_count) && Number.isInteger(gpu.temp_sensor_count) ? t('hardware.gpuTemperatureSensorCount', { count: gpu.temp_sensor_count }) : t('hardware.gpuTemperatureSensorCountMissing');
  return [t('hardware.appleGpuTemperatureScope'), gpu.temperature_source === 'smc' ? t('hardware.appleGpuTemperatureSource') : '', maximum, count].filter(Boolean).join(' ');
}
