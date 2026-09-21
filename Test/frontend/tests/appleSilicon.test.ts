import { describe, it, expect } from 'vitest';
import { isAppleSilicon } from '../../../frontend/src/api/types';

describe('isAppleSilicon', () => {
  it('macOS arm64 返回 true', () => {
    expect(isAppleSilicon({ platform: 'macOS-15.7.9-arm64-arm-64bit' })).toBe(true);
    expect(isAppleSilicon({ platform: 'Darwin-24.0.0-arm64' })).toBe(true);
  });

  it('Intel Mac / Linux / Windows 返回 false', () => {
    expect(isAppleSilicon({ platform: 'macOS-12.0-x86_64' })).toBe(false);
    expect(isAppleSilicon({ platform: 'Linux-6.1-x86_64' })).toBe(false);
    expect(isAppleSilicon({ platform: 'Windows-11-amd64' })).toBe(false);
  });

  it('空值兜底 false', () => {
    expect(isAppleSilicon(null)).toBe(false);
    expect(isAppleSilicon(undefined)).toBe(false);
    expect(isAppleSilicon({})).toBe(false);
  });
});
