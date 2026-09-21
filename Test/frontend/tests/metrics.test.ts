import { describe, it, expect } from 'vitest';
import { shapeValidationSeries, mergeValidationPoint, appendCapped } from '../../../frontend/src/utils/metrics';
import { ValidationPoint } from '../../../frontend/src/api/types';

const v = (step: number, per_t: Record<string, number>, mean: number): ValidationPoint =>
  ({ step, per_t, mean } as ValidationPoint);

describe('shapeValidationSeries', () => {
  it('每个固定时间步一条线 + 均值线', () => {
    const input = [
      v(100, { '0.1': 0.3, '0.5': 0.2 }, 0.25),
      v(200, { '0.1': 0.24, '0.5': 0.16 }, 0.2),
    ];
    const { steps, series } = shapeValidationSeries(input);
    expect(steps).toEqual([100, 200]);
    expect(series.map((s) => s.name)).toEqual(['t=0.1', 't=0.5', 'mean']);
    expect(series[0].data).toEqual([
      [100, 0.3],
      [200, 0.24],
    ]);
    expect(series[2].data).toEqual([
      [100, 0.25],
      [200, 0.2],
    ]);
  });

  it('乱序与重复 step 去重（保留最后一个）', () => {
    const input = [
      v(200, { '0.5': 0.16 }, 0.2),
      v(100, { '0.5': 0.2 }, 0.25),
      v(200, { '0.5': 0.1 }, 0.15),
    ];
    const { steps, series } = shapeValidationSeries(input);
    expect(steps).toEqual([100, 200]);
    // step 200 保留最后一条
    expect(series.find((s) => s.name === 'mean')!.data).toEqual([
      [100, 0.25],
      [200, 0.15],
    ]);
  });

  it('部分点缺失某时间步时跳过该点', () => {
    const input = [
      v(100, { '0.1': 0.3 }, 0.3),
      v(200, { '0.1': 0.2, '0.9': 0.1 }, 0.15),
    ];
    const { series } = shapeValidationSeries(input);
    const t09 = series.find((s) => s.name === 't=0.9')!;
    expect(t09.data).toEqual([[200, 0.1]]);
  });

  it('空输入返回空结构', () => {
    const { steps, series } = shapeValidationSeries([]);
    expect(steps).toEqual([]);
    expect(series.map((s) => s.name)).toEqual(['mean']);
  });
});

describe('mergeValidationPoint', () => {
  it('按 step 去重并保持升序', () => {
    let acc: ValidationPoint[] = [];
    acc = mergeValidationPoint(acc, v(200, {}, 0.2));
    acc = mergeValidationPoint(acc, v(100, {}, 0.3));
    acc = mergeValidationPoint(acc, v(200, {}, 0.99)); // 重复 → 忽略
    expect(acc.map((p) => p.step)).toEqual([100, 200]);
    expect(acc[1].mean).toBe(0.2);
  });
});

describe('appendCapped (日志环形缓存)', () => {
  it('未超上限时正常追加', () => {
    expect(appendCapped([1, 2], [3], 10)).toEqual([1, 2, 3]);
  });

  it('超上限时丢弃最旧行', () => {
    const lines = Array.from({ length: 5 }, (_, i) => i);
    expect(appendCapped(lines, [5, 6], 5)).toEqual([2, 3, 4, 5, 6]);
  });

  it('单次大批量超过上限时只保留末尾 cap 行', () => {
    const incoming = Array.from({ length: 10 }, (_, i) => i);
    const result = appendCapped([], incoming, 4);
    expect(result).toEqual([6, 7, 8, 9]);
  });

  it('5 万行上限语义', () => {
    const lines = Array.from({ length: 50000 }, (_, i) => i);
    const result = appendCapped(lines, [50000, 50001], 50000);
    expect(result.length).toBe(50000);
    expect(result[0]).toBe(2);
    expect(result[49999]).toBe(50001);
  });
});
