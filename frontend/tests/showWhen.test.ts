import { describe, it, expect } from 'vitest';
import { evaluateShowWhen, ShowWhenError } from '../src/schema/showWhen';

describe('showWhen evaluation suite (synced 1:1 with backend test_config.py)', () => {
  // Config matching Python's TrainConfig() defaults
  const sampleConfig = {
    adapter: {
      algo: 'lokr',
      rank: 16,
      alpha: 16.0,
      factor: -1,
      decompose_both: false,
      rs_lora: false,
      dora: false,
      init: 'default',
    },
    loop: {
      epochs: 10,
      max_steps: null,
    },
    optimizer: {
      type: 'adamw8bit',
      lr: 0.0001,
      weight_decay: 0.01,
    },
    dataset: {
      batch_size: 4,
      resolutions: [1024],
    },
    sampling: {
      enabled: false,
    },
  };

  // 11 test cases ported from test_config.py @pytest.mark.parametrize("expr,expected")
  const backendCases: [string, boolean][] = [
    ["adapter.algo == 'lokr'", true],
    ["adapter.algo != 'lokr'", false],
    ["adapter.algo in ['lora','lokr']", true],
    ["adapter.rank > 8 && loop.epochs >= 10", true],
    ["!(adapter.dora == true)", true],
    ["loop.max_steps == null", true],
    ["missing.path == null", true],
    ["missing.path > 1", false],
    ["(adapter.algo == 'lora' || adapter.algo == 'lokr') && adapter.rs_lora == false", true],
    ["sampling.enabled == true", false],
    ["dataset.resolutions in [1024]", false],
  ];

  backendCases.forEach(([expr, expected], idx) => {
    it(`backend case ${idx + 1}: ${expr}`, () => {
      expect(evaluateShowWhen(expr, sampleConfig)).toBe(expected);
    });
  });

  // Additional edge cases: double quotes, numbers, nulls
  it('supports double quoted strings', () => {
    expect(evaluateShowWhen('adapter.algo == "lokr"', sampleConfig)).toBe(true);
    expect(evaluateShowWhen('adapter.algo in ["lokr", "lora"]', sampleConfig)).toBe(true);
  });

  it('handles null comparisons rigorously', () => {
    expect(evaluateShowWhen('loop.max_steps == null', sampleConfig)).toBe(true);
    expect(evaluateShowWhen('loop.max_steps != null', sampleConfig)).toBe(false);
    expect(evaluateShowWhen('loop.max_steps < 10', sampleConfig)).toBe(false);
    expect(evaluateShowWhen('loop.max_steps > 10', sampleConfig)).toBe(false);
  });

  it('handles inequality comparisons with null returning false', () => {
    expect(evaluateShowWhen('null < 1', sampleConfig)).toBe(false);
    expect(evaluateShowWhen('1 < null', sampleConfig)).toBe(false);
    expect(evaluateShowWhen('null <= null', sampleConfig)).toBe(false);
  });

  // Error handling tests: invalid syntax must throw ShowWhenError
  it('throws ShowWhenError on invalid operator syntax', () => {
    expect(() => evaluateShowWhen('adapter.algo ==', {})).toThrow(ShowWhenError);
    expect(() => evaluateShowWhen('a.b $ 1', {})).toThrow(ShowWhenError);
    expect(() => evaluateShowWhen('(a == 1', {})).toThrow(ShowWhenError);
    expect(() => evaluateShowWhen('a in [', {})).toThrow(ShowWhenError);
    expect(() => evaluateShowWhen('!', {})).toThrow(ShowWhenError);
  });
});
