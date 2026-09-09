import { describe, it, expect } from 'vitest';
import { evaluateShowWhen } from '../src/schema/showWhen';

describe('showWhen evaluation suite (≥20 tests)', () => {
  const sampleConfig = {
    adapter: {
      algo: 'lokr',
      rank: 16,
      factor: -1,
      decompose_both: true,
      dora: false,
    },
    optimizer: {
      type: 'adamw8bit',
      lr: 0.0001,
      weight_decay: 0.01,
    },
    dataset: {
      batch_size: 4,
      repeats: 10,
    },
    memory: {
      block_swap: 0,
      fp8_base: true,
    },
    tags: null,
  };

  it('1. basic equality string', () => {
    expect(evaluateShowWhen("adapter.algo == 'lokr'", sampleConfig)).toBe(true);
  });

  it('2. basic inequality string', () => {
    expect(evaluateShowWhen("adapter.algo != 'lora'", sampleConfig)).toBe(true);
  });

  it('3. false equality string', () => {
    expect(evaluateShowWhen("adapter.algo == 'lora'", sampleConfig)).toBe(false);
  });

  it('4. number comparison >', () => {
    expect(evaluateShowWhen('adapter.rank > 8', sampleConfig)).toBe(true);
  });

  it('5. number comparison <=', () => {
    expect(evaluateShowWhen('dataset.batch_size <= 4', sampleConfig)).toBe(true);
  });

  it('6. number comparison < with float', () => {
    expect(evaluateShowWhen('optimizer.lr < 0.001', sampleConfig)).toBe(true);
  });

  it('7. boolean equality true', () => {
    expect(evaluateShowWhen('adapter.decompose_both == true', sampleConfig)).toBe(true);
  });

  it('8. boolean equality false', () => {
    expect(evaluateShowWhen('adapter.dora == false', sampleConfig)).toBe(true);
  });

  it('9. logical AND (&&) true', () => {
    expect(evaluateShowWhen("adapter.algo == 'lokr' && adapter.rank == 16", sampleConfig)).toBe(true);
  });

  it('10. logical AND (&&) false', () => {
    expect(evaluateShowWhen("adapter.algo == 'lokr' && adapter.rank == 32", sampleConfig)).toBe(false);
  });

  it('11. logical OR (||) true', () => {
    expect(evaluateShowWhen("adapter.algo == 'lora' || adapter.rank == 16", sampleConfig)).toBe(true);
  });

  it('12. logical NOT (!) on boolean', () => {
    expect(evaluateShowWhen('!adapter.dora', sampleConfig)).toBe(true);
  });

  it('13. parentheses grouping', () => {
    expect(evaluateShowWhen("(adapter.algo == 'lokr' || adapter.algo == 'lora') && adapter.rank > 0", sampleConfig)).toBe(true);
  });

  it('14. in operator with strings match', () => {
    expect(evaluateShowWhen("adapter.algo in ['lora', 'lokr']", sampleConfig)).toBe(true);
  });

  it('15. in operator with strings no match', () => {
    expect(evaluateShowWhen("adapter.algo in ['lora', 'full']", sampleConfig)).toBe(false);
  });

  it('16. in operator with numbers', () => {
    expect(evaluateShowWhen('adapter.rank in [8, 16, 32]', sampleConfig)).toBe(true);
  });

  it('17. null comparison', () => {
    expect(evaluateShowWhen('tags == null', sampleConfig)).toBe(true);
  });

  it('18. missing field comparison', () => {
    expect(evaluateShowWhen('non.existent.path == null', sampleConfig)).toBe(true);
  });

  it('19. missing field != value evaluates correctly', () => {
    expect(evaluateShowWhen("missing.field != 'test'", sampleConfig)).toBe(true);
  });

  it('20. complex nested expression', () => {
    expect(
      evaluateShowWhen(
        "adapter.algo in ['lokr'] && (optimizer.lr <= 0.0001 || memory.fp8_base == false) && !adapter.dora",
        sampleConfig
      )
    ).toBe(true);
  });

  it('21. negative numbers check', () => {
    expect(evaluateShowWhen('adapter.factor == -1', sampleConfig)).toBe(true);
  });

  it('22. fallback safely on empty or invalid expressions', () => {
    expect(evaluateShowWhen('', sampleConfig)).toBe(true);
    expect(evaluateShowWhen('///invalid syntax///', sampleConfig)).toBe(true);
  });
});
