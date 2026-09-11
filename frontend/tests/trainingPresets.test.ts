import { describe, expect, it } from 'vitest';
import { applyTrainingPreset, reusableTrainingPreset } from '../src/utils/trainingPresets';

describe('version-safe training presets', () => {
  it('applies training choices while keeping current data, captions, cache and run paths', () => {
    const source = { path: '/current/images', caption_ext: '.caption', repeats: 3 };
    const current = {
      dataset: { sources: [source], cache_dir: '/current/cache', batch_size: 1 },
      validation: { sources: [{ path: '/current/validation' }] },
      checkpoint: { output_dir: '/current/runs', resume: null },
      logging: { events_path: null },
    };
    const oldPreset = {
      dataset: { sources: [{ path: '/old/images' }], cache_dir: '/old/cache', batch_size: 4 },
      validation: { sources: [{ path: '/old/validation' }], enabled: true },
      checkpoint: { output_dir: '/old/runs', resume: '/old/state', save_every_epochs: 2 },
      logging: { events_path: '/old/events.jsonl', level: 'debug' },
      optimizer: { lr: 0.001 },
    };
    const merged = applyTrainingPreset(current, oldPreset);
    expect(merged.dataset).toEqual({ sources: [source], cache_dir: '/current/cache', batch_size: 4 });
    expect(merged.validation.sources).toEqual(current.validation.sources);
    expect(merged.checkpoint).toEqual({ output_dir: '/current/runs', resume: null, save_every_epochs: 2 });
    expect(merged.logging).toEqual({ events_path: null, level: 'debug' });
    expect(merged.optimizer.lr).toBe(0.001);
    expect(oldPreset.dataset.sources[0].path).toBe('/old/images');
  });

  it('saves reusable choices without version paths and does not mutate the current configuration', () => {
    const config = { dataset: { sources: [{ path: '/version' }], caption: { trigger_word: 'subject' } }, checkpoint: { resume: '/state' } };
    expect(reusableTrainingPreset(config)).toEqual({ dataset: { caption: { trigger_word: 'subject' } }, checkpoint: {} });
    expect(config.dataset.sources).toHaveLength(1);
  });
});
