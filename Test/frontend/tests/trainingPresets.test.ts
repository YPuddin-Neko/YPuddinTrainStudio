import { describe, expect, it } from 'vitest';
import { applyTrainingPreset, reusableTrainingPreset } from '../../../frontend/src/utils/trainingPresets';

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

  it('preserves current family and every project-owned model, sampling and resume path', () => {
    const current = {model:{family:'anima',dit_path:'current-dit',text_encoder_path:'current-text',vae_path:'current-vae',tokenizer_path:'current-tokenizer'},adapter:{resume_weights:'current-adapter'},sampling:{output_dir:'current-samples',prompts_file:'current-prompts',steps:25}};
    const old = {model:{family:'krea2',dit_path:'old-dit',text_encoder_path:'old-text',vae_path:'old-vae',tokenizer_path:'old-tokenizer'},adapter:{resume_weights:'old-adapter',rank:16},sampling:{output_dir:'old-samples',prompts_file:'old-prompts',steps:28}};
    expect(reusableTrainingPreset(old)).toEqual({model:{family:'krea2'},adapter:{rank:16},sampling:{steps:28}});
    expect(applyTrainingPreset(current,old)).toEqual({...current,adapter:{resume_weights:'current-adapter',rank:16},sampling:{...current.sampling,steps:28}});
    expect(old.model.dit_path).toBe('old-dit');
  });
});
