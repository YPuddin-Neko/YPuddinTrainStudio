import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import type { FamilyInfo } from '../src/api/types';
import trainSchema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';

// The server schema includes this internal field; keep the assertion meaningful
// even when the bundled fallback schema predates its addition.
const schema = {
  ...trainSchema,
  $defs: {
    ...trainSchema.$defs,
    SamplingConfig: {
      ...trainSchema.$defs.SamplingConfig,
      properties: {
        ...trainSchema.$defs.SamplingConfig.properties,
        output_dir: { anyOf: [{ type: 'string' }, { type: 'null' }], default: null, title: 'Output Dir' },
      },
    },
  },
};
const family = {
  name: 'anima', label: 'Anima', architecture: 'anima', adapter_prefix: 'lora',
  objective: 'rectified_flow', sampling_samplers: ['euler', 'heun', 'er_sde'], sampling_schedulers: ['uniform', 'simple', 'sgm_uniform', 'normal'],
  objective_timestep_sampling: ['uniform', 'logit_normal', 'shift', 'resolution_shift', 'mode', 'cosmap'], objective_weighting: ['none', 'sigma_sqrt', 'cosmap', 'snr_like', 'cosmos'],
  capabilities: [], text_modes: [], presets: [], default_preset: '',
  sampling: { steps: 25, cfg: 4, shift: 3, sampler: 'euler' },
  latent: { channels: 16, stride: 8, patch: 2, align: 16 },
  text_max_len: 512, weights: [], linear_modules: 1,
} satisfies FamilyInfo;

function Editor({ initial, compact = true, selectedFamily = family }: {
  initial: Record<string, unknown>; compact?: boolean; selectedFamily?: FamilyInfo | null;
}) {
  const [value, setValue] = React.useState(initial);
  return <>
    <SchemaForm schema={schema} value={value} onChange={setValue} showAdvanced compact={compact}
      family={selectedFamily ?? undefined} groupFilter={['sampling']} />
    <output data-testid="sampling-config">{JSON.stringify(value)}</output>
  </>;
}
const current = () => JSON.parse(screen.getByTestId('sampling-config').textContent!).sampling;
const input = (key: string) => within(screen.getByTestId(`field-sampling.${key}`)).getByRole('spinbutton');
const enable = () => within(screen.getByTestId('field-sampling.enabled')).getByRole('checkbox');

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

describe('sampling form state and service-owned paths', () => {
  it.each([false, true])('toggles sampling without losing imported settings or exposing its assigned output in compact=%s', compact => {
    const sampling = {
      enabled: true, output_dir: '/studio/project/chara/v2/samples/j_01',
      every_steps: 50, every_epochs: null, at_start: true,
      width: 768, height: 512, steps: 12, cfg: 2.5, shift: null, seed: 0,
      prompts_file: '/prompts/custom.txt', sampler: 'euler',
      prompts: [{ prompt: 'chara, sunset', negative: 'blur', seed: 7, steps: 8 }],
    };
    render(<Editor compact={compact} initial={{ sampling, checkpoint: { output_dir: '/weights' } }} />);
    expect(screen.queryByTestId('field-sampling.output_dir')).not.toBeInTheDocument();
    expect(screen.queryByDisplayValue(sampling.output_dir)).not.toBeInTheDocument();
    fireEvent.change(input('steps'), { target: { value: '16' } });
    fireEvent.click(enable());
    expect(screen.queryByTestId('field-sampling.steps')).not.toBeInTheDocument();
    expect(screen.queryByTestId('prompts-editor')).not.toBeInTheDocument();
    expect(current()).toEqual({ ...sampling, enabled: false, steps: 16 });
    fireEvent.click(enable());
    expect(input('steps')).toHaveValue(16);
    expect(screen.getByDisplayValue('chara, sunset')).toBeInTheDocument();
    expect(current()).toEqual({ ...sampling, steps: 16 });
    expect(screen.queryByTestId('field-sampling.output_dir')).not.toBeInTheDocument();
    expect(JSON.parse(screen.getByTestId('sampling-config').textContent!).checkpoint.output_dir).toBe('/weights');
  });

  it.each(['steps', 'every_steps', 'every_epochs'])('restores nullable %s to a valid positive integer, then preserves an explicit null', key => {
    render(<Editor selectedFamily={null} initial={{ sampling: { enabled: true, [key]: null } }} />);
    const unset = screen.getByRole('checkbox', { name: `sampling.${key}.unset` });
    expect(unset).toBeChecked();
    fireEvent.click(unset);
    expect(current()[key]).toBe(1);
    expect(input(key)).toHaveValue(1);
    expect(input(key)).toBeValid();
    fireEvent.change(input(key), { target: { value: '9' } });
    expect(current()[key]).toBe(9);
    fireEvent.click(unset);
    expect(current()[key]).toBeNull();
    expect(input(key)).toHaveValue(null);
  });

  it('restores nullable shift above its exclusive minimum', () => {
    render(<Editor selectedFamily={null} initial={{ sampling: { enabled: true, shift: null } }} />);
    fireEvent.click(screen.getByRole('checkbox', { name: 'sampling.shift.unset' }));
    expect(current().shift).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('checkbox', { name: 'sampling.shift.unset' }));
    expect(current().shift).toBeNull();
  });

  it('uses each displayed numeric family default when disabling inheritance', () => {
    render(<Editor initial={{ sampling: { enabled: true, steps: null, cfg: null, shift: null } }} />);
    for (const [key, expected] of Object.entries({ steps: 25, cfg: 4, shift: 3 })) {
      fireEvent.click(screen.getByRole('checkbox', { name: `sampling.${key}.unset` }));
      expect(current()[key]).toBe(expected);
      expect(input(key)).toHaveValue(expected);
    }
  });

  it('shows changing family defaults as placeholders without writing them to inherited values', () => {
    const initial = { sampling: { enabled: true, steps: null, cfg: null, shift: null, prompts: [] } };
    const { rerender } = render(<Editor initial={initial} />);
    expect(input('steps')).toHaveAttribute('placeholder', '25');
    expect(input('cfg')).toHaveAttribute('placeholder', '4');
    expect(input('shift')).toHaveAttribute('placeholder', '3');
    rerender(<Editor initial={initial} selectedFamily={{ ...family, name: 'krea2', sampling: { steps: 28, cfg: 5.5, shift: null, sampler: 'euler' } }} />);
    expect(input('steps')).toHaveAttribute('placeholder', '28');
    expect(input('cfg')).toHaveAttribute('placeholder', '5.5');
    expect(input('shift')).toHaveAttribute('placeholder', expect.stringContaining('自动'));
    expect(current()).toEqual(initial.sampling);
    fireEvent.change(input('cfg'), { target: { value: '0' } });
    expect(current().cfg).toBe(0);
    expect(input('cfg')).toHaveValue(0);
    fireEvent.change(input('cfg'), { target: { value: '' } });
    expect(current().cfg).toBeNull();
    expect(input('cfg')).toHaveAttribute('placeholder', '5.5');
  });
});
