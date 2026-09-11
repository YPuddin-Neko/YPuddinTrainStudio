import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import schema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';

const initial = { adapter: { algo: 'lora', dropout: 0.2, rank_dropout: 0, module_dropout: 1 }, dataset: { caption: { tag_dropout: 0.35, caption_dropout: 0 } }, scheduler: { min_lr_ratio: 0.1, warmup_steps: 0.05 }, loop: { ema: true, ema_decay: 0.999 }, validation: { enabled: true, split_ratio: 0.2 } };
function Editor() {
  const [value, setValue] = React.useState<Record<string, any>>(initial);
  const [advanced, setAdvanced] = React.useState(false);
  return <><button onClick={() => setAdvanced(!advanced)}>Advanced</button><SchemaForm schema={schema} value={value} onChange={setValue} compact showAdvanced={advanced}/><output data-testid="config-value">{JSON.stringify(value)}</output></>;
}
const config = () => JSON.parse(screen.getByTestId('config-value').textContent!);
const field = (name: string) => within(screen.getByTestId(`field-${name}`));
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

describe('percentage controls', () => {
  it('uses the JSON schema bounds, synchronizes slider and number, and stores fractions exactly once', () => {
    render(<Editor/>);
    const tag = field('dataset.caption.tag_dropout');
    expect(tag.getByRole('slider')).toHaveValue('35');
    expect(tag.getByRole('spinbutton')).toHaveValue(35);
    expect(tag.getByRole('slider')).toHaveAttribute('min', '0');
    expect(tag.getByRole('slider')).toHaveAttribute('max', '100');
    expect(tag.getByRole('spinbutton')).toHaveAccessibleDescription('%');
    expect(tag.getAllByText('%')).toHaveLength(1);
    fireEvent.change(tag.getByRole('slider'), { target: { value: '100' } });
    expect(tag.getByRole('spinbutton')).toHaveValue(100);
    expect(config().dataset.caption.tag_dropout).toBe(1);
    fireEvent.change(tag.getByRole('spinbutton'), { target: { value: '12.5' } });
    expect(tag.getByRole('slider')).toHaveValue('12.5');
    expect(config().dataset.caption.tag_dropout).toBe(0.125);
    fireEvent.change(tag.getByRole('spinbutton'), { target: { value: '0' } });
    expect(config().dataset.caption.tag_dropout).toBe(0);
    expect(tag.getByRole('slider')).toHaveValue('0');
    expect(config().scheduler.warmup_steps).toBe(0.05);
  });

  it('keeps advanced and basic probabilities consistent without changing decay or mixed-unit fields', () => {
    render(<Editor/>);
    expect(screen.queryByTestId('field-adapter.dropout')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Advanced'));
    for (const [name, expected] of Object.entries({ 'adapter.dropout': 20, 'adapter.rank_dropout': 0, 'adapter.module_dropout': 100, 'scheduler.min_lr_ratio': 10, 'validation.split_ratio': 20 })) {
      expect(field(name).getByRole('spinbutton')).toHaveValue(expected);
      expect(field(name).getByRole('slider')).toHaveValue(String(expected));
    }
    expect(field('loop.ema_decay').getByRole('spinbutton')).toHaveValue(0.999);
    expect(field('scheduler.warmup_steps').getByRole('spinbutton')).toHaveValue(0.05);
    expect(field('scheduler.warmup_steps').queryByRole('slider')).not.toBeInTheDocument();
    fireEvent.change(field('adapter.dropout').getByRole('spinbutton'), { target: { value: '45' } });
    fireEvent.click(screen.getByText('Advanced'));
    fireEvent.click(screen.getByText('Advanced'));
    expect(field('adapter.dropout').getByRole('spinbutton')).toHaveValue(45);
    expect(config().adapter.dropout).toBe(0.45);
    // Hidden fields leave no empty placeholder groups/fields in the compact layout.
    for (const group of screen.getByTestId('schema-form').querySelectorAll('.config-group')) {
      expect(group.querySelectorAll('.config-field').length).toBeGreaterThan(0);
      expect(group.parentElement).toBe(screen.getByTestId('schema-form'));
    }
  });
});
