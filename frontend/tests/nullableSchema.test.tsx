import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import trainSchema from '../src/schema/train-schema.json';
import '../src/i18n';

function Editor({ initialValue = { dataset: { caption: { trigger_word: null } }, logging: { wandb: null } } }: { initialValue?: any }) {
  const [value, setValue] = React.useState<any>(initialValue);
  return <><SchemaForm schema={trainSchema} value={value} onChange={setValue} showAdvanced /><output data-testid="value">{JSON.stringify(value)}</output></>;
}

describe('nullable schema editing', () => {
  it('accepts a textual trigger word and preserves an explicit cleared null', () => {
    render(<Editor />);
    const field = screen.getByTestId('field-dataset.caption.trigger_word');
    const input = within(field).getByRole('textbox');
    fireEvent.change(input, { target: { value: 'sora_character' } });
    expect(JSON.parse(screen.getByTestId('value').textContent!).dataset.caption.trigger_word).toBe('sora_character');
    fireEvent.change(input, { target: { value: '' } });
    expect(JSON.parse(screen.getByTestId('value').textContent!).dataset.caption.trigger_word).toBeNull();
  });

  it('enables a nullable W&B object, edits nested strings, and unsets the object', () => {
    render(<Editor />);
    fireEvent.click(screen.getByRole('checkbox', { name: 'logging.wandb.unset' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'logging.wandb.project' }), { target: { value: 'training-project' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'logging.wandb.run_name' }), { target: { value: 'experiment-1' } });
    expect(JSON.parse(screen.getByTestId('value').textContent!).logging.wandb).toEqual({ project: 'training-project', run_name: 'experiment-1' });
    fireEvent.click(screen.getByRole('checkbox', { name: 'logging.wandb.unset' }));
    expect(JSON.parse(screen.getByTestId('value').textContent!).logging.wandb).toBeNull();
  });

  it('renders a newly registered family even when the bundled schema has no enum entry', () => {
    render(<SchemaForm schema={trainSchema} value={{ model: { family: 'future-family' } }} onChange={() => {}}
      families={[{ name: 'future-family', label: 'Future family' } as any]} />);
    expect(screen.getByRole('combobox', { name: 'model.family' })).toHaveValue('future-family');
    expect(screen.getByRole('option', { name: 'Future family' })).toBeInTheDocument();
  });

  it('preserves inherited prompt values, edits steps and CFG, and clears back to null', () => {
    render(<Editor initialValue={{ sampling: { enabled: true, seed: 0, width: 64, height: 64, steps: 20, cfg: 1,
      prompts: [{ prompt: 'imported prompt', steps: 2, seed: null, width: null, height: null, cfg: null }] } }} />);
    const field = (index: number, key: string) => screen.getByRole('spinbutton', { name: `sampling.prompts.${index}.${key}` });
    const value = () => JSON.parse(screen.getByTestId('value').textContent!).sampling;
    for (const key of ['seed', 'width', 'height', 'cfg']) expect(field(0, key)).toHaveValue(null);
    expect(field(0, 'seed')).toHaveAttribute('placeholder', '继承全局设置 (0)');
    expect(field(0, 'width')).toHaveAttribute('placeholder', '继承全局设置 (64)');
    expect(field(0, 'steps')).toHaveValue(2);
    fireEvent.change(field(0, 'steps'), { target: { value: '4' } });
    fireEvent.change(field(0, 'cfg'), { target: { value: '2.5' } });
    fireEvent.change(field(0, 'seed'), { target: { value: '0' } });
    expect(value().prompts[0]).toMatchObject({ steps: 4, cfg: 2.5, seed: 0, width: null, height: null });
    fireEvent.change(field(0, 'steps'), { target: { value: '' } });
    fireEvent.change(field(0, 'cfg'), { target: { value: '' } });
    expect(value().prompts[0]).toMatchObject({ steps: null, cfg: null });
    fireEvent.click(screen.getByTestId('add-prompt'));
    expect(value().prompts[1]).toEqual({ prompt: '', negative: '', seed: null, width: null, height: null, steps: null, cfg: null });
    expect(field(1, 'steps')).toHaveValue(null);
    expect(value().width).toBe(64);
  });
});
