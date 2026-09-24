import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { SchemaForm } from '../../../frontend/src/schema/SchemaForm/SchemaForm';
import trainSchema from '../../../frontend/src/schema/train-schema.json';
import '../../../frontend/src/i18n';

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

  it.each([false,true])('hides cloud logging controls while retaining historical values in compact=%s', compact => {
    const legacy = {project:'old-project',run_name:'old-run',entity:'old-team'};
    function LegacyEditor() { const [value,setValue]=React.useState({logging:{level:'info',wandb:legacy}}); return <><SchemaForm schema={trainSchema} value={value} onChange={setValue as any} compact={compact} showAdvanced/><output data-testid="legacy-value">{JSON.stringify(value)}</output></>; }
    render(<LegacyEditor/>);
    expect(screen.queryByTestId('field-logging.wandb')).not.toBeInTheDocument();
    expect(screen.queryByRole('switch',{name:'logging.wandb.unset'})).not.toBeInTheDocument();
    expect(screen.queryByText('Weights & Biases')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('combobox',{name:'日志级别'}));fireEvent.click(screen.getByRole('option',{name:'debug · 最详细'}));
    expect(JSON.parse(screen.getByTestId('legacy-value').textContent!).logging).toEqual({level:'debug',wandb:legacy});
  });

  it('still supports nullable objects independently of retired cloud logging controls', () => {
    const schema={properties:{options:{anyOf:[{type:'object',properties:{name:{type:'string'}}},{type:'null'}],default:null}}};
    function OptionalEditor() {const [value,setValue]=React.useState<Record<string,any>>({options:null});return <><SchemaForm schema={schema} value={value} onChange={setValue}/><output data-testid="optional-value">{JSON.stringify(value)}</output></>;}
    render(<OptionalEditor/>);
    fireEvent.click(screen.getByRole('combobox',{name:'options.mode'}));
    fireEvent.click(screen.getByRole('option',{name:'填写参数'}));
    fireEvent.change(screen.getByRole('textbox',{name:'options.name'}),{target:{value:'local-option'}});
    expect(JSON.parse(screen.getByTestId('optional-value').textContent!).options).toEqual({name:'local-option'});
    fireEvent.click(screen.getByRole('combobox',{name:'options.mode'}));
    fireEvent.click(screen.getByRole('option',{name:'不使用'}));
    expect(JSON.parse(screen.getByTestId('optional-value').textContent!).options).toBeNull();
  });

  it('renders a newly registered family even when the bundled schema has no enum entry', () => {
    render(<SchemaForm schema={trainSchema} value={{ model: { family: 'future-family' } }} onChange={() => {}}
      families={[{ name: 'future-family', label: 'Future family' } as any]} />);
    expect(screen.getByRole('combobox', { name: 'model.family' })).toHaveTextContent('Future family');
    fireEvent.click(screen.getByRole('combobox',{name:'model.family'}));
    expect(screen.getByRole('option', { name: 'Future family' })).toBeInTheDocument();
  });

  it('retains zero and numeric enum types with keyboard selection and nullable reset', () => {
    const schema={properties:{mode:{type:'integer',enum:[0,1,2]},optional:{anyOf:[{type:'integer',enum:[0,1]},{type:'null'}]}}};
    function EnumEditor(){const [value,setValue]=React.useState<Record<string,any>>({mode:0,optional:null});return <><SchemaForm schema={schema} value={value} onChange={setValue}/><output data-testid="enum-value">{JSON.stringify(value)}</output></>;}
    render(<EnumEditor/>);
    const mode=screen.getByRole('combobox',{name:'mode'});
    expect(mode).toHaveTextContent('0');
    fireEvent.keyDown(mode,{key:'ArrowDown'});fireEvent.keyDown(mode,{key:'End'});fireEvent.keyDown(mode,{key:'Enter'});
    expect(JSON.parse(screen.getByTestId('enum-value').textContent!).mode).toBe(2);
    fireEvent.click(screen.getByRole('combobox',{name:'optional'}));fireEvent.click(screen.getByRole('option',{name:'0'}));
    expect(JSON.parse(screen.getByTestId('enum-value').textContent!).optional).toBe(0);
    fireEvent.click(screen.getByRole('combobox',{name:'optional'}));
    fireEvent.click(screen.getByRole('option',{name:'自动'}));
    expect(JSON.parse(screen.getByTestId('enum-value').textContent!).optional).toBeNull();
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

it('does not write defaults just because an inherited numeric field received focus', () => {
  const changed = vi.fn();
  render(<SchemaForm schema={trainSchema} value={{dataset:{bucket_step:null}}} onChange={changed} showAdvanced compact groupFilter={['dataset']}/>);
  const field = screen.getByRole('spinbutton',{name:'dataset.bucket_step'});
  expect(field).toHaveAttribute('placeholder','自动对齐');
  fireEvent.focus(field);fireEvent.blur(field);
  expect(changed).not.toHaveBeenCalled();
});
