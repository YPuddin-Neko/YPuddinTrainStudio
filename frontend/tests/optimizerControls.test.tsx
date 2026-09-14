import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import schema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

function NumericEditor({initial}: {initial: Record<string, unknown>}) {
  const [value, setValue] = React.useState(initial);
  return <><SchemaForm schema={schema} value={value} onChange={setValue} compact showAdvanced groupFilter={['optimizer', 'adapter']}/>
    <output data-testid="numeric-configuration">{JSON.stringify(value)}</output></>;
}

it('shows the exact learning rate in scientific notation while preserving decimal and exponent editing', async () => {
  render(<NumericEditor initial={{optimizer:{type:'adamw',lr:0.0001}}}/>);
  const input = screen.getByRole('spinbutton', {name:'学习率'});
  const scientific = screen.getByLabelText('学习率 · 科学计数法');
  expect(scientific).toHaveTextContent(/^1e-4$/);
  for (const [raw, notation] of [['0.2','2e-1'],['2e-5','2e-5'],['0.000123456789','1.23456789e-4']]) {
    await act(async () => { await userEvent.clear(input); });
    expect(scientific).toHaveTextContent('—');
    await act(async () => { await userEvent.type(input, raw); });
    expect(scientific).toHaveTextContent(notation);
    expect(JSON.parse(screen.getByTestId('numeric-configuration').textContent || '{}').optimizer.lr).toBe(Number(raw));
  }
});

it.each(['2.5', '0.25', '1e-6'])('keeps a cleared D Coef empty while entering %s', async (raw) => {
  render(<NumericEditor initial={{optimizer: {type: 'prodigy_plus_sf', d_coef: 1}}}/>);
  const input = screen.getByRole('spinbutton', {name: '自适应步长倍率（D Coef）'});
  await act(async () => { await userEvent.clear(input); });
  expect(input).toHaveValue(null);
  expect(JSON.parse(screen.getByTestId('numeric-configuration').textContent || '{}').optimizer.d_coef).toBe('');
  await act(async () => { await userEvent.type(input, raw); });
  expect(input).toHaveValue(Number(raw));
  expect(JSON.parse(screen.getByTestId('numeric-configuration').textContent || '{}').optimizer.d_coef).toBe(Number(raw));
});

it('allows clearing beta, rank, learning rate and slider numbers before replacement', async () => {
  render(<NumericEditor initial={{optimizer: {type: 'adamw'}, adapter: {algo: 'lokr', rank: 16}}}/>);
  for (const [name, next] of [['方向平滑 β1', '0.85'], ['Rank / 秩', '32'], ['学习率', '1e-5'], ['optimizer.eps', '1e-8'], ['输出丢弃率', '12.5']]) {
    const input = screen.getByRole('spinbutton', {name});
    await act(async () => { await userEvent.clear(input); });
    expect(input).toHaveValue(null);
    await act(async () => { await userEvent.type(input, next); });
    expect(input).toHaveValue(Number(next));
  }
});

it.each([
  ['prodigy_plus_sf', 'eps', 1e-8, '1e-7'],
  ['prodigy_plus_sf', 'beta3', 0.9, '0.95'],
  ['prodigy', 'growth_rate', 1.02, '1.05'],
])('only changes nullable %s %s to automatic mode through its checkbox', async (type, name, initial, raw) => {
  render(<NumericEditor initial={{optimizer: {type, [name]: initial}}}/>);
  const input = screen.getByRole('spinbutton', {name: `optimizer.${name}`});
  const automatic = screen.getByRole('checkbox', {name: `optimizer.${name}.unset`});
  await act(async () => { await userEvent.clear(input); });
  expect(input).toHaveValue(null);
  expect(automatic).not.toBeChecked();
  expect(JSON.parse(screen.getByTestId('numeric-configuration').textContent || '{}').optimizer[name]).toBe('');
  await act(async () => { await userEvent.type(input, String(raw)); });
  expect(input).toHaveValue(Number(raw));
  await act(async () => { await userEvent.click(automatic); });
  expect(automatic).toBeChecked();
  expect(JSON.parse(screen.getByTestId('numeric-configuration').textContent || '{}').optimizer[name]).toBe(null);
  await act(async () => { await userEvent.click(automatic); });
  expect(automatic).not.toBeChecked();
  expect(input).not.toHaveValue(null);
});

it('gives each beta a separate readable name and only changes the chosen value', () => {
  const changed = vi.fn();
  render(<SchemaForm schema={schema} value={{optimizer: {type: 'adamw', betas: [0.9, 0.99]}}} onChange={changed} compact showAdvanced groupFilter={['optimizer']}/>);
  const field = screen.getByTestId('field-optimizer.betas');
  expect(within(field).getByText('更新平滑程度')).toBeInTheDocument();
  expect(within(field).getByRole('spinbutton', {name: '方向平滑 β1'})).toHaveValue(0.9);
  fireEvent.change(within(field).getByRole('spinbutton', {name: '幅度平滑 β2'}), {target: {value: '0.98'}});
  expect(changed).toHaveBeenLastCalledWith({optimizer: {type: 'adamw', betas: [0.9, 0.98]}});
  fireEvent.click(within(field).getByRole('button', {name: '更新平滑程度 说明'}));
  expect(screen.getByText(/通常保留优化器默认值；调大后反应更平缓/)).toBeInTheDocument();
});

it('uses one labelled boolean control across groups, including keyboard activation and read-only', async () => {
  const boolSchema = {type: 'object', properties: {
    optimizer: {type: 'object', properties: {kahan: {type: 'boolean', default: false, description: 'Kahan', 'x-ui': {group: 'optimizer'}}}},
    sampling: {type: 'object', properties: {enabled: {type: 'boolean', default: false, description: 'Preview', 'x-ui': {group: 'sampling'}}}},
  }};
  const changed = vi.fn();
  const value = {optimizer: {kahan: false}, sampling: {enabled: true}};
  const view = render(<SchemaForm schema={boolSchema} value={value} onChange={changed} compact/>);
  for (const name of ['低精度更新补偿', '生成训练预览']) {
    const input = screen.getByRole('checkbox', {name});
    expect(input.closest('label')).toHaveClass('config-toggle-control');
    expect(input.closest('[data-field-path]')).toHaveClass('config-field-boolean');
    expect(input.closest('[data-field-path]')).not.toHaveClass('config-field-toggle');
  }
  const input = screen.getByRole('checkbox', {name: '低精度更新补偿'});
  input.focus();
  await userEvent.keyboard(' ');
  expect(changed).toHaveBeenLastCalledWith({...value, optimizer: {kahan: true}});
  changed.mockClear();
  view.rerender(<SchemaForm schema={boolSchema} value={value} onChange={changed} compact readOnly/>);
  expect(screen.getByRole('checkbox', {name: '低精度更新补偿'})).toBeDisabled();
  await userEvent.click(screen.getByText('未开启'));
  expect(changed).not.toHaveBeenCalled();
});

it('never exposes schema-hidden unsupported controls even in advanced search', () => {
  const hiddenSchema = {type: 'object', properties: {unsupported: {type: 'boolean', description: 'Unsupported', 'x-ui': {hidden: true}}}};
  render(<SchemaForm schema={hiddenSchema} value={{unsupported: true}} onChange={vi.fn()} compact showAdvanced search="unsupported"/>);
  expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
});
