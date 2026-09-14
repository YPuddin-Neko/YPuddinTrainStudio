import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import schema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

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
