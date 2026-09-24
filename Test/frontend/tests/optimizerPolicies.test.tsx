import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { SchemaForm } from '../../../frontend/src/schema/SchemaForm/SchemaForm';
import { normalizeOptimizerConfig } from '../../../frontend/src/utils/optimizerCapabilities';
import schema from '../../../frontend/src/schema/train-schema.json';
import i18n from '../../../frontend/src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
function Editor({initial}: {initial: Record<string, any>}) {
  const [value, setValue] = React.useState(initial);
  return <><SchemaForm schema={schema} value={value} onChange={setValue} compact groupFilter={['optimizer', 'scheduler']}/><output data-testid="configuration">{JSON.stringify(value)}</output></>;
}
const value = () => JSON.parse(screen.getByTestId('configuration').textContent || '{}');
function select(name: string, option: string) {
  fireEvent.click(screen.getByRole('combobox', {name}));
  fireEvent.click(screen.getByRole('option', {name: option}));
}

it('shows automatic values for old PPSF drafts without saving or losing legacy D Coef', () => {
  const changed = vi.fn();
  render(<SchemaForm schema={schema} value={{optimizer: {type: 'prodigy_plus_sf', lr: .0001, args: {d_coef: 2}}, scheduler: {type: 'cosine', warmup_steps: 50}}} onChange={changed} compact groupFilter={['optimizer', 'scheduler']}/>);
  const rate = screen.getByTestId('field-optimizer.lr');
  expect(within(rate).queryByRole('spinbutton')).not.toBeInTheDocument();
  expect(within(rate).getByRole('status', {name: '学习率'})).toHaveTextContent('1');
  expect(screen.getByRole('spinbutton', {name: '自适应步长倍率（D Coef）'})).toHaveValue(2);
  expect(screen.queryByRole('combobox', {name: '学习率调度'})).not.toBeInTheDocument();
  expect(screen.getByRole('spinbutton', {name: '权重平均 β1'})).toHaveValue(.9);
  expect(screen.getByRole('switch', {name: '免调度权重平均'})).toBeChecked();
  expect(changed).not.toHaveBeenCalled();
  for (const path of ['optimizer.kahan','optimizer.group_lr','scheduler.warmup_steps','scheduler.min_lr_ratio']) expect(screen.queryByTestId(`field-${path}`)).not.toBeInTheDocument();
});

it('keeps all PPSF typed parameters visible without advanced mode', () => {
  render(<Editor initial={{optimizer: {type: 'prodigy_plus_sf'}}}/>);
  for (const name of schema['x-optimizer-capabilities'].prodigy_plus_sf.typed_fields) {
    expect(screen.getByTestId(`field-optimizer.${name}`)).toBeInTheDocument();
  }
  expect(screen.queryByTestId('field-optimizer.args')).not.toBeInTheDocument();
  expect(screen.queryByTestId('field-optimizer.fused_backward')).not.toBeInTheDocument();
});

it('restores manual optimizer learning rates and scheduler when switching back', () => {
  render(<Editor initial={{optimizer: {type: 'adamw', lr: .0003, betas: [.8, .98]}, scheduler: {type: 'linear', warmup_steps: 7}}}/>);
  select('优化器', 'Prodigy Plus Schedule-Free');
  expect(value().optimizer).toMatchObject({type: 'prodigy_plus_sf', lr: 1, betas: [.9, .99], d_coef: 1});
  expect(value().scheduler).toMatchObject({type: 'constant', warmup_steps: 0});
  fireEvent.change(screen.getByRole('spinbutton', {name: '自适应步长倍率（D Coef）'}), {target: {value: '1.5'}});
  select('优化器', 'AdamW');
  expect(value().optimizer).toMatchObject({type: 'adamw', lr: .0003, betas: [.8, .98]});
  expect(value().scheduler).toMatchObject({type: 'linear', warmup_steps: 7});
  select('优化器', 'Prodigy Plus Schedule-Free');
  expect(value().optimizer.d_coef).toBe(1.5);
});

it('makes external scheduling editable when PPSF weight averaging is turned off', () => {
  render(<Editor initial={{optimizer: {type: 'prodigy_plus_sf', use_schedulefree: true}}}/>);
  fireEvent.click(screen.getByRole('switch', {name: '免调度权重平均'}));
  expect(screen.getByRole('spinbutton', {name: '方向平滑 β1'})).toBeInTheDocument();
  expect(screen.queryByRole('spinbutton', {name: '权重平均 β1'})).not.toBeInTheDocument();
  select('学习率调度', 'cosine');
  expect(value().scheduler.type).toBe('cosine');
  expect(value().optimizer.lr).toBe(1);
});

it('locks Automagic initialization while exposing its adjustable bounds', () => {
  render(<Editor initial={{optimizer: {type: 'adamw', lr: .0001}}}/>);
  select('优化器', 'Automagic');
  expect(value().optimizer).toMatchObject({type: 'automagic', lr: .000001, eps: 1e-30, grad_clip_norm: 0});
  expect(screen.getByRole('spinbutton', {name: '自适应学习率下限'})).toBeEnabled();
  expect(screen.getByRole('spinbutton', {name: '自适应学习率上限'})).toBeEnabled();
  expect(screen.getByRole('spinbutton', {name: '学习率调整增量'})).toBeEnabled();
  expect(screen.queryByTestId('field-optimizer.d_coef')).not.toBeInTheDocument();
});

it('normalizes known class aliases and preserves conflicts for backend validation', () => {
  const next = normalizeOptimizerConfig(schema, {optimizer: {type: 'prodigyplus.ProdigyPlusScheduleFree', d_coef: 3, args: {d_coef: 2, lr: .0001, future_option: true}}});
  expect(next.optimizer).toMatchObject({type: 'prodigy_plus_sf', lr: 1, d_coef: 3, args: {d_coef: 2, lr: .0001, future_option: true}});
  const migrated = normalizeOptimizerConfig(schema, {optimizer: {type: 'prodigy_plus_sf', d_coef: 1, args: {d_coef: 2}}});
  expect(migrated.optimizer.d_coef).toBe(2);
  expect(migrated.optimizer.args).toEqual({});
});

it('clears per-layer and per-group rate bypasses and explains the disabled layer control', () => {
  const initial = {optimizer: {type: 'prodigy', group_lr: {te: .001}}, adapter: {algo: 'lokr', lr_scale: {te: 2}, rules: [{match: 'layers.*', lr: .002}]}};
  expect(normalizeOptimizerConfig(schema, initial)).toMatchObject({optimizer: {lr: 1, group_lr: {}}, adapter: {lr_scale: {}, rules: [{match: 'layers.*', lr: null}]}});
  render(<SchemaForm schema={schema} value={initial} onChange={vi.fn()} compact showAdvanced groupFilter={['adapter']}/>);
  expect(screen.getByRole('spinbutton', {name: 'LR 1'})).toBeDisabled();
  expect(screen.getByRole('spinbutton', {name: 'LR 1'})).toHaveAttribute('title', expect.stringContaining('统一管理步长'));
});

it('does not retain one preset’s temporary optimizer values when a different draft is loaded', () => {
  const changed = vi.fn();
  const view = render(<SchemaForm schema={schema} value={{optimizer: {type: 'adamw', lr: .005}}} onChange={changed} compact groupFilter={['optimizer']}/>);
  select('优化器', 'Prodigy');
  view.rerender(<SchemaForm schema={schema} value={{optimizer: {type: 'prodigy_plus_sf', d_coef: 4}}} onChange={changed} compact groupFilter={['optimizer']}/>);
  select('优化器', 'AdamW');
  expect(changed.mock.lastCall?.[0].optimizer.lr).toBe(.0001);
});

it('preserves custom-class constructor arguments and unsupported built-in arguments', () => {
  const custom = {optimizer: {type: 'torch.optim.AdamW', lr: .0003, args: {betas: [.7, .8], eps: 1e-7, weight_decay: .2}}};
  expect(normalizeOptimizerConfig(schema, custom)).toEqual(custom);
  const sgd = {optimizer: {type: 'sgd', args: {betas: [.7, .8], eps: 1e-7}}};
  expect(normalizeOptimizerConfig(schema, sgd)).toEqual(sgd);
});

it('uses English labels and explanations throughout the optimizer section', async () => {
  await i18n.changeLanguage('en');
  const {container, rerender} = render(<SchemaForm schema={schema} value={{optimizer: {type: 'prodigy_plus_sf'}}} onChange={vi.fn()} compact groupFilter={['optimizer']}/>);
  expect(container.querySelector('[data-group="optimizer"]')?.textContent).not.toMatch(/[\u3400-\u9fff]/);
  expect(screen.getByRole('spinbutton', {name: 'Weight averaging β1'})).toBeInTheDocument();
  expect(screen.getByRole('status', {name: 'Learning rate'})).toHaveTextContent('1');
  rerender(<SchemaForm schema={schema} value={{optimizer: {type: 'automagic'}}} onChange={vi.fn()} compact groupFilter={['optimizer']}/>);
  expect(container.querySelector('[data-group="optimizer"]')?.textContent).not.toMatch(/[\u3400-\u9fff]/);
  expect(screen.getByRole('spinbutton', {name: 'Learning-rate increment'})).toBeInTheDocument();
});

it('hides inactive dependent controls and restores their values when the parent is enabled', () => {
  render(<Editor initial={{optimizer:{type:'prodigy_plus_sf',schedulefree_c:2,split_groups_mean:true,factored_fp32:false}}}/>);
  for (const [toggle, path] of [['免调度权重平均','schedulefree_c'], ['各参数组独立估计步长','split_groups_mean'], ['分解统计以节省显存','factored_fp32']]) {
    const field = screen.getByTestId(`field-optimizer.${path}`);
    expect(field).toBeInTheDocument();
    fireEvent.click(screen.getByRole('switch', {name:toggle}));
    expect(screen.queryByTestId(`field-optimizer.${path}`)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('switch', {name:toggle}));
    expect(screen.getByTestId(`field-optimizer.${path}`)).toBeInTheDocument();
  }
  expect(value().optimizer).toMatchObject({schedulefree_c:2, split_groups_mean:true, factored_fp32:false});
});
