import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import schema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
type Draft = {adapter: Record<string, unknown>};
const initial = (algo = 'lokr', rank: number | string = 'full'): Draft => ({adapter: {algo, rank, alpha: 8, factor: 8, dora: true, preset: 'attn-mlp', rules: []}});
function Editor({algo = 'lokr', rank = 'full', compact = true, advanced = false, replacement}: {algo?: string; rank?: number | string; compact?: boolean; advanced?: boolean; replacement?: Draft}) {
  const [value, setValue] = React.useState(initial(algo, rank));
  return <><SchemaForm schema={schema} value={value} onChange={setValue as React.Dispatch<React.SetStateAction<Record<string, unknown>>>} compact={compact} showAdvanced={advanced} groupFilter={['adapter']}/>
    {replacement && <button onClick={() => setValue(structuredClone(replacement))}>加载另一个预设</button>}
    <output data-testid="config-value">{JSON.stringify(value)}</output></>;
}
const config = () => JSON.parse(screen.getByTestId('config-value').textContent!);
function choose(label: string, option: string) {
  fireEvent.click(screen.getByRole('combobox', {name: label}));
  fireEvent.click(screen.getByRole('option', {name: option}));
}
const mode = (option: string) => choose('LoKr 参数形式', option);
const numeric = (key: string) => within(screen.getByTestId(`field-adapter.${key}`)).getByRole('spinbutton');

describe('LoKr parameter mode', () => {
  it('separates mode from numeric Rank and restores edited Rank and Alpha after Full', () => {
    render(<Editor/>);
    expect(screen.getByRole('combobox', {name: 'LoKr 参数形式'})).toHaveTextContent('Full · 完整因子矩阵');
    expect(screen.queryByTestId('field-adapter.rank')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-adapter.alpha')).not.toBeInTheDocument();
    expect(within(screen.getByTestId('field-adapter.parameter_mode')).queryByRole('spinbutton')).not.toBeInTheDocument();
    expect(config().adapter).toMatchObject({algo: 'lokr', rank: 'full', alpha: 8, factor: 8, dora: true, preset: 'attn-mlp', rules: []});
    fireEvent.click(screen.getByRole('button', {name: 'LoKr 参数形式 说明'}));
    expect(screen.getByRole('tooltip')).toHaveTextContent('不是全量微调，也不使用 Alpha');
    fireEvent.keyDown(document, {key: 'Escape'});

    mode('低秩 · 分解因子矩阵');
    expect(config().adapter.rank).toBe(16);
    fireEvent.change(numeric('rank'), {target: {value: '32'}});
    fireEvent.change(numeric('alpha'), {target: {value: '16'}});
    mode('Full · 完整因子矩阵');
    expect(config().adapter).toMatchObject({rank: 'full', alpha: 16, factor: 8, dora: true, preset: 'attn-mlp', rules: []});
    expect(config().adapter).not.toHaveProperty('parameter_mode');
    expect(screen.queryByTestId('field-adapter.alpha')).not.toBeInTheDocument();
    mode('低秩 · 分解因子矩阵');
    expect(numeric('rank')).toHaveValue(32);
    expect(numeric('alpha')).toHaveValue(16);
  });

  it.each([true, false])('groups DoRA with structure and only shows relevant advanced sections in compact=%s', compact => {
    const {rerender} = render(<Editor rank={7} compact={compact}/>);
    const structure = screen.getByRole('heading', {name: '训练结构'}).parentElement!;
    const capacity = screen.getByRole('heading', {name: '参数规模'}).parentElement!;
    for (const key of ['algo', 'preset', 'dora']) expect(structure).toContainElement(screen.getByTestId(`field-adapter.${key}`));
    for (const key of ['parameter_mode', 'factor', 'rank', 'alpha']) expect(capacity).toContainElement(screen.getByTestId(`field-adapter.${key}`));
    expect(screen.getAllByRole('heading').map(node => node.textContent)).toEqual(['训练结构', '参数规模']);
    expect(numeric('rank')).toHaveValue(7);
    rerender(<Editor rank={7} compact={compact} advanced/>);
    expect(screen.getByRole('heading', {name: '初始化与继续训练'}).parentElement).toContainElement(screen.getByTestId('field-adapter.init'));
    expect(screen.getByRole('heading', {name: '训练正则'}).parentElement).toContainElement(screen.getByTestId('field-adapter.dropout'));
    expect(screen.getByRole('heading', {name: '计算与学习率'}).parentElement).toContainElement(screen.getByTestId('field-adapter.mode'));
    expect(capacity).toContainElement(screen.getByTestId('field-adapter.rs_lora'));
    expect(capacity).toContainElement(screen.getByTestId('field-adapter.decompose_both'));
    expect(structure).toContainElement(screen.getByTestId('field-adapter.dora'));
    expect(screen.getByTestId('field-adapter.lr_scale')).toHaveClass('config-field-wide');
    expect(screen.getByTestId('field-adapter.rules')).toHaveClass('config-field-wide');
  });

  it('retains ordinary LoRA Rank and Alpha without LoKr-only modes or a full checkbox', () => {
    render(<Editor algo="lora" rank={7}/>);
    expect(screen.queryByRole('combobox', {name: 'LoKr 参数形式'})).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-adapter.factor')).not.toBeInTheDocument();
    expect(numeric('rank')).toHaveValue(7);
    expect(within(screen.getByTestId('field-adapter.rank')).queryByRole('checkbox')).not.toBeInTheDocument();
    expect(screen.getAllByRole('heading').map(node => node.textContent)).toEqual(['训练结构', '参数规模']);
    expect(screen.getByRole('heading', {name: '训练结构'}).parentElement).toContainElement(screen.getByTestId('field-adapter.dora'));
    fireEvent.change(numeric('alpha'), {target: {value: '4'}});
    expect(config().adapter).toMatchObject({algo: 'lora', rank: 7, alpha: 4, factor: 8});
  });

  it('keeps Full mode and DoRA in separate meaningful sections without an empty tuning group', () => {
    render(<Editor/>);
    const capacity = screen.getByRole('heading', {name: '参数规模'}).parentElement!;
    expect(capacity).toContainElement(screen.getByTestId('field-adapter.parameter_mode'));
    expect(capacity).toContainElement(screen.getByTestId('field-adapter.factor'));
    expect(capacity).not.toContainElement(screen.getByTestId('field-adapter.dora'));
    expect(screen.getAllByRole('heading').map(node => node.textContent)).toEqual(['训练结构', '参数规模']);
  });

  it.each(['lora', 'loha'])('restores a valid remembered numeric rank when switching Full to %s', algo => {
    render(<Editor rank={7}/>);
    mode('Full · 完整因子矩阵');
    choose('适配器算法', algo);
    expect(config().adapter).toMatchObject({algo, rank: 7, alpha: 8, factor: 8, dora: true, preset: 'attn-mlp', rules: []});
    expect(numeric('rank')).toHaveValue(7);
    expect(screen.queryByTestId('field-adapter.factor')).not.toBeInTheDocument();
  });

  it('uses the default numeric rank when imported Full has no local rank history', () => {
    render(<Editor/>);
    choose('适配器算法', 'lora');
    expect(config().adapter).toMatchObject({algo: 'lora', rank: 16, alpha: 8});
  });

  it('does not restore invalid edits and does not transfer rank history into another preset', () => {
    const replacement = initial();
    replacement.adapter.alpha = 23;
    render(<Editor rank={7} replacement={replacement}/>);
    fireEvent.change(numeric('rank'), {target: {value: '0'}});
    mode('Full · 完整因子矩阵');
    mode('低秩 · 分解因子矩阵');
    expect(numeric('rank')).toHaveValue(16);
    fireEvent.change(numeric('rank'), {target: {value: '64'}});
    mode('Full · 完整因子矩阵');
    fireEvent.click(screen.getByRole('button', {name: '加载另一个预设'}));
    mode('低秩 · 分解因子矩阵');
    expect(numeric('rank')).toHaveValue(16);
    expect(numeric('alpha')).toHaveValue(23);
  });

  it('keeps mode searchable, read-only, and locatable by the real rank validation path', () => {
    const onChange = vi.fn();
    const {rerender} = render(<SchemaForm schema={schema} value={initial()} onChange={onChange} compact readOnly groupFilter={['adapter']} search="参数形式" errors={[{loc: 'adapter.rank', msg: '检查当前 Rank 配置'}]}/>);
    expect(screen.getByRole('combobox', {name: 'LoKr 参数形式'})).toBeDisabled();
    expect(screen.getByRole('combobox', {name: 'LoKr 参数形式'})).toHaveAttribute('aria-invalid', 'true');
    expect(document.getElementById('field-adapter.rank')).toBe(screen.getByTestId('field-adapter.parameter_mode'));
    expect(screen.getByText('检查当前 Rank 配置')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name: 'LoKr 参数形式 说明'}));
    expect(screen.getByRole('tooltip')).toHaveTextContent('完整因子矩阵');
    expect(onChange).not.toHaveBeenCalled();
    rerender(<SchemaForm schema={schema} value={initial('lokr', 7)} onChange={onChange} compact readOnly groupFilter={['adapter']} errors={[{loc: 'adapter.rank', msg: '检查当前 Rank 配置'}]}/>);
    expect(numeric('rank')).toBeDisabled();
    expect(numeric('rank')).toHaveAttribute('aria-invalid', 'true');
    expect(numeric('alpha')).toBeDisabled();
  });
});
