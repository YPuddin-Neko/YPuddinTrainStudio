import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import schema from '../src/schema/train-schema.json';
import { schemaDefaults } from '../src/utils/config';
import type { FamilyInfo } from '../src/api/types';
import i18n from '../src/i18n';

const family = { name: 'anima', label: 'Anima 2B', text_modes: ['auto', 'cached', 'online'], presets: [] } as unknown as FamilyInfo;
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
function Editor({ initial, groups = ['dataset', 'caption'], advanced = false }: { initial: Record<string, any>; groups?: string[]; advanced?: boolean }) {
  const [value, setValue] = React.useState(initial);
  return <><SchemaForm schema={schema} value={value} onChange={setValue} family={family} groupFilter={groups} compact showAdvanced={advanced} /><output data-testid="caption-configuration">{JSON.stringify(value)}</output></>;
}
const current = () => JSON.parse(screen.getByTestId('caption-configuration').textContent!);
function select(label: string, option: string) {
  fireEvent.click(screen.getByRole('combobox', { name: label }));
  fireEvent.click(screen.getByRole('option', { name: option }));
}

describe('compact data and caption parameter layout', () => {
  it('starts new configurations at 1024 and edits resolutions solely through the input', () => {
    const initial = schemaDefaults(schema);
    expect(initial.dataset.resolutions).toEqual([1024]);
    render(<Editor initial={initial} />);
    const field = screen.getByTestId('field-dataset.resolutions');
    const input = within(field).getByRole('textbox');
    expect(input).toHaveValue('1024');
    for (const size of ['512', '768', '1024']) expect(within(field).queryByRole('button', { name: size })).not.toBeInTheDocument();
    fireEvent.change(input, { target: { value: '768, 1152' } });
    expect(current().dataset.resolutions).toEqual([768, 1152]);
  });

  it('keeps an existing 768 resolution when mounting or expanding advanced settings', () => {
    const initial = { dataset: { resolutions: [768], caption: { shuffle: false } } };
    const { rerender } = render(<Editor initial={initial} />);
    expect(within(screen.getByTestId('field-dataset.resolutions')).getByRole('textbox')).toHaveValue('768');
    rerender(<Editor initial={initial} advanced />);
    expect(within(screen.getByTestId('field-dataset.resolutions')).getByRole('textbox')).toHaveValue('768');
    expect(current()).toEqual(initial);
  });

  it('hides resource controls in basic mode while preserving legacy values and uses descriptive text modes', () => {
    const initial = { dataset: { resolutions: [768], cache_latents: false, text_encoding: 'online', caption: { shuffle: false } } };
    const { rerender } = render(<Editor initial={initial} />);
    expect(screen.queryByTestId('field-dataset.cache_latents')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-dataset.text_encoding')).not.toBeInTheDocument();
    expect(current()).toEqual(initial);
    rerender(<Editor initial={initial} advanced />);
    const cache = within(screen.getByTestId('field-dataset.cache_latents')).getByRole('checkbox');
    expect(cache).not.toBeChecked();
    const mode = screen.getByTestId('text-encoding-select');
    expect(mode).toHaveTextContent('每步处理标签');
    fireEvent.click(mode);
    expect(screen.queryByRole('option', { name: /在线|预缓存/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('option', { name: '训练前缓存标签' }));
    expect(current().dataset).toEqual({ ...initial.dataset, text_encoding: 'cached' });
  });

  it('edits each source format in the caption group without changing other sources or their metadata', () => {
    const sources = [
      { path: 'D:/project/v2/traindata/portraits', repeats: 3, caption_ext: '.txt', is_reg: false },
      { path: 'D:/project/v2/reg/class', repeats: 1, caption_ext: '.labels', is_reg: true, prior_weight: .5 },
    ];
    const initial = { dataset: { sources, caption: { shuffle: false } } };
    const { rerender } = render(<Editor initial={initial} groups={['dataset']} />);
    expect(screen.queryByRole('combobox', { name: '标签格式 · portraits' })).not.toBeInTheDocument();
    rerender(<Editor initial={initial} groups={['caption']} />);
    const first = screen.getByRole('combobox', { name: '标签格式 · portraits' });
    expect(first.closest('[data-group="caption"]')).not.toBeNull();
    expect(first).toHaveTextContent('仅 TXT');
    expect(screen.getByRole('combobox', { name: '标签格式 · class' })).toHaveTextContent('自定义扩展名');
    expect(screen.getByRole('textbox', { name: '自定义标签扩展名' })).toHaveValue('.labels');
    select('标签格式 · portraits', '自动 · JSON 优先，其次 TXT');
    expect(current().dataset.sources).toEqual([{ ...sources[0], caption_ext: 'auto' }, sources[1]]);
    select('标签格式 · portraits', '仅 JSON');
    expect(current().dataset.sources[0].caption_ext).toBe('.json');
    fireEvent.change(screen.getByRole('textbox', { name: '自定义标签扩展名' }), { target: { value: '.caption' } });
    expect(current().dataset.sources).toEqual([{ ...sources[0], caption_ext: '.json' }, { ...sources[1], caption_ext: '.caption' }]);
    select('标签格式 · class', '仅 TXT');
    expect(current().dataset.sources[1]).toEqual({ ...sources[1], caption_ext: '.txt' });
    expect(screen.queryByRole('textbox', { name: '自定义标签扩展名' })).not.toBeInTheDocument();
  });


  it('warns about nonempty legacy label rewrites and reveals their original values without changing them', () => {
    const initial = { dataset: { resolutions: [768], caption: { prefix: 'legacy prefix', suffix: 'legacy suffix', trigger_word: 'old_character', shuffle: false } } };
    render(<Editor initial={initial} groups={['caption']} />);
    expect(screen.getByText('当前配置会额外改写已有标签。')).toBeInTheDocument();
    for (const key of ['prefix', 'suffix', 'trigger_word']) expect(screen.queryByTestId(`field-dataset.caption.${key}`)).not.toBeInTheDocument();
    expect(current()).toEqual(initial);
    fireEvent.click(screen.getByRole('button', { name: '编辑额外标签改写' }));
    for (const [key, expected] of Object.entries(initial.dataset.caption).filter(([key]) => key !== 'shuffle')) {
      expect(within(screen.getByTestId(`field-dataset.caption.${key}`)).getByRole('textbox')).toHaveValue(String(expected));
    }
    expect(current()).toEqual(initial);
    fireEvent.change(within(screen.getByTestId('field-dataset.caption.prefix')).getByRole('textbox'), { target: { value: 'edited prefix' } });
    expect(current().dataset).toEqual({ ...initial.dataset, caption: { ...initial.dataset.caption, prefix: 'edited prefix' } });
  });

  it('does not advertise empty rewrite values in basic mode and exposes them under Advanced', () => {
    const initial = { dataset: { caption: { prefix: '', suffix: '', trigger_word: null, shuffle: false } } };
    const { rerender } = render(<Editor initial={initial} groups={['caption']} />);
    expect(screen.queryByText('当前配置会额外改写已有标签。')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '编辑额外标签改写' })).not.toBeInTheDocument();
    for (const key of ['prefix', 'suffix', 'trigger_word']) expect(screen.queryByTestId(`field-dataset.caption.${key}`)).not.toBeInTheDocument();
    rerender(<Editor initial={initial} groups={['caption']} advanced />);
    for (const key of ['prefix', 'suffix', 'trigger_word']) expect(within(screen.getByTestId(`field-dataset.caption.${key}`)).getByRole('textbox')).toHaveValue('');
    expect(current()).toEqual(initial);
  });

});
