import React from 'react';
import {fireEvent, render, screen, within} from '@testing-library/react';
import {beforeEach, expect, it} from 'vitest';
import {SchemaForm} from '../src/schema/SchemaForm/SchemaForm';
import schema from '../src/schema/train-schema.json';
import {schemaDefaults} from '../src/utils/config';
import {selectTrainingComponents} from '../src/utils/trainingSelection';
import i18n from '../src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
function Editor({initial}: {initial?: Record<string, any>}) {
  const [value,setValue] = React.useState(initial || schemaDefaults(schema));
  return <><SchemaForm schema={schema} value={value} onChange={setValue} compact showAdvanced groupFilter={['training','adapter','memory','dataset']}/><output data-testid="training-config">{JSON.stringify(value)}</output></>;
}
const config = () => JSON.parse(screen.getByTestId('training-config').textContent || '{}');

it('changes actual training components and dependent encoding controls without exposing adapter controls in full mode', () => {
  render(<Editor/>);
  fireEvent.click(screen.getByRole('combobox',{name:'训练方式'}));
  fireEvent.click(screen.getByRole('option',{name:'全量微调'}));
  expect(config().training).toMatchObject({mode:'full', train_backbone:true, train_text_encoder:false});
  expect(screen.queryByTestId('field-adapter.algo')).not.toBeInTheDocument();
  expect(screen.getByRole('status',{name:'底模存储精度'})).toHaveTextContent('FP32 · 32 位');
  fireEvent.click(screen.getByRole('checkbox',{name:'训练文本编码器'}));
  expect(config().training.train_text_encoder).toBe(true);
  expect(config().dataset.text_encoding).toBe('online');
  expect(config().memory.offload_text_encoder).toBe(false);
  expect(within(screen.getByTestId('field-dataset.text_encoding')).queryByRole('combobox')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('checkbox',{name:'训练主模型（UNet / DiT）'}));
  expect(config().training).toMatchObject({train_backbone:false, train_text_encoder:true});
  fireEvent.click(screen.getByRole('combobox',{name:'训练方式'}));
  fireEvent.click(screen.getByRole('option',{name:'LoRA / LoKr 适配器'}));
  expect(config().training).toMatchObject({mode:'adapter',train_backbone:true,train_text_encoder:false});
  expect(screen.getByTestId('field-adapter.algo')).toBeInTheDocument();
});

it('allows correcting incompatible full-mode precision from an imported draft instead of locking an invalid value', () => {
  const initial = schemaDefaults(schema);
  initial.training.mode='full'; initial.memory.base_precision='bf16';
  render(<Editor initial={initial}/>);
  fireEvent.click(screen.getByRole('combobox',{name:'底模存储精度'}));
  fireEvent.click(screen.getByRole('option',{name:'FP32 · 32 位'}));
  expect(config().memory.base_precision).toBe('fp32');
  expect(screen.getByRole('status',{name:'底模存储精度'})).toHaveTextContent('FP32 · 32 位');
});

it('keeps unrelated settings intact and does not mutate the previous draft', () => {
  const original = schemaDefaults(schema);
  original.dataset.sources=[{path:'/my/training/images',repeats:3}];
  original.optimizer.lr=0.0003;
  const next=structuredClone(original); next.training.mode='full'; next.training.train_text_encoder=true;
  const changed=selectTrainingComponents(next,original);
  expect(changed.dataset.sources).toEqual(original.dataset.sources);
  expect(changed.optimizer).toEqual(original.optimizer);
  expect(original.training.mode).toBe('adapter');
  expect(changed.memory.base_precision).toBe('fp32');
});

it('offers GPU count in common settings and persists the numeric choice', () => {
  function Devices() {
    const [value, setValue] = React.useState(schemaDefaults(schema));
    return <><SchemaForm schema={schema} value={value} onChange={setValue} compact showAdvanced={false} groupFilter={['loop']}/><output data-testid="training-config">{JSON.stringify(value)}</output></>;
  }
  render(<Devices/>);
  const field = screen.getByLabelText('训练显卡数量');
  expect(field).toHaveValue(1);
  fireEvent.change(field, {target: {value: '2'}});
  fireEvent.blur(field);
  expect(config().loop.gpu_count).toBe(2);
  expect(screen.getByText('1 为单卡；多卡分担图片计算，显存不会合并。')).toBeInTheDocument();
  expect(screen.getByTestId('field-dataset.batch_size')).toHaveTextContent('每张显卡一次处理的图片数');
});
