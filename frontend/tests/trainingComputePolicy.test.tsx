import React from 'react';
import {fireEvent, render, screen, within} from '@testing-library/react';
import {beforeEach, expect, it} from 'vitest';
import {SchemaForm} from '../src/schema/SchemaForm/SchemaForm';
import schema from '../src/schema/train-schema.json';
import {schemaDefaults} from '../src/utils/config';
import {configFieldHelp, configOptionLabel} from '../src/utils/configPresentation';
import {confirmedTrainingComputePolicy, currentTrainingComputePolicy} from '../src/utils/trainingComputePolicy';
import i18n from '../src/i18n';

const policy = {id:'dtk-full-fp32-math-v1',mixed_precision:'no',allow_tf32:false,attention:'sdpa',sdpa_backend:'math'};
const fullConfig = () => {
  const value = schemaDefaults(schema);
  value.model.family = 'anima'; value.model.attention = 'xformers';
  value.training.mode = 'full'; value.training.train_backbone = true;
  value.loop.deterministic = true; value.loop.mixed_precision = 'bf16';
  value.memory.allow_tf32 = true;
  return value;
};
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

it('only uses a confirmed policy for the current server-checked draft', () => {
  const config = fullConfig(); const encoded = JSON.stringify(config);
  expect(currentTrainingComputePolicy(policy,config,encoded,false)).toEqual(policy);
  expect(currentTrainingComputePolicy(policy,config,encoded,true)).toBeNull();
  const changed = structuredClone(config); changed.loop.seed += 1;
  expect(currentTrainingComputePolicy(policy,changed,encoded,false)).toBeNull();
  expect(currentTrainingComputePolicy(null,config,encoded,false)).toBeNull();
});

it.each(['anima','sdxl','krea2'])('accepts the server-confirmed full-backbone policy for %s', family => {
  const config = fullConfig(); config.model.family = family;
  expect(confirmedTrainingComputePolicy(policy,config)).toEqual(policy);
});

it('does not infer an effective FP32 policy for another family, adapter, text-only or disabled training', () => {
  const config = fullConfig();
  for (const changed of [
    {...config,model:{...config.model,family:'flux2'}},
    {...config,training:{...config.training,mode:'adapter'}},
    {...config,training:{...config.training,train_backbone:false,train_text_encoder:true}},
    {...config,loop:{...config.loop,deterministic:false}},
  ]) expect(confirmedTrainingComputePolicy(policy,changed)).toBeNull();
  expect(confirmedTrainingComputePolicy({...policy,id:'future-policy'},config)).toBeNull();
  expect(confirmedTrainingComputePolicy({...policy,allow_tf32:true},config)).toBeNull();
});

it('displays actual managed values and restores original choices without changing the draft', () => {
  const initial = fullConfig();
  function Editor() {
    const [value,setValue] = React.useState(initial);
    return <><SchemaForm schema={schema} value={value} onChange={setValue} computePolicy={policy} compact showAdvanced groupFilter={['loop','memory']}/><output data-testid="draft">{JSON.stringify(value)}</output></>;
  }
  render(<Editor/>);
  expect(screen.getByRole('status',{name:'混合精度'})).toHaveTextContent('FP32 计算（关闭混合精度）');
  expect(screen.getByRole('status',{name:'允许 TF32'})).toHaveTextContent('关闭');
  expect(screen.getByRole('status',{name:'注意力后端'})).toHaveTextContent('PyTorch SDPA（数学实现）');
  expect(screen.getByTestId('field-loop.deterministic')).toHaveTextContent('本次 DTK 主模型全量微调');
  expect(screen.getByTestId('field-loop.deterministic')).toHaveTextContent('激活显存和训练耗时');
  expect(JSON.parse(screen.getByTestId('draft').textContent!)).toEqual(initial);
  fireEvent.click(screen.getByRole('checkbox',{name:'可复现训练'}));
  expect(screen.getByRole('combobox',{name:'混合精度'})).toHaveTextContent('BF16');
  expect(screen.getByRole('checkbox',{name:'允许 TF32'})).toBeChecked();
  expect(screen.getByRole('combobox',{name:'注意力后端'})).toHaveTextContent('xFormers');
  const saved = JSON.parse(screen.getByTestId('draft').textContent!);
  expect(saved.loop.mixed_precision).toBe('bf16');
  expect(saved.memory.allow_tf32).toBe(true);
  expect(saved.model.attention).toBe('xformers');
});

it('leaves original controls editable without a current runtime policy and does not label no as FP32', () => {
  render(<SchemaForm schema={schema} value={fullConfig()} onChange={() => {}} compact showAdvanced groupFilter={['loop','memory']}/>);
  expect(screen.getByRole('combobox',{name:'混合精度'})).toHaveTextContent('BF16');
  expect(screen.getByRole('checkbox',{name:'允许 TF32'})).toBeChecked();
  expect(screen.queryByText('FP32 计算（关闭混合精度）')).not.toBeInTheDocument();
  expect(configOptionLabel('loop.mixed_precision','no')).toBe('关闭自动混合精度');
  expect(configFieldHelp('loop.mixed_precision','')).toContain('不等同于所有训练都使用 FP32');
});

it('presents the same effective policy in English without leaking Chinese fallback text', async () => {
  await i18n.changeLanguage('en');
  render(<SchemaForm schema={schema} value={fullConfig()} onChange={() => {}} computePolicy={policy} compact showAdvanced groupFilter={['loop','memory']}/>);
  const mixed = screen.getByTestId('field-loop.mixed_precision');
  expect(within(mixed).getByRole('status')).toHaveTextContent('FP32 computation (mixed precision off)');
  expect(mixed).toHaveTextContent('Your original choice is retained');
  const deterministic = screen.getByTestId('field-loop.deterministic');
  expect(deterministic).toHaveTextContent('Activation memory and training time may increase');
  expect(deterministic.textContent).not.toMatch(/[\u4e00-\u9fff]/);
  fireEvent.click(within(deterministic).getByRole('button',{name:'Reproducible training help'}));
  expect(screen.getByRole('tooltip')).toHaveTextContent('Bitwise equality is not guaranteed');
});
