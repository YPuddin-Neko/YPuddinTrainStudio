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
const bf16Policy = {...policy,id:'dtk-krea2-fsdp-bf16-linear-fp32-backward-v1',mixed_precision:'bf16',linear_forward:'native-bf16',linear_backward:'fp32-contractions-grad-original-dtype',linear_backward_implementation:'linear-bf16-forward-fp32-backward-v1',fsdp_param_dtype:'bfloat16',fsdp_reduce_dtype:'float32'};
const sdxlPolicy = {...policy,id:'dtk-sdxl-bf16-conv-fp32-linear-backward-v1',mixed_precision:'bf16',linear_forward:'native-bf16',linear_backward:'fp32-contractions-grad-original-dtype',linear_backward_implementation:'linear-bf16-forward-fp32-backward-v1',conv_forward:'fp32-output-bf16',conv_implementation:'conv2d-fp32-output-bf16-v1'};
const sdxlShardedPolicy = {...sdxlPolicy,id:'dtk-sdxl-fsdp-bf16-conv-fp32-linear-backward-v1',fsdp_param_dtype:'bfloat16',fsdp_reduce_dtype:'float32'};
const animaPolicy = {...policy,id:'dtk-anima-bf16-linear-fp32-compute-v1',mixed_precision:'bf16',linear_forward:'bf16-rounded-operands-fp32-contraction-bf16-output',linear_backward:'fp32-contractions-grad-original-dtype',linear_backward_implementation:'linear-bf16-operands-fp32-compute-v1'};
const fullConfig = () => {
  const value = schemaDefaults(schema);
  value.model.family = 'anima'; value.model.attention = 'xformers';
  value.training.mode = 'full'; value.training.train_backbone = true;
  value.loop.deterministic = true; value.loop.mixed_precision = 'bf16';
  value.memory.allow_tf32 = true;
  return value;
};
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

const animaLokrConfig = () => {
  const value = fullConfig();
  value.training.mode = 'adapter'; value.adapter.algo = 'lokr'; value.adapter.mode = 'auto';
  value.adapter.param_dtype = 'fp32'; value.adapter.dora = false; value.adapter.rules = [];
  value.memory.base_precision = 'auto';
  return value;
};

it.each(['full','adapter'])('confirms the Anima %s policy only for supported single-GPU drafts', mode => {
  const config = mode === 'full' ? fullConfig() : animaLokrConfig();
  for (const checkpointing of ['none','block']) {
    expect(confirmedTrainingComputePolicy(animaPolicy,{...config,memory:{...config.memory,activation_checkpointing:checkpointing}})).toEqual(animaPolicy);
  }
  // At one GPU, the selected distributed strategy does not change the backend scope.
  expect(confirmedTrainingComputePolicy(animaPolicy,{...config,loop:{...config.loop,distributed_strategy:'fsdp'}})).toEqual(animaPolicy);
  for (const changed of [
    {...config,model:{...config.model,family:'sdxl'}},
    {...config,training:{...config.training,train_backbone:false}},
    {...config,training:{...config.training,train_text_encoder:true}},
    {...config,loop:{...config.loop,deterministic:false}},
    {...config,loop:{...config.loop,mixed_precision:'no'}},
    {...config,loop:{...config.loop,mixed_precision:'fp16'}},
    {...config,loop:{...config.loop,gpu_count:2,distributed_strategy:'ddp'}},
    {...config,loop:{...config.loop,gpu_count:2,distributed_strategy:'fsdp'}},
    {...config,memory:{...config.memory,activation_checkpointing:'unsloth'}},
  ]) expect(confirmedTrainingComputePolicy(animaPolicy,changed)).toBeNull();
});

it('rejects unverified Anima adapter paths, including per-layer algorithm overrides', () => {
  const config = animaLokrConfig();
  for (const mode of ['auto','bypass']) {
    expect(confirmedTrainingComputePolicy(animaPolicy,{...config,adapter:{...config.adapter,mode,rules:[{match:'a',algo:null},{match:'b',algo:'lokr'},{match:'c',algo:'none'}]}})).toEqual(animaPolicy);
  }
  for (const adapter of [
    {...config.adapter,algo:'lora'}, {...config.adapter,mode:'merged'},
    {...config.adapter,dora:true}, {...config.adapter,param_dtype:'bf16'},
    {...config.adapter,rules:[{match:'*',algo:'loha'}]},
    {...config.adapter,rules:[null]}, {...config.adapter,rules:{}},
  ]) expect(confirmedTrainingComputePolicy(animaPolicy,{...config,adapter})).toBeNull();
  for (const base_precision of ['fp8_e4m3','fp8_e5m2']) {
    expect(confirmedTrainingComputePolicy(animaPolicy,{...config,memory:{...config.memory,base_precision}})).toBeNull();
  }
  // Adapter options are inactive during full-model training.
  expect(confirmedTrainingComputePolicy(animaPolicy,{...fullConfig(),adapter:{...config.adapter,algo:'loha',dora:true}})).toEqual(animaPolicy);
});

it('rejects incomplete, forged or stale Anima compute identities', () => {
  const config = fullConfig();
  for (const key of Object.keys(animaPolicy)) {
    const incomplete: Record<string, unknown> = {...animaPolicy}; delete incomplete[key];
    expect(confirmedTrainingComputePolicy(incomplete,config)).toBeNull();
  }
  for (const changed of [
    {...animaPolicy,id:'diagnostic-anima-linear-fp32-native-lokr-candidate-v1'},
    {...animaPolicy,linear_forward:'native-bf16'},
    {...animaPolicy,linear_backward:'native-bf16'},
    {...animaPolicy,linear_backward_implementation:'linear-bf16-forward-fp32-backward-v1'},
    {...animaPolicy,allow_tf32:true}, {...animaPolicy,sdpa_backend:'flash'},
    {...animaPolicy,fsdp_param_dtype:'bfloat16'}, {...animaPolicy,fsdp_reduce_dtype:'float32'},
    {...animaPolicy,conv_forward:'fp32-output-bf16'}, {...animaPolicy,conv_implementation:'conv2d-fp32-output-bf16-v1'},
  ]) expect(confirmedTrainingComputePolicy(changed,config)).toBeNull();
  const checked = JSON.stringify(config);
  expect(currentTrainingComputePolicy(animaPolicy,config,checked,false)).toEqual(animaPolicy);
  expect(currentTrainingComputePolicy(animaPolicy,config,checked,true)).toBeNull();
  expect(currentTrainingComputePolicy(animaPolicy,{...config,loop:{...config.loop,gpu_count:2}},checked,false)).toBeNull();
});

it('describes Anima rounded BF16 operands and FP32 linear computation without changing the draft', () => {
  const initial = animaLokrConfig();
  function Editor() {
    const [value,setValue] = React.useState(initial);
    return <><SchemaForm schema={schema} value={value} onChange={setValue} computePolicy={animaPolicy} compact showAdvanced groupFilter={['loop','memory']}/><output data-testid="draft">{JSON.stringify(value)}</output></>;
  }
  render(<Editor/>);
  expect(screen.getByRole('status',{name:'混合精度'})).toHaveTextContent('BF16（线性层 FP32 运算）');
  const hint = screen.getByTestId('field-loop.deterministic');
  expect(hint).toHaveTextContent('本次 Anima 使用单卡');
  expect(hint).toHaveTextContent('先按 BF16 舍入输入与参数');
  expect(hint).toHaveTextContent('以 FP32 进行矩阵运算并返回 BF16');
  expect(hint).not.toHaveTextContent('保留 BF16 前向');
  expect(hint).not.toHaveTextContent('多卡显存分片');
  expect(JSON.parse(screen.getByTestId('draft').textContent!)).toEqual(initial);
  fireEvent.click(screen.getByRole('checkbox',{name:'可复现训练'}));
  expect(screen.getByRole('combobox',{name:'混合精度'})).toHaveTextContent('BF16');
  expect(screen.getByRole('checkbox',{name:'允许 TF32'})).toBeChecked();
  expect(screen.getByRole('combobox',{name:'注意力后端'})).toHaveTextContent('xFormers');
});

it('presents Anima computation and checkpointing limits in English', async () => {
  await i18n.changeLanguage('en');
  render(<SchemaForm schema={schema} value={fullConfig()} onChange={() => {}} computePolicy={animaPolicy} compact showAdvanced groupFilter={['loop','memory']}/>);
  expect(screen.getByTestId('field-loop.mixed_precision')).toHaveTextContent('BF16 (FP32 linear operations)');
  const hint = screen.getByTestId('field-loop.deterministic');
  expect(hint).toHaveTextContent('uses one GPU');
  expect(hint).toHaveTextContent('rounded to BF16');
  expect(hint).toHaveTextContent('matrix operations use FP32');
  expect(hint.textContent).not.toMatch(/[\u4e00-\u9fff]/);
  expect(hint.textContent).not.toContain('native BF16');
  fireEvent.click(within(hint).getByRole('button',{name:'Reproducible training help'}));
  expect(screen.getByRole('tooltip')).toHaveTextContent('Activation checkpointing must be off or use standard blocks');
});

const shardedKreaConfig = () => {
  const value = fullConfig();
  value.model.family = 'krea2';
  value.training.train_text_encoder = false;
  value.loop.distributed_strategy = 'fsdp'; value.loop.gpu_count = 2;
  return value;
};

it('only confirms complete BF16 policy metadata for the matching multi-GPU Krea draft', () => {
  const config = shardedKreaConfig();
  expect(confirmedTrainingComputePolicy(bf16Policy,config)).toEqual(bf16Policy);
  for (const changed of [
    {...config,model:{...config.model,family:'anima'}},
    {...config,training:{...config.training,train_text_encoder:true}},
    {...config,loop:{...config.loop,distributed_strategy:'ddp'}},
    {...config,loop:{...config.loop,gpu_count:1}},
    {...config,loop:{...config.loop,mixed_precision:'no'}},
  ]) expect(confirmedTrainingComputePolicy(bf16Policy,changed)).toBeNull();
  for (const changed of [
    {...bf16Policy,linear_backward:undefined},
    {...bf16Policy,linear_backward_implementation:undefined},
    {...bf16Policy,linear_forward:'fp32'},
    {...bf16Policy,fsdp_reduce_dtype:'bfloat16'},
    {...bf16Policy,id:'diagnostic-dtk-krea2-full-bf16-math-linear-backward-v2'},
  ]) expect(confirmedTrainingComputePolicy(changed,config)).toBeNull();
});

it('renders actual BF16 forward and FP32 backward without changing the selected draft', () => {
  const initial = shardedKreaConfig();
  function Editor() {
    const [value,setValue] = React.useState(initial);
    return <><SchemaForm schema={schema} value={value} onChange={setValue} computePolicy={bf16Policy} compact showAdvanced groupFilter={['loop','memory']}/><output data-testid="draft">{JSON.stringify(value)}</output></>;
  }
  render(<Editor/>);
  expect(screen.getByRole('status',{name:'混合精度'})).toHaveTextContent('BF16 前向（线性层反向 FP32）');
  expect(screen.getByTestId('field-loop.deterministic')).toHaveTextContent('本次 Krea2 使用多卡显存分片');
  expect(screen.queryByText('FP32 计算（关闭混合精度）')).not.toBeInTheDocument();
  expect(JSON.parse(screen.getByTestId('draft').textContent!)).toEqual(initial);
  fireEvent.click(screen.getByRole('checkbox',{name:'可复现训练'}));
  expect(screen.getByRole('combobox',{name:'混合精度'})).toHaveTextContent('BF16');
  expect(screen.getByRole('checkbox',{name:'允许 TF32'})).toBeChecked();
});

it('shows the BF16 policy in English with no Chinese fallback', async () => {
  await i18n.changeLanguage('en');
  render(<SchemaForm schema={schema} value={shardedKreaConfig()} onChange={() => {}} computePolicy={bf16Policy} compact showAdvanced groupFilter={['loop','memory']}/>);
  expect(screen.getByTestId('field-loop.mixed_precision')).toHaveTextContent('BF16 forward (FP32 linear backward)');
  const hint = screen.getByTestId('field-loop.deterministic');
  expect(hint).toHaveTextContent('shards the model across GPUs');
  expect(hint.textContent).not.toMatch(/[\u4e00-\u9fff]/);
});

it('shows the SDXL convolution policy only for the checked single-GPU backbone configuration', () => {
  const config = fullConfig(); config.model.family = 'sdxl'; config.loop.gpu_count = 1;
  expect(confirmedTrainingComputePolicy(sdxlPolicy,config)).toEqual(sdxlPolicy);
  expect(confirmedTrainingComputePolicy({...sdxlPolicy,conv_implementation:undefined},config)).toBeNull();
  expect(confirmedTrainingComputePolicy(sdxlPolicy,{...config,loop:{...config.loop,gpu_count:2}})).toBeNull();
  expect(confirmedTrainingComputePolicy(sdxlPolicy,{...config,model:{...config.model,family:'anima'}})).toBeNull();
  render(<SchemaForm schema={schema} value={config} onChange={() => {}} computePolicy={sdxlPolicy} compact showAdvanced groupFilter={['loop','memory']}/>);
  expect(screen.getByRole('status',{name:'混合精度'})).toHaveTextContent('BF16（FP32 卷积、线性层反向）');
  expect(screen.getByTestId('field-loop.deterministic')).toHaveTextContent('卷积输出转回 BF16');
  expect(screen.getByTestId('field-loop.deterministic')).not.toHaveTextContent('本次 Krea2');
});

it('shows the same verified SDXL policy for supported LoKr, without accepting unverified adapter paths', () => {
  const config = fullConfig(); config.model.family = 'sdxl'; config.loop.gpu_count = 1;
  config.training.mode = 'adapter'; config.adapter.algo = 'lokr'; config.adapter.mode = 'auto';
  config.adapter.param_dtype = 'fp32'; config.adapter.dora = false; config.adapter.rules = [];
  config.memory.base_precision = 'auto';
  expect(confirmedTrainingComputePolicy(sdxlPolicy,config)).toEqual(sdxlPolicy);
  expect(confirmedTrainingComputePolicy(policy,config)).toBeNull();
  for (const adapter of [
    {...config.adapter,algo:'lora'}, {...config.adapter,mode:'merged'},
    {...config.adapter,dora:true}, {...config.adapter,param_dtype:'bf16'},
    {...config.adapter,rules:[{match:'*',algo:'loha'}]},
  ]) expect(confirmedTrainingComputePolicy(sdxlPolicy,{...config,adapter})).toBeNull();
  expect(confirmedTrainingComputePolicy(sdxlPolicy,{...config,memory:{...config.memory,base_precision:'fp8_e4m3'}})).toBeNull();
  render(<SchemaForm schema={schema} value={config} onChange={() => {}} computePolicy={sdxlPolicy} compact showAdvanced groupFilter={['loop','memory']}/>);
  expect(screen.getByRole('status',{name:'混合精度'})).toHaveTextContent('BF16（FP32 卷积、线性层反向）');
  expect(screen.getByTestId('field-loop.deterministic')).toHaveTextContent('本次 SDXL 保留 BF16');
});

it('only uses a confirmed policy for the current server-checked draft', () => {
  const config = fullConfig(); const encoded = JSON.stringify(config);
  expect(currentTrainingComputePolicy(policy,config,encoded,false)).toEqual(policy);
  expect(currentTrainingComputePolicy(policy,config,encoded,true)).toBeNull();
  const changed = structuredClone(config); changed.loop.seed += 1;
  expect(currentTrainingComputePolicy(policy,changed,encoded,false)).toBeNull();
  expect(currentTrainingComputePolicy(null,config,encoded,false)).toBeNull();
});

it('keeps the verified BF16 LoKr settings for DDP but rejects adapter sharding and full DDP', () => {
  const config = fullConfig(); config.model.family = 'sdxl'; config.training.mode = 'adapter';
  config.adapter.algo = 'lokr'; config.adapter.param_dtype = 'fp32'; config.adapter.dora = false;
  config.adapter.mode = 'auto'; config.adapter.rules = []; config.memory.base_precision = 'auto';
  config.loop.gpu_count = 2; config.loop.distributed_strategy = 'ddp';
  expect(confirmedTrainingComputePolicy(sdxlPolicy,config)).toEqual(sdxlPolicy);
  expect(confirmedTrainingComputePolicy(sdxlPolicy,{...config,loop:{...config.loop,distributed_strategy:'fsdp'}})).toBeNull();
  expect(confirmedTrainingComputePolicy(sdxlPolicy,{...config,training:{...config.training,mode:'full'}})).toBeNull();
  render(<SchemaForm schema={schema} value={config} onChange={() => {}} computePolicy={sdxlPolicy} compact showAdvanced groupFilter={['loop','memory']}/>);
  expect(screen.getByRole('status',{name:'混合精度'})).toHaveTextContent('BF16（FP32 卷积、线性层反向）');
});

it('shows SDXL sharding only for the matching full-model policy and BF16 gather metadata', () => {
  const config = shardedKreaConfig(); config.model.family = 'sdxl';
  expect(confirmedTrainingComputePolicy(sdxlShardedPolicy,config)).toEqual(sdxlShardedPolicy);
  expect(confirmedTrainingComputePolicy(sdxlPolicy,config)).toBeNull();
  expect(confirmedTrainingComputePolicy(sdxlShardedPolicy,{...config,loop:{...config.loop,distributed_strategy:'ddp'}})).toBeNull();
  expect(confirmedTrainingComputePolicy(sdxlShardedPolicy,{...config,training:{...config.training,mode:'adapter'}})).toBeNull();
  for (const changed of [
    {...sdxlShardedPolicy,fsdp_param_dtype:'float32'}, {...sdxlShardedPolicy,fsdp_reduce_dtype:'bfloat16'},
    {...sdxlShardedPolicy,conv_implementation:undefined}, {...sdxlShardedPolicy,linear_backward_implementation:undefined},
  ]) expect(confirmedTrainingComputePolicy(changed,config)).toBeNull();
  render(<SchemaForm schema={schema} value={config} onChange={() => {}} computePolicy={sdxlShardedPolicy} compact showAdvanced groupFilter={['loop','memory']}/>);
  expect(screen.getByRole('status',{name:'混合精度'})).toHaveTextContent('BF16（FP32 卷积、线性层反向）');
  expect(screen.getByTestId('field-loop.deterministic')).toHaveTextContent('本次 SDXL 使用多卡显存分片');
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
