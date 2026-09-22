import React from 'react';
import {fireEvent, render, screen, within} from '@testing-library/react';
import {beforeEach, expect, it} from 'vitest';
import {SchemaForm} from '../../../frontend/src/schema/SchemaForm/SchemaForm';
import schema from '../../../frontend/src/schema/train-schema.json';
import {schemaDefaults} from '../../../frontend/src/utils/config';
import {configFieldHelp, configOptionLabel} from '../../../frontend/src/utils/configPresentation';
import {confirmedTrainingComputePolicy, currentTrainingComputePolicy, trainingComputePolicyHint} from '../../../frontend/src/utils/trainingComputePolicy';
import i18n from '../../../frontend/src/i18n';

const policy = {id:'dtk-full-fp32-math-v1',mixed_precision:'no',allow_tf32:false,attention:'sdpa',sdpa_backend:'math'};
const bf16Policy = {...policy,id:'dtk-krea2-fsdp-bf16-linear-fp32-backward-v1',mixed_precision:'bf16',linear_forward:'native-bf16',linear_backward:'fp32-contractions-grad-original-dtype',linear_backward_implementation:'linear-bf16-forward-fp32-backward-v1',fsdp_param_dtype:'bfloat16',fsdp_reduce_dtype:'float32'};
const sdxlPolicy = {...policy,id:'dtk-sdxl-bf16-conv-fp32-linear-backward-v1',mixed_precision:'bf16',linear_forward:'native-bf16',linear_backward:'fp32-contractions-grad-original-dtype',linear_backward_implementation:'linear-bf16-forward-fp32-backward-v1',conv_forward:'fp32-output-bf16',conv_implementation:'conv2d-fp32-output-bf16-v1'};
const sdxlShardedPolicy = {...sdxlPolicy,id:'dtk-sdxl-fsdp-bf16-conv-fp32-linear-backward-v1',fsdp_param_dtype:'bfloat16',fsdp_reduce_dtype:'float32'};
const animaPolicy = {...policy,id:'dtk-anima-bf16-linear-fp32-compute-v1',mixed_precision:'bf16',linear_forward:'bf16-rounded-operands-fp32-contraction-bf16-output',linear_backward:'fp32-contractions-grad-original-dtype',linear_backward_implementation:'linear-bf16-operands-fp32-compute-v1'};
const animaDdpPolicy = {...animaPolicy,id:'dtk-anima-ddp-bf16-linear-fp32-compute-v1'};
const animaFsdpPolicy = {...animaPolicy,id:'dtk-anima-fsdp-bf16-linear-fp32-compute-v1',fsdp_param_dtype:'bfloat16',fsdp_reduce_dtype:'float32'};
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

const animaDualConfig = (strategy: 'ddp' | 'fsdp') => {
  const value = strategy === 'ddp' ? animaLokrConfig() : fullConfig();
  value.loop.gpu_count = 2; value.loop.distributed_strategy = strategy;
  return value;
};

function textLoraCase(family: string, backbone = false, gpus = 1) {
  const config = animaLokrConfig();
  config.model.family = family; config.model.sdxl_max_token_length = 150;
  config.adapter.algo = 'lora';
  config.training.train_backbone = backbone; config.training.train_text_encoder = true;
  config.dataset.text_encoding = 'online'; config.memory.offload_text_encoder = false;
  config.loop.gpu_count = gpus; config.loop.distributed_strategy = 'ddp';
  const components = family === 'sdxl' ? ['text_encoder', 'text_encoder_2'] : ['text_encoder'];
  const candidate = {
    ...animaPolicy, id: `dtk-${family}-text-lora-bf16-fp32-contractions-v1`,
    text_linear_forward: 'bf16-rounded-operands-fp32-contraction-bf16-output',
    text_linear_backward_implementation: 'linear-bf16-operands-fp32-compute-v1',
    adapter_implementation: 'lora-bf16-operands-fp32-contractions-v1',
    adapter_forward: 'bf16-rounded-operands-fp32-contractions-bf16-intermediates',
    adapter_backward: 'fp32-contractions-grad-original-dtype',
    trainable_components: backbone ? ['backbone', ...components] : components,
    operator_components: ['backbone', ...components],
    distributed_strategy: gpus === 1 ? 'single' : 'ddp',
    ...(family === 'sdxl' ? {conv_forward: 'fp32-output-bf16', conv_implementation: 'conv2d-fp32-output-bf16-v1', sdxl_max_token_length: 150} : {}),
  };
  return {config, candidate};
}

function backboneAdapterCase(family: string, algo: 'lora' | 'lokr', strategy: 'single' | 'ddp' | 'fsdp', tokens = 75) {
  const config = animaLokrConfig();
  config.model.family = family; config.model.sdxl_max_token_length = tokens;
  config.adapter.algo = algo; config.training.train_text_encoder = false;
  config.loop.gpu_count = strategy === 'single' ? 1 : 2;
  config.loop.distributed_strategy = strategy === 'fsdp' ? 'fsdp' : 'ddp';
  const linear = family === 'sdxl' && algo === 'lokr' && tokens === 75 ? sdxlPolicy : animaPolicy;
  const stablePreview = family === 'sdxl' && tokens > 75 && strategy !== 'single';
  const candidate = {
    ...linear, id: `dtk-${family}-backbone-${algo}-${strategy}-bf16-compute-${stablePreview ? 'preview-v2' : 'v1'}`,
    adapter_algorithm: algo, trainable_components: ['backbone'], operator_components: ['backbone'], distributed_strategy: strategy,
    ...(algo === 'lora' ? {
      adapter_implementation: 'lora-bf16-operands-fp32-contractions-v1',
      adapter_forward: 'bf16-rounded-operands-fp32-contractions-bf16-intermediates',
      adapter_backward: 'fp32-contractions-grad-original-dtype',
    } : {}),
    ...(strategy === 'fsdp' ? {fsdp_param_dtype: 'bfloat16', fsdp_reduce_dtype: 'float32'} : {}),
    ...(family === 'sdxl' ? {conv_forward: 'fp32-output-bf16', conv_implementation: 'conv2d-fp32-output-bf16-v1', sdxl_max_token_length: tokens} : {}),
    ...(stablePreview ? {
      preview_operator_components: ['backbone'],
      preview_linear_forward: 'bf16-rounded-operands-fp32-contraction-bf16-output',
      preview_linear_implementation: 'linear-native-dispatch-bf16-operands-fp32-preview-v1',
    } : {}),
  };
  return {config, candidate};
}

const backboneCases = [...['anima', 'sdxl'].flatMap(family =>
  (['lora', 'lokr'] as const).flatMap(algo =>
    (algo === 'lora' ? ['single', 'ddp', 'fsdp'] as const : ['fsdp'] as const).flatMap(strategy =>
      (family === 'sdxl' ? [75, 150, 225] : [75]).map(tokens => ({family, algo, strategy, tokens})))),
), ...[150, 225].map(tokens => ({family:'sdxl', algo:'lokr' as const, strategy:'ddp' as const, tokens}))];

it.each(backboneCases)('confirms only the complete $family $algo $strategy $tokens-token backbone contract', ({family, algo, strategy, tokens}) => {
  const {config, candidate} = backboneAdapterCase(family, algo, strategy, tokens);
  expect(confirmedTrainingComputePolicy(candidate, config)).toEqual(candidate);
  // Frozen encoders can be cached and offloaded; do not apply text-training restrictions.
  expect(confirmedTrainingComputePolicy(candidate, {...config, dataset:{...config.dataset, text_encoding:'cached'}, memory:{...config.memory, offload_text_encoder:true}})).toEqual(candidate);
  for (const mode of ['auto', 'bypass']) for (const activation_checkpointing of ['none', 'block']) {
    expect(confirmedTrainingComputePolicy(candidate, {...config,
      adapter:{...config.adapter, mode, rules:[{match:'a', algo:null}, {match:'b', algo}, {match:'c', algo:'none'}]},
      memory:{...config.memory, activation_checkpointing},
    })).toEqual(candidate);
  }
  if (strategy === 'single') {
    expect(confirmedTrainingComputePolicy(candidate, {...config, loop:{...config.loop, distributed_strategy:'fsdp'}})).toEqual(candidate);
  }
  for (const key of Object.keys(candidate)) {
    const missing: Record<string, unknown> = {...candidate}; delete missing[key];
    expect(confirmedTrainingComputePolicy(missing, config), key).toBeNull();
  }
  for (const invalid of [
    {...candidate, id:candidate.id.replace(/-v[12]$/, '-v99')},
    {...candidate, adapter_algorithm:algo === 'lora' ? 'lokr' : 'lora'},
    {...candidate, trainable_components:['backbone', 'text_encoder']},
    {...candidate, operator_components:['text_encoder']},
    {...candidate, distributed_strategy:strategy === 'fsdp' ? 'ddp' : 'fsdp'},
    {...candidate, linear_forward:candidate.linear_forward === 'native-bf16' ? 'bf16-rounded-operands-fp32-contraction-bf16-output' : 'native-bf16'},
    {...candidate, linear_backward_implementation:'unknown'},
    {...candidate, text_linear_forward:'bf16-rounded-operands-fp32-contraction-bf16-output'},
    {...candidate, fsdp_param_dtype:'float32'}, {...candidate, fsdp_reduce_dtype:'bfloat16'},
    ...(strategy !== 'fsdp' ? [{...candidate, fsdp_param_dtype:'bfloat16', fsdp_reduce_dtype:'float32'}] : []),
    ...(algo === 'lokr' ? [{...candidate, adapter_implementation:'lora-bf16-operands-fp32-contractions-v1'}] : [{...candidate, adapter_forward:'native-bf16'}]),
    {...candidate, sdxl_max_token_length:family === 'sdxl' ? (tokens === 75 ? 150 : 75) : 75},
    ...(family === 'anima' ? [{...candidate, conv_forward:'fp32-output-bf16'}] : [{...candidate, conv_implementation:'unknown'}]),
  ]) expect(confirmedTrainingComputePolicy(invalid, config)).toBeNull();
  for (const changed of [
    {...config, model:{...config.model, family:family === 'anima' ? 'sdxl' : 'anima'}},
    {...config, training:{...config.training, mode:'full'}},
    {...config, training:{...config.training, train_backbone:false}},
    {...config, training:{...config.training, train_text_encoder:true}},
    {...config, loop:{...config.loop, deterministic:false}},
    {...config, loop:{...config.loop, mixed_precision:'fp16'}},
    ...[0, 1.5, '2'].map(gpu_count => ({...config, loop:{...config.loop, gpu_count}})),
    {...config, adapter:{...config.adapter, algo:algo === 'lora' ? 'lokr' : 'lora'}},
    {...config, adapter:{...config.adapter, mode:'merged'}},
    {...config, adapter:{...config.adapter, dora:true}},
    {...config, adapter:{...config.adapter, param_dtype:'bf16'}},
    {...config, adapter:{...config.adapter, rules:[{match:'*', algo:'loha'}]}},
    {...config, adapter:{...config.adapter, rules:[null]}},
    {...config, memory:{...config.memory, base_precision:'fp8_e4m3'}},
    {...config, memory:{...config.memory, compile:true}},
    {...config, memory:{...config.memory, blocks_to_swap:1}},
    {...config, memory:{...config.memory, activation_checkpointing:'unsloth'}},
  ]) expect(confirmedTrainingComputePolicy(candidate, changed)).toBeNull();
  const checked = JSON.stringify(config);
  expect(currentTrainingComputePolicy(candidate, config, checked, false)).toEqual(candidate);
  expect(currentTrainingComputePolicy(candidate, config, checked, true)).toBeNull();
  expect(currentTrainingComputePolicy(candidate, {...config, loop:{...config.loop, seed:config.loop.seed+1}}, checked, false)).toBeNull();
  expect(confirmedTrainingComputePolicy(family === 'anima' ? animaPolicy : sdxlPolicy, config)).toBeNull();
});

it.each(['anima', 'sdxl'])('does not expose new %s LoKr policies for single-GPU or DDP runs', family => {
  for (const strategy of ['single', 'ddp'] as const) {
    const {config, candidate} = backboneAdapterCase(family, 'lokr', strategy);
    expect(confirmedTrainingComputePolicy(candidate, config)).toBeNull();
  }
});

it.each(backboneCases.filter(row => row.family === 'sdxl' && row.tokens > 75 && row.strategy !== 'single'))(
  'requires the complete preview identity for $algo $strategy $tokens-token training', ({family, algo, strategy, tokens}) => {
    const {config, candidate} = backboneAdapterCase(family, algo, strategy, tokens);
    for (const invalid of [
      {...candidate, preview_operator_components:['backbone', 'text_encoder']},
      {...candidate, preview_linear_forward:'native-bf16'},
      {...candidate, preview_linear_implementation:'linear-bf16-operands-fp32-compute-v1'},
      {...candidate, preview_adapter_implementation:'lora-bf16-operands-fp32-contractions-v1'},
    ]) expect(confirmedTrainingComputePolicy(invalid, config)).toBeNull();
    const obsolete: Record<string, unknown> = {...candidate};
    for (const key of Object.keys(obsolete)) if (key.startsWith('preview_')) delete obsolete[key];
    obsolete.id = `dtk-sdxl-backbone-${algo}-${strategy}-bf16-compute-v1`;
    expect(confirmedTrainingComputePolicy(obsolete, config)).toBeNull();
    expect(confirmedTrainingComputePolicy(candidate, {...config, loop:{...config.loop, gpu_count:1}})).toBeNull();
    expect(confirmedTrainingComputePolicy(candidate, {...config, model:{...config.model, sdxl_max_token_length:75}})).toBeNull();
    if (algo === 'lokr' && strategy === 'ddp') {
      const legacy = {...animaPolicy, id:'dtk-sdxl-long-text-bf16-conv-fp32-linear-compute-v1', conv_forward:'fp32-output-bf16', conv_implementation:'conv2d-fp32-output-bf16-v1', sdxl_max_token_length:tokens};
      expect(confirmedTrainingComputePolicy(legacy, config)).toBeNull();
    }
  },
);

it('does not extend preview metadata to single-GPU, short-caption, Anima or text-training policies', () => {
  const rows = [
    backboneAdapterCase('sdxl', 'lora', 'single', 150),
    backboneAdapterCase('sdxl', 'lora', 'ddp', 75),
    backboneAdapterCase('anima', 'lora', 'ddp'),
    textLoraCase('sdxl', true, 2),
  ];
  for (const {config, candidate} of rows) {
    expect(confirmedTrainingComputePolicy(candidate, config)).toEqual(candidate);
    expect(confirmedTrainingComputePolicy({...candidate, preview_operator_components:['backbone']}, config)).toBeNull();
    expect(trainingComputePolicyHint(confirmedTrainingComputePolicy(candidate, config), false)).not.toContain('重复预览');
  }
});

it.each(['single', 'ddp', 'fsdp'] as const)('describes frozen-encoder LoRA %s and restores the original draft on disable', strategy => {
  const {config:initial, candidate} = backboneAdapterCase('sdxl', 'lora', strategy, 150);
  function Editor() {
    const [value,setValue] = React.useState(initial);
    return <><SchemaForm schema={schema} value={value} onChange={setValue} computePolicy={candidate} compact showAdvanced groupFilter={['loop','memory']}/><output data-testid="draft">{JSON.stringify(value)}</output></>;
  }
  render(<Editor/>);
  expect(screen.getByRole('status', {name:'混合精度'})).toHaveTextContent('BF16（FP32 线性层与 LoRA 运算）');
  const hint = screen.getByTestId('field-loop.deterministic');
  expect(hint).toHaveTextContent(strategy === 'fsdp' ? 'FSDP 将模型参数、适配器梯度和优化器状态分摊到多卡' : strategy === 'ddp' ? 'DDP 每卡保留完整模型' : '本次使用单卡');
  expect(hint).toHaveTextContent('文本编码器保持冻结');
  expect(hint).toHaveTextContent('保留 BF16 舍入和输出');
  expect(hint).toHaveTextContent('LoRA 也保留 BF16 操作数舍入和中间结果');
  expect(hint).toHaveTextContent('卷积使用 FP32，输出转回 BF16');
  expect(hint).toHaveTextContent('可能增加显存和耗时');
  expect(hint).toHaveTextContent('相同计算策略和运行环境及标签长度');
  expect(hint).toHaveTextContent('不可混用旧策略的训练状态');
  if (strategy !== 'single') expect(hint).toHaveTextContent('提高重复预览的一致性');
  else expect(hint).not.toHaveTextContent('提高重复预览的一致性');
  expect(JSON.parse(screen.getByTestId('draft').textContent!)).toEqual(initial);
  fireEvent.click(screen.getByRole('checkbox', {name:'可复现训练'}));
  expect(screen.getByRole('combobox', {name:'混合精度'})).toHaveTextContent('BF16');
  expect(screen.getByRole('checkbox', {name:'允许 TF32'})).toBeChecked();
  expect(screen.getByRole('combobox', {name:'注意力后端'})).toHaveTextContent('xFormers');
  expect(JSON.parse(screen.getByTestId('draft').textContent!)).toEqual({...initial, loop:{...initial.loop, deterministic:false}});
});

it.each(backboneCases)('explains $family $algo $strategy $tokens-token arithmetic in both languages', ({family, algo, strategy, tokens}) => {
  const {config, candidate} = backboneAdapterCase(family, algo, strategy, tokens);
  const confirmed = confirmedTrainingComputePolicy(candidate, config);
  const chinese = trainingComputePolicyHint(confirmed, false)!;
  const english = trainingComputePolicyHint(confirmed, true)!;
  expect(english).not.toMatch(/[\u4e00-\u9fff]/);
  expect(english).toContain('Text encoders stay frozen');
  expect(english).toContain('Memory use and runtime may increase');
  expect(english).toContain('states from previous policies cannot be mixed');
  if (family === 'sdxl' && tokens > 75 && strategy !== 'single') {
    expect(english).toContain('Preview backbone linear layers also use FP32 operations with BF16 rounding and outputs');
    expect(chinese).toContain('预览的主模型线性层也保留 BF16 舍入与输出');
  } else {
    expect(english).not.toContain('Preview backbone');
    expect(chinese).not.toContain('重复预览');
  }
  if (algo === 'lokr') {
    expect(chinese).toContain('LoKr 运算保持原生实现');
    expect(english).toContain('LoKr operations remain native');
    expect(chinese).not.toContain('LoRA');
  }
  if (family === 'sdxl' && algo === 'lokr' && tokens === 75) {
    expect(chinese).toContain('原生 BF16 前向');
    expect(english).toContain('linear forward uses native BF16');
  } else {
    expect(chinese).toContain('线性层保留 BF16 舍入和输出，矩阵运算使用 FP32');
    expect(english).toContain('linear operations retain BF16 rounding and outputs, with FP32 matrix operations');
  }
  if (strategy === 'fsdp') {
    expect(english).toContain('gathering parameters in BF16 and reducing gradients in FP32');
    expect(chinese).not.toContain('每卡保留完整模型');
  } else if (strategy === 'ddp') {
    expect(english).toContain('DDP keeps the complete model on each GPU');
    expect(chinese).not.toContain('分摊到多卡');
  }
});

it.each(['anima', 'sdxl', 'krea2'])('confirms %s text-only and joint LoRA policies including exact component and length identities', family => {
  for (const backbone of [false, true]) for (const gpus of [1, 2]) {
    const {config, candidate} = textLoraCase(family, backbone, gpus);
    expect(confirmedTrainingComputePolicy(candidate, config)).toEqual(candidate);
    expect(confirmedTrainingComputePolicy(candidate, {...config, dataset: {...config.dataset, text_encoding: 'auto'}})).toEqual(candidate);
    for (const key of Object.keys(candidate)) {
      const missing: Record<string, unknown> = {...candidate}; delete missing[key];
      expect(confirmedTrainingComputePolicy(missing, config)).toBeNull();
    }
    for (const invalid of [
      {...candidate, trainable_components: []}, {...candidate, operator_components: ['text_encoder']},
      {...candidate, distributed_strategy: gpus === 1 ? 'ddp' : 'single'},
      {...candidate, linear_forward: 'native-bf16'}, {...candidate, adapter_implementation: 'unknown'},
      {...candidate, fsdp_param_dtype: 'bfloat16'},
      {...candidate, sdxl_max_token_length: family === 'sdxl' ? 225 : 150},
    ]) expect(confirmedTrainingComputePolicy(invalid, config)).toBeNull();
    expect(currentTrainingComputePolicy(candidate, config, JSON.stringify(config), true)).toBeNull();
    expect(currentTrainingComputePolicy(candidate, {...config, training: {...config.training, train_backbone: !backbone}}, JSON.stringify(config), false)).toBeNull();
  }
});

it('rejects unsupported text LoRA drafts instead of displaying a managed precision', () => {
  const {config, candidate} = textLoraCase('anima');
  for (const invalid of [
    {...config, training: {...config.training, train_text_encoder: false}},
    {...config, loop: {...config.loop, deterministic: false}},
    {...config, loop: {...config.loop, distributed_strategy: 'fsdp'}},
    {...config, loop: {...config.loop, gpu_count: 1.5}},
    {...config, adapter: {...config.adapter, algo: 'lokr'}},
    {...config, adapter: {...config.adapter, mode: 'merged'}},
    {...config, adapter: {...config.adapter, param_dtype: 'bf16'}},
    {...config, adapter: {...config.adapter, dora: true}},
    {...config, adapter: {...config.adapter, rules: [{match:'*', algo:'loha'}]}},
    {...config, memory: {...config.memory, offload_text_encoder: true}},
    {...config, memory: {...config.memory, base_precision: 'fp8_e4m3'}},
    {...config, memory: {...config.memory, compile: true}},
    {...config, memory: {...config.memory, blocks_to_swap: 1}},
    {...config, memory: {...config.memory, activation_checkpointing: 'unsloth'}},
    {...config, dataset: {...config.dataset, text_encoding: 'cached'}},
  ]) expect(confirmedTrainingComputePolicy(candidate, invalid)).toBeNull();
});

it('shows effective text-only precision and restores the original draft when reproducibility is disabled', () => {
  const {config: initial, candidate} = textLoraCase('sdxl');
  function Editor() {
    const [value, setValue] = React.useState(initial);
    return <><SchemaForm schema={schema} value={value} onChange={setValue} computePolicy={candidate} compact showAdvanced groupFilter={['loop','memory']}/><output data-testid="draft">{JSON.stringify(value)}</output></>;
  }
  render(<Editor/>);
  expect(screen.getByRole('status', {name:'混合精度'})).toHaveTextContent('使用 FP32 运算提高可复现性');
  expect(screen.getByTestId('field-loop.deterministic')).toHaveTextContent('文本编码器 LoRA 训练');
  expect(JSON.parse(screen.getByTestId('draft').textContent!)).toEqual(initial);
  fireEvent.click(screen.getByRole('checkbox', {name:'可复现训练'}));
  expect(screen.getByRole('combobox', {name:'混合精度'})).toHaveTextContent('BF16');
  expect(screen.getByRole('checkbox', {name:'允许 TF32'})).toBeChecked();
  expect(JSON.parse(screen.getByTestId('draft').textContent!)).toEqual({...initial, loop:{...initial.loop, deterministic:false}});
});

it.each([150, 225])('keeps SDXL %s-token LoKr separate from the previous short-caption policy', length => {
  const config = animaLokrConfig(); config.model.family = 'sdxl'; config.model.sdxl_max_token_length = length;
  const candidate = {...animaPolicy, id:'dtk-sdxl-long-text-bf16-conv-fp32-linear-compute-v1', conv_forward:'fp32-output-bf16', conv_implementation:'conv2d-fp32-output-bf16-v1', sdxl_max_token_length:length};
  expect(confirmedTrainingComputePolicy(candidate, config)).toEqual(candidate);
  expect(confirmedTrainingComputePolicy(sdxlPolicy, config)).toBeNull();
  for (const invalid of [
    {...candidate, sdxl_max_token_length: 75}, {...candidate, linear_forward: 'native-bf16'},
    {...candidate, conv_implementation: 'unknown'}, {...candidate, adapter_implementation: 'lora-bf16-operands-fp32-contractions-v1'},
  ]) expect(confirmedTrainingComputePolicy(invalid, config)).toBeNull();
  expect(confirmedTrainingComputePolicy(candidate, {...config, model:{...config.model, sdxl_max_token_length:75}})).toBeNull();
});

it.each(['ddp','fsdp'] as const)('accepts Anima %s only for its complete supported multi-GPU scope', strategy => {
  const config = animaDualConfig(strategy);
  const verified = strategy === 'ddp' ? animaDdpPolicy : animaFsdpPolicy;
  for (const gpu_count of [2,4]) for (const activation_checkpointing of ['none','block']) {
    expect(confirmedTrainingComputePolicy(verified,{...config,loop:{...config.loop,gpu_count},memory:{...config.memory,activation_checkpointing}})).toEqual(verified);
  }
  for (const changed of [
    ...[0,1,1.5,'2'].map(gpu_count=>({...config,loop:{...config.loop,gpu_count}})),
    {...config,loop:{...config.loop,distributed_strategy:strategy==='ddp'?'fsdp':'ddp'}},
    {...config,training:{...config.training,mode:strategy==='ddp'?'full':'adapter'}},
    {...config,training:{...config.training,train_backbone:false}},
    {...config,training:{...config.training,train_text_encoder:true}},
    {...config,loop:{...config.loop,deterministic:false}},
    {...config,loop:{...config.loop,mixed_precision:'fp16'}},
    {...config,model:{...config.model,family:'sdxl'}},
    {...config,memory:{...config.memory,activation_checkpointing:'unsloth'}},
  ]) expect(confirmedTrainingComputePolicy(verified,changed)).toBeNull();
  if (strategy === 'ddp') {
    for (const adapter of [
      {...config.adapter,algo:'lora'}, {...config.adapter,mode:'weight'},
      {...config.adapter,dora:true}, {...config.adapter,param_dtype:'bf16'},
      {...config.adapter,rules:[{match:'*',algo:'loha'}]},
    ]) expect(confirmedTrainingComputePolicy(verified,{...config,adapter})).toBeNull();
    expect(confirmedTrainingComputePolicy(verified,{...config,memory:{...config.memory,base_precision:'fp8_e4m3'}})).toBeNull();
  }
});

it.each(['ddp','fsdp'] as const)('rejects incomplete or conflicting Anima %s metadata without accepting stale validation', strategy => {
  const config = animaDualConfig(strategy);
  const verified = strategy === 'ddp' ? animaDdpPolicy : animaFsdpPolicy;
  for (const key of Object.keys(verified)) {
    const incomplete: Record<string, unknown> = {...verified}; delete incomplete[key];
    expect(confirmedTrainingComputePolicy(incomplete,config)).toBeNull();
  }
  for (const changed of [
    {...verified,id:'diagnostic-anima-dual-candidate'},
    {...verified,linear_forward:'native-bf16'},
    {...verified,linear_backward_implementation:'linear-bf16-forward-fp32-backward-v1'},
    {...verified,conv_forward:'fp32-output-bf16'},
    {...verified,fsdp_param_dtype:'float32'}, {...verified,fsdp_reduce_dtype:'bfloat16'},
    ...(strategy === 'ddp' ? [{...verified,fsdp_param_dtype:'bfloat16',fsdp_reduce_dtype:'float32'}] : []),
  ]) expect(confirmedTrainingComputePolicy(changed,config)).toBeNull();
  const checked = JSON.stringify(config);
  expect(currentTrainingComputePolicy(verified,config,checked,false)).toEqual(verified);
  expect(currentTrainingComputePolicy(verified,config,checked,true)).toBeNull();
  for (const changed of [
    {...config,loop:{...config.loop,seed:config.loop.seed+1}},
    {...config,loop:{...config.loop,gpu_count:4}},
    {...config,loop:{...config.loop,distributed_strategy:strategy==='ddp'?'fsdp':'ddp'}},
  ]) expect(currentTrainingComputePolicy(verified,changed,checked,false)).toBeNull();
});

it('does not interchange the three Anima policy identities', () => {
  const rows = [[animaPolicy,fullConfig()],[animaDdpPolicy,animaDualConfig('ddp')],[animaFsdpPolicy,animaDualConfig('fsdp')]] as const;
  for (const [index,[candidate]] of rows.entries()) for (const [other,[,config]] of rows.entries()) {
    expect(confirmedTrainingComputePolicy(candidate,config)).toEqual(index===other ? candidate : null);
  }
});

it.each(['ddp','fsdp'] as const)('renders Anima %s managed fields and restores the requested draft when disabled', strategy => {
  const initial = animaDualConfig(strategy);
  const candidate = strategy === 'ddp' ? animaDdpPolicy : animaFsdpPolicy;
  function Editor() {
    const [value,setValue] = React.useState(initial);
    return <><SchemaForm schema={schema} value={value} onChange={setValue} computePolicy={candidate} compact showAdvanced groupFilter={['loop','memory']}/><output data-testid="draft">{JSON.stringify(value)}</output></>;
  }
  render(<Editor/>);
  expect(screen.getByRole('status',{name:'混合精度'})).toHaveTextContent('BF16（线性层 FP32 运算）');
  const hint=screen.getByTestId('field-loop.deterministic');
  expect(hint).toHaveTextContent(strategy==='ddp' ? '每卡保留完整主模型' : '分担参数、梯度和优化器状态');
  expect(hint).toHaveTextContent('先按 BF16 舍入输入与参数');
  expect(hint).not.toHaveTextContent('使用单卡');
  if (strategy==='fsdp') expect(hint).toHaveTextContent('按 BF16 汇集参数，以 FP32 汇总梯度');
  expect(JSON.parse(screen.getByTestId('draft').textContent!)).toEqual(initial);
  fireEvent.click(screen.getByRole('checkbox',{name:'可复现训练'}));
  expect(screen.getByRole('combobox',{name:'混合精度'})).toHaveTextContent('BF16');
  expect(screen.getByRole('checkbox',{name:'允许 TF32'})).toBeChecked();
  expect(screen.getByRole('combobox',{name:'注意力后端'})).toHaveTextContent('xFormers');
  expect(JSON.parse(screen.getByTestId('draft').textContent!)).toEqual({...initial,loop:{...initial.loop,deterministic:false}});
});

it.each(['ddp','fsdp'] as const)('describes Anima %s parallelism accurately in English', async strategy => {
  await i18n.changeLanguage('en');
  render(<SchemaForm schema={schema} value={animaDualConfig(strategy)} onChange={()=>{}} computePolicy={strategy==='ddp'?animaDdpPolicy:animaFsdpPolicy} compact showAdvanced groupFilter={['loop','memory']}/>);
  const hint=screen.getByTestId('field-loop.deterministic');
  expect(hint).toHaveTextContent(strategy==='ddp'?'each GPU keeps the complete backbone':'shards parameters, gradients and optimizer states');
  expect(hint).toHaveTextContent('return BF16 outputs');
  expect(hint.textContent).not.toMatch(/[\u4e00-\u9fff]/);
  expect(hint.textContent).not.toContain('all gradients use FP32');
  expect(screen.getByRole('status',{name:'Mixed Precision'})).toHaveTextContent('BF16 (FP32 linear operations)');
});

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
  expect(screen.getByRole('tooltip')).toHaveTextContent('Configuration validation explains the effective settings');
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
  expect(configFieldHelp('loop.mixed_precision','')).toContain('不等同于全程 FP32');
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
