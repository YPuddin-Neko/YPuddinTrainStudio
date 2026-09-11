import React from 'react';
import {fireEvent, render, screen, within} from '@testing-library/react';
import {describe, it, expect} from 'vitest';
import {presentConfigIssues, configTabForPath} from '../src/utils/configPresentation';
import {SchemaForm} from '../src/schema/SchemaForm/SchemaForm';
import BucketInspector from '../src/pages/TrainConfig/BucketInspector';
import type {Plan} from '../src/api/types';
import '../src/i18n';
import trainSchema from '../src/schema/train-schema.json';

describe('compact configuration workbench contracts', () => {
  it('normalizes FastAPI locations and translates required components without crashing', () => {
    const issues = presentConfigIssues([
      {loc:['body','config','model','dit_path'],msg:'model.dit_path is required for anima'},
      {loc:'model',msg:'model.dit_path is required for anima'},
      {loc:['body','config','dataset','sources',0,'path'],msg:'file is missing'},
      {loc:['body','priority'],msg:'Input should be a valid integer'},
    ]);
    expect(issues).toHaveLength(3);
    expect(issues[0]).toMatchObject({path:'model.dit_path',label:'主模型 / DiT',tab:'model',message:'请填写或选择主模型 / DiT'});
    expect(issues[1]).toMatchObject({path:'dataset.sources.0.path',tab:'data'});
    expect(issues[2].path).toBe('priority');
    expect(configTabForPath('dataset.batch_size')).toBe('train');
  });

  it('allows renaming optimizer keys without overwriting a duplicate and preserves JSON value types', () => {
    const schema = {properties:{optimizer:{$ref:'#/$defs/Optimizer'}},$defs:{Optimizer:{type:'object',properties:{args:{type:'object',additionalProperties:true,'x-ui':{control:'key-value'}}}}}};
    let latest: any;
    function Harness() {const [value,setValue]=React.useState<Record<string, any>>({optimizer:{args:{epsilon:1,existing:2}}});latest=value;return <SchemaForm schema={schema} value={value} onChange={setValue} compact/>;}
    render(<Harness/>);
    const editor=screen.getByTestId('key-value-editor');
    const key=within(editor).getByRole('textbox',{name:'键 epsilon'});
    fireEvent.change(key,{target:{value:'existing'}});fireEvent.blur(key);
    expect(screen.getByRole('alert')).toHaveTextContent('参数名不能为空或重复');
    expect(latest.optimizer.args).toEqual({epsilon:1,existing:2});
    fireEvent.change(key,{target:{value:'betas'}});fireEvent.blur(key);
    const value=within(editor).getByRole('textbox',{name:'值 betas'});
    fireEvent.change(value,{target:{value:'[0.9,0.999]'}});
    expect(latest.optimizer.args).toEqual({betas:[0.9,0.999],existing:2});
  });

  it('renders actual bucket counts and supports shape selection and a detailed table even if weights are missing', () => {
    const plan={ok:false,images:8,items:16,captioned:7,total_steps:12,steps_per_epoch:6,buckets:[{w:512,h:768,items:10,batches:4},{w:768,h:512,items:6,batches:2}]} as Plan;
    render(<BucketInspector plan={plan} loading={false} onData={() => {}}/>);
    expect(screen.getByRole('button',{name:'512 × 768, 10 样本'})).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'512 × 768, 10 样本'}));
    expect(screen.getByText('10 样本 · 4 批次 / 轮')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'分桶明细表'}));
    expect(screen.getByRole('table')).toHaveTextContent('512 × 768104');
    expect(screen.getByRole('table')).toHaveTextContent('768 × 51262');
  });

  it('switches native and bucket controls without deleting retained settings or hiding legacy defaults', () => {
    function Harness() { const [value, setValue] = React.useState<Record<string, any>>({ dataset: { resolutions: [768], aspect_ratio_limit: 2 } }); return <><SchemaForm schema={trainSchema} value={value} onChange={setValue} compact showAdvanced groupFilter={['dataset']}/><output data-testid="native-config">{JSON.stringify(value)}</output></>; }
    render(<Harness/>);
    expect(screen.getByTestId('field-dataset.resolutions')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('combobox',{name:'分辨率模式'}));
    fireEvent.click(screen.getByRole('option',{name:'原生 · 每图独立尺寸'}));
    for (const name of ['resolutions','aspect_ratio_limit','area_tolerance','bucket_step','bucket_no_upscale']) expect(screen.queryByTestId(`field-dataset.${name}`)).not.toBeInTheDocument();
    expect(screen.getByRole('spinbutton',{name:'原生像素预算'})).toHaveValue(1048576);
    expect(screen.getByRole('combobox',{name:'超出预算时'})).toHaveTextContent('等比缩小到预算内');
    fireEvent.change(screen.getByRole('spinbutton',{name:'原生最长边'}),{target:{value:'2048'}});
    fireEvent.click(screen.getByRole('combobox',{name:'分辨率模式'}));
    fireEvent.click(screen.getByRole('option',{name:'分桶 · 统一基准面积'}));
    expect(screen.queryByRole('spinbutton',{name:'原生像素预算'})).not.toBeInTheDocument();
    expect(screen.getByRole('spinbutton',{name:'最大长宽比'})).toHaveValue(2);
    expect(JSON.parse(screen.getByTestId('native-config').textContent!).dataset).toEqual({resolution_mode:'bucket',resolutions:[768],aspect_ratio_limit:2,native_max_side:2048});
  });

  it('distinguishes native forwards from logical batches in the plan', () => {
    const plan = {ok:true,images:4,items:8,captioned:4,total_steps:4,steps_per_epoch:2,buckets:[{w:512,h:768,items:8,batches:4}],native:{images:4,downscaled:1,sizes:1,logical_batches:2,max_pixels:1048576,alignment:32,batch_size:4,forward_groups:4}} as Plan;
    render(<BucketInspector plan={plan} loading={false} onData={()=>{}}/>);
    expect(screen.getByText('原生尺寸分布')).toBeInTheDocument();
    expect(screen.getByText('逻辑批次 / 轮').nextElementSibling).toHaveTextContent('2');
    expect(screen.getByText('分组前向 / 轮').nextElementSibling).toHaveTextContent('4');
    fireEvent.click(screen.getByRole('button',{name:'分桶明细表'}));
    expect(screen.getByRole('columnheader',{name:'前向次数'})).toBeInTheDocument();
    expect(screen.queryByRole('columnheader',{name:'批次'})).not.toBeInTheDocument();
  });
});
