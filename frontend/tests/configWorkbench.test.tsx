import React from 'react';
import {fireEvent, render, screen, within} from '@testing-library/react';
import {describe, it, expect} from 'vitest';
import {presentConfigIssues, configTabForPath} from '../src/utils/configPresentation';
import {SchemaForm} from '../src/schema/SchemaForm/SchemaForm';
import BucketInspector from '../src/pages/TrainConfig/BucketInspector';
import type {Plan} from '../src/api/types';
import '../src/i18n';

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
});
