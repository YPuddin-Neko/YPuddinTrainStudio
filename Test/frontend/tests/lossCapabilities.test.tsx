import React from 'react';
import {fireEvent, render, screen} from '@testing-library/react';
import {describe, expect, it} from 'vitest';
import {SchemaForm} from '../../../frontend/src/schema/SchemaForm/SchemaForm';
import schema from '../../../frontend/src/schema/train-schema.json';
import {schemaDefaults} from '../../../frontend/src/utils/config';
import type {FamilyInfo} from '../../../frontend/src/api/types';
import '../../../frontend/src/i18n';

function Editor({ddpm=false, prediction='epsilon', objective={}}: {ddpm?:boolean;prediction?:string;objective?:Record<string,unknown>}) {
  const initial=schemaDefaults(schema);
  const [value,setValue]=React.useState({...initial,model:{...initial.model,family:ddpm?'sdxl':'anima',prediction_type:prediction},objective:{...initial.objective,...objective}});
  const family={name:ddpm?'sdxl':'anima',objective:ddpm?'ddpm':'rectified_flow',capabilities:[],weights:[],presets:[],sampling:{}} as unknown as FamilyInfo;
  return <><SchemaForm schema={schema} value={value} onChange={setValue} family={family} compact showAdvanced groupFilter={['objective']}/><output data-testid="draft">{JSON.stringify(value)}</output></>;
}

describe('prediction-specific loss controls',()=>{
  it('hides inactive DDPM modifiers for Flow but lets imported incompatible values be cleared',()=>{
    render(<Editor objective={{scale_v_pred_loss_like_noise_pred:true,v_pred_like_loss:0.2,debiased_estimation_loss:true}}/>);
    expect(screen.getAllByText('此损失参数仅适用于 SDXL，请关闭或设为 0 后再使用当前模型。')).toHaveLength(3);
    fireEvent.click(screen.getByRole('checkbox',{name:'按 ε 预测尺度缩放 v 损失'}));
    fireEvent.change(screen.getByRole('spinbutton',{name:'附加 v 预测损失系数'}),{target:{value:'0'}});
    fireEvent.click(screen.getByRole('checkbox',{name:'去偏损失加权'}));
    expect(screen.queryByTestId('field-objective.scale_v_pred_loss_like_noise_pred')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-objective.v_pred_like_loss')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-objective.debiased_estimation_loss')).not.toBeInTheDocument();
    expect(JSON.parse(screen.getByTestId('draft').textContent!).objective).toMatchObject({scale_v_pred_loss_like_noise_pred:false,v_pred_like_loss:0,debiased_estimation_loss:false});
  });
  it('keeps an incompatible v scaling switch editable in an epsilon draft',()=>{
    render(<Editor ddpm objective={{scale_v_pred_loss_like_noise_pred:true}}/>);
    expect(screen.getByText('此参数与当前预测方式不兼容，请关闭此项或选择对应的预测方式。')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox',{name:'按 ε 预测尺度缩放 v 损失'}));
    expect(screen.queryByTestId('field-objective.scale_v_pred_loss_like_noise_pred')).not.toBeInTheDocument();
  });
  it('keeps an incompatible epsilon-only modifier editable in a v draft',()=>{
    render(<Editor ddpm prediction="v_prediction" objective={{v_pred_like_loss:0.2}}/>);
    fireEvent.change(screen.getByRole('spinbutton',{name:'附加 v 预测损失系数'}),{target:{value:'0'}});
    expect(screen.queryByTestId('field-objective.v_pred_like_loss')).not.toBeInTheDocument();
    expect(screen.getByTestId('field-objective.scale_v_pred_loss_like_noise_pred')).toBeInTheDocument();
  });
});
