import React from 'react';
import {fireEvent,render,screen} from '@testing-library/react';
import {beforeEach,expect,it} from 'vitest';
import {SchemaForm} from '../src/schema/SchemaForm/SchemaForm';
import schema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';

beforeEach(async()=>{await i18n.changeLanguage('zh-CN');});
function Editor({initial,groups}: {initial:Record<string,any>;groups:string[]}) {
  const [value,setValue]=React.useState(initial);
  return <><SchemaForm schema={schema} value={value} onChange={setValue} groupFilter={groups} compact showAdvanced/><output data-testid="value">{JSON.stringify(value)}</output></>;
}
const value=()=>JSON.parse(screen.getByTestId('value').textContent!);
function select(name:string,option:string) {fireEvent.click(screen.getByRole('combobox',{name}));fireEvent.click(screen.getByRole('option',{name:option}));}

it('selects a supported optimizer and preserves the explicit custom-class escape hatch',()=>{
  render(<Editor initial={{optimizer:{type:'adamw',lr:.0002}}} groups={['optimizer']}/>);
  expect(screen.queryByRole('textbox',{name:'优化器'})).not.toBeInTheDocument();
  select('优化器','Lion');expect(value().optimizer.type).toBe('lion');
  select('优化器','自定义 Python 类…');
  fireEvent.change(screen.getByRole('textbox',{name:'优化器 自定义类'}),{target:{value:'my_package.CustomOptimizer'}});
  expect(value().optimizer.type).toBe('my_package.CustomOptimizer');
  select('优化器','AdamW');expect(screen.queryByRole('textbox',{name:'优化器 自定义类'})).not.toBeInTheDocument();
});

it('pairs independent inference choices and shows ER controls only for ER-SDE',()=>{
  render(<Editor initial={{sampling:{enabled:true,sampler:'euler',scheduler:'uniform'}}} groups={['sampling']}/>);
  expect(screen.queryByRole('combobox',{name:'ER-SDE 求解阶数'})).not.toBeInTheDocument();
  select('采样器','ER-SDE');select('采样调度器','SGM Uniform');
  expect(value().sampling).toMatchObject({sampler:'er_sde',scheduler:'sgm_uniform'});
  select('ER-SDE 求解阶数','2');expect(value().sampling.er_sde_order).toBe(2);
  select('采样器','Heun');expect(screen.queryByRole('combobox',{name:'ER-SDE 求解阶数'})).not.toBeInTheDocument();
});

it('keeps prior weighting out of training sources and exposes it through external-source compatibility settings',()=>{
  render(<Editor initial={{dataset:{sources:[{path:'/dataset',repeats:1,caption_ext:'.txt',is_reg:false,prior_weight:.5}]}}} groups={['dataset']}/>);
  expect(screen.queryByRole('combobox',{name:'标签格式 1'})).not.toBeInTheDocument();
  expect(screen.queryByRole('spinbutton',{name:'正则损失权重 1'})).not.toBeInTheDocument();
  expect(screen.queryByRole('checkbox',{name:'这是正则集'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByText('高级来源设置'));
  fireEvent.click(screen.getByText('外部 / 旧版来源兼容设置'));
  fireEvent.click(screen.getByRole('checkbox',{name:'外部来源用于正则训练'}));
  expect(screen.getByRole('spinbutton',{name:'正则损失权重 1'})).toHaveValue(.5);
  expect(value().dataset.sources[0]).toMatchObject({path:'/dataset',caption_ext:'.txt',is_reg:true,prior_weight:.5});
});
