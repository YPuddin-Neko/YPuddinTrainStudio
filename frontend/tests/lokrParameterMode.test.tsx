import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import schema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';

beforeEach(async()=>{await i18n.changeLanguage('zh-CN');});
function Editor({algo='lokr',rank='full'}:{algo?:string;rank?:number|string}){
  const [value,setValue]=React.useState({adapter:{algo,rank,alpha:8,factor:8}});
  return <><SchemaForm schema={schema} value={value} onChange={setValue as any} compact showAdvanced groupFilter={['adapter']}/><output data-testid="config-value">{JSON.stringify(value)}</output></>;
}
const config=()=>JSON.parse(screen.getByTestId('config-value').textContent!);
function mode(option:string){fireEvent.click(screen.getByRole('combobox',{name:'LoKr 参数形式'}));fireEvent.click(screen.getByRole('option',{name:option}));}

describe('LoKr parameter mode',()=>{
  it('switches Full and low rank through real values while hiding only the unused Alpha control',()=>{
    render(<Editor/>);
    expect(screen.getByRole('combobox',{name:'LoKr 参数形式'})).toHaveTextContent('Full · 完整因子矩阵');
    expect(screen.getByText('仍保留 LoKr 结构，不是全量微调；此模式不使用 Alpha 缩放。')).toBeInTheDocument();
    expect(screen.queryByTestId('field-adapter.alpha')).not.toBeInTheDocument();
    expect(within(screen.getByTestId('field-adapter.rank')).queryByRole('spinbutton')).not.toBeInTheDocument();
    expect(config().adapter).toMatchObject({algo:'lokr',rank:'full',alpha:8,factor:8});
    mode('低秩 · 分解因子矩阵');
    expect(config().adapter.rank).toBe(16);
    const rank=within(screen.getByTestId('field-adapter.rank')).getByRole('spinbutton');
    const alpha=within(screen.getByTestId('field-adapter.alpha')).getByRole('spinbutton');
    expect(alpha).toHaveValue(8);
    fireEvent.change(rank,{target:{value:'32'}});fireEvent.change(alpha,{target:{value:'16'}});
    expect(config().adapter).toMatchObject({rank:32,alpha:16});
    mode('Full · 完整因子矩阵');
    expect(config().adapter).toMatchObject({algo:'lokr',rank:'full',alpha:16,factor:8});
    expect(screen.queryByTestId('field-adapter.alpha')).not.toBeInTheDocument();
    mode('低秩 · 分解因子矩阵');
    expect(within(screen.getByTestId('field-adapter.alpha')).getByRole('spinbutton')).toHaveValue(16);
  });

  it('retains ordinary LoRA rank and Alpha fields without presenting LoKr-only modes',()=>{
    render(<Editor algo="lora" rank={16}/>);
    expect(screen.queryByRole('combobox',{name:'LoKr 参数形式'})).not.toBeInTheDocument();
    expect(screen.getByTestId('field-adapter.rank')).toBeInTheDocument();
    const alpha=within(screen.getByTestId('field-adapter.alpha')).getByRole('spinbutton');
    fireEvent.change(alpha,{target:{value:'4'}});
    expect(config().adapter).toMatchObject({algo:'lora',rank:16,alpha:4});
  });
});
