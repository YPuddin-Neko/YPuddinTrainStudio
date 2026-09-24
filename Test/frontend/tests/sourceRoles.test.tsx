import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import { SchemaForm, type SourceRoleInfo } from '../../../frontend/src/schema/SchemaForm/SchemaForm';
import schema from '../../../frontend/src/schema/train-schema.json';
import i18n from '../../../frontend/src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
function Editor({ roles, initial, scoped = true }: { roles: SourceRoleInfo[]; initial: Record<string, any>; scoped?: boolean }) {
  const [value, setValue] = React.useState(initial);
  return <><SchemaForm schema={schema} value={value} onChange={setValue} sourceRoles={roles} versionSources={scoped} groupFilter={['dataset']} compact/><output data-testid="saved">{JSON.stringify(value)}</output></>;
}
const saved = () => JSON.parse(screen.getByTestId('saved').textContent!);
const source = { path: '/project/v2/reg/dogs', is_reg: false, repeats: 2, caption_ext: '.json', prior_weight: .3 };

it('shows the server-derived managed purpose/count and never exposes a manual role toggle', () => {
  render(<Editor initial={{dataset:{sources:[source]}}} roles={[{path:source.path,section:'dataset',is_reg:true,managed:true,root:'/project/v2/reg',origin:'version',images:17}]}/>);
  expect(screen.getByText('dogs')).toBeVisible();
  expect(screen.getByText('正则集 · 17 张图片')).toBeVisible();
  expect(screen.queryByText('高级来源设置')).not.toBeInTheDocument();
  expect(screen.getByRole('spinbutton', {name: '重复次数 1'})).toBeVisible();
  expect(screen.queryByRole('combobox',{name:'用途 1'})).not.toBeInTheDocument();
  expect(screen.getByRole('textbox', {name: '图片目录 1'})).toHaveValue(source.path);
  expect(screen.queryByText('目录归属：当前版本')).not.toBeInTheDocument();
  expect(screen.queryByRole('combobox',{name:'用途 1'})).not.toBeInTheDocument();
  expect(screen.queryByRole('combobox',{name:/标签格式/})).not.toBeInTheDocument();
  expect(screen.getByRole('spinbutton',{name:'正则损失权重 1'})).toHaveValue(.3);
  fireEvent.change(screen.getByRole('spinbutton',{name:'重复次数 1'}),{target:{value:'3'}});
  expect(saved().dataset.sources[0]).toMatchObject({path:source.path,repeats:3,caption_ext:'.json'});
});

it('preserves legacy role and format until the visible purpose control changes it', () => {
  const path = 'D:\\pictures\\reg_named_folder';
  render(<Editor initial={{dataset:{sources:[{...source,path,is_reg:false}]}}} roles={[{path,section:'dataset',is_reg:false,managed:false,root:null,origin:'registered',images:2}]}/>);
  expect(screen.getByText('reg_named_folder')).toBeVisible();
  expect(screen.getByText('训练集 · 2 张图片')).toBeVisible();
  fireEvent.click(screen.getByRole('combobox', {name:'用途 1'}));
  fireEvent.click(screen.getByRole('option', {name:'正则集'}));
  expect(saved().dataset.sources[0]).toMatchObject({path,is_reg:true,caption_ext:'.json'});
});

it('does not guess role from folder text or allow compatibility changes while ownership is pending', () => {
  render(<Editor initial={{dataset:{sources:[source]}}} roles={[]}/>);
  expect(screen.getByText(/用途待确认/)).toBeVisible();
  expect(screen.getByText(/用途待确认/)).toBeVisible();
  expect(screen.queryByText('外部来源用途')).not.toBeInTheDocument();
  expect(screen.queryByRole('combobox',{name:'用途 1'})).not.toBeInTheDocument();
  expect(saved().dataset.sources[0].is_reg).toBe(false);
});

it('shows registered legacy metadata when a supplied recipe omits the compatibility flag', () => {
  render(<Editor initial={{dataset:{sources:[{path:'/legacy/class',repeats:1}]}}} roles={[{path:'/legacy/class',section:'dataset',is_reg:true,managed:false,root:null,origin:'registered',images:3}]}/>);
  expect(screen.getByText('正则集 · 3 张图片')).toBeVisible();
  expect(screen.getByRole('combobox',{name:'用途 1'})).toHaveTextContent('正则集');
  fireEvent.click(screen.getByRole('combobox', {name:'用途 1'}));
  fireEvent.click(screen.getByRole('option', {name:'训练集'}));
  expect(saved().dataset.sources[0].is_reg).toBe(false);
});

it('names dataset source groups independently from sampling prompt groups',()=>{
  render(<Editor initial={{dataset:{sources:[source,{...source,path:'/project/v2/traindata/portraits'}]}}} roles={[]}/>);
  expect(screen.getByRole('group',{name:'数据源 #1'})).toBeInTheDocument();
  expect(screen.getByRole('group',{name:'数据源 #2'})).toBeInTheDocument();
  expect(screen.queryByRole('group',{name:/提示词/})).not.toBeInTheDocument();
});
