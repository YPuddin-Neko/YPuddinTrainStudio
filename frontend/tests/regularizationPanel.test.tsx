import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import RegularizationPanel, { type RegularizationTask } from '../src/components/datasets/RegularizationPanel';
import { apiClient } from '../src/api/client';
import i18n from '../src/i18n';

vi.mock('../src/pages/ProjectDetail/ProjectDataImport',()=>({default:({defaultIsReg}:{defaultIsReg:boolean})=><div>{defaultIsReg?'Import regularization data':'Import training data'}</div>}));
const task = (extra: Partial<RegularizationTask> = {}): RegularizationTask => ({id:'reg_123',source:'ai',status:'running',phase:'generating',done:1,total:2,logs:[],error:null,created_at:1,can_cancel:true,dataset_id:null,path:null,images:0,duplicates:0,...extra});
let snapshot: {path:string;images:number;operations:RegularizationTask[]};
beforeEach(async()=>{
  await i18n.changeLanguage('zh-CN');
  snapshot={path:'D:/studio/project/dogs/v1/reg',images:0,operations:[]};
  vi.spyOn(apiClient,'get').mockImplementation(async()=>snapshot as any);
});
afterEach(()=>vi.restoreAllMocks());
function show(readOnly=false) {
  const changed=vi.fn();
  const client=new QueryClient({defaultOptions:{queries:{retry:false,gcTime:0}}});
  render(<MemoryRouter><QueryClientProvider client={client}><RegularizationPanel projectId="dogs" versionId="v1" readOnly={readOnly} onChanged={changed}/></QueryClientProvider></MemoryRouter>);
  return {client,changed};
}

describe('regularization preparation',()=>{
  it('starts explicit class generation and notifies the dataset owner only after publication',async()=>{
    vi.spyOn(apiClient,'post').mockImplementation(async()=>{snapshot={...snapshot,operations:[task()]};return task() as any;});
    const {client,changed}=show();
    await screen.findByRole('button',{name:'生成正则图'});
    expect(screen.getByRole('button',{name:'生成正则图'})).toBeDisabled();
    fireEvent.change(screen.getByRole('textbox',{name:'类别提示词'}),{target:{value:'a photo of a dog'}});
    fireEvent.change(screen.getByRole('spinbutton',{name:'图片数量'}),{target:{value:'2'}});
    fireEvent.click(screen.getByRole('button',{name:'生成正则图'}));
    await screen.findByRole('progressbar',{name:'正则图准备进度'});
    expect(apiClient.post).toHaveBeenCalledWith('/projects/dogs/versions/v1/regularization',expect.objectContaining({source:'ai',prompt:'a photo of a dog',count:2,prior_weight:1,repeats:1}),{silent:true});
    expect(changed).not.toHaveBeenCalled();
    snapshot={...snapshot,images:2,operations:[task({status:'completed',done:2,images:2,dataset_id:'d_reg',can_cancel:false})]};
    await act(async()=>{await client.invalidateQueries({queryKey:['regularization','dogs','v1']});});
    expect(await screen.findByText('新增 2 张图片')).toBeInTheDocument();
    expect(changed).toHaveBeenCalledOnce();
    expect(screen.getByRole('link',{name:'查看图片与标签'})).toHaveAttribute('href',expect.stringContaining('data_step=captions'));
    await act(async()=>{await client.invalidateQueries({queryKey:['regularization','dogs','v1']});});
    expect(changed).toHaveBeenCalledOnce();
  });

  it('sends a site query and one-use credentials, then clears the secret',async()=>{
    vi.spyOn(apiClient,'post').mockResolvedValue(task({source:'gelbooru'}) as any);
    show(); await screen.findByRole('button',{name:'生成正则图'});
    fireEvent.click(screen.getByRole('combobox',{name:'图片来源'}));
    fireEvent.click(screen.getByRole('option',{name:'Gelbooru'}));
    fireEvent.change(screen.getByRole('textbox',{name:'站点检索标签'}),{target:{value:'dog solo'}});
    fireEvent.click(screen.getByText('站点凭据（可选）'));
    fireEvent.change(screen.getByRole('textbox',{name:'用户名或用户 ID'}),{target:{value:'123'}});
    const secret=screen.getByLabelText('API Key');
    fireEvent.change(secret,{target:{value:'test-registry-key'}});
    fireEvent.click(screen.getByRole('button',{name:'收集正则图'}));
    await waitFor(()=>expect(apiClient.post).toHaveBeenCalledWith(expect.any(String),expect.objectContaining({source:'gelbooru',prompt:'dog solo',username:'123',api_key:'test-registry-key'}),{silent:true}));
    await waitFor(()=>expect(secret).toHaveValue(''));
    expect(screen.queryByText('test-registry-key')).not.toBeInTheDocument();
  });

  it('shows cancellation and a failed preparation without claiming images were registered',async()=>{
    snapshot.operations=[task()];
    vi.spyOn(apiClient,'post').mockImplementation(async()=>{snapshot={...snapshot,operations:[task({status:'failed',can_cancel:false,error:'Base model weights are missing'})]};return {} as any;});
    show();
    fireEvent.click(await screen.findByRole('button',{name:'取消 reg_123'}));
    await screen.findByText('Base model weights are missing');
    expect(apiClient.post).toHaveBeenCalledWith('/regularization/reg_123/cancel',{}, {silent:true});
    expect(screen.queryByRole('link',{name:'查看图片与标签'})).not.toBeInTheDocument();
    expect(screen.queryByText('新增 0 张图片')).not.toBeInTheDocument();
  });

  it('permits archived data viewing but prevents generation, cancellation and import',async()=>{
    snapshot.images=12; snapshot.operations=[task({status:'completed',images:12,dataset_id:'d_reg',can_cancel:false})];
    show(true);
    expect(await screen.findByText('已有 12 张')).toBeInTheDocument();
    expect(screen.getByRole('button',{name:'生成正则图'})).toBeDisabled();
    expect(screen.queryByText('导入已有正则图')).not.toBeInTheDocument();
    expect(screen.getByRole('link',{name:'查看图片与标签'})).toBeInTheDocument();
  });
});
