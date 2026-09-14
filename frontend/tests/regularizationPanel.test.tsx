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
let keyConfigured = true;
beforeEach(async()=>{
  await i18n.changeLanguage('zh-CN');
  snapshot={path:'D:/studio/project/dogs/v1/reg',images:0,operations:[]};
  keyConfigured = true;
  vi.spyOn(apiClient,'get').mockImplementation(async endpoint=>endpoint === '/credentials' ? {huggingface:{configured:false},modelscope:{configured:false},danbooru:{configured:keyConfigured},gelbooru:{configured:keyConfigured}} as any : snapshot as any);
});
afterEach(()=>vi.restoreAllMocks());
function show(readOnly=false) {
  const changed=vi.fn();
  const client=new QueryClient({defaultOptions:{queries:{retry:false,gcTime:0}}});
  render(<MemoryRouter><QueryClientProvider client={client}><RegularizationPanel projectId="dogs" versionId="v1" readOnly={readOnly} onChanged={changed}/></QueryClientProvider></MemoryRouter>);
  return {client,changed};
}

describe('regularization preparation',()=>{
  it('previews training-tag exclusions and selected scope before starting the exact checked plan',async()=>{
    const pending: {body:any;resolve:(value:any)=>void}[]=[];
    const preview={signature:'a'.repeat(64),sources:[{id:'d_first',name:'first',path:'/train/first'},{id:'d_second',name:'second',path:'/train/second'}],top_tags:[{tag:'subject_trigger',count:2},{tag:'dog',count:1}],source_images:3,existing_images:0,missing_captions:1,invalid_captions:0,empty_after_exclusion:0,eligible_images:2,planned_images:2,remaining_images:0,max_batch_images:200,examples:[{source_id:'d_first',rel_path:'same.png',prompt:'subject_trigger, dog'}]};
    vi.spyOn(apiClient,'post').mockImplementation((url,body)=>{
      if(url.endsWith('/plan'))return new Promise(resolve=>pending.push({body,resolve}));
      snapshot={...snapshot,operations:[task()]};return Promise.resolve(task() as any);
    });
    show();
    fireEvent.click(await screen.findByRole('combobox',{name:'提示词来源'}));
    fireEvent.click(screen.getByRole('option',{name:'按训练图片标签逐张生成'}));
    await waitFor(()=>expect(pending).toHaveLength(1));
    expect(screen.getByRole('button',{name:'生成正则图'})).toBeDisabled();
    expect(pending[0].body.prompt_source).toBe('training_tags');
    await act(async()=>pending[0].resolve(preview));
    const exclude=await screen.findByRole('button',{name:'排除标签：subject_trigger，2 张图片'});
    fireEvent.click(exclude);
    expect(exclude).toHaveAttribute('aria-pressed','true');
    expect(screen.getByRole('button',{name:'生成正则图'})).toBeDisabled();
    fireEvent.click(screen.getByRole('combobox',{name:'训练目录范围'}));
    fireEvent.click(screen.getByRole('option',{name:'first · /train/first'}));
    fireEvent.click(screen.getByRole('combobox',{name:'生成范围'}));
    fireEvent.click(screen.getByRole('option',{name:'全部：另建一个批次'}));
    await waitFor(()=>expect(pending.length).toBeGreaterThan(1));
    await waitFor(()=>expect(pending.at(-1)?.body).toMatchObject({excluded_tags:['subject_trigger'],source_ids:['d_first'],generation_scope:'all'}));
    await act(async()=>pending.at(-1)!.resolve({...preview,signature:'b'.repeat(64),planned_images:1,eligible_images:1,examples:[{source_id:'d_first',rel_path:'same.png',prompt:'dog'}]}));
    await waitFor(()=>expect(screen.getByRole('button',{name:'生成正则图'})).toBeEnabled());
    expect(screen.getByText(/原图片与标签保持原样/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'生成正则图'}));
    await waitFor(()=>expect(apiClient.post).toHaveBeenCalledWith('/projects/dogs/versions/v1/regularization',expect.objectContaining({prompt_source:'training_tags',excluded_tags:['subject_trigger'],source_ids:['d_first'],generation_scope:'all',plan_signature:'b'.repeat(64)}),{silent:true}));
  });
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

  it('uses stored site credentials without putting keys in the form or task body',async()=>{
    vi.spyOn(apiClient,'post').mockResolvedValue(task({source:'gelbooru'}) as any);
    show(); await screen.findByRole('button',{name:'生成正则图'});
    fireEvent.click(screen.getByRole('combobox',{name:'图片来源'}));
    fireEvent.click(screen.getByRole('option',{name:'Gelbooru'}));
    await screen.findByText('gelbooru 访问密钥已配置');
    fireEvent.change(screen.getByRole('textbox',{name:'站点检索标签'}),{target:{value:'dog solo'}});
    expect(screen.queryByLabelText('API Key')).not.toBeInTheDocument();
    expect(document.querySelector('input[type="password"]')).toBeNull();
    expect(screen.getByRole('link',{name:'管理访问密钥'})).toHaveAttribute('href','/settings/environment?tab=credentials#credentials-gelbooru');
    fireEvent.click(screen.getByRole('button',{name:'收集正则图'}));
    await waitFor(()=>expect(apiClient.post).toHaveBeenCalledWith(expect.any(String),expect.objectContaining({source:'gelbooru',prompt:'dog solo'}),{silent:true}));
    const body=vi.mocked(apiClient.post).mock.calls[0][1] as Record<string,unknown>;
    expect(body).not.toHaveProperty('api_key'); expect(body).not.toHaveProperty('username'); expect(body).not.toHaveProperty('user_id');
  });

  it('requires Gelbooru setup and refreshes configured status after saving in Settings',async()=>{
    keyConfigured=false; show();
    await screen.findByRole('button',{name:'生成正则图'});
    fireEvent.click(screen.getByRole('combobox',{name:'图片来源'}));
    fireEvent.click(screen.getByRole('option',{name:'Gelbooru'}));
    await screen.findByText('Gelbooru 需要先配置用户 ID 和 API Key。');
    fireEvent.change(screen.getByRole('textbox',{name:'站点检索标签'}),{target:{value:'dog'}});
    expect(screen.getByRole('button',{name:'收集正则图'})).toBeDisabled();
    keyConfigured=true;
    await act(async()=>{window.dispatchEvent(new Event('credentials.changed'));});
    await screen.findByText('gelbooru 访问密钥已配置');
    expect(screen.getByRole('button',{name:'收集正则图'})).toBeEnabled();
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
