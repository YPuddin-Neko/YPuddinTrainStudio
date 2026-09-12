import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DatasetPipelinePanel, { type PipelineSnapshot } from '../src/components/datasets/DatasetPipelinePanel';
import { apiClient } from '../src/api/client';
import '../src/i18n';
vi.mock('../src/events/useEventStream', () => ({useEventStream: () => {}}));
let state: PipelineSnapshot;
let submitted: {url:string; body:any}[];
const image = {dataset_id:'d_1',path:'/data/a.png',hash:'abc',width:640,height:480,caption:'portrait',has_mask:true,roles:['train'],issues:[{severity:'warning' as const,code:'duplicate',message:'duplicate'}],editable:true};
const operation = {id:'dp_1',action:'exclude',status:'completed',phase:'completed',done:1,total:1,error:null,job_id:null,can_undo:true,can_cancel:false,result:{changed_files:3},logs:[{time:1,message:'Backed up a.png'}]};
function RouteProbe(){const location=useLocation();const navigate=useNavigate();return <><output data-testid="pipeline-location">{location.search}</output><button onClick={()=>navigate(-1)}>Browser back</button></>;}
function show(extra={}) {
  return render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter><RouteProbe/><DatasetPipelinePanel projectId="p_1" versionId="v_2" config={{dataset:{resolution_mode:'native'}}} importPanel={<div>Upload files</div>} datasetList={<div>Dataset links</div>} onChanged={vi.fn()} {...extra}/></MemoryRouter></QueryClientProvider>);
}
beforeEach(() => {
  sessionStorage.clear();
  submitted=[];
  state={signature:'one',inspection:{images:[{...image,rel_path:'a.png'},{...image,rel_path:'b.png',caption:'',hash:'def'}],duplicate_groups:[[0,1]],errors:0,warnings:3,captioned:1,masks:2,source_issues:[]},plan:null,operations:[],busy:false,archived:false,stale:false,ready_to_train:false,prepared_job_id:null};
  vi.spyOn(apiClient,'get').mockImplementation(async url => {
    if(url==='/projects/p_1/versions/v_2/regularization')return {path:'/project/p_1/v2/reg',images:0,operations:[]} as any;
    if(url==='/projects/p_1/datasets') return [{source:{id:'d_1',path:'/data/images'},index_status:'ready'}] as any;
    if(url==='/datasets/d_1/images') return {items:[{...image,rel_path:'a.png'},{...image,rel_path:'b.png',caption:'',hash:'def'}],page:1,page_size:40,total:2} as any;
    return structuredClone(state) as any;
  });
  vi.spyOn(apiClient,'post').mockImplementation(async (url,body) => {submitted.push({url,body});return operation as any;});
});
afterEach(() => vi.restoreAllMocks());
describe('dataset pipeline', () => {
  it('persists the selected stage in URL and per-version memory with browser back support', async()=>{
    const first=show();
    fireEvent.click(screen.getByRole('button',{name:/涂抹与遮罩/}));
    expect(screen.getByTestId('pipeline-location')).toHaveTextContent('data_step=paint');
    fireEvent.click(screen.getByRole('button',{name:/标签查看/}));
    fireEvent.click(screen.getByRole('button',{name:'Browser back'}));
    expect(screen.getByRole('button',{name:/涂抹与遮罩/})).toHaveAttribute('aria-current','step');
    first.unmount();
    const second=show();
    expect(screen.getByRole('button',{name:/涂抹与遮罩/})).toHaveAttribute('aria-current','step');
    second.unmount();
    show({versionId:'v_3'});
    expect(screen.getByText('Upload files')).toBeInTheDocument();
  });
  it('opens the version-scoped regularization step and preserves its URL',async()=>{
    show();fireEvent.click(screen.getByRole('button',{name:/正则图/}));
    expect(screen.getByTestId('pipeline-location')).toHaveTextContent('data_step=reg');
    await waitFor(()=>expect(apiClient.get).toHaveBeenCalledWith('/projects/p_1/versions/v_2/regularization',expect.anything()));
    expect(submitted).toEqual([]);
  });
  it('selects duplicate copies while keeping one and sends version-scoped exclusion', async () => {
    // Existing reports can put the uncaptioned download first. The richer copy wins.
    state.inspection!.duplicate_groups = [[1,0]];
    show(); fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
    await screen.findByRole('checkbox',{name:'选择 a.png'});
    fireEvent.click(screen.getByRole('button',{name:'选择重复副本（每组保留一张）'}));
    expect(screen.getByRole('checkbox',{name:'选择 a.png'})).not.toBeChecked();
    expect(screen.getByRole('checkbox',{name:'选择 b.png'})).toBeChecked();
    fireEvent.click(screen.getByRole('button',{name:'排除 1 张选中图片'}));
    await waitFor(() => expect(submitted[0]).toEqual({url:'/projects/p_1/versions/v_2/pipeline/operations',body:{action:'exclude',images:[{dataset_id:'d_1',rel_path:'b.png'}]}}));
  });
  it('opens paint and masks without a prior inspection or crop controls', async () => {
    state.inspection=null;
    show();fireEvent.click(screen.getByRole('button',{name:/涂抹与遮罩/}));
    await screen.findByRole('button',{name:'打开涂抹与遮罩编辑器'});
    expect(screen.queryByRole('button',{name:'检查数据'})).not.toBeInTheDocument();
    expect(screen.queryByRole('combobox',{name:'处理方式'})).not.toBeInTheDocument();
    expect(submitted).toEqual([]);
    expect(apiClient.get).toHaveBeenCalledWith('/projects/p_1/datasets',expect.objectContaining({params:{version_id:'v_2'}}));
  });
  it('views existing captions without an inspection or mutation and preserves recovery of old operations', async () => {
    state.inspection=null;
    state.operations=[{...operation,action:'tag'},{...operation,id:'dp_old_tag',action:'tag',status:'failed',can_undo:false,error:'old tagging failed'},{...operation,id:'dp_failed',status:'failed',can_undo:false,error:'disk is full',result:{rolled_back:true}}];
    show(); fireEvent.click(screen.getByRole('button',{name:/标签查看/}));
    expect(await screen.findByTestId('existing-caption')).toHaveTextContent('portrait');
    expect(screen.queryByRole('button',{name:'检查数据'})).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox',{name:'标签文本'})).not.toBeInTheDocument();
    expect(screen.queryByRole('form',{name:/WD14/})).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'查看标签: b.png'}));
    expect(screen.getByTestId('existing-caption')).toHaveTextContent('此图片暂无标签');
    expect(submitted).toEqual([]);
    expect(apiClient.get).toHaveBeenCalledWith('/projects/p_1/datasets',expect.objectContaining({params:{version_id:'v_2'}}));
    fireEvent.click(screen.getByRole('button',{name:'恢复此操作前的文件'}));
    await waitFor(() => expect(submitted[0]?.body).toEqual({action:'restore',restore_operation_id:'dp_1'}));
    await waitFor(() => expect(screen.getByRole('button',{name:'重试'})).not.toBeDisabled());
    fireEvent.click(screen.getByRole('button',{name:'重试'}));
    await waitFor(() => expect(submitted[1]?.url).toBe('/dataset-pipeline/operations/dp_failed/retry'));
  });
  it('maps the remembered preprocessing stage to the paint workspace',async()=>{
    sessionStorage.setItem('studio.pipeline.stage.p_1.v_2','preprocess');
    show();
    expect(screen.getByRole('button',{name:/涂抹与遮罩/})).toHaveAttribute('aria-current','step');
    expect(await screen.findByRole('button',{name:'打开涂抹与遮罩编辑器'})).toBeEnabled();
    expect(screen.queryByRole('button',{name:'单图可视裁剪'})).not.toBeInTheDocument();
    expect(sessionStorage.getItem('studio.pipeline.stage.p_1.v_2')).toBe('paint');
    expect(submitted).toEqual([]);
  });
  it('shows cache progress and cancellation while keeping direct training available', async () => {
    state.operations=[{...operation,action:'prepare',status:'running',phase:'cache',done:2,total:7,can_undo:false,can_cancel:true,job_id:'j_cache'}];
    show(); fireEvent.click(screen.getByRole('button',{name:/训练准备/}));
    expect(await screen.findByText('编码与缓存')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('value','2');
    expect(screen.getByRole('button',{name:'检查并构建训练缓存'})).toBeDisabled();
    expect(screen.getByRole('link',{name:'进入训练参数'})).toHaveAttribute('href','/projects/p_1/v/v_2/train');
    expect(screen.getByRole('link',{name:'选择训练模型'})).toHaveAttribute('href','/projects/p_1/v/v_2/train?tab=model');
    fireEvent.click(screen.getByRole('button',{name:'取消'}));
    await waitFor(() => expect(submitted[0]?.url).toBe('/dataset-pipeline/operations/dp_1/cancel'));
  });
  it('makes legacy copy discoverable and keeps archived mutations disabled', async () => {
    state.inspection!.images.forEach(item => item.editable=false);
    show({readOnly:true}); fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
    await screen.findByText(/旧版或外部引用素材/);
    expect(screen.getByRole('button',{name:'检查数据'})).toBeDisabled();
    expect(screen.getByRole('checkbox',{name:'选择 a.png'})).toBeDisabled();
    expect(screen.getByRole('button',{name:'复制为可处理的新版本'})).toBeDisabled();
  });
});

it('names failed painting and returns to its editor instead of posting an unsupported retry', async()=>{
  state.operations=[{...operation,action:'paint',status:'failed',can_undo:false,error:'disk full',result:{rolled_back:true}}];
  show();
  expect(await screen.findByText('图像涂抹与遮罩')).toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'重试'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'返回涂抹与遮罩'}));
  expect(screen.getByTestId('pipeline-location')).toHaveTextContent('data_step=paint');
  expect(await screen.findByRole('button',{name:'打开涂抹与遮罩编辑器'})).toBeEnabled();
  expect(apiClient.get).toHaveBeenCalledWith('/projects/p_1/datasets',expect.objectContaining({params:{version_id:'v_2'}}));
  expect(submitted).toEqual([]);
});

it('shows the real paint staging phase in readable language',async()=>{
  state.operations=[{...operation,action:'paint',status:'running',phase:'staging',can_undo:false}];
  show();
  expect(await screen.findByText('准备绘制文件')).toBeInTheDocument();
  expect(screen.getByText('准备绘制文件').closest('[role="status"]')).toHaveTextContent('图像涂抹与遮罩');
});
