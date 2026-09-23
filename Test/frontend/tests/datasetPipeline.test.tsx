import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DatasetPipelinePanel, { type PipelineSnapshot } from '../../../frontend/src/components/datasets/DatasetPipelinePanel';
import { apiClient } from '../../../frontend/src/api/client';
import '../../../frontend/src/i18n';
vi.mock('../../../frontend/src/events/useEventStream', () => ({useEventStream: () => {}}));
vi.mock('../../../frontend/src/components/datasets/CaptionWorkspace', () => ({default: ({projectId,versionId,readOnly}: {projectId:string;versionId:string;readOnly:boolean}) => <div data-testid="caption-workspace-route">{projectId}/{versionId}/{String(readOnly)}</div>}));
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
  it('shows the selected JSON filename and specific reason without hiding it behind a generic badge', async()=>{
    const message='Invalid JSON caption labels/a.json: unrecognized caption format; use a supported tags/nl structure or select a TXT caption explicitly';
    state.inspection={...state.inspection!,errors:1,images:[{...image,rel_path:'a.png',caption:'',issues:[{severity:'error',code:'caption_format_unsupported',path:'labels/a.json',message}]}]};
    const original=structuredClone(state);
    show();fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
    const summary=await screen.findByText('此 JSON 标签结构不受支持 · labels/a.json');
    fireEvent.click(summary);
    expect(screen.getByText(message)).toBeVisible();
    expect(screen.getByText('请使用 tags / nl 标签结构，或在标签格式中明确选择 TXT。原文件未修改。')).toBeVisible();
    expect(state).toEqual(original);expect(submitted).toEqual([]);
  });
  it('shows Anima field previews without writing captions and navigates to the existing editor', async () => {
    state.inspection = {...state.inspection!, caption_profile:'anima', images:[{
      ...image,rel_path:'a.png',caption:'Some_Artist, Blue_Hair',
      issues:[{severity:'warning',code:'anima_artist_prefix',message:'Anima recommends @ before an artist name'}],
      caption_format:{profile:'anima',suggestions:[
        {path:['fixed','artist'],role:'artist',before:'Some_Artist',after:'@some artist'},
        {path:['ai_output','tags'],role:'tags',before:['Blue_Hair','score_7','MY_Trigger'],after:['blue hair','score_7','MY_Trigger']},
      ]},
    }]};
    const original=structuredClone(state);
    show(); fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
    await screen.findByText('Anima 标签格式建议 · 1 张可预览');
    fireEvent.click(screen.getByText('Anima 标签格式建议 · 1 张可预览'));
    expect(screen.getByText(/不会自动修改文件/)).toBeInTheDocument();
    expect(screen.getByText('@some artist')).toBeInTheDocument();
    expect(screen.getByText(/TXT 含完整句子时/)).toBeInTheDocument();
    expect(screen.getByRole('link',{name:'查看模型作者的标签指南'})).toHaveAttribute('href','https://huggingface.co/circlestone-labs/Anima#prompting');
    expect(screen.getByText('画师名前建议加 @')).toBeInTheDocument();
    expect(state).toEqual(original);
    expect(submitted).toEqual([]);
    fireEvent.click(screen.getByRole('button',{name:'打开标签编辑'}));
    expect(screen.getByTestId('pipeline-location')).toHaveTextContent('data_step=captions');
    expect(submitted).toEqual([]);
  });

  it('does not apply Anima advice to other model families or old reports', async () => {
    show(); fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
    await screen.findByRole('checkbox',{name:'选择 a.png'});
    expect(screen.queryByText(/Anima 标签格式建议/)).not.toBeInTheDocument();
  });

  it('separates transparent pixels from fully opaque alpha channels and only filters on request', async () => {
    state.inspection = {...state.inspection!, transparent_images:2, alpha_images:3, errors:0, warnings:2, images:[
      {...image,rel_path:'semi.png',has_alpha:true,has_transparency:true,issues:[{severity:'warning',code:'transparent_image',message:'Transparent pixels'}]},
      {...image,rel_path:'palette.png',has_alpha:true,has_transparency:true,issues:[{severity:'warning',code:'transparent_image',message:'Transparent pixels'}]},
      {...image,rel_path:'opaque.png',has_alpha:true,has_transparency:false,issues:[]},
      {...image,rel_path:'rgb.png',has_alpha:false,has_transparency:false,issues:[]},
    ]};
    show(); fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
    await screen.findByRole('checkbox',{name:'选择 semi.png'});
    expect(screen.getByText('2 张含透明像素')).toBeInTheDocument();
    expect(screen.getByText('3 张带透明通道')).toBeInTheDocument();
    expect(screen.getAllByText('含透明或半透明像素')).toHaveLength(2);
    expect(screen.getByText(/透明通道全部不透明/)).toBeInTheDocument();
    expect(screen.getByText(/透明区域以白色合成/)).toBeInTheDocument();
    const filter = screen.getByRole('combobox',{name:'显示'});
    fireEvent.click(filter);
    fireEvent.click(screen.getByRole('option',{name:'含透明像素'}));
    expect(screen.getByRole('checkbox',{name:'选择 palette.png'})).toBeInTheDocument();
    expect(screen.queryByRole('checkbox',{name:'选择 opaque.png'})).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox',{name:'选择 rgb.png'})).not.toBeInTheDocument();
    fireEvent.click(filter);
    fireEvent.click(screen.getByRole('option',{name:'带透明通道（含全不透明）'}));
    expect(screen.getByRole('checkbox',{name:'选择 opaque.png'})).toBeInTheDocument();
    expect(screen.queryByRole('checkbox',{name:'选择 rgb.png'})).not.toBeInTheDocument();
    expect(submitted).toEqual([]);
  });

  it('does not label an older inspection as having zero transparent images', async () => {
    show(); fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
    await screen.findByRole('checkbox',{name:'选择 a.png'});
    expect(screen.getByText('这份检查结果尚未包含透明像素检测，重新检查可补充。')).toBeInTheDocument();
    expect(screen.queryByText('0 张含透明像素')).not.toBeInTheDocument();
  });
  it('clears hidden selections when changing filters and excludes only visible results', async () => {
    state.inspection = {...state.inspection!, transparent_images:1, alpha_images:2, images:[
      {...image,rel_path:'subtle.png',has_alpha:true,has_transparency:true,transparent_pixels:1,min_alpha:254,issues:[]},
      {...image,rel_path:'opaque.png',has_alpha:true,has_transparency:false,issues:[]},
    ]};
    show(); fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
    fireEvent.click(await screen.findByRole('checkbox',{name:'选择 opaque.png'}));
    expect(screen.getByText(/1 个透明或半透明像素/)).toHaveTextContent('最低不透明度 99.6%');
    fireEvent.click(screen.getByRole('combobox',{name:'显示'}));
    fireEvent.click(screen.getByRole('option',{name:'含透明像素'}));
    expect(screen.getByRole('button',{name:'排除 0 张选中图片'})).toBeDisabled();
    fireEvent.click(screen.getByRole('button',{name:'选择当前筛选结果'}));
    fireEvent.click(screen.getByRole('button',{name:'排除 1 张选中图片'}));
    await waitFor(() => expect(submitted[0].body.images).toEqual([{dataset_id:'d_1',rel_path:'subtle.png'}]));
  });
  it('persists the selected stage in URL and per-version memory with browser back support', async()=>{
    const first=show();
    fireEvent.click(screen.getByRole('button',{name:/涂抹与遮罩/}));
    expect(screen.getByTestId('pipeline-location')).toHaveTextContent('data_step=paint');
    fireEvent.click(screen.getByRole('button',{name:/标签编辑/}));
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
    expect(screen.queryByTestId('existing-caption')).not.toBeInTheDocument();
    expect(screen.queryByText('暂无标签')).not.toBeInTheDocument();
    expect(submitted).toEqual([]);
    expect(apiClient.get).toHaveBeenCalledWith('/projects/p_1/datasets',expect.objectContaining({params:{version_id:'v_2'}}));
  });
  it('opens the dedicated caption workspace without inspection and preserves operation recovery', async () => {
    state.inspection=null;
    state.operations=[{...operation,action:'tag'},{...operation,id:'dp_old_tag',action:'tag',status:'failed',can_undo:false,error:'old tagging failed'},{...operation,id:'dp_failed',status:'failed',can_undo:false,error:'disk is full',result:{rolled_back:true}}];
    show(); fireEvent.click(screen.getByRole('button',{name:/标签编辑/}));
    expect(await screen.findByTestId('caption-workspace-route')).toHaveTextContent('p_1/v_2/false');
    expect(screen.queryByRole('button',{name:'检查数据'})).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox',{name:'标签文本'})).not.toBeInTheDocument();
    expect(screen.queryByRole('form',{name:/WD14/})).not.toBeInTheDocument();
    expect(screen.queryByRole('region',{name:'涂抹与遮罩'})).not.toBeInTheDocument();
    expect(submitted).toEqual([]);
    fireEvent.click(await screen.findByText('操作记录'));
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
    show(); fireEvent.click(screen.getByRole('button',{name:/训练缓存/}));
    expect(await screen.findByText('编码与缓存')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow','28');
    expect(screen.getByRole('button',{name:'提前生成缓存（可选）'})).toBeDisabled();
    expect(screen.getByRole('link',{name:'训练参数'})).toHaveAttribute('href','/projects/p_1/v/v_2/train');
    expect(screen.getByRole('link',{name:'选择训练模型'})).toHaveAttribute('href','/projects/p_1/v/v_2/train?tab=model');
    fireEvent.click(screen.getByRole('button',{name:'取消'}));
    await waitFor(() => expect(submitted[0]?.url).toBe('/dataset-pipeline/operations/dp_1/cancel'));
  });
  it('makes legacy copy discoverable and keeps archived mutations disabled', async () => {
    state.inspection!.images.forEach(item => item.editable=false);
    show({readOnly:true}); fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
    await screen.findByText(/外部或旧版素材需先复制/);
    expect(screen.getByRole('button',{name:'检查数据'})).toBeDisabled();
    expect(screen.getByRole('checkbox',{name:'选择 a.png'})).toBeDisabled();
    expect(screen.getByRole('button',{name:'复制为可处理的新版本'})).toBeDisabled();
  });
});

it('names failed painting and returns to its editor instead of posting an unsupported retry', async()=>{
  state.operations=[{...operation,action:'paint',status:'failed',can_undo:false,error:'disk full',result:{rolled_back:true}}];
  show();
  await screen.findByText('操作记录');
  expect(screen.getByText('操作记录').closest('details')).not.toHaveAttribute('open');
  fireEvent.click(screen.getByText('操作记录'));
  expect(screen.getByText('图像涂抹与遮罩')).toBeInTheDocument();
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
  expect(screen.getByRole('region',{name:'图像涂抹与遮罩'})).toHaveTextContent('准备绘制文件');
});

it('keeps inspection scope and operation history without repeating read-only notices',async()=>{
  state.operations=[{...operation,action:'inspect',can_undo:false,result:{}}];
  show();fireEvent.click(screen.getByRole('button',{name:/检查与筛选/}));
  await screen.findByText('检查哪些内容');
  fireEvent.click(screen.getByText('检查哪些内容'));
  expect(screen.getByText(/按文件内容识别完全相同/)).toBeVisible();
  await screen.findByText('操作记录');
  const footer=screen.getByTestId('dataset-pipeline').lastElementChild;
  expect(footer?.tagName).toBe('FOOTER');
  fireEvent.click(screen.getByText('操作记录'));
  expect(screen.queryByText('只读取数据，未修改文件。')).not.toBeInTheDocument();
  expect(screen.getByText('数据检查')).toBeVisible();
  expect(screen.queryByRole('button',{name:'恢复此操作前的文件'})).not.toBeInTheDocument();
});
it('does not invent a percentage before an operation knows its total',async()=>{
  state.operations=[{...operation,action:'inspect',status:'running',phase:'inspecting',done:0,total:0,can_undo:false}];
  show();
  expect(await screen.findByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
});
