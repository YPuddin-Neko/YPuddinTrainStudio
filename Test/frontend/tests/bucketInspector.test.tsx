import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import BucketInspector from '../../../frontend/src/pages/TrainConfig/BucketInspector';
import i18n from '../../../frontend/src/i18n';
import type {Plan} from '../../../frontend/src/api/types';

const balancePlan=(repeats=2)=>({ok:true,errors:[],warnings:[],source_balance:[
  {source_index:0,path:'D:/训练数据/角色',is_reg:false,images:3,repeats,repeated_images:3*repeats,resolution_variants:1,items:3*repeats},
  {source_index:1,path:'D:/训练数据/正则',is_reg:true,images:2,repeats:1,repeated_images:2,resolution_variants:1,items:2},
]} as unknown as Plan);

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

it('shows native training results without embedding parameter editors in the inspector',()=>{
  const plan={ok:true,errors:[],warnings:[],buckets:[{base:0,w:864,h:1200,items:133,batches:133}],native:{max_pixels:1048576,downscaled:152}} as unknown as Plan;
  render(<BucketInspector plan={plan} loading={false} onData={()=>{}} dataset={{resolution_mode:'native'}}/>);
  expect(screen.getByRole('heading',{name:'实际训练尺寸'})).toBeVisible();
  expect(screen.getByRole('button',{name:'864 × 1200, 133 样本'})).toBeVisible();
  expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument();
});

it('shows real source proportions, repeat formulas and a separate regularization group',()=>{
  const {rerender}=render(<BucketInspector plan={balancePlan()} loading={false} hasSources onData={()=>{}}/>);
  const train=screen.getByRole('button',{name:'角色 · 3 张 × 2 次 = 6 项 · 75.0%'});
  expect(screen.getByRole('button',{name:'正则 · 2 张 × 1 次 = 2 项 · 25.0%'})).toBeVisible();
  expect(screen.getByRole('heading',{name:/正则集\s*25\.0%/})).toBeVisible();
  fireEvent.focus(train);
  expect(screen.getByRole('tooltip')).toHaveTextContent('D:/训练数据/角色');
  expect(screen.getByRole('tooltip')).toHaveTextContent('3 张 × 2 次 = 6 项 · 75.0%');
  rerender(<BucketInspector plan={balancePlan(6)} loading={false} hasSources onData={()=>{}}/>);
  expect(screen.getByRole('button',{name:'角色 · 3 张 × 6 次 = 18 项 · 90.0%'})).toBeVisible();
  expect(screen.getByRole('img')).toHaveAccessibleName('角色 90.0% · 正则 10.0%');
});

it('keeps unknown and pending counts distinct from zero and labels old results during recalculation',()=>{
  const {rerender}=render(<BucketInspector plan={null} loading hasSources onData={()=>{}}/>);
  expect(screen.getByRole('status')).toHaveTextContent('正在计算数据分布');
  expect(screen.queryByText('0 项 / 轮')).not.toBeInTheDocument();
  rerender(<BucketInspector plan={balancePlan()} loading hasSources onData={()=>{}}/>);
  expect(screen.getByRole('status')).toHaveTextContent('正在更新数据分布，以下为上次结果');
  expect(screen.getByRole('button',{name:'角色 · 3 张 × 2 次 = 6 项 · 75.0%'})).toBeVisible();
  rerender(<BucketInspector plan={{...balancePlan(),source_balance:null}} loading={false} hasSources onData={()=>{}}/>);
  expect(screen.getByText(/配平统计尚不可用/)).toBeVisible();
  expect(screen.queryByText('0 项 / 轮')).not.toBeInTheDocument();
});

it('exposes a failed recalculation and retries without pretending the last result is current',()=>{
  const retry=vi.fn();
  render(<BucketInspector plan={balancePlan()} loading={false} error="Request timed out" onRetry={retry} onData={()=>{}}/>);
  expect(screen.getByRole('alert')).toHaveTextContent('数据分布计算失败');
  expect(screen.getByRole('alert')).toHaveTextContent('以下为上次计算结果');
  fireEvent.click(screen.getByRole('button',{name:'重新计算'}));
  expect(retry).toHaveBeenCalledOnce();
});

it('keeps indexed data visible when incomplete sampling settings block the training plan', () => {
  const onIssues = vi.fn(); const onData = vi.fn();
  render(<BucketInspector plan={null} loading={false} hasSources indexed={{images:600,captioned:533}} onIssues={onIssues} onData={onData}/>);
  expect(screen.getByText('600')).toBeInTheDocument();
  expect(screen.getByText('533')).toBeInTheDocument();
  expect(screen.getByText('请完成待配置项后计算。')).toBeInTheDocument();
  expect(screen.queryByText('尚无训练图片。')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'检查待配置项'}));
  expect(onIssues).toHaveBeenCalledOnce(); expect(onData).not.toHaveBeenCalled();
});

it('shows the dataset action for a workspace that has no configured sources', () => {
  const onData = vi.fn();
  render(<BucketInspector plan={null} loading={false} onData={onData}/>);
  expect(screen.getByText('尚无训练图片。')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'配置训练数据'}));
  expect(onData).toHaveBeenCalledOnce();
});

it('explains each GPU batch and memory when a distributed plan is selected', () => {
  const plan = {ok:true, errors:[], warnings:[], images:16, items:16, captioned:16, buckets:[{base: 64,w:64,h:64,items:16,batches:8}], steps_per_epoch:2, total_steps:8, params:{base:1000,trainable:100,adapted_layers:1,training_mode:'adapter' as const}, memory:{weights_mb:0,swapped_mb:0,text_encoder_mb:0,adapter_mb:0,optimizer_mb:0,gradients_mb:0,estimate_scope:'per_device' as const,communication_mb_estimate:0,optimizer_workspace_mb_estimate:0,heuristic:true,peak_mb_estimate:4096}, distributed:{strategy:'ddp' as const,parameter_storage:'replicated' as const,gradient_storage:'replicated' as const,optimizer_storage:'replicated' as const,world_size:2,per_device_batch_size:2,effective_batch_size:8,batches_per_rank:4,dropped_samples:1,tail_policy:'drop_incomplete_rank_group'}};
  render(<BucketInspector plan={plan} loading={false} onData={() => {}}/>);
  expect(screen.getByText('训练显卡').parentElement).toHaveTextContent('2');
  expect(screen.getByText('每卡批量').parentElement).toHaveTextContent('2');
  expect(screen.getByText('有效批量上限').parentElement).toHaveTextContent('8');
  expect(screen.getByText('首轮末尾略过').parentElement).toHaveTextContent('1 张');
  expect(screen.getByText('每卡显存峰值估算')).toBeInTheDocument();
});

it('keeps equal bucket sizes from different base resolutions in separately labelled groups', () => {
  const plan = {ok:true,errors:[],warnings:[],buckets:[{base:1024,w:1024,h:1024,items:2,batches:1},{base:1536,w:1024,h:1024,items:3,batches:2}]} as unknown as Plan;
  render(<BucketInspector plan={plan} loading={false} onData={()=>{}}/>);
  expect(screen.getByRole('region',{name:'分辨率 1024'})).toHaveTextContent('2 样本');
  expect(screen.getByRole('region',{name:'分辨率 1536'})).toHaveTextContent('3 样本');
  fireEvent.click(screen.getByRole('button',{name:'分辨率 1536 · 1024 × 1024, 3 样本'}));
  expect(screen.getByText('分辨率 1536 · 1024 × 1024')).toBeVisible();
  fireEvent.click(screen.getByRole('button',{name:'分桶明细表'}));
  expect(screen.getByRole('columnheader',{name:'分辨率'})).toBeVisible();
  expect(screen.getByRole('cell',{name:'1536'})).toBeVisible();
});
