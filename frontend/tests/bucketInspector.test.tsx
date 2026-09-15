import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import BucketInspector from '../src/pages/TrainConfig/BucketInspector';
import i18n from '../src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

it('keeps indexed data visible when incomplete sampling settings block the training plan', () => {
  const onIssues = vi.fn(); const onData = vi.fn();
  render(<BucketInspector plan={null} loading={false} hasSources indexed={{images:600,captioned:533}} onIssues={onIssues} onData={onData}/>);
  expect(screen.getByText('600')).toBeInTheDocument();
  expect(screen.getByText('533')).toBeInTheDocument();
  expect(screen.getByText(/已配置数据来源/)).toBeInTheDocument();
  expect(screen.queryByText(/添加训练图片后/)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'检查待配置项'}));
  expect(onIssues).toHaveBeenCalledOnce(); expect(onData).not.toHaveBeenCalled();
});

it('shows the dataset action for a workspace that has no configured sources', () => {
  const onData = vi.fn();
  render(<BucketInspector plan={null} loading={false} onData={onData}/>);
  expect(screen.getByText(/添加训练图片后/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'配置训练数据'}));
  expect(onData).toHaveBeenCalledOnce();
});

it('explains each GPU batch and memory when a distributed plan is selected', () => {
  const plan = {ok:true, errors:[], warnings:[], images:16, items:16, captioned:16, buckets:[{w:64,h:64,items:16,batches:8}], steps_per_epoch:2, total_steps:8, params:{base:1000,trainable:100,adapted_layers:1,training_mode:'adapter' as const}, memory:{weights_mb:0,swapped_mb:0,text_encoder_mb:0,adapter_mb:0,optimizer_mb:0,gradients_mb:0,estimate_scope:'per_device' as const,communication_mb_estimate:0,optimizer_workspace_mb_estimate:0,heuristic:true,peak_mb_estimate:4096}, distributed:{strategy:'ddp' as const,parameter_storage:'replicated' as const,gradient_storage:'replicated' as const,optimizer_storage:'replicated' as const,world_size:2,per_device_batch_size:2,effective_batch_size:8,batches_per_rank:4,dropped_samples:1,tail_policy:'drop_incomplete_rank_group'}};
  render(<BucketInspector plan={plan} loading={false} onData={() => {}}/>);
  expect(screen.getByText('训练显卡').parentElement).toHaveTextContent('2');
  expect(screen.getByText('每卡批量').parentElement).toHaveTextContent('2');
  expect(screen.getByText('有效批量上限').parentElement).toHaveTextContent('8');
  expect(screen.getByText('首轮末尾略过').parentElement).toHaveTextContent('1 张');
  expect(screen.getByText('每卡显存峰值估算')).toBeInTheDocument();
});
