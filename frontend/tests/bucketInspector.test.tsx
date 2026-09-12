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
