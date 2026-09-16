import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import GpuDevicePicker from '../src/components/GpuDevicePicker';
import { apiClient } from '../src/api/client';
import { gpuSelectionValid } from '../src/utils/gpuDevices';
import i18n from '../src/i18n';

const devices = [{ device: 'cuda:0', name: 'Hygon DCU', job_id: 'train', job_name: '正在训练', status: 'running', mem_free_mb: 1024 }, { device: 'cuda:1', name: 'Hygon DCU', job_id: null, job_name: null, status: 'free', mem_free_mb: 64000 }];
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); vi.spyOn(apiClient, 'get').mockResolvedValue({ devices, max_concurrent: null }); });
afterEach(() => vi.restoreAllMocks());
function Harness({ count = 1 }: { count?: number }) {
  const [selected, setSelected] = React.useState<string[]>([]);
  const [valid, setValid] = React.useState(true);
  return <><GpuDevicePicker value={selected} onChange={setSelected} count={count} onValidityChange={setValid}/><output data-testid="selected">{JSON.stringify(selected)}</output><button disabled={!valid}>提交</button></>;
}
function choose(label: RegExp) { fireEvent.click(screen.getByRole('combobox', { name: '运行显卡' })); fireEvent.click(screen.getByRole('option', { name: label })); }
it('selects the explicit free GPU without translating its logical service id', async () => {
  render(<Harness/>); await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  choose(/GPU 1.*空闲/);
  expect(screen.getByTestId('selected')).toHaveTextContent('["cuda:1"]');
  expect(screen.getByRole('button', { name: '提交' })).toBeEnabled();
  choose(/自动选择空闲显卡/); expect(screen.getByTestId('selected')).toHaveTextContent('[]');
});
it('allows requesting a busy card and clearly explains queued execution', async () => {
  render(<Harness/>); await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  choose(/GPU 0.*占用中/);
  expect(screen.getByText('所选显卡正在使用，任务会等待它空闲后启动。')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: '提交' })).toBeEnabled();
});
it('requires exactly the configured multi-GPU count while allowing automatic selection', async () => {
  render(<Harness count={2}/>); const gpu0 = await screen.findByRole('checkbox', { name: /GPU 0/ });
  expect(screen.getByRole('button', { name: '提交' })).toBeEnabled();
  fireEvent.click(gpu0); expect(screen.getByRole('button', { name: '提交' })).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox', { name: /GPU 1/ })); expect(screen.getByRole('button', { name: '提交' })).toBeEnabled();
  expect(screen.getByTestId('selected')).toHaveTextContent('["cuda:0","cuda:1"]');
});
it('does not silently replace a missing explicit card with automatic selection', async () => {
  const changed = vi.fn(), valid = vi.fn();
  render(<GpuDevicePicker value={['cuda:7']} onChange={changed} onValidityChange={valid}/>);
  await screen.findByText('所选显卡已不可用，请重新选择。');
  expect(valid).toHaveBeenLastCalledWith(false); expect(changed).not.toHaveBeenCalled();
  expect(screen.getByRole('combobox', { name: '运行显卡' })).toHaveTextContent('GPU 7');
});
it('keeps CPU/default-device scheduling usable when there are no GPUs', async () => {
  vi.mocked(apiClient.get).mockResolvedValue({ devices: [], max_concurrent: null });
  render(<Harness/>); await screen.findByText('未检测到可调度显卡，将使用当前环境的默认设备。');
  expect(screen.getByRole('button', { name: '提交' })).toBeEnabled();
});
it('keeps the compact picker open when selecting an option rendered in its portal', async () => {
  const changed = vi.fn();
  const { container } = render(<GpuDevicePicker compact value={[]} onChange={changed}/>);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  const details = container.querySelector('details')!;
  fireEvent.click(screen.getByText('自动选卡'));
  expect(details.open).toBe(true);
  fireEvent.click(screen.getByRole('combobox', { name: '运行显卡' }));
  const option = screen.getByRole('option', { name: /GPU 1.*空闲/ });
  fireEvent.mouseDown(option);
  expect(details.open).toBe(true);
  fireEvent.click(option);
  expect(changed).toHaveBeenCalledWith(['cuda:1']);
  fireEvent.mouseDown(document.body);
  expect(details.open).toBe(false);
});

function TrainingHarness({initial = [], initialCount = 1}: {initial?: string[]; initialCount?: number}) {
  const [selected, setSelected] = React.useState(initial);
  const [count, setCount] = React.useState(initialCount);
  const [valid, setValid] = React.useState(true);
  return <><input type="number" aria-label="显卡数量" value={count} onChange={event => setCount(Number(event.target.value))}/>
    <GpuDevicePicker training value={selected} count={count} onValidityChange={setValid} onChange={setSelected}/>
    <output data-testid="request">{JSON.stringify({count,devices:selected})}</output><button disabled={!valid}>提交</button></>;
}
it('training single-device selection replaces the previous GPU without changing the configured count', async () => {
  render(<TrainingHarness/>);
  expect(screen.getByRole('button', {name:'训练显卡'})).toHaveTextContent('自动');
  fireEvent.click(screen.getByRole('button', {name:'训练显卡'}));
  fireEvent.click(await screen.findByRole('radio', {name:/GPU 0/}));
  fireEvent.click(screen.getByRole('radio', {name:/GPU 1/}));
  expect(screen.getByTestId('request')).toHaveTextContent('{"count":1,"devices":["cuda:1"]}');
  fireEvent.keyDown(document, {key:'Escape'});
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.getByRole('button', {name:'训练显卡'})).toHaveTextContent('GPU 1 · Hygon DCU');
  expect(screen.getByRole('button', {name:'训练显卡'})).toHaveFocus();
});
it('training multi-device choice enforces the independent count and explains incomplete requests', async () => {
  vi.mocked(apiClient.get).mockResolvedValue({devices:[...devices,{device:'cuda:2',name:'Third GPU',status:'free'}],max_concurrent:null});
  render(<TrainingHarness initialCount={2}/>);
  fireEvent.click(screen.getByRole('button', {name:'训练显卡'}));
  fireEvent.click(await screen.findByRole('checkbox', {name:/GPU 0/}));
  expect(screen.getByRole('button', {name:'提交'})).toBeDisabled();
  expect(screen.getByText('已选 1 张，还需选择 1 张；也可改为自动。')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('checkbox', {name:/GPU 1/}));
  expect(screen.getByRole('checkbox', {name:/GPU 2/})).toBeDisabled();
  expect(screen.getByRole('button', {name:'提交'})).toBeEnabled();
  expect(screen.getByTestId('request')).toHaveTextContent('{"count":2,"devices":["cuda:0","cuda:1"]}');
  fireEvent.click(screen.getByRole('checkbox', {name:'自动选择 2 张空闲显卡'}));
  expect(screen.getByTestId('request')).toHaveTextContent('{"count":2,"devices":[]}');
  expect(screen.queryByRole('combobox', {name:'自动选择数量'})).not.toBeInTheDocument();
  fireEvent.mouseDown(document.body);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});
it('focuses the portaled training menu without scrolling the parameter page', async () => {
  const focus = vi.spyOn(HTMLElement.prototype, 'focus');
  render(<TrainingHarness/>);
  fireEvent.click(screen.getByRole('button', {name:'训练显卡'}));
  await screen.findByRole('radio', {name:/GPU 0/});
  expect(focus).toHaveBeenCalledWith({preventScroll:true});
});
it('training preserves and rejects missing devices, allows removal and disables unavailable alternatives', async () => {
  vi.mocked(apiClient.get).mockResolvedValue({devices:[...devices,{device:'cuda:2',name:'Offline',status:'unavailable'}],max_concurrent:null});
  render(<TrainingHarness initial={['cuda:7']}/>);
  await screen.findByText('所选显卡已不可用，请重新选择。');
  expect(screen.getByRole('button', {name:'提交'})).toBeDisabled();
  fireEvent.click(screen.getByRole('button', {name:'训练显卡'}));
  expect(screen.getByRole('radio', {name:/GPU 2/})).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox', {name:/GPU 7/}));
  expect(screen.getByTestId('request')).toHaveTextContent('{"count":1,"devices":[]}');
  expect(screen.getByRole('button', {name:'提交'})).toBeEnabled();
});
it('training does not silently reduce an imported automatic count when fewer devices are available', async () => {
  render(<TrainingHarness initialCount={4}/>);
  await screen.findByText('当前可用设备不足 4 张，请调整显卡数量。');
  expect(screen.getByRole('button', {name:'提交'})).toBeDisabled();
  expect(screen.getByTestId('request')).toHaveTextContent('{"count":4,"devices":[]}');
});
it('shared training validity rejects unavailable, duplicate and mismatched requests without excluding busy cards', () => {
  const snapshot = {devices, max_concurrent:null};
  expect(gpuSelectionValid(['cuda:0'],1,snapshot)).toBe(true);
  expect(gpuSelectionValid(['cuda:0'],2,snapshot)).toBe(false);
  expect(gpuSelectionValid(['cuda:0','cuda:0'],2,snapshot)).toBe(false);
  expect(gpuSelectionValid(['cuda:1'],1,null)).toBe(false);
  expect(gpuSelectionValid([],2,{devices:[],max_concurrent:null})).toBe(false);
  expect(gpuSelectionValid([],1,{devices:[],max_concurrent:null})).toBe(true);
});
