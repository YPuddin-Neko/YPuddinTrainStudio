import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Link, MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import SystemTelemetry from '../../../frontend/src/components/SystemTelemetry';
import Layout from '../../../frontend/src/components/Layout';
import { apiClient } from '../../../frontend/src/api/client';
import type { SystemStats } from '../../../frontend/src/api/types';
import i18n from '../../../frontend/src/i18n';

const stream = vi.hoisted(() => ({ status: 'connected', callbacks: new Map<string, (data: any) => void>() }));
vi.mock('../../../frontend/src/events/useEventStream', () => ({ useEventStream: (type: string, callback: (data: any) => void) => { stream.callbacks.set(type, callback); }, useEventStreamStatus: () => stream.status }));
const snapshot = (): SystemStats => ({ cpu_pct: 7.6, ram: { used_mb: 12288, total_mb: 32768 }, disks: [{ path: 'D:/Studio/data', used_gb: 512, total_gb: 1024 }], gpus: [{ index: 0, kind: 'cuda', name: 'RTX test', util_pct: 49, mem_used_mb: 3072, mem_total_mb: 12288, power_w: 193.2, temp_c: 58 }] });
const settings = { ui: { theme: 'light', language: 'zh-CN' } };
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN'); stream.status = 'connected'; stream.callbacks.clear();
  vi.spyOn(apiClient, 'get').mockImplementation(async endpoint => {
    if (endpoint === '/settings') return settings as any;
    if (endpoint === '/system/stats') return snapshot() as any;
    if (endpoint === '/system/info') return { ypuddin: 'test' } as any;
    if (endpoint === '/jobs') return { items: [] } as any;
    throw new Error(`Unexpected ${endpoint}`);
  });
});
afterEach(() => { vi.restoreAllMocks(); });

function RouteState() {
  const location = useLocation();
  return <output>{location.state?.backgroundLocation?.pathname || 'No background'}</output>;
}
function shell(child: React.ReactNode = <div>Page content</div>) {
  return <MemoryRouter initialEntries={['/projects/p1/train']}><Routes><Route element={<Layout />}>
    <Route path="/projects/p1/train" element={child}/><Route path="/projects" element={<div>Projects destination</div>}/><Route path="/settings" element={<RouteState/>}/>
  </Route></Routes></MemoryRouter>;
}

describe('stable system telemetry', () => {
  it('orders CPU, memory, GPU and disk with measured percentages and all four GPU readings', () => {
    const { container } = render(<SystemTelemetry stats={snapshot()}/>);
    expect([...container.querySelectorAll('.telemetry-group')].map(node => node.getAttribute('data-testid'))).toEqual(['telemetry-cpu', 'telemetry-memory', 'telemetry-gpu', 'telemetry-disk']);
    expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('8%');
    expect(screen.getByLabelText('内存占用率')).toHaveTextContent('38%');
    expect(screen.getByLabelText('GPU 占用率')).toHaveTextContent('49%');
    expect(screen.getByLabelText('显存占用率')).toHaveTextContent('25%');
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('193 W');
    expect(screen.getByLabelText('GPU 温度')).toHaveTextContent('58 °C');
    expect(screen.getByLabelText('硬盘占用率')).toHaveTextContent('50%');
    expect(screen.getByTestId('telemetry-disk')).toHaveAttribute('title', expect.stringContaining('D:/Studio/data'));
    expect(container.querySelector('.lucide-gpu')).toBeInTheDocument();
    expect(container.querySelector('.lucide-zap')).not.toBeInTheDocument();
  });

  it('keeps missing values unknown and preserves genuine zero readings', () => {
    const { rerender } = render(<SystemTelemetry stats={null}/>);
    for (const label of ['CPU 占用率', '内存占用率', 'GPU 占用率', '显存占用率', 'GPU 功率', 'GPU 温度', '硬盘占用率']) expect(screen.getByLabelText(label)).toHaveTextContent(/^—$/);
    const data = snapshot(); data.cpu_pct = 0; data.gpus[0] = { ...data.gpus[0], util_pct: 0, power_w: 0, temp_c: null, mem_used_mb: 0 };
    rerender(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('0 W');
    expect(screen.getByLabelText('GPU 占用率')).toHaveTextContent('0%');
    expect(screen.getByLabelText('显存占用率')).toHaveTextContent('0%');
    expect(screen.getByLabelText('GPU 温度')).toHaveTextContent(/^—$/);
    data.gpus[0] = { ...data.gpus[0], mem_total_mb: 0, power_w: Number.NaN };
    rerender(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('显存占用率')).toHaveTextContent(/^—$/);
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent(/^—$/);
  });

  it('retains dedicated number and unit slots from missing readings through zero and three digits', () => {
    const { rerender } = render(<SystemTelemetry stats={null}/>);
    const power = screen.getByLabelText('GPU 功率');
    const numberSlot = power.querySelector('.telemetry-number');
    const unitSlot = power.querySelector('.telemetry-unit');
    expect(numberSlot).toHaveTextContent('—');
    expect(unitSlot).toBeEmptyDOMElement();
    for (const value of [0, 9, 99, 100, 999]) {
      const data = snapshot(); data.gpus[0].power_w = value; data.cpu_pct = value > 100 ? 100 : value;
      rerender(<SystemTelemetry stats={data}/>);
      expect(power.querySelector('.telemetry-number')).toBe(numberSlot);
      expect(power.querySelector('.telemetry-unit')).toBe(unitSlot);
      expect(numberSlot).toHaveTextContent(String(value));
      expect(unitSlot).toHaveTextContent('W');
      expect(screen.getByLabelText('CPU 占用率').querySelector('.telemetry-unit')).toHaveTextContent('%');
    }
    expect(screen.getByTestId('telemetry-memory')).toHaveAttribute('title', expect.stringContaining('12.0 / 32.0 GiB'));
  });

  it('keeps the selected GPU identity across reordered updates and falls back when it disappears', () => {
    const data = snapshot(); const second = { ...data.gpus[0], index: 3, name: 'Second GPU', power_w: 280, util_pct: 85 };
    data.gpus.push(second);
    const { rerender } = render(<SystemTelemetry stats={data}/>);
    fireEvent.click(screen.getByRole('combobox', { name: '选择监控显卡' }));
    fireEvent.click(screen.getByRole('option', { name: 'GPU 3 · Second GPU' }));
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('280 W');
    rerender(<SystemTelemetry stats={{ ...data, gpus: [{ ...second, power_w: 285 }, data.gpus[0]] }}/>);
    expect(screen.getByRole('combobox')).toHaveTextContent(/^GPU 3CUDA$/);
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('285 W');
    rerender(<SystemTelemetry stats={snapshot()}/>);
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('193 W');
  });

  it('defaults to per-card averages and keeps full device names in the menu only', () => {
    const data = snapshot();
    data.gpus.push({ ...data.gpus[0], index: 3, name: 'Another long GPU model name', util_pct: 75, mem_used_mb: 12288, mem_total_mb: 24576, power_w: 280.8, temp_c: 66 });
    const {rerender} = render(<SystemTelemetry stats={data}/>);
    expect(screen.getByRole('combobox')).toHaveTextContent(/^多卡平均CUDA · 2 卡$/);
    expect(screen.getByLabelText('GPU 占用率')).toHaveTextContent('62%');
    expect(screen.getByLabelText('显存占用率')).toHaveTextContent('38%');
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('237 W');
    expect(screen.getByLabelText('GPU 温度')).toHaveTextContent('62 °C');
    expect(screen.getByLabelText('显存占用率').parentElement).toHaveAttribute('title', expect.stringContaining('每张显卡已用显存的百分比'));
    fireEvent.click(screen.getByRole('combobox'));
    expect(screen.getByRole('option', {name:'多卡平均 · 2 张显卡'})).toHaveAttribute('aria-selected','true');
    fireEvent.click(screen.getByRole('option', {name:'GPU 3 · Another long GPU model name'}));
    expect(screen.getByRole('combobox')).toHaveTextContent(/^GPU 3CUDA$/);
    expect(screen.getByLabelText('GPU 占用率')).toHaveTextContent('75%');
    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByRole('option', {name:'多卡平均 · 2 张显卡'}));
    rerender(<SystemTelemetry stats={{...data, gpus:[{...data.gpus[1],util_pct:51},data.gpus[0]]}}/>);
    expect(screen.getByLabelText('GPU 占用率')).toHaveTextContent('50%');
  });

  it('never treats a missing device reading as zero in an average', () => {
    const data = snapshot();
    data.gpus.push({ ...data.gpus[0], index: 1, util_pct: null, power_w: 0, temp_c: Number.NaN, mem_used_mb: 0 });
    const {rerender} = render(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('GPU 占用率')).toHaveTextContent(/^—$/);
    expect(screen.getByLabelText('GPU 温度')).toHaveTextContent(/^—$/);
    expect(screen.getByLabelText('GPU 占用率').parentElement).toHaveAttribute('title',expect.stringContaining('1/2'));
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('97 W');
    expect(screen.getByLabelText('显存占用率')).toHaveTextContent('13%');
    rerender(<SystemTelemetry stats={{...data, gpus:data.gpus.map(item=>({...item,util_pct:0,power_w:0,temp_c:0,mem_used_mb:0}))}}/>);
    for (const label of ['GPU 占用率','显存占用率']) expect(screen.getByLabelText(label)).toHaveTextContent('0%');
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('0 W');
    expect(screen.getByLabelText('GPU 温度')).toHaveTextContent('0 °C');
  });

  it('labels MPS memory as system unified memory and does not invent GPU load or power', () => {
    const data = snapshot(); data.gpus[0] = { index: 0, kind: 'mps', name: 'Apple GPU', mem_used_mb: 8192, mem_total_mb: 16384, util_pct: null, power_w: null, temp_c: null };
    render(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('系统统一内存占用率')).toHaveTextContent('50%');
    expect(screen.queryByLabelText('显存占用率')).not.toBeInTheDocument();
    expect(screen.getByTestId('telemetry-gpu')).toHaveAttribute('title', expect.stringContaining('并非 GPU 专用显存或训练进程分配量'));
    expect(screen.getByLabelText('GPU 估算功率')).toHaveTextContent(/^—$/);
  });

  it.each([['dtk', 'DTK'], ['rocm', 'ROCm']] as const)('labels %s and keeps unavailable driver readings unknown', (kind, label) => {
    const data = snapshot();
    data.gpus = [
      { index: 0, kind, name: 'BW GPU 0', mem_total_mb: 65536, mem_used_mb: 4096, util_pct: null, power_w: null, temp_c: null, telemetry_source: 'torch-hip', telemetry_note: 'hip_driver_metrics_unavailable' },
      { index: 1, kind, name: 'BW GPU 1', mem_total_mb: 65536, mem_used_mb: 8192, util_pct: null, power_w: null, temp_c: null, telemetry_source: 'torch-hip', telemetry_note: 'hip_driver_metrics_unavailable' },
    ];
    render(<SystemTelemetry stats={data}/>);
    expect(screen.getByText(`${label} · 2 卡`)).toBeInTheDocument();
    expect(screen.queryByText('CUDA')).not.toBeInTheDocument();
    expect(screen.getByLabelText('GPU 占用率').parentElement).toHaveAttribute('title', expect.stringContaining('0/2'));
    expect(screen.getByTestId('telemetry-gpu')).not.toHaveAttribute('title', expect.stringContaining('hardware.hip'));
    for (const reading of ['GPU 占用率', 'GPU 功率', 'GPU 温度']) expect(screen.getByLabelText(reading)).toHaveTextContent(/^—$/);
    expect(screen.getByLabelText('显存占用率')).toHaveTextContent('9%');
    fireEvent.click(screen.getByRole('combobox', { name: '选择监控显卡' }));
    fireEvent.click(screen.getByRole('option', { name: 'GPU 1 · BW GPU 1' }));
    expect(screen.getByLabelText('显存占用率')).toHaveTextContent('13%');
    expect(screen.getByTestId('telemetry-gpu')).toHaveAttribute('title', expect.stringContaining('暂时无法可靠获取这张卡'));
  });

  it.each([0, 46])('shows Apple GPU driver utilization %s while missing power and temperature stay independent', value => {
    const data = snapshot();
    data.gpus[0] = {index:0, kind:'mps', name:'Apple M4 GPU', telemetry_source:'ioreg', telemetry_note:'mps_power_unavailable', util_pct:value, power_w:null, temp_c:null, mem_used_mb:8192, mem_total_mb:16384};
    const {rerender}=render(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('全系统 GPU 占用率')).toHaveTextContent(`${value}%`);
    const group=screen.getByTestId('telemetry-gpu');
    expect(group).toHaveAttribute('title',expect.stringContaining('全系统 Apple GPU 利用率'));
    expect(group).toHaveAttribute('title',expect.stringContaining('包含桌面与其他应用'));
    expect(group).toHaveAttribute('title',expect.stringContaining('IORegistry'));
    expect(group.getAttribute('title')).not.toContain('不提供');
    expect(group.getAttribute('title')).not.toContain('未取得 GPU 利用率');
    expect(screen.getByLabelText('GPU 估算功率')).toHaveTextContent(/^—$/);
    expect(screen.getByLabelText('GPU 估算功率').parentElement).toHaveAttribute('title','当前未取得 GPU 功率读数。');
    expect(screen.getByLabelText('GPU 均温').parentElement).toHaveAttribute('title','当前未取得 GPU 温度读数。');
    rerender(<SystemTelemetry stats={{...data,gpus:[{...data.gpus[0],telemetry_source:'mps',util_pct:null}]}}/>);
    expect(screen.getByLabelText('全系统 GPU 占用率')).toHaveTextContent(/^—$/);
    expect(group.getAttribute('title')).toContain('当前未取得 GPU 利用率读数');
    expect(group.getAttribute('title')).not.toContain('IORegistry');
    expect(screen.getByLabelText('系统统一内存占用率')).toHaveTextContent('50%');
  });

  it('explains Apple readings in English without claiming MPS cannot expose them', async () => {
    await i18n.changeLanguage('en');
    const data=snapshot(); data.gpus[0]={...data.gpus[0],kind:'mps',telemetry_source:'ioreg',telemetry_note:'mps_power_unavailable',util_pct:46,power_w:null,temp_c:null};
    render(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('System GPU utilization')).toHaveTextContent('46%');
    expect(screen.getByTestId('telemetry-gpu').getAttribute('title')).toContain('including the desktop and other applications');
    expect(screen.getByLabelText('Estimated GPU power').parentElement).toHaveAttribute('title','No GPU power reading is currently available.');
    expect(screen.getByLabelText('Mean GPU temperature').parentElement).toHaveAttribute('title','No GPU temperature reading is currently available.');
    expect(screen.getByTestId('telemetry-gpu').getAttribute('title')).not.toContain('does not expose');
  });

  it.each([[0,'0'],[0.37,'0.37'],[0.004,'<0.01']])('shows Apple estimated watts %s without rounding positive idle readings to zero', (power, displayed) => {
    const data=snapshot(); data.gpus[0]={...data.gpus[0],kind:'mps',telemetry_source:'ioreg',power_w:power as number,power_source:'ioreport',power_estimated:true,power_sample_seconds:1.25,temp_c:45.64,temp_max_c:49.83,temp_sensor_count:8,temperature_source:'smc'};
    render(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('GPU 估算功率')).toHaveTextContent(`${displayed} W`);
    expect(screen.getByText('估算功率')).toBeInTheDocument();
    const powerCell=screen.getByLabelText('GPU 估算功率').parentElement!;
    expect(powerCell).toHaveAttribute('title',expect.stringContaining('IOReport'));
    expect(powerCell).toHaveAttribute('title',expect.stringContaining('平均采样区间：1.25 秒'));
    expect(powerCell).toHaveAttribute('title',expect.stringContaining('采样期间全系统 GPU 的估算平均功率'));
    expect(screen.getByText('均温')).toBeInTheDocument();
    const temperature=screen.getByLabelText('GPU 均温');
    expect(temperature).toHaveTextContent('45.6 °C');
    expect(temperature.parentElement).toHaveAttribute('title',expect.stringContaining('最高温：49.8 °C'));
    expect(temperature.parentElement).toHaveAttribute('title',expect.stringContaining('有效传感器：8 个'));
    expect(temperature.parentElement).toHaveAttribute('title',expect.stringContaining('不是 GPU 核心数'));
    expect(temperature.parentElement).toHaveAttribute('title',expect.stringContaining('SMC'));
  });

  it('keeps sensor failures independent and does not fabricate missing sampling metadata', () => {
    const data=snapshot(); data.gpus[0]={...data.gpus[0],kind:'mps',power_w:0.37,power_estimated:true,power_sample_seconds:null,power_source:'ioreport',temp_c:null,temp_max_c:null,temp_sensor_count:null,temperature_source:null};
    const {rerender}=render(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('GPU 估算功率')).toHaveTextContent('0.37 W');
    expect(screen.getByLabelText('GPU 估算功率').parentElement).toHaveAttribute('title',expect.stringContaining('平均采样区间未取得'));
    expect(screen.getByLabelText('GPU 均温')).toHaveTextContent(/^—$/);
    rerender(<SystemTelemetry stats={{...data,gpus:[{...data.gpus[0],power_w:null,temp_c:44.1}]}}/>);
    expect(screen.getByLabelText('GPU 估算功率')).toHaveTextContent(/^—$/);
    expect(screen.getByLabelText('GPU 均温')).toHaveTextContent('44.1 °C');
    expect(screen.getByLabelText('GPU 均温').parentElement).toHaveAttribute('title',expect.stringContaining('最高温未取得'));
    expect(screen.getByLabelText('GPU 均温').parentElement).toHaveAttribute('title',expect.stringContaining('有效传感器数量未取得'));
  });

  it('keeps CUDA integer readings and labels without Apple sensor explanations', () => {
    render(<SystemTelemetry stats={snapshot()}/>);
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('193 W');
    expect(screen.getByLabelText('GPU 温度')).toHaveTextContent('58 °C');
    expect(screen.queryByText('估算功率')).not.toBeInTheDocument();
    expect(screen.queryByText('均温')).not.toBeInTheDocument();
    const tip=screen.getByTestId('telemetry-gpu').getAttribute('title');
    expect(tip).not.toMatch(/Apple|IOReport|SMC|均温|估算平均功率/);
  });

  it('explains ROCm driver power and edge temperature without claiming isolated APU GPU power', () => {
    const data=snapshot();
    data.gpus[0]={...data.gpus[0],kind:'rocm',power_source:'hwmon',temperature_source:'hwmon-edge'};
    render(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('GPU 功率').parentElement).toHaveAttribute('title',expect.stringContaining('包含 CPU 功耗'));
    expect(screen.getByLabelText('GPU 温度').parentElement).toHaveAttribute('title',expect.stringContaining('边缘温度'));
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('193 W');
  });

  it('removes normal connection text, keeps disconnect feedback, and updates live readings', async () => {
    const { rerender } = render(shell());
    await waitFor(() => expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('193 W'));
    expect(screen.queryByTestId('event-connection')).not.toBeInTheDocument();
    const next = snapshot(); next.gpus[0].power_w = 211;
    act(() => stream.callbacks.get('system.stats')?.(next));
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('211 W');
    stream.status = 'disconnected'; rerender(shell());
    expect(screen.getByTestId('event-connection')).toHaveTextContent(i18n.t('connection.disconnected'));
    expect(screen.getByTestId('app-topbar')).toBeInTheDocument();
  });

  it('shows initial telemetry failures, retries, and never replaces live data with a late cold response', async () => {
    const ordinaryGet = vi.mocked(apiClient.get).getMockImplementation()!;
    let complete: (value: SystemStats) => void = () => {};
    let attempts = 0;
    vi.mocked(apiClient.get).mockImplementation(endpoint => {
      if (endpoint !== '/system/stats') return ordinaryGet(endpoint);
      attempts += 1;
      return attempts === 1 ? Promise.reject(new Error('Hardware probe failed')) : new Promise(resolve => { complete = resolve as (value: SystemStats) => void; });
    });
    render(shell());
    const retry = await screen.findByRole('button', { name: '硬件状态读取失败，点击重试' });
    expect(screen.getByRole('alert')).toHaveTextContent('Hardware probe failed');
    fireEvent.click(retry);
    const latest = snapshot(); latest.gpus[0].power_w = 255;
    act(() => stream.callbacks.get('system.stats')?.(latest));
    await act(async () => complete(snapshot()));
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('255 W');
    expect(screen.queryByRole('button', { name: '硬件状态读取失败，点击重试' })).not.toBeInTheDocument();
  });

  it('retains the same shell and page frame while lazy content loads, and carries settings context', async () => {
    let finish: (value: { default: React.ComponentType }) => void = () => {};
    const LazyPage = React.lazy(() => new Promise<{ default: React.ComponentType }>(resolve => { finish = resolve; }));
    render(shell(<LazyPage/>));
    const header = screen.getByTestId('app-topbar');
    const frame = screen.getByTestId('app-page-frame');
    expect(screen.getByTestId('app-page-loading')).toBeInTheDocument();
    await act(async () => finish({ default: () => <Link to="/projects">Open projects</Link> }));
    expect(screen.getByTestId('app-topbar')).toBe(header);
    expect(screen.getByTestId('app-page-frame')).toBe(frame);
    fireEvent.click(within(screen.getByRole('complementary')).getByRole('link', { name: i18n.t('nav.settings') }));
    expect(await screen.findByText('/projects/p1/train')).toBeInTheDocument();
    expect(screen.getByTestId('app-topbar')).toBe(header);
    expect(screen.getByTestId('app-page-frame')).toBe(frame);
  });
});
