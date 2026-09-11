import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Link, MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import SystemTelemetry from '../src/components/SystemTelemetry';
import Layout from '../src/components/Layout';
import { apiClient } from '../src/api/client';
import type { SystemStats } from '../src/api/types';
import i18n from '../src/i18n';

const stream = vi.hoisted(() => ({ status: 'connected', callbacks: new Map<string, (data: any) => void>() }));
vi.mock('../src/events/useEventStream', () => ({ useEventStream: (type: string, callback: (data: any) => void) => { stream.callbacks.set(type, callback); }, useEventStreamStatus: () => stream.status }));
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
    fireEvent.change(screen.getByRole('combobox', { name: '选择监控显卡' }), { target: { value: '3' } });
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('280 W');
    rerender(<SystemTelemetry stats={{ ...data, gpus: [{ ...second, power_w: 285 }, data.gpus[0]] }}/>);
    expect(screen.getByRole('combobox')).toHaveValue('3');
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('285 W');
    rerender(<SystemTelemetry stats={snapshot()}/>);
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent('193 W');
  });

  it('labels MPS memory as system unified memory and does not invent GPU load or power', () => {
    const data = snapshot(); data.gpus[0] = { index: 0, kind: 'mps', name: 'Apple GPU', mem_used_mb: 8192, mem_total_mb: 16384, util_pct: null, power_w: null, temp_c: null };
    render(<SystemTelemetry stats={data}/>);
    expect(screen.getByLabelText('系统统一内存占用率')).toHaveTextContent('50%');
    expect(screen.queryByLabelText('显存占用率')).not.toBeInTheDocument();
    expect(screen.getByTestId('telemetry-gpu')).toHaveAttribute('title', expect.stringContaining('并非训练进程分配量'));
    expect(screen.getByLabelText('GPU 功率')).toHaveTextContent(/^—$/);
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
    fireEvent.click(within(screen.getByRole('navigation', { name: '主导航' })).getByRole('link', { name: i18n.t('nav.settings') }));
    expect(await screen.findByText('/projects/p1/train')).toBeInTheDocument();
    expect(screen.getByTestId('app-topbar')).toBe(header);
    expect(screen.getByTestId('app-page-frame')).toBe(frame);
  });
});
