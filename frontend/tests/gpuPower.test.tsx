import { render, screen, cleanup } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { GpuCard } from '../src/components/GpuCard';
import i18n from '../src/i18n';

beforeEach(async()=>{await i18n.changeLanguage('zh-CN');});

afterEach(cleanup);
describe('GPU telemetry presentation', () => {
  it('shows idle zero watts and separates the driver limit', () => {
    render(<MemoryRouter><GpuCard gpu={{index: 0, name: 'RTX', kind: 'cuda', power_w: 0, power_limit_w: 450, telemetry_source: 'nvidia-smi'}} /></MemoryRouter>);
    expect(screen.getByTestId('gpu-power')).toHaveTextContent('0W');
    expect(screen.getByText('功率上限 450 W')).toBeInTheDocument();
  });
  it('retains an unavailable power reading and explains its cause', () => {
    render(<MemoryRouter><GpuCard gpu={{index: 0, name: 'RTX', kind: 'cuda', power_w: null, telemetry_note: 'nvidia_power_unavailable'}} /></MemoryRouter>);
    expect(screen.getByTestId('gpu-power')).toHaveTextContent('未提供');
    expect(screen.getByText(/驱动暂未返回功率/)).toBeInTheDocument();
    expect(screen.getByRole('link', {name: '查看环境'})).toHaveAttribute('href', '/settings#environment');
  });
  it.each([0,46])('shows system Apple GPU utilization %s independently from unavailable sensors',value=>{
    const gpu={index:0,name:'Apple M4 GPU',kind:'mps' as const,telemetry_source:'ioreg',telemetry_note:'mps_power_unavailable',util_pct:value,power_w:null,temp_c:null,mem_used_mb:8192,mem_total_mb:16384};
    const {rerender}=render(<MemoryRouter><GpuCard gpu={gpu}/></MemoryRouter>);
    expect(screen.getByText('全系统 GPU 占用率').nextElementSibling).toHaveTextContent(`${value}%`);
    expect(screen.getByText(/全系统 Apple GPU 活动/)).toHaveTextContent('IORegistry');
    expect(screen.getByText('系统统一内存').parentElement).toHaveAttribute('title',expect.stringContaining('并非 GPU 专用显存或训练进程分配量'));
    expect(screen.getByText(/当前未取得 GPU 功率读数/)).toHaveTextContent('当前未取得 GPU 温度读数');
    expect(screen.queryByText(/不提供功率|适配开发中/)).not.toBeInTheDocument();
    rerender(<MemoryRouter><GpuCard gpu={{...gpu,util_pct:null,power_w:22,temp_c:45}}/></MemoryRouter>);
    expect(screen.getByText('全系统 GPU 占用率').nextElementSibling).toHaveTextContent('—');
    expect(screen.getByTestId('gpu-power')).toHaveTextContent('22W');
    expect(screen.getByText('温度').nextElementSibling).toHaveTextContent('45°');
    expect(screen.getByText('当前未取得 GPU 利用率读数。')).toBeInTheDocument();
    expect(screen.queryByText(/当前未取得 GPU 功率读数/)).not.toBeInTheDocument();
    expect(screen.queryByText(/当前未取得 GPU 温度读数/)).not.toBeInTheDocument();
  });

});
