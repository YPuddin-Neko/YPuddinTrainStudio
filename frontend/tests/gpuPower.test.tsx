import { render, screen, cleanup } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { GpuCard } from '../src/components/GpuCard';
import '../src/i18n';

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
});
