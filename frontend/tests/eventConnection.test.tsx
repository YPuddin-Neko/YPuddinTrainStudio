import { act, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
vi.mock('../src/events/mockEventSource', () => ({ shouldUseMockEvents: () => false, createMockEventSource: vi.fn() }));
import { useEventStream, useEventStreamStatus } from '../src/events/useEventStream';

class FakeSource {
  static instances: FakeSource[] = [];
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  listeners = new Map<string, (event: any) => void>();
  close = vi.fn();
  constructor(public url: string) { FakeSource.instances.push(this); }
  addEventListener(type: string, listener: (event: any) => void) { this.listeners.set(type, listener); }
}
function Status({ onStep }: { onStep: (event: any) => void }) {
  const status = useEventStreamStatus();
  useEventStream('job.step', onStep);
  return <span>{status}</span>;
}
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('event stream reconnection', () => {
  it('reports disconnects, reconnects with the last id, and releases the connection on unmount', () => {
    vi.useFakeTimers();
    vi.stubGlobal('EventSource', FakeSource);
    const onStep = vi.fn();
    const { unmount } = render(<Status onStep={onStep} />);
    const first = FakeSource.instances[0];
    act(() => first.onopen?.());
    expect(screen.getByText('connected')).toBeInTheDocument();
    act(() => first.listeners.get('job.step')?.({ data: '{"job_id":"one","step":5}', lastEventId: '42' }));
    expect(onStep).toHaveBeenCalledWith({ job_id: 'one', step: 5 });
    act(() => first.onerror?.());
    expect(screen.getByText('disconnected')).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(3000));
    const second = FakeSource.instances[1];
    expect(new URL(second.url).searchParams.get('last_event_id')).toBe('42');
    act(() => second.onopen?.());
    expect(screen.getByText('connected')).toBeInTheDocument();
    unmount();
    expect(second.close).toHaveBeenCalled();
  });
});
