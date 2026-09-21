import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useWorkspaceHeight } from '../../../frontend/src/components/projects/useWorkspaceHeight';

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

function Workspace({ ready, variable = '--toolbar-height' }: { ready: boolean; variable?: string }) {
  const toolbarRef = useWorkspaceHeight(variable, ready);
  return <main data-testid="workspace">{ready && <div ref={toolbarRef}>Toolbar</div>}</main>;
}

describe('workspace toolbar height observation', () => {
  it('starts after a delayed version becomes ready and tracks wrapping without retaining stale offsets', () => {
    let height = 46;
    let notify = () => {};
    const observe = vi.fn();
    const disconnect = vi.fn();
    vi.stubGlobal('ResizeObserver', class {
      constructor(callback: () => void) { notify = callback; }
      observe = observe;
      disconnect = disconnect;
    });
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => ({ height } as DOMRect));
    const mounted = render(<Workspace ready={false}/>);
    const workspace = screen.getByTestId('workspace');
    expect(observe).not.toHaveBeenCalled();
    mounted.rerender(<Workspace ready/>);
    expect(observe).toHaveBeenCalledWith(screen.getByText('Toolbar'));
    expect(workspace.style.getPropertyValue('--toolbar-height')).toBe('46px');
    height = 88;
    act(() => notify());
    expect(workspace.style.getPropertyValue('--toolbar-height')).toBe('88px');
    mounted.rerender(<Workspace ready={false}/>);
    expect(disconnect).toHaveBeenCalledOnce();
    expect(workspace.style.getPropertyValue('--toolbar-height')).toBe('');
    height = 46;
    mounted.rerender(<Workspace ready/>);
    expect(workspace.style.getPropertyValue('--toolbar-height')).toBe('46px');
  });

  it('updates on window resize when ResizeObserver is unavailable and cleans renamed measurements', () => {
    let height = 54;
    vi.stubGlobal('ResizeObserver', undefined);
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => ({ height } as DOMRect));
    const mounted = render(<Workspace ready/>);
    const workspace = screen.getByTestId('workspace');
    expect(workspace.style.getPropertyValue('--toolbar-height')).toBe('54px');
    height = 96;
    act(() => window.dispatchEvent(new Event('resize')));
    expect(workspace.style.getPropertyValue('--toolbar-height')).toBe('96px');
    mounted.rerender(<Workspace ready variable="--launch-height"/>);
    expect(workspace.style.getPropertyValue('--toolbar-height')).toBe('');
    expect(workspace.style.getPropertyValue('--launch-height')).toBe('96px');
    mounted.unmount();
    expect(workspace.style.getPropertyValue('--launch-height')).toBe('');
  });
});
