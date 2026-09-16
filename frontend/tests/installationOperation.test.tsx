import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import InstallationOperation, { InstallationLog } from '../src/components/InstallationOperation';

let contentHeight = 600;
beforeEach(() => {
  contentHeight = 600;
  const offsets = new WeakMap<HTMLElement, number>();
  // jsdom does not lay out text or clamp scrolling as a browser does.
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(200);
  vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(() => contentHeight);
  vi.spyOn(HTMLElement.prototype, 'scrollTop', 'get').mockImplementation(function (this: HTMLElement) {
    return offsets.get(this) ?? 0;
  });
  vi.spyOn(HTMLElement.prototype, 'scrollTop', 'set').mockImplementation(function (this: HTMLElement, value: number) {
    offsets.set(this, Math.max(0, Math.min(contentHeight - 200, value)));
  });
});
afterEach(() => vi.restoreAllMocks());

it('starts at the latest log and follows appends and a fixed-length rotating buffer', () => {
  const view = render(<InstallationLog label="安装日志" logs={['old', 'latest']}/>);
  const log = screen.getByLabelText('安装日志');
  expect(log.scrollTop).toBe(400);
  contentHeight = 800;
  view.rerender(<InstallationLog label="安装日志" logs={['old', 'latest', 'appended']}/>);
  expect(log.scrollTop).toBe(600);

  // Same number of entries, but the new message wraps to more lines.
  contentHeight = 1000;
  view.rerender(<InstallationLog label="安装日志" logs={['latest', 'appended', 'long new message']}/>);
  expect(log.scrollTop).toBe(800);
  expect(log).toHaveTextContent('long new message');
});

it('keeps the reader in history until they scroll back to the bottom', () => {
  const view = render(<InstallationLog label="安装日志" logs={['old', 'latest']}/>);
  const log = screen.getByLabelText('安装日志');
  log.scrollTop = 120;
  fireEvent.scroll(log);
  contentHeight = 800;
  view.rerender(<InstallationLog label="安装日志" logs={['old', 'latest', 'appended']}/>);
  expect(log.scrollTop).toBe(120);
  contentHeight = 1000;
  view.rerender(<InstallationLog label="安装日志" logs={['latest', 'appended', 'long new message']}/>);
  expect(log.scrollTop).toBe(120);

  log.scrollTop = 800;
  fireEvent.scroll(log);
  contentHeight = 1200;
  view.rerender(<InstallationLog label="安装日志" logs={['appended', 'long new message', 'complete']}/>);
  expect(log.scrollTop).toBe(1000);
});

it('opens the latest log again after the operation is collapsed', () => {
  const operation = (logs: string[]) => <InstallationOperation title="PyTorch" status="正在安装">
    <InstallationLog label="安装日志" logs={logs}/>
  </InstallationOperation>;
  const view = render(operation(['old', 'latest']));
  const log = screen.getByLabelText('安装日志');
  log.scrollTop = 120;
  fireEvent.scroll(log);
  fireEvent.click(screen.getByRole('button', { name: /PyTorch/ }));
  expect(screen.queryByLabelText('安装日志')).not.toBeInTheDocument();
  contentHeight = 1000;
  view.rerender(operation(['latest', 'new while collapsed']));
  fireEvent.click(screen.getByRole('button', { name: /PyTorch/ }));
  expect(screen.getByLabelText('安装日志')).toHaveTextContent('new while collapsed');
  expect(screen.getByLabelText('安装日志').scrollTop).toBe(800);
});
