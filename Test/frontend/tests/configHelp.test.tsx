import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Link, MemoryRouter } from 'react-router-dom';
import ConfigHelp from '../../../frontend/src/components/ConfigHelp';
import { SchemaForm } from '../../../frontend/src/schema/SchemaForm/SchemaForm';
import '../../../frontend/src/i18n';

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

function geometry(width: number, height: number, x: number, y: number, mainLeft = 0) {
  vi.stubGlobal('innerWidth', width);
  vi.stubGlobal('innerHeight', height);
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    if (this.tagName === 'MAIN') return new DOMRect(mainLeft, 48, width - mainLeft, height - 48);
    if (this.classList.contains('config-help-popover')) return new DOMRect(0, 0, 320, 180);
    return new DOMRect(x, y, 24, 24);
  });
  vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockImplementation(function (this: HTMLElement) {
    return this.classList.contains('config-help-popover') ? 180 : height;
  });
}

describe('configuration help', () => {
  it.each([
    { width: 1004, height: 773, x: 204, y: 220, mainLeft: 184, above: false },
    { width: 1004, height: 773, x: 972, y: 726, mainLeft: 184, above: true },
    { width: 390, height: 773, x: 12, y: 720, mainLeft: 0, above: true },
    { width: 390, height: 773, x: 350, y: 80, mainLeft: 0, above: false },
  ])('keeps the portal in its workspace at $width px (anchor $x, $y)', fixture => {
    geometry(fixture.width, fixture.height, fixture.x, fixture.y, fixture.mainLeft);
    render(<main style={{ overflow: 'auto' }}><div style={{ overflow: 'hidden' }}><ConfigHelp label="不放大小图 说明">解释较长的训练行为，不应被字段或侧栏裁剪。</ConfigHelp></div></main>);
    fireEvent.click(screen.getByRole('button', { name: '不放大小图 说明' }));
    const popup = screen.getByRole('tooltip');
    expect(popup.parentElement).toBe(document.body);
    const left = Number.parseFloat(popup.style.left);
    const top = Number.parseFloat(popup.style.top);
    const actualWidth = Number.parseFloat(popup.style.width);
    const actualHeight = Math.min(180, Number.parseFloat(popup.style.maxHeight));
    expect(left).toBeGreaterThanOrEqual(fixture.mainLeft + 8);
    expect(left + actualWidth).toBeLessThanOrEqual(fixture.width - 8);
    expect(top).toBeGreaterThanOrEqual(56);
    expect(top + actualHeight).toBeLessThanOrEqual(fixture.height - 8);
    if (fixture.above) expect(top + actualHeight).toBeLessThanOrEqual(fixture.y - 6);
    else expect(top).toBeGreaterThanOrEqual(fixture.y + 24 + 6);
  });

  it('dismisses on outside pointerdown but preserves text selection and scrolling inside', async () => {
    const user = userEvent.setup();
    render(<><ConfigHelp label="字段说明">可以选择与滚动阅读。</ConfigHelp><button>空白处的其他操作</button></>);
    const button = screen.getByRole('button', { name: '字段说明' });
    await user.click(button);
    const popup = screen.getByRole('tooltip');
    fireEvent.pointerDown(popup);
    fireEvent.scroll(popup);
    expect(popup).toBeInTheDocument();
    fireEvent.pointerDown(screen.getByRole('button', { name: '空白处的其他操作' }));
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    expect(button).toHaveAttribute('aria-expanded', 'false');
  });

  it('allows keyboard opening, Escape dismissal and normal focus navigation', async () => {
    const user = userEvent.setup();
    render(<><ConfigHelp label="字段说明">帮助内容</ConfigHelp><input aria-label="下一参数" /></>);
    const button = screen.getByRole('button', { name: '字段说明' });
    await user.tab();
    await user.keyboard('{Enter}');
    expect(button).toHaveAttribute('aria-describedby', screen.getByRole('tooltip').id);
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    expect(button).toHaveFocus();
    await user.keyboard(' ');
    expect(screen.getByRole('tooltip')).toBeInTheDocument();
    await user.tab();
    expect(screen.getByRole('textbox', { name: '下一参数' })).toHaveFocus();
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  it('keeps only one open and closes on ancestor scroll, route changes and unmount', () => {
    const view = render(<MemoryRouter><main><ConfigHelp label="A说明">内容A</ConfigHelp><ConfigHelp label="B说明">内容B</ConfigHelp><Link to="/?tab=sampling">采样参数</Link></main></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: 'A说明' }));
    fireEvent.click(screen.getByRole('button', { name: 'B说明' }));
    expect(screen.getAllByRole('tooltip')).toHaveLength(1);
    expect(screen.getByRole('tooltip')).toHaveTextContent('内容B');
    expect(screen.getByRole('button', { name: 'A说明' })).toHaveAttribute('aria-expanded', 'false');
    fireEvent.scroll(screen.getByRole('main'));
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'A说明' }));
    // click alone has no pointerdown/focus transition, so this exercises routing.
    fireEvent.click(screen.getByRole('link', { name: '采样参数' }));
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'A说明' }));
    view.unmount();
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  it('uses the shared popup in SchemaForm and removes it when its section disappears', () => {
    const schema = { properties: { data: { type: 'boolean', title: 'No upscale', description: 'Never enlarge small images.' } } };
    const view = render(<SchemaForm schema={schema} value={{ data: true }} onChange={() => {}} compact />);
    fireEvent.click(screen.getByRole('button', { name: /说明|help/ }));
    expect(screen.getByRole('tooltip')).toHaveTextContent('Never enlarge small images.');
    view.rerender(<SchemaForm schema={schema} value={{ data: true }} onChange={() => {}} compact search="not-present" />);
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });
});
