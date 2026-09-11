import React from 'react';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import StudioSelect, { type StudioSelectOption } from '../src/components/StudioSelect';
import SettingsDrawer from '../src/components/SettingsDrawer';
import '../src/i18n';

const options: StudioSelectOption[] = [
  { value: 'alpha', label: 'Alpha' },
  { value: 'blocked', label: 'Blocked', disabled: true },
  { value: 'bravo', label: 'Bravo' },
  { value: 'charlie', label: 'Charlie' },
  { value: 'delta', label: 'Delta' },
];
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
function Controlled({ onChange = () => {}, initial = 'alpha' }: { onChange?: (value: string) => void; initial?: string }) {
  const [value, setValue] = React.useState(initial);
  return <><StudioSelect aria-label="Choice" value={value} options={options} onValueChange={next => { onChange(next); setValue(next); }} /><button>Next field</button></>;
}
function activeLabel() {
  const id = screen.getByRole('combobox', { name: 'Choice' }).getAttribute('aria-activedescendant');
  return id ? document.getElementById(id)?.textContent : null;
}

describe('shared select interaction', () => {
  it('opens with the keyboard, skips disabled choices and commits only on Enter', async () => {
    const changed = vi.fn();
    const user = userEvent.setup();
    render(<Controlled onChange={changed} />);
    await user.tab();
    const trigger = screen.getByRole('combobox', { name: 'Choice' });
    await user.keyboard('{ArrowDown}');
    expect(trigger).toHaveFocus();
    expect(screen.getByRole('listbox')).toBeInTheDocument();
    expect(activeLabel()).toBe('Alpha');
    await user.keyboard('{ArrowDown}');
    expect(activeLabel()).toBe('Bravo');
    expect(changed).not.toHaveBeenCalled();
    expect(trigger).toHaveTextContent('Alpha');
    await user.keyboard('{ArrowUp}');
    expect(activeLabel()).toBe('Alpha');
    await user.keyboard('{ArrowUp}');
    expect(activeLabel()).toBe('Delta');
    await user.keyboard('{Enter}');
    expect(changed).toHaveBeenCalledExactlyOnceWith('delta');
    expect(trigger).toHaveTextContent('Delta');
    expect(trigger).toHaveFocus();
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  it('supports Home, End, Space and prefix search without selecting disabled matches', async () => {
    const user = userEvent.setup();
    render(<Controlled initial="bravo" />);
    await user.tab();
    await user.keyboard(' ');
    await user.keyboard('{End}');
    expect(activeLabel()).toBe('Delta');
    await user.keyboard('{Home}');
    expect(activeLabel()).toBe('Alpha');
    await user.keyboard('b');
    expect(activeLabel()).toBe('Bravo');
    await user.keyboard('ra');
    expect(activeLabel()).toBe('Bravo');
    await user.keyboard(' ');
    expect(screen.getByRole('combobox')).toHaveTextContent('Bravo');
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  it('Escape abandons a pending highlight and Tab moves focus to the next field', async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    render(<Controlled onChange={changed} />);
    await user.tab();
    await user.keyboard('{Enter}{End}{Escape}');
    const trigger = screen.getByRole('combobox');
    expect(trigger).toHaveFocus();
    expect(trigger).toHaveTextContent('Alpha');
    expect(changed).not.toHaveBeenCalled();
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    await user.keyboard('{Enter}');
    await user.tab();
    expect(screen.getByRole('button', { name: 'Next field' })).toHaveFocus();
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    expect(changed).not.toHaveBeenCalled();
  });

  it('ignores disabled options and retains the controlled value until the owner confirms a click', async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    const { rerender } = render(<StudioSelect aria-label="Choice" value="alpha" options={options} onValueChange={changed} />);
    const trigger = screen.getByRole('combobox');
    await user.click(trigger);
    await user.click(screen.getByRole('option', { name: 'Blocked' }));
    expect(changed).not.toHaveBeenCalled();
    expect(screen.getByRole('listbox')).toBeInTheDocument();
    await user.click(screen.getByRole('option', { name: 'Charlie' }));
    expect(changed).toHaveBeenCalledExactlyOnceWith('charlie');
    expect(trigger).toHaveTextContent('Alpha');
    expect(trigger).toHaveFocus();
    rerender(<StudioSelect aria-label="Choice" value="charlie" options={options} onValueChange={changed} disabled />);
    expect(trigger).toHaveTextContent('Charlie');
    await user.click(trigger);
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  it('positions a sidebar popup above its trigger and inside a narrow viewport', async () => {
    vi.stubGlobal('innerWidth', 390);
    vi.stubGlobal('innerHeight', 844);
    render(<Controlled />);
    const trigger = screen.getByRole('combobox');
    vi.spyOn(trigger, 'getBoundingClientRect').mockReturnValue({ x: 310, y: 790, left: 310, right: 385, top: 790, bottom: 824, width: 75, height: 34, toJSON: () => ({}) });
    fireEvent.click(trigger);
    const list = screen.getByRole('listbox');
    expect(list).toHaveStyle({ position: 'fixed', width: '160px', left: '222px', bottom: '59px' });
    expect(Number.parseFloat(list.style.left) + Number.parseFloat(list.style.width)).toBeLessThanOrEqual(382);
    expect(within(list).getAllByRole('option')).toHaveLength(5);
    fireEvent.pointerDown(document.body);
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  });

  it('closes only the select on Escape inside the settings drawer', async () => {
    const close = vi.fn();
    const user = userEvent.setup();
    render(<SettingsDrawer onClose={close}><Controlled /></SettingsDrawer>);
    await user.click(screen.getByRole('combobox'));
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    expect(close).not.toHaveBeenCalled();
    expect(screen.getByRole('combobox')).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(close).toHaveBeenCalledOnce();
  });

  it('leaves a nested path dialog in control of its own Escape key', async () => {
    const close = vi.fn();
    const innerClose = vi.fn();
    const user = userEvent.setup();
    render(<SettingsDrawer onClose={close}><section role="dialog" aria-label="Path picker" onKeyDown={event => {if (event.key === 'Escape') {event.stopPropagation();innerClose();}}}><input aria-label="Path"/></section></SettingsDrawer>);
    await user.click(screen.getByRole('textbox', {name:'Path'}));
    await user.keyboard('{Escape}');
    expect(innerClose).toHaveBeenCalledOnce();
    expect(close).not.toHaveBeenCalled();
  });
});
