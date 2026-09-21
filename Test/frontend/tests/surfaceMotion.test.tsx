import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Dialog from '../../../frontend/src/components/Dialog';
import SettingsDrawer from '../../../frontend/src/components/SettingsDrawer';
import '../../../frontend/src/i18n';
let reduced=false;
beforeEach(()=>{
  vi.useFakeTimers();
  vi.stubGlobal('matchMedia',(query:string)=>({matches:query==='(prefers-reduced-motion: reduce)'&&reduced,media:query,addEventListener:vi.fn(),removeEventListener:vi.fn()}));
});
afterEach(()=>{vi.useRealTimers();vi.unstubAllGlobals();reduced=false;});
function Fixture({drawer=false,onClose,disabled=false}:{drawer?:boolean;onClose:()=>void;disabled?:boolean}){
  const [open,setOpen]=React.useState(false);
  const close=()=>{onClose();setOpen(false);};
  return <><button onClick={()=>setOpen(true)}>Open surface</button>{open&&(drawer?<SettingsDrawer onClose={close}><input aria-label="Draft"/></SettingsDrawer>:<Dialog title="Edit draft" onClose={close} closeDisabled={disabled}><input aria-label="Draft"/></Dialog>)}</>;
}
describe('dialog and settings drawer exit lifecycle',()=>{
  it.each([false,true])('retains the %s surface for its exit once, then unmounts and returns focus',drawer=>{
    const closed=vi.fn();render(<Fixture drawer={drawer} onClose={closed}/>);
    const trigger=screen.getByRole('button',{name:'Open surface'});trigger.focus();fireEvent.click(trigger);
    const surface=screen.getByRole('dialog');fireEvent.change(screen.getByRole('textbox',{name:'Draft'}),{target:{value:'still mounted during exit'}});
    const close=screen.getByRole('button',{name:drawer?'关闭设置，返回工作区':'关闭'});fireEvent.click(close);fireEvent.click(close);fireEvent.keyDown(surface,{key:'Escape'});
    expect(surface.parentElement).toHaveClass('is-closing');expect(screen.getByRole('textbox',{name:'Draft'})).toHaveValue('still mounted during exit');expect(closed).not.toHaveBeenCalled();
    act(()=>vi.advanceTimersByTime(179));expect(surface).toBeInTheDocument();act(()=>vi.advanceTimersByTime(1));
    expect(closed).toHaveBeenCalledOnce();expect(screen.queryByRole('dialog')).not.toBeInTheDocument();expect(trigger).toHaveFocus();
  });
  it.each([false,true])('closes %s immediately when the user requests reduced motion',drawer=>{
    reduced=true;const closed=vi.fn();render(<Fixture drawer={drawer} onClose={closed}/>);fireEvent.click(screen.getByRole('button',{name:'Open surface'}));
    fireEvent.keyDown(screen.getByRole('dialog'),{key:'Escape'});
    expect(closed).toHaveBeenCalledOnce();expect(screen.queryByRole('dialog')).not.toBeInTheDocument();act(()=>vi.advanceTimersByTime(1000));expect(closed).toHaveBeenCalledOnce();
  });
  it('cancels a pending callback when its surface unmounts for another reason',()=>{
    const closed=vi.fn();const mounted=render(<Fixture onClose={closed}/>);fireEvent.click(screen.getByRole('button',{name:'Open surface'}));fireEvent.click(screen.getByRole('button',{name:'关闭'}));
    mounted.unmount();act(()=>vi.advanceTimersByTime(1000));expect(closed).not.toHaveBeenCalled();
  });
  it('does not dismiss a busy dialog through its button, Escape, or backdrop',()=>{
    const closed=vi.fn();render(<Fixture onClose={closed} disabled/>);fireEvent.click(screen.getByRole('button',{name:'Open surface'}));
    expect(screen.getByRole('button',{name:'关闭'})).toBeDisabled();fireEvent.keyDown(screen.getByRole('dialog'),{key:'Escape'});fireEvent.mouseDown(screen.getByRole('dialog').parentElement!);
    act(()=>vi.advanceTimersByTime(1000));expect(closed).not.toHaveBeenCalled();expect(screen.getByRole('dialog')).toBeInTheDocument();
  });
});
