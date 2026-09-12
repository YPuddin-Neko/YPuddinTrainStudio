import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ImageEditor from '../src/components/masks/ImageEditor';
import * as paintApi from '../src/components/masks/paintApi';
import { ApiError } from '../src/api/types';
import '../src/i18n';
vi.mock('../src/components/masks/paintApi',async original=>({...await original<typeof paintApi>(),loadPaint:vi.fn(),savePaint:vi.fn(),restorePaint:vi.fn()}));
const initial:paintApi.PaintInfo={width:8,height:8,image_id:'old-hash',rel_path:'person.png',revision:'original',has_mask:false,mask_source:'alpha',can_restore:false};
const rgba=()=>new Uint8ClampedArray(Array.from({length:64},()=>[20,40,60,128]).flat());
let savedPixels:Uint8ClampedArray;let savedMask:Uint8Array;let latest:paintApi.PaintInfo;
const props=()=>({datasetId:'d_v2',imageId:'old-hash',relPath:'person.png',onClose:vi.fn(),onSaved:vi.fn(),onReloadList:vi.fn(),onEnableTraining:vi.fn<()=>Promise<void>>().mockResolvedValue(undefined)});
beforeEach(()=>{
  latest={...initial};savedPixels=rgba();savedMask=new Uint8Array(64).fill(128);
  vi.mocked(paintApi.loadPaint).mockImplementation(async()=>({info:{...latest},pixels:savedPixels.slice(),mask:savedMask.slice()}));
  vi.mocked(paintApi.savePaint).mockImplementation(async(_did,_info,pixels,mask)=>{
    if(pixels)savedPixels=pixels.slice();if(mask)savedMask=mask.slice();
    latest={...latest,image_id:'new-hash',revision:'saved',has_mask:!!mask||latest.has_mask,can_restore:true,operation_id:'op1'};return latest;
  });
  vi.mocked(paintApi.restorePaint).mockImplementation(async()=>{savedPixels=rgba();savedMask=new Uint8Array(64).fill(128);latest={...initial};return latest;});
  vi.spyOn(HTMLCanvasElement.prototype,'getContext').mockImplementation(()=>({createImageData:(w:number,h:number)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData:vi.fn()}) as any);
  vi.stubGlobal('PointerEvent',MouseEvent);HTMLCanvasElement.prototype.setPointerCapture=vi.fn();
});
afterEach(()=>{vi.restoreAllMocks();vi.clearAllMocks();vi.unstubAllGlobals();});
async function draw(){
  const canvas=await screen.findByLabelText('图像与遮罩绘制画布');
  vi.spyOn(canvas,'getBoundingClientRect').mockReturnValue({left:0,top:0,width:80,height:80} as DOMRect);
  fireEvent.change(screen.getByRole('spinbutton',{name:'笔刷像素'}),{target:{value:'1'}});
  fireEvent.pointerDown(canvas,{button:0,clientX:35,clientY:35,pointerId:1});fireEvent.pointerUp(canvas,{pointerId:1});
}
describe('image paint and loss-mask editor',()=>{
  it.each([404,409])('does not retry the stale image hash after HTTP %i and confirms before closing and refreshing the list',async(status)=>{
    const callbacks=props();render(<ImageEditor {...callbacks}/>);await screen.findByLabelText('图像与遮罩绘制画布');await draw();
    vi.mocked(paintApi.savePaint).mockRejectedValueOnce(new ApiError(status,{code:'paint.conflict',message:'Image identity changed'}));
    fireEvent.click(screen.getByRole('button',{name:'保存修改'}));await screen.findByRole('alert');
    expect(screen.queryByRole('button',{name:'重新读取'})).not.toBeInTheDocument();expect(screen.getByRole('button',{name:'保存修改'})).toBeDisabled();
    const confirm=vi.spyOn(window,'confirm').mockReturnValue(false);fireEvent.click(screen.getByRole('button',{name:'关闭并刷新列表'}));
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('放弃未保存'));expect(callbacks.onClose).not.toHaveBeenCalled();expect(callbacks.onReloadList).not.toHaveBeenCalled();
    expect(paintApi.loadPaint).toHaveBeenCalledTimes(1);confirm.mockReturnValue(true);fireEvent.click(screen.getByRole('button',{name:'关闭并刷新列表'}));
    expect(callbacks.onReloadList).toHaveBeenCalledOnce();expect(callbacks.onClose).toHaveBeenCalledOnce();expect(callbacks.onSaved).not.toHaveBeenCalled();expect(paintApi.loadPaint).toHaveBeenCalledTimes(1);
  });
  it('retains two independent drafts across modes, samples colors and atomically submits only actual modified layers',async()=>{
    const callbacks=props();render(<ImageEditor {...callbacks}/>);await screen.findByLabelText('图像与遮罩绘制画布');
    fireEvent.change(screen.getByLabelText('涂抹颜色'),{target:{value:'#ff0000'}});await draw();
    fireEvent.click(screen.getByRole('button',{name:'撤销'}));expect(screen.getByRole('button',{name:'保存修改'})).toBeDisabled();
    fireEvent.click(screen.getByRole('button',{name:'重做'}));
    fireEvent.click(screen.getByRole('tab',{name:'训练遮罩'}));fireEvent.click(screen.getByRole('button',{name:'全黑'}));
    fireEvent.click(screen.getByRole('tab',{name:/图像涂抹/}));
    fireEvent.click(screen.getByRole('button',{name:'取色'}));await draw();expect(screen.getByLabelText('涂抹颜色')).toHaveValue('#ff0000');
    fireEvent.click(screen.getByRole('button',{name:'保存修改'}));
    await waitFor(()=>expect(callbacks.onSaved).toHaveBeenCalledOnce());
    const [dataset,info,image,mask]=vi.mocked(paintApi.savePaint).mock.calls[0];
    expect(dataset).toBe('d_v2');expect(info.revision).toBe('original');expect(info.image_id).toBe('old-hash');
    expect(Array.from(image!.slice((3*8+3)*4,(3*8+3)*4+4))).toEqual([255,0,0,128]);expect(image!.slice(0,4)).toEqual(rgba().slice(0,4));
    expect(mask!.every(value=>value===0)).toBe(true);
    await waitFor(()=>expect(screen.getByRole('button',{name:'保存修改'})).toBeDisabled());
    expect(paintApi.loadPaint).toHaveBeenLastCalledWith('d_v2','new-hash','person.png',expect.any(AbortSignal));
  });
  it('keeps draft pixels on conflict, protects close, and only enables masked training after a successful save',async()=>{
    const callbacks=props();render(<ImageEditor {...callbacks}/>);await screen.findByLabelText('图像与遮罩绘制画布');
    fireEvent.click(screen.getByRole('tab',{name:'训练遮罩'}));fireEvent.click(screen.getByRole('button',{name:'全黑'}));
    vi.spyOn(window,'confirm').mockReturnValue(false);fireEvent.click(screen.getByRole('button',{name:'关闭图片编辑器'}));expect(callbacks.onClose).not.toHaveBeenCalled();
    vi.mocked(paintApi.savePaint).mockRejectedValueOnce(new Error('image.conflict: changed in another client'));
    fireEvent.click(screen.getByRole('button',{name:'保存并启用遮罩训练'}));expect(await screen.findByRole('alert')).toHaveTextContent('image.conflict');expect(callbacks.onEnableTraining).not.toHaveBeenCalled();
    expect(vi.mocked(paintApi.savePaint).mock.calls[0][2]).toBeUndefined();
    fireEvent.click(screen.getByRole('button',{name:'保存并启用遮罩训练'}));await waitFor(()=>expect(callbacks.onEnableTraining).toHaveBeenCalledOnce());
    expect(vi.mocked(paintApi.savePaint).mock.calls[1][3]!.every(value=>value===0)).toBe(true);
  });
  it('restores the previous save with its current revision and re-reads restored pixels',async()=>{
    latest={...initial,can_restore:true,revision:'after-edit'};
    render(<ImageEditor {...props()}/>);await screen.findByLabelText('图像与遮罩绘制画布');vi.spyOn(window,'confirm').mockReturnValue(true);
    fireEvent.click(screen.getByRole('button',{name:'恢复上次保存前'}));
    await waitFor(()=>expect(paintApi.restorePaint).toHaveBeenCalledWith('d_v2',expect.objectContaining({revision:'after-edit'})));
    await waitFor(()=>expect(screen.getByRole('button',{name:'恢复上次保存前'})).toBeDisabled());
    expect(await screen.findByText('已恢复保存前的文件。')).toBeInTheDocument();
  });
  it('blocks writes after a load error and keeps failed reload distinct from a successful file save',async()=>{
    vi.mocked(paintApi.loadPaint).mockRejectedValueOnce(new Error('outside allowed root'));
    render(<ImageEditor {...props()}/>);expect(await screen.findByRole('alert')).toHaveTextContent('outside allowed root');expect(screen.getByRole('button',{name:'保存修改'})).toBeDisabled();
    fireEvent.click(screen.getByRole('button',{name:'重新读取'}));await screen.findByLabelText('图像与遮罩绘制画布');await draw();
    vi.mocked(paintApi.loadPaint).mockRejectedValueOnce(new Error('connection lost'));
    fireEvent.click(screen.getByRole('button',{name:'保存修改'}));await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('文件已保存，但重新读取失败'));
    expect(screen.getByRole('button',{name:'保存修改'})).toBeDisabled();fireEvent.click(screen.getByRole('button',{name:'重新读取'}));
    await screen.findByLabelText('图像与遮罩绘制画布');expect(screen.getByRole('button',{name:'保存修改'})).toBeDisabled();
  });
});
