import Switch from '../Switch';
import React from 'react';
import { createPortal } from 'react-dom';
import { Brush, ChevronLeft, ChevronRight, Eraser, Hand, Pipette, Redo2, Undo2, X, ZoomIn, ZoomOut } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { ApiError } from '../../api/types';
import { MaskDocument, imagePoint, paintSegment, type MaskOperation } from './maskDocument';
import { PaintDocument, type PaintStroke } from './paintDocument';
import { loadPaint, restorePaint, savePaint, type PaintInfo } from './paintApi';
import './image-editor.css';
import { SlidingIndicator } from '../motion';
import { LoadingNote } from '../Loading';
import OverflowStrip from '../OverflowStrip';
import { useConfirmation } from '../useConfirmation';
import { useImageViewport } from './useImageViewport';
import Dialog from '../Dialog';

type Mode = 'paint' | 'mask';
type Tool = 'brush' | 'erase' | 'maskErase' | 'pan' | 'pick';
const maskStrokeValues = { brush: 255, erase: 0, maskErase: 255 } as const;
export type ImageEditorIdentity = { datasetId: string; imageId: string; relPath: string };
export type ImageEditorNavigationTarget = ImageEditorIdentity & { position: number; total: number };
interface Props {
  datasetId: string; imageId: string; relPath: string;
  onClose: () => void; onSaved: (info?: PaintInfo, signal?: AbortSignal) => void | Promise<void>;
  navigation?: {
    position: number; total: number;
    resolve: (direction: -1 | 1, current: ImageEditorIdentity, signal: AbortSignal) => Promise<ImageEditorNavigationTarget | null>;
    onNavigated: (target: ImageEditorNavigationTarget) => void;
  };
  onEnableTraining?: () => Promise<void>;
  navigationError?: string;
  onReloadList?: () => void;
}
export default function ImageEditor({datasetId,imageId,relPath,onClose,onSaved,onEnableTraining,navigation,navigationError,onReloadList}:Props) {
  const text = useWorkspaceText();
  const { confirm, confirmation } = useConfirmation();
  const [mode,setMode] = React.useState<Mode>('paint');
  const [tool,setTool] = React.useState<Tool>('brush');
  const [color,setColor] = React.useState('#ffffff');
  const [diameter,setDiameter] = React.useState(24);
  const [opacity,setOpacity] = React.useState(.5);
  const [maskOnly,setMaskOnly] = React.useState(false);
  const [zoom,setZoom] = React.useState(1);
  const [bounds,setBounds] = React.useState({width:900,height:550});
  const [info,setInfo] = React.useState<PaintInfo|null>(null);
  const [loading,setLoading] = React.useState(true);
  const [verified,setVerified] = React.useState(false);
  const [saving,setSaving] = React.useState(false);
  const [switching,setSwitching] = React.useState(false);
  const [pendingDirection,setPendingDirection] = React.useState<-1|1|null>(null);
  const [switchError,setSwitchError] = React.useState('');
  const [error,setError] = React.useState('');
  const [requiresReopen,setRequiresReopen] = React.useState(false);
  const [message,setMessage] = React.useState('');
  const [revision,redraw] = React.useReducer((value:number)=>value+1,0);
  const [reload,setReload] = React.useState(0);
  const paint = React.useRef<PaintDocument|null>(null);
  const mask = React.useRef<MaskDocument|null>(null);
  const identity = React.useRef({imageId,relPath});
  const mounted = React.useRef(false);
  const actionBusy = React.useRef(false);
  const readController = React.useRef<AbortController|null>(null);
  const pendingSync = React.useRef<PaintInfo|null>(null);
  const savedCallback = React.useRef(onSaved);savedCallback.current=onSaved;
  const pendingChoice = React.useRef<-1|1|null>(null);
  const switchSequence = React.useRef(0);
  const busy = saving || switching || pendingDirection!==null;
  const unavailable = loading || !verified;
  const canvas = React.useRef<HTMLCanvasElement>(null);
  const viewport = React.useRef<HTMLDivElement>(null);
  const dialog = React.useRef<HTMLElement>(null);
  const closeButton = React.useRef<HTMLButtonElement>(null);
  const navigationControls = React.useRef<HTMLDivElement>(null);
  const returnFocus = React.useRef<-1|1|null>(null);
  const cursor = React.useRef<HTMLDivElement>(null);
  const stroke = React.useRef<PaintStroke|Extract<MaskOperation,{kind:'stroke'}>|null>(null);
  const frame = React.useRef<number|null>(null);
  const titleId = React.useId();
  const dirty = () => !!paint.current?.dirty || !!mask.current?.dirty || !!stroke.current;
  const reportError=(error:unknown)=>{setRequiresReopen(error instanceof ApiError&&[404,409].includes(error.status));setError(formatApiError(error));};
  const scale = info ? Math.max(.01,Math.min(1,(bounds.width-24)/info.width,(bounds.height-24)/info.height))*zoom : 1;
  const renderView = React.useCallback(()=>{
    const doc=paint.current, weights=mask.current, element=canvas.current;
    if(!doc || !weights || !element)return;
    const resolution = Math.min(1,1536/Math.max(doc.width,doc.height));
    const width=Math.max(1,Math.round(doc.width*resolution)),height=Math.max(1,Math.round(doc.height*resolution));
    if(element.width!==width)element.width=width;if(element.height!==height)element.height=height;
    const context=element.getContext('2d');if(!context)return;
    const output=context.createImageData(width,height);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const pixel=Math.min(doc.height-1,Math.floor(y/resolution))*doc.width+Math.min(doc.width-1,Math.floor(x/resolution));
      const source=pixel*4,target=(y*width+x)*4,weight=weights.pixels[pixel],alpha=doc.pixels[source+3]/255;
      const overlay=mode==='mask'&&!maskOnly?(1-weight/255)*opacity:0;
      for(let channel=0;channel<3;channel++){
        const base=doc.pixels[source+channel]*alpha+255*(1-alpha);
        output.data[target+channel]=mode==='mask'&&maskOnly?weight:Math.round(base*(1-overlay)+[239,68,68][channel]*overlay);
      }
      output.data[target+3]=255;
    }
    context.putImageData(output,0,0);
  },[mode,opacity,maskOnly]);
  const schedule=()=>{if(frame.current===null)frame.current=requestAnimationFrame(()=>{frame.current=null;renderView();});};
  const install=(loaded:Awaited<ReturnType<typeof loadPaint>>)=>{
    if(frame.current!==null){cancelAnimationFrame(frame.current);frame.current=null;}
    stroke.current=null;
    paint.current=new PaintDocument(loaded.info.width,loaded.info.height,loaded.pixels);
    mask.current=new MaskDocument(loaded.info.width,loaded.info.height,loaded.mask);
    identity.current={imageId:loaded.info.image_id,relPath:loaded.info.rel_path};
    setInfo(loaded.info);setVerified(true);redraw();
  };
  React.useEffect(()=>{
    mounted.current=true;
    const previous=document.activeElement as HTMLElement|null;
    closeButton.current?.focus();
    return()=>{mounted.current=false;readController.current?.abort();previous?.focus();if(frame.current!==null)cancelAnimationFrame(frame.current);};
  },[]);
  React.useEffect(()=>{
    const controller=new AbortController();setLoading(true);setError('');setRequiresReopen(false);
    void loadPaint(datasetId,identity.current.imageId,identity.current.relPath,controller.signal).then(async loaded=>{
      if(controller.signal.aborted)return;
      install(loaded);
      if(pendingSync.current){await savedCallback.current(loaded.info,controller.signal);if(!controller.signal.aborted)pendingSync.current=null;}
    }).catch(error=>{if(!controller.signal.aborted)reportError(error);}).finally(()=>{if(!controller.signal.aborted)setLoading(false);});
    return()=>controller.abort();
  },[datasetId,reload]);
  React.useEffect(()=>{renderView();},[revision,renderView,loading]);
  React.useEffect(()=>{
    if(busy||returnFocus.current===null)return;
    const buttons=navigationControls.current?.querySelectorAll<HTMLButtonElement>('button');
    const preferred=buttons?.[returnFocus.current===-1?0:1];
    (preferred&&!preferred.disabled?preferred:canvas.current)?.focus();returnFocus.current=null;
  },[busy]);
  React.useEffect(()=>{
    const host=viewport.current;if(!host)return;
    // Scrollbars must not change the fit scale after a wheel zoom.
    const update=()=>setBounds({width:host.offsetWidth||900,height:host.offsetHeight||550});update();
    const observer=new ResizeObserver(update);observer.observe(host);return()=>observer.disconnect();
  },[loading]);
  React.useEffect(()=>{
    const unload=(event:BeforeUnloadEvent)=>{if(dirty()||actionBusy.current){event.preventDefault();event.returnValue='';}};
    window.addEventListener('beforeunload',unload);return()=>window.removeEventListener('beforeunload',unload);
  },[saving]);
  const close=async()=>{
    if(actionBusy.current)return;
    if(dirty()&&!await confirm({title:text('放弃图片与遮罩修改','Discard image and mask edits'),message:text('图片或遮罩尚未保存，放弃这些修改并关闭？','Discard the unsaved image and mask changes?'),confirmLabel:text('放弃并关闭','Discard and close'),danger:true}))return;
    if(pendingSync.current)onReloadList?.();
    onClose();
  };
  const closeAndRefresh=async()=>{
    if(actionBusy.current)return;
    if(dirty()&&!await confirm({title:text('关闭并刷新列表','Close and refresh list'),message:text('关闭并刷新列表会放弃未保存的图片和遮罩修改，继续？','Close and refresh the list, discarding unsaved image and mask edits?'),confirmLabel:text('放弃并刷新列表','Discard and refresh list'),danger:true}))return;
    onReloadList?.();onClose();
  };
  const reloadImage=async()=>{
    if(actionBusy.current)return;
    if(dirty()&&!await confirm({title:text('重新读取图片与遮罩','Reload image and mask'),message:text('重新读取会放弃未保存修改，继续？','Reload and discard unsaved changes?'),confirmLabel:text('放弃并重新读取','Discard and reload'),danger:true}))return;
    setReload(value=>value+1);
  };
  const finishStroke=()=>{
    if(stroke.current){
      if('color' in stroke.current)paint.current?.apply(stroke.current,true);else mask.current?.apply(stroke.current,true);
      stroke.current=null;setMessage('');redraw();
    }
  };
  const changeMode=(next:Mode)=>{
    finishStroke();setMode(next);
    if(tool==='pick'||(next==='paint'&&tool==='maskErase'))setTool('brush');
  };
  const { panning, viewportEvents } = useImageViewport({ viewport, canvas, preview:cursor, zoom, setZoom, enabled:!!info&&!unavailable&&!busy, panTool:tool==='pan', isDrawing:()=>!!stroke.current, finishStroke });
  const point=(event:React.PointerEvent<HTMLCanvasElement>)=>imagePoint(event.clientX,event.clientY,event.currentTarget.getBoundingClientRect(),info!.width,info!.height);
  const pointerDown=(event:React.PointerEvent<HTMLCanvasElement>)=>{
    if(!paint.current||!mask.current||actionBusy.current||unavailable||event.button!==0)return;
    event.preventDefault();event.currentTarget.setPointerCapture(event.pointerId);
    const at=point(event);
    if(mode==='paint'&&(tool==='pick'||event.altKey)){setColor(paint.current.colorAt(at));setTool('brush');return;}
    if(mode==='paint'){
      stroke.current={kind:'stroke',points:[at],diameter,color,erase:tool==='erase'};
      paint.current.segment(at,at,diameter,color,tool==='erase');
    }else if(tool in maskStrokeValues){
      stroke.current={kind:'stroke',points:[at],diameter,value:maskStrokeValues[tool as keyof typeof maskStrokeValues]};
      paintSegment(mask.current.pixels,mask.current.width,mask.current.height,at,at,diameter,stroke.current.value);
    }
    schedule();
  };
  const pointerMove=(event:React.PointerEvent<HTMLCanvasElement>)=>{
    if(cursor.current){const rect=event.currentTarget.getBoundingClientRect();cursor.current.style.left=`${event.clientX-rect.left}px`;cursor.current.style.top=`${event.clientY-rect.top}px`;cursor.current.style.visibility=tool==='pan'||tool==='pick'?'hidden':'visible';}
    if(!stroke.current||!paint.current||!mask.current)return;
    const at=point(event),previous=stroke.current.points.at(-1)!;
    if('color' in stroke.current)paint.current.segment(previous,at,stroke.current.diameter,stroke.current.color,stroke.current.erase);
    else paintSegment(mask.current.pixels,mask.current.width,mask.current.height,previous,at,stroke.current.diameter,stroke.current.value);
    stroke.current.points.push(at);schedule();
  };
  const current=mode==='paint'?paint.current:mask.current;
  const history=(redo=false)=>{if(actionBusy.current||unavailable||stroke.current)return;if(redo)current?.redo();else current?.undo();setMessage('');redraw();};
  const operation=(op:MaskOperation)=>{if(!actionBusy.current&&!stroke.current){mask.current?.apply(op);setMessage('');redraw();}};
  const persist=async(enable=false):Promise<boolean>=>{
    if(!info||!paint.current||!mask.current||unavailable||requiresReopen)return false;
    finishStroke();setSaving(true);setError('');setMessage('');
    let committed:PaintInfo|undefined;
    let reloaded=false;
    try{
      const imageChanged=paint.current.dirty,maskChanged=mask.current.dirty||(enable&&!info.has_mask);
      if(imageChanged||maskChanged){
        committed=await savePaint(datasetId,info,imageChanged?paint.current.pixels:undefined,maskChanged?mask.current.pixels:undefined);
        if(!mounted.current)return false;
        identity.current={imageId:committed.image_id,relPath:committed.rel_path};setInfo(committed);
        setVerified(false);
        paint.current.markSaved();mask.current.markSaved();pendingSync.current=committed;
        // Re-read encoded pixels (e.g. JPEG) so the next brush operation starts from the file actually saved.
        const controller=new AbortController();readController.current=controller;
        const loaded=await loadPaint(datasetId,committed.image_id,committed.rel_path,controller.signal);
        if(!mounted.current||controller.signal.aborted)return false;
        install(loaded);
        reloaded=true;
      }
      if(pendingSync.current){await onSaved(pendingSync.current,readController.current?.signal);if(!mounted.current)return false;pendingSync.current=null;}
      setMessage(text('修改已保存；原文件备份已保留，下次训练将读取更新后的数据。','Changes saved with original backups. The next training run will use the updated data.'));
      if(enable)await onEnableTraining?.();
      return true;
    }catch(error){
      if(!mounted.current)return false;
      if(committed&&!reloaded){setLoading(true);setRequiresReopen(error instanceof ApiError&&[404,409].includes(error.status));setError(`${text('文件已保存，但重新读取失败，请重新读取后继续。','Files were saved, but reloading failed. Reload before continuing.')}\n${formatApiError(error)}`);}
      else if(pendingSync.current){setError(`${text('文件已保存，但列表刷新失败，请重试。','Files were saved, but refreshing the list failed. Try again.')}\n${formatApiError(error)}`);}
      else reportError(error);
      return false;
    }finally{if(mounted.current){setSaving(false);redraw();}}
  };
  const save=async(enable=false)=>{
    if(actionBusy.current)return;
    actionBusy.current=true;
    try{await persist(enable);}finally{actionBusy.current=false;}
  };
  const cancelSwitch=()=>{
    if(saving)return;
    readController.current?.abort();switchSequence.current++;
    actionBusy.current=false;setSwitching(false);
  };
  const performSwitch=async(direction:-1|1,saveFirst:boolean)=>{
    if(!navigation)return;
    const sequence=++switchSequence.current;
    setPendingDirection(null);setSwitching(true);setSwitchError('');
    const controller=new AbortController();readController.current=controller;
    try{
      const target=await navigation.resolve(direction,{datasetId,...identity.current},controller.signal);
      if(!mounted.current||controller.signal.aborted||!target)return;
      // Resolve before saving: modified-time sorting can move the current image after a write.
      if(saveFirst&&!await persist())return;
      if(pendingSync.current){
        await onSaved(pendingSync.current,controller.signal);
        if(!mounted.current||controller.signal.aborted)return;
        pendingSync.current=null;
      }
      if(!mounted.current||controller.signal.aborted)return;
      readController.current=controller;
      const loaded=await loadPaint(target.datasetId,target.imageId,target.relPath,controller.signal);
      if(!mounted.current||controller.signal.aborted)return;
      navigation.onNavigated({...target,imageId:loaded.info.image_id,relPath:loaded.info.rel_path});
      install(loaded);setZoom(1);viewport.current?.scrollTo(0,0);
      if(cursor.current)cursor.current.style.visibility='hidden';
      setMessage('');setError('');setRequiresReopen(false);pendingSync.current=null;
    }catch(error){
      if(mounted.current&&!controller.signal.aborted)setSwitchError(`${text('切换失败：','Could not switch images: ')}${formatApiError(error)}`);
    }finally{
      controller.abort();
      if(sequence===switchSequence.current){actionBusy.current=false;if(mounted.current)setSwitching(false);}
    }
  };
  const requestSwitch=(direction:-1|1)=>{
    if(actionBusy.current||unavailable||!info||!navigation)return;
    returnFocus.current=direction;
    finishStroke();actionBusy.current=true;
    if(dirty()){pendingChoice.current=direction;setPendingDirection(direction);}
    else void performSwitch(direction,false);
  };
  const cancelPending=()=>{pendingChoice.current=null;setPendingDirection(null);actionBusy.current=false;};
  const confirmSwitch=(saveFirst:boolean)=>{
    const direction=pendingChoice.current;if(direction===null)return;
    pendingChoice.current=null;
    void performSwitch(direction,saveFirst);
  };
  const restore=async()=>{
    if(!info||actionBusy.current||unavailable)return;
    if(!await confirm({title:text('恢复上次保存前','Restore previous save'),message:text('恢复最近一次保存前的图片和遮罩？当前未保存修改也会放弃。','Restore the image and mask from before the latest save? Unsaved edits will also be discarded.'),confirmLabel:text('恢复文件','Restore files'),danger:true}))return;
    if(actionBusy.current||!mounted.current)return;
    actionBusy.current=true;setSaving(true);setError('');
    try{
      const restored=await restorePaint(datasetId,info);if(!mounted.current)return;
      identity.current={imageId:restored.image_id,relPath:restored.rel_path};setInfo(restored);setVerified(false);
      paint.current?.markSaved();mask.current?.markSaved();pendingSync.current=restored;
      setLoading(true);setReload(value=>value+1);setMessage(text('已恢复保存前的文件。','Restored the files from before that save.'));
    }catch(error){if(mounted.current)reportError(error);}finally{actionBusy.current=false;if(mounted.current)setSaving(false);}
  };
  const keyDown=(event:React.KeyboardEvent)=>{
    if(!(event.target instanceof Element)||event.target.closest('[role="dialog"]')!==dialog.current)return;
    if(event.key==='Escape'){event.preventDefault();event.stopPropagation();close();}
    const formField=event.target instanceof HTMLInputElement;
    if(event.target instanceof HTMLElement&&event.target.getAttribute('role')==='tab'&&['ArrowLeft','ArrowRight','Home','End'].includes(event.key)&&!busy){
      event.preventDefault();
      const next=event.key==='Home'?'paint':event.key==='End'?'mask':mode==='paint'?'mask':'paint';
      changeMode(next);
      const tabs=dialog.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]');tabs?.[next==='paint'?0:1].focus();
    }
    if(!formField&&(event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();history(event.shiftKey);}
    if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();void save();}
    if(event.key==='Tab'){
      const controls=dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),canvas[tabindex]');
      if(!controls?.length)return;const first=controls[0],last=controls[controls.length-1];
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}
      else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
    }
  };
  return createPortal(<><div className="image-editor-backdrop" onClick={()=>void close()}>
    <section className="image-editor" role="dialog" aria-modal="true" aria-labelledby={titleId} ref={dialog} onKeyDown={keyDown} onClick={event=>event.stopPropagation()}>
      <header><div className="image-editor-heading"><h2 id={titleId}>{text('涂抹与遮罩','Paint and mask')}</h2><p title={identity.current.relPath}>{identity.current.relPath}{info&&` · ${info.width} × ${info.height}`}</p></div><div className="image-editor-header-actions">
        {navigation&&<div ref={navigationControls} className="image-editor-navigation" role="group" aria-label={text('切换图片','Switch image')}>
          <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一张','Previous image')} title={text('上一张','Previous image')} disabled={busy||unavailable||navigation.position<=1} onClick={()=>requestSwitch(-1)}><ChevronLeft size={15}/></button>
          <span aria-live="polite">{navigation.position} / {navigation.total}</span>
          <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一张','Next image')} title={text('下一张','Next image')} disabled={busy||unavailable||navigation.position>=navigation.total} onClick={()=>requestSwitch(1)}><ChevronRight size={15}/></button>
          {switching&&<button type="button" className="ui-btn ui-btn-sm" disabled={saving} onClick={cancelSwitch}>{text('取消切换','Cancel switch')}</button>}
        </div>}
        <button ref={closeButton} type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('关闭图片编辑器','Close image editor')} disabled={busy} onClick={close}><X size={17}/></button></div></header>
      <div className="image-editor-tools">
        <div className="ui-segmented" role="tablist" aria-label={text('编辑模式','Editing mode')}>{(['paint','mask'] as const).map(value=><button key={value} type="button" role="tab" aria-selected={mode===value} disabled={busy} onClick={()=>changeMode(value)}>{value==='paint'?text('图像涂抹','Image paint'):text('训练遮罩','Training mask')}{(value==='paint'?paint.current?.dirty:mask.current?.dirty)?' *':''}</button>)}<SlidingIndicator className="ui-segmented-thumb"/></div>
        <OverflowStrip className="ui-segmented" containerClassName="image-editor-tool-strip" role="group" label={text('绘制工具','Drawing tools')} activeKey={tool}>
          <button type="button" aria-pressed={tool==='brush'} disabled={busy||unavailable} onClick={()=>setTool('brush')}><Brush size={14}/>{mode==='paint'?text('画笔','Brush'):text('参与训练','Include')}</button>
          <button type="button" aria-pressed={tool==='erase'} disabled={busy||unavailable} onClick={()=>setTool('erase')}><Eraser size={14}/>{mode==='paint'?text('擦回原图','Erase paint'):text('忽略区域','Ignore')}</button>
          {mode==='mask'&&<button type="button" aria-pressed={tool==='maskErase'} title={text('擦除遮罩，恢复为参与训练区域','Erase mask to include this area in training')} disabled={busy||unavailable} onClick={()=>setTool('maskErase')}><Eraser size={14}/>{text('橡皮擦','Eraser')}</button>}
          <button type="button" aria-pressed={tool==='pan'} disabled={busy||unavailable} onClick={()=>setTool('pan')}><Hand size={14}/>{text('移动','Pan')}</button>
          {mode==='paint'&&<button type="button" aria-pressed={tool==='pick'} disabled={busy||unavailable} onClick={()=>setTool('pick')}><Pipette size={14}/>{text('取色','Pick color')}</button>}
          <SlidingIndicator className="ui-segmented-thumb"/>
        </OverflowStrip>
        {mode==='paint'&&<label>{text('颜色','Color')}<input type="color" aria-label={text('涂抹颜色','Paint color')} value={color} disabled={busy||unavailable} onChange={event=>setColor(event.target.value)}/></label>}
        <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('撤销','Undo')} title={text('撤销','Undo')} disabled={busy||unavailable||!current?.canUndo} onClick={()=>history()}><Undo2 size={14}/></button><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('重做','Redo')} title={text('重做','Redo')} disabled={busy||unavailable||!current?.canRedo} onClick={()=>history(true)}><Redo2 size={14}/></button>
        {mode==='paint'?<button type="button" className="ui-btn ui-btn-sm" disabled={busy||unavailable} onClick={()=>{paint.current?.apply({kind:'clear'});setMessage('');redraw();}}>{text('清除本次涂抹','Clear session paint')}</button>:<><button type="button" className="ui-btn ui-btn-sm" disabled={busy||unavailable} onClick={()=>operation({kind:'fill',value:255})}>{text('全白','All white')}</button><button type="button" className="ui-btn ui-btn-sm" disabled={busy||unavailable} onClick={()=>operation({kind:'fill',value:0})}>{text('全黑','All black')}</button><button type="button" className="ui-btn ui-btn-sm" disabled={busy||unavailable} onClick={()=>operation({kind:'invert'})}>{text('反转','Invert')}</button></>}
      </div>
      <div className="image-editor-tools"><label>{text('笔刷直径','Brush diameter')}<input aria-label={text('笔刷直径','Brush diameter')} type="range" min={1} max={1024} value={diameter} disabled={busy} onChange={event=>setDiameter(Number(event.target.value))}/><input aria-label={text('笔刷像素','Brush pixels')} type="number" min={1} max={1024} value={diameter} disabled={busy} onChange={event=>setDiameter(Math.max(1,Math.min(1024,Number(event.target.value)||1)))}/>px</label>
        {mode==='mask'&&<><label>{text('叠加透明度','Overlay opacity')}<input aria-label={text('叠加透明度','Overlay opacity')} type="range" min={0} max={1} step={.05} value={opacity} disabled={busy} onChange={event=>setOpacity(Number(event.target.value))}/></label><Switch disabled={busy} checked={maskOnly} onCheckedChange={setMaskOnly}>{text('仅看遮罩','Mask only')}</Switch></>}
        <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('缩小','Zoom out')} title={text('缩小','Zoom out')} disabled={busy} onClick={()=>setZoom(value=>Math.max(.25,value/1.5))}><ZoomOut size={14}/></button><button type="button" className="ui-btn ui-btn-sm" disabled={busy} onClick={()=>{setZoom(1);viewport.current?.scrollTo(0,0);}}>{text('适应','Fit')} {Math.round(scale*100)}%</button><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('放大','Zoom in')} title={text('放大','Zoom in')} disabled={busy} onClick={()=>setZoom(value=>Math.min(8,value*1.5))}><ZoomIn size={14}/></button>
      </div>
      <p className="image-editor-hint">{text('滚轮缩放，中键拖动。','Scroll to zoom; drag with the middle mouse button.')} {mode==='paint'?text('Alt + 点击取色；擦回原图可恢复本次编辑前的像素。','Alt + click picks a color. Erase restores pixels from before this editing session.'):text('白色参与训练，黑色忽略；红色为忽略区域。无独立遮罩时使用图片透明度。','White trains, black is ignored; red marks ignored areas. Without a separate mask, image transparency is used.')}</p>
      {error&&<p className="image-editor-status" role="alert">{error} {requiresReopen?<><span>{text('图片身份或文件已变化，请刷新列表后重新打开。','The image identity or files changed. Refresh the list and reopen the image.')}</span><button type="button" className="ui-btn ui-btn-sm" disabled={busy} onClick={()=>void closeAndRefresh()}>{text('关闭并刷新列表','Close and refresh list')}</button></>:<button type="button" className="ui-btn ui-btn-sm" disabled={busy} onClick={()=>void reloadImage()}>{text('重新读取','Reload')}</button>}</p>}
      {switching&&<LoadingNote className="image-editor-status" label={text('正在切换图片…','Switching images…')}/>}
      {switchError&&<p className="image-editor-status" role="alert">{switchError}</p>}
      {navigationError&&<p className="image-editor-status" role="alert">{navigationError}</p>}
      {loading?<LoadingNote className="image-editor-status" label={text('正在读取图片与遮罩…','Loading image and mask…')}/>:info&&<div ref={viewport} {...viewportEvents} className="image-editor-viewport" data-testid="paint-viewport" style={{cursor:panning?'grabbing':tool==='pan'?'grab':undefined}}><div className="image-editor-stage" style={{width:Math.max(1,info.width*scale),height:Math.max(1,info.height*scale)}}><canvas ref={canvas} aria-label={text('图像与遮罩绘制画布','Image and mask drawing canvas')} tabIndex={0} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={finishStroke} onPointerCancel={finishStroke} onLostPointerCapture={finishStroke} onPointerLeave={()=>{if(cursor.current)cursor.current.style.visibility='hidden';}} style={{cursor:panning?'grabbing':tool==='pan'?'grab':tool==='pick'?'copy':'crosshair'}}/><div ref={cursor} aria-hidden="true" className="image-editor-cursor" style={{visibility:'hidden',width:diameter*scale,height:diameter*scale}}/></div></div>}
      {message&&<p role="status" className="image-editor-status">{message}</p>}
      <footer><small>{dirty()?text('有未保存修改','Unsaved changes'):text('已同步','Up to date')} · {text('Ctrl/Cmd + Z 撤销，Ctrl/Cmd + S 保存','Ctrl/Cmd + Z to undo, Ctrl/Cmd + S to save')}</small><div className="image-editor-footer-actions"><button type="button" className="ui-btn" disabled={busy||unavailable||requiresReopen||!info?.can_restore} onClick={()=>void restore()}>{text('恢复上次保存前','Restore previous save')}</button><button type="button" className="ui-btn" disabled={busy} onClick={close}>{text('返回图片列表','Back to images')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={busy||unavailable||requiresReopen||!info||!dirty()} onClick={()=>void save()}>{saving?text('保存中…','Saving…'):text('保存修改','Save changes')}</button>{mode==='mask'&&onEnableTraining&&<button type="button" className="ui-btn" disabled={busy||unavailable||requiresReopen||!info} onClick={()=>void save(true)}>{text('保存并启用遮罩训练','Save and enable masked training')}</button>}</div></footer>
    </section>
  </div>{confirmation}{pendingDirection!==null&&<Dialog nested title={text('未保存修改','Unsaved changes')} onClose={cancelPending}>
    <p className="workspace-confirm-message">{text('保存当前图片和遮罩的修改后再切换？','Save changes to the current image and mask before switching?')}</p>
    <div className="workspace-confirm-actions image-editor-switch-actions">
      <button type="button" className="ui-btn" onClick={cancelPending}>{text('取消','Cancel')}</button>
      <button type="button" className="ui-btn ui-btn-danger" onClick={()=>confirmSwitch(false)}>{text('放弃修改并切换','Discard and switch')}</button>
      <button type="button" className="ui-btn ui-btn-primary" onClick={()=>confirmSwitch(true)}>{text('保存并切换','Save and switch')}</button>
    </div>
  </Dialog>}</>,document.body);
}
