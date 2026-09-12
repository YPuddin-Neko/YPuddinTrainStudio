import React from 'react';
import { createPortal } from 'react-dom';
import { Brush, Eraser, Hand, Pipette, Redo2, Undo2, X, ZoomIn, ZoomOut } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { ApiError } from '../../api/types';
import { MaskDocument, imagePoint, paintSegment, type MaskOperation } from './maskDocument';
import { PaintDocument, type PaintStroke } from './paintDocument';
import { loadPaint, restorePaint, savePaint, type PaintInfo } from './paintApi';
import './image-editor.css';

type Mode = 'paint' | 'mask';
type Tool = 'brush' | 'erase' | 'pan' | 'pick';
interface Props {
  datasetId: string; imageId: string; relPath: string;
  onClose: () => void; onSaved: (info?: PaintInfo) => void;
  onEnableTraining?: () => Promise<void>;
  navigationError?: string;
  onReloadList?: () => void;
}
export default function ImageEditor({datasetId,imageId,relPath,onClose,onSaved,onEnableTraining,navigationError,onReloadList}:Props) {
  const text = useWorkspaceText();
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
  const [saving,setSaving] = React.useState(false);
  const [error,setError] = React.useState('');
  const [requiresReopen,setRequiresReopen] = React.useState(false);
  const [message,setMessage] = React.useState('');
  const [revision,redraw] = React.useReducer((value:number)=>value+1,0);
  const [reload,setReload] = React.useState(0);
  const paint = React.useRef<PaintDocument|null>(null);
  const mask = React.useRef<MaskDocument|null>(null);
  const identity = React.useRef({imageId,relPath});
  const canvas = React.useRef<HTMLCanvasElement>(null);
  const viewport = React.useRef<HTMLDivElement>(null);
  const dialog = React.useRef<HTMLElement>(null);
  const closeButton = React.useRef<HTMLButtonElement>(null);
  const cursor = React.useRef<HTMLDivElement>(null);
  const stroke = React.useRef<PaintStroke|Extract<MaskOperation,{kind:'stroke'}>|null>(null);
  const pan = React.useRef<{x:number;y:number;left:number;top:number}|null>(null);
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
    paint.current=new PaintDocument(loaded.info.width,loaded.info.height,loaded.pixels);
    mask.current=new MaskDocument(loaded.info.width,loaded.info.height,loaded.mask);
    identity.current={imageId:loaded.info.image_id,relPath:loaded.info.rel_path};
    setInfo(loaded.info);redraw();
  };
  React.useEffect(()=>{
    const previous=document.activeElement as HTMLElement|null;
    closeButton.current?.focus();
    return()=>{previous?.focus();if(frame.current!==null)cancelAnimationFrame(frame.current);};
  },[]);
  React.useEffect(()=>{
    const controller=new AbortController();setLoading(true);setError('');setRequiresReopen(false);
    void loadPaint(datasetId,identity.current.imageId,identity.current.relPath,controller.signal).then(loaded=>{
      if(!controller.signal.aborted)install(loaded);
    }).catch(error=>{if(!controller.signal.aborted)reportError(error);}).finally(()=>{if(!controller.signal.aborted)setLoading(false);});
    return()=>controller.abort();
  },[datasetId,reload]);
  React.useEffect(()=>{renderView();},[revision,renderView,loading]);
  React.useEffect(()=>{
    const host=viewport.current;if(!host)return;
    const update=()=>setBounds({width:host.clientWidth||900,height:host.clientHeight||550});update();
    const observer=new ResizeObserver(update);observer.observe(host);return()=>observer.disconnect();
  },[loading]);
  React.useEffect(()=>{
    const unload=(event:BeforeUnloadEvent)=>{if(dirty()||saving){event.preventDefault();event.returnValue='';}};
    window.addEventListener('beforeunload',unload);return()=>window.removeEventListener('beforeunload',unload);
  },[saving]);
  const close=()=>{if(!saving&&(!dirty()||window.confirm(text('图片或遮罩尚未保存，放弃这些修改并关闭？','Discard the unsaved image and mask changes?'))))onClose();};
  const closeAndRefresh=()=>{
    if(saving||dirty()&&!window.confirm(text('关闭并刷新列表会放弃未保存的图片和遮罩修改，继续？','Close and refresh the list, discarding unsaved image and mask edits?')))return;
    onReloadList?.();onClose();
  };
  const finishStroke=()=>{
    if(stroke.current){
      if('color' in stroke.current)paint.current?.apply(stroke.current,true);else mask.current?.apply(stroke.current,true);
      stroke.current=null;setMessage('');redraw();
    }
    pan.current=null;
  };
  const point=(event:React.PointerEvent<HTMLCanvasElement>)=>imagePoint(event.clientX,event.clientY,event.currentTarget.getBoundingClientRect(),info!.width,info!.height);
  const pointerDown=(event:React.PointerEvent<HTMLCanvasElement>)=>{
    if(!paint.current||!mask.current||saving||loading||event.button!==0)return;
    event.preventDefault();event.currentTarget.setPointerCapture(event.pointerId);
    if(tool==='pan'){pan.current={x:event.clientX,y:event.clientY,left:viewport.current?.scrollLeft||0,top:viewport.current?.scrollTop||0};return;}
    const at=point(event);
    if(mode==='paint'&&(tool==='pick'||event.altKey)){setColor(paint.current.colorAt(at));setTool('brush');return;}
    if(mode==='paint'){
      stroke.current={kind:'stroke',points:[at],diameter,color,erase:tool==='erase'};
      paint.current.segment(at,at,diameter,color,tool==='erase');
    }else{
      stroke.current={kind:'stroke',points:[at],diameter,value:tool==='erase'?0:255};
      paintSegment(mask.current.pixels,mask.current.width,mask.current.height,at,at,diameter,stroke.current.value);
    }
    schedule();
  };
  const pointerMove=(event:React.PointerEvent<HTMLCanvasElement>)=>{
    if(cursor.current){const rect=event.currentTarget.getBoundingClientRect();cursor.current.style.left=`${event.clientX-rect.left}px`;cursor.current.style.top=`${event.clientY-rect.top}px`;cursor.current.style.visibility=tool==='pan'||tool==='pick'?'hidden':'visible';}
    if(pan.current&&viewport.current){viewport.current.scrollLeft=pan.current.left-(event.clientX-pan.current.x);viewport.current.scrollTop=pan.current.top-(event.clientY-pan.current.y);return;}
    if(!stroke.current||!paint.current||!mask.current)return;
    const at=point(event),previous=stroke.current.points.at(-1)!;
    if('color' in stroke.current)paint.current.segment(previous,at,stroke.current.diameter,stroke.current.color,stroke.current.erase);
    else paintSegment(mask.current.pixels,mask.current.width,mask.current.height,previous,at,stroke.current.diameter,stroke.current.value);
    stroke.current.points.push(at);schedule();
  };
  const current=mode==='paint'?paint.current:mask.current;
  const history=(redo=false)=>{if(saving||stroke.current)return;if(redo)current?.redo();else current?.undo();setMessage('');redraw();};
  const operation=(op:MaskOperation)=>{if(!saving&&!stroke.current){mask.current?.apply(op);setMessage('');redraw();}};
  const save=async(enable=false)=>{
    if(!info||!paint.current||!mask.current||saving||loading||requiresReopen)return;
    finishStroke();setSaving(true);setError('');setMessage('');
    let committed:PaintInfo|undefined;
    let reloaded=false;
    try{
      const imageChanged=paint.current.dirty,maskChanged=mask.current.dirty||(enable&&!info.has_mask);
      if(imageChanged||maskChanged){
        committed=await savePaint(datasetId,info,imageChanged?paint.current.pixels:undefined,maskChanged?mask.current.pixels:undefined);
        identity.current={imageId:committed.image_id,relPath:committed.rel_path};setInfo(committed);
        paint.current.markSaved();mask.current.markSaved();onSaved(committed);
        // Re-read encoded pixels (e.g. JPEG) so the next brush operation starts from the file actually saved.
        install(await loadPaint(datasetId,committed.image_id,committed.rel_path,new AbortController().signal));
        reloaded=true;
      }
      setMessage(text('修改已保存；原文件备份已保留，下次训练将读取更新后的数据。','Changes saved with original backups. The next training run will use the updated data.'));
      if(enable)await onEnableTraining?.();
    }catch(error){
      if(committed&&!reloaded){setLoading(true);setRequiresReopen(error instanceof ApiError&&[404,409].includes(error.status));setError(`${text('文件已保存，但重新读取失败，请重新读取后继续。','Files were saved, but reloading failed. Reload before continuing.')}\n${formatApiError(error)}`);}
      else reportError(error);
    }finally{setSaving(false);redraw();}
  };
  const restore=async()=>{
    if(!info||saving||!window.confirm(text('恢复最近一次保存前的图片和遮罩？当前未保存修改也会放弃。','Restore the image and mask from before the latest save? Unsaved edits will also be discarded.')))return;
    setSaving(true);setError('');
    try{
      const restored=await restorePaint(datasetId,info);identity.current={imageId:restored.image_id,relPath:restored.rel_path};setInfo(restored);
      paint.current?.markSaved();mask.current?.markSaved();onSaved(restored);
      setLoading(true);setReload(value=>value+1);setMessage(text('已恢复保存前的文件。','Restored the files from before that save.'));
    }catch(error){reportError(error);}finally{setSaving(false);}
  };
  const keyDown=(event:React.KeyboardEvent)=>{
    if(event.key==='Escape'){event.preventDefault();event.stopPropagation();close();}
    const formField=event.target instanceof HTMLInputElement;
    if(event.target instanceof HTMLElement&&event.target.getAttribute('role')==='tab'&&['ArrowLeft','ArrowRight','Home','End'].includes(event.key)&&!saving){
      event.preventDefault();finishStroke();
      const next=event.key==='Home'?'paint':event.key==='End'?'mask':mode==='paint'?'mask':'paint';
      setMode(next);if(tool==='pick')setTool('brush');
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
  return createPortal(<div className="image-editor-backdrop" onClick={close}>
    <section className="image-editor" role="dialog" aria-modal="true" aria-labelledby={titleId} ref={dialog} onKeyDown={keyDown} onClick={event=>event.stopPropagation()}>
      <header><div><h2 id={titleId}>{text('涂抹与遮罩','Paint and mask')}</h2><p title={relPath}>{relPath}{info&&` · ${info.width} × ${info.height}`}</p></div><button ref={closeButton} aria-label={text('关闭图片编辑器','Close image editor')} disabled={saving} onClick={close}><X size={17}/></button></header>
      <div className="image-editor-tools">
        <div role="tablist" aria-label={text('编辑模式','Editing mode')}>{(['paint','mask'] as const).map(value=><button key={value} role="tab" aria-selected={mode===value} disabled={saving} onClick={()=>{finishStroke();setMode(value);if(tool==='pick')setTool('brush');}}>{value==='paint'?text('图像涂抹','Image paint'):text('训练遮罩','Training mask')}{(value==='paint'?paint.current?.dirty:mask.current?.dirty)?' *':''}</button>)}</div>
        <button aria-pressed={tool==='brush'} disabled={saving||loading} onClick={()=>setTool('brush')}><Brush size={14}/>{mode==='paint'?text('画笔','Brush'):text('参与训练','Include')}</button>
        <button aria-pressed={tool==='erase'} disabled={saving||loading} onClick={()=>setTool('erase')}><Eraser size={14}/>{mode==='paint'?text('擦回原图','Erase paint'):text('忽略区域','Ignore')}</button>
        <button aria-pressed={tool==='pan'} disabled={saving||loading} onClick={()=>setTool('pan')}><Hand size={14}/>{text('移动','Pan')}</button>
        {mode==='paint'&&<><label>{text('颜色','Color')}<input type="color" aria-label={text('涂抹颜色','Paint color')} value={color} disabled={saving||loading} onChange={event=>setColor(event.target.value)}/></label><button aria-pressed={tool==='pick'} disabled={saving||loading} onClick={()=>setTool('pick')}><Pipette size={14}/>{text('取色','Pick color')}</button></>}
        <button aria-label={text('撤销','Undo')} disabled={saving||loading||!current?.canUndo} onClick={()=>history()}><Undo2 size={14}/></button><button aria-label={text('重做','Redo')} disabled={saving||loading||!current?.canRedo} onClick={()=>history(true)}><Redo2 size={14}/></button>
        {mode==='paint'?<button disabled={saving||loading} onClick={()=>{paint.current?.apply({kind:'clear'});setMessage('');redraw();}}>{text('清除本次涂抹','Clear session paint')}</button>:<><button disabled={saving||loading} onClick={()=>operation({kind:'fill',value:255})}>{text('全白','All white')}</button><button disabled={saving||loading} onClick={()=>operation({kind:'fill',value:0})}>{text('全黑','All black')}</button><button disabled={saving||loading} onClick={()=>operation({kind:'invert'})}>{text('反转','Invert')}</button></>}
      </div>
      <div className="image-editor-tools"><label>{text('笔刷直径','Brush diameter')}<input aria-label={text('笔刷直径','Brush diameter')} type="range" min={1} max={1024} value={diameter} disabled={saving} onChange={event=>setDiameter(Number(event.target.value))}/><input aria-label={text('笔刷像素','Brush pixels')} type="number" min={1} max={1024} value={diameter} disabled={saving} onChange={event=>setDiameter(Math.max(1,Math.min(1024,Number(event.target.value)||1)))}/>px</label>
        {mode==='mask'&&<><label>{text('叠加透明度','Overlay opacity')}<input aria-label={text('叠加透明度','Overlay opacity')} type="range" min={0} max={1} step={.05} value={opacity} onChange={event=>setOpacity(Number(event.target.value))}/></label><label><input type="checkbox" checked={maskOnly} onChange={event=>setMaskOnly(event.target.checked)}/>{text('仅看遮罩','Mask only')}</label></>}
        <button aria-label={text('缩小','Zoom out')} onClick={()=>setZoom(value=>Math.max(.25,value/1.5))}><ZoomOut size={14}/></button><button onClick={()=>{setZoom(1);viewport.current?.scrollTo(0,0);}}>{text('适应','Fit')} {Math.round(scale*100)}%</button><button aria-label={text('放大','Zoom in')} onClick={()=>setZoom(value=>Math.min(8,value*1.5))}><ZoomIn size={14}/></button>
      </div>
      <p className="image-editor-hint">{mode==='paint'?text('涂抹会改变图像颜色，保留尺寸、透明通道和标签。Alt + 点击可取色；擦回原图恢复本次打开时的像素。','Painting changes image colors while preserving dimensions, alpha and captions. Alt + click picks a color. Erase restores pixels from this editing session.'):text('白色参与训练，黑色忽略；红色叠加表示忽略区域。原图没有遮罩时使用 Alpha，否则全图参与。','White participates in training, black is ignored; red indicates ignored areas. Without a sidecar mask, alpha is used, or the whole image participates.')}</p>
      {error&&<p className="image-editor-status" role="alert">{error} {requiresReopen?<><span>{text('图片身份或文件已变化，请刷新列表后重新打开。','The image identity or files changed. Refresh the list and reopen the image.')}</span><button disabled={saving} onClick={closeAndRefresh}>{text('关闭并刷新列表','Close and refresh list')}</button></>:<button disabled={saving} onClick={()=>{if(!dirty()||window.confirm(text('重新读取会放弃未保存修改，继续？','Reload and discard unsaved changes?'))){setReload(value=>value+1);}}}>{text('重新读取','Reload')}</button>}</p>}
      {navigationError&&<p className="image-editor-status" role="alert">{navigationError}</p>}
      {loading?<p role="status" className="image-editor-status">{text('正在读取图片与遮罩…','Loading image and mask…')}</p>:info&&<div ref={viewport} className="image-editor-viewport" data-testid="paint-viewport"><div className="image-editor-stage" style={{width:Math.max(1,info.width*scale),height:Math.max(1,info.height*scale)}}><canvas ref={canvas} aria-label={text('图像与遮罩绘制画布','Image and mask drawing canvas')} tabIndex={0} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={finishStroke} onPointerCancel={finishStroke} onLostPointerCapture={finishStroke} onPointerLeave={()=>{if(cursor.current)cursor.current.style.visibility='hidden';}} style={{cursor:tool==='pan'?'grab':tool==='pick'?'copy':'crosshair'}}/><div ref={cursor} aria-hidden="true" className="image-editor-cursor" style={{visibility:'hidden',width:diameter*scale,height:diameter*scale}}/></div></div>}
      {message&&<p role="status" className="image-editor-status">{message}</p>}
      <footer><small>{dirty()?text('有未保存修改','Unsaved changes'):text('已同步','Up to date')} · {text('Ctrl/Cmd + Z 撤销，Ctrl/Cmd + S 保存','Ctrl/Cmd + Z to undo, Ctrl/Cmd + S to save')}</small><div className="image-editor-footer-actions"><button disabled={saving||loading||requiresReopen||!info?.can_restore} onClick={()=>void restore()}>{text('恢复上次保存前','Restore previous save')}</button><button disabled={saving} onClick={close}>{text('返回图片列表','Back to images')}</button><button className="image-editor-primary" disabled={saving||loading||requiresReopen||!info||!dirty()} onClick={()=>void save()}>{saving?text('保存中…','Saving…'):text('保存修改','Save changes')}</button>{mode==='mask'&&onEnableTraining&&<button disabled={saving||loading||requiresReopen||!info} onClick={()=>void save(true)}>{text('保存并启用遮罩训练','Save and enable masked training')}</button>}</div></footer>
    </section>
  </div>,document.body);
}
