import { useRef, useState, type PointerEvent } from 'react';
import { apiUrl } from '../../api/client';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../StudioSelect';

type Rectangle = {x:number;y:number;width:number;height:number};
type Props = {image:{dataset_id:string;hash:string;width:number;height:number;rel_path:string};busy:boolean;onApply:(rectangle:Rectangle)=>void};
type Gesture = {mode:string;start:{x:number;y:number};box:Rectangle};
const bound = (n:number,min:number,max:number) => Math.max(min,Math.min(max,n));
export default function VisualCropEditor({image,busy,onApply}:Props) {
  const text=useWorkspaceText();
  const [box,setBox]=useState<Rectangle>({x:0,y:0,width:image.width,height:image.height});
  const [ratio,setRatio]=useState('free');
  const gesture=useRef<Gesture|null>(null);
  const proportion=ratio==='original'?image.width/image.height:ratio==='free'?0:Number(ratio);
  const normalize=(next:Rectangle)=>{const x=bound(Math.round(next.x),0,image.width-1);const y=bound(Math.round(next.y),0,image.height-1);return {x,y,width:bound(Math.round(next.width),1,image.width-x),height:bound(Math.round(next.height),1,image.height-y)};};
  const position=(event:PointerEvent<SVGSVGElement>)=>{const rect=event.currentTarget.getBoundingClientRect();return {x:bound((event.clientX-rect.left)/rect.width*image.width,0,image.width),y:bound((event.clientY-rect.top)/rect.height*image.height,0,image.height)};};
  const anchored=(anchor:{x:number;y:number},end:{x:number;y:number})=>{
    const sx=end.x>=anchor.x?1:-1,sy=end.y>=anchor.y?1:-1;
    let width=Math.abs(end.x-anchor.x),height=Math.abs(end.y-anchor.y);
    if(proportion){width=Math.max(width,height*proportion);height=width/proportion;const scale=Math.min(1,(sx>0?image.width-anchor.x:anchor.x)/Math.max(width,1),(sy>0?image.height-anchor.y:anchor.y)/Math.max(height,1));width*=scale;height*=scale;}
    return normalize({x:sx>0?anchor.x:anchor.x-width,y:sy>0?anchor.y:anchor.y-height,width,height});
  };
  const down=(event:PointerEvent<SVGSVGElement>)=>{if(busy||event.button!==0)return;event.preventDefault();const mode=(event.target as SVGElement).getAttribute('data-crop-handle')||'draw';gesture.current={mode,start:position(event),box};event.currentTarget.setPointerCapture?.(event.pointerId);};
  const move=(event:PointerEvent<SVGSVGElement>)=>{const current=gesture.current;if(!current||busy)return;const point=position(event),before=current.box;
    if(current.mode==='move')setBox({...before,x:Math.round(bound(before.x+point.x-current.start.x,0,image.width-before.width)),y:Math.round(bound(before.y+point.y-current.start.y,0,image.height-before.height))});
    else {const anchor=current.mode==='draw'?current.start:{x:current.mode.includes('w')?before.x+before.width:before.x,y:current.mode.includes('n')?before.y+before.height:before.y};setBox(anchored(anchor,point));}
  };
  const changeRatio=(value:string)=>{setRatio(value);const r=value==='original'?image.width/image.height:value==='free'?0:Number(value);if(!r)return;const width=Math.min(box.width,box.height*r);const height=width/r;setBox(normalize({x:box.x+(box.width-width)/2,y:box.y+(box.height-height)/2,width,height}));};
  const edit=(field:keyof Rectangle,value:number)=>{let next={...box,[field]:value};if(proportion){next.x=bound(next.x,0,image.width-1);next.y=bound(next.y,0,image.height-1);const wanted=field==='height'?value*proportion:next.width;const width=Math.min(wanted,image.width-next.x,(image.height-next.y)*proportion);next={...next,width,height:width/proportion};}setBox(normalize(next));};
  const handleSize=Math.max(8,Math.min(image.width,image.height)/35);
  return <div className="pipeline-visual-crop" data-testid="visual-crop-editor"><div className="pipeline-crop-canvas" style={{aspectRatio:`${image.width}/${image.height}`,maxWidth:`${Math.min(700,300*image.width/image.height)}px`}}><svg role="img" aria-label={text('可拖动裁剪区域','Draggable crop area')} viewBox={`0 0 ${image.width} ${image.height}`} onPointerDown={down} onPointerMove={move} onPointerUp={()=>{gesture.current=null;}} onPointerCancel={()=>{gesture.current=null;}}>
    <image href={apiUrl(`/datasets/${image.dataset_id}/images/${image.hash}/file`)} width={image.width} height={image.height}/>
    <path d={`M0 0H${image.width}V${image.height}H0Z M${box.x} ${box.y}V${box.y+box.height}H${box.x+box.width}V${box.y}Z`} fill="black" fillOpacity=".55" fillRule="evenodd"/>
    <rect data-crop-handle="move" x={box.x} y={box.y} width={box.width} height={box.height} fill="transparent" stroke="white" strokeWidth={Math.max(1,handleSize/8)} style={{cursor:'move'}}/>
    {[['nw',box.x,box.y],['ne',box.x+box.width,box.y],['sw',box.x,box.y+box.height],['se',box.x+box.width,box.y+box.height]].map(([name,x,y])=><rect key={name} data-crop-handle={name} x={Number(x)-handleSize/2} y={Number(y)-handleSize/2} width={handleSize} height={handleSize} fill="white" stroke="#2563eb" strokeWidth={handleSize/7} style={{cursor:`${name==='ne'||name==='sw'?'nesw':'nwse'}-resize`}}/>)}
  </svg></div><div className="pipeline-crop-controls"><strong>{image.rel_path}</strong><p>{text('拖动矩形移动，拖动四角调整；也可在图上重新框选。','Drag the rectangle to move it, drag corners to resize, or draw a new selection.')}</p><label>{text('裁剪比例','Crop ratio')}<StudioSelect aria-label={text('裁剪比例','Crop ratio')} value={ratio} onValueChange={changeRatio} disabled={busy} options={[
    {value:'free',label:text('自由比例','Free')},{value:'original',label:text('原图比例','Original')},...Object.entries({'1':'1:1','0.6666666667':'2:3','0.75':'3:4','1.3333333333':'4:3','1.5':'3:2','1.7777777778':'16:9'}).map(([value,label])=>({value,label}))
  ]}/></label><div className="pipeline-crop-coordinates">{(['x','y','width','height'] as const).map(field=><label key={field}>{field==='width'?text('裁剪宽度','Crop width'):field==='height'?text('裁剪高度','Crop height'):field.toUpperCase()}<input aria-label={field==='width'?text('裁剪宽度','Crop width'):field==='height'?text('裁剪高度','Crop height'):`${text('裁剪','Crop')} ${field.toUpperCase()}`} type="number" min={field==='x'||field==='y'?0:1} max={field==='x'?image.width-1:field==='y'?image.height-1:field==='width'?image.width-box.x:image.height-box.y} value={box[field]} disabled={busy} onChange={event=>edit(field,Number(event.target.value))}/></label>)}</div><output>{text('实际输出','Output')}: {box.width} × {box.height} px</output><small>{text('按原图像素裁切，不额外缩放。遮罩同步裁切，原文件可恢复。','Crop original pixels without resizing. Masks follow the same rectangle; originals can be restored.')}</small><div><button disabled={busy} onClick={()=>{setBox({x:0,y:0,width:image.width,height:image.height});setRatio('free');}}>{text('重置区域','Reset selection')}</button><button className="pipeline-primary" disabled={busy} onClick={()=>onApply(box)}>{text('应用此裁剪','Apply this crop')}</button></div></div></div>;
}
