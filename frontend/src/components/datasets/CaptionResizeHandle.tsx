import {useRef} from 'react';

export default function CaptionResizeHandle({label,value,min,max,onChange,horizontal=false,reverse=false}: {
  label:string;value:number;min:number;max:number;onChange:(value:number)=>void;horizontal?:boolean;reverse?:boolean;
}) {
  const drag=useRef<{pointer:number;start:number;value:number}|null>(null);
  const update=(next:number)=>onChange(Math.max(min,Math.min(max,Math.round(next))));
  return <div className={`caption-resize-handle${horizontal?' caption-resize-row':''}`} role="separator" tabIndex={0}
    aria-label={label} aria-orientation={horizontal?'horizontal':'vertical'} aria-valuemin={min} aria-valuemax={max} aria-valuenow={value}
    onPointerDown={event=>{if(event.button!==0)return;event.preventDefault();event.currentTarget.setPointerCapture(event.pointerId);drag.current={pointer:event.pointerId,start:horizontal?event.clientY:event.clientX,value};}}
    onPointerMove={event=>{const start=drag.current;if(start?.pointer===event.pointerId)update(start.value+((horizontal?event.clientY:event.clientX)-start.start)*(reverse?-1:1));}}
    onPointerUp={event=>{drag.current=null;if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId);}}
    onLostPointerCapture={()=>{drag.current=null;}}
    onKeyDown={event=>{const keys=horizontal?['ArrowUp','ArrowDown']:['ArrowLeft','ArrowRight'];if(keys.includes(event.key)){event.preventDefault();update(value+(event.key===keys[0]?-16:16)*(reverse?-1:1));}else if(event.key==='Home'||event.key==='End'){event.preventDefault();update(event.key==='Home'?min:max);}}}/>
}
