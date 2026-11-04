import React from 'react';
import { Plus, X } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';
import { splitCaptionTags } from '../../utils/captionDrafts';
import type { CaptionTagDragBinding } from './useCaptionTagDrag';
import './caption-tag.css';

export default function CaptionTagList({values,onChange,adding,onAdding,disabled=false,label,drag}:{values:string[];onChange:(values:string[])=>void;adding:string;onAdding:(value:string)=>void;disabled?:boolean;label:string;drag?:CaptionTagDragBinding}){
  const text=useWorkspaceText();
  const [addingOpen,setAddingOpen]=React.useState(false);
  const [editing,setEditing]=React.useState<{index:number;original:string}|null>(null);
  const add=()=>{const tokens=splitCaptionTags(adding);if(tokens.length){onChange([...values,...tokens]);onAdding('');}};
  return <div className="caption-chip-list" data-caption-tag-field={drag?.field} aria-label={label}>
    {values.map((tag,index)=><span className={`caption-tag${drag?.dragging===index?' is-dragging':''}${drag?.insertionIndex===index?' is-drop-target':''}`} data-caption-tag-index={index} key={index} onPointerDown={event=>drag?.onPointerDown(event,index,tag)}>
      {editing?.index===index?<input className="caption-tag-control" autoFocus aria-label={text(`编辑标签：${tag}`,`Edit tag: ${tag}`)} size={Math.min(28,Math.max(6,tag.length+1))} value={tag} disabled={disabled} onChange={event=>onChange(values.map((value,i)=>i===index?event.target.value:value))} onBlur={()=>setEditing(null)} onKeyDown={event=>{if(event.key==='Enter'){event.preventDefault();setEditing(null);}if(event.key==='Escape'){onChange(values.map((value,i)=>i===index?editing.original:value));setEditing(null);}}}/>:<span className="caption-tag-control caption-tag-text" role="button" tabIndex={disabled?-1:0} aria-label={text(`编辑标签：${tag}`,`Edit tag: ${tag}`)} title={text('双击编辑；Alt + 左右方向键移动','Double-click to edit; Alt + Left/Right to move')} onDoubleClick={()=>{if(!disabled)setEditing({index,original:tag});}} onKeyDown={event=>{if(disabled)return;if(event.key==='Enter'){setEditing({index,original:tag});event.preventDefault();}if(event.altKey&&['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();const next=[...values],at=Math.max(0,Math.min(values.length-1,index+(event.key==='ArrowLeft'?-1:1)));next.splice(index,1);next.splice(at,0,tag);onChange(next);}}}>{tag}</span>}
      <button type="button" className="caption-tag-control caption-tag-remove" disabled={disabled} aria-label={text(`删除标签：${tag}`,`Remove tag: ${tag}`)} onClick={()=>onChange(values.filter((_,i)=>i!==index))}><X size={12}/></button>
    </span>)}
    <div className="caption-chip-add">{addingOpen||adding?<><input autoFocus aria-label={text(`添加${label}标签`,`Add ${label} tags`)} placeholder={text('添加标签…','Add tags…')} value={adding} disabled={disabled} onBlur={()=>{if(!adding.trim())setAddingOpen(false);}} onChange={event=>onAdding(event.target.value)} onKeyDown={event=>{if(event.key==='Enter'){event.preventDefault();add();}}}/><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={text(`添加${label}标签`,`Add ${label} tags`)} disabled={disabled||!adding.trim()} onClick={add}><Plus size={14}/></button></>:<button type="button" className={values.length?"caption-tag-add":"caption-tag-empty"} disabled={disabled} aria-label={text(`添加${label}标签`,`Add ${label} tags`)} onClick={()=>setAddingOpen(true)}>{values.length?<Plus size={13}/>:text('点击添加标签','Click to add tags')}</button>}</div>
  </div>;
}
