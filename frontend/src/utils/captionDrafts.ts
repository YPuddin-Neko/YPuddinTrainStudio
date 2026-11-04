import type { DatasetImage } from '../api/types';
import { captionFieldChanges, captionFieldKey, getCaptionStructure, type CaptionField, type CaptionFieldDraft, type CaptionStructure } from './captionStructure';

export const captionDraftKey=(datasetId:string,relPath:string)=>JSON.stringify([datasetId,relPath]);
export const tagKey=(value:string)=>value.trim().toLowerCase().replace(/ß/g,'ss').replace(/ς/g,'σ');
/** Match training's comma boundaries; escaped commas and weighted groups remain one token. */
export function splitCaptionTags(value:string):string[]{
  const result:string[]=[];let start=0,round=0,square=0;
  for(let i=0;i<value.length;i++){
    const char=value[i];if(char==='\\'){i++;continue;}
    if(char===','&&!round&&!square){const token=value.slice(start,i).trim();if(token)result.push(token);start=i+1;}
    else if(char==='(')round++;else if(char===')')round=Math.max(0,round-1);
    else if(char==='[')square++;else if(char===']')square=Math.max(0,square-1);
  }
  const tail=value.slice(start).trim();if(tail)result.push(tail);return result;
}
export const joinCaptionTags=(tags:string[])=>tags.map(tag=>tag.trim()).filter(Boolean).join(', ');
export const captionIsJson=(image:DatasetImage)=>image.caption_format?.toLowerCase().replace(/^\./,'')==='json';
export interface CaptionDraft { datasetId:string; image:DatasetImage; caption:string; fields:CaptionFieldDraft; additions:Record<string,string> }
export type CaptionDraftMap=Record<string,CaptionDraft>;
export function newCaptionDraft(datasetId:string,image:DatasetImage):CaptionDraft{return {datasetId,image,caption:image.caption_tags??image.caption??'',fields:{},additions:{}};}
export const fieldTokens=(field:CaptionField,value=field.value)=>Array.isArray(value)?[...value]:field.role==='trigger'?(value.trim()?[value.trim()]:[]):splitCaptionTags(value);
export function draftFieldTokens(draft:CaptionDraft,field:CaptionField){return fieldTokens(field,draft.fields[captionFieldKey(field.path)]??field.value);}
export function materializeCaptionDraft(draft:CaptionDraft):CaptionDraft{
  let next={...draft,fields:{...draft.fields},additions:{}};
  if(!captionIsJson(draft.image)){
    if(draft.additions.txt?.trim())next={...next,caption:joinCaptionTags([...splitCaptionTags(next.caption),...splitCaptionTags(draft.additions.txt)])};
  }else for(const field of getCaptionStructure(draft.image)?.fields||[]){
    const key=captionFieldKey(field.path),adding=draft.additions[key];
    if(!adding?.trim()||field.role==='nl')continue;
    const tokens=[...draftFieldTokens(next,field),...splitCaptionTags(adding)];
    next.fields[key]=Array.isArray(field.value)?tokens:joinCaptionTags(tokens);
  }
  return next;
}
export function captionDraftDirty(draft:CaptionDraft){
  return Object.values(draft.additions).some(value=>!!value.trim())||(captionIsJson(draft.image)?captionFieldChanges(getCaptionStructure(draft.image),draft.fields).length>0:draft.caption!==(draft.image.caption_tags??draft.image.caption??''));
}
export function captionEditable(image:DatasetImage){return !image.caption_error&&image.caption_status!=='invalid'&&(!captionIsJson(image)||!!getCaptionStructure(image)?.editable);}
export function allDraftTags(draft:CaptionDraft){
  if(!captionIsJson(draft.image))return captionEditorPresentation(draft.image)==='text'?[]:splitCaptionTags(draft.caption);
  const fields=getCaptionStructure(draft.image)?.fields||[];
  return fields.filter(field=>field.role!=='nl').flatMap(field=>draftFieldTokens(draft,field));
}
export function draftNaturalText(draft:CaptionDraft){
  if(!captionIsJson(draft.image))return draft.caption;
  return (getCaptionStructure(draft.image)?.fields||[]).filter(field=>field.role==='nl').map(field=>draft.fields[captionFieldKey(field.path)]??field.value).join('\n');
}
export interface CaptionBatchEdit { action:'add'|'remove'|'replace'|'deduplicate'; tags?:string[]; from?:string; to?:string; position?:'start'|'end'; field?:string; mode?:'tags'|'text' }
export function editCaptionDraft(input:CaptionDraft,edit:CaptionBatchEdit):CaptionDraft{
  if(!captionIsJson(input.image)&&((edit.mode==='text')!==(captionEditorPresentation(input.image)==='text')))return input;
  const draft=materializeCaptionDraft(input);
  if(!captionEditable(draft.image))return input;
  const structure=getCaptionStructure(draft.image);
  const updateTokens=(tokens:string[],seen?:Set<string>)=>{
    let next=[...tokens];
    if(edit.action==='remove'){const remove=new Set((edit.tags||[]).map(tagKey));next=next.filter(tag=>!remove.has(tagKey(tag)));}
    if(edit.action==='replace')next=next.map(tag=>tagKey(tag)===tagKey(edit.from||'')?edit.to!.trim():tag).filter(Boolean);
    if(edit.action==='add'){
      const existing=new Set(next.map(tagKey)),fresh=(edit.tags||[]).filter(tag=>{const key=tagKey(tag);if(!key||existing.has(key))return false;existing.add(key);return true;});
      next=edit.position==='start'?[...fresh,...next]:[...next,...fresh];
    }
    if(edit.action==='deduplicate'){const known=seen||new Set<string>();next=next.filter(tag=>{const key=tagKey(tag);if(known.has(key))return false;known.add(key);return true;});}
    return next;
  };
  const updateText=(value:string)=>{
    const raw=edit.from??(edit.tags||[]).join(', '),to=edit.to??'';
    if(edit.action==='add'){const added=(edit.tags||[]).join(', ');return edit.position==='start'?added+value:value+added;}
    if(edit.action==='replace'||edit.action==='remove')return raw?value.split(raw).join(edit.action==='remove'?'':to):value;
    return value;
  };
  if(!captionIsJson(draft.image))return {...draft,caption:edit.mode==='text'?updateText(draft.caption):joinCaptionTags(updateTokens(splitCaptionTags(draft.caption)))};
  const fields=(structure?.fields||[]).filter(field=>(edit.mode==='text'?field.role==='nl':field.role!=='nl')&&(!edit.field||edit.field==='all'||captionFieldKey(field.path)===edit.field));
  const target=fields.find(field=>field.role==='tags')||fields.find(field=>Array.isArray(field.value))||fields[0];
  const seen=new Set<string>();
  for(const field of [...fields].sort((a,b)=>Number(b.role==='trigger')-Number(a.role==='trigger'))){
    if(edit.action==='add'&&field!==target)continue;
    const key=captionFieldKey(field.path),value=draft.fields[key]??field.value;
    if(edit.mode==='text')draft.fields[key]=updateText(typeof value==='string'?value:value.join(', '));
    else {const tokens=updateTokens(fieldTokens(field,value),seen);draft.fields[key]=Array.isArray(field.value)?tokens:joinCaptionTags(tokens);}
  }
  return draft;
}
export function captionDraftPayload(input:CaptionDraft){
  const draft=materializeCaptionDraft(input);
  return captionIsJson(draft.image)?{caption_fields:captionFieldChanges(getCaptionStructure(draft.image),draft.fields),caption_revision:getCaptionStructure(draft.image)!.revision}:{caption:draft.caption};
}
export function settledCaptionImage(draft:CaptionDraft,result:{caption:string;caption_structure?:CaptionStructure|null}):DatasetImage{
  return {...draft.image,caption:result.caption,caption_tags:captionIsJson(draft.image)?allDraftTags(draft).join(', '):draft.caption,caption_description:draftNaturalText(draft),caption_structure:result.caption_structure?{...result.caption_structure,fields:result.caption_structure.fields.map(field=>({...field}))}:draft.image.caption_structure,caption_status:result.caption?'captioned':'missing'};
}

/** Presentation only: ambiguous TXT stays editable as its original text. */
export function captionEditorPresentation(image:DatasetImage):'tags'|'text'|'json'{
  if(captionIsJson(image))return 'json';
  const raw=(image.caption_tags??image.caption??'').trim();
  if(!raw)return 'tags';
  if(/[\r\n]/.test(raw)||/[.!?。！？]$/.test(raw))return 'text';
  const tokens=splitCaptionTags(raw);
  const short=(token:string)=>token.length<=70&&token.split(/\s+/).length<=4&&!/[.!?。！？]$/.test(token);
  if(tokens.length>1&&tokens.every(short))return 'tags';
  if(tokens.length===1&&short(tokens[0])&&(/\\,|^\(.*\)$|^\[.*\]$/.test(raw)||raw.split(/\s+/).length<=2&&!/^(?:a|an|the)\s/i.test(raw)))return 'tags';
  return 'text';
}
