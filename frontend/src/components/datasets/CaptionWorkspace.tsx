import React from 'react';
import { UNSAFE_DataRouterContext, useLocation, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowDownWideNarrow, CheckCircle2, ChevronLeft, ChevronRight, Code, CopyX, FileText, Filter, Folder, Image as ImageIcon, ListPlus, Loader2, MinusCircle, Pencil, PlusCircle, RefreshCw, Save, Search, Tags, Trash2, X } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { CaptionFile, DatasetImage, DatasetInfo } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { allDraftTags, captionEditorPresentation, captionDraftDirty, captionDraftKey, captionDraftPayload, captionEditable, captionFileTarget, captionIsJson, draftFieldTokens, draftNaturalText, editCaptionDraft, joinCaptionTags, materializeCaptionDraft, newCaptionDraft, settledCaptionImage, splitCaptionTags, tagKey, type CaptionBatchEdit, type CaptionDraft, type CaptionDraftMap } from '../../utils/captionDrafts';
import { captionFieldKey, captionMetadata, captionStructureLabel, gbcRegions, getCaptionStructure, isSceneCaption, type CaptionField, type CaptionStructure } from '../../utils/captionStructure';
import StudioSelect from '../StudioSelect';
import Dialog from '../Dialog';
import DatasetNavigationGuard from './DatasetNavigationGuard';
import CaptionResizeHandle from './CaptionResizeHandle';
import CaptionTagList from './CaptionTagList';
import CaptionGbcRegions from './CaptionGbcRegions';
import StructuredCaptionEditor from './StructuredCaptionEditor';
import { useCaptionTagDrag } from './useCaptionTagDrag';
import { useDatasetImageCatalog } from './useDatasetImageCatalog';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { SlidingIndicator } from '../motion';
import { LazyImage, LoadingNote } from '../Loading';
import './caption-workspace.css';
import './caption-tag.css';

type StatsOrder='frequency_desc'|'frequency_asc'|'name_asc'|'name_desc';
type BatchScope='current'|'all';
type BatchDialog=CaptionBatchEdit&{scope:BatchScope;raw:string;tokens?:string[]};

function draftCaptionStatus(draft:CaptionDraft,tags:string[],prose:string):'captioned'|'missing'|'invalid'{
  const image=draft.image;
  if(draft.captionFile==='gbc'||(captionIsJson(image)&&!getCaptionStructure(image)&&image.caption_status==='captioned'&&!image.caption_error))return image.caption_status||'missing';
  if(isSceneCaption(getCaptionStructure(image))&&!image.caption_error&&image.caption_status!=='invalid')return image.caption_status==='captioned'?'captioned':'missing';
  if(!captionEditable(image))return 'invalid';
  return tags.length||prose.trim()?'captioned':'missing';
}
const EMPTY:DatasetImage[]=[];
const PAGE_SIZE=30;
const FILE_CHOICES=['txt','json','gbc'] as const;
const FILE_PREFERENCES='studio.caption.files';
const nameOf=(path:string)=>path.replace(/\\/g,'/').split('/').filter(Boolean).pop()||path;
const FIELD_LABELS:Record<string,[string,string]>={quality:['画面质量','Quality'],count:['人物数量','Subject count'],character:['角色','Character'],character_name:['角色名称','Character name'],character_variant:['角色形态','Character variant'],character_full:['角色完整名称','Full character name'],series:['作品 / 系列','Series'],artist:['画师','Artist'],appearance:['外观与服装','Appearance and clothing'],tags:['画面内容','Image content'],environment:['环境与背景','Environment and background'],nl:['自然语言描述','Natural-language description'],trigger:['触发词','Trigger word']};

export default function CaptionWorkspace({projectId,versionId,initialDatasetId='',readOnly=false,onChanged}:{projectId:string;versionId:string;initialDatasetId?:string;readOnly?:boolean;onChanged?:()=>void}){
  const text=useWorkspaceText();
  const client=useQueryClient();
  const navigate=useNavigate();
  const location=useLocation();
  const dataRouter=!!React.useContext(UNSAFE_DataRouterContext);
  const workspace=React.useRef<HTMLElement>(null),content=React.useRef<HTMLDivElement>(null),grid=React.useRef<HTMLDivElement>(null);
  const [datasetId,setDatasetId]=React.useState(initialDatasetId);
  const [galleryWidth,setGalleryWidth]=React.useState(220);
  const [rightWidths,setRightWidths]=React.useState<Partial<Record<'tags'|'json'|'text',number>>>({});
  const [previewHeights,setPreviewHeights]=React.useState<Partial<Record<'tags'|'json',number>>>({});
  const [view,setView]=React.useState({did:'',page:1,search:'',status:'',active:''});
  const [showRegionBoxes,setShowRegionBoxes]=React.useState(true);
  const [previewFallback,setPreviewFallback]=React.useState('');
  const [filePreferences,setFilePreferences]=React.useState<Record<string,CaptionFile>>(()=>{
    try{return JSON.parse(localStorage.getItem(FILE_PREFERENCES)||'{}')||{};}catch{return {};}
  });
  const [drafts,setDrafts]=React.useState<CaptionDraftMap>({});
  const draftsRef=React.useRef(drafts);draftsRef.current=drafts;
  const [saving,setSaving]=React.useState(false),[message,setMessage]=React.useState(''),[failures,setFailures]=React.useState<Record<string,string>>({});
  const [statsSearch,setStatsSearch]=React.useState(''),[statsLimit,setStatsLimit]=React.useState(100);
  const [statsOrder,setStatsOrder]=React.useState<StatsOrder>('frequency_desc'),[commonOnly,setCommonOnly]=React.useState(false);
  const [selectedTags,setSelectedTags]=React.useState(new Set<string>()),[tagFilter,setTagFilter]=React.useState(false);
  const statsAnchor=React.useRef('');
  const [batch,setBatch]=React.useState<BatchDialog|null>(null);
  const [leaveOpen,setLeaveOpen]=React.useState(false);
  const leavePending=React.useRef<((leave:boolean)=>void)|null>(null);
  const mounted=React.useRef(true);
  const datasets=useQuery<DatasetInfo[]>({queryKey:['caption-datasets',projectId,versionId],refetchOnWindowFocus:false,refetchOnReconnect:false,
    refetchInterval:query=>query.state.data?.some(item=>item.index_status==='indexing')?1500:false,
    queryFn:({signal})=>apiClient.get<DatasetInfo[]>(`/projects/${projectId}/datasets`,{params:{version_id:versionId,include_cache:false},signal,silent:true})});
  const source=datasets.data?.find(item=>item.source.id===datasetId)||datasets.data?.[0];
  const did=source?.source.id||'';
  const requestedFile=filePreferences[did];
  const captionFile:CaptionFile=FILE_CHOICES.some(choice=>choice===requestedFile)?requestedFile:'default';
  const graphMode=captionFile==='gbc';
  const catalog=useDatasetImageCatalog(did||undefined,'filename',false,captionFile);
  const catalogItems=catalog.data||EMPTY;
  const items=catalog.isPlaceholderData?EMPTY:catalogItems;
  const availableFiles=React.useMemo(()=>new Set(catalogItems.flatMap(item=>item.caption_files||[])),[catalogItems]);
  const rememberFile=React.useCallback((id:string,file:CaptionFile)=>setFilePreferences(previous=>{
    const next={...previous,[id]:file};
    try{localStorage.setItem(FILE_PREFERENCES,JSON.stringify(next));}catch{/* applies for this session */}
    return next;
  }),[]);
  React.useEffect(()=>{
    if(catalog.data&&!catalog.isPlaceholderData&&!catalog.isFetching&&captionFile!=='default'&&!availableFiles.has(captionFile))rememberFile(did,'default');
  },[availableFiles,captionFile,catalog.data,catalog.isFetching,catalog.isPlaceholderData,did,rememberFile]);
  const configuredFile=source?.source.caption_ext?.toLowerCase().replace(/^\./,'');
  const showFileChoice=availableFiles.size>1||availableFiles.has('gbc')||captionFile!=='default'||!!configuredFile&&configuredFile!=='auto'&&[...availableFiles].some(file=>file!==configuredFile);
  const fileChoices=FILE_CHOICES.filter(file=>availableFiles.has(file)||file===captionFile&&(catalog.isPending||catalog.isPlaceholderData));
  const currentView=view.did===did?view:{did,page:1,search:'',status:'',active:''};
  const changeView=(patch:Partial<typeof view>)=>setView({...currentView,...patch});
  const analysis=React.useMemo(()=>{
    const datasetCounts={all:items.length,captioned:0,missing:0,invalid:0};
    const frequency=new Map<string,{tag:string;count:number}>();
    let taggedCount=0;
    const byPath=new Map(items.map(item=>{
      const target=captionFileTarget(item,captionFile),key=captionDraftKey(did,item.rel_path,target);
      const draft=drafts[key]||newCaptionDraft(did,item,target);
      const tags=graphMode?[]:allDraftTags(draft),prose=draftNaturalText(draft),status=draftCaptionStatus(draft,tags,prose);
      const tagsByKey=new Map<string,string>();
      for(const tag of tags){const key=tagKey(tag);if(!tagsByKey.has(key))tagsByKey.set(key,tag);}
      datasetCounts[status]++;
      if(tags.length)taggedCount++;
      if(status!=='invalid')for(const [key,tag] of tagsByKey){const entry=frequency.get(key);frequency.set(key,{tag:entry?.tag||tag,count:(entry?.count||0)+1});}
      const graphText=graphMode||isSceneCaption(getCaptionStructure(item))||captionIsJson(item)&&!getCaptionStructure(item)?item.caption||'':'';
      return [item.rel_path,{item,draft,tags,status,tagsByKey,searchText:`${item.rel_path} ${tags.join(', ')} ${prose} ${graphText}`.toLocaleLowerCase()}] as const;
    }));
    return {byPath,datasetCounts,frequency,taggedCount};
  },[items,did,captionFile,graphMode,drafts]);
  const {datasetCounts,frequency,taggedCount}=analysis;
  const getDraft=(item:DatasetImage)=>analysis.byPath.get(item.rel_path)!.draft;
  const stats=React.useMemo(()=>[...frequency.values()].filter(item=>(!commonOnly||item.count===taggedCount)&&item.tag.toLocaleLowerCase().includes(statsSearch.trim().toLocaleLowerCase())).sort((a,b)=>(statsOrder.startsWith('frequency')?a.count-b.count:a.tag.localeCompare(b.tag))*(statsOrder.endsWith('asc')?1:-1)),[frequency,commonOnly,taggedCount,statsSearch,statsOrder]);
  const filtered=React.useMemo(()=>{
    const search=currentView.search.trim().toLocaleLowerCase();
    const required=tagFilter?[...selectedTags].map(tagKey):[];
    return [...analysis.byPath.values()].filter(row=>(!currentView.status||currentView.status===row.status)&&required.every(key=>row.tagsByKey.has(key))&&row.searchText.includes(search)).map(row=>row.item);
  },[analysis,currentView.search,currentView.status,tagFilter,selectedTags]);
  const pages=Math.max(1,Math.ceil(filtered.length/PAGE_SIZE)),page=Math.min(currentView.page,pages),visible=filtered.slice((page-1)*PAGE_SIZE,page*PAGE_SIZE);
  const image=analysis.byPath.get(currentView.active)?.item||visible[0]||(catalog.isPlaceholderData?(catalogItems.find(item=>item.rel_path===currentView.active)||catalogItems[0]):undefined);
  const filteredIndex=React.useMemo(()=>new Map(filtered.map((item,index)=>[item.rel_path,index])),[filtered]);
  const imageIndex=image?filteredIndex.get(image.rel_path)??-1:-1;
  const previewKey=image?`${did}/${image.hash}`:'';
  const originalImage=image?apiUrl(`/datasets/${did}/images/${image.hash}/file`):'';
  const previewSize=Math.min(2048,1024*Math.ceil(window.devicePixelRatio||1));
  const previewImage=image&&previewFallback!==previewKey?apiUrl(`/datasets/${did}/images/${image.hash}/thumb?size=${previewSize}&orient=true`):originalImage;
  const activeDraft=image&&!catalog.isPlaceholderData?getDraft(image):null;
  const key=activeDraft?captionDraftKey(did,activeDraft.image.rel_path,activeDraft.captionFile):'';
  const listStructure=getCaptionStructure(activeDraft?.image);
  const needsDocument=!!activeDraft&&(graphMode?!!activeDraft.image.caption_files?.includes('gbc'):!listStructure&&captionIsJson(activeDraft.image)&&activeDraft.image.caption_status==='captioned'&&!activeDraft.image.caption_error);
  const documentQuery=useQuery({queryKey:['caption-document',did,image?.hash,image?.rel_path,captionFile],enabled:needsDocument,gcTime:0,retry:false,refetchOnWindowFocus:false,refetchOnReconnect:false,
    queryFn:async({signal})=>getCaptionStructure(await apiClient.get<{caption_structure?:CaptionStructure|null}>(`/datasets/${did}/images/${image!.hash}/caption`,{params:{rel_path:image!.rel_path,caption_file:captionFile},signal,silent:true}))});
  const structure=documentQuery.data??listStructure;
  const sceneCaption=graphMode||needsDocument||isSceneCaption(structure);
  const regions=gbcRegions(structure);
  const isJson=!!image&&captionIsJson(image);
  const fields=structure?.fields||[];
  const dirtyEntries=React.useMemo(()=>Object.entries(drafts).filter(([,draft])=>captionDraftDirty(draft)),[drafts]);
  const dirtyCount=dirtyEntries.length;
  const activeDirty=!!activeDraft&&captionDraftDirty(activeDraft);
  const locked=readOnly||saving||catalog.isFetching||sceneCaption||!!image&&!captionEditable(image);
  const effectiveMode=graphMode?'text':activeDraft?captionEditorPresentation(activeDraft.image):'text';
  const statsWidth=rightWidths[effectiveMode]??(effectiveMode==='text'?320:effectiveMode==='json'?220:250);
  const previewHeight=effectiveMode==='text'?undefined:previewHeights[effectiveMode];
  const setPreviewHeight=(height:number)=>{if(effectiveMode!=='text')setPreviewHeights(previous=>({...previous,[effectiveMode]:height}));};
  const setStatsWidth=(width:number)=>setRightWidths(previous=>({...previous,[effectiveMode]:width}));
  const statsOrders=[{value:'frequency_desc',label:text('频次高→低','Frequency: high to low')},{value:'frequency_asc',label:text('频次低→高','Frequency: low to high')},{value:'name_asc',label:text('名称 A→Z','Name: A to Z')},{value:'name_desc',label:text('名称 Z→A','Name: Z to A')}];
  const statsOrderLabel=statsOrders.find(option=>option.value===statsOrder)!.label;
  const editHint=selectedTags.size===1?text('编辑选中标签','Edit selected tag'):text('手动填写原标签和新标签','Enter the old and new tags');
  const fieldLabel=(field:CaptionField)=>text(...(FIELD_LABELS[field.role]||[field.role,field.role]))+(field.path[0]==='from_path'?text(' · 来自目录',' · From folder'):'');
  const updateActive=(patch:Partial<CaptionDraft>)=>{
    if(!activeDraft||locked)return;
    setDrafts(previous=>({...previous,[key]:{...(previous[key]||activeDraft),...patch}}));setMessage('');
    setFailures(previous=>{const next={...previous};delete next[key];return next;});
  };
  const changeField=(field:CaptionField,tags:string[])=>{if(!activeDraft)return;const fieldKey=captionFieldKey(field.path);updateActive({fields:{...activeDraft.fields,[fieldKey]:Array.isArray(field.value)?tags:joinCaptionTags(tags)}});};
  const tagDrag=useCaptionTagDrag({scope:key,revision:activeDraft,disabled:locked,onDrop:(from,to)=>{
    if(!activeDraft)return;
    const next={...activeDraft,fields:{...activeDraft.fields}},fromField=fields.find(field=>captionFieldKey(field.path)===from.field),toField=fields.find(field=>captionFieldKey(field.path)===to.field);
    if(isJson&&(!fromField||!toField||fromField.role==='nl'||toField.role==='nl'))return;
    const sourceTags=isJson?draftFieldTokens(next,fromField!):splitCaptionTags(next.caption),targetTags=from.field===to.field?sourceTags:isJson?draftFieldTokens(next,toField!):sourceTags;
    if(sourceTags[from.index]!==from.value)return;
    const [value]=sourceTags.splice(from.index,1),position=to.index-(from.field===to.field&&from.index<to.index?1:0);targetTags.splice(Math.max(0,position),0,value);
    if(isJson){next.fields[from.field]=Array.isArray(fromField!.value)?sourceTags:joinCaptionTags(sourceTags);next.fields[to.field]=Array.isArray(toField!.value)?targetTags:joinCaptionTags(targetTags);}else next.caption=joinCaptionTags(targetTags);
    updateActive(next);
  }});
  React.useLayoutEffect(()=>{
    const root=workspace.current,panel=content.current;if(!root||!panel)return;let frame=0;
    const update=()=>{frame=0;let top=panel.getBoundingClientRect().top+window.scrollY;for(let parent=panel.parentElement;parent&&parent!==document.body;parent=parent.parentElement)top+=parent.scrollTop;root.style.setProperty('--caption-editor-height',`${Math.max(440,Math.floor(innerHeight-top-20))}px`);};
    const schedule=()=>{if(!frame)frame=requestAnimationFrame(update);};const observer=new ResizeObserver(schedule);observer.observe(root);window.addEventListener('resize',schedule);update();return()=>{observer.disconnect();window.removeEventListener('resize',schedule);if(frame)cancelAnimationFrame(frame);};
  },[did,items.length]);
  React.useEffect(()=>{if(grid.current)grid.current.scrollTop=0;},[page,did,currentView.search,currentView.status,tagFilter]);
  React.useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;leavePending.current?.(false);};},[]);
  const saveDrafts=async(onlyCurrent=false)=>{
    if(readOnly||saving)return false;
    const pending=Object.entries(draftsRef.current).filter(([draftKey,draft])=>captionDraftDirty(draft)&&(!onlyCurrent||draftKey===key));
    if(!pending.length)return true;
    setSaving(true);setMessage('');setFailures({});let savedCount=0;const errors:Record<string,string>={};
    const updatedDatasets=new Set<string>();
    try{
      for(const [draftKey,original] of pending){
        const draft=materializeCaptionDraft(original);
        try{
          const params={rel_path:draft.image.rel_path,...(draft.captionFile!=='default'?{caption_file:draft.captionFile}:{})};
          const result=await apiClient.put<{caption:string;caption_structure?:CaptionStructure|null}>(`/datasets/${draft.datasetId}/images/${draft.image.hash}/caption`,captionDraftPayload(draft),{params,silent:true});
          const nextImage=settledCaptionImage(draft,result);savedCount++;
          updatedDatasets.add(draft.datasetId);
          for(const [queryKey,rows] of client.getQueriesData<DatasetImage[]>({queryKey:['dataset-image-catalog',draft.datasetId]})){
            const file=queryKey[4] as CaptionFile|undefined;
            if(!rows||!file||file==='gbc')continue;
            const match=rows.findIndex(item=>item.rel_path===draft.image.rel_path&&captionFileTarget(item,file)===draft.captionFile);
            if(match>=0)client.setQueryData(queryKey,rows.map((item,index)=>index===match?nextImage:item));
          }
          setDrafts(previous=>{if(previous[draftKey]!==original)return previous;const next={...previous};delete next[draftKey];return next;});
          void client.invalidateQueries({queryKey:['caption-images',projectId,versionId,draft.datasetId]});
        }catch(error){errors[draftKey]=formatApiError(error);}
      }
      if(mounted.current){setFailures(errors);setMessage(text(`已保存 ${savedCount} 张${Object.keys(errors).length?`，${Object.keys(errors).length} 张失败，草稿已保留`:''}`,`Saved ${savedCount} images${Object.keys(errors).length?`; ${Object.keys(errors).length} failed, drafts retained`:''}`));if(savedCount){for(const id of updatedDatasets)void client.invalidateQueries({queryKey:['dataset-image-catalog',id]});void datasets.refetch();onChanged?.();}}
      return !Object.keys(errors).length;
    }finally{if(mounted.current)setSaving(false);}
  };
  const beforeLeave=()=>{
    if(saving)return Promise.reject(new Error(''));
    if(!dirtyCount)return Promise.resolve();
    return new Promise<void>((resolve,reject)=>{leavePending.current=leave=>leave?resolve():reject(new Error(''));setLeaveOpen(true);});
  };
  const finishLeave=(leave:boolean)=>{if(leave)setDrafts({});setLeaveOpen(false);leavePending.current?.(leave);leavePending.current=null;};
  const latestLeave=React.useRef({dirtyCount,saving,beforeLeave});latestLeave.current={dirtyCount,saving,beforeLeave};
  React.useEffect(()=>{
    const unload=(event:BeforeUnloadEvent)=>{if(latestLeave.current.dirtyCount||latestLeave.current.saving){event.preventDefault();event.returnValue='';}};
    const click=(event:MouseEvent)=>{
      if(dataRouter||!latestLeave.current.dirtyCount||event.defaultPrevented||event.button||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
      const anchor=event.target instanceof Element?event.target.closest<HTMLAnchorElement>('a[href]'):null;if(!anchor||anchor.hasAttribute('download')||anchor.target&&anchor.target!=='_self')return;
      const next=new URL(anchor.href,window.location.origin);if(next.origin!==window.location.origin)return;event.preventDefault();event.stopPropagation();
      void latestLeave.current.beforeLeave().then(()=>navigate(`${next.pathname}${next.search}${next.hash}`,next.pathname.startsWith('/settings')?{state:{backgroundLocation:location}}:undefined)).catch(()=>{});
    };
    window.addEventListener('beforeunload',unload);document.addEventListener('click',click,true);return()=>{window.removeEventListener('beforeunload',unload);document.removeEventListener('click',click,true);};
  },[dataRouter,navigate,location]);
  useEventStream<{dataset_id?:string}>(EVENT_TYPES.DATASET_CHANGED,event=>{if(event.dataset_id===did&&!dirtyCount&&!saving){void catalog.refetch();void datasets.refetch();if(needsDocument)void documentQuery.refetch();}});
  const selectImage=(item:DatasetImage)=>changeView({active:item.rel_path});
  const moveImage=(delta:number)=>{const next=filtered[imageIndex+delta];if(next)changeView({active:next.rel_path,page:Math.floor((imageIndex+delta)/PAGE_SIZE)+1});};
  const clearTagSelection=()=>{setSelectedTags(new Set());setTagFilter(false);setStatsSearch('');};
  const changeSource=(id:string)=>{setDatasetId(id);clearTagSelection();};
  const changeFile=(file:CaptionFile)=>{if(image)changeView({active:image.rel_path});rememberFile(did,file);clearTagSelection();setShowRegionBoxes(true);};
  const refresh=()=>{void datasets.refetch();void catalog.refetch();if(needsDocument)void documentQuery.refetch();};
  const openBatch=(action:CaptionBatchEdit['action'],options:Partial<BatchDialog>={})=>setBatch({action,scope:'all',raw:'',from:'',to:'',position:'start',field:'all',mode:effectiveMode==='text'?'text':'tags',...options});
  const batchTargets=batch?(batch.scope==='current'?(activeDraft?[activeDraft.image]:[]):items):[];
  const batchEdit:CaptionBatchEdit|null=batch?{...batch,tags:batch.tokens??(batch.mode==='text'?[batch.raw]:splitCaptionTags(batch.raw))}:null;
  const changedTargets=batchEdit?batchTargets.filter(item=>captionEditable(item)&&JSON.stringify(captionDraftPayload(editCaptionDraft(getDraft(item),batchEdit)))!==JSON.stringify(captionDraftPayload(materializeCaptionDraft(getDraft(item))))):[];
  const applyBatch=()=>{
    if(!batch||!batchEdit||readOnly||saving)return;
    setDrafts(previous=>{const next={...previous};for(const item of changedTargets){const draft=getDraft(item),draftKey=captionDraftKey(did,item.rel_path,draft.captionFile);next[draftKey]=editCaptionDraft(previous[draftKey]||draft,batchEdit);}return next;});
    if(batchEdit.mode==='tags'&&batchEdit.action==='replace')setSelectedTags(previous=>new Set([...previous].map(tag=>tagKey(tag)===tagKey(batchEdit.from||'')?batchEdit.to!.trim():tag)));
    if(batchEdit.mode==='tags'&&batchEdit.action==='remove'&&batch.scope==='all')setSelectedTags(previous=>new Set([...previous].filter(tag=>!batchEdit.tags?.some(removed=>tagKey(removed)===tagKey(tag)))));
    setMessage(text(`已修改 ${changedTargets.length} 张图片的草稿`,`Updated drafts for ${changedTargets.length} images`));setBatch(null);
  };
  const deduplicateAll=()=>{
    if(readOnly||saving)return;
    const updates:CaptionDraftMap={};
    for(const item of items){if(!captionEditable(item))continue;const before=getDraft(item),draftKey=captionDraftKey(did,item.rel_path,before.captionFile),after=editCaptionDraft(before,{action:'deduplicate',mode:'tags'});if(JSON.stringify(captionDraftPayload(before))!==JSON.stringify(captionDraftPayload(after)))updates[draftKey]=after;}
    setDrafts(previous=>({...previous,...updates}));
    const changed=Object.keys(updates).length;setMessage(text(`已去除 ${changed} 张图片草稿中的重复标签`,`Removed duplicate tags from ${changed} image drafts`));
  };
  const selectStat=(tag:string,event:React.MouseEvent)=>{
    const anchor=statsAnchor.current;if(!event.shiftKey)statsAnchor.current=tag;
    setSelectedTags(previous=>{if(event.shiftKey&&anchor){const names=stats.map(item=>item.tag),a=names.indexOf(anchor),b=names.indexOf(tag);if(a>=0&&b>=0)return new Set([...(event.ctrlKey||event.metaKey?previous:[]),...names.slice(Math.min(a,b),Math.max(a,b)+1)]);}if(event.ctrlKey||event.metaKey){const next=new Set(previous);if(next.has(tag))next.delete(tag);else next.add(tag);return next;}return previous.has(tag)&&previous.size===1?new Set():new Set([tag]);});
  };
  const jsonFields=React.useMemo(()=>Array.from(new Map(items.flatMap(item=>(getCaptionStructure(item)?.fields||[]).map(field=>[captionFieldKey(field.path),field]))).values()),[items]);
  const currentTagKeys=(activeDraft&&analysis.byPath.get(activeDraft.image.rel_path)?.tagsByKey)||new Map<string,string>();
  const editSelectedCurrent=(action:'add'|'remove')=>{if(activeDraft&&!locked)updateActive(editCaptionDraft(activeDraft,{action,tags:[...selectedTags],mode:'tags',field:'all',position:'end'}));};
  const fieldEditor=(field:CaptionField)=>{
    if(!activeDraft)return null;const fieldKey=captionFieldKey(field.path),value=activeDraft.fields[fieldKey]??field.value,label=fieldLabel(field);
    return <section className="caption-json-field" key={fieldKey} aria-label={label}><h4 title={field.path.join('.')}>{label}</h4>{field.role==='nl'?<textarea aria-label={label} value={typeof value==='string'?value:value.join(', ')} disabled={locked} rows={4} onChange={event=>updateActive({fields:{...activeDraft.fields,[fieldKey]:event.target.value}})}/>:<CaptionTagList label={label} values={draftFieldTokens(activeDraft,field)} adding={activeDraft.additions[fieldKey]||''} onAdding={value=>updateActive({additions:{...activeDraft.additions,[fieldKey]:value}})} onChange={tags=>changeField(field,tags)} disabled={locked} drag={tagDrag.bindField(fieldKey)}/>}</section>;
  };
  let editorBody:React.ReactNode=null;
  if(catalog.isPlaceholderData||needsDocument&&documentQuery.isPending){
    editorBody=<LoadingNote label={text('读取标签…','Loading caption…')}/>;
  }else if(documentQuery.isError){
    editorBody=<p className="caption-workspace-error" role="alert">{graphMode?text('无法读取 GBC 数据。','Could not read GBC data.'):text('无法读取标签结构。','Could not read the caption structure.')}{' '}<button type="button" className="ui-link" onClick={()=>void documentQuery.refetch()}>{text('重试','Retry')}</button></p>;
  }else if(graphMode&&!structure){
    editorBody=<p className="caption-workspace-empty">{text('这张图片没有 GBC 数据。','This image has no GBC data.')}</p>;
  }else if(activeDraft){
    if(sceneCaption)editorBody=<StructuredCaptionEditor structure={structure} draft={activeDraft.fields} onChange={fields=>updateActive({fields})} readOnly/>;
    else if(isJson)editorBody=<>{fields.map(fieldEditor)}{effectiveMode==='json'&&structure&&<details className="caption-json-metadata"><summary>{text('其他元数据','Other metadata')}</summary><pre>{JSON.stringify(captionMetadata(structure),null,2)}</pre></details>}</>;
    else if(effectiveMode==='text')editorBody=<textarea aria-label={text('标签文本','Caption text')} rows={7} value={activeDraft.caption} disabled={locked} onChange={event=>updateActive({caption:event.target.value})}/>;
    else editorBody=<CaptionTagList label={text('画面内容','Image content')} values={splitCaptionTags(activeDraft.caption)} adding={activeDraft.additions.txt||''} onAdding={value=>updateActive({additions:{...activeDraft.additions,txt:value}})} onChange={values=>updateActive({caption:joinCaptionTags(values)})} disabled={locked} drag={tagDrag.bindField('txt')}/>;
  }
  const currentEditor=(
        <section className="caption-workspace-editor" role="region" aria-label={text('当前图片标签编辑器','Current image caption editor')}>
          <header className="caption-editor-heading">
            {graphMode||isJson?<Code size={13}/>:effectiveMode==='text'?<FileText size={13}/>:<Tags size={13}/>}
            <h3>{graphMode?'GBC JSON':isJson?text(captionStructureLabel(structure),captionStructureLabel(structure,true)):text('标签文本','Caption text')}</h3>
            {activeDraft&&effectiveMode!=='text'&&!sceneCaption&&<span className="caption-editor-tag-count">{allDraftTags(activeDraft).length}</span>}
            {!sceneCaption&&<span>{image?.caption_format?.toUpperCase()||'TXT'}{activeDirty?text(' · 待保存',' · Unsaved'):''}</span>}
            {!graphMode&&<button type="button" className={`ui-btn ui-btn-sm${activeDirty?' ui-btn-primary':''}`} disabled={!activeDirty||locked} onClick={()=>void saveDrafts(true)}><Save size={13}/>{text('保存当前','Save current')}</button>}
          </header>
          <div className="caption-workspace-editor-fields" {...(sceneCaption?{}:tagDrag.editorProps)}>
            {activeDraft&&!captionEditable(activeDraft.image)&&(!sceneCaption||!!image?.caption_error&&!documentQuery.isError)&&<p className="caption-workspace-error" role="alert">{image?.caption_error||text('此标签格式仅可查看，不能编辑。','This caption format is read-only.')}</p>}
            {editorBody}
          </div>
        </section>
  );
  return <section ref={workspace} className="caption-workspace" style={{'--caption-gallery-width':`${galleryWidth}px`,'--caption-stats-width':`${statsWidth}px`,'--caption-preview-height':previewHeight==null?(effectiveMode==='json'?'220px':'calc((100% - 8px) * .75)'):`${previewHeight}px`} as React.CSSProperties} aria-label={text('标签工作区','Caption workspace')}>
    {dataRouter&&<DatasetNavigationGuard shouldBlock={()=>!!dirtyCount||saving} beforeLeave={beforeLeave} onError={()=>{}}/>}
    <div className="caption-workspace-toolbar" role="group" aria-label={text('选择图片目录','Choose image folder')}>
      <label className="caption-workspace-source"><StudioSelect searchable icon={<Folder size={14}/>} aria-label={text('图片目录','Image folder')} value={did} disabled={saving||!source} options={(datasets.data||[]).map(item=>({value:item.source.id,label:nameOf(item.source.path)}))} onValueChange={changeSource}/></label>
      {showFileChoice&&<div className="caption-workspace-format"><StudioSelect aria-label={text('标签文件','Caption file')} value={captionFile} disabled={saving} options={[{value:'default',label:'AUTO'},...fileChoices.map(file=>({value:file,label:file.toUpperCase()}))]} onValueChange={file=>changeFile(file as CaptionFile)}/></div>}
      <button type="button" className={`ui-btn ui-btn-sm caption-save-all${dirtyCount?' ui-btn-primary':''}`} disabled={readOnly||saving||!dirtyCount} onClick={()=>void saveDrafts()}>{saving?<Loader2 size={14} className="animate-spin"/>:<Save size={14}/>} {text('保存全部','Save all')}{dirtyCount?` (${dirtyCount})`:''}</button>
    </div>
    {(datasets.error||catalog.error)&&<p role="alert" className="caption-workspace-error">{formatApiError(datasets.error||catalog.error)}</p>}
    {message&&<div className="caption-workspace-message" role="status">{message}{!!Object.keys(failures).length&&<details><summary>{text('查看失败图片','View failed images')}</summary>{Object.entries(failures).map(([draftKey,error])=><p role="alert" key={draftKey}>{drafts[draftKey]?.image.rel_path}: {error}</p>)}</details>}</div>}
    {datasets.isPending||catalog.isPending?<LoadingNote block label={text('读取图片与标签…','Loading images and captions…')}/>:!source?<p className="caption-workspace-empty">{text('请先导入图片。','Import images first.')}</p>:<div className="caption-workspace-content" ref={content}>
      <section className="caption-workspace-library" aria-label={text('数据集图片','Dataset images')}>
        <div className="caption-workspace-library-head" role="search" aria-label={text('筛选当前目录图片','Filter images in this folder')}>
          <div className="caption-workspace-search-row"><label className="caption-workspace-search"><Search size={13}/><input type="search" aria-label={text('搜索文件名或标签','Search filenames or captions')} placeholder={text('搜索文件名或标签…','Search filenames or captions…')} value={currentView.search} disabled={saving} onChange={event=>changeView({search:event.target.value,page:1})}/></label><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" disabled={saving||catalog.isFetching} title={text('刷新标签与统计','Refresh captions and statistics')} aria-label={text('刷新标签与统计','Refresh captions and statistics')} onClick={refresh}><RefreshCw size={13} className={catalog.isFetching?'animate-spin':undefined}/></button></div>
          <div className="caption-workspace-status-filter" role="group" aria-label={text('标签状态筛选','Caption status filter')}>{([['',text('全部','All'),datasetCounts.all],['missing',text('空标','Untagged'),datasetCounts.missing],['captioned',text('已标','Tagged'),datasetCounts.captioned],...(datasetCounts.invalid?[['invalid',text('错误','Invalid'),datasetCounts.invalid]]:[])] as [string,string,number][]).map(([value,label,count])=><button type="button" key={value} aria-pressed={currentView.status===value} disabled={saving} onClick={()=>changeView({status:value,page:1})}>{label} {count}</button>)}<SlidingIndicator className="ui-segmented-thumb caption-status-indicator"/></div>
          {tagFilter&&!!selectedTags.size&&<button type="button" className="caption-workspace-clear" onClick={()=>setTagFilter(false)}><Filter size={12}/>{text(`标签筛选 ${selectedTags.size} 项`,`Filtering ${selectedTags.size} tags`)}<X size={12}/></button>}
        </div>
        <div className="caption-workspace-gallery" ref={grid} role="region" aria-label={text('图片缩略图','Image thumbnails')} aria-busy={catalog.isFetching} tabIndex={0}>{visible.map(item=>{
          const row=analysis.byPath.get(item.rel_path)!;
          const count=row.tags.length,dirty=captionDraftDirty(row.draft);
          return <div className="caption-library-item" key={item.rel_path}><button type="button" title={item.rel_path} aria-label={text(`选择图片：${item.rel_path}`,`Select image: ${item.rel_path}`)} aria-pressed={image?.rel_path===item.rel_path} disabled={saving} onClick={()=>selectImage(item)}><LazyImage src={apiUrl(`/datasets/${did}/images/${item.hash}/thumb?size=256`)} alt={item.rel_path} loading="lazy"/>{(count>0||dirty)&&<span className="caption-image-tag-count" data-dirty={dirty} aria-label={text(`${count} 个标签${dirty?'，待保存':''}`,`${count} tags${dirty?', unsaved':''}`)}>{count}</span>}</button></div>;
        })}{!visible.length&&!catalog.isPlaceholderData&&<p className="caption-workspace-empty">{text('没有符合条件的图片。','No matching images.')}</p>}</div>
        <footer className="caption-workspace-pagination"><span>{filtered.length}/{items.length}</span><div className="caption-workspace-pager"><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一页','Previous page')} disabled={saving||page<=1} onClick={()=>changeView({page:page-1})}><ChevronLeft size={12}/></button><span>{page}/{pages}</span><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一页','Next page')} disabled={saving||page>=pages} onClick={()=>changeView({page:page+1})}><ChevronRight size={12}/></button></div></footer>
      </section>
      <CaptionResizeHandle label={text('调整缩略图栏宽度','Resize thumbnail column')} value={galleryWidth} min={160} max={380} onChange={setGalleryWidth}/>
      <div className={`caption-workspace-body${effectiveMode==='text'?' is-prose':''}`}>
        <div className="caption-workspace-visual"><div className="caption-preview-heading"><ImageIcon size={13}/><strong title={image?.rel_path}>{image?.rel_path||text('图片预览','Image preview')}</strong><span>{image?`${image.width}×${image.height}`:''}</span>
          {!!regions.length&&<button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" aria-pressed={showRegionBoxes} onClick={()=>setShowRegionBoxes(value=>!value)}>{text('区域框','Region boxes')}</button>}
          {!!originalImage&&<a className="ui-btn ui-btn-sm ui-btn-quiet" href={originalImage} target="_blank" rel="noreferrer" aria-label={text('查看原图','View original image')}>{text('原图','Original')}</a>}
          <div><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一张','Previous image')} disabled={saving||imageIndex<=0} onClick={()=>moveImage(-1)}><ChevronLeft size={14}/></button><span>{imageIndex<0?'—':imageIndex+1}/{filtered.length}</span><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一张','Next image')} disabled={saving||imageIndex<0||imageIndex>=filtered.length-1} onClick={()=>moveImage(1)}><ChevronRight size={14}/></button></div></div>
          <div className="caption-workspace-preview">{image&&<LazyImage src={previewImage} alt={text(`大图：${image.rel_path}`,`Full image: ${image.rel_path}`)} onError={()=>{if(previewImage!==originalImage)setPreviewFallback(previewKey);}}/>}
            {showRegionBoxes&&!!regions.length&&<CaptionGbcRegions key={`${key}:${previewImage}`} regions={regions}/>}
          </div>
        </div>
        {effectiveMode!=='text'&&<CaptionResizeHandle horizontal label={text('调整图片预览高度','Resize image preview')} value={previewHeight??(effectiveMode==='json'?220:Math.round((content.current?.clientHeight||600)*.75))} min={80} max={Math.max(80,(content.current?.clientHeight||600)-108)} onChange={setPreviewHeight}/>}
        {effectiveMode!=='text'&&currentEditor}
      </div>
      <CaptionResizeHandle label={text('调整标签统计栏宽度','Resize tag statistics column')} value={statsWidth} min={200} max={500} reverse onChange={setStatsWidth}/>
      <aside className="caption-workspace-right" aria-label={text('标签统计与工具','Tag statistics and tools')}>
        {effectiveMode==='text'?currentEditor:<><section className="caption-workspace-statistics" aria-label={text('整个数据集的标签统计','Statistics for the whole dataset')}><header><div><h3>{text('标签频次','Tag frequency')}</h3><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm" aria-pressed={commonOnly} onClick={()=>{setCommonOnly(value=>!value);setStatsLimit(100);}}>{commonOnly?text('公共','Common'):text('全部','All')}</button></div><div className="caption-stats-search"><input aria-label={text('搜索标签统计','Search tag statistics')} placeholder={text('筛选标签','Filter tags')} value={statsSearch} onChange={event=>{setStatsSearch(event.target.value);setStatsLimit(100);}}/><span className="caption-stats-sort" title={`${text('标签统计排序','Tag statistics sort')}：${statsOrderLabel}`}><StudioSelect icon={<ArrowDownWideNarrow size={14}/>} aria-label={`${text('标签统计排序','Tag statistics sort')}：${statsOrderLabel}`} value={statsOrder} onValueChange={value=>setStatsOrder(value as StatsOrder)} options={statsOrders}/></span></div></header>
          <div className="caption-workspace-frequencies">{stats.slice(0,statsLimit).map(item=><div key={item.tag} className="caption-stat-row" data-current={currentTagKeys.has(tagKey(item.tag))}><button type="button" className="caption-stat-select" title={item.tag} aria-label={text(`选择标签：${item.tag}，${item.count} 张图片`,`Select tag: ${item.tag}, ${item.count} images`)} aria-pressed={selectedTags.has(item.tag)} onClick={event=>selectStat(item.tag,event)}><span className="caption-stat-label"><span>{item.tag}</span><span className="caption-stat-bar" role="meter" aria-label={text(`标签频次：${item.tag}`,`Tag frequency: ${item.tag}`)} aria-valuemin={0} aria-valuemax={items.length} aria-valuenow={item.count}><span style={{width:`${items.length?item.count/items.length*100:0}%`}}/></span></span><small>{item.count}</small>{currentTagKeys.has(tagKey(item.tag))&&<CheckCircle2 size={12} aria-label={text('当前图片含此标签','This image has this tag')}/>}</button></div>)}{!stats.length&&<p className="caption-workspace-empty">{text('没有符合条件的标签。','No matching tags.')}</p>}{stats.length>statsLimit&&<button type="button" className="ui-btn ui-btn-quiet ui-btn-sm" onClick={()=>setStatsLimit(value=>value+100)}>{text('显示更多','Show more')}</button>}</div>
          <footer>{text(`${frequency.size} 个标签 · ${datasetCounts.captioned}/${items.length} 张已标`,`${frequency.size} tags · ${datasetCounts.captioned}/${items.length} tagged`)}</footer>
        </section>
        <div className="caption-tool-sidebar" role="toolbar" aria-label={text('标签批量工具','Caption batch tools')}>
          <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('按选中标签筛选','Filter by selected tags')} title={text('按选中标签筛选','Filter by selected tags')} aria-pressed={tagFilter} disabled={!items.length} onClick={()=>{setTagFilter(value=>!value);changeView({page:1});}}><Filter size={16}/></button>
          <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('编辑／替换标签','Edit / replace tags')} aria-description={editHint} title={editHint} disabled={readOnly||saving||!items.length} onClick={()=>{const tag=selectedTags.size===1?[...selectedTags][0]:'';openBatch('replace',{from:tag,to:tag,mode:'tags'});}}><Pencil size={16}/></button>
          <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('批量添加','Batch add')} title={text('批量添加','Batch add')} disabled={readOnly||saving||!items.length} onClick={()=>openBatch('add')}><ListPlus size={16}/></button>
          <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('去除重复标签','Deduplicate tags')} title={text('去除重复标签','Deduplicate tags')} disabled={readOnly||saving||!items.length} onClick={deduplicateAll}><CopyX size={16}/></button>
          <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('删除选中标签','Remove selected tags')} title={text('删除选中标签','Remove selected tags')} disabled={readOnly||saving||!selectedTags.size} onClick={()=>openBatch('remove',{raw:[...selectedTags].join(', '),tokens:[...selectedTags],mode:'tags'})}><Trash2 size={16}/></button>
          <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('添加选中标签到当前图片','Add selected tags to current image')} title={text('添加选中标签到当前图片','Add selected tags to current image')} disabled={locked||!activeDraft||!selectedTags.size} onClick={()=>editSelectedCurrent('add')}><PlusCircle size={16}/></button>
          <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('从当前图片移除选中标签','Remove selected tags from current image')} title={text('从当前图片移除选中标签','Remove selected tags from current image')} disabled={locked||!activeDraft||!selectedTags.size||![...selectedTags].every(tag=>currentTagKeys.has(tagKey(tag)))} onClick={()=>editSelectedCurrent('remove')}><MinusCircle size={16}/></button>
        </div></>}
      </aside>
    </div>}
    {batch&&<Dialog title={{add:text('批量添加标签','Batch add tags'),remove:text('批量删除标签','Batch remove tags'),replace:text('批量替换标签','Batch replace tags'),deduplicate:text('去除重复标签','Deduplicate tags')}[batch.action]} onClose={()=>setBatch(null)}><div className="caption-batch-form">
      <fieldset><legend>{text('应用范围','Apply to')}</legend><div className="ui-segmented ui-segmented-sm" role="group" aria-label={text('应用范围','Apply to')}>{([['current',text('当前图片','Current')],['all',text('全部图片','All images')]] as [BatchScope,string][]).map(([value,label])=><button type="button" key={value} aria-pressed={batch.scope===value} disabled={value==='current'&&!image} onClick={()=>setBatch({...batch,scope:value})}>{label}</button>)}<SlidingIndicator className="ui-segmented-thumb"/></div></fieldset>
      {!!jsonFields.length&&<label>{text('JSON 目标字段','JSON target field')}<StudioSelect aria-label={text('JSON 目标字段','JSON target field')} value={batch.field||'all'} onValueChange={field=>setBatch({...batch,field,mode:jsonFields.find(item=>captionFieldKey(item.path)===field)?.role==='nl'?'text':field==='all'&&effectiveMode==='text'?'text':'tags'})} options={[{value:'all',label:batch.action==='add'?text('画面内容（缺少时使用首个标签字段）','Image content (or the first tag field)'):text('全部分类','All categories')},...jsonFields.map(field=>({value:captionFieldKey(field.path),label:fieldLabel(field)}))]}/></label>}
      {batch.action==='add'&&<fieldset><legend>{text('添加位置','Insert at')}</legend><div className="ui-segmented ui-segmented-sm"><button type="button" aria-pressed={batch.position==='start'} onClick={()=>setBatch({...batch,position:'start'})}>{text('头部','Start')}</button><button type="button" aria-pressed={batch.position==='end'} onClick={()=>setBatch({...batch,position:'end'})}>{text('尾部','End')}</button><SlidingIndicator className="ui-segmented-thumb"/></div></fieldset>}
      {batch.action==='replace'?<><label>{text('原标签','Old tag')}<input aria-label={text('原标签','Old tag')} value={batch.from||''} onChange={event=>setBatch({...batch,from:event.target.value})}/></label><label>{text('新标签','New tag')}<input aria-label={text('新标签','New tag')} value={batch.to||''} onChange={event=>setBatch({...batch,to:event.target.value})}/></label></>:batch.action==='remove'&&batch.tokens?<div className="caption-delete-tags">{batch.tokens.map(tag=><span className="caption-tag" key={tag}>{tag}</span>)}</div>:batch.action!=='deduplicate'&&<label>{batch.mode==='text'?text('文本','Text'):text('标签（逗号分隔）','Tags (comma separated)')}<textarea aria-label={text('批量标签','Batch tags')} rows={3} value={batch.raw} onChange={event=>setBatch({...batch,raw:event.target.value,tokens:undefined})}/></label>}
      <p className="caption-batch-count">{text(`范围 ${batchTargets.length} 张 · 将修改 ${changedTargets.length} 张`,`Scope: ${batchTargets.length} images · ${changedTargets.length} will change`)}{batchTargets.some(item=>!captionEditable(item))&&text(` · 跳过 ${batchTargets.filter(item=>!captionEditable(item)).length} 张不可编辑图片`,` · ${batchTargets.filter(item=>!captionEditable(item)).length} read-only images skipped`)}</p>
      <div className="workspace-confirm-actions"><button type="button" className="ui-btn" onClick={()=>setBatch(null)}>{text('取消','Cancel')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={!changedTargets.length||batch.action==='replace'&&(!batch.from?.trim()||!batch.to?.trim())||batch.action==='add'&&!batch.raw.trim()} onClick={applyBatch}>{text('应用到草稿','Apply to drafts')}</button></div>
    </div></Dialog>}
    {leaveOpen&&<Dialog title={text('标签修改尚未保存','Unsaved caption changes')} closeDisabled={saving} onClose={()=>finishLeave(false)}><p>{text(`有 ${dirtyCount} 张图片的标签尚未保存。`,`Captions for ${dirtyCount} images are unsaved.`)}</p><div className="workspace-confirm-actions"><button type="button" className="ui-btn" disabled={saving} onClick={()=>finishLeave(false)}>{text('继续编辑','Keep editing')}</button><button type="button" className="ui-btn ui-btn-danger" disabled={saving} onClick={()=>finishLeave(true)}>{text('放弃并继续','Discard and continue')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={readOnly||saving} onClick={()=>void saveDrafts().then(saved=>{if(saved)finishLeave(true);})}>{text('保存全部并继续','Save all and continue')}</button></div></Dialog>}
  </section>;
}
